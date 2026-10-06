# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""Benchmark inference speed under different DEIMV2 query counts.

The script is designed for DEIMV2GSDGuided, but the fixed-query modes also work
for regular DETR-like models with ``num_queries``.

Examples:
    python tools/analysis_tools/benchmark_dynamic_queries.py \
        configs/deimv2/deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det_fine.py \
        --checkpoint work_dirs/xxx/epoch_117.pth \
        --modes dynamic force:300 force:600 force:800 fixed:300 fixed:800 \
        --max-iter 200 --num-warmup 20 --batch-size 1 --num-workers 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


StageStats = MutableMapping[str, List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark dynamic/fixed query inference speed.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument(
        '--modes',
        nargs='+',
        default=['dynamic', 'force:300', 'force:600', 'force:800'],
        help=(
            'Benchmark modes. dynamic uses model query grouping; '
            'force:N keeps original num_queries/top-k then forces grouping to N; '
            'fixed:N sets model.num_queries=N and disables query grouping.'))
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--num-warmup', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--dataset-type', choices=['val', 'test'], default='test')
    parser.add_argument('--rescale', action='store_true', help='rescale output boxes')
    parser.add_argument('--amp', action='store_true', help='use CUDA autocast fp16')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--out-json', default='', help='optional summary json path')
    parser.add_argument('--out-csv', default='', help='optional summary csv path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, in key=value format')
    parser.add_argument('--local-rank', type=int, default=0)
    return parser.parse_args()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize_ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return dict(mean=0.0, median=0.0, p90=0.0, min=0.0, max=0.0, std=0.0)
    return dict(
        mean=statistics.mean(values) * 1000.0,
        median=statistics.median(values) * 1000.0,
        p90=percentile(values, 0.90) * 1000.0,
        min=min(values) * 1000.0,
        max=max(values) * 1000.0,
        std=(statistics.pstdev(values) if len(values) > 1 else 0.0) * 1000.0,
    )


def timed_call(stats: StageStats, name: str, device: torch.device, func, *args, **kwargs):
    sync_if_cuda(device)
    start = time.perf_counter()
    out = func(*args, **kwargs)
    sync_if_cuda(device)
    stats[name].append(time.perf_counter() - start)
    return out


@contextmanager
def force_query_count(model, count: Optional[int]):
    """Temporarily force DEIMV2GSDGuided query grouping to a fixed count."""
    if count is None or not hasattr(model, '_choose_query_count'):
        yield
        return

    original = model._choose_query_count
    old_group_enabled = getattr(model, 'query_group_enabled', None)
    if old_group_enabled is not None:
        model.query_group_enabled = True

    def _forced_choose(self, query, memory, gsd_emb, spatial_shapes,
                       dn_meta=None, logits=None):
        q_total = query.size(1)
        if self.training:
            num_dn = int(dn_meta['num_denoising_queries']) if dn_meta else 0
            q_total = max(1, q_total - num_dn)
        q = max(1, min(int(count), int(q_total)))
        return torch.full(
            (query.size(0), ), q, dtype=torch.long, device=query.device)

    model._choose_query_count = MethodType(_forced_choose, model)
    try:
        yield
    finally:
        model._choose_query_count = original
        if old_group_enabled is not None:
            model.query_group_enabled = old_group_enabled


@contextmanager
def fixed_num_queries(model, count: Optional[int]):
    """Temporarily set model.num_queries and disable query grouping."""
    if count is None:
        yield
        return

    old_num_queries = getattr(model, 'num_queries', None)
    old_group_enabled = getattr(model, 'query_group_enabled', None)
    model.num_queries = int(count)
    if old_group_enabled is not None:
        model.query_group_enabled = False
    try:
        yield
    finally:
        if old_num_queries is not None:
            model.num_queries = old_num_queries
        if old_group_enabled is not None:
            model.query_group_enabled = old_group_enabled


def parse_mode(mode: str) -> Tuple[str, Optional[int]]:
    if mode == 'dynamic':
        return mode, None
    if ':' in mode:
        prefix, raw = mode.split(':', 1)
        if prefix in ('force', 'fixed'):
            return prefix, int(raw)
    raise ValueError(
        f'Invalid mode "{mode}". Use dynamic, force:N, or fixed:N.')


def build_model(cfg: Config, checkpoint: str, device: torch.device,
                fuse_conv_bn: bool = False) -> torch.nn.Module:
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    if fuse_conv_bn:
        from mmengine.model import revert_sync_batchnorm
        from mmcv.cnn import fuse_conv_bn as mmcv_fuse_conv_bn
        model = revert_sync_batchnorm(model)
        model = mmcv_fuse_conv_bn(model)
    model.to(device)
    model.eval()
    return model


def build_loader(cfg: Config, dataset_type: str, batch_size: int,
                 num_workers: int):
    if dataset_type == 'val':
        dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    else:
        dataloader_cfg = copy.deepcopy(cfg.test_dataloader)
    dataloader_cfg.batch_size = batch_size
    dataloader_cfg.num_workers = num_workers
    dataloader_cfg.persistent_workers = False
    dataloader_cfg.setdefault('collate_fn', dict(type='pseudo_collate'))
    # Some configs keep persistent_workers=True even when num_workers=0.
    if num_workers == 0:
        dataloader_cfg.persistent_workers = False
        dataloader_cfg.collate_fn = pseudo_collate
    return Runner.build_dataloader(dataloader_cfg)


def move_data_to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    # Let the model data_preprocessor handle device transfer and normalization.
    return data


def run_profile_once(model: torch.nn.Module,
                     data: Dict[str, Any],
                     device: torch.device,
                     stats: StageStats,
                     rescale: bool,
                     amp: bool) -> Tuple[int, List[int]]:
    sync_if_cuda(device)
    total_start = time.perf_counter()

    with torch.no_grad(), torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
            enabled=(amp and device.type == 'cuda')):
        data = move_data_to_device(data, device)
        data = timed_call(stats, 'data_preprocessor', device,
                          model.data_preprocessor, data, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']

        if hasattr(model, '_last_query_group_logits'):
            model._last_query_group_logits = None
        if hasattr(model, '_last_group_queries'):
            model._last_group_queries = None

        if hasattr(model, '_update_online_gsd_cache'):
            timed_call(stats, 'gsd_predictor_cache', device,
                       model._update_online_gsd_cache, batch_inputs)

        img_feats = timed_call(stats, 'backbone_neck', device,
                               model.extract_feat, batch_inputs)
        encoder_inputs_dict, decoder_inputs_dict = timed_call(
            stats, 'pre_transformer', device, model.pre_transformer,
            img_feats, batch_data_samples)
        encoder_outputs_dict = timed_call(
            stats, 'encoder', device, lambda: model.forward_encoder(
                **encoder_inputs_dict))
        tmp_dec_in, head_inputs_dict = timed_call(
            stats, 'pre_decoder_query_select', device, lambda:
            model.pre_decoder(
                **encoder_outputs_dict,
                batch_data_samples=batch_data_samples))
        decoder_inputs_dict.update(tmp_dec_in)
        effective_query = int(decoder_inputs_dict['query'].shape[1])
        group_queries = []
        if getattr(model, '_last_group_queries', None) is not None:
            group_queries = [int(v) for v in model._last_group_queries.tolist()]

        decoder_outputs_dict = timed_call(
            stats, 'decoder', device,
            lambda: model.forward_decoder(**decoder_inputs_dict))
        head_inputs_dict.update(decoder_outputs_dict)
        results_list = timed_call(
            stats, 'bbox_head_predict_postprocess', device,
            lambda: model.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples))
        timed_call(stats, 'add_pred_to_datasample', device,
                   model.add_pred_to_datasample, batch_data_samples,
                   results_list)

    sync_if_cuda(device)
    stats['total'].append(time.perf_counter() - total_start)
    return effective_query, group_queries


def run_mode(model: torch.nn.Module, dataloader, mode: str,
             max_iter: int, num_warmup: int, device: torch.device,
             rescale: bool, amp: bool) -> Dict[str, Any]:
    mode_type, count = parse_mode(mode)
    stats: StageStats = defaultdict(list)
    query_counts: List[int] = []
    per_image_group_queries: List[int] = []
    num_samples = 0

    mode_context = force_query_count(model, count) if mode_type == 'force' \
        else fixed_num_queries(model, count) if mode_type == 'fixed' \
        else force_query_count(model, None)

    with mode_context:
        for i, data in enumerate(dataloader):
            if i >= max_iter:
                break

            # Run warmup iterations, but discard their timing.
            iter_stats: StageStats = defaultdict(list)
            eff_q, group_q = run_profile_once(
                model=model,
                data=data,
                device=device,
                stats=iter_stats,
                rescale=rescale,
                amp=amp)

            if i < num_warmup:
                continue

            for key, values in iter_stats.items():
                stats[key].extend(values)
            query_counts.append(eff_q)
            per_image_group_queries.extend(group_q)
            num_samples += len(data['data_samples'])

    stage_summary = {name: summarize_ms(values) for name, values in stats.items()}
    total_values = stats.get('total', [])
    total_mean = stage_summary.get('total', {}).get('mean', 0.0)
    total_time = sum(total_values)
    fps = num_samples / total_time if total_time > 0 else 0.0
    ms_per_image = total_time * 1000.0 / num_samples if num_samples > 0 else 0.0
    return dict(
        mode=mode,
        num_iters=len(total_values),
        num_samples=num_samples,
        fps=fps,
        ms_per_batch=total_mean,
        ms_per_image=ms_per_image,
        effective_query_mean=statistics.mean(query_counts) if query_counts else 0,
        effective_query_min=min(query_counts) if query_counts else 0,
        effective_query_max=max(query_counts) if query_counts else 0,
        group_query_hist={
            str(q): per_image_group_queries.count(q)
            for q in sorted(set(per_image_group_queries))
        },
        stages_ms=stage_summary)


def print_summary(results: List[Dict[str, Any]]) -> None:
    print('\n========== Dynamic Query Benchmark ==========')
    print('Note: inference has no Hungarian matching; matching is a training-only cost.')
    header = (
        f'{"mode":<12} {"img/s":>9} {"ms/img":>9} {"ms/batch":>10} {"query(avg)":>11} '
        f'{"pre_dec":>9} {"decoder":>9} {"head+post":>10} {"gsd":>9}')
    print(header)
    print('-' * len(header))
    for item in results:
        stages = item['stages_ms']
        pre_dec = stages.get('pre_decoder_query_select', {}).get('mean', 0.0)
        dec = stages.get('decoder', {}).get('mean', 0.0)
        head = stages.get('bbox_head_predict_postprocess', {}).get('mean', 0.0)
        gsd = stages.get('gsd_predictor_cache', {}).get('mean', 0.0)
        print(
            f"{item['mode']:<12} {item['fps']:>9.2f} "
            f"{item['ms_per_image']:>9.2f} "
            f"{item['ms_per_batch']:>10.2f} "
            f"{item['effective_query_mean']:>11.1f} "
            f"{pre_dec:>9.2f} {dec:>9.2f} {head:>10.2f} {gsd:>9.2f}")
    print('\nStage details are saved in JSON/CSV if --out-json/--out-csv is set.')


def save_outputs(results: List[Dict[str, Any]], out_json: str,
                 out_csv: str) -> None:
    if out_json:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding='utf-8')
        print(f'Saved JSON: {path}')

    if out_csv:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        stage_names = sorted({
            stage
            for item in results
            for stage in item.get('stages_ms', {}).keys()
        })
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'mode', 'fps', 'ms_per_batch', 'effective_query_mean',
                'ms_per_image',
                *[f'{stage}_mean_ms' for stage in stage_names]
            ])
            for item in results:
                stages = item.get('stages_ms', {})
                writer.writerow([
                    item['mode'], item['fps'], item['ms_per_batch'],
                    item['effective_query_mean'], item['ms_per_image'],
                    *[stages.get(stage, {}).get('mean', 0.0)
                      for stage in stage_names]
                ])
        print(f'Saved CSV: {path}')


def main() -> int:
    args = parse_args()
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    device = torch.device('cuda', args.local_rank) if torch.cuda.is_available() \
        else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    model = build_model(
        cfg=cfg,
        checkpoint=args.checkpoint,
        device=device,
        fuse_conv_bn=args.fuse_conv_bn)
    dataloader = build_loader(
        cfg=cfg,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.max_iter <= args.num_warmup:
        raise ValueError('--max-iter must be larger than --num-warmup.')

    results = []
    for mode in args.modes:
        print(f'\nRunning mode: {mode}')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        result = run_mode(
            model=model,
            dataloader=dataloader,
            mode=mode,
            max_iter=args.max_iter,
            num_warmup=args.num_warmup,
            device=device,
            rescale=args.rescale,
            amp=args.amp)
        if device.type == 'cuda':
            result['max_cuda_memory_mb'] = torch.cuda.max_memory_allocated(
                device) / 1024 / 1024
        results.append(result)

    print_summary(results)
    save_outputs(results, args.out_json, args.out_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
