# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""Profile one training step by stages for DEIMV2/D-FINE style detectors.

This profiler manually follows ``model.loss``:

    data_preprocessor -> gsd predictor -> backbone -> neck -> encoder ->
    pre_decoder -> decoder -> loss/cost/matching -> backward -> optimizer step

For D-FINE/DEIM heads, Hungarian matching is not called through
``HungarianAssigner.assign``. It is implemented inside
``DFINEHead._get_match_indices``. This script temporarily replaces that method
on the current bbox head so the matching path is split into:

    loss_cost_build, hungarian_cpu_transfer, hungarian_scipy, matching_total

Example:
    python tools/analysis_tools/profile_train_stages.py \
        configs/deimv2/deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det_fine.py \
        --checkpoint work_dirs/xxx/epoch_117.pth \
        --max-iter 50 --num-warmup 5 --batch-size 2 --num-workers 2 \
        --out-json work_dirs/profile/train_stage_profile.json \
        --out-csv work_dirs/profile/train_stage_profile.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_repo_package(package: str = 'mmdet') -> None:
    """Prefer this checkout when another editable mmdet exists in PYTHONPATH."""
    repo_root = str(REPO_ROOT)
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO_ROOT]
    sys.path.insert(0, repo_root)

    pkg = sys.modules.get(package)
    pkg_file = getattr(pkg, '__file__', '') if pkg is not None else ''
    if pkg_file and not Path(pkg_file).resolve().is_relative_to(REPO_ROOT):
        for name in list(sys.modules):
            if name == package or name.startswith(f'{package}.'):
                del sys.modules[name]


ensure_repo_package('mmdet')

import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment


StageStats = MutableMapping[str, List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Profile training-stage timing for DETR-like detectors.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', default='', help='optional checkpoint file')
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument(
        '--epoch',
        type=int,
        default=0,
        help='epoch value written to model.set_epoch(), affects assigner switch')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--optimizer-step',
        action='store_true',
        help='also run optimizer.step(); otherwise only backward is measured')
    parser.add_argument(
        '--no-profile-matching',
        action='store_true',
        help='do not replace DFINEHead._get_match_indices')
    parser.add_argument('--out-json', default='', help='optional output json')
    parser.add_argument('--out-csv', default='', help='optional output csv')
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
        std=(statistics.pstdev(values) if len(values) > 1 else 0.0) * 1000.0)


def timed_call(stats: StageStats, name: str, device: torch.device, func,
               *args, **kwargs):
    sync_if_cuda(device)
    start = time.perf_counter()
    out = func(*args, **kwargs)
    sync_if_cuda(device)
    stats[name].append(time.perf_counter() - start)
    return out


def add_time(stats: StageStats, name: str, seconds: float) -> None:
    stats[name].append(float(seconds))


def parse_losses(losses: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    log_vars = {}
    total_loss = None
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            loss = loss_value.mean()
        elif isinstance(loss_value, list):
            loss = sum(v.mean() for v in loss_value)
        else:
            continue
        log_vars[loss_name] = float(loss.detach().cpu())
        if 'loss' in loss_name:
            total_loss = loss if total_loss is None else total_loss + loss
    if total_loss is None:
        raise RuntimeError('No loss tensor found in loss dict.')
    log_vars['loss'] = float(total_loss.detach().cpu())
    return total_loss, log_vars


def build_model(cfg: Config, checkpoint: str, device: torch.device):
    ensure_repo_package('mmdet')
    # The fine-grained configs use custom modules added in this checkout. Import
    # them explicitly so registry state does not depend on the caller's import
    # order or an existing editable MMDetection installation.
    import mmdet.models.dense_heads.hierarchical_deimv2_head  # noqa: F401

    from mmdet.registry import MODELS

    model = MODELS.build(cfg.model)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    model.train()
    return model


def build_loader(cfg: Config,
                 batch_size: int,
                 num_workers: int,
                 seed: Optional[int] = None):
    dataloader_cfg = copy.deepcopy(cfg.train_dataloader)
    if dataloader_cfg.get('batch_sampler', None) is not None:
        # Some MMEngine versions pass top-level batch_size through to
        # torch.DataLoader even when batch_sampler is set. PyTorch only accepts
        # batch_size=1 in that case, so keep the real batch size inside the
        # batch sampler config.
        if isinstance(dataloader_cfg.batch_sampler, dict):
            dataloader_cfg.batch_sampler.batch_size = batch_size
        dataloader_cfg.batch_size = 1
        dataloader_cfg.drop_last = False
        dataloader_cfg.shuffle = False
    else:
        dataloader_cfg.batch_size = batch_size
    dataloader_cfg.num_workers = num_workers
    dataloader_cfg.persistent_workers = bool(num_workers > 0)
    dataloader_cfg.setdefault('collate_fn', dict(type='pseudo_collate'))
    if num_workers == 0:
        dataloader_cfg.persistent_workers = False
        dataloader_cfg.collate_fn = pseudo_collate
    return Runner.build_dataloader(dataloader_cfg, seed=seed)


def build_optimizer(model, cfg: Config):
    optim_cfg = copy.deepcopy(cfg.get('optim_wrapper', {}))
    if not optim_cfg:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-6,
            weight_decay=0.0)
        return optimizer

    # Prefer the exact repo optimizer wrapper when the config is complete.
    try:
        optim_wrapper = Runner.build_optim_wrapper(model, optim_cfg)
        return optim_wrapper
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-6,
            weight_decay=0.0)
        return optimizer


def zero_grad(optim_obj) -> None:
    if hasattr(optim_obj, 'zero_grad'):
        optim_obj.zero_grad()
    elif hasattr(optim_obj, 'optimizer'):
        optim_obj.optimizer.zero_grad(set_to_none=True)


def optimizer_step(optim_obj) -> None:
    if hasattr(optim_obj, 'step'):
        optim_obj.step()
    elif hasattr(optim_obj, 'optimizer'):
        optim_obj.optimizer.step()


@contextmanager
def profiled_dfine_matching(head, stats: StageStats, device: torch.device):
    """Patch the current head's _get_match_indices with stage timers."""
    from mmengine.dist import get_dist_info
    from mmdet.models.dense_heads.dfine_head import DFINEHead
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    if not isinstance(head, DFINEHead) or not hasattr(head, '_get_match_indices'):
        yield
        return

    original = head._get_match_indices
    alert_counts = defaultdict(int)
    rank, _ = get_dist_info()

    @torch.no_grad()
    def _profiled_get_match_indices(
            self, all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas):
        sync_if_cuda(device)
        total_start = time.perf_counter()

        num_imgs = all_layers_matching_cls_scores[0].shape[0]
        gt_instances = InstanceData.cat(batch_gt_instances)
        num_target_list = list(map(len, batch_gt_instances))
        add_time(stats, 'matching_num_gts', float(sum(num_target_list)) / 1000.0)

        img_shapes = {tuple(img_meta['img_shape']) for img_meta in batch_img_metas}
        assert len(img_shapes) == 1, (
            f'All images must have the same shape, but got {img_shapes}.')

        img_meta = batch_img_metas[0]
        img_h, img_w = img_meta['img_shape']
        factor = all_layers_matching_bbox_preds[0].new_tensor(
            [img_w, img_h, img_w, img_h]).unsqueeze(0)

        all_layers_match_indices = []
        for layer_idx, (cls_score, bbox_pred) in enumerate(
                zip(all_layers_matching_cls_scores,
                    all_layers_matching_bbox_preds)):
            if cls_score is None or bbox_pred is None:
                all_layers_match_indices.append(None)
                continue

            num_imgs, num_queries, _ = cls_score.shape
            add_time(stats, 'matching_num_queries', float(num_queries) / 1000.0)

            sync_if_cuda(device)
            prep_start = time.perf_counter()
            cls_score = cls_score.flatten(0, 1).float()
            bbox_pred = bbox_pred.flatten(0, 1).float()
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
            bbox_pred = bbox_pred * factor
            pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
            sync_if_cuda(device)
            add_time(stats, 'matching_prepare', time.perf_counter() - prep_start)

            total_cost = None
            for match_cost in self.assigner.match_costs:
                sync_if_cuda(device)
                cost_start = time.perf_counter()
                cost = match_cost(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    img_meta=img_meta)
                if not torch.all(torch.isfinite(cost)):
                    cost_name = match_cost.__class__.__name__
                    alert_counts[cost_name] += 1
                    add_time(stats, f'nonfinite_cost/{cost_name}', 1.0)
                    if rank == 0 and alert_counts[cost_name] == 1:
                        print(
                            f'ALERT: {cost_name} produced NaN/Inf values. '
                            f'Further alerts for this cost are suppressed.')
                cost.nan_to_num_(nan=1.0).float()
                total_cost = cost if total_cost is None else cost + total_cost
                sync_if_cuda(device)
                elapsed = time.perf_counter() - cost_start
                add_time(stats, 'loss_cost_build', elapsed)
                add_time(stats, f'loss_cost_build/{match_cost.__class__.__name__}',
                         elapsed)

            sync_if_cuda(device)
            view_start = time.perf_counter()
            total_cost = total_cost.view(num_imgs, num_queries, -1)
            sync_if_cuda(device)
            add_time(stats, 'loss_cost_view', time.perf_counter() - view_start)

            batch_match_indices = []
            for bid, cost in enumerate(total_cost.split(num_target_list, -1)):
                sync_if_cuda(device)
                cpu_start = time.perf_counter()
                cost_cpu = cost[bid].cpu()
                cpu_elapsed = time.perf_counter() - cpu_start
                add_time(stats, 'hungarian_cpu_transfer', cpu_elapsed)

                hung_start = time.perf_counter()
                row, col = linear_sum_assignment(cost_cpu)
                hung_elapsed = time.perf_counter() - hung_start
                add_time(stats, 'hungarian_scipy', hung_elapsed)
                add_time(stats, 'hungarian_total', cpu_elapsed + hung_elapsed)

                tensor_start = time.perf_counter()
                batch_match_indices.append(
                    (torch.from_numpy(row).to(torch.long),
                     torch.from_numpy(col).to(torch.long)))
                add_time(stats, 'hungarian_tensorize', time.perf_counter() - tensor_start)

            all_layers_match_indices.append(batch_match_indices)

        sync_if_cuda(device)
        add_time(stats, 'matching_total', time.perf_counter() - total_start)
        return all_layers_match_indices

    head._get_match_indices = MethodType(_profiled_get_match_indices, head)
    try:
        yield
    finally:
        head._get_match_indices = original


def forward_loss_by_stages(model, data: Dict[str, Any], device: torch.device,
                           stats: StageStats, amp: bool) -> Dict[str, Any]:
    data = timed_call(stats, 'data_preprocessor', device,
                      model.data_preprocessor, data, True)
    batch_inputs = data['inputs']
    batch_data_samples = data['data_samples']

    if hasattr(model, '_last_query_group_logits'):
        model._last_query_group_logits = None
    if hasattr(model, '_last_group_queries'):
        model._last_group_queries = None

    with torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
            enabled=(amp and device.type == 'cuda')):
        if hasattr(model, '_update_online_gsd_cache'):
            timed_call(stats, 'gsd_predictor_cache', device,
                       model._update_online_gsd_cache, batch_inputs)

        sync_if_cuda(device)
        backbone_start = time.perf_counter()
        feats = model.backbone(batch_inputs)
        sync_if_cuda(device)
        stats['backbone'].append(time.perf_counter() - backbone_start)

        if getattr(model, 'with_neck', False):
            feats = timed_call(stats, 'neck', device, model.neck, feats)
        else:
            stats['neck'].append(0.0)

        encoder_inputs_dict, decoder_inputs_dict = timed_call(
            stats, 'pre_transformer', device, model.pre_transformer,
            feats, batch_data_samples)
        encoder_outputs_dict = timed_call(
            stats, 'encoder', device,
            lambda: model.forward_encoder(**encoder_inputs_dict))
        tmp_dec_in, head_inputs_dict = timed_call(
            stats, 'pre_decoder', device,
            lambda: model.pre_decoder(
                **encoder_outputs_dict,
                batch_data_samples=batch_data_samples))
        decoder_inputs_dict.update(tmp_dec_in)

        effective_query = int(decoder_inputs_dict['query'].shape[1])
        stats['effective_query'].append(float(effective_query) / 1000.0)
        dn_meta = head_inputs_dict.get('dn_meta', None)
        num_dn = int(dn_meta.get('num_denoising_queries', 0)) \
            if dn_meta is not None else 0
        matching_query = max(0, effective_query - num_dn)
        stats['effective_matching_query'].append(
            float(matching_query) / 1000.0)
        stats['effective_dn_query'].append(float(num_dn) / 1000.0)
        group_queries = getattr(model, '_last_group_queries', None)
        if group_queries is not None:
            for q in group_queries.reshape(-1).tolist():
                stats['group_query_per_image'].append(float(q) / 1000.0)

        decoder_outputs_dict = timed_call(
            stats, 'decoder', device,
            lambda: model.forward_decoder(**decoder_inputs_dict))
        head_inputs_dict.update(decoder_outputs_dict)

        losses = timed_call(
            stats, 'loss_total_with_matching', device,
            lambda: model.bbox_head.loss(
                **head_inputs_dict,
                batch_data_samples=batch_data_samples))

        if (getattr(model, 'query_group_enabled', False)
                and getattr(model, '_last_query_group_logits', None) is not None):
            group_gsd = model._get_batch_gsd(
                batch_data_samples=batch_data_samples,
                device=model._last_query_group_logits.device,
                dtype=model._last_query_group_logits.dtype) \
                if hasattr(model, '_get_batch_gsd') else None
            aux = timed_call(
                stats, 'query_group_aux_loss', device,
                lambda: model._query_group_aux_loss(
                    logits=model._last_query_group_logits,
                    batch_data_samples=batch_data_samples,
                    gsd=group_gsd))
            losses.update(aux)

    return losses


def run_profile_step(model, data, optim_obj, device: torch.device,
                     amp: bool, optimizer_step_enabled: bool,
                     profile_matching: bool) -> Tuple[StageStats, Dict[str, float]]:
    stats: StageStats = defaultdict(list)
    sync_if_cuda(device)
    step_start = time.perf_counter()

    timed_call(stats, 'zero_grad', device, zero_grad, optim_obj)
    context = profiled_dfine_matching(model.bbox_head, stats, device) \
        if profile_matching else nullcontext()

    with context:
        losses = forward_loss_by_stages(model, data, device, stats, amp)

    total_loss, log_vars = parse_losses(losses)
    if not torch.isfinite(total_loss.detach()):
        log_vars['nonfinite_loss'] = 1.0
        stats['backward'].append(0.0)
        stats['optimizer_step'].append(0.0)
        zero_grad(optim_obj)
        sync_if_cuda(device)
        stats['train_step_total'].append(time.perf_counter() - step_start)
        return stats, log_vars

    timed_call(stats, 'backward', device, total_loss.backward)
    if optimizer_step_enabled:
        timed_call(stats, 'optimizer_step', device, optimizer_step, optim_obj)
    else:
        stats['optimizer_step'].append(0.0)

    sync_if_cuda(device)
    stats['train_step_total'].append(time.perf_counter() - step_start)
    return stats, log_vars


def merge_stats(dst: StageStats, src: StageStats) -> None:
    for key, values in src.items():
        dst[key].extend(values)


def print_summary(summary: Dict[str, Any]) -> None:
    rows = [
        'data_preprocessor', 'gsd_predictor_cache', 'backbone', 'neck',
        'pre_transformer', 'encoder', 'pre_decoder', 'decoder',
        'loss_total_with_matching', 'loss_cost_build', 'hungarian_cpu_transfer',
        'hungarian_scipy', 'hungarian_total', 'matching_total',
        'query_group_aux_loss', 'backward', 'optimizer_step',
        'train_step_total'
    ]
    print('\n========== Training Stage Profile ==========')
    print(f"iters={summary['num_iters']} samples={summary['num_samples']}")
    print(
        f'{"stage":<28} {"mean(ms)":>10} {"median":>10} '
        f'{"p90":>10} {"std":>10}')
    print('-' * 72)
    stages = summary['stages_ms']
    for row in rows:
        if row not in stages:
            continue
        item = stages[row]
        print(
            f'{row:<28} {item["mean"]:>10.2f} {item["median"]:>10.2f} '
            f'{item["p90"]:>10.2f} {item["std"]:>10.2f}')

    cost_rows = [k for k in stages if k.startswith('loss_cost_build/')]
    if cost_rows:
        print('\n-- Cost Components --')
        for row in sorted(cost_rows):
            item = stages[row]
            print(f'{row:<44} {item["mean"]:>10.2f} ms')

    print(
        '\nNote: matching_total includes cost build, CPU transfer, SciPy '
        'linear_sum_assignment, and tensor conversion inside _get_match_indices.')


def save_outputs(summary: Dict[str, Any], out_json: str, out_csv: str) -> None:
    if out_json:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding='utf-8')
        print(f'Saved JSON: {path}')

    if out_csv:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['stage', 'mean_ms', 'median_ms', 'p90_ms', 'std_ms',
                             'min_ms', 'max_ms', 'num_records'])
            for stage, values in summary['raw_seconds'].items():
                ms = summary['stages_ms'][stage]
                writer.writerow([
                    stage, ms['mean'], ms['median'], ms['p90'], ms['std'],
                    ms['min'], ms['max'], len(values)
                ])
        print(f'Saved CSV: {path}')


def main() -> int:
    args = parse_args()
    if args.max_iter <= args.num_warmup:
        raise ValueError('--max-iter must be larger than --num-warmup.')

    from mmdet.utils import register_all_modules

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

    model = build_model(cfg, args.checkpoint, device)
    if hasattr(model, 'set_epoch'):
        model.set_epoch(args.epoch)
    dataloader = build_loader(cfg, args.batch_size, args.num_workers)
    optim_obj = build_optimizer(model, cfg)

    all_stats: StageStats = defaultdict(list)
    last_log_vars = {}
    num_samples = 0
    kept_iters = 0

    from mmengine.logging import MessageHub
    message_hub = MessageHub.get_current_instance()
    message_hub.update_info('epoch', args.epoch)

    for i, data in enumerate(dataloader):
        if i >= args.max_iter:
            break
        message_hub.update_info('iter', i)
        step_stats, log_vars = run_profile_step(
            model=model,
            data=data,
            optim_obj=optim_obj,
            device=device,
            amp=args.amp,
            optimizer_step_enabled=args.optimizer_step,
            profile_matching=(not args.no_profile_matching))
        if i < args.num_warmup:
            continue
        merge_stats(all_stats, step_stats)
        last_log_vars = log_vars
        num_samples += len(data['data_samples'])
        kept_iters += 1
        print(
            f'Profile iter {i + 1}/{args.max_iter}: '
            f'loss={log_vars.get("loss", 0.0):.4f}', flush=True)

    raw_seconds = {k: [float(v) for v in vals] for k, vals in all_stats.items()}
    summary = dict(
        num_iters=kept_iters,
        num_samples=num_samples,
        batch_size=args.batch_size,
        amp=args.amp,
        optimizer_step=args.optimizer_step,
        stages_ms={k: summarize_ms(v) for k, v in raw_seconds.items()},
        raw_seconds=raw_seconds,
        last_log_vars=last_log_vars)
    print_summary(summary)
    save_outputs(summary, args.out_json, args.out_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
