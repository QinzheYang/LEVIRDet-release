# Copyright (c) OpenMMLab. All rights reserved.
# -*- coding: utf-8 -*-
"""Benchmark training speed before/after training-time dynamic queries.

The script builds two config variants from one config file:

1. baseline: query_group is kept, but train-time query cropping is disabled and
   QueryAwareBatchSampler is removed. This matches the previous training flow.
2. dynamic_train: GT/GSD proxy query cropping is enabled during training and a
   QueryAwareBatchSampler is used.

Example:
    python tools/analysis_tools/benchmark_train_query_modes.py \
        configs/deimv2/deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det_fine.py \
        --checkpoint work_dirs/xxx/epoch_117.pth \
        --max-iter 60 --num-warmup 10 --batch-size 2 --num-workers 2 \
        --amp --optimizer-step \
        --out-json work_dirs/profile/query_train_benchmark.json

    torchrun --nproc_per_node=8 tools/analysis_tools/benchmark_train_query_modes.py \
        configs/deimv2/deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det_fine.py \
        --checkpoint epoch_117_hierarchy_pretrain_vehicle_ll.pth \
        --full-epoch --num-warmup 0 --batch-size 2 --num-workers 2 \
        --amp --modes baseline dynamic_no_sampler dynamic_train \
        --launcher pytorch --log-interval 50
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_repo_package(package: str = 'mmdet') -> None:
    """Prefer this checkout when another editable mmdet exists in PYTHONPATH."""
    repo_root = str(REPO_ROOT)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != repo_root]
    sys.path.insert(0, repo_root)

    pkg = sys.modules.get(package)
    pkg_file = getattr(pkg, '__file__', '') if pkg is not None else ''
    if pkg_file and not os.path.abspath(pkg_file).startswith(
            os.path.abspath(repo_root)):
        for name in list(sys.modules):
            if name == package or name.startswith(f'{package}.'):
                del sys.modules[name]


ensure_repo_package('mmdet')

import torch
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.logging import MessageHub
from mmengine.registry import init_default_scope

from tools.analysis_tools.profile_train_stages import (
    StageStats, build_loader, build_model, build_optimizer, merge_stats,
    run_profile_step, summarize_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare baseline training speed with dynamic-query training.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', default='', help='optional checkpoint file')
    parser.add_argument('--max-iter', type=int, default=60)
    parser.add_argument(
        '--full-epoch',
        '--full-train-data',
        dest='full_epoch',
        action='store_true',
        help='ignore --max-iter and iterate over the whole train dataloader '
        'once; in distributed mode each rank runs its own data shard and '
        'rank0 reports aggregated statistics')
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument(
        '--seed',
        type=int,
        default=20260530,
        help='fixed seed for sampler, dataloader workers, and torch RNG')
    parser.add_argument(
        '--summary-stat',
        choices=['mean', 'median'],
        default='median',
        help='statistic used in the comparison table')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='rank0 progress print interval in iterations; set larger for '
        '--full-epoch runs')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--optimizer-step',
        action='store_true',
        help='kept for compatibility; by default benchmark does not mutate '
        'weights, because repeated manual optimizer steps can pollute timing')
    parser.add_argument(
        '--keep-lr',
        action='store_true',
        help='deprecated alias; use --real-optimizer-step to really update')
    parser.add_argument(
        '--real-optimizer-step',
        action='store_true',
        help='actually run optimizer.step(); not recommended for speed-only '
        'benchmark because it changes model state across iterations')
    parser.add_argument(
        '--stop-on-nonfinite',
        action='store_true',
        help='raise immediately when a non-finite loss is encountered')
    parser.add_argument(
        '--no-profile-matching',
        action='store_true',
        help='do not split D-FINE matching internals')
    parser.add_argument(
        '--modes',
        nargs='+',
        default=['baseline', 'dynamic_train'],
        choices=['baseline', 'dynamic_train', 'dynamic_no_sampler'],
        help='benchmark modes to run')
    parser.add_argument(
        '--no-sync-across-ranks',
        action='store_true',
        help='disable train-time all_reduce(MAX) for query count in dynamic modes')
    parser.add_argument(
        '--train-begin-epoch',
        type=int,
        default=0,
        help='train_begin_epoch for dynamic modes')
    parser.add_argument(
        '--gsd-weight',
        type=float,
        default=None,
        help='override query_group_cfg.gsd_weight in dynamic modes')
    parser.add_argument(
        '--query-bins',
        type=int,
        nargs='+',
        default=None,
        help='override query bins, e.g. --query-bins 300 600 800')
    parser.add_argument(
        '--out-json',
        default='',
        help='optional output json path for raw and summarized results')
    parser.add_argument(
        '--out-csv',
        default='',
        help='optional output csv path for the comparison table')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options before mode-specific edits')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='initialize distributed env; model is not DDP-wrapped')
    parser.add_argument('--local-rank', type=int, default=0)
    return parser.parse_args()


def ensure_query_group_cfg(cfg: Config) -> ConfigDict:
    cfg.model.setdefault('gsd_cfg', ConfigDict())
    cfg.model.gsd_cfg.setdefault('query_group_cfg', ConfigDict())
    return cfg.model.gsd_cfg.query_group_cfg


def build_query_batch_sampler(qcfg: ConfigDict) -> ConfigDict:
    return ConfigDict(
        type='QueryAwareBatchSampler',
        bins=list(qcfg.get('bins', [300, 600, 800])),
        query_per_gt=float(qcfg.get('query_per_gt', 3.0)),
        small_obj_weight=float(qcfg.get('small_obj_weight', 15.0)),
        density_weight=float(qcfg.get('density_weight', 0.2)),
        scale_bonus_weight=float(qcfg.get('scale_bonus_weight', 8.0)),
        scale_ref=float(qcfg.get('scale_ref', 0.08)),
        drop_last=False)


def make_mode_cfg(base_cfg: Config, mode: str, args: argparse.Namespace) -> Config:
    cfg = copy.deepcopy(base_cfg)
    qcfg = ensure_query_group_cfg(cfg)

    if args.query_bins is not None:
        qcfg.bins = list(args.query_bins)
    if args.gsd_weight is not None:
        qcfg.gsd_weight = float(args.gsd_weight)

    if mode == 'baseline':
        # Previous behavior: grouping head may still have aux loss, but decoder
        # query tensors are not cropped during training.
        qcfg.train_enabled = False
        cfg.train_dataloader.batch_sampler = None
    else:
        qcfg.enabled = True
        qcfg.train_enabled = True
        qcfg.train_begin_epoch = int(args.train_begin_epoch)
        qcfg.train_use_gsd = True
        qcfg.train_sync_across_ranks = not args.no_sync_across_ranks

        if mode == 'dynamic_train':
            cfg.train_dataloader.batch_sampler = build_query_batch_sampler(qcfg)
            cfg.train_dataloader.setdefault(
                'sampler', ConfigDict(type='DefaultSampler', shuffle=True))
        elif mode == 'dynamic_no_sampler':
            cfg.train_dataloader.batch_sampler = None

    return cfg


def get_effective_query(summary: Dict[str, Any]) -> float:
    # profile_train_stages stores query/1000 as seconds so summarize_ms yields
    # the actual query count. Keep this small oddity contained here.
    return stage_value(summary, 'effective_query', 'mean')


def get_stage_count(summary: Dict[str, Any], key: str) -> float:
    return stage_value(summary, key, 'mean')


def stage_value(summary: Dict[str, Any], key: str, stat: str) -> float:
    return float(summary['stages_ms'].get(key, {}).get(stat, 0.0))


def set_benchmark_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_query_values(summary: Dict[str, Any], key: str) -> List[int]:
    values = summary.get('raw_seconds', {}).get(key, [])
    return [int(round(float(v) * 1000.0)) for v in values]


def query_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return dict(count=0, mean=0.0, min=0.0, max=0.0)
    return dict(
        count=len(values),
        mean=sum(values) / len(values),
        min=float(min(values)),
        max=float(max(values)))


def format_bin_distribution(values: List[int], bins: List[int]) -> str:
    if not values:
        return 'none'
    counts = Counter(values)
    total = len(values)
    parts = []
    used = 0
    for bin_value in bins:
        count = counts.get(bin_value, 0)
        used += count
        parts.append(f'{bin_value}:{count / total * 100:.1f}%({count})')
    other = total - used
    if other:
        common_other = ','.join(
            f'{k}:{v}' for k, v in counts.most_common()
            if k not in bins)[:60]
        parts.append(f'other:{other / total * 100:.1f}%({other}; {common_other})')
    return ' | '.join(parts)


def set_optimizer_lr(optim_obj, lr: float) -> None:
    optimizer = getattr(optim_obj, 'optimizer', optim_obj)
    for group in getattr(optimizer, 'param_groups', []):
        group['lr'] = lr


def dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    return torch.distributed.get_world_size() if dist_initialized() else 1


def aggregate_summary_across_ranks(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Merge per-rank benchmark summaries for rank0 reporting."""
    if not dist_initialized():
        summary['world_size'] = 1
        summary['global_batch_size'] = int(summary.get('batch_size', 1))
        return summary

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, summary)

    if rank != 0:
        return summary

    merged = copy.deepcopy(summary)
    per_rank_iters = [
        int(s.get('num_iters', 0)) for s in gathered if s
    ]
    merged['world_size'] = world_size
    merged['global_batch_size'] = int(summary.get('batch_size', 1)) * world_size
    merged['num_iters'] = max(per_rank_iters) if per_rank_iters else 0
    merged['num_rank_iters'] = sum(per_rank_iters)
    merged['num_iters_per_rank'] = per_rank_iters
    merged['num_samples'] = sum(
        int(s.get('num_samples', 0)) for s in gathered if s)
    merged['per_rank'] = [
        dict(
            rank=i,
            num_iters=int(s.get('num_iters', 0)),
            num_samples=int(s.get('num_samples', 0)))
        for i, s in enumerate(gathered) if s
    ]

    raw_seconds: Dict[str, List[float]] = defaultdict(list)
    for item in gathered:
        if not item:
            continue
        for key, values in item.get('raw_seconds', {}).items():
            raw_seconds[key].extend(float(v) for v in values)

    merged['raw_seconds'] = dict(raw_seconds)
    merged['stages_ms'] = {
        key: summarize_ms(values)
        for key, values in raw_seconds.items()
    }
    return merged


def should_log_iter(iter_idx: int, log_interval: int) -> bool:
    if log_interval <= 0:
        return False
    return (iter_idx + 1) == 1 or (iter_idx + 1) % log_interval == 0


def run_one_mode(mode: str,
                 cfg: Config,
                 args: argparse.Namespace,
                 device: torch.device,
                 rank: int = 0) -> Dict[str, Any]:
    if rank == 0:
        print(f'\n========== Running mode: {mode} ==========', flush=True)
    set_benchmark_seed(args.seed)
    model = build_model(cfg, args.checkpoint, device)
    if hasattr(model, 'set_epoch'):
        model.set_epoch(args.epoch)
    dataloader = build_loader(
        cfg, args.batch_size, args.num_workers, seed=args.seed)
    optim_obj = build_optimizer(model, cfg)
    optimizer_step_enabled = bool(args.real_optimizer_step)
    if optimizer_step_enabled and not args.keep_lr:
        set_optimizer_lr(optim_obj, 0.0)
    elif args.optimizer_step and rank == 0:
        print(
            '--optimizer-step was requested, but this benchmark keeps weights '
            'unchanged by default. Add --real-optimizer-step to actually update.',
            flush=True)

    all_stats: StageStats = defaultdict(list)
    last_log_vars: Dict[str, float] = {}
    num_samples = 0
    kept_iters = 0
    try:
        dataloader_len = len(dataloader)
    except TypeError:
        dataloader_len = None
    target_iters = dataloader_len if args.full_epoch else args.max_iter
    target_text = str(target_iters) if target_iters is not None else 'all'

    message_hub = MessageHub.get_current_instance()
    message_hub.update_info('epoch', args.epoch)

    for i, data in enumerate(dataloader):
        if (not args.full_epoch) and i >= args.max_iter:
            break
        message_hub.update_info('iter', i)
        step_stats, log_vars = run_profile_step(
            model=model,
            data=data,
            optim_obj=optim_obj,
            device=device,
            amp=args.amp,
            optimizer_step_enabled=optimizer_step_enabled,
            profile_matching=(not args.no_profile_matching))
        if log_vars.get('nonfinite_loss', 0.0):
            if args.stop_on_nonfinite:
                raise RuntimeError(
                    f'Non-finite loss in mode={mode}, iter={i + 1}.')
            if rank == 0 and should_log_iter(i, args.log_interval):
                print(
                    f'{mode} iter {i + 1}/{target_text}: '
                    'non-finite loss, skipped from summary.',
                    flush=True)
            continue
        if i < args.num_warmup:
            continue

        merge_stats(all_stats, step_stats)
        last_log_vars = log_vars
        num_samples += len(data['data_samples'])
        kept_iters += 1
        total_ms = step_stats['train_step_total'][-1] * 1000.0
        eff_q = int(step_stats.get('effective_query', [0.0])[-1] * 1000.0)
        match_q = int(
            step_stats.get('effective_matching_query', [0.0])[-1] * 1000.0)
        if rank == 0 and should_log_iter(i, args.log_interval):
            print(
                f'{mode} iter {i + 1}/{target_text}: '
                f'{total_ms:.1f} ms, query={eff_q}, match_query={match_q}, '
                f'loss={log_vars.get("loss", 0.0):.4f}',
                flush=True)

    raw_seconds = {k: [float(v) for v in vals] for k, vals in all_stats.items()}
    summary = dict(
        mode=mode,
        num_iters=kept_iters,
        num_samples=num_samples,
        batch_size=args.batch_size,
        global_batch_size=args.batch_size * get_world_size(),
        world_size=get_world_size(),
        amp=args.amp,
        optimizer_step=optimizer_step_enabled,
        seed=args.seed,
        summary_stat=args.summary_stat,
        full_epoch=args.full_epoch,
        num_warmup=args.num_warmup,
        train_enabled=bool(
            cfg.model.get('gsd_cfg', {}).get('query_group_cfg', {}).get(
                'train_enabled', False)),
        batch_sampler_type=(
            cfg.train_dataloader.batch_sampler.get('type')
            if isinstance(cfg.train_dataloader.get('batch_sampler'), dict)
            else None),
        stages_ms={k: summarize_ms(v) for k, v in raw_seconds.items()},
        raw_seconds=raw_seconds,
        last_log_vars=last_log_vars)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def print_comparison(results: List[Dict[str, Any]]) -> None:
    print('\n========== Training Query Mode Benchmark ==========')
    print('Note: baseline keeps query_group aux loss but disables train-time query cropping.')
    stat = results[0].get('summary_stat', 'mean') if results else 'mean'
    print(f'Summary stat: {stat}')
    world_size = int(results[0].get('world_size', 1)) if results else 1
    if world_size > 1:
        print(
            'Distributed summary: timings are aggregated from rank-local '
            'measurements; img/s uses global batch size.')
    header = (
        f'{"mode":<20} {"img/s":>8} {"ms/img":>9} {"ms/iter":>9} '
        f'{"query":>8} {"match_q":>8} {"decoder":>9} {"loss+match":>11} '
        f'{"matching":>9} {"backward":>9}')
    print(header)
    print('-' * len(header))

    baseline_ms = None
    for item in results:
        stages = item['stages_ms']
        iter_ms = stage_value(item, 'train_step_total', stat)
        if item['mode'] == 'baseline':
            baseline_ms = iter_ms
        global_batch_size = int(item.get('global_batch_size',
                                         item.get('batch_size', 1)))
        ms_img = iter_ms / max(1, global_batch_size)
        img_s = 1000.0 / ms_img if ms_img > 0 else 0.0
        print(
            f'{item["mode"]:<20} {img_s:>8.2f} {ms_img:>9.2f} '
            f'{iter_ms:>9.2f} {stage_value(item, "effective_query", stat):>8.1f} '
            f'{stage_value(item, "effective_matching_query", stat):>8.1f} '
            f'{stage_value(item, "decoder", stat):>9.2f} '
            f'{stage_value(item, "loss_total_with_matching", stat):>11.2f} '
            f'{stage_value(item, "matching_total", stat):>9.2f} '
            f'{stage_value(item, "backward", stat):>9.2f}')

    if baseline_ms:
        print('\n-- Speedup vs baseline --')
        for item in results:
            iter_ms = stage_value(item, 'train_step_total', stat)
            if item['mode'] == 'baseline' or iter_ms <= 0:
                continue
            speedup = baseline_ms / iter_ms
            saved = baseline_ms - iter_ms
            print(f'{item["mode"]}: {speedup:.3f}x, saved {saved:.2f} ms/iter')

    print('\n-- Query Distribution --')
    bin_values = [300, 600, 800]
    for item in results:
        total_q = raw_query_values(item, 'effective_query')
        match_q = raw_query_values(item, 'effective_matching_query')
        dn_q = raw_query_values(item, 'effective_dn_query')
        group_q = raw_query_values(item, 'group_query_per_image')
        total_stats = query_stats(total_q)
        match_stats = query_stats(match_q)
        dn_stats = query_stats(dn_q)
        print(
            f'{item["mode"]}: total_q mean={total_stats["mean"]:.1f} '
            f'range=[{total_stats["min"]:.0f},{total_stats["max"]:.0f}], '
            f'match_q mean={match_stats["mean"]:.1f} '
            f'range=[{match_stats["min"]:.0f},{match_stats["max"]:.0f}], '
            f'dn_q mean={dn_stats["mean"]:.1f}')
        print(
            f'  batch match_q: {format_bin_distribution(match_q, bin_values)}')
        if group_q:
            print(
                f'  image proxy_q: {format_bin_distribution(group_q, bin_values)}')
        else:
            print('  image proxy_q: none')


def save_outputs(results: List[Dict[str, Any]],
                 out_json: str,
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
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'mode', 'summary_stat', 'img_per_s', 'ms_per_img', 'ms_per_iter',
                'world_size', 'batch_size_per_rank', 'global_batch_size',
                'num_iters', 'num_samples', 'full_epoch',
                'effective_query', 'effective_matching_query',
                'effective_dn_query', 'decoder_ms',
                'loss_total_with_matching_ms', 'matching_total_ms',
                'backward_ms', 'batch_sampler_type', 'batch_match_query_dist',
                'image_proxy_query_dist'
            ])
            for item in results:
                stages = item['stages_ms']
                stat = item.get('summary_stat', 'mean')
                iter_ms = stage_value(item, 'train_step_total', stat)
                global_batch_size = int(item.get('global_batch_size',
                                                 item.get('batch_size', 1)))
                ms_img = iter_ms / max(1, global_batch_size)
                img_s = 1000.0 / ms_img if ms_img > 0 else 0.0
                writer.writerow([
                    item['mode'], stat, img_s, ms_img, iter_ms,
                    item.get('world_size', 1), item.get('batch_size', 1),
                    global_batch_size, item.get('num_iters', 0),
                    item.get('num_samples', 0), item.get('full_epoch', False),
                    stage_value(item, 'effective_query', stat),
                    stage_value(item, 'effective_matching_query', stat),
                    stage_value(item, 'effective_dn_query', stat),
                    stage_value(item, 'decoder', stat),
                    stage_value(item, 'loss_total_with_matching', stat),
                    stage_value(item, 'matching_total', stat),
                    stage_value(item, 'backward', stat),
                    item.get('batch_sampler_type'),
                    format_bin_distribution(
                        raw_query_values(item, 'effective_matching_query'),
                        [300, 600, 800]),
                    format_bin_distribution(
                        raw_query_values(item, 'group_query_per_image'),
                        [300, 600, 800])
                ])
        print(f'Saved CSV: {path}')


def main() -> int:
    args = parse_args()
    if (not args.full_epoch) and args.max_iter <= args.num_warmup:
        raise ValueError('--max-iter must be larger than --num-warmup.')

    ensure_repo_package('mmdet')
    from mmdet.utils import register_all_modules
    import mmdet

    mmdet_path = Path(mmdet.__file__).resolve()
    if not mmdet_path.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            f'Imported mmdet from {mmdet_path}, expected it under '
            f'{REPO_ROOT}. Please remove the other checkout from PYTHONPATH.')

    register_all_modules(init_default_scope=False)
    base_cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        base_cfg.merge_from_dict(args.cfg_options)
    init_default_scope(base_cfg.get('default_scope', 'mmdet'))

    if args.launcher != 'none':
        from mmengine.dist import get_rank, init_dist
        init_dist(args.launcher)
        rank = get_rank()
    else:
        rank = 0

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() \
        else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    results = []
    for mode in args.modes:
        if dist_initialized():
            torch.distributed.barrier()
        cfg = make_mode_cfg(base_cfg, mode, args)
        local_summary = run_one_mode(mode, cfg, args, device, rank=rank)
        summary = aggregate_summary_across_ranks(local_summary)
        if rank == 0:
            results.append(summary)
        if dist_initialized():
            torch.distributed.barrier()

    if rank == 0:
        print_comparison(results)
        save_outputs(results, args.out_json, args.out_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
