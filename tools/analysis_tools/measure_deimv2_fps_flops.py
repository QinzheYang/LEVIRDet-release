# Copyright (c) OpenMMLab. All rights reserved.
"""Measure inference FPS and FLOPs for a DEIMV2/GSD config.

Examples:
    python tools/analysis_tools/measure_deimv2_fps_flops.py \
        --checkpoint work_dirs/xxx/best_coco_bbox_mAP_epoch_120.pth

    python tools/analysis_tools/measure_deimv2_fps_flops.py \
        configs/deimv2/deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det.py \
        --checkpoint epoch_117.pth --max-iter 200 --num-warmup 20 \
        --flops-images 20 --batch-size 1 --num-workers 0
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError as exc:
    raise ImportError('Please install/upgrade mmengine with analysis support.') from exc

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


DEFAULT_CONFIG = (
    'configs/deimv2/deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det.py')


class DEIMV2InferenceFlopsWrapper(torch.nn.Module):
    """Tensor-only inference path for FLOPs analysis.

    ``DetectionTransformer._forward`` does not pass ``batch_data_samples`` to
    ``pre_decoder``. DEIMV2GSDGuided needs those samples to activate GSD-guided
    query modulation/grouping, so this wrapper mirrors the predict path up to
    bbox-head tensor outputs while leaving post-processing out of FLOPs.
    """

    def __init__(self, model: torch.nn.Module, data_samples) -> None:
        super().__init__()
        self.model = model
        self.data_samples = data_samples

    def forward(self, batch_inputs: torch.Tensor):
        if hasattr(self.model, '_last_query_group_logits'):
            self.model._last_query_group_logits = None
        if hasattr(self.model, '_last_group_queries'):
            self.model._last_group_queries = None
        if hasattr(self.model, '_update_online_gsd_cache'):
            self.model._update_online_gsd_cache(batch_inputs)

        img_feats = self.model.extract_feat(batch_inputs)
        encoder_inputs_dict, decoder_inputs_dict = self.model.pre_transformer(
            img_feats, self.data_samples)
        encoder_outputs_dict = self.model.forward_encoder(
            **encoder_inputs_dict)
        tmp_dec_in, head_inputs_dict = self.model.pre_decoder(
            **encoder_outputs_dict,
            batch_data_samples=self.data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.model.forward_decoder(
            **decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return self.model.bbox_head.forward(**head_inputs_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure DEIMV2 inference FPS and FLOPs.')
    parser.add_argument(
        'config',
        nargs='?',
        default=DEFAULT_CONFIG,
        help='config file path')
    parser.add_argument(
        '--checkpoint',
        default='',
        help='checkpoint path. If omitted, random/config init weights are used')
    parser.add_argument(
        '--dataset-type',
        choices=['val', 'test'],
        default='test',
        help='which dataloader to benchmark')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--num-warmup', type=int, default=20)
    parser.add_argument(
        '--flops-images',
        type=int,
        default=20,
        help='number of dataloader images used to average FLOPs')
    parser.add_argument(
        '--skip-fps',
        action='store_true',
        help='only compute FLOPs')
    parser.add_argument(
        '--skip-flops',
        action='store_true',
        help='only measure FPS')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='measure FPS with CUDA autocast fp16')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='fuse Conv-BN before benchmarking when supported')
    parser.add_argument(
        '--out',
        default='',
        help='optional JSON file to save the summary')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, key=value format')
    parser.add_argument('--local-rank', type=int, default=0)
    return parser.parse_args()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def build_dataloader(cfg: Config, dataset_type: str, batch_size: int,
                     num_workers: int):
    dataloader_cfg = copy.deepcopy(
        cfg.val_dataloader if dataset_type == 'val' else cfg.test_dataloader)
    dataloader_cfg.batch_size = batch_size
    dataloader_cfg.num_workers = num_workers
    dataloader_cfg.persistent_workers = False
    if num_workers == 0:
        dataloader_cfg.collate_fn = pseudo_collate
    else:
        dataloader_cfg.setdefault('collate_fn', dict(type='pseudo_collate'))
    return Runner.build_dataloader(dataloader_cfg)


def build_model(cfg: Config, checkpoint: str, device: torch.device,
                fuse_conv_bn: bool) -> torch.nn.Module:
    model = MODELS.build(cfg.model)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model = revert_sync_batchnorm(model)
    if fuse_conv_bn:
        from mmcv.cnn import fuse_conv_bn as mmcv_fuse_conv_bn
        model = mmcv_fuse_conv_bn(model)
    model.to(device)
    model.eval()
    return model


def measure_fps(model: torch.nn.Module, dataloader, device: torch.device,
                max_iter: int, num_warmup: int, amp: bool) -> Dict[str, Any]:
    if max_iter <= num_warmup:
        raise ValueError('--max-iter must be larger than --num-warmup.')

    elapsed: List[float] = []
    num_images = 0
    amp_enabled = amp and device.type == 'cuda'

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= max_iter:
                break

            sync_if_cuda(device)
            start = time.perf_counter()
            with torch.autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=amp_enabled):
                model.test_step(data)
            sync_if_cuda(device)
            duration = time.perf_counter() - start

            if idx >= num_warmup:
                elapsed.append(duration)
                num_images += len(data['data_samples'])

    total = sum(elapsed)
    fps = num_images / total if total > 0 else 0.0
    ms_per_image = 1000.0 / fps if fps > 0 else 0.0
    return dict(
        fps=fps,
        ms_per_image=ms_per_image,
        num_iters=len(elapsed),
        num_images=num_images,
        mean_ms_per_batch=statistics.mean(elapsed) * 1000.0
        if elapsed else 0.0,
        median_ms_per_batch=statistics.median(elapsed) * 1000.0
        if elapsed else 0.0)


def measure_flops(model: torch.nn.Module, dataloader, flops_images: int,
                  device: torch.device) -> Dict[str, Any]:
    flops_values: List[int] = []
    params: Optional[int] = None
    input_shapes = []

    for idx, data in enumerate(dataloader):
        if idx >= flops_images:
            break
        if len(data['data_samples']) != 1:
            raise ValueError('FLOPs measurement expects batch_size=1.')

        with torch.no_grad():
            processed = model.data_preprocessor(data, False)
        batch_inputs = processed['inputs']
        data_samples = processed['data_samples']
        wrapper = DEIMV2InferenceFlopsWrapper(model, data_samples).to(device)
        wrapper.eval()

        outputs = get_model_complexity_info(
            wrapper,
            None,
            inputs=batch_inputs,
            show_table=False,
            show_arch=False)
        flops_values.append(int(outputs['flops']))
        params = int(outputs['params'])

        sample = data_samples[0]
        shape = getattr(sample, 'batch_input_shape',
                        getattr(sample, 'pad_shape', None))
        input_shapes.append(tuple(shape) if shape is not None else None)

    if not flops_values:
        raise RuntimeError('No samples were processed for FLOPs measurement.')

    mean_flops = int(statistics.mean(flops_values))
    return dict(
        flops=mean_flops,
        flops_formatted=_format_size(mean_flops),
        params=params,
        params_formatted=_format_size(params) if params is not None else '',
        num_images=len(flops_values),
        input_shapes=sorted({str(v) for v in input_shapes}))


def print_summary(config: str, checkpoint: str, device: torch.device,
                  fps_result: Optional[Dict[str, Any]],
                  flops_result: Optional[Dict[str, Any]]) -> None:
    print('\n========== DEIMV2 FPS / FLOPs ==========')
    print(f'Config: {config}')
    print(f'Checkpoint: {checkpoint or "<not loaded>"}')
    print(f'Device: {device}')
    if fps_result is not None:
        print('\n[Inference speed]')
        print(f"FPS: {fps_result['fps']:.2f} img/s")
        print(f"Latency: {fps_result['ms_per_image']:.2f} ms/img")
        print(f"Iterations: {fps_result['num_iters']}, "
              f"images: {fps_result['num_images']}")
        print(f"Mean batch latency: "
              f"{fps_result['mean_ms_per_batch']:.2f} ms")
    if flops_result is not None:
        print('\n[Complexity]')
        print(f"FLOPs: {flops_result['flops_formatted']} "
              f"({flops_result['flops']})")
        print(f"Params: {flops_result['params_formatted']} "
              f"({flops_result['params']})")
        print(f"FLOPs samples: {flops_result['num_images']}")
        print(f"Input shapes: {', '.join(flops_result['input_shapes'])}")
        print('Note: FLOPs include the online GSD branch and transformer '
              'tensor path, but exclude post-processing/NMS and may miss ops '
              'unsupported by mmengine analysis.')


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

    fps_result = None
    if not args.skip_fps:
        fps_loader = build_dataloader(
            cfg, args.dataset_type, args.batch_size, args.num_workers)
        fps_result = measure_fps(
            model=model,
            dataloader=fps_loader,
            device=device,
            max_iter=args.max_iter,
            num_warmup=args.num_warmup,
            amp=args.amp)

    flops_result = None
    if not args.skip_flops:
        flops_loader = build_dataloader(
            cfg, args.dataset_type, batch_size=1, num_workers=0)
        flops_result = measure_flops(
            model=model,
            dataloader=flops_loader,
            flops_images=args.flops_images,
            device=device)

    print_summary(args.config, args.checkpoint, device, fps_result,
                  flops_result)

    if args.out:
        out = dict(
            config=args.config,
            checkpoint=args.checkpoint,
            device=str(device),
            fps=fps_result,
            flops=flops_result)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding='utf-8')
        print(f'\nSaved summary: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
