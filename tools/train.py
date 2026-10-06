# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

#from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmdet.utils import register_all_modules, setup_cache_size_limit_of_dynamo
from tools.misc.loader import build_dataloader

Runner.build_dataloader = staticmethod(build_dataloader)

def _extract_oom_sample_info(data_batch: Any) -> List[Tuple[int, str, int]]:
    """Collect per-sample metadata for OOM diagnostics.

    Returns:
        list[tuple[int, str, int]]: (sample_index, img_path, num_gt_bboxes)
    """
    results: List[Tuple[int, str, int]] = []
    if not isinstance(data_batch, dict):
        return results

    data_samples = data_batch.get('data_samples')
    if not isinstance(data_samples, (list, tuple)):
        return results

    for i, sample in enumerate(data_samples):
        img_path = ''
        num_gt = -1
        if hasattr(sample, 'metainfo'):
            img_path = sample.metainfo.get('img_path', '')
        if hasattr(sample, 'gt_instances'):
            gt_instances = sample.gt_instances
            if hasattr(gt_instances, 'bboxes'):
                num_gt = int(gt_instances.bboxes.shape[0])
        results.append((i, img_path, num_gt))
    return results


def _patch_train_step_for_oom_diagnostics(runner: Runner) -> None:
    """Patch model.train_step to print batch/sample info when CUDA OOM occurs."""
    original_train_step = runner.model.train_step

    def _safe_log_oom(message: str) -> None:
        """Best-effort OOM logging that never raises secondary errors."""
        try:
            print_log(message, logger='current', level=logging.ERROR)
        except Exception:
            # Fallback to plain stderr printing when logger is unavailable.
            print(message, flush=True)

    def wrapped_train_step(data: Dict[str, Any], optim_wrapper: Any):
        try:
            return original_train_step(data, optim_wrapper)
        except RuntimeError as err:
            if 'out of memory' in str(err).lower():
                rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                _safe_log_oom(
                    f'[OOM-DIAG][rank{rank}] CUDA OOM captured in train_step. '
                    'Dumping batch sample info...')
                for sample_idx, img_path, num_gt in _extract_oom_sample_info(data):
                    _safe_log_oom(
                        f'[OOM-DIAG][rank{rank}] sample={sample_idx}, '
                         f'num_gt_bboxes={num_gt}, img_path={img_path}')
                # Clear cache to make sure logging and shutdown can proceed.
                torch.cuda.empty_cache()
            raise

    runner.model.train_step = wrapped_train_step


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # Ensure all mmdet modules are imported and registered before building
    # objects from registries (e.g., custom detectors such as DEIMV2).
    register_all_modules()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    _patch_train_step_for_oom_diagnostics(runner)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
