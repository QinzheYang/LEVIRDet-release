# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class BatchTraceHook(Hook):
    """Persist each rank's latest batch for distributed-hang diagnosis.

    The trace is written outside ``work_dir`` by default so this hook does not
    add per-iteration traffic to shared checkpoint storage. Each write uses an
    atomic replacement; files remain readable after NCCL aborts the process.
    """

    priority = 'VERY_HIGH'

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir
        self._trace_dir: Optional[Path] = None

    @staticmethod
    def _sample_info(sample: Any) -> Dict[str, Any]:
        metainfo = getattr(sample, 'metainfo', {}) or {}
        info = {
            'img_id': metainfo.get('img_id'),
            'img_path': metainfo.get('img_path', ''),
            'ori_shape': metainfo.get('ori_shape'),
            'img_shape': metainfo.get('img_shape'),
        }
        gt_instances = getattr(sample, 'gt_instances', None)
        if gt_instances is not None:
            bboxes = getattr(gt_instances, 'bboxes', None)
            labels = getattr(gt_instances, 'labels', None)
            if bboxes is not None:
                info['num_gt_bboxes'] = int(len(bboxes))
            if labels is not None:
                try:
                    info['gt_labels'] = labels.detach().cpu().tolist()
                except Exception:
                    info['gt_labels'] = '<unavailable>'
        return info

    @classmethod
    def _batch_info(cls, data_batch: Any) -> List[Dict[str, Any]]:
        if not isinstance(data_batch, dict):
            return []
        samples = data_batch.get('data_samples')
        if not isinstance(samples, (list, tuple)):
            return []
        return [cls._sample_info(sample) for sample in samples]

    def _resolve_trace_dir(self, runner: Runner) -> Path:
        if self._trace_dir is not None:
            return self._trace_dir
        configured = self.output_dir or os.environ.get(
            'TRAIN_BATCH_TRACE_DIR')
        if configured:
            trace_dir = Path(configured)
        else:
            job_tag = os.environ.get('TRAIN_JOB_TAG')
            if not job_tag:
                job_tag = Path(str(runner.work_dir)).name
            trace_dir = Path(tempfile.gettempdir()) / 'levirdet-batch-trace' \
                / job_tag
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_dir = trace_dir
        return trace_dir

    @staticmethod
    def _atomic_dump(path: Path, payload: Dict[str, Any]) -> None:
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        with temporary.open('w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    def _write(self, runner: Runner, batch_idx: int, data_batch: Any,
               phase: str) -> None:
        rank = get_rank()
        payload = {
            'phase': phase,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'rank': rank,
            'local_rank': int(os.environ.get('LOCAL_RANK', rank)),
            'epoch_zero_based': int(runner.epoch),
            'epoch_one_based': int(runner.epoch) + 1,
            'batch_idx_zero_based': int(batch_idx),
            'runner_iter_zero_based': int(runner.iter),
            'samples': self._batch_info(data_batch),
        }
        path = self._resolve_trace_dir(runner) / f'rank_{rank}.json'
        self._atomic_dump(path, payload)

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch: Optional[dict] = None) -> None:
        self._write(runner, batch_idx, data_batch, 'train_step_started')

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        self._write(runner, batch_idx, data_batch, 'train_step_completed')
