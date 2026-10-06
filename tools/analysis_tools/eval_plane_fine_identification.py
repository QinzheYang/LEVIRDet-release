import argparse
import json
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
from mmengine.config import Config, DictAction
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmdet.registry import METRICS, RUNNERS
from mmdet.utils import register_all_modules, setup_cache_size_limit_of_dynamo


TARGET_CLASSES = (
    'B-1_Lancer', 'B-2_Spirit', 'B-21', 'F-2A', 'F-15',
    'F-16F_Fighting_Falcon', 'F-22', 'F-35B_Lightning_II', 'KC-10',
    'KC-130', 'KJ-600', 'KJ-2000', 'E3', 'XQ-58A', 'YFQ-42A',
    'YFQ-44A')


def bbox_iou_matrix(pred_bboxes, gt_bboxes):
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return np.zeros((len(pred_bboxes), len(gt_bboxes)), dtype=np.float32)

    pred = np.asarray(pred_bboxes, dtype=np.float32)
    gt = np.asarray(gt_bboxes, dtype=np.float32)
    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    pred_area = np.maximum(pred[:, 2] - pred[:, 0], 0.0) * np.maximum(
        pred[:, 3] - pred[:, 1], 0.0)
    gt_area = np.maximum(gt[:, 2] - gt[:, 0], 0.0) * np.maximum(
        gt[:, 3] - gt[:, 1], 0.0)
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0)


def maximum_iou_matching(iou_matrix, iou_thr, pred_scores):
    """Return maximum one-to-one matches in the IoU-threshold graph."""
    num_pred, num_gt = iou_matrix.shape
    if num_pred == 0 or num_gt == 0:
        return 0

    pred_order = np.argsort(-np.asarray(pred_scores), kind='stable')
    adjacency = []
    for pred_idx in range(num_pred):
        valid = np.flatnonzero(iou_matrix[pred_idx] >= iou_thr)
        valid = valid[np.argsort(-iou_matrix[pred_idx, valid], kind='stable')]
        adjacency.append(valid.tolist())

    matched_pred_by_gt = [-1] * num_gt

    def augment(pred_idx, visited_gt):
        for gt_idx in adjacency[pred_idx]:
            if visited_gt[gt_idx]:
                continue
            visited_gt[gt_idx] = True
            previous_pred = matched_pred_by_gt[gt_idx]
            if previous_pred == -1 or augment(previous_pred, visited_gt):
                matched_pred_by_gt[gt_idx] = pred_idx
                return True
        return False

    matches = 0
    for pred_idx in pred_order:
        if augment(int(pred_idx), [False] * num_gt):
            matches += 1
    return matches


def to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@METRICS.register_module()
class PlaneFineIdentificationAccuracyMetric(BaseMetric):
    """Class-aware detection recall named P by the experiment definition."""

    default_prefix = 'identification'

    def __init__(self,
                 model_classes,
                 target_classes=TARGET_CLASSES,
                 iou_thr=0.5,
                 score_thr=0.3,
                 output_file=None,
                 collect_device='cpu',
                 prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.model_classes = tuple(model_classes)
        self.target_classes = tuple(target_classes)
        self.target_class_set = set(self.target_classes)
        self.iou_thr = float(iou_thr)
        self.score_thr = float(score_thr)
        self.output_file = output_file

        missing = set(self.target_classes) - set(self.model_classes)
        if missing:
            raise ValueError(
                f'Target classes are absent from model_classes: {sorted(missing)}')

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            if 'instances' not in data_sample:
                raise KeyError(
                    'Validation sample has no raw `instances`. Add `instances` '
                    'to PackDetInputs.meta_keys in the test pipeline.')

            gt_bboxes = []
            gt_labels = []
            for instance in data_sample['instances']:
                label = int(instance['bbox_label'])
                class_name = self.model_classes[label]
                if class_name in self.target_class_set:
                    gt_bboxes.append(instance['bbox'])
                    gt_labels.append(class_name)

            pred = data_sample['pred_instances']
            bboxes = to_numpy(pred['bboxes']).reshape(-1, 4)
            labels = to_numpy(pred['labels']).astype(np.int64).reshape(-1)
            scores = to_numpy(pred['scores']).reshape(-1)
            valid = scores >= self.score_thr
            bboxes = bboxes[valid]
            labels = labels[valid]
            scores = scores[valid]

            pred_bboxes = []
            pred_labels = []
            pred_scores = []
            for bbox, label, score in zip(bboxes, labels, scores):
                if label < 0 or label >= len(self.model_classes):
                    continue
                class_name = self.model_classes[int(label)]
                if class_name in self.target_class_set:
                    pred_bboxes.append(bbox.tolist())
                    pred_labels.append(class_name)
                    pred_scores.append(float(score))

            self.results.append(
                dict(
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    pred_bboxes=pred_bboxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores))

    def compute_metrics(self, results):
        gt_count = OrderedDict((name, 0) for name in self.target_classes)
        correct_count = OrderedDict((name, 0) for name in self.target_classes)

        for result in results:
            gt_labels = np.asarray(result['gt_labels'], dtype=object)
            pred_labels = np.asarray(result['pred_labels'], dtype=object)
            gt_bboxes = np.asarray(result['gt_bboxes'], dtype=np.float32).reshape(
                -1, 4)
            pred_bboxes = np.asarray(
                result['pred_bboxes'], dtype=np.float32).reshape(-1, 4)
            pred_scores = np.asarray(result['pred_scores'], dtype=np.float32)

            for class_name in self.target_classes:
                gt_indices = np.flatnonzero(gt_labels == class_name)
                pred_indices = np.flatnonzero(pred_labels == class_name)
                gt_count[class_name] += len(gt_indices)
                if len(gt_indices) == 0 or len(pred_indices) == 0:
                    continue
                ious = bbox_iou_matrix(
                    pred_bboxes[pred_indices], gt_bboxes[gt_indices])
                correct_count[class_name] += maximum_iou_matching(
                    ious, self.iou_thr, pred_scores[pred_indices])

        total_gt = sum(gt_count.values())
        total_correct = sum(correct_count.values())
        overall_p = total_correct / total_gt if total_gt else 0.0

        summary = OrderedDict(
            iou_threshold=self.iou_thr,
            score_threshold=self.score_thr,
            total_correct=total_correct,
            total_samples=total_gt,
            P=overall_p,
            per_class=OrderedDict())
        metrics = OrderedDict(
            P=overall_p,
            correct=float(total_correct),
            total=float(total_gt))

        logger = MMLogger.get_current_instance()
        rows = ['class | correct | total | P', '-' * 52]
        for class_name in self.target_classes:
            count = gt_count[class_name]
            correct = correct_count[class_name]
            class_p = correct / count if count else 0.0
            summary['per_class'][class_name] = dict(
                correct=correct, total=count, P=class_p)
            metrics[f'P_{class_name}'] = class_p
            rows.append(
                f'{class_name:<27} | {correct:>7} | {count:>5} | '
                f'{class_p:.6f}')
        rows.append('-' * 52)
        rows.append(
            f'overall                     | {total_correct:>7} | '
            f'{total_gt:>5} | {overall_p:.6f}')
        logger.info(
            '\nPlane fine-grained identification accuracy\n' + '\n'.join(rows))

        if self.output_file:
            output_dir = osp.dirname(osp.abspath(self.output_file))
            os.makedirs(output_dir, exist_ok=True)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f'Identification summary saved to {self.output_file}')

        return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-grained plane identification accuracy P.')
    parser.add_argument('config', help='model config file')
    parser.add_argument('checkpoint', help='model checkpoint file')
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Prediction score threshold. Use 0 to disable score filtering.')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--work-dir')
    parser.add_argument('--out-json')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None)
    parser.add_argument(
        '--launcher',
        choices=('none', 'pytorch', 'slurm', 'mpi'),
        default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def add_instances_to_test_pipeline(cfg):
    dataset_cfg = cfg.test_dataloader.dataset
    while 'dataset' in dataset_cfg:
        dataset_cfg = dataset_cfg.dataset
    for transform in reversed(dataset_cfg.pipeline):
        if transform.get('type') == 'PackDetInputs':
            meta_keys = list(transform.get('meta_keys', ()))
            if 'instances' not in meta_keys:
                meta_keys.append('instances')
                transform['meta_keys'] = tuple(meta_keys)
            return
    raise RuntimeError('Cannot find PackDetInputs in the test pipeline.')


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.checkpoint
    cfg.resume = False
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.batch_size:
        cfg.test_dataloader.batch_size = args.batch_size

    add_instances_to_test_pipeline(cfg)
    cfg.test_evaluator = dict(
        type='PlaneFineIdentificationAccuracyMetric',
        model_classes=tuple(cfg.classes),
        target_classes=TARGET_CLASSES,
        iou_thr=args.iou_thr,
        score_thr=args.score_thr,
        output_file=args.out_json)

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.test()


if __name__ == '__main__':
    main()
