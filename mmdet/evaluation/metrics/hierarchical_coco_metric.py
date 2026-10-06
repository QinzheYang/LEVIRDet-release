# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from mmengine.fileio import dump, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.utils.hierarchy import build_class_hierarchy
from .coco_metric import CocoMetric


@METRICS.register_module()
class HierarchicalCocoMetric(CocoMetric):
    """COCO bbox metric with coarse and hierarchy-compatible views."""

    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 class_names: Sequence[str],
                 class_hierarchy: Mapping[str, Sequence[str]],
                 base_classes: Sequence[str],
                 coarse_30_weight: float = 0.7,
                 coarse_all_weight: float = 0.3,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if self.metrics != ['bbox']:
            raise ValueError('HierarchicalCocoMetric currently supports bbox only.')

        self.class_names = tuple(class_names)
        self.class_hierarchy = dict(class_hierarchy)
        self.base_classes = tuple(base_classes)
        self.coarse_30_weight = coarse_30_weight
        self.coarse_all_weight = coarse_all_weight

        hierarchy = build_class_hierarchy(self.class_names,
                                          self.class_hierarchy)
        parent = hierarchy['parent'].tolist()
        self.ancestor_paths = []
        for label in range(len(self.class_names)):
            path = []
            cur = label
            while cur != -1:
                path.append(cur)
                cur = int(parent[cur])
            self.ancestor_paths.append(path)

        base_name_to_label = {
            name: idx
            for idx, name in enumerate(self.base_classes)
        }
        self.label_to_base_label = []
        for path in self.ancestor_paths:
            root_name = self.class_names[path[-1]]
            if root_name not in base_name_to_label:
                raise KeyError(
                    f'Hierarchy root {root_name!r} is not in base_classes.')
            self.label_to_base_label.append(base_name_to_label[root_name])

    def _categories(self, names: Sequence[str]) -> List[dict]:
        return [dict(id=idx, name=name) for idx, name in enumerate(names)]

    def _gt_json(self, gt_dicts: Sequence[dict], names: Sequence[str],
                 label_mode: str) -> dict:
        image_infos = []
        annotations = []
        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_infos.append(
                dict(
                    id=img_id,
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                    file_name=''))
            for ann in gt_dict['anns']:
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]
                label = int(ann['bbox_label'])
                if label_mode == 'coarse_30':
                    labels = [self.label_to_base_label[label]]
                elif label_mode == 'coarse_all':
                    labels = self.ancestor_paths[label]
                else:
                    raise ValueError(f'Unsupported label_mode {label_mode!r}.')

                for out_label in labels:
                    annotations.append(
                        dict(
                            id=len(annotations) + 1,
                            image_id=img_id,
                            bbox=coco_bbox,
                            iscrowd=ann.get('ignore_flag', 0),
                            category_id=int(out_label),
                            area=coco_bbox[2] * coco_bbox[3]))

        return dict(
            info=dict(
                date_created=str(datetime.datetime.now()),
                description='Hierarchy-aware COCO json converted by mmdet.'),
            images=image_infos,
            categories=self._categories(names),
            licenses=None,
            annotations=annotations)

    def _gt_label_sets(self, gt_dicts: Sequence[dict]) -> List[set]:
        label_sets = []
        for gt_dict in gt_dicts:
            labels = set()
            for ann in gt_dict['anns']:
                labels.update(self.ancestor_paths[int(ann['bbox_label'])])
            label_sets.append(labels)
        return label_sets

    def _pred_json(self, preds: Sequence[dict], label_mode: str,
                   gt_label_sets: Optional[Sequence[set]] = None) -> List[dict]:
        results = []
        for idx, pred in enumerate(preds):
            image_id = pred.get('img_id', idx)
            labels = pred['labels']
            bboxes = pred['bboxes']
            scores = pred['scores']
            gt_labels = gt_label_sets[idx] if gt_label_sets is not None else None

            for i, label in enumerate(labels):
                label = int(label)
                if label_mode == 'coarse_30':
                    out_labels = [self.label_to_base_label[label]]
                elif label_mode == 'coarse_all':
                    out_labels = [
                        path_label for path_label in self.ancestor_paths[label]
                        if path_label in gt_labels
                    ]
                    if not out_labels:
                        out_labels = [label]
                else:
                    raise ValueError(
                        f'Unsupported label_mode {label_mode!r}.')

                for out_label in out_labels:
                    results.append(
                        dict(
                            image_id=image_id,
                            bbox=self.xyxy2xywh(bboxes[i]),
                            score=float(scores[i]),
                            category_id=int(out_label)))
        return results

    def _evaluate_one(self, gt_json: dict, pred_json: List[dict],
                      outfile_prefix: str, name: str,
                      logger: MMLogger) -> Dict[str, float]:
        gt_path = f'{outfile_prefix}.{name}.gt.json'
        pred_path = f'{outfile_prefix}.{name}.bbox.json'
        dump(gt_json, gt_path)
        dump(pred_json, pred_path)

        coco_gt = COCO(gt_path)
        cat_ids = coco_gt.get_cat_ids(cat_names=[
            category['name'] for category in gt_json['categories']
        ])
        img_ids = coco_gt.get_img_ids()
        predictions = load(pred_path)
        metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
        }
        if len(predictions) == 0:
            logger.error(f'The {name} testing results are empty.')
            return OrderedDict(
                (f'{name}_bbox_{item}', 0.0) for item in metric_names)

        coco_dt = coco_gt.loadRes(predictions)
        if self.use_mp_eval:
            coco_eval = COCOevalMP(coco_gt, coco_dt, 'bbox')
        else:
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = img_ids
        coco_eval.params.maxDets = list(self.proposal_nums)
        coco_eval.params.iouThrs = self.iou_thrs
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = OrderedDict()
        if self.classwise:
            results.update(
                self._classwise_results(coco_eval, coco_gt, cat_ids, name,
                                        logger))

        for item, stat_idx in metric_names.items():
            results[f'{name}_bbox_{item}'] = float(coco_eval.stats[stat_idx])
        logger.info(
            f'{name}_bbox_mAP_copypaste: '
            f'{coco_eval.stats[0]:.4f} {coco_eval.stats[1]:.4f} '
            f'{coco_eval.stats[2]:.4f} {coco_eval.stats[3]:.4f} '
            f'{coco_eval.stats[4]:.4f} {coco_eval.stats[5]:.4f}')
        return results

    def _classwise_results(self, coco_eval: COCOeval, coco_gt: COCO,
                           cat_ids: Sequence[int], name: str,
                           logger: MMLogger) -> Dict[str, float]:
        """Log AP for each category from the COCO precision tensor."""
        precisions = coco_eval.eval['precision']
        # precision shape: (iou, recall, class, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]

        def _mean_precision(precision: np.ndarray) -> float:
            precision = precision[precision > -1]
            if precision.size:
                return float(np.mean(precision))
            return float('nan')

        def _get_ann_ids(coco: COCO,
                         cat_id: int,
                         iscrowd: Optional[bool] = None) -> list:
            if hasattr(coco, 'get_ann_ids'):
                kwargs = dict(cat_ids=[cat_id])
                if iscrowd is not None:
                    kwargs['iscrowd'] = iscrowd
                return coco.get_ann_ids(**kwargs)
            kwargs = dict(catIds=[cat_id])
            if iscrowd is not None:
                kwargs['iscrowd'] = iscrowd
            return coco.getAnnIds(**kwargs)

        results = OrderedDict()
        results_per_category = []
        for idx, cat_id in enumerate(cat_ids):
            category = coco_gt.loadCats(cat_id)[0]
            category_name = category['name']

            row = [category_name]
            ap = _mean_precision(precisions[:, :, idx, 0, -1])
            row.append(f'{round(ap, 3)}')
            results[f'{name}_{category_name}_precision'] = round(ap, 3)
            if np.isnan(ap):
                total_gt_count = len(_get_ann_ids(coco_gt, cat_id))
                non_ignored_gt_count = len(
                    _get_ann_ids(coco_gt, cat_id, iscrowd=False))
                ignored_gt_count = total_gt_count - non_ignored_gt_count
                dt_count = 0
                if coco_eval.cocoDt is not None:
                    dt_count = len(_get_ann_ids(coco_eval.cocoDt, cat_id))
                logger.warning(
                    f'{name} category {category_name!r} AP is NaN. '
                    'COCOeval saw '
                    f'non_ignored_gt={non_ignored_gt_count}, '
                    f'ignored_gt={ignored_gt_count}, '
                    f'total_gt={total_gt_count}, dt_count={dt_count}. '
                    'This means this category has no non-ignored GT in the '
                    'evaluated dataset.')

            for iou_idx in [0, 5]:
                if iou_idx < precisions.shape[0]:
                    ap = _mean_precision(precisions[iou_idx, :, idx, 0, -1])
                else:
                    ap = float('nan')
                row.append(f'{round(ap, 3)}')

            for area_idx in [1, 2, 3]:
                ap = _mean_precision(precisions[:, :, idx, area_idx, -1])
                row.append(f'{round(ap, 3)}')
            results_per_category.append(tuple(row))

        if not results_per_category:
            return results

        num_columns = len(results_per_category[0])
        results_flatten = list(itertools.chain(*results_per_category))
        headers = [
            f'{name} category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m',
            'mAP_l'
        ]
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns] for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        logger.info(f'\n{name}_bbox_per_category_AP:\n{table.table}')
        return results

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        coarse_30_gt = self._gt_json(gts, self.base_classes, 'coarse_30')
        coarse_30_pred = self._pred_json(preds, 'coarse_30')
        eval_results.update(
            self._evaluate_one(coarse_30_gt, coarse_30_pred, outfile_prefix,
                               'coarse_30', logger))

        coarse_all_gt = self._gt_json(gts, self.class_names, 'coarse_all')
        coarse_all_pred = self._pred_json(
            preds, 'coarse_all', gt_label_sets=self._gt_label_sets(gts))
        eval_results.update(
            self._evaluate_one(coarse_all_gt, coarse_all_pred, outfile_prefix,
                               'coarse_all', logger))

        weighted_items = ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m',
                          'mAP_l')
        weight_sum = self.coarse_30_weight + self.coarse_all_weight
        for item in weighted_items:
            coarse_30 = eval_results[f'coarse_30_bbox_{item}']
            coarse_all = eval_results[f'coarse_all_bbox_{item}']
            eval_results[f'weighted_bbox_{item}'] = (
                coarse_30 * self.coarse_30_weight +
                coarse_all * self.coarse_all_weight) / weight_sum

        logger.info(
            'weighted_bbox_mAP: '
            f'{eval_results["weighted_bbox_mAP"]:.4f} '
            f'(coarse_30={self.coarse_30_weight}, '
            f'coarse_all={self.coarse_all_weight})')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
