# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.layers.transformer.dfine_layers import bbox2distance
from mmdet.models.losses import VarifocalLoss
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.structures.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean
from mmdet.utils.hierarchy import build_class_hierarchy
from ..utils import multi_apply
from .deimv2_head import DEIMV2Head
from .dfine_head import unimodal_distribution_focal_loss


@MODELS.register_module()
class HierarchicalDEIMV2Head(DEIMV2Head):
    """DEIMV2 head with hierarchy-aware sigmoid classification targets."""

    def __init__(self,
                 *args,
                 class_names: Sequence[str],
                 class_hierarchy: Mapping[str, Sequence[str]],
                 sibling_negative_weight: float = 0.25,
                 family_negative_weight: float = 0.5,
                 descendant_negative_weight: float = 0.0,
                 descendant_match_weight: float = 1.0,
                 collapse_tree_predictions: bool = True,
                 use_path_score: bool = True,
                 depth_score_factor: float = 0.05,
                 **kwargs) -> None:
        self.class_names = tuple(class_names)
        self.class_hierarchy = dict(class_hierarchy)
        self.sibling_negative_weight = sibling_negative_weight
        self.family_negative_weight = family_negative_weight
        self.descendant_negative_weight = descendant_negative_weight
        self.descendant_match_weight = descendant_match_weight
        self.collapse_tree_predictions = collapse_tree_predictions
        self.use_path_score = use_path_score
        self.depth_score_factor = depth_score_factor
        super().__init__(*args, **kwargs)

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                '`class_names` length must equal `num_classes`, got '
                f'{len(self.class_names)} and {self.num_classes}.')

        hierarchy = build_class_hierarchy(
            self.class_names,
            self.class_hierarchy,
            sibling_negative_weight=sibling_negative_weight,
            family_negative_weight=family_negative_weight,
            descendant_negative_weight=descendant_negative_weight,
            descendant_match_weight=descendant_match_weight)
        self.register_buffer(
            'hierarchy_positive_mask', hierarchy['ancestor'], persistent=False)
        self.register_buffer(
            'hierarchy_loss_weight', hierarchy['loss_weight'], persistent=False)
        self.register_buffer(
            'hierarchy_depth', hierarchy['depth'], persistent=False)

    def _hierarchical_cls_loss(self, cls_scores: Tensor, bbox_preds: Tensor,
                               labels: Tensor, label_weights: Tensor,
                               bbox_targets: Tensor,
                               cls_avg_factor: float) -> Tensor:
        """Build multi-label path targets and compute varifocal-style loss."""
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero()
        pos_inds = pos_inds.squeeze(1)
        cls_iou_targets = cls_scores.new_zeros(cls_scores.shape)
        cls_weights = cls_scores.new_ones(cls_scores.shape)

        if pos_inds.numel() > 0:
            pos_labels = labels[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            pos_ious = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).type_as(cls_scores)

            pos_path = self.hierarchy_positive_mask[pos_labels].to(
                device=cls_scores.device, dtype=cls_scores.dtype)
            pos_weights = self.hierarchy_loss_weight[pos_labels].to(
                device=cls_scores.device, dtype=cls_scores.dtype)
            cls_iou_targets[pos_inds] = pos_path * pos_ious[:, None]
            cls_weights[pos_inds] = pos_weights

        cls_weights = cls_weights * label_weights[:, None].to(cls_weights.dtype)
        return self.loss_cls(
            cls_scores,
            cls_iou_targets,
            weight=cls_weights,
            avg_factor=cls_avg_factor)

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            bbox_corners: Optional[Tensor],
                            teacher: Optional[Tuple[Tensor, Tensor]],
                            batch_match_indices: Optional[List[Tuple[Tensor,
                                                                     Tensor]]],
                            initial_bbox_preds: Optional[Tensor],
                            merged_match_indices: List[Tuple[Tensor, Tensor]],
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        if not isinstance(self.loss_cls, VarifocalLoss):
            return super().loss_by_feat_single(
                cls_scores, bbox_preds, bbox_corners, teacher,
                batch_match_indices, initial_bbox_preds, merged_match_indices,
                batch_gt_instances, batch_img_metas)

        num_imgs, num_queries, _ = cls_scores.shape
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         bbox_num_pos_list) = multi_apply(
             self._get_cls_targets_single,
             batch_match_indices,
             batch_gt_instances,
             batch_img_metas,
             num_queries=num_queries,
             device=bbox_preds.device)
        num_total_pos = sum(bbox_num_pos_list)
        num_total_neg = num_imgs * num_queries - num_total_pos
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])).item()
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self._hierarchical_cls_loss(
            cls_scores, bbox_preds, labels, label_weights, bbox_targets,
            cls_avg_factor)

        if num_queries not in self.cached_bbox_targets or not self.use_uni_set:
            if self.use_uni_set:
                (bbox_targets_list, bbox_weights_list,
                 bbox_num_pos_list) = multi_apply(
                     self._get_bbox_targets_single,
                     merged_match_indices,
                     batch_gt_instances,
                     batch_img_metas,
                     num_queries=num_queries,
                     device=bbox_preds.device)
                num_total_bbox_pos = sum(bbox_num_pos_list)
                bbox_targets = torch.cat(bbox_targets_list, 0)
            else:
                num_total_bbox_pos = num_total_pos

            bbox_weights = torch.cat(bbox_weights_list, 0)
            bbox_avg_factor = bbox_preds.new_tensor([num_total_bbox_pos])
            bbox_avg_factor = torch.clamp(
                reduce_mean(bbox_avg_factor), min=1).item()
            self.cached_bbox_targets[num_queries] = (bbox_targets,
                                                     bbox_weights,
                                                     num_total_bbox_pos,
                                                     bbox_avg_factor)
        else:
            (bbox_targets, bbox_weights, num_total_bbox_pos,
             bbox_avg_factor) = self.cached_bbox_targets[num_queries]

        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=bbox_avg_factor)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=bbox_avg_factor)

        if bbox_corners is None:
            loss_fgl = loss_ddf = None
            with_fgl_loss = with_dff_loss = False
        else:
            loss_fgl = loss_ddf = bbox_corners.new_tensor(0)
            with_fgl_loss = self.fgl_loss_weight is not None
            with_dff_loss = self.loss_ld is not None and teacher is not None

        if with_fgl_loss or with_dff_loss:
            bbox_pos_inds = torch.nonzero(
                bbox_weights.sum(-1) > 0, as_tuple=False).squeeze(-1).unique()
            pos_ious = bbox_overlaps(
                bboxes[bbox_pos_inds], bboxes_gt[bbox_pos_inds],
                is_aligned=True).detach()

        if with_fgl_loss:
            initial_bbox_preds = initial_bbox_preds.reshape(-1, 4)
            bbox_corners = bbox_corners.reshape(-1, 4, self.reg_max + 1)
            weight_targets = pos_ious.unsqueeze(-1).repeat(1, 4).reshape(-1)

            if self.cached_fgl_targets is None:
                self.cached_fgl_targets = bbox2distance(
                    initial_bbox_preds[bbox_pos_inds],
                    bbox_cxcywh_to_xyxy(bbox_targets[bbox_pos_inds]),
                    self.reg_max, self.reg_scale, 0.5)
            target_corners, weight_right, weight_left = self.cached_fgl_targets

            loss_fgl = self.fgl_loss_weight * unimodal_distribution_focal_loss(
                bbox_corners[bbox_pos_inds].reshape(-1, self.reg_max + 1),
                target_corners,
                weight_right=weight_right,
                weight_left=weight_left,
                weight=weight_targets,
                avg_factor=bbox_avg_factor)

        if with_dff_loss:
            teacher_scores, teacher_corners = teacher
            teacher_scores = teacher_scores.reshape(-1, self.cls_out_channels)
            teacher_corners = teacher_corners.reshape(-1, self.reg_max + 1)
            bbox_corners = bbox_corners.reshape(-1, self.reg_max + 1)

            weight_targets_local = teacher_scores.sigmoid().max(dim=-1)[0]
            weight_targets_local[bbox_pos_inds] = \
                pos_ious.type_as(weight_targets_local)
            weight_targets_local = \
                weight_targets_local.unsqueeze(-1).repeat(1, 4).reshape(-1)

            loss_match_local = self.loss_ld(bbox_corners, teacher_corners,
                                            weight_targets_local) * (
                                                self.reg_max + 1)

            mask = bbox_weights.bool().reshape(-1)
            num_total_bbox_neg = bbox_weights.size(0) - num_total_bbox_pos
            if self.num_pos is None:
                self.num_pos = (num_total_bbox_pos * 4 * 8 / num_imgs)**0.5
                self.num_neg = (num_total_bbox_neg * 4 * 8 / num_imgs)**0.5
            loss_match_local1 = loss_match_local[mask].mean() \
                if num_total_bbox_pos > 0 else 0
            loss_match_local2 = loss_match_local[~mask].mean() \
                if num_total_bbox_neg > 0 else 0
            loss_ddf = (loss_match_local1 * self.num_pos +
                        loss_match_local2 * self.num_neg) / (
                            self.num_pos + self.num_neg)

        return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_bbox_corners: Optional[Tensor],
                        teacher: Optional[Tuple[Tensor, Tensor]],
                        initial_dn_bbox_preds: Optional[Tensor],
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        if not isinstance(self.loss_cls, VarifocalLoss):
            return super()._loss_dn_single(
                dn_cls_scores, dn_bbox_preds, dn_bbox_corners, teacher,
                initial_dn_bbox_preds, batch_gt_instances, batch_img_metas,
                dn_meta)

        if dn_cls_scores.size(1) == 0:
            loss_cls = dn_cls_scores.new_tensor(0)
            loss_bbox = loss_iou = dn_bbox_preds.new_tensor(0)
            loss_fgl = loss_ddf = dn_bbox_corners.new_tensor(0) \
                if dn_bbox_corners is not None else None
            return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

        if self.cached_dn_targets is None:
            cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                                  batch_img_metas, dn_meta)
            (labels_list, label_weights_list, bbox_targets_list,
             bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            cls_avg_factor = \
                num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(
                    dn_bbox_preds.new_tensor([cls_avg_factor])).item()
            cls_avg_factor = max(cls_avg_factor, 1)

            if self.bg_cls_weight == 0:
                bbox_avg_factor = cls_avg_factor
            else:
                bbox_avg_factor = dn_bbox_preds.new_tensor([num_total_pos])
                bbox_avg_factor = torch.clamp(
                    reduce_mean(bbox_avg_factor), min=1).item()

            self.cached_dn_targets = (labels, label_weights, bbox_targets,
                                      bbox_weights, num_total_pos,
                                      cls_avg_factor, bbox_avg_factor)
        else:
            (labels, label_weights, bbox_targets, bbox_weights, num_total_pos,
             cls_avg_factor, bbox_avg_factor) = self.cached_dn_targets

        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        loss_cls = self._hierarchical_cls_loss(
            cls_scores, dn_bbox_preds, labels, label_weights, bbox_targets,
            cls_avg_factor)

        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=bbox_avg_factor)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=bbox_avg_factor)

        if dn_bbox_corners is None:
            loss_fgl = loss_ddf = None
            with_fgl_loss = with_dff_loss = False
        else:
            loss_fgl = loss_ddf = dn_bbox_corners.new_tensor(0)
            with_fgl_loss = self.fgl_loss_weight is not None
            with_dff_loss = self.loss_ld is not None and teacher is not None

        if with_fgl_loss or with_dff_loss:
            bbox_pos_inds = torch.nonzero(
                bbox_weights.sum(-1) > 0, as_tuple=False).squeeze(-1).unique()
            pos_ious = bbox_overlaps(
                bboxes[bbox_pos_inds], bboxes_gt[bbox_pos_inds],
                is_aligned=True).detach()

        if with_fgl_loss:
            initial_dn_bbox_preds = initial_dn_bbox_preds.reshape(-1, 4)
            dn_bbox_corners = dn_bbox_corners.reshape(-1, 4, self.reg_max + 1)
            weight_targets = pos_ious.unsqueeze(-1).repeat(1, 4).reshape(-1)

            if self.cached_dn_fgl_targets is None:
                self.cached_dn_fgl_targets = bbox2distance(
                    initial_dn_bbox_preds[bbox_pos_inds],
                    bbox_cxcywh_to_xyxy(bbox_targets[bbox_pos_inds]),
                    self.reg_max, self.reg_scale, 0.5)
            (target_corners, weight_right,
             weight_left) = self.cached_dn_fgl_targets

            loss_fgl = self.fgl_loss_weight * unimodal_distribution_focal_loss(
                dn_bbox_corners[bbox_pos_inds].reshape(-1, self.reg_max + 1),
                target_corners,
                weight_right=weight_right,
                weight_left=weight_left,
                weight=weight_targets,
                avg_factor=bbox_avg_factor)

        if with_dff_loss:
            teacher_scores, teacher_corners = teacher
            teacher_scores = teacher_scores.reshape(-1, self.cls_out_channels)
            teacher_corners = teacher_corners.reshape(-1, self.reg_max + 1)
            dn_bbox_corners = dn_bbox_corners.reshape(-1, self.reg_max + 1)

            weight_targets_local = teacher_scores.sigmoid().max(dim=-1)[0]
            weight_targets_local[bbox_pos_inds] = \
                pos_ious.type_as(weight_targets_local)
            weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(
                1, 4).reshape(-1)

            loss_match_local = self.loss_ld(dn_bbox_corners, teacher_corners,
                                            weight_targets_local) * (
                                                self.reg_max + 1)

            mask = bbox_weights.bool().reshape(-1)
            num_total_bbox_pos = num_total_pos
            num_total_bbox_neg = bbox_weights.size(0) - num_total_bbox_pos
            loss_match_local1 = loss_match_local[mask].mean() \
                if num_total_bbox_pos > 0 else 0
            loss_match_local2 = loss_match_local[~mask].mean() \
                if num_total_bbox_neg > 0 else 0
            loss_ddf = (loss_match_local1 * self.num_pos +
                        loss_match_local2 * self.num_neg) / (
                            self.num_pos + self.num_neg)

        return loss_cls, loss_bbox, loss_iou, loss_fgl, loss_ddf

    @torch.no_grad()
    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict,
                               dn_meta: Dict[str, int]) -> tuple:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def _apply_path_score(self, scores: Tensor) -> Tensor:
        path_mask = self.hierarchy_positive_mask.to(
            device=scores.device, dtype=torch.bool)
        path_scores = scores[:, None, :].masked_fill(~path_mask[None], 1.0)
        return path_scores.min(dim=-1).values

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        if not self.loss_cls.use_sigmoid:
            return super()._predict_by_feat_single(
                cls_score, bbox_pred, img_meta, rescale=rescale)

        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        max_per_img = min(max_per_img, len(cls_score))
        img_shape = img_meta['img_shape']

        scores_per_query = cls_score.sigmoid()
        if self.use_path_score:
            scores_per_query = self._apply_path_score(scores_per_query)

        if self.collapse_tree_predictions:
            depth_weight = 1.0 + torch.clamp(
                self.hierarchy_depth.to(scores_per_query.device) - 1.0,
                min=0.0) * self.depth_score_factor
            rank_scores = scores_per_query * depth_weight[None, :]
            _, det_labels = rank_scores.max(dim=-1)
            query_scores = scores_per_query.gather(
                1, det_labels[:, None]).squeeze(1)
            scores, bbox_index = query_scores.topk(max_per_img)
            det_labels = det_labels[bbox_index]
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, indexes = scores_per_query.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

        nms_cfg = self.test_cfg.get('nms', None)
        if nms_cfg is not None:
            _, keeps = batched_nms(
                boxes=results.bboxes,
                scores=results.scores,
                idxs=results.labels,
                nms_cfg=nms_cfg)
            results = results[keeps]

        return results
