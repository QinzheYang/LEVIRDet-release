# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

from torch import Tensor

from mmdet.models.losses import VarifocalLoss
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, reduce_mean
from .hierarchical_deimv2_head import HierarchicalDEIMV2Head


@MODELS.register_module()
class DistributedSafeHierarchicalDEIMV2Head(HierarchicalDEIMV2Head):
    """Hierarchy head robust to an all-empty local DDP batch.

    Denoising query count is derived from the local maximum number of GTs. If
    one rank has no GT, its DN loss returns early while other ranks synchronize
    DN normalization factors. This subclass joins those collectives before the
    early return without changing the existing hierarchy or DFINE heads.
    """

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_bbox_corners: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: Optional[InstanceList] = None
    ) -> Dict[str, Tensor]:
        self._empty_dn_sync_done = False
        return super().loss_by_feat(
            all_layers_cls_scores=all_layers_cls_scores,
            all_layers_bbox_preds=all_layers_bbox_preds,
            all_layers_bbox_corners=all_layers_bbox_corners,
            enc_cls_scores=enc_cls_scores,
            enc_bbox_preds=enc_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        dn_bbox_corners: Optional[Tensor],
                        teacher: Optional[Tuple[Tensor, Tensor]],
                        initial_dn_bbox_preds: Optional[Tensor],
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        if (isinstance(self.loss_cls, VarifocalLoss)
                and dn_cls_scores.size(1) == 0
                and not getattr(self, '_empty_dn_sync_done', False)):
            if self.sync_cls_avg_factor:
                reduce_mean(dn_bbox_preds.new_tensor([0.0])).item()
            if self.bg_cls_weight != 0:
                reduce_mean(dn_bbox_preds.new_tensor([0.0])).item()
            self._empty_dn_sync_done = True

        return super()._loss_dn_single(
            dn_cls_scores=dn_cls_scores,
            dn_bbox_preds=dn_bbox_preds,
            dn_bbox_corners=dn_bbox_corners,
            teacher=teacher,
            initial_dn_bbox_preds=initial_dn_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)
