# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class FineCocoDataset(CocoDataset):
    """COCO dataset that can read fine-grained labels from annotation fields.

    The raw COCO ``category_id`` may stay at the coarse class. If a fine label
    field is present and exists in ``metainfo['classes']``, it is used as the
    training label; otherwise the coarse ``category_id`` label is used.
    """

    def __init__(self,
                 *args,
                 fine_label_key: str = 'fine_category_name',
                 fine_level_keys: Sequence[str] = (
                     'fine_category_level4', 'fine_category_level3',
                     'fine_category_level2', 'fine_category_level1'),
                 fine_label_aliases: Optional[Dict[str, str]] = None,
                 category_label_aliases: Optional[Dict[str, str]] = None,
                 ignore_crowd: bool = True,
                 **kwargs) -> None:
        self.fine_label_key = fine_label_key
        self.fine_level_keys = tuple(fine_level_keys)
        self.fine_label_aliases = dict(fine_label_aliases or {})
        self.category_label_aliases = dict(category_label_aliases or {})
        self.ignore_crowd = ignore_crowd
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        self.class_name_to_label = {
            name: idx
            for idx, name in enumerate(self.metainfo['classes'])
        }

        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        # Build the coarse category mapping through category-only aliases.
        # This keeps a raw COCO parent category named "car" mapped to the
        # model parent "vehicle" without changing the fine leaf label "car".
        self.cat_ids = []
        self.cat2label = {}
        for cat_info in self.coco.dataset.get('categories', []):
            cat_id = cat_info['id']
            cat_name = self._normalize_category_label(cat_info.get('name'))
            if cat_name in self.class_name_to_label:
                self.cat_ids.append(cat_id)
                self.cat2label[cat_id] = self.class_name_to_label[cat_name]
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def _normalize_fine_label(self, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        return self.fine_label_aliases.get(name, name)

    def _normalize_category_label(self, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        return self.category_label_aliases.get(name, name)

    def _get_ann_label(self, ann: dict) -> int:
        fine_name = self._normalize_fine_label(ann.get(self.fine_label_key))
        if fine_name in self.class_name_to_label:
            return self.class_name_to_label[fine_name]

        for key in self.fine_level_keys:
            fine_name = self._normalize_fine_label(ann.get(key))
            if fine_name in self.class_name_to_label:
                return self.class_name_to_label[fine_name]

        return self.cat2label[ann['category_id']]

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for ann in ann_info:
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if self.ignore_crowd and ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self._get_ann_label(ann)

            fine_name = ann.get(self.fine_label_key)
            if fine_name is not None:
                instance[self.fine_label_key] = fine_name

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
