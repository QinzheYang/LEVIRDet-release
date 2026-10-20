# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

from torch.utils.data import BatchSampler, Sampler

from mmdet.datasets.samplers.track_img_sampler import TrackImgSampler
from mmdet.registry import DATA_SAMPLERS


# TODO: maybe replace with a data_loader wrapper
@DATA_SAMPLERS.register_module()
class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # two groups for w < h and w >= h
        self._aspect_ratio_buckets = [[] for _ in range(2)]

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info['width'], data_info['height']
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[
            1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[:self.batch_size]
                left_data = left_data[self.batch_size:]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


@DATA_SAMPLERS.register_module()
class QueryAwareBatchSampler(BatchSampler):
    """Group images with similar GT-proxy query demand into one batch.

    The detector still uses one tensor shape per batch. This sampler reduces
    wasted training compute by making each mini-batch mostly contain images
    that fall into the same query bin.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool = False,
                 bins: Sequence[int] = (300, 600, 800),
                 query_per_gt: float = 3.0,
                 small_obj_weight: float = 15.0,
                 density_weight: float = 0.2,
                 scale_bonus_weight: float = 8.0,
                 scale_ref: float = 0.08,
                 small_area_thr: float = 0.001,
                 include_ignored: bool = True,
                 gsd_key: str = '',
                 gsd_weight: float = 0.0,
                 gsd_ref: float = 1.0,
                 gsd_clip: float = 2.0) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bins = sorted(int(v) for v in bins)
        if len(self.bins) == 0:
            raise ValueError('bins must contain at least one query count.')
        self.query_per_gt = float(query_per_gt)
        self.small_obj_weight = float(small_obj_weight)
        self.density_weight = float(density_weight)
        self.scale_bonus_weight = float(scale_bonus_weight)
        self.scale_ref = float(scale_ref)
        self.small_area_thr = float(small_area_thr)
        self.include_ignored = include_ignored
        self.gsd_key = gsd_key
        self.gsd_weight = float(gsd_weight)
        self.gsd_ref = float(gsd_ref)
        self.gsd_clip = float(gsd_clip)
        self._query_buckets = [[] for _ in range(len(self.bins))]

    def _query_score(self, data_info: dict) -> float:
        width = float(max(1, data_info.get('width', 1024)))
        height = float(max(1, data_info.get('height', 1024)))
        img_area = width * height
        instances = data_info.get('instances', [])
        areas = []
        for inst in instances:
            if (not self.include_ignored) and inst.get('ignore_flag', 0):
                continue
            bbox = inst.get('bbox', None)
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            if area > 0:
                areas.append(area)

        n = len(areas)
        if n == 0:
            return 1.0

        area_ratios = [max(1e-6, area / img_area) for area in areas]
        small_ratio = sum(r < self.small_area_thr for r in area_ratios) / n
        mean_scale = sum(math.sqrt(r) for r in area_ratios) / n
        density = n / (img_area / (1024.0 * 1024.0))
        scale_bonus = max(0.0, self.scale_ref - mean_scale)
        score = (
            n * self.query_per_gt
            + self.small_obj_weight * small_ratio
            + self.density_weight * density
            + self.scale_bonus_weight * scale_bonus)

        if self.gsd_key and self.gsd_weight != 0 and self.gsd_key in data_info:
            gsd = max(1e-6, float(data_info[self.gsd_key]))
            ref = max(1e-6, self.gsd_ref)
            gsd_boost = max(
                -self.gsd_clip,
                min(self.gsd_clip, math.log(ref / gsd)))
            score += self.gsd_weight * gsd_boost

        return max(1.0, score)

    def _bucket_id(self, data_info: dict) -> int:
        score = self._query_score(data_info)
        distances = [abs(score - query_count) for query_count in self.bins]
        return min(range(len(self.bins)), key=lambda i: distances[i])

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_idx = idx[0] if isinstance(idx, tuple) else idx
            data_info = self.sampler.dataset.get_data_info(data_idx)
            bucket_id = self._bucket_id(data_info)
            bucket = self._query_buckets[bucket_id]
            bucket.append(idx)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        left_data = []
        for bucket in self._query_buckets:
            left_data.extend(bucket)
        self._query_buckets = [[] for _ in range(len(self.bins))]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[:self.batch_size]
                left_data = left_data[self.batch_size:]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


@DATA_SAMPLERS.register_module()
class TrackAspectRatioBatchSampler(AspectRatioBatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            # hard code to solve TrackImgSampler
            if isinstance(self.sampler, TrackImgSampler):
                video_idx, _ = idx
            else:
                video_idx = idx
            # video_idx
            data_info = self.sampler.dataset.get_data_info(video_idx)
            # data_info {video_id, images, video_length}
            img_data_info = data_info['images'][0]
            width, height = img_data_info['width'], img_data_info['height']
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[
            1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[:self.batch_size]
                left_data = left_data[self.batch_size:]


@DATA_SAMPLERS.register_module()
class MultiDataAspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch for multi-source datasets.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (Sequence(int)): Size of mini-batch for multi-source
        datasets.
        num_datasets(int): Number of multi-source datasets.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
        its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: Sequence[int],
                 num_datasets: int,
                 drop_last: bool = True) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_datasets = num_datasets
        self.drop_last = drop_last
        # two groups for w < h and w >= h for each dataset --> 2 * num_datasets
        self._buckets = [[] for _ in range(2 * self.num_datasets)]

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info['width'], data_info['height']
            dataset_source_idx = self.sampler.dataset.get_dataset_source(idx)
            aspect_ratio_bucket_id = 0 if width < height else 1
            bucket_id = dataset_source_idx * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size[dataset_source_idx]:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        for i in range(self.num_datasets):
            left_data = self._buckets[i * 2 + 0] + self._buckets[i * 2 + 1]
            while len(left_data) > 0:
                if len(left_data) <= self.batch_size[i]:
                    if not self.drop_last:
                        yield left_data[:]
                    left_data = []
                else:
                    yield left_data[:self.batch_size[i]]
                    left_data = left_data[self.batch_size[i]:]

        self._buckets = [[] for _ in range(2 * self.num_datasets)]

    def __len__(self) -> int:
        sizes = [0 for _ in range(self.num_datasets)]
        for idx in self.sampler:
            dataset_source_idx = self.sampler.dataset.get_dataset_source(idx)
            sizes[dataset_source_idx] += 1

        if self.drop_last:
            lens = 0
            for i in range(self.num_datasets):
                lens += sizes[i] // self.batch_size[i]
            return lens
        else:
            lens = 0
            for i in range(self.num_datasets):
                lens += (sizes[i] + self.batch_size[i] -
                         1) // self.batch_size[i]
            return lens
