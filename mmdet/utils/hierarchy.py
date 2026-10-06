# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor


def build_class_hierarchy(
    class_names: Sequence[str],
    hierarchy: Mapping[str, Sequence[str]],
    sibling_negative_weight: float = 0.25,
    family_negative_weight: float = 0.5,
    descendant_negative_weight: float = 0.0,
    descendant_match_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Build dense hierarchy tensors indexed by flat class id.

    Rows are ground-truth labels and columns are model output labels.
    """
    class_names = tuple(class_names)
    if len(set(class_names)) != len(class_names):
        raise ValueError('`class_names` contains duplicate entries.')

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    parent = torch.full((num_classes, ), -1, dtype=torch.long)

    for parent_name, children in hierarchy.items():
        if parent_name not in name_to_idx:
            raise KeyError(f'Hierarchy parent {parent_name!r} is not a class.')
        parent_idx = name_to_idx[parent_name]
        for child_name in children:
            if child_name not in name_to_idx:
                raise KeyError(
                    f'Hierarchy child {child_name!r} is not a class.')
            child_idx = name_to_idx[child_name]
            if parent[child_idx] != -1:
                old_parent = class_names[int(parent[child_idx])]
                raise ValueError(
                    f'Class {child_name!r} has multiple parents: '
                    f'{old_parent!r} and {parent_name!r}.')
            parent[child_idx] = parent_idx

    ancestor = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    depth = torch.zeros((num_classes, ), dtype=torch.float32)
    root = torch.arange(num_classes, dtype=torch.long)

    for idx in range(num_classes):
        seen = set()
        cur = idx
        last = idx
        while cur != -1:
            if cur in seen:
                raise ValueError(f'Hierarchy cycle detected at {class_names[idx]!r}.')
            seen.add(cur)
            ancestor[idx, cur] = 1.0
            last = cur
            cur = int(parent[cur])
        depth[idx] = float(len(seen))
        root[idx] = last

    descendant = ancestor.t().contiguous()
    descendant.fill_diagonal_(0.0)

    same_parent = parent[:, None].eq(parent[None, :]) & parent[:, None].ge(0)
    same_root = root[:, None].eq(root[None, :])

    loss_weight = torch.ones((num_classes, num_classes), dtype=torch.float32)
    loss_weight[same_root] = family_negative_weight
    loss_weight[same_parent] = sibling_negative_weight
    loss_weight[descendant.bool()] = descendant_negative_weight
    loss_weight[ancestor.bool()] = 1.0

    match_weight = ancestor.clone()
    match_weight = torch.maximum(match_weight, descendant * descendant_match_weight)

    return dict(
        parent=parent,
        ancestor=ancestor,
        descendant=descendant,
        loss_weight=loss_weight,
        match_weight=match_weight,
        depth=depth,
        root=root)
