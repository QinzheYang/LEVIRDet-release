# Copyright (c) OpenMMLab. All rights reserved.
"""Expand a coarse-class checkpoint for the hierarchy-aware detector config.

The converter keeps all compatible weights and expands class-shaped tensors:

- bbox_head.cls_branches.*.{weight,bias}: 30 -> target class count
- dn_query_generator.label_embedding.weight: 30 -> target class count

New fine-grained rows are initialized from the closest ancestor that exists in
the source checkpoint classes.
"""

import argparse
import copy
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Expand a 30-class checkpoint to a hierarchy checkpoint.')
    parser.add_argument('checkpoint', help='Source coarse-class checkpoint.')
    parser.add_argument('config', help='Target hierarchy config file.')
    parser.add_argument('output', help='Output converted checkpoint path.')
    parser.add_argument(
        '--source',
        choices=('state_dict', 'ema_state_dict'),
        default='state_dict',
        help='Which weight dict to convert from. Defaults to state_dict.')
    return parser.parse_args()


def load_config_namespace(config_path: str) -> dict:
    namespace: dict = {}
    code = Path(config_path).read_text(encoding='utf-8')
    exec(compile(code, config_path, 'exec'), namespace)
    return namespace


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]
                        ) -> OrderedDict:
    stripped = OrderedDict()
    for key, value in state_dict.items():
        if key == 'steps':
            continue
        if key.startswith('module.'):
            key = key[len('module.'):]
        stripped[key] = value
    return stripped


def build_parent_map(
    hierarchy: Mapping[str, Sequence[str]]
) -> Dict[str, Optional[str]]:
    parent: Dict[str, Optional[str]] = {}
    for parent_name, children in hierarchy.items():
        parent.setdefault(parent_name, None)
        for child_name in children:
            if child_name in parent and parent[child_name] is not None:
                raise ValueError(f'{child_name!r} has multiple parents.')
            parent[child_name] = parent_name
    return parent


def find_source_ancestor(name: str, source_name_to_idx: Mapping[str, int],
                         parent: Mapping[str, Optional[str]],
                         source_class_aliases: Mapping[str, str]) -> str:
    cur = name
    while cur is not None:
        source_name = source_class_aliases.get(cur, cur)
        if source_name in source_name_to_idx:
            return source_name
        cur = parent.get(cur)
    raise KeyError(f'Cannot map target class {name!r} to source classes.')


def build_class_index_map(source_classes: Sequence[str],
                          target_classes: Sequence[str],
                          hierarchy: Mapping[str, Sequence[str]],
                          source_class_aliases: Mapping[str,
                                                        str]) -> Dict[int, int]:
    source_name_to_idx = {
        name: idx
        for idx, name in enumerate(source_classes)
    }
    parent = build_parent_map(hierarchy)
    index_map = {}
    for target_idx, name in enumerate(target_classes):
        source_name = find_source_ancestor(name, source_name_to_idx, parent,
                                           source_class_aliases)
        index_map[target_idx] = source_name_to_idx[source_name]
    return index_map


def should_expand_key(key: str, value: torch.Tensor,
                      source_num_classes: int) -> bool:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return False
    if value.shape[0] != source_num_classes:
        return False
    if re.fullmatch(r'bbox_head\.cls_branches\.\d+\.(weight|bias)', key):
        return True
    if key == 'dn_query_generator.label_embedding.weight':
        return True
    return False


def expand_tensor(value: torch.Tensor, index_map: Mapping[int, int],
                  target_num_classes: int) -> torch.Tensor:
    expanded_shape = (target_num_classes, ) + tuple(value.shape[1:])
    expanded = value.new_empty(expanded_shape)
    for target_idx in range(target_num_classes):
        expanded[target_idx].copy_(value[index_map[target_idx]])
    return expanded


def main() -> int:
    args = parse_args()
    cfg = load_config_namespace(args.config)
    target_classes = tuple(cfg['classes'])
    hierarchy = dict(cfg['class_hierarchy'])
    source_class_aliases = dict(cfg.get('source_class_aliases', {}))

    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    if args.source not in checkpoint:
        raise KeyError(f'{args.source!r} is not in checkpoint.')

    source_state_dict = strip_module_prefix(checkpoint[args.source])
    source_classes = tuple(
        checkpoint.get('meta', {}).get('dataset_meta', {}).get('classes', ()))
    if not source_classes:
        source_classes = tuple(cfg.get('base_classes', ()))
    if not source_classes:
        raise KeyError('Cannot infer source classes from checkpoint or config.')

    index_map = build_class_index_map(source_classes, target_classes,
                                      hierarchy, source_class_aliases)
    target_num_classes = len(target_classes)
    source_num_classes = len(source_classes)

    converted_state_dict = OrderedDict()
    expanded_keys = []
    for key, value in source_state_dict.items():
        if should_expand_key(key, value, source_num_classes):
            converted_state_dict[key] = expand_tensor(value, index_map,
                                                      target_num_classes)
            expanded_keys.append(key)
        else:
            converted_state_dict[key] = value

    meta = copy.deepcopy(checkpoint.get('meta', {}))
    dataset_meta = copy.deepcopy(meta.get('dataset_meta', {}))
    dataset_meta['classes'] = target_classes
    meta['dataset_meta'] = dataset_meta
    meta['hierarchy_pretrain'] = dict(
        source_checkpoint=str(args.checkpoint),
        source=args.source,
        source_num_classes=source_num_classes,
        target_num_classes=target_num_classes,
        source_class_aliases=source_class_aliases,
        expanded_keys=expanded_keys)

    output = dict(meta=meta, state_dict=converted_state_dict)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)

    print(f'source_classes={source_num_classes}')
    print(f'target_classes={target_num_classes}')
    print(f'expanded_keys={len(expanded_keys)}')
    for key in expanded_keys:
        print(f'  {key}: {tuple(source_state_dict[key].shape)} -> '
              f'{tuple(converted_state_dict[key].shape)}')
    print(f'saved={args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
