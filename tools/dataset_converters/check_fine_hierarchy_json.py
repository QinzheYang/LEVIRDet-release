# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def load_config_symbols(config: Path) -> dict:
    namespace = {}
    exec(compile(config.read_text(encoding='utf-8'), str(config), 'exec'),
         namespace)
    return namespace


def normalize(name: str, aliases: Mapping[str, str]) -> str:
    return aliases.get(name, name)


def deepest_label(ann: dict, level_keys: Sequence[str]) -> str:
    if ann.get('fine_category_name'):
        return ann['fine_category_name']
    for key in level_keys:
        if ann.get(key):
            return ann[key]
    return ''


def iter_parent_pairs(path: List[str]) -> Iterable[tuple]:
    for parent, child in zip(path, path[1:]):
        yield parent, child


def check_file(path: Path, classes: Sequence[str],
               hierarchy: Mapping[str, Sequence[str]],
               aliases: Mapping[str, str], level_keys: Sequence[str]) -> int:
    classes = tuple(classes)
    class_set = set(classes)
    children = {
        parent: set(values)
        for parent, values in hierarchy.items()
    }
    categories_by_id = {}
    errors = []
    fine_counts = Counter()
    fallback_count = 0
    level_counts = defaultdict(Counter)

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    for cat in data.get('categories', []):
        categories_by_id[cat['id']] = cat['name']

    for ann in data.get('annotations', []):
        raw_path = [
            ann.get('fine_category_level1'),
            ann.get('fine_category_level2'),
            ann.get('fine_category_level3'),
            ann.get('fine_category_level4'),
        ]
        raw_path = [name for name in raw_path if name]
        norm_path = [normalize(name, aliases) for name in raw_path]

        fine_name = deepest_label(ann, level_keys)
        if fine_name:
            fine_name = normalize(fine_name, aliases)
            fine_counts[fine_name] += 1
            if fine_name not in class_set:
                errors.append(f'unknown fine label {fine_name!r}')
            if norm_path and fine_name != norm_path[-1]:
                errors.append(
                    f'fine label {fine_name!r} != deepest level '
                    f'{norm_path[-1]!r}')
        else:
            fallback_count += 1

        for level_key in (
                'fine_category_level1', 'fine_category_level2',
                'fine_category_level3', 'fine_category_level4'):
            raw_name = ann.get(level_key)
            if not raw_name:
                continue
            norm_name = normalize(raw_name, aliases)
            level_counts[level_key][norm_name] += 1
            if norm_name not in class_set:
                errors.append(f'unknown level label {raw_name!r}')

        for parent, child in iter_parent_pairs(norm_path):
            if parent == child:
                continue
            if child not in children.get(parent, set()):
                errors.append(f'invalid hierarchy edge {parent!r}->{child!r}')

        coarse = categories_by_id.get(ann.get('category_id'))
        if coarse is not None and norm_path:
            coarse = normalize(coarse, aliases)
            if coarse != norm_path[0]:
                errors.append(
                    f'category_id coarse {coarse!r} != level1 '
                    f'{norm_path[0]!r}')

    unique_errors = Counter(errors)
    print(path)
    print(f'  images={len(data.get("images", []))} '
          f'annotations={len(data.get("annotations", []))} '
          f'categories={len(data.get("categories", []))}')
    print(f'  fine_unique={len(fine_counts)} fallback_to_coarse={fallback_count}')
    print(f'  top_fine={fine_counts.most_common(10)}')
    print(f'  unique_level_values='
          f'{dict((k, len(v)) for k, v in level_counts.items())}')
    if unique_errors:
        print(f'  errors={sum(unique_errors.values())} '
              f'unique_errors={len(unique_errors)}')
        for message, count in unique_errors.most_common(20):
            print(f'    {count}x {message}')
    else:
        print('  hierarchy_check=OK')
    return 1 if unique_errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_files', nargs='+', type=Path)
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(
            'configs/deimv2/'
            'deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det.py'))
    parser.add_argument('--alias', action='append', default=['vehicle=car'])
    args = parser.parse_args()

    symbols = load_config_symbols(args.config)
    classes = symbols['classes']
    hierarchy = symbols['class_hierarchy']
    aliases = dict(item.split('=', 1) for item in args.alias)
    level_keys = (
        'fine_category_level4', 'fine_category_level3',
        'fine_category_level2', 'fine_category_level1')

    status = 0
    for ann_file in args.ann_files:
        status |= check_file(ann_file, classes, hierarchy, aliases, level_keys)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
