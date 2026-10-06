import argparse
import json
import os
import random
import re
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the plane JSON annotations to COCO detection '
        'format and create a train/test split.')
    parser.add_argument(
        'input_root',
        help='Directory containing the original plane *.json and image files.')
    parser.add_argument(
        'out_root',
        help='Output dataset root. The script writes annotations/, train/imgs/, '
        'and test/imgs/ under this directory.')
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Fraction of images reserved for the test split.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for splitting.')
    parser.add_argument(
        '--split-by',
        choices=('group', 'image'),
        default='group',
        help='Use group to split by port id and reduce leakage.')
    parser.add_argument(
        '--group-regex',
        default=r'^(port_\d+)',
        help='Regex used to extract split groups from image names.')
    parser.add_argument(
        '--channel',
        default='R',
        help='Preferred bbox channel in obj_position. Falls back to the first '
        'available bbox if this channel is absent.')
    parser.add_argument(
        '--keep-original-classes',
        action='store_true',
        help='Keep raw aircraft subclasses as COCO categories. By default all '
        'classes are merged into one category named plane.')
    parser.add_argument(
        '--image-mode',
        choices=('link', 'copy', 'hardlink', 'none'),
        default='link',
        help='How to materialize split images. link tries symlink, hardlink, '
        'then copy. none only writes JSON annotations.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search json/images recursively under input_root.')
    parser.add_argument(
        '--keep-empty',
        action='store_true',
        help='Keep images without valid boxes in the COCO json.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only print the discovered statistics and split sizes.')
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def iter_files(root, suffixes, recursive=False):
    if recursive:
        iterator = root.rglob('*')
    else:
        iterator = root.glob('*')
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in suffixes)


def build_image_index(root, recursive=False):
    images = iter_files(root, IMAGE_SUFFIXES, recursive=recursive)
    index = defaultdict(list)
    for path in images:
        index[path.name].append(path)
        index[path.stem].append(path)
    return index, images


def find_image_path(root, image_index, image_name):
    image_name = str(image_name)
    direct = root / image_name
    if direct.is_file():
        return direct
    matches = image_index.get(Path(image_name).name, [])
    if matches:
        return matches[0]
    stem_matches = image_index.get(Path(image_name).stem, [])
    return stem_matches[0] if stem_matches else None


def read_image_size(path):
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        pass

    with path.open('rb') as f:
        header = f.read(32)
        if header.startswith(b'\x89PNG\r\n\x1a\n'):
            return struct.unpack('>II', header[16:24])

        f.seek(0)
        if f.read(2) == b'\xff\xd8':
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    break
                if marker_start != b'\xff':
                    continue
                marker = f.read(1)
                while marker == b'\xff':
                    marker = f.read(1)
                if marker in (b'\xc0', b'\xc1', b'\xc2', b'\xc3',
                              b'\xc5', b'\xc6', b'\xc7', b'\xc9',
                              b'\xca', b'\xcb', b'\xcd', b'\xce',
                              b'\xcf'):
                    f.read(3)
                    height, width = struct.unpack('>HH', f.read(4))
                    return width, height
                size_bytes = f.read(2)
                if len(size_bytes) != 2:
                    break
                size = struct.unpack('>H', size_bytes)[0]
                f.seek(size - 2, os.SEEK_CUR)

    raise RuntimeError(f'Cannot read image size: {path}')


def select_bbox(obj, preferred_channel):
    positions = obj.get('obj_position') or []
    fallback = None
    for item in positions:
        if not isinstance(item, dict):
            continue
        bbox = item.get('bbox')
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if fallback is None:
            fallback = bbox
        if str(item.get('channel', '')).upper() == preferred_channel.upper():
            return bbox
    return fallback


def clip_bbox(bbox, width, height):
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(0.0, min(x, float(width)))
    y1 = max(0.0, min(y, float(height)))
    x2 = max(0.0, min(x + w, float(width)))
    y2 = max(0.0, min(y + h, float(height)))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return [x1, y1, clipped_w, clipped_h]


def group_name(image_name, group_regex):
    match = re.search(group_regex, image_name)
    if match:
        return match.group(1)
    return Path(image_name).stem


def collect_records(args):
    input_root = Path(args.input_root)
    image_index, image_files = build_image_index(
        input_root, recursive=args.recursive)
    json_files = iter_files(input_root, ('.json', ), recursive=args.recursive)

    records = []
    raw_class_counts = Counter()
    skipped = Counter()
    duplicate_image_names = {
        name: paths for name, paths in image_index.items()
        if '.' in name and len(paths) > 1
    }

    for ann_path in json_files:
        try:
            data = load_json(ann_path)
        except Exception as exc:
            skipped[f'bad_json:{type(exc).__name__}'] += 1
            continue

        if not isinstance(data, dict) or 'obj_label' not in data:
            skipped['not_plane_annotation'] += 1
            continue

        image_name = data.get('img_name') or f'{ann_path.stem}.png'
        image_path = find_image_path(input_root, image_index, image_name)
        if image_path is None:
            skipped['missing_image'] += 1
            continue

        try:
            width, height = read_image_size(image_path)
        except Exception:
            skipped['bad_image_size'] += 1
            continue

        objects = []
        for obj in data.get('obj_label') or []:
            if not isinstance(obj, dict):
                skipped['bad_object'] += 1
                continue
            raw_class = str(obj.get('obj_class', '')).strip()
            if not raw_class:
                skipped['missing_class'] += 1
                continue
            bbox = select_bbox(obj, args.channel)
            if bbox is None:
                skipped['missing_bbox'] += 1
                continue
            bbox = clip_bbox(bbox, width, height)
            if bbox is None:
                skipped['invalid_bbox'] += 1
                continue
            raw_class_counts[raw_class] += 1
            objects.append(
                dict(
                    raw_class=raw_class,
                    bbox=bbox,
                    area=bbox[2] * bbox[3],
                    obj_angle=obj.get('obj_angle'),
                    obj_speed=obj.get('obj_speed'),
                    obj_attitude=obj.get('obj_attitude')))

        if objects or args.keep_empty:
            rel_image_path = image_path.relative_to(input_root)
            records.append(
                dict(
                    ann_path=ann_path,
                    image_path=image_path,
                    rel_image_path=rel_image_path,
                    image_name=Path(image_name).name,
                    width=width,
                    height=height,
                    objects=objects,
                    group=group_name(Path(image_name).name, args.group_regex),
                    img_resolution=data.get('img_resolution'),
                    shooting_mode=data.get('shooting_mode'),
                    source_group=data.get('source_group')))
        else:
            skipped['empty_image'] += 1

    return records, raw_class_counts, skipped, len(json_files), len(image_files), duplicate_image_names


def split_records(records, test_ratio, seed, split_by):
    rng = random.Random(seed)
    records = sorted(records, key=lambda item: item['rel_image_path'].as_posix())
    if not records:
        return [], []

    if split_by == 'image':
        shuffled = list(records)
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * test_ratio)))
        test_set = set(id(item) for item in shuffled[:test_count])
        train = [item for item in records if id(item) not in test_set]
        test = [item for item in records if id(item) in test_set]
        return train, test

    grouped = defaultdict(list)
    for item in records:
        grouped[item['group']].append(item)
    groups = sorted(grouped)
    rng.shuffle(groups)

    target_test_count = max(1, int(round(len(records) * test_ratio)))
    test_groups = []
    test_count = 0
    for group in groups:
        if test_count >= target_test_count and test_groups:
            break
        test_groups.append(group)
        test_count += len(grouped[group])

    test_group_set = set(test_groups)
    train = [item for item in records if item['group'] not in test_group_set]
    test = [item for item in records if item['group'] in test_group_set]
    if not train and len(test) > 1:
        train, test = test[1:], test[:1]
    return train, test


def ensure_image(src, dst, mode):
    if mode == 'none':
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == 'copy':
        shutil.copy2(src, dst)
        return
    if mode == 'hardlink':
        os.link(src, dst)
        return

    rel_src = os.path.relpath(src, start=dst.parent)
    try:
        os.symlink(rel_src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def build_categories(records, keep_original_classes):
    if not keep_original_classes:
        return [{'id': 1, 'name': 'plane'}], {'plane': 1}

    class_names = sorted({
        obj['raw_class']
        for record in records for obj in record['objects']
    })
    categories = [
        {
            'id': idx + 1,
            'name': name
        } for idx, name in enumerate(class_names)
    ]
    return categories, {item['name']: item['id'] for item in categories}


def build_coco(split_name, records, categories, category_by_name,
               keep_original_classes):
    images = []
    annotations = []
    ann_id = 1

    for img_id, record in enumerate(records, start=1):
        images.append(
            dict(
                id=img_id,
                file_name=record['rel_image_path'].as_posix(),
                width=record['width'],
                height=record['height'],
                img_resolution=record['img_resolution'],
                shooting_mode=record['shooting_mode'],
                source_group=record['source_group'],
                source_json=record['ann_path'].name))

        for obj in record['objects']:
            category_name = obj['raw_class'] if keep_original_classes else 'plane'
            annotations.append(
                dict(
                    id=ann_id,
                    image_id=img_id,
                    category_id=category_by_name[category_name],
                    bbox=obj['bbox'],
                    area=obj['area'],
                    iscrowd=0,
                    raw_category=obj['raw_class'],
                    obj_angle=obj['obj_angle'],
                    obj_speed=obj['obj_speed'],
                    obj_attitude=obj['obj_attitude']))
            ann_id += 1

    return dict(
        info=dict(
            description='Plane dataset converted from per-image JSON labels',
            split=split_name),
        licenses=[],
        categories=categories,
        images=images,
        annotations=annotations)


def write_split(out_root, split_name, records, categories, category_by_name,
                keep_original_classes, image_mode):
    split_img_root = out_root / split_name / 'imgs'
    for record in records:
        dst = split_img_root / record['rel_image_path']
        ensure_image(record['image_path'], dst, image_mode)

    coco = build_coco(split_name, records, categories, category_by_name,
                      keep_original_classes)
    ann_dir = out_root / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    with (ann_dir / f'{split_name}.json').open('w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)


def print_summary(records, train, test, raw_class_counts, skipped,
                  num_json_files, num_image_files, duplicate_image_names):
    print(f'JSON files: {num_json_files}')
    print(f'Image files: {num_image_files}')
    print(f'Usable images: {len(records)}')
    print(f'Train images: {len(train)}')
    print(f'Test images: {len(test)}')
    print(f'Usable boxes: {sum(len(item["objects"]) for item in records)}')
    print(f'Raw classes: {len(raw_class_counts)}')
    for name, count in raw_class_counts.most_common():
        print(f'  {name}: {count}')
    if skipped:
        print('Skipped:')
        for name, count in skipped.most_common():
            print(f'  {name}: {count}')
    if duplicate_image_names:
        print(f'Duplicate image names: {len(duplicate_image_names)}')


def main():
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise RuntimeError('--test-ratio must be between 0 and 1.')

    out_root = Path(args.out_root)
    records, raw_class_counts, skipped, num_json_files, num_image_files, duplicate_image_names = collect_records(args)
    train, test = split_records(records, args.test_ratio, args.seed,
                                args.split_by)

    print_summary(records, train, test, raw_class_counts, skipped,
                  num_json_files, num_image_files, duplicate_image_names)
    if args.dry_run:
        return

    categories, category_by_name = build_categories(
        records, keep_original_classes=args.keep_original_classes)
    write_split(out_root, 'train', train, categories, category_by_name,
                args.keep_original_classes, args.image_mode)
    write_split(out_root, 'test', test, categories, category_by_name,
                args.keep_original_classes, args.image_mode)

    summary = {
        'num_json_files': num_json_files,
        'num_image_files': num_image_files,
        'num_usable_images': len(records),
        'num_train_images': len(train),
        'num_test_images': len(test),
        'num_usable_boxes': sum(len(item['objects']) for item in records),
        'raw_class_counts': dict(raw_class_counts),
        'skipped': dict(skipped),
        'merge_to_plane': not args.keep_original_classes,
        'split_by': args.split_by,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
    }
    ann_dir = out_root / 'annotations'
    with (ann_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Wrote COCO dataset to: {out_root}')


if __name__ == '__main__':
    main()
