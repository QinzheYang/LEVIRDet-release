# Copyright (c) OpenMMLab. All rights reserved.
"""Crop fine-grained class examples and render a class contact sheet.

This script reads COCO-like hierarchy annotations, samples up to N boxes per
class, saves expanded crops, and creates a 3-column overview image:

    class name | crop 1 | crop 2 | ...

The original 30 coarse classes are shown first, followed by the fine-grained
hierarchy nodes.
"""

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


BASE_CLASSES = (
    'plane', 'storage_tank', 'tenniscourt', 'baseball_diamond',
    'basketball_court', 'ground_track_field', 'vehicle', 'bridge', 'harbor',
    'ship', 'parking_lot', 'overpass', 'swimming-pool', 'roundabout',
    'soccer-ball-field', 'pylon', 'stadium', 'container', 'container-crane',
    'windmill', 'helipad', 'rugby-count', 'helicoper',
    'Expressway-toll-station', 'chimney', 'dam', 'golffield',
    'trainstation', 'Expressway-Service-area', 'airport')

PLANE_AIRLINER_CLASSES = (
    'A220', 'A321', 'A330', 'A350', 'ARJ21', 'Boeing737', 'Boeing747',
    'Boeing777', 'Boeing787', 'C919')
PLANE_NOTAIRLINER_CLASSES = (
    'A-10', 'A-26', 'B-1', 'B-1B', 'B-2', 'B-29', 'B-52', 'C-130',
    'C-135', 'C-17', 'C-21', 'C-5', 'E-3', 'E-8', 'F-15', 'F-16', 'F-22',
    'F-5', 'FA-18', 'KC-10', 'KC-135', 'P-3C', 'P-63', 'SU-24', 'SU-34',
    'SU-35', 'T-43', 'T-6', 'TU-160', 'TU-22', 'TU-95', 'U-2')
VEHICLE_FINE_CLASSES = (
    'car', 'bus', 'camping_car', 'Cargo Truck', 'dump truck', 'Excavator',
    'other', 'pickup', 'small-vehicle', 'tractor', 'Trailer',
    'Truck Tractor', 'van')
SHIP_CIVIL_CLASSES = (
    'bargePontoon', 'bulkCarrier', 'Car_carrier', 'coastGuard',
    'Container_Ship', 'Dock', 'dredgerReclamation', 'dredging', 'drill',
    'Engineering_Ship', 'Fishing_Vessel', 'Hovercraft', 'Large_sail_ship',
    'Liquid_Cargo_Ship', 'lpg', 'Merchant', 'offshore', 'oreCarrier',
    'passenger', 'RoRo', 'serviceCraft', 'Small_leisure_craft',
    'small_Ro-Ro_ferry', 'tiny_boat', 'tiny_ship', 'Tugboat', 'yacht')
SHIP_AS_CLASSES = ('Auxiliary Ships', 'Masyuu AS', 'Sanantonio AS', 'other AS')
SHIP_CRUISER_CLASSES = ('other cruiser', 'Ticonderoga')
SHIP_COMMANDER_CLASSES = ('other Commander', 'USS Blue Ridge (LCC-19)')
SHIP_CV_CLASSES = ('Enterprise', 'Midway', 'Nimitz', 'other Aircraft carrier')
SHIP_DD_CLASSES = (
    'Arleigh Burke DD', 'Asagiri DD', 'Atago DD', 'Hatsuyuki DD',
    'Hyuga DDH', 'other Destroyer')
SHIP_FF_CLASSES = ('Perry FF', 'Frigate')
SHIP_LANDING_CLASSES = (
    'Austin LL', 'LHA LL', 'LSD_41 LL', 'Osumi LL', 'Wasp LL', 'LL',
    'other landing')
SHIP_LCS_CLASSES = ('DULI', )

FINE_CLASSES = (
    'Airliner', *PLANE_AIRLINER_CLASSES, 'notairliner',
    *PLANE_NOTAIRLINER_CLASSES, 'other-airplane', *VEHICLE_FINE_CLASSES,
    'civil_ship', *SHIP_CIVIL_CLASSES, 'war', 'AOE', 'AS', *SHIP_AS_CLASSES,
    'C', *SHIP_CRUISER_CLASSES, 'commander', *SHIP_COMMANDER_CLASSES, 'CV',
    *SHIP_CV_CLASSES, 'DD', *SHIP_DD_CLASSES, 'EPF', 'FF', *SHIP_FF_CLASSES,
    'Landing', *SHIP_LANDING_CLASSES, 'LCS', *SHIP_LCS_CLASSES,
    'Medical ship', 'other Warship', 'patrolForce', 'Submarine', 'Test ship')

CLASS_ORDER = tuple(dict.fromkeys((*BASE_CLASSES, *FINE_CLASSES)))
CLASS_SET = set(CLASS_ORDER)
FINE_ALIASES = {
    'YuDao LL': 'LL',
    'YuDeng LL': 'LL',
    'YuTing LL': 'LL',
    'YuZhao LL': 'LL',
}
CATEGORY_ALIASES = {'car': 'vehicle'}
CLASS_HIERARCHY = {
    'plane': ('Airliner', 'notairliner', 'other-airplane'),
    'Airliner': PLANE_AIRLINER_CLASSES,
    'notairliner': PLANE_NOTAIRLINER_CLASSES,
    'vehicle': VEHICLE_FINE_CLASSES,
    'ship': ('civil_ship', 'war'),
    'civil_ship': SHIP_CIVIL_CLASSES,
    'war': (
        'AOE', 'AS', 'C', 'commander', 'CV', 'DD', 'EPF', 'FF', 'Landing',
        'LCS', 'Medical ship', 'other Warship', 'patrolForce', 'Submarine',
        'Test ship'),
    'AS': SHIP_AS_CLASSES,
    'C': SHIP_CRUISER_CLASSES,
    'commander': SHIP_COMMANDER_CLASSES,
    'CV': SHIP_CV_CLASSES,
    'DD': SHIP_DD_CLASSES,
    'FF': SHIP_FF_CLASSES,
    'Landing': SHIP_LANDING_CLASSES,
    'LCS': SHIP_LCS_CLASSES,
}


@dataclass
class SampleRef:
    split: str
    image_id: int
    file_name: str
    category: str
    source_category: str
    ann_id: int
    bbox: Tuple[float, float, float, float]
    augmentation_id: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Crop fine-class examples and make a class overview sheet.')
    parser.add_argument(
        '--train-json',
        default='train_fine_tmp_merged_cleaned_hierarchy.json')
    parser.add_argument(
        '--val-json',
        default='val_fine_tmp_merged_cleaned_hierarchy.json')
    parser.add_argument(
        '--train-img-root',
        default=r'H:\dataset\Levir-Det\train_tmp\imgs')
    parser.add_argument(
        '--val-img-root',
        default=r'H:\dataset\Levir-Det\val_tmp\imgs')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/fine_class_crop_overview',
        help='Output directory for crops, manifest and contact sheet.')
    parser.add_argument(
        '--max-crops-per-class',
        type=int,
        default=9,
        help='Maximum saved crops for each class.')
    parser.add_argument(
        '--show-per-class',
        type=int,
        default=9,
        help='Number of crops shown in the contact-sheet row.')
    parser.add_argument(
        '--expand-ratio',
        type=float,
        default=1.2,
        help='BBox expansion ratio around the box center.')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--columns', type=int, default=3)
    parser.add_argument('--thumb-size', type=int, default=84)
    parser.add_argument(
        '--crop-size',
        type=int,
        default=160,
        help='Saved square crop size after resizing.')
    parser.add_argument('--label-width', type=int, default=260)
    parser.add_argument('--row-height', type=int, default=96)
    parser.add_argument('--gap', type=int, default=10)
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=92,
        help='JPEG quality for saved crops.')
    parser.add_argument(
        '--index-missing-images',
        action='store_true',
        help='If root/file_name is missing, scan image roots by basename.')
    parser.add_argument(
        '--max-ann-per-json',
        type=int,
        default=None,
        help='Debug option: stop after N annotations per json.')
    parser.add_argument(
        '--save-pdf',
        action='store_true',
        default=True,
        help='Also save the contact sheet as PDF. Enabled by default.')
    parser.add_argument(
        '--no-save-pdf',
        action='store_false',
        dest='save_pdf',
        help='Disable PDF export.')
    parser.add_argument(
        '--pdf-resolution',
        type=int,
        default=300,
        help='Resolution metadata used when saving the PDF.')
    parser.add_argument(
        '--no-fill-parent-with-descendants',
        action='store_true',
        help='Disable filling empty/underfilled parent classes with descendant '
        'samples in the saved crops and contact sheet.')
    return parser.parse_args()


def make_progress(iterable: Iterable, desc: str, total: Optional[int] = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        print(desc)
        return iterable


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', '_', name)
    name = re.sub(r'\s+', '_', name).strip('._ ')
    return name or 'unknown'


def normalize_label(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return FINE_ALIASES.get(name, name)


def deepest_fine_label(ann: dict) -> Optional[str]:
    for key in ('fine_category_name', 'fine_category_level4',
                'fine_category_level3', 'fine_category_level2',
                'fine_category_level1', 'fine_supercategory'):
        name = normalize_label(ann.get(key))
        if name in CLASS_SET:
            return name
    return None


def get_category_name(ann: dict,
                      cat_id_to_name: Dict[int, str]) -> Optional[str]:
    fine = deepest_fine_label(ann)
    if fine is not None:
        return fine
    name = cat_id_to_name.get(int(ann['category_id']))
    return CATEGORY_ALIASES.get(name, name)


def build_descendant_map() -> Dict[str, Tuple[str, ...]]:
    descendants: Dict[str, Tuple[str, ...]] = {}

    def collect(name: str) -> Tuple[str, ...]:
        children = CLASS_HIERARCHY.get(name, ())
        values: List[str] = []
        for child in children:
            values.append(child)
            values.extend(collect(child))
        return tuple(dict.fromkeys(values))

    for cls in CLASS_ORDER:
        descendants[cls] = collect(cls)
    return descendants


DESCENDANTS = build_descendant_map()


def load_json_full(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def try_import_ijson():
    try:
        import ijson
        return ijson
    except ImportError:
        return None


def iter_json_array_std(json_path: Path,
                        item: str,
                        chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream a top-level array using only the Python standard library."""
    decoder = json.JSONDecoder()
    pattern = f'"{item}"'
    with json_path.open('r', encoding='utf-8') as f:
        buffer = ''
        pos = 0

        while pattern not in buffer:
            chunk = f.read(chunk_size)
            if not chunk:
                raise KeyError(f'{item!r} not found in {json_path}')
            buffer += chunk
            if pattern in buffer:
                break
            if len(buffer) > len(pattern) * 2:
                buffer = buffer[-len(pattern) * 2:]

        pos = buffer.index(pattern) + len(pattern)
        while True:
            bracket = buffer.find('[', pos)
            if bracket >= 0:
                pos = bracket + 1
                break
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f'Cannot find array start for {item!r}.')
            buffer += chunk

        eof = False
        while True:
            while True:
                while pos < len(buffer) and buffer[pos] in ' \r\n\t,':
                    pos += 1
                if pos < len(buffer):
                    break
                chunk = f.read(chunk_size)
                if not chunk:
                    eof = True
                    break
                buffer = buffer[pos:] + chunk
                pos = 0
            if eof:
                break
            if buffer[pos] == ']':
                break

            while True:
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                    yield obj
                    pos = end
                    if pos > chunk_size:
                        buffer = buffer[pos:]
                        pos = 0
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk


def iter_json_array(path: Path, item: str) -> Iterator[dict]:
    """Iterate a top-level JSON array with optional ijson support."""
    ijson = try_import_ijson()
    if ijson is not None:
        with path.open('rb') as f:
            yield from ijson.items(f, f'{item}.item')
        return

    yield from iter_json_array_std(path, item)


def get_json_counts(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Return images/annotations counts if cheap enough."""
    ijson = try_import_ijson()
    if ijson is not None:
        return None, None
    data = load_json_full(path)
    return len(data.get('images', [])), len(data.get('annotations', []))


def load_image_map(json_path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    cat_id_to_name: Dict[int, str] = {}
    image_id_to_file: Dict[int, str] = {}

    for cat in iter_json_array(json_path, 'categories'):
        cat_id_to_name[int(cat['id'])] = cat['name']
    for img in make_progress(iter_json_array(json_path, 'images'),
                             f'images {json_path.name}'):
        image_id_to_file[int(img['id'])] = img['file_name']
    return cat_id_to_name, image_id_to_file


def reservoir_add(samples: Dict[str, List[SampleRef]],
                  seen: Dict[str, int], sample: SampleRef, max_count: int,
                  rng: random.Random) -> None:
    cls = sample.category
    seen[cls] += 1
    bucket = samples[cls]
    if len(bucket) < max_count:
        bucket.append(sample)
        return
    idx = rng.randint(0, seen[cls] - 1)
    if idx < max_count:
        bucket[idx] = sample


def collect_samples_for_json(json_path: Path, split: str, img_root: Path,
                             samples: Dict[str, List[SampleRef]],
                             seen: Dict[str, int], args: argparse.Namespace,
                             rng: random.Random) -> None:
    cat_id_to_name, image_id_to_file = load_image_map(json_path)
    ann_iter = iter_json_array(json_path, 'annotations')
    total = None

    if args.max_ann_per_json is not None:
        ann_iter = iter_limited(ann_iter, args.max_ann_per_json)
        total = args.max_ann_per_json

    for ann in make_progress(
            ann_iter, f'annotations {json_path.name}', total=total):
        category = get_category_name(ann, cat_id_to_name)
        if category not in CLASS_SET:
            continue
        image_id = int(ann['image_id'])
        file_name = image_id_to_file.get(image_id)
        if file_name is None:
            continue
        bbox = tuple(float(v) for v in ann['bbox'])
        sample = SampleRef(
            split=split,
            image_id=image_id,
            file_name=file_name,
            category=category,
            source_category=category,
            ann_id=int(ann.get('id', -1)),
            bbox=bbox)
        reservoir_add(samples, seen, sample, args.max_crops_per_class, rng)


def iter_limited(iterator: Iterator[dict], limit: int) -> Iterator[dict]:
    for idx, item in enumerate(iterator):
        if idx >= limit:
            break
        yield item


def build_basename_index(roots: Sequence[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for root in roots:
        for path in make_progress(root.rglob('*'), f'index {root}'):
            if path.is_file():
                index.setdefault(path.name, path)
    return index


def resolve_image_path(root: Path, file_name: str,
                       basename_index: Optional[Dict[str, Path]] = None
                       ) -> Optional[Path]:
    raw = Path(file_name)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend((root / file_name, root / raw.name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if basename_index is not None:
        return basename_index.get(raw.name)
    return None


def expanded_square_xyxy(bbox: Sequence[float], ratio: float
                         ) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    cx = x + w * 0.5
    cy = y + h * 0.5
    side = max(1.0, max(w, h) * ratio)
    x1 = math.floor(cx - side * 0.5)
    y1 = math.floor(cy - side * 0.5)
    x2 = math.ceil(cx + side * 0.5)
    y2 = math.ceil(cy + side * 0.5)
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return x1, y1, x2, y2


def copy_for_category(ref: SampleRef, category: str) -> SampleRef:
    return SampleRef(
        split=ref.split,
        image_id=ref.image_id,
        file_name=ref.file_name,
        category=category,
        source_category=ref.source_category,
        ann_id=ref.ann_id,
        bbox=ref.bbox,
        augmentation_id=ref.augmentation_id)


def copy_with_augmentation(ref: SampleRef, augmentation_id: int) -> SampleRef:
    return SampleRef(
        split=ref.split,
        image_id=ref.image_id,
        file_name=ref.file_name,
        category=ref.category,
        source_category=ref.source_category,
        ann_id=ref.ann_id,
        bbox=ref.bbox,
        augmentation_id=augmentation_id)


def build_output_samples(samples: Dict[str, List[SampleRef]],
                         args: argparse.Namespace,
                         rng: random.Random) -> Dict[str, List[SampleRef]]:
    output = {
        cls: list(samples.get(cls, []))[:args.max_crops_per_class]
        for cls in CLASS_ORDER
    }
    if args.no_fill_parent_with_descendants:
        return output

    for cls in CLASS_ORDER:
        current = list(output.get(cls, []))
        if len(current) >= args.max_crops_per_class:
            continue

        candidates: List[SampleRef] = []
        for child in DESCENDANTS.get(cls, ()):
            candidates.extend(samples.get(child, []))
        if not candidates:
            continue

        rng.shuffle(candidates)
        existing = {(ref.split, ref.image_id, ref.ann_id) for ref in current}
        for ref in candidates:
            key = (ref.split, ref.image_id, ref.ann_id)
            if key in existing:
                continue
            current.append(copy_for_category(ref, cls))
            existing.add(key)
            if len(current) >= args.max_crops_per_class:
                break
        output[cls] = current

    for cls in CLASS_ORDER:
        current = list(output.get(cls, []))
        if not current or len(current) >= args.max_crops_per_class:
            continue

        base_refs = list(current)
        aug_id = 1
        while len(current) < args.max_crops_per_class:
            ref = base_refs[(len(current) - len(base_refs)) % len(base_refs)]
            current.append(copy_with_augmentation(ref, aug_id))
            aug_id += 1
        output[cls] = current
    return output


def augment_crop(crop: Image.Image, augmentation_id: int) -> Image.Image:
    if augmentation_id <= 0:
        return crop

    mode = (augmentation_id - 1) % 8
    if mode in (0, 3, 5, 7):
        crop = ImageOps.mirror(crop)
    if mode in (1, 4, 6):
        crop = ImageOps.flip(crop)
    if mode == 2:
        crop = crop.rotate(90, expand=False)
    elif mode == 3:
        crop = crop.rotate(180, expand=False)
    elif mode == 4:
        crop = crop.rotate(270, expand=False)
    elif mode == 5:
        crop = ImageEnhance.Contrast(crop).enhance(1.12)
    elif mode == 6:
        crop = ImageEnhance.Brightness(crop).enhance(1.08)
    elif mode == 7:
        crop = ImageEnhance.Color(crop).enhance(0.88)
    return crop


def save_crops(samples: Dict[str, List[SampleRef]], args: argparse.Namespace,
               out_dir: Path) -> List[dict]:
    roots = {
        'train': Path(args.train_img_root),
        'val': Path(args.val_img_root),
    }
    basename_index = build_basename_index(tuple(roots.values())) \
        if args.index_missing_images else None

    refs = [ref for cls in CLASS_ORDER for ref in samples.get(cls, [])]
    by_image: Dict[Tuple[str, str], List[SampleRef]] = defaultdict(list)
    for ref in refs:
        by_image[(ref.split, ref.file_name)].append(ref)

    crop_root = out_dir / 'crops'
    crop_root.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    per_class_index: Dict[str, int] = defaultdict(int)

    for (split, file_name), image_refs in make_progress(
            by_image.items(), 'crop images', total=len(by_image)):
        image_path = resolve_image_path(
            roots[split], file_name, basename_index=basename_index)
        if image_path is None:
            print(f'Warning: image not found: split={split}, file={file_name}')
            continue
        try:
            image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
        except Exception as exc:
            print(f'Warning: failed to open {image_path}: {exc}')
            continue

        for ref in image_refs:
            per_class_index[ref.category] += 1
            crop_box = expanded_square_xyxy(ref.bbox, args.expand_ratio)
            crop = image.crop(crop_box)
            crop = augment_crop(crop, ref.augmentation_id)
            if crop.size != (args.crop_size, args.crop_size):
                crop = crop.resize(
                    (args.crop_size, args.crop_size), Image.LANCZOS)
            cls_dir = crop_root / safe_name(ref.category)
            cls_dir.mkdir(parents=True, exist_ok=True)
            crop_name = (
                f'{per_class_index[ref.category]:03d}__{ref.split}'
                f'__img{ref.image_id}__ann{ref.ann_id}'
                f'__src_{safe_name(ref.source_category)}'
                f'__aug{ref.augmentation_id}.jpg')
            crop_path = cls_dir / crop_name
            crop.save(crop_path, quality=args.jpeg_quality)
            manifest.append({
                'class': ref.category,
                'source_class': ref.source_category,
                'split': ref.split,
                'image_id': ref.image_id,
                'annotation_id': ref.ann_id,
                'image_path': str(image_path),
                'bbox_xywh': list(ref.bbox),
                'crop_xyxy': list(crop_box),
                'augmentation_id': ref.augmentation_id,
                'crop_path': str(crop_path),
            })
    return manifest


def save_manifest(manifest: List[dict], out_dir: Path) -> None:
    json_path = out_dir / 'manifest.json'
    csv_path = out_dir / 'manifest.csv'
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = [
        'class', 'source_class', 'split', 'image_id', 'annotation_id',
        'image_path', 'bbox_xywh', 'crop_xyxy', 'augmentation_id',
        'crop_path'
    ]
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        r'C:\Windows\Fonts\times.ttf',
        r'C:\Windows\Fonts\timesbd.ttf',
        '/usr/share/fonts/truetype/msttcorefonts/times.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
        r'C:\Windows\Fonts\arial.ttf',
        r'C:\Windows\Fonts\msyh.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def ellipsize(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    dummy = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = '...'
    while text and draw.textlength(text + suffix, font=font) > max_width:
        text = text[:-1]
    return text + suffix


def make_thumbnail(path: Path, size: int) -> Image.Image:
    try:
        image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        image.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new('RGB', (size, size), (245, 245, 245))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas
    except Exception:
        return Image.new('RGB', (size, size), (230, 230, 230))


def render_contact_sheet(manifest: List[dict], args: argparse.Namespace,
                         out_dir: Path) -> Path:
    by_class: Dict[str, List[Path]] = defaultdict(list)
    for row in manifest:
        by_class[row['class']].append(Path(row['crop_path']))

    columns = args.columns
    rows = math.ceil(len(CLASS_ORDER) / columns)
    thumb = args.thumb_size
    gap = args.gap
    label_w = args.label_width
    row_h = max(args.row_height, thumb + gap)
    col_w = label_w + args.show_per_class * (thumb + gap) + gap
    header_h = 34
    width = columns * col_w + gap
    height = header_h + rows * row_h + gap

    sheet = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = load_font(18)
    small_font = load_font(15)
    header_font = load_font(24)

    draw.text((gap, 6), 'Fine-grained class crop overview',
              fill=(20, 20, 20), font=header_font)
    for idx, cls in enumerate(CLASS_ORDER):
        col = idx % columns
        row = idx // columns
        x0 = gap + col * col_w
        y0 = header_h + row * row_h

        fill = (245, 248, 255) if idx < len(BASE_CLASSES) else (250, 250, 250)
        draw.rectangle(
            [x0, y0, x0 + col_w - gap, y0 + row_h - 2],
            fill=fill,
            outline=(225, 225, 225))
        label = ellipsize(f'{idx + 1:03d} {cls}', font, label_w - 6)
        text_y = y0 + max(2, (row_h - 22) // 2)
        draw.text((x0 + 4, text_y), label, fill=(20, 20, 20), font=font)

        crop_paths = by_class.get(cls, [])[:args.show_per_class]
        tx = x0 + label_w
        for crop_idx in range(args.show_per_class):
            thumb_x = tx + crop_idx * (thumb + gap)
            thumb_y = y0 + (row_h - thumb) // 2
            if crop_idx < len(crop_paths):
                tile = make_thumbnail(crop_paths[crop_idx], thumb)
                sheet.paste(tile, (thumb_x, thumb_y))
            else:
                draw.rectangle(
                    [thumb_x, thumb_y, thumb_x + thumb, thumb_y + thumb],
                    fill=(238, 238, 238),
                    outline=(220, 220, 220))
                draw.text(
                    (thumb_x + 18, thumb_y + 20),
                    '-',
                    fill=(150, 150, 150),
                    font=small_font)

    out_path = out_dir / 'class_contact_sheet.png'
    sheet.save(out_path, dpi=(args.pdf_resolution, args.pdf_resolution))
    if args.save_pdf:
        sheet.save(
            out_dir / 'class_contact_sheet.pdf',
            'PDF',
            resolution=args.pdf_resolution)
    return out_path


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: Dict[str, List[SampleRef]] = defaultdict(list)
    seen: Dict[str, int] = defaultdict(int)

    inputs = [
        (Path(args.train_json), 'train', Path(args.train_img_root)),
        (Path(args.val_json), 'val', Path(args.val_img_root)),
    ]
    for json_path, split, img_root in inputs:
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        if not img_root.exists():
            raise FileNotFoundError(img_root)
        collect_samples_for_json(json_path, split, img_root, samples, seen,
                                 args, rng)

    output_samples = build_output_samples(samples, args, rng)
    manifest = save_crops(output_samples, args, out_dir)
    save_manifest(manifest, out_dir)
    sheet_path = render_contact_sheet(manifest, args, out_dir)

    summary = {
        cls: {
            'seen_annotations': seen.get(cls, 0),
            'saved_crops': sum(1 for row in manifest if row['class'] == cls),
            'direct_saved_crops': sum(
                1 for row in manifest
                if row['class'] == cls and row['source_class'] == cls),
        }
        for cls in CLASS_ORDER
    }
    (out_dir / 'class_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'classes={len(CLASS_ORDER)}')
    print(f'saved_crops={len(manifest)}')
    print(f'crop_dir={out_dir / "crops"}')
    print(f'manifest={out_dir / "manifest.csv"}')
    print(f'contact_sheet={sheet_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
