"""Prepare a new COCO task from the released LEVIRDetNet-30 checkpoint.

Run inside the project's Docker/Conda environment, from the repository root:
    python tools/transfer_ckpt.py --new-class plane,ship --data-root /data/my_dataset
    python tools/train.py work_dirs/finetune/finetune.py
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_KEYS = tuple(f'bbox_head.cls_branches.{i}.{suffix}'
                        for i in range(7) for suffix in ('weight', 'bias'))
EMBEDDING_KEY = 'dn_query_generator.label_embedding.weight'
CLASS_KEYS = CLASSIFIER_KEYS + (EMBEDDING_KEY,)


def parse_classes(values):
    """Accept space-separated names, ASCII commas, or Chinese commas."""
    names = [name.strip() for value in values for name in re.split('[,，]', value)]
    if not names or any(not name or any(ord(c) < 32 for c in name) for name in names):
        raise ValueError('Class names must be nonempty, without control characters.')
    if len(names) != len(set(names)):
        raise ValueError('Class names must be unique (exact spelling, case-sensitive).')
    return tuple(names)


def transfer_state(checkpoint, new_classes, source_classes, seed=0):
    """Reuse exact-name class rows; initialize novel rows without changing source."""
    new_classes = parse_classes(list(new_classes))
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError('Expected the complete epoch_117.pth checkpoint with state_dict.')
    saved_classes = ((checkpoint.get('meta') or {}).get('dataset_meta') or {}).get('classes')
    if len(source_classes) != 30 or tuple(saved_classes or ()) != tuple(source_classes):
        raise ValueError('Checkpoint metadata must match the released 30-class configuration. Use epoch_117.pth.')
    state = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError('state_dict must contain named tensors.')
        name = key.removeprefix('module.')
        if name in state:
            raise ValueError(f'Duplicate checkpoint key after removing module.: {name}')
        state[name] = value
    actual = {k for k in state if k.startswith('bbox_head.cls_branches.')}
    if actual != set(CLASSIFIER_KEYS) or EMBEDDING_KEY not in state:
        raise ValueError('Expected all seven DFINE classification branches and the denoising label embedding.')
    for key in CLASS_KEYS:
        shape = (30,) if key.endswith('.bias') else (30, 256)
        if tuple(state[key].shape) != shape or not state[key].is_floating_point():
            raise ValueError(f'Unexpected 30-class parameter shape/dtype: {key}: {tuple(state[key].shape)}')
    for prefix in ('backbone.dinov3.', 'backbone.sta.', 'encoder.', 'decoder.',
                   'bbox_head.reg_branches.', 'gsd_predictor.'):
        if not any(k.startswith(prefix) for k in state):
            raise ValueError(f'The complete detector checkpoint is missing {prefix} parameters.')

    generator = torch.Generator(device='cpu').manual_seed(seed)
    mapping = [source_classes.index(name) if name in source_classes else None for name in new_classes]
    converted = OrderedDict(state)
    for key in CLASS_KEYS:
        source = state[key].detach().cpu()
        target = torch.empty((len(new_classes), *source.shape[1:]), dtype=source.dtype)
        if key == EMBEDDING_KEY:
            target.normal_(mean=0, std=1, generator=generator)
        elif key.endswith('.bias'):
            target.fill_(math.log(0.01 / 0.99))
        else:
            torch.nn.init.kaiming_uniform_(target, a=math.sqrt(5), generator=generator)
        for target_index, source_index in enumerate(mapping):
            if source_index is not None:
                target[target_index].copy_(source[source_index])
        converted[key] = target
    report = dict(
        source_classes=list(source_classes), target_classes=list(new_classes),
        class_mapping=[dict(name=name, target_index=i, source_index=mapping[i],
                            initialization='copied' if mapping[i] is not None else 'new')
                       for i, name in enumerate(new_classes)],
        adapted_keys=list(CLASS_KEYS), preserved_tensor_count=len(state)-len(CLASS_KEYS),
        source_state='state_dict', seed=seed,
    )
    # New task, not resume: no old optimizer, EMA, scheduler, epoch or palette.
    result = dict(state_dict=converted,
                  meta=dict(dataset_meta=dict(classes=tuple(new_classes)), transfer_info=report))
    return result, report


def _integer_id(value):
    return isinstance(value, int) and not isinstance(value, bool)


def prepare_annotations(path, classes, image_dir, *, crowd_as_normal=False):
    """Validate a COCO split and return a category-ordered copy, preserving IDs."""
    path, image_dir = Path(path), Path(image_dir)
    if not path.is_file():
        raise FileNotFoundError(f'Missing COCO annotation file: {path}')
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Missing image directory: {image_dir}')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or any(not isinstance(data.get(k), list)
                                         for k in ('images', 'annotations', 'categories')):
        raise ValueError(f'{path}: expected COCO images, annotations and categories arrays.')
    if any(not isinstance(item, dict) for key in ('images', 'annotations', 'categories') for item in data[key]):
        raise ValueError(f'{path}: COCO arrays must contain JSON objects.')
    categories = data['categories']
    names = [c.get('name') for c in categories]
    cat_ids = [c.get('id') for c in categories]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError(f'{path}: category names must be unique strings.')
    if any(not _integer_id(i) for i in cat_ids) or len(set(cat_ids)) != len(cat_ids):
        raise ValueError(f'{path}: category IDs must be unique integers.')
    if set(names) != set(classes):
        raise ValueError(f'{path}: categories must contain exactly --new-class names. '
                         f'Missing={sorted(set(classes)-set(names))}; extra={sorted(set(names)-set(classes))}. '
                         'Prepare target-task annotations first; the script does not silently relabel/filter other classes.')
    by_name = {c['name']: c for c in categories}
    by_id = {c['id']: c['name'] for c in categories}
    image_ids = set()
    root = image_dir.resolve()
    for item in data['images']:
        ident, name = item.get('id'), item.get('file_name')
        if not _integer_id(ident) or ident in image_ids:
            raise ValueError(f'{path}: image IDs must be unique integers.')
        image_ids.add(ident)
        if not isinstance(name, str) or not name:
            raise ValueError(f'{path}: each image needs a relative file_name.')
        posix, windows = PurePosixPath(name), PureWindowsPath(name)
        if posix.is_absolute() or windows.drive or windows.root or '..' in posix.parts or '..' in windows.parts:
            raise ValueError(f'{path}: file_name must stay inside the split image directory: {name}')
        if '\\' in name:
            raise ValueError(f'{path}: use forward slashes in COCO file_name for Linux portability: {name}')
        image_path = (root / name).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f'{path}: image link must stay inside the split image directory: {name}')
        if not image_path.is_file():
            raise FileNotFoundError(f'Missing image referenced by {path.name}: {image_path}')
        if any(not isinstance(item.get(k), (int, float)) or not math.isfinite(item[k]) or item[k] <= 0
               for k in ('width', 'height')):
            raise ValueError(f'{path}: positive image width/height required for image {ident}.')
    if not image_ids:
        raise ValueError(f'{path}: the split contains no images.')
    ann_ids, crowd_count = set(), 0
    counts = dict.fromkeys(classes, 0)
    annotations = []
    for ann in data['annotations']:
        ident = ann.get('id')
        if not _integer_id(ident) or ident in ann_ids:
            raise ValueError(f'{path}: annotation IDs must be unique integers.')
        ann_ids.add(ident)
        if (not _integer_id(ann.get('image_id')) or not _integer_id(ann.get('category_id'))
                or ann['image_id'] not in image_ids or ann['category_id'] not in by_id):
            raise ValueError(f'{path}: annotation {ident} references an unknown image/category.')
        box = ann.get('bbox')
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in box)
                or box[2] <= 0 or box[3] <= 0):
            raise ValueError(f'{path}: annotation {ident} needs a finite COCO [x,y,width,height] box.')
        if not isinstance(ann.get('area'), (int, float)) or not math.isfinite(ann['area']) or ann['area'] <= 0:
            raise ValueError(f'{path}: annotation {ident} needs a positive area.')
        if ann.get('iscrowd', 0) not in (0, 1):
            raise ValueError(f'{path}: annotation {ident} iscrowd must be 0 or 1.')
        crowd_count += int(bool(ann.get('iscrowd', 0)))
        counts[by_id[ann['category_id']]] += 1
        # Add a missing standard flag only to the copy; retain every special field.
        item = dict(ann)
        item['iscrowd'] = 0 if crowd_as_normal else int(bool(ann.get('iscrowd', 0)))
        annotations.append(item)
    prepared = dict(data)
    prepared['categories'] = [by_name[name] for name in classes]
    prepared['annotations'] = annotations
    report = dict(source=str(path.resolve()), images=len(image_ids), annotations=len(annotations),
                  category_ids=[by_name[name]['id'] for name in classes], counts=counts,
                  original_crowd=crowd_count, crowd_as_normal=crowd_as_normal)
    return prepared, report


def render_config(template, replacements):
    """Replace only explicit top-level settings; keep the readable config body."""
    text = Path(template).read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                if name in found:
                    raise ValueError(f'Duplicate template setting: {name}')
                found[name] = node
    if set(found) != set(replacements):
        raise ValueError(f'Missing template settings: {sorted(set(replacements)-set(found))}')
    for name, node in sorted(found.items(), key=lambda pair: pair[1].lineno, reverse=True):
        lines[node.lineno-1:node.end_lineno] = [f'{name} = {replacements[name]!r}\n']
    text = ''.join(lines)
    ast.parse(text)
    return '# Generated by tools/transfer_ckpt.py. Run training from the project root.\n' + text


def _path_setting(path):
    """Keep paths under the current project portable; retain external paths."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Transfer LEVIRDetNet-30 to a new COCO detection task.')
    parser.add_argument('--new-class', '--new-classes', dest='new_classes', nargs='+', required=True,
                        help='Exact COCO category names, e.g. plane,ship or plane ship')
    parser.add_argument('--weights', default='epoch_117.pth', help='Trusted released 30-class checkpoint')
    parser.add_argument('--data-root', default='data/my_dataset')
    parser.add_argument('--train-ann', default='annotations/train.json')
    parser.add_argument('--val-ann', default='annotations/val.json')
    parser.add_argument('--train-images', default='train/images')
    parser.add_argument('--val-images', default='val/images')
    parser.add_argument('--out-dir', default='work_dirs/finetune', help='A new directory; existing directories are not overwritten')
    parser.add_argument('--epochs', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=2, help='Training batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--crowd-as-normal', action='store_true', help='Set iscrowd=0 in generated annotation copies only')
    args = parser.parse_args(argv)
    try:
        classes = parse_classes(args.new_classes)
        if args.epochs < 2 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError('Use epochs >= 2, batch-size >= 1, and a positive finite lr.')
        if not 0 <= args.seed < 2**63:
            raise ValueError('seed must be in [0, 2**63).')
        out = Path(args.out_dir).expanduser().resolve()
        if out.exists():
            raise FileExistsError(f'Output directory already exists: {out}. Choose a new --out-dir; no existing run is overwritten.')
        root = Path(args.data_root).expanduser().resolve()
        train_images, val_images = root / args.train_images, root / args.val_images
        print('Checking annotations, class order and image paths...', flush=True)
        train_data, train_report = prepare_annotations(root / args.train_ann, classes, train_images,
                                                       crowd_as_normal=args.crowd_as_normal)
        val_data, val_report = prepare_annotations(root / args.val_ann, classes, val_images,
                                                   crowd_as_normal=args.crowd_as_normal)
        source = Path(args.weights).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'Missing 30-class checkpoint: {source}')
        from mmengine.config import Config
        base = PROJECT_ROOT / 'configs/levirdetnet/levirdetnet-30class.py'
        source_cfg = Config.fromfile(str(base))
        if source_cfg.model.bbox_head.type != 'DFINEHead' or source_cfg.model.bbox_head.num_classes != 30:
            raise ValueError('Expected the released 30-class DFINEHead configuration.')
        print('Reading state_dict from the 30-class checkpoint...', flush=True)
        checkpoint = torch.load(str(source), map_location='cpu', weights_only=False, mmap=True)
        converted, report = transfer_state(checkpoint, classes, tuple(source_cfg.classes), args.seed)
        del checkpoint
        report.update(source_checkpoint=str(source), source_checkpoint_bytes=source.stat().st_size,
                      train=train_report, val=val_report)
        try:
            base_path = Path(os.path.relpath(base, out)).as_posix()
        except ValueError:  # Different Windows drive; generate inside the intended runtime.
            base_path = base.as_posix()
        config_text = render_config(PROJECT_ROOT/'configs/levirdetnet/levirdetnet-finetune.py', dict(
            _base_=base_path, classes=classes, data_root=_path_setting(root),
            train_ann_file=_path_setting(out/'annotations/train.json'),
            val_ann_file=_path_setting(out/'annotations/val.json'),
            train_image_dir=Path(args.train_images).as_posix(), val_image_dir=Path(args.val_images).as_posix(),
            load_from=_path_setting(out/'init.pth'), work_dir=_path_setting(out),
            max_epochs=args.epochs, batch_size=args.batch_size, base_lr=args.lr))

        # All validation precedes output creation. Only write a new task directory.
        out.mkdir(parents=True, exist_ok=False)
        (out/'annotations').mkdir()
        print(f'Writing the new task to {out}...', flush=True)
        torch.save(converted, str(out/'init.pth.tmp'))
        (out/'init.pth.tmp').rename(out/'init.pth')
        for name, data in [('train', train_data), ('val', val_data)]:
            (out/f'annotations/{name}.json').write_text(json.dumps(data, ensure_ascii=False)+'\n', encoding='utf-8')
        (out/'transfer_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (out/'finetune.py').write_text(config_text, encoding='utf-8')
        for item in report['class_mapping']:
            print(f"  {item['name']}: " + (f"copied source row {item['source_index']}" if item['source_index'] is not None else 'newly initialized'))
        print(f"Adapted {len(CLASS_KEYS)} tensors; preserved {report['preserved_tensor_count']} tensors.")
        print('Start a new training run (do not use --resume for the first run):')
        print(f'python tools/train.py "{_path_setting(out / "finetune.py")}"')
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
