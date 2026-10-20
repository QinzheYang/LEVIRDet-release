"""CPU regression tests for the released 30-class transfer workflow.

No detector, compiled MMCV operations, GPU, or large checkpoint is required.
Run: python -m pytest tests/test_utils/test_transfer_ckpt.py
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools' / 'transfer_ckpt.py'
TEMPLATE = ROOT / 'configs' / 'levirdetnet' / 'levirdetnet-finetune.py'
SPEC = importlib.util.spec_from_file_location('levir_transfer_under_test', SCRIPT)
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)
SOURCE_CLASSES = ('plane',) + tuple(f'class_{i}' for i in range(1, 9)) + ('ship',) + tuple(
    f'class_{i}' for i in range(10, 30))


@pytest.fixture
def checkpoint():
    state = {}
    for index, key in enumerate(transfer.CLASS_KEYS):
        shape = (30,) if key.endswith('.bias') else (30, 256)
        state[key] = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + index * 10000
    for index, prefix in enumerate(('backbone.dinov3.', 'backbone.sta.', 'encoder.',
                                    'decoder.', 'bbox_head.reg_branches.', 'gsd_predictor.')):
        state[prefix + 'test_weight'] = torch.tensor([index + 0.125])
    state['backbone.sta.bn.num_batches_tracked'] = torch.tensor(41)
    return {
        'state_dict': state,
        'meta': {'dataset_meta': {'classes': SOURCE_CLASSES, 'palette': [(1, 2, 3)] * 30},
                 'epoch': 117, 'iter': 999, 'cfg': 'obsolete configuration', 'seed': 7},
        'optimizer': {'state': 'obsolete'},
        'param_schedulers': ['obsolete'],
        'message_hub': {'runtime_info': {'epoch': 117}},
        'ema_state_dict': {'obsolete': torch.tensor(1)},
        'ema_dynamic_momentum_dict': {'restart_epoch': 166},
    }


def make_split(root, split, categories=None, *, crowd=1):
    categories = categories or [
        {'id': 71, 'name': 'ship', 'supercategory': 'transport', 'special': {'part': 'p3'}},
        {'id': 8, 'name': 'plane', 'supercategory': 'transport', 'special': {'part': 'p1'}},
    ]
    image_dir = root / split / 'images'
    image_dir.mkdir(parents=True)
    (image_dir / 'sample.jpg').write_bytes(b'image existence is sufficient for annotation validation')
    data = {
        'info': {'description': 'custom fields survive', 'hierarchy': {'vehicle': ['ship']}},
        'licenses': [{'id': 12, 'name': 'dataset license'}],
        'custom_top': {'provenance': ['part1', 'part3']},
        'images': [{'id': 21, 'file_name': 'sample.jpg', 'width': 1024, 'height': 768,
                    'part': 'part1', 'gsd': 0.7}],
        'categories': categories,
        'annotations': [
            {'id': 301 + i, 'image_id': 21, 'category_id': c['id'],
             'bbox': [4, 5, 20, 30], 'area': 600, 'iscrowd': crowd,
             'special': {'ancestors': [999], 'source': 'part1'}, 'segmentation': []}
            for i, c in enumerate(categories)
        ],
    }
    path = root / 'annotations' / (split + '.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path, image_dir, data


def write_project(root):
    config_dir = root / 'configs' / 'levirdetnet'
    config_dir.mkdir(parents=True)
    # Deliberately inherited stale values verify that the generated config
    # actually replaces the source training setup, rather than only max_epochs.
    base = f'''classes = {SOURCE_CLASSES!r}
model = dict(type='DEIMV2GSDGuided',
             backbone=dict(type='DINOv3STAs', weights_path='old_dino.pth', init_cfg=dict(type='Pretrained')),
             bbox_head=dict(type='DFINEHead', num_classes=30),
             gsd_cfg=dict(ckpt='old_gsd.pt', online_trainable=True),
             data_preprocessor=dict(batch_augments=[dict(type='BatchSyncRandomResize')]),
             train_cfg=dict(assigner=dict(type='OldAssigner'), switch_assigner=dict(switch_epoch=149)))
custom_hooks = [dict(type='EMADynamicMomentumHook', restart_epoch=166), dict(type='PipelineSwitchHook')]
train_dataloader = dict(dataset=dict(type='MultiImageMixDataset', dataset=dict(old=True)))
val_dataloader = dict(dataset=dict(type='OldDataset'))
test_dataloader = val_dataloader
val_evaluator = dict(type='OldMetric', old=True)
test_evaluator = val_evaluator
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=192, val_interval=5)
optim_wrapper = dict(type='OldOptimWrapper', paramwise_cfg=dict(custom_keys={{'obsolete': dict(lr_mult=0)}}))
param_scheduler = [dict(type='OldScheduler', end=192)]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=50, save_best='obsolete'))
resume = True
'''
    (config_dir / 'levirdetnet-30class.py').write_text(base, encoding='utf-8')
    (config_dir / 'levirdetnet-finetune.py').write_text(TEMPLATE.read_text(encoding='utf-8'), encoding='utf-8')


@pytest.mark.parametrize('values', [['plane,ship'], ['plane，ship'], ['plane', 'ship'], [' plane ', ' ship ']])
def test_class_cli_syntax(values):
    assert transfer.parse_classes(values) == ('plane', 'ship')


@pytest.mark.parametrize('values', [[], [''], ['plane,'], ['plane,,ship'], ['plane', 'plane'], ['bad\nname']])
def test_invalid_class_names(values):
    with pytest.raises(ValueError):
        transfer.parse_classes(values)


def test_all_fifteen_parameters_follow_requested_class_order(checkpoint):
    original = copy.deepcopy(checkpoint)
    out, report = transfer.transfer_state(checkpoint, ('ship', 'plane'), SOURCE_CLASSES)
    assert len(report['adapted_keys']) == 15
    assert report['class_mapping'] == [
        dict(name='ship', target_index=0, source_index=9, initialization='copied'),
        dict(name='plane', target_index=1, source_index=0, initialization='copied')]
    for key in transfer.CLASS_KEYS:
        assert torch.equal(out['state_dict'][key], original['state_dict'][key][[9, 0]])
    for key, tensor in original['state_dict'].items():
        assert torch.equal(checkpoint['state_dict'][key], tensor), key
        if key not in transfer.CLASS_KEYS:
            assert torch.equal(out['state_dict'][key], tensor), key
    assert report['preserved_tensor_count'] == len(checkpoint['state_dict']) - 15


def test_novel_classes_are_reproducible_and_do_not_consume_global_rng(checkpoint):
    torch.manual_seed(1234)
    rng = torch.random.get_rng_state().clone()
    first, report = transfer.transfer_state(checkpoint, ('novel', 'plane'), SOURCE_CLASSES, seed=19)
    assert torch.equal(torch.random.get_rng_state(), rng)
    second, _ = transfer.transfer_state(checkpoint, ('novel', 'plane'), SOURCE_CLASSES, seed=19)
    third, _ = transfer.transfer_state(checkpoint, ('novel', 'plane'), SOURCE_CLASSES, seed=20)
    for key in transfer.CLASS_KEYS:
        assert torch.equal(first['state_dict'][key], second['state_dict'][key])
        assert torch.equal(first['state_dict'][key][1], checkpoint['state_dict'][key][0])
        if key.endswith('.bias'):
            assert first['state_dict'][key][0].item() == pytest.approx(math.log(0.01 / 0.99))
        else:
            assert not torch.equal(first['state_dict'][key][0], third['state_dict'][key][0])
            assert torch.isfinite(first['state_dict'][key]).all()
            if key != transfer.EMBEDDING_KEY:
                assert first['state_dict'][key][0].abs().max() <= 1 / math.sqrt(256)
    assert report['class_mapping'][0]['initialization'] == 'new'


def test_same_class_count_still_remaps_rows(checkpoint):
    names = tuple(reversed(SOURCE_CLASSES))
    out, _ = transfer.transfer_state(checkpoint, names, SOURCE_CLASSES)
    for key in transfer.CLASS_KEYS:
        assert torch.equal(out['state_dict'][key], checkpoint['state_dict'][key].flip(0))


def test_transfer_metadata_discards_resume_state_and_old_palette(checkpoint):
    out, report = transfer.transfer_state(checkpoint, ('plane', 'ship'), SOURCE_CLASSES)
    assert set(out) == {'state_dict', 'meta'}
    assert set(out['meta']) == {'dataset_meta', 'transfer_info'}
    assert out['meta']['dataset_meta'] == {'classes': ('plane', 'ship')}
    assert out['meta']['transfer_info'] == report
    assert report['source_state'] == 'state_dict'


def test_module_prefix_is_removed_and_collision_rejected(checkpoint):
    checkpoint['state_dict'] = {'module.' + k: v for k, v in checkpoint['state_dict'].items()}
    out, _ = transfer.transfer_state(checkpoint, ('plane',), SOURCE_CLASSES)
    assert all(not key.startswith('module.') for key in out['state_dict'])
    key = transfer.CLASS_KEYS[0]
    checkpoint['state_dict'][key] = checkpoint['state_dict']['module.' + key]
    with pytest.raises(ValueError, match='Duplicate checkpoint key'):
        transfer.transfer_state(checkpoint, ('plane',), SOURCE_CLASSES)


@pytest.mark.parametrize('key', transfer.CLASS_KEYS)
def test_missing_each_class_parameter_is_rejected(checkpoint, key):
    del checkpoint['state_dict'][key]
    with pytest.raises(ValueError, match='seven DFINE classification branches'):
        transfer.transfer_state(checkpoint, ('plane',), SOURCE_CLASSES)


@pytest.mark.parametrize('shape', [(159, 256), (30, 255), (30, 1, 256)])
def test_wrong_class_weight_shape_rejected(checkpoint, shape):
    checkpoint['state_dict'][transfer.CLASS_KEYS[0]] = torch.zeros(shape)
    with pytest.raises(ValueError, match='shape/dtype'):
        transfer.transfer_state(checkpoint, ('plane',), SOURCE_CLASSES)


@pytest.mark.parametrize('meta_classes', [tuple(f'class_{i}' for i in range(159)), SOURCE_CLASSES[::-1], None])
def test_wrong_source_class_metadata_rejected(checkpoint, meta_classes):
    checkpoint['meta']['dataset_meta']['classes'] = meta_classes
    with pytest.raises(ValueError, match='released 30-class configuration'):
        transfer.transfer_state(checkpoint, ('plane',), SOURCE_CLASSES)


def test_missing_nonclass_architecture_or_integer_classifier_rejected(checkpoint):
    del checkpoint['state_dict']['gsd_predictor.test_weight']
    with pytest.raises(ValueError, match='gsd_predictor'):
        transfer.transfer_state(checkpoint, ('plane',), SOURCE_CLASSES)
    checkpoint['state_dict']['gsd_predictor.test_weight'] = torch.ones(1)
    checkpoint['state_dict'][transfer.EMBEDDING_KEY] = torch.zeros((30, 256), dtype=torch.long)
    with pytest.raises(ValueError, match='shape/dtype'):
        transfer.transfer_state(checkpoint, ('plane',), SOURCE_CLASSES)


def test_train_val_ids_and_order_are_independent_but_names_align(tmp_path):
    train, train_imgs, _ = make_split(tmp_path, 'train')
    val, val_imgs, _ = make_split(tmp_path, 'val', [
        {'id': 700, 'name': 'plane'}, {'id': 2, 'name': 'ship'}])
    a, ar = transfer.prepare_annotations(train, ('plane', 'ship'), train_imgs)
    b, br = transfer.prepare_annotations(val, ('plane', 'ship'), val_imgs)
    assert [c['name'] for c in a['categories']] == ['plane', 'ship']
    assert [c['name'] for c in b['categories']] == ['plane', 'ship']
    assert ar['category_ids'] == [8, 71]
    assert br['category_ids'] == [700, 2]
    assert [ann['category_id'] for ann in a['annotations']] == [71, 8]
    assert [ann['category_id'] for ann in b['annotations']] == [700, 2]


@pytest.mark.parametrize('crowd_as_normal', [False, True])
def test_special_fields_and_source_bytes_survive_annotation_copy(tmp_path, crowd_as_normal):
    path, images, original = make_split(tmp_path, 'train')
    old_bytes = path.read_bytes()
    out, report = transfer.prepare_annotations(path, ('plane', 'ship'), images,
                                                crowd_as_normal=crowd_as_normal)
    assert path.read_bytes() == old_bytes
    for key in ('info', 'licenses', 'custom_top', 'images'):
        assert out[key] == original[key]
    assert {c['name']: c for c in out['categories']} == {c['name']: c for c in original['categories']}
    for new, old in zip(out['annotations'], original['annotations']):
        assert {k: v for k, v in new.items() if k != 'iscrowd'} == {k: v for k, v in old.items() if k != 'iscrowd'}
        assert new['iscrowd'] == (0 if crowd_as_normal else 1)
    assert report['original_crowd'] == 2
    assert report['crowd_as_normal'] is crowd_as_normal


@pytest.mark.parametrize('classes', [('plane',), ('plane', 'ship', 'novel')])
def test_missing_or_extra_category_fails_without_relabeling(tmp_path, classes):
    path, images, _ = make_split(tmp_path, 'train')
    old_bytes = path.read_bytes()
    with pytest.raises(ValueError, match='exactly --new-class names'):
        transfer.prepare_annotations(path, classes, images)
    assert path.read_bytes() == old_bytes


def test_missing_image_fails(tmp_path):
    path, images, _ = make_split(tmp_path, 'train')
    (images / 'sample.jpg').unlink()
    with pytest.raises(FileNotFoundError, match='Missing image referenced'):
        transfer.prepare_annotations(path, ('plane', 'ship'), images)


@pytest.mark.parametrize('name', ['../outside.jpg', 'sub/../../outside.jpg', '/absolute.jpg',
                                  'C:/outside.jpg', r'sub\sample.jpg'])
def test_unsafe_or_nonportable_image_path_fails(tmp_path, name):
    path, images, data = make_split(tmp_path, 'train')
    data['images'][0]['file_name'] = name
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        transfer.prepare_annotations(path, ('plane', 'ship'), images)


def test_symlink_outside_image_root_rejected(tmp_path):
    path, images, data = make_split(tmp_path, 'train')
    outside = tmp_path / 'outside.jpg'
    outside.write_bytes(b'outside')
    link = images / 'link.jpg'
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip('Creating symlinks is unavailable on this host')
    data['images'][0]['file_name'] = 'link.jpg'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        transfer.prepare_annotations(path, ('plane', 'ship'), images)


def test_render_config_missing_setting_is_rejected(tmp_path):
    source = tmp_path / 'simple.py'
    source.write_text('classes = ("plane",)\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Missing template settings'):
        transfer.render_config(source, {'unknown': 2})


def test_cli_complete_output_config_and_no_overwrite(tmp_path, checkpoint, monkeypatch):
    from mmengine.config import Config
    project = tmp_path / 'project'
    write_project(project)
    dataset = project / 'data' / 'my_dataset'
    train, _, _ = make_split(dataset, 'train')
    val, _, _ = make_split(dataset, 'val', [{'id': 1, 'name': 'plane'}, {'id': 98, 'name': 'ship'}])
    original = {p: p.read_bytes() for p in (train, val)}
    torch.save(checkpoint, project / 'epoch_117.pth')
    monkeypatch.chdir(project)
    monkeypatch.setattr(transfer, 'PROJECT_ROOT', project)
    args = ['--new-class', 'ship，plane', '--epochs', '8', '--batch-size', '3',
            '--lr', '0.0002', '--crowd-as-normal']
    assert transfer.main(args) == 0
    out = project / 'work_dirs' / 'finetune'
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()) == [
        'annotations/train.json', 'annotations/val.json', 'finetune.py', 'init.pth', 'transfer_report.json']
    result = torch.load(str(out / 'init.pth'), weights_only=False)
    assert result['meta']['dataset_meta']['classes'] == ('ship', 'plane')
    assert torch.equal(result['state_dict'][transfer.EMBEDDING_KEY], checkpoint['state_dict'][transfer.EMBEDDING_KEY][[9, 0]])
    cfg = Config.fromfile(str(out / 'finetune.py'))
    assert cfg.classes == ('ship', 'plane')
    assert cfg.model.bbox_head.num_classes == 2
    assert cfg.model.backbone.weights_path is None and cfg.model.backbone.init_cfg is None
    assert cfg.model.backbone.finetune is True and cfg.model.gsd_cfg.ckpt == ''
    assert cfg.model.gsd_cfg.online_trainable is True
    assert cfg.model.data_preprocessor.batch_augments is None
    assert 'switch_assigner' not in cfg.model.train_cfg
    assert cfg.model.train_cfg.assigner.type == 'HungarianAssigner'
    assert cfg.custom_hooks == [dict(type='SetEpochInfoHook')]
    assert cfg.train_cfg.max_epochs == 8 and cfg.train_cfg.val_interval == 1
    assert cfg.train_dataloader.batch_size == 3
    assert cfg.train_dataloader.dataset.type == 'CocoDataset'
    assert 'dataset' not in cfg.train_dataloader.dataset
    assert cfg.train_dataloader.dataset.metainfo.classes == ('ship', 'plane')
    assert cfg.val_dataloader.dataset.metainfo.classes == ('ship', 'plane')
    assert Path(cfg.train_dataloader.dataset.ann_file) == out / 'annotations' / 'train.json'
    assert Path(cfg.val_dataloader.dataset.ann_file) == out / 'annotations' / 'val.json'
    assert cfg.test_dataloader == cfg.val_dataloader
    assert cfg.val_evaluator.type == 'CocoMetric' and 'old' not in cfg.val_evaluator
    assert cfg.test_evaluator == cfg.val_evaluator
    assert cfg.optim_wrapper.type == 'OptimWrapper'
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(0.0002)
    assert set(cfg.optim_wrapper.paramwise_cfg.custom_keys) == {'backbone.dinov3', 'gsd_predictor'}
    assert [p.type for p in cfg.param_scheduler] == ['LinearLR', 'CosineRestartLR']
    assert cfg.param_scheduler[-1].end == 8
    assert cfg.param_scheduler[-1].periods == [7]
    assert cfg.param_scheduler[-1].restart_weights == [1]
    assert cfg.default_hooks.checkpoint.interval == 1
    assert cfg.default_hooks.checkpoint.save_best == 'coco/bbox_mAP'
    assert cfg.resume is False
    assert Path(cfg.load_from) == Path('work_dirs/finetune/init.pth')
    assert cfg.auto_scale_lr.enable is False
    for source, contents in original.items():
        assert source.read_bytes() == contents
    generated_bytes = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(SystemExit) as error:
        transfer.main(args)
    assert error.value.code == 2
    for path, contents in generated_bytes.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize('max_epochs', [2, 48])
@pytest.mark.parametrize('iters_per_epoch', [1, 3, 10])
def test_warmup_then_cosine_restores_each_group_base_lr(tmp_path, max_epochs, iters_per_epoch):
    """Even one warmup iteration must not leave cosine at 10% of base LR.

    Build the real template schedulers with MMEngine's epoch-to-iteration
    conversion. Check every training step and two differently scaled optimizer
    groups, rather than checking only config strings or scheduler attributes.
    """
    from mmengine.config import Config
    from mmengine.optim.scheduler import CosineRestartLR, LinearLR

    project = tmp_path / 'project'
    write_project(project)
    config_dir = project / 'configs' / 'levirdetnet'
    generated = config_dir / 'scheduler_test.py'
    generated.write_text(transfer.render_config(TEMPLATE, {
        '_base_': './levirdetnet-30class.py', 'max_epochs': max_epochs,
    }), encoding='utf-8')
    cfg = Config.fromfile(str(generated))
    params = [torch.nn.Parameter(torch.ones(())) for _ in range(2)]
    base_rates = (1e-4, 1e-5)
    optimizer = torch.optim.AdamW([
        {'params': [parameter], 'lr': rate}
        for parameter, rate in zip(params, base_rates)
    ], lr=base_rates[0])
    classes = {'LinearLR': LinearLR, 'CosineRestartLR': CosineRestartLR}
    schedulers = []
    for setting in cfg.param_scheduler:
        spec = dict(setting)
        scheduler_class = classes[spec.pop('type')]
        assert spec.pop('convert_to_iter_based') is True
        schedulers.append(scheduler_class.build_iter_from_epoch(
            optimizer, epoch_length=iters_per_epoch, **spec))
    assert schedulers[1].periods == [(max_epochs - 1) * iters_per_epoch]

    used_rates = []
    for _ in range(max_epochs * iters_per_epoch):
        # This is the rate used by training, before the after-iteration hook.
        used_rates.append([group['lr'] for group in optimizer.param_groups])
        optimizer.zero_grad()
        sum(parameter.square() for parameter in params).backward()
        optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

    for group_index, base_rate in enumerate(base_rates):
        actual = [row[group_index] for row in used_rates]
        assert actual[0] == pytest.approx(0.1 * base_rate, rel=1e-12, abs=1e-15)
        assert actual[iters_per_epoch] == pytest.approx(base_rate, rel=1e-12, abs=1e-15)
        assert all(left <= right + 1e-15 for left, right in zip(
            actual[:iters_per_epoch - 1], actual[1:iters_per_epoch]))
        assert all(left >= right - 1e-15 for left, right in zip(
            actual[iters_per_epoch:-1], actual[iters_per_epoch + 1:]))
        for iteration in range(iters_per_epoch, len(actual)):
            phase = (iteration - iters_per_epoch) / ((max_epochs - 1) * iters_per_epoch)
            expected = base_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * phase)))
            assert actual[iteration] == pytest.approx(expected, rel=1e-10, abs=1e-15)


def test_cli_subprocess_and_validation_failure_create_no_output(tmp_path, checkpoint):
    project = tmp_path / 'project'
    write_project(project)
    tools_dir = project / 'tools'
    tools_dir.mkdir()
    script = tools_dir / 'transfer_ckpt.py'
    script.write_text(SCRIPT.read_text(encoding='utf-8'), encoding='utf-8')
    dataset = project / 'data' / 'my_dataset'
    make_split(dataset, 'train')
    make_split(dataset, 'val')
    torch.save(checkpoint, project / 'epoch_117.pth')
    command = [sys.executable, str(script), '--new-class', 'plane,ship']
    failed = subprocess.run(command + ['--epochs', '1'], cwd=project, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert failed.returncode == 2
    assert 'epochs >= 2' in failed.stderr
    assert not (project / 'work_dirs' / 'finetune').exists()
    success = subprocess.run(command, cwd=project, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=60)
    assert success.returncode == 0, success.stdout + success.stderr
    assert 'Adapted 15 tensors' in success.stdout
    assert 'python tools/train.py' in success.stdout
    assert (project / 'work_dirs' / 'finetune' / 'finetune.py').is_file()
