"""CPU regression checks for the public LEVIR feature convenience API.

The expensive backbone is replaced only at the construction boundary; image
preprocessing, checkpoint selection, validation, and PCA run as implemented.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn


MODULE_PATH = Path(__file__).resolve().parents[2] / 'levir_features.py'
SPEC = importlib.util.spec_from_file_location('tested_levir_features', MODULE_PATH)
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def get_intermediate_layers(self, image, *, n, reshape):
        self.calls.append(('patch', n, reshape, torch.is_inference_mode_enabled()))
        return (image.new_full((image.shape[0], 384, image.shape[-2] // 16,
                                image.shape[-1] // 16), self.scale.item()),)

    def forward_features(self, image):
        self.calls.append(('cls', torch.is_inference_mode_enabled()))
        return {'x_norm_clstoken': image.new_full((image.shape[0], 384), 3)}


class FakeBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dinov3 = FakeDino()
        self.constructor_options = kwargs
        self.calls = []

    def forward(self, image):
        self.calls.append(('pyramid', torch.is_inference_mode_enabled()))
        return tuple(image.new_full((image.shape[0], 256, image.shape[-2] // stride,
                                    image.shape[-1] // stride), stride)
                     for stride in (8, 16, 32))


def make_model(size=(32, 64)):
    return api.LEVIRFeatureExtractor(FakeBackbone(), MEAN, STD, size)


def test_rgb_normalization_and_non_square_dimensions():
    model = make_model()
    image = np.zeros((9, 7, 3), dtype=np.uint8)
    image[...] = [255, 0, 127]
    original = image.copy()
    prepared = model.prepare_image(image)
    expected = torch.tensor([(255 - MEAN[0]) / STD[0], -MEAN[1] / STD[1],
                             (127 - MEAN[2]) / STD[2]])
    assert prepared.shape == (1, 3, 32, 64)
    torch.testing.assert_close(prepared[0, :, 0, 0], expected)
    torch.testing.assert_close(prepared[0, :, -1, -1], expected)
    np.testing.assert_array_equal(image, original)


def test_pil_path_and_numpy_have_identical_preprocessing(tmp_path):
    pixels = np.arange(13 * 11 * 3, dtype=np.uint16).reshape(13, 11, 3).astype(np.uint8)
    path = tmp_path / 'sample.png'
    image = Image.fromarray(pixels)
    image.save(path)
    model = make_model()
    torch.testing.assert_close(model.prepare_image(path), model.prepare_image(image), rtol=0, atol=0)
    torch.testing.assert_close(model.prepare_image(path), model.prepare_image(pixels), rtol=0, atol=0)


def test_load_image_detaches_file_and_converts_to_rgb(tmp_path):
    path = tmp_path / 'alpha.png'
    Image.new('RGBA', (3, 4), (10, 20, 30, 100)).save(path)
    image = api.load_image(path)
    path.unlink()
    assert image.mode == 'RGB'
    assert image.size == (3, 4)
    assert image.getpixel((0, 0)) == (10, 20, 30)


def test_float_tensor_and_uint8_tensor_agree_without_mutation():
    model = make_model()
    integer = torch.tensor([0, 127, 255], dtype=torch.uint8).view(3, 1, 1).expand(3, 32, 64).clone()
    floating = integer.float().div(255).requires_grad_(True)
    before = floating.detach().clone()
    prepared = model.prepare_image(floating)
    torch.testing.assert_close(prepared, model.prepare_image(integer), rtol=0, atol=1e-6)
    torch.testing.assert_close(floating, before, rtol=0, atol=0)
    assert model.prepare_image(integer.unsqueeze(0).repeat(2, 1, 1, 1)).shape == (2, 3, 32, 64)


@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan'), float('inf')])
def test_rejects_normalized_out_of_range_or_nonfinite_float(value):
    image = torch.zeros(3, 8, 8)
    image[0, 0, 0] = value
    with pytest.raises(ValueError, match=r'finite.*\[0,1\]'):
        make_model().prepare_image(image)


@pytest.mark.parametrize('image', [
    np.zeros((4, 5, 3), dtype=np.float32),
    np.zeros((4, 5), dtype=np.uint8),
    np.zeros((4, 5, 4), dtype=np.uint8),
    np.zeros((0, 5, 3), dtype=np.uint8),
    torch.zeros(2, 5, 5),
    torch.zeros(0, 3, 5, 5),
    torch.zeros(3, 0, 5),
])
def test_rejects_ambiguous_shapes_and_numpy_ranges(image):
    with pytest.raises(ValueError):
        make_model().prepare_image(image)


def test_rejects_integer_tensor_and_unknown_input():
    with pytest.raises(TypeError, match='uint8 or floating'):
        make_model().prepare_image(torch.zeros(3, 4, 4, dtype=torch.int16))
    with pytest.raises(TypeError, match='image path'):
        make_model().prepare_image([1, 2, 3])


@pytest.mark.parametrize('size', [0, 31, 33, True, (32, 33), (32, 0), (32,), (32.0, 64)])
def test_invalid_image_sizes_fail(size):
    with pytest.raises(ValueError, match='multiple'):
        make_model(size)


def test_all_feature_outputs_use_eval_frozen_inference():
    model = make_model()
    assert not model.training and not model.backbone.training and not model.backbone.dinov3.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    image = torch.full((2, 3, 16, 24), 0.5, requires_grad=True)
    patch_features = model(image)
    assert patch_features.shape == (2, 384, 2, 4)
    assert not patch_features.requires_grad
    assert model.backbone.dinov3.calls[-1] == ('patch', 1, True, True)
    assert model(image, output='cls').shape == (2, 384)
    assert model.backbone.dinov3.calls[-1] == ('cls', True)
    pyramid = model(image, output='pyramid')
    assert [tuple(item.shape) for item in pyramid] == [(2, 256, 4, 8), (2, 256, 2, 4), (2, 256, 1, 2)]
    assert model.backbone.calls[-1] == ('pyramid', True)
    with pytest.raises(ValueError, match='inference-only'):
        model.train()
    assert model.eval() is model
    with pytest.raises(ValueError, match='output must'):
        model(image, output='unknown')


def test_vis_deterministic_non_mutating_and_preserves_rng():
    values = torch.arange(7 * 9 * 5, dtype=torch.float32).reshape(7, 9, 5)
    feature = torch.sin(values * 0.173).requires_grad_(True)
    before = feature.detach().clone()
    rng_before = torch.random.get_rng_state().clone()
    first = api.vis(feature, save_path=None)
    second = api.vis(feature.unsqueeze(0), save_path=None)
    assert first.mode == 'RGB' and first.size == (5, 9)
    np.testing.assert_array_equal(first, second)
    assert np.ptp(np.asarray(first)) > 0
    torch.testing.assert_close(feature, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(384, 4, 4), (1, 3, 5), (2, 1, 1)])
def test_vis_constant_features_return_finite_black_image(shape):
    image = api.vis(torch.full(shape, 0.7), save_path=None)
    pixels = np.asarray(image)
    assert pixels.dtype == np.uint8
    assert pixels.shape == (shape[1], shape[2], 3)
    assert not pixels.any()


def test_vis_resizes_and_saves_to_nested_directory(tmp_path):
    path = tmp_path / 'a/b/features.png'
    result = api.vis(torch.arange(60).reshape(3, 4, 5).float(), path, size=(40, 50))
    assert result.size == (50, 40)
    with Image.open(path) as loaded:
        np.testing.assert_array_equal(loaded, result)


@pytest.mark.parametrize('feature', [torch.zeros(2, 384, 4, 4), torch.zeros(384),
                                    torch.zeros(0, 4, 4), torch.full((3, 4, 4), float('nan')),
                                    torch.full((3, 4, 4), float('inf'))])
def test_vis_rejects_batch_cls_empty_and_nonfinite(feature):
    with pytest.raises(ValueError):
        api.vis(feature, save_path=None)


@pytest.mark.parametrize('size', [(1, 0), (True, 1), (1.0, 4), 32])
def test_vis_rejects_invalid_output_size(size):
    with pytest.raises(ValueError, match='positive'):
        api.vis(torch.zeros(3, 4, 4), save_path=None, size=size)


@contextmanager
def fake_dependencies():
    config = {'classes': tuple(f'class_{i}' for i in range(30))}
    class Config(dict):
        model = SimpleNamespace(
            bbox_head=SimpleNamespace(num_classes=30),
            backbone=dict(type='DINOv3STAs', name='dinov3_vits16plus', weights_path='must-not-load.pth'),
            data_preprocessor=dict(mean=MEAN, std=STD, bgr_to_rgb=True),
        )

        @staticmethod
        def fromfile(path):
            return Config(config)

    config_module = ModuleType('mmengine.config')
    config_module.Config = Config
    adapter_module = ModuleType('mmdet.models.backbones.dinov3_adapter')
    adapter_module.DINOv3STAs = FakeBackbone
    with patch.dict(sys.modules, {'mmengine.config': config_module,
                                'mmdet.models.backbones.dinov3_adapter': adapter_module}):
        yield config


def small_checkpoint():
    return {'meta': {'dataset_meta': {'classes': tuple(f'class_{i}' for i in range(30))}, 'epoch': 117},
            'state_dict': {'backbone.dinov3.scale': torch.tensor(7.),
                           'bbox_head.cls_branches.0.weight': torch.zeros(30, 256)},
            'ema_state_dict': {'module.backbone.dinov3.scale': torch.tensor(99.)}}


def test_loader_strictly_uses_main_state_and_disables_initializers(tmp_path):
    path = tmp_path / 'checkpoint.pth'
    checkpoint = small_checkpoint()
    checkpoint['state_dict'] = {'module.' + key: value for key, value in checkpoint['state_dict'].items()}
    torch.save(checkpoint, path)
    with fake_dependencies():
        model = api.load_model(path, device='cpu', image_size=32)
    assert model.backbone.dinov3.scale.item() == 7
    assert model.backbone.constructor_options['weights_path'] is None
    assert model.backbone.constructor_options['init_cfg'] is None
    assert model.checkpoint_info['checkpoint_state'] == 'state_dict'
    assert model.checkpoint_info['epoch'] == 117
    assert model.checkpoint_info['backbone_tensor_count'] == 1


@pytest.mark.parametrize('kind', ['159class', 'class_order', 'missing_backbone', 'shape', 'prefix_collision'])
def test_loader_rejects_mismatched_checkpoints(tmp_path, kind):
    checkpoint = small_checkpoint()
    if kind == '159class':
        checkpoint['state_dict']['bbox_head.cls_branches.0.weight'] = torch.zeros(159, 256)
    elif kind == 'class_order':
        checkpoint['meta']['dataset_meta']['classes'] = tuple(reversed(checkpoint['meta']['dataset_meta']['classes']))
    elif kind == 'missing_backbone':
        checkpoint['state_dict'].pop('backbone.dinov3.scale')
    elif kind == 'shape':
        checkpoint['state_dict']['backbone.dinov3.scale'] = torch.ones(3)
    else:
        checkpoint['state_dict']['module.backbone.dinov3.scale'] = torch.tensor(8.)
    path = tmp_path / f'{kind}.pth'
    torch.save(checkpoint, path)
    with fake_dependencies(), pytest.raises((ValueError, RuntimeError)):
        api.load_model(path, device='cpu', image_size=32)
