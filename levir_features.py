"""Small inference API for the backbone trained in LEVIRDetNet-30.

Use the project's existing Docker or Conda environment. No network downloads
are performed. Only load checkpoints from a trusted source.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['load_model', 'load_image', 'vis', 'LEVIRFeatureExtractor']


def load_image(path):
    """Read one local image as an independent, EXIF-oriented RGB PIL image."""
    with Image.open(Path(path).expanduser()) as image:
        return ImageOps.exif_transpose(image).convert('RGB').copy()


def _image_size(value):
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, value)
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, int)
                   or v <= 0 or v % 32 for v in value)):
        raise ValueError('image_size must be a positive multiple of 32, or (height, width) with both multiples of 32.')
    return tuple(value)


class LEVIRFeatureExtractor(nn.Module):
    """Frozen 30-class backbone with image preprocessing and three outputs.

    ``model(img)`` returns final normalized DINO patch features [B,384,H/16,W/16].
    ``output='cls'`` returns global CLS features [B,384].
    ``output='pyramid'`` returns the complete DINOv3STAs adapter outputs at
    strides 8,16,32, each with 256 channels for the released checkpoint.

    Accepts an image path, a PIL image, uint8 RGB HWC numpy array, or a RGB
    CHW/BCHW tensor (uint8 0..255 or floating point 0..1, not normalized).
    Inputs resize to image_size without preserving aspect ratio, as in the
    release's 1024-square inference. Tensor resize uses PyTorch bilinear;
    PIL/array resize uses the release pipeline's OpenCV bilinear method.
    """

    def __init__(self, backbone, mean, std, image_size=1024, checkpoint_info=None):
        super().__init__()
        self.backbone = backbone
        self.image_size = _image_size(image_size)
        self.checkpoint_info = dict(checkpoint_info or {})
        self.register_buffer('_mean', torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_std', torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        if not torch.isfinite(self._mean).all() or not torch.isfinite(self._std).all() or not (self._std > 0).all():
            raise ValueError('Normalization requires finite means and positive finite standard deviations.')
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):
        if mode:
            raise ValueError('This convenience API is inference-only. Use the training configuration for fine-tuning.')
        return super().train(False)

    def prepare_image(self, img):
        """Resize, convert RGB to a batch, normalize, and move to the model device."""
        if isinstance(img, (str, Path)):
            img = load_image(img)
        if isinstance(img, Image.Image):
            img = np.asarray(ImageOps.exif_transpose(img).convert('RGB'))
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) < 1:
                raise ValueError('Numpy input must be nonempty uint8 RGB with shape [H,W,3]. Convert OpenCV BGR to RGB first.')
            import cv2
            height, width = self.image_size
            resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            batch = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).unsqueeze(0)
            batch = batch.to(device=self._mean.device, dtype=self._mean.dtype)
        elif isinstance(img, torch.Tensor):
            batch = img.unsqueeze(0) if img.ndim == 3 else img
            if batch.ndim != 4 or batch.shape[1] != 3 or min(batch.shape) < 1:
                raise ValueError('Tensor input must have nonempty RGB shape [3,H,W] or [B,3,H,W].')
            if batch.dtype == torch.uint8:
                batch = batch.to(device=self._mean.device, dtype=self._mean.dtype)
            elif batch.is_floating_point():
                if not torch.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
                    raise ValueError('Floating RGB tensors must be finite and in [0,1]; do not normalize them first.')
                batch = batch.to(device=self._mean.device, dtype=self._mean.dtype) * 255.0
            else:
                raise TypeError('Tensor input must be uint8 or floating point RGB in [0,1].')
            if tuple(batch.shape[-2:]) != self.image_size:
                batch = F.interpolate(batch, size=self.image_size, mode='bilinear', align_corners=False)
        else:
            raise TypeError('Use an image path, PIL RGB image, uint8 RGB numpy array, or RGB tensor.')
        return (batch - self._mean) / self._std

    @torch.inference_mode()
    def forward(self, img, output='patch'):
        if output not in ('patch', 'cls', 'pyramid'):
            raise ValueError("output must be 'patch', 'cls', or 'pyramid'.")
        batch = self.prepare_image(img)
        if output == 'pyramid':
            return self.backbone(batch)
        if output == 'cls':
            return self.backbone.dinov3.forward_features(batch)['x_norm_clstoken']
        return self.backbone.dinov3.get_intermediate_layers(batch, n=1, reshape=True)[0]


def load_model(weights='epoch_117.pth', *, config=None, device=None, image_size=1024):
    """Strictly load the 30-class checkpoint's backbone, without detector/GSD.

    ``weights`` is a trusted local complete detector checkpoint. ``state_dict``
    is the same deployed state used by image_demo.py; ema_state_dict is not
    substituted. The default config is resolved beside this module, independent
    of the working directory. Relative weights paths use the working directory.
    CUDA is selected when available; otherwise CPU is used (slower).
    """
    from mmengine.config import Config
    from mmdet.models.backbones.dinov3_adapter import DINOv3STAs

    _image_size(image_size)
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'30-class checkpoint not found: {path}. Download epoch_117.pth from the project model links.')
    config_path = Path(config).expanduser() if config else Path(__file__).resolve().parent / 'configs/levirdetnet/levirdetnet-30class.py'
    cfg = Config.fromfile(str(config_path))
    classes = tuple(cfg.get('classes', ()))
    if len(classes) != 30 or cfg.model.bbox_head.num_classes != 30:
        raise ValueError('This API expects the 30-class LEVIRDetNet configuration.')
    backbone_cfg = dict(cfg.model.backbone)
    if backbone_cfg.pop('type', None) != 'DINOv3STAs' or backbone_cfg.get('name') != 'dinov3_vits16plus':
        raise ValueError('Expected the released DINOv3STAs ViT-S+/16 backbone.')
    backbone_cfg.update(weights_path=None, init_cfg=None)
    normalization = cfg.model.data_preprocessor
    if not normalization.get('bgr_to_rgb', False):
        raise ValueError('Expected the released RGB preprocessing configuration.')
    # Memory mapping avoids eagerly allocating the full optimizer/EMA payload
    # of this multi-GB training checkpoint. No remote checkpoint is consulted.
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError('Expected a complete detector checkpoint with state_dict, such as epoch_117.pth.')
    state = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        key = key.removeprefix('module.')
        if key in state:
            raise ValueError('Checkpoint keys collide after removing the module. prefix.')
        state[key] = value
    head_weights = [value for key, value in state.items()
                    if key.startswith('bbox_head.cls_branches.') and key.endswith('.weight')]
    if not head_weights or any(not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 30 for value in head_weights):
        raise ValueError('Checkpoint classification layers are not the released 30-class detector. Use epoch_117.pth.')
    meta = checkpoint.get('meta') or {}
    saved_classes = (meta.get('dataset_meta') or {}).get('classes')
    if saved_classes is not None and tuple(saved_classes) != classes:
        raise ValueError('Checkpoint class names/order differ from the 30-class configuration.')
    backbone_state = OrderedDict((key[len('backbone.'):], value) for key, value in state.items() if key.startswith('backbone.'))
    # The original constructor logs "from scratch" when no external initializer
    # is supplied. Here all trained parameters are loaded strictly just below.
    with redirect_stdout(StringIO()):
        backbone = DINOv3STAs(**backbone_cfg)
    expected = backbone.state_dict()
    missing = sorted(set(expected) - set(backbone_state))
    extra = sorted(set(backbone_state) - set(expected))
    conflicts = sorted(key for key in set(expected) & set(backbone_state)
                       if not isinstance(backbone_state[key], torch.Tensor)
                       or backbone_state[key].shape != expected[key].shape)
    if missing or extra or conflicts:
        raise RuntimeError(f'Backbone checkpoint mismatch: missing={missing}, unexpected={extra}, shape_conflicts={conflicts}')
    backbone.load_state_dict(backbone_state, strict=True)
    info = dict(checkpoint=str(path), epoch=meta.get('epoch'), checkpoint_state='state_dict',
                class_count=30, backbone=backbone_cfg['name'], backbone_tensor_count=len(backbone_state))
    model = LEVIRFeatureExtractor(backbone, normalization['mean'], normalization['std'], image_size, info)
    del checkpoint, state, backbone_state, head_weights, expected
    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Check GPU access, or use device='cpu'.")
    return model.to(device).eval()


def vis(feature, save_path='work_dirs/features.png', *, size=None):
    """Save a deterministic three-component PCA RGB view and return a PIL image.

    Input: one [C,H,W] or [1,C,H,W] dense feature tensor. For a batch, call
    vis(feature[i]); for pyramid output, select a level first. CLS vectors are
    not spatial maps. PCA colors are computed per image, do not denote semantic
    classes, and are not directly comparable between independently fitted images.
    ``size=(height,width)`` optionally resizes the result. ``save_path=None``
    returns an image without writing a file (suitable for notebook display).
    """
    if not isinstance(feature, torch.Tensor):
        raise TypeError('vis expects a dense torch.Tensor; select one pyramid level if needed.')
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3 or min(feature.shape) < 1:
        raise ValueError('vis expects one [C,H,W] or [1,C,H,W] map, not a batch or CLS vector.')
    if not torch.isfinite(feature).all():
        raise ValueError('Features contain NaN or Inf.')
    channels, height, width = feature.shape
    values = feature.detach().float().cpu().permute(1, 2, 0).reshape(-1, channels).numpy().astype(np.float64)
    values -= values.mean(axis=0, keepdims=True)
    # Symmetric eigendecomposition of CxC covariance avoids a full pixel SVD.
    _, vectors = np.linalg.eigh(values.T @ values / max(1, len(values) - 1))
    components = vectors[:, -min(3, channels):][:, ::-1].copy()
    for index in range(components.shape[1]):
        pivot = np.abs(components[:, index]).argmax()
        if components[pivot, index] < 0:
            components[:, index] *= -1
    projected = values @ components
    low, high = np.percentile(projected, [2, 98], axis=0)
    rgb = np.zeros((len(values), 3), dtype=np.float64)
    active = high - low > 1e-12
    colors = np.zeros_like(projected)
    colors[:, active] = np.clip((projected[:, active] - low[active]) / (high[active] - low[active]), 0, 1)
    rgb[:, :colors.shape[1]] = colors
    image = Image.fromarray(np.rint(rgb.reshape(height, width, 3) * 255).astype(np.uint8))
    if size is not None:
        if not isinstance(size, (tuple, list)) or len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size):
            raise ValueError('size must be a positive (height, width) pair.')
        image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    if save_path is not None:
        target = Path(save_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
    return image
