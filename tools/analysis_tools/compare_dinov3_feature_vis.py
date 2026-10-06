# Copyright (c) OpenMMLab. All rights reserved.
"""Compare detector-backbone features with plain DINOv3 PCA features.

Examples:
    # Compare the epoch_117 detector backbone against a local HF DINOv3 model.
    python tools/analysis_tools/compare_dinov3_feature_vis.py \
        /path/to/images \
        --dinov3-backend transformers \
        --dinov3-model /path/to/dinov3-hf-dir

    # Compare against the DINOv3 implementation and weights configured in mmdet.
    python tools/analysis_tools/compare_dinov3_feature_vis.py \
        /path/to/images --dinov3-backend mmdet

    # DINOv3 input = 2048, ours input = 4096 by default.
    python tools/analysis_tools/compare_dinov3_feature_vis.py \
        /path/to/images --size 2048

    # Manually set ours input size.
    python tools/analysis_tools/compare_dinov3_feature_vis.py \
        /path/to/images --size 2048 --own-size 4096
"""

import argparse
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize detector backbone features side-by-side with '
        'plain DINOv3 features.')
    parser.add_argument('image_dir', help='Directory containing input images.')
    parser.add_argument(
        '--config',
        default='configs/deimv2/deimv2_dinov3_vitb_8xb4-48e_tmp_1024_gsd_det.py',
        help='Detector config used to build the trained backbone.')
    parser.add_argument(
        '--checkpoint',
        default='epoch_117.pth',
        help='Detector checkpoint. Only backbone weights are loaded.')
    parser.add_argument(
        '--checkpoint-source',
        choices=('state_dict', 'ema_state_dict'),
        default='state_dict',
        help='Which state dict to read from the detector checkpoint.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/feature_vis',
        help='Directory to save visualization PNG files.')
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device used for inference.')
    parser.add_argument(
        '--size',
        type=int,
        default=2048,
        help='DINOv3 input resize size. Images are resized to size x size '
        'for DINOv3. The detector backbone input defaults to 2 * size.')
    parser.add_argument(
        '--own-size',
        type=int,
        default=None,
        help='Detector-backbone input resize size. If not set, defaults to '
        '2 * --size, so ours output can align with DINOv3 output when ours '
        'has stride 32 and DINOv3 has patch size 16.')
    parser.add_argument(
        '--own-level',
        type=int,
        default=-1,
        help='Which detector-backbone feature level to visualize. '
        'Default -1 uses the last output level.')
    parser.add_argument(
        '--dinov3-backend',
        choices=('auto', 'transformers', 'mmdet'),
        default='auto',
        help='Reference DINOv3 backend. auto uses transformers when '
        '--dinov3-model is set, otherwise mmdet.')
    parser.add_argument(
        '--dinov3-model',
        default=None,
        help='Local HuggingFace/Transformers DINOv3 model directory or name.')
    parser.add_argument(
        '--dinov3-register-tokens',
        type=int,
        default=4,
        help='Register-token count for Transformers DINOv3 outputs.')
    parser.add_argument(
        '--transformers-local-files-only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use Transformers in offline/local-only mode. Defaults to true.')
    parser.add_argument(
        '--trust-remote-code',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Allow local custom Transformers model code. Defaults to true.')
    parser.add_argument(
        '--dinov3-name',
        default=None,
        help='DINOv3 model name for the mmdet backend. Defaults to the config '
        'backbone name.')
    parser.add_argument(
        '--dinov3-weights',
        default=None,
        help='DINOv3 .pth weights for the mmdet backend. Defaults to the '
        'config backbone weights_path.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search image_dir.')
    parser.add_argument(
        '--save-separate',
        action='store_true',
        help='Also save individual PCA maps for ours and DINOv3.')
    return parser.parse_args()


def torch_load(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def find_images(image_dir: str, recursive: bool,
                max_images: Optional[int]) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f'Image directory not found: {root}')

    iterator: Iterable[Path] = root.rglob('*') if recursive else root.iterdir()
    images = sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    if max_images is not None:
        images = images[:max_images]

    if not images:
        raise FileNotFoundError(f'No images found in {root}')

    return images


def load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def preprocess_for_mmdet(image: Image.Image, cfg: Config, size: int,
                         device: torch.device) -> torch.Tensor:
    image = image.resize((size, size), Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32)

    data_preprocessor = cfg.model.get('data_preprocessor', {})
    bgr_to_rgb = bool(data_preprocessor.get('bgr_to_rgb', True))
    if not bgr_to_rgb:
        arr = arr[..., ::-1].copy()

    mean = np.asarray(
        data_preprocessor.get('mean', [123.675, 116.28, 103.53]),
        dtype=np.float32)
    std = np.asarray(
        data_preprocessor.get('std', [58.395, 57.12, 57.375]),
        dtype=np.float32)

    if float(mean.max()) <= 1.5 and float(std.max()) <= 1.5:
        arr = arr / 255.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.from_numpy(mean).view(1, 3, 1, 1)
    std_t = torch.from_numpy(std).view(1, 3, 1, 1)
    tensor = (tensor - mean_t) / std_t
    return tensor.to(device)


def pca_to_rgb(feature: torch.Tensor) -> np.ndarray:
    """Convert a CxHxW or 1xCxHxW feature map to an RGB PCA image."""
    if feature.ndim == 4:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f'Expected CxHxW feature, got {tuple(feature.shape)}')

    c, h, w = feature.shape
    flat = feature.detach().float().permute(1, 2, 0).reshape(h * w, c).cpu()
    flat = flat - flat.mean(dim=0, keepdim=True)
    flat = flat / (flat.std(dim=0, keepdim=True) + 1e-6)

    try:
        _, _, v = torch.pca_lowrank(flat, q=3, center=False)
        projected = flat @ v[:, :3]
    except Exception:
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        projected = flat @ vh[:3].t()

    projected = projected.reshape(h, w, 3).numpy()
    min_val = projected.min(axis=(0, 1), keepdims=True)
    max_val = projected.max(axis=(0, 1), keepdims=True)
    projected = (projected - min_val) / (max_val - min_val + 1e-8)
    return np.clip(projected, 0.0, 1.0)


def strip_module_prefix(key: str) -> str:
    return key[len('module.'):] if key.startswith('module.') else key


def get_checkpoint_state(checkpoint_path: str, source: str) -> OrderedDict:
    checkpoint = torch_load(checkpoint_path)

    if isinstance(checkpoint, dict) and source in checkpoint:
        state_dict = checkpoint[source]
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f'Unsupported checkpoint type: {type(checkpoint)}')

    output = OrderedDict()
    for key, value in state_dict.items():
        if key == 'steps':
            continue
        output[strip_module_prefix(key)] = value
    return output


def load_backbone_weights(backbone: torch.nn.Module, checkpoint_path: str,
                          source: str) -> Tuple[int, int]:
    state_dict = get_checkpoint_state(checkpoint_path, source)
    target_state = backbone.state_dict()
    loadable = OrderedDict()
    skipped = 0

    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            key = key[len('backbone.'):]

        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            loadable[key] = value
        elif key in target_state:
            skipped += 1

    missing, unexpected = backbone.load_state_dict(loadable, strict=False)

    print(
        f'Loaded backbone tensors: {len(loadable)}; '
        f'shape-skipped: {skipped}; missing: {len(missing)}; '
        f'unexpected: {len(unexpected)}')

    return len(loadable), skipped


def build_own_backbone(cfg: Config, checkpoint_path: str, source: str,
                       device: torch.device) -> torch.nn.Module:
    register_all_modules(init_default_scope=True)
    backbone = MODELS.build(cfg.model.backbone)
    load_backbone_weights(backbone, checkpoint_path, source)
    backbone.to(device).eval()
    return backbone


def build_mmdet_dinov3(cfg: Config, args: argparse.Namespace,
                       device: torch.device) -> torch.nn.Module:
    from mmdet.models.backbones.dinov3 import DinoVisionTransformer

    backbone_cfg = cfg.model.backbone
    name = args.dinov3_name or backbone_cfg.get('name', 'dinov3_vits16plus')
    weights = args.dinov3_weights or backbone_cfg.get('weights_path', None)

    model = DinoVisionTransformer(name=name)

    if weights:
        weights_path = Path(weights)
        if weights_path.exists():
            model.load_state_dict(torch_load(str(weights_path)), strict=False)
            print(f'Loaded mmdet DINOv3 weights: {weights_path}')
        else:
            print(f'Warning: DINOv3 weights not found: {weights_path}')

    model.to(device).eval()
    return model


def infer_own_feature(backbone: torch.nn.Module, tensor: torch.Tensor,
                      own_level: int) -> torch.Tensor:
    with torch.inference_mode():
        feats = backbone(tensor)

    if not isinstance(feats, (tuple, list)):
        feats = (feats,)

    level = own_level if own_level >= 0 else len(feats) + own_level

    if level < 0 or level >= len(feats):
        raise IndexError(
            f'own_level={own_level} is out of range for {len(feats)} levels.')

    return feats[level]


def infer_mmdet_dinov3_feature(model: torch.nn.Module,
                               tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        if hasattr(model, 'get_intermediate_layers'):
            return model.get_intermediate_layers(
                tensor, n=1, reshape=True, norm=True)[-1]

        outputs = model(tensor)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[-1]
        return outputs


def manual_transformers_inputs(image: Image.Image, size: int,
                               device: torch.device) -> dict:
    image = image.resize((size, size), Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    return {'pixel_values': ((tensor - mean) / std).to(device)}


def get_last_hidden_state(outputs):
    hidden = getattr(outputs, 'last_hidden_state', None)
    if hidden is not None:
        return hidden

    if isinstance(outputs, dict):
        for key in ('last_hidden_state', 'x_norm_patchtokens',
                    'patch_tokens', 'features'):
            if key in outputs:
                return outputs[key]

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]

    raise ValueError(
        'Cannot find last_hidden_state in Transformers DINOv3 outputs.')


def infer_transformers_dinov3_feature(model, processor, image: Image.Image,
                                      size: int, device: torch.device,
                                      register_tokens: int) -> torch.Tensor:
    if processor is None:
        inputs = manual_transformers_inputs(image, size, device)
    else:
        inputs = processor(
            images=image,
            return_tensors='pt',
            do_resize=True,
            size={
                'height': size,
                'width': size
            },
            do_center_crop=False,
        ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    hidden = get_last_hidden_state(outputs)

    if hidden.ndim == 4:
        return hidden

    candidates = [1 + register_tokens, 5, 1, 0]
    for start in dict.fromkeys(candidates):
        n = hidden.shape[1] - start
        side = int(math.sqrt(n))
        if n > 0 and side * side == n:
            tokens = hidden[:, start:, :]
            return tokens.reshape(
                hidden.shape[0],
                side,
                side,
                hidden.shape[2]
            ).permute(0, 3, 1, 2).contiguous()

    raise ValueError(
        'Cannot infer a square patch grid from Transformers DINOv3 output. '
        f'last_hidden_state shape={tuple(hidden.shape)}')


def build_transformers_dinov3(args: argparse.Namespace,
                              device: torch.device):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError(
            'Please install transformers or use --dinov3-backend mmdet.'
        ) from exc

    processor = None
    common_kwargs = dict(
        local_files_only=args.transformers_local_files_only,
        trust_remote_code=args.trust_remote_code)

    try:
        processor = AutoImageProcessor.from_pretrained(
            args.dinov3_model, **common_kwargs)
    except Exception as exc:
        print(
            'Warning: AutoImageProcessor could not load this DINOv3 '
            f'directory ({exc}). Falling back to manual ImageNet '
            'resize/normalize preprocessing.')

    try:
        model = AutoModel.from_pretrained(
            args.dinov3_model, **common_kwargs).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            'AutoModel could not load the DINOv3 directory in offline mode. '
            'If this is not a standard Transformers model, use '
            '--dinov3-backend mmdet with a .pth DINOv3 weight, or provide a '
            'Transformers directory with config/model files that your '
            f'installed transformers version can load. Original error: {exc}'
        ) from exc

    return model, processor


def image_to_vis_array(image: Image.Image,
                       target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert PIL image to RGB float array for visualization.

    target_hw is (height, width). When provided, the image is resized to the
    same spatial size as the feature PCA maps, so the three panels are aligned.
    """
    if target_hw is not None:
        target_h, target_w = target_hw
        image = image.resize((target_w, target_h), Image.BICUBIC)

    arr = np.asarray(image).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def save_comparison(original_image: Image.Image,
                    own_vis: np.ndarray,
                    dino_vis: np.ndarray,
                    out_path: Path,
                    title_original: str = 'image',
                    title_ours: str = 'ours',
                    title_dino: str = 'DINOv3') -> None:
    """Save one-row visualization: original image + ours PCA + DINOv3 PCA."""
    target_hw = dino_vis.shape[:2]
    image_vis = image_to_vis_array(original_image, target_hw=target_hw)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)

    axes[0].imshow(image_vis)
    axes[0].set_title(title_original)
    axes[0].axis('off')

    axes[1].imshow(own_vis)
    axes[1].set_title(title_ours)
    axes[1].axis('off')

    axes[2].imshow(dino_vis)
    axes[2].set_title(title_dino)
    axes[2].axis('off')

    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    cfg = Config.fromfile(args.config)

    dino_size = args.size
    own_size = args.own_size if args.own_size is not None else args.size * 2

    images = find_images(args.image_dir, args.recursive, args.max_images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'DINOv3 input size: {dino_size} x {dino_size}')
    print(f'Ours input size:   {own_size} x {own_size}')

    own_backbone = build_own_backbone(
        cfg, args.checkpoint, args.checkpoint_source, device)

    backend = args.dinov3_backend
    if backend == 'auto':
        backend = 'transformers' if args.dinov3_model else 'mmdet'

    if backend == 'transformers':
        if not args.dinov3_model:
            raise ValueError(
                '--dinov3-model is required for transformers backend.')
        dino_model, processor = build_transformers_dinov3(args, device)
        dino_title = 'DINOv3(transformers)'
    else:
        dino_model = build_mmdet_dinov3(cfg, args, device)
        processor = None
        dino_title = 'DINOv3(mmdet)'

    print(f'Processing {len(images)} images...')

    for image_path in images:
        image = load_rgb(image_path)

        # Ours uses a larger input by default.
        # Example: DINOv3 2048 -> 128x128 tokens with patch 16.
        #          Ours   4096 -> 128x128 feature map with stride 32.
        own_input = preprocess_for_mmdet(image, cfg, own_size, device)
        own_feat = infer_own_feature(own_backbone, own_input, args.own_level)

        if backend == 'transformers':
            dino_feat = infer_transformers_dinov3_feature(
                dino_model,
                processor,
                image,
                dino_size,
                device,
                args.dinov3_register_tokens)
        else:
            dino_input = preprocess_for_mmdet(image, cfg, dino_size, device)
            dino_feat = infer_mmdet_dinov3_feature(dino_model, dino_input)

        own_vis = pca_to_rgb(own_feat)
        dino_vis = pca_to_rgb(dino_feat)

        if own_vis.shape[:2] != dino_vis.shape[:2]:
            print(
                f'Warning: PCA map spatial shapes are still different for '
                f'{image_path.name}: ours={own_vis.shape[:2]}, '
                f'dinov3={dino_vis.shape[:2]}. '
                f'Check --own-size, --size, own stride, and DINOv3 patch size.')

        stem = image_path.stem
        save_path = out_dir / f'{stem}_image_ours_vs_dinov3.png'

        save_comparison(
            image,
            own_vis,
            dino_vis,
            save_path,
            title_original='image',
            title_ours='ours',
            title_dino=dino_title)

        if args.save_separate:
            plt.imsave(out_dir / f'{stem}_ours_pca.png', own_vis)
            plt.imsave(out_dir / f'{stem}_dinov3_pca.png', dino_vis)

        print(
            f'Saved {save_path} | '
            f'own_input=({own_size}, {own_size}) '
            f'dino_input=({dino_size}, {dino_size}) | '
            f'ours={tuple(own_feat.shape)} '
            f'dinov3={tuple(dino_feat.shape)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())