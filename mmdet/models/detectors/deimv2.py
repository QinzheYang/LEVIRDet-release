# Copyright (c) OpenMMLab. All rights reserved.
import math
import json
from pathlib import Path
from typing import Dict, Optional

from mmengine.logging import MMLogger
import torch
import torch.distributed as dist
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import ConfigType
from ..layers import DEIMV2TransformerDecoder, RTDETRHybridEncoder
from ..layers.transformer.dfine_layers import (
    LQE, Gate, MultiNumPointsMultiScaleDeformableAttention)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .deim import DEIMDFINE
from .rtdetr_ins import RTDETRInsMixup, RTDETRInsPlusMixup

##  new add gsd_pred  ##

def _rgb_to_gray(x: Tensor) -> Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class _RadialFFTSpectrumEncoder(nn.Module):

    def __init__(self, radial_bins: int = 64, out_dim: int = 256):
        super().__init__()
        self.radial_bins = radial_bins
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self._cache_hw = None
        self._radial_bin_idx: Optional[Tensor] = None
        self._radial_bin_cnt: Optional[Tensor] = None

    @torch.no_grad()
    def _build_radial_bins(self, h: int, w: int, device):
        wx = w // 2 + 1
        fy = torch.fft.fftfreq(h, d=1.0, device=device)
        fx = torch.fft.rfftfreq(w, d=1.0, device=device)
        yy = fy[:, None].expand(h, wx)
        xx = fx[None, :].expand(h, wx)
        rr = torch.sqrt(xx * xx + yy * yy)
        rmax = rr.max().clamp_min(1e-12)
        rr01 = rr / rmax
        r = self.radial_bins
        bin_idx = torch.clamp((rr01 * (r - 1)).long(), 0, r - 1)
        flat = bin_idx.reshape(-1)
        cnt = torch.bincount(flat, minlength=r).float().clamp_min(1.0)
        self._cache_hw = (h, w)
        self._radial_bin_idx = flat
        self._radial_bin_cnt = cnt

    def forward(self, x_rgb: Tensor) -> Tensor:
        n, _, h, w = x_rgb.shape
        device = x_rgb.device
        if self._cache_hw != (h, w) or self._radial_bin_idx is None:
            self._build_radial_bins(h, w, device)

        g = _rgb_to_gray(x_rgb).squeeze(1)
        freq = torch.fft.rfft2(g, norm='ortho')
        mag = torch.log1p(torch.abs(freq))

        flat_mag = mag.reshape(n, -1)
        bin_idx = self._radial_bin_idx
        r = self.radial_bins

        spec_sum = torch.zeros((n, r), device=device, dtype=flat_mag.dtype)
        spec_sum.scatter_add_(1, bin_idx[None, :].expand(n, -1), flat_mag)
        spec = spec_sum / self._radial_bin_cnt[None, :].to(spec_sum.dtype)

        z = self.conv(spec.unsqueeze(1))
        z = self.proj(z)
        return z


class _EdgeStatEncoder(nn.Module):

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(
                1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(
                1, 1, 3, 3)
        lap = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(
                1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)
        self.register_buffer('lap', lap, persistent=False)

    def forward(self, x_rgb: Tensor) -> Tensor:
        g = _rgb_to_gray(x_rgb)
        gx = F.conv2d(g, self.sobel_x, padding=1)
        gy = F.conv2d(g, self.sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        lap = F.conv2d(g, self.lap, padding=1)

        mag_f = mag.flatten(1)
        lap_f = lap.flatten(1)

        mag_mean = mag_f.mean(dim=1)
        mag_std = mag_f.std(dim=1)
        k90 = max(1, int(0.90 * mag_f.shape[1]))
        k99 = max(1, int(0.99 * mag_f.shape[1]))
        mag_p90 = mag_f.kthvalue(k90, dim=1).values
        mag_p99 = mag_f.kthvalue(k99, dim=1).values
        lap_var = lap_f.var(dim=1)
        lap_abs_mean = lap_f.abs().mean(dim=1)

        feat = torch.stack(
            [mag_mean, mag_std, mag_p90, mag_p99, lap_var, lap_abs_mean], dim=1)
        return self.mlp(feat)


class _GSDRegressorCNNFFTEdge(nn.Module):
    """Patch-based GSD regressor used for online guidance."""

    def __init__(self,
                 backbone_name: str = 'resnet50',
                 embed_dim: int = 256,
                 fft_radial_bins: int = 64,
                 agg: str = 'attn',
                 fusion: str = 'gated'):
        super().__init__()
        assert agg in ('mean', 'max', 'attn')
        assert fusion in ('concat_mlp', 'gated')
        self.embed_dim = embed_dim
        self.agg = agg
        self.fusion = fusion

        if backbone_name == 'resnet18':
            net = torchvision.models.resnet18(weights=None)
            feat_dim = 512
        elif backbone_name == 'resnet34':
            net = torchvision.models.resnet34(weights=None)
            feat_dim = 512
        elif backbone_name == 'resnet50':
            net = torchvision.models.resnet50(weights=None)
            feat_dim = 2048
        else:
            raise ValueError(backbone_name)

        self.cnn = nn.Sequential(*list(net.children())[:-1])
        self.cnn_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.fft_enc = _RadialFFTSpectrumEncoder(
            radial_bins=fft_radial_bins, out_dim=embed_dim)
        self.edge_enc = _EdgeStatEncoder(out_dim=embed_dim)

        if fusion == 'concat_mlp':
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.gate = None
        else:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 3, 128),
                nn.GELU(),
                nn.Linear(128, 3),
            )
            self.fuse = None

        if agg == 'attn':
            self.attn_score = nn.Linear(embed_dim, 1)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, 3, H, W), ImageNet normalized
        b, k, c, h, w = x.shape
        x = x.view(b * k, c, h, w)
        f = self.cnn(x).flatten(1)
        e_cnn = self.cnn_proj(f)
        e_fft = self.fft_enc(x)
        e_edge = self.edge_enc(x)

        cat = torch.cat([e_cnn, e_fft, e_edge], dim=1)
        if self.fusion == 'concat_mlp':
            e = self.fuse(cat)
        else:
            g = torch.softmax(self.gate(cat), dim=1)
            e = g[:, 0:1] * e_cnn + g[:, 1:2] * e_fft + g[:, 2:3] * e_edge

        e = e.view(b, k, self.embed_dim)
        if self.agg == 'mean':
            g_img = e.mean(dim=1)
        elif self.agg == 'max':
            g_img = e.max(dim=1).values
        else:
            a = torch.softmax(self.attn_score(e).squeeze(-1), dim=1)
            g_img = (e * a.unsqueeze(-1)).sum(dim=1)
        return self.head(g_img)



@MODELS.register_module()
class DEIMV2(DEIMDFINE):
    """Implementation of `Real-Time Object Detection Meets DINOv3.

    <https://arxiv.org/abs/2509.20787>`_
    """

    def __init__(self,
                 *args,
                 train_cfg: ConfigType = dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         match_costs=[
                             dict(type='ClassificationCost', weight=1.),
                             dict(
                                 type='BBoxL1Cost',
                                 weight=5.0,
                                 box_format='xywh'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ]),
                     switch_assigner=dict(
                         switch_epoch=45,
                         assigner=dict(
                             type='HungarianAssigner',
                             match_costs=[
                                 dict(
                                     type='DEIMV2LossCost',
                                     iou_order_alpha=4.0,
                                     weight=1.)
                             ]))),
                 **kwargs) -> None:
        super().__init__(*args, train_cfg=train_cfg, **kwargs)

        if train_cfg and 'switch_assigner' in train_cfg:
            switch_assigner_cfg = train_cfg['switch_assigner']
            self.switch_assigner_epoch = switch_assigner_cfg['switch_epoch']
            self.switch_assigner = TASK_UTILS.build(
                switch_assigner_cfg['assigner'])
            self.assigner_has_switched = False

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = DEIMV2TransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Identity()
        self.memory_trans_norm = nn.Identity()

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m,
                          (Gate, MultiNumPointsMultiScaleDeformableAttention,
                           MultiScaleDeformableAttention)):
                m.init_weights()
            elif isinstance(m, LQE):
                for layer in m.reg_conf.layers[:-1]:
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                m.init_weights()

    def _switch_assigner(self) -> None:
        """Switch to the new assigner during training."""
        if hasattr(self, 'switch_assigner_epoch'):
            if not hasattr(self, 'epoch'):
                raise AttributeError(
                    'Please set the current epoch number to the model '
                    'before calling loss function. Use `SetEpochInfoHook`')
            epoch_to_be_switched = self.epoch >= self.switch_assigner_epoch
            if epoch_to_be_switched and not self.assigner_has_switched:
                logger = MMLogger.get_current_instance()
                logger.info('Switching to the new assigner at epoch '
                            f'{self.epoch}.')
                assert hasattr(self.bbox_head, 'assigner'), \
                    'The bbox_head must have an assigner to be switched.'
                self.bbox_head.assigner = self.switch_assigner
                self.assigner_has_switched = True

    def set_epoch(self, value: int) -> None:
        """Set current epoch number and switch assigner if needed.

        Note:
            This function is called by `SetEpochInfoHook` during training.
        """
        self.epoch = value
        self._switch_assigner()


@MODELS.register_module()
class DEIMV2Ins(RTDETRInsMixup, DEIMV2):
    """DEIMV2 for Instance."""


@MODELS.register_module()
class DEIMV2InsPlus(RTDETRInsPlusMixup, DEIMV2Ins):
    """DEIMV2InsPlus with C2"""


@MODELS.register_module()
class DEIMV2GSDGuided(DEIMV2):
    """DEIMV2 with GSD-guided query modulation.

    This module consumes pre-computed GSD predictions (json) and injects a
    learnable GSD embedding into decoder queries.

    Args:
        gsd_cfg (dict, optional): Configuration for GSD guidance.
            - enabled (bool): Enable/disable guidance. Default: True.
            - gsd_json (str): Path to json file. Each record supports
              {"img": "...", "gsd": float} or {"img_path": "...", "gsd": float}.
            - default_gsd (float): Fallback value when image has no entry.
            - use_log_gsd (bool): Use log(gsd) as input scalar.
            - modulate (str): "add" or "film".
            - scale (float): Residual scale factor.
            - hidden_dim (int): Hidden dimension for gsd mlp.
    """

    def __init__(self, *args, gsd_cfg: Optional[Dict] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        gsd_cfg = gsd_cfg or {}
        self.gsd_enabled = gsd_cfg.get('enabled', True)
        self.gsd_source = gsd_cfg.get('source', 'lookup')
        self.default_gsd = float(gsd_cfg.get('default_gsd', 1.0))
        self.use_log_gsd = bool(gsd_cfg.get('use_log_gsd', True))
        self.gsd_modulate = gsd_cfg.get('modulate', 'add')
        self.gsd_scale = float(gsd_cfg.get('scale', 1.0))
        hidden_dim = int(gsd_cfg.get('hidden_dim', 128))

        self.gsd_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )
        if self.gsd_modulate == 'film':
            self.gsd_film = nn.Linear(self.embed_dims, self.embed_dims * 2)

        self.gsd_table = {}
        self._cached_batch_gsd: Optional[Tensor] = None

        # online predictor settings
        self.gsd_patch_size = int(gsd_cfg.get('patch_size', 224))
        self.gsd_num_patches = int(gsd_cfg.get('num_patches', 5))
        self.gsd_small_img_threshold = int(gsd_cfg.get('small_img_threshold', 512))
        self.gsd_online_trainable = bool(gsd_cfg.get('online_trainable', False))
        self.gsd_predictor = None

        query_group_cfg = gsd_cfg.get('query_group_cfg', {})
        self.query_group_enabled = bool(query_group_cfg.get('enabled', False))
        self.query_group_bins = list(query_group_cfg.get('bins', [200, 400, 600, 800]))
        if len(self.query_group_bins) == 0:
            self.query_group_bins = [self.num_queries]
        self.query_group_bins = sorted(int(v) for v in self.query_group_bins)
        q_hidden_dim = int(query_group_cfg.get('hidden_dim', 128))
        self.query_group_scale_alpha = float(query_group_cfg.get('scale_alpha', 0.5))
        self.query_group_density_alpha = float(
            query_group_cfg.get('density_alpha', 0.5))
        self.query_group_loss_weight = float(
            query_group_cfg.get('loss_weight', 0.1))
        self.query_group_query_per_gt = float(
            query_group_cfg.get('query_per_gt', 4.0))
        self.query_group_small_obj_weight = float(
            query_group_cfg.get('small_obj_weight', 15.0))
        self.query_group_density_weight = float(
            query_group_cfg.get('density_weight', 0.2))
        self.query_group_scale_bonus_weight = float(
            query_group_cfg.get('scale_bonus_weight', 8.0))
        self.query_group_scale_ref = float(
            query_group_cfg.get('scale_ref', 0.08))
        self.query_group_gsd_weight = float(
            query_group_cfg.get('gsd_weight', 0.0))
        self.query_group_gsd_ref = float(
            query_group_cfg.get('gsd_ref', self.default_gsd))
        self.query_group_gsd_clip = float(
            query_group_cfg.get('gsd_clip', 2.0))
        self.query_group_train_enabled = bool(
            query_group_cfg.get('train_enabled', False))
        self.query_group_train_begin_epoch = int(
            query_group_cfg.get('train_begin_epoch', 0))
        train_epoch_ranges = query_group_cfg.get('train_epoch_ranges', None)
        if train_epoch_ranges is None:
            self.query_group_train_epoch_ranges = None
        else:
            self.query_group_train_epoch_ranges = []
            for epoch_range in train_epoch_ranges:
                if len(epoch_range) != 2:
                    raise ValueError(
                        'query_group_cfg.train_epoch_ranges expects '
                        '(begin, end) pairs.')
                begin, end = epoch_range
                begin = int(begin)
                end = None if end is None else int(end)
                self.query_group_train_epoch_ranges.append((begin, end))
        self.query_group_train_use_gsd = bool(
            query_group_cfg.get('train_use_gsd', True))
        self.query_group_train_sync_across_ranks = bool(
            query_group_cfg.get('train_sync_across_ranks', True))
        self.query_group_head = nn.Sequential(
            nn.Linear(self.embed_dims * 2 + 2, q_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(q_hidden_dim, len(self.query_group_bins)))
        self._last_group_queries: Optional[Tensor] = None
        self._last_query_group_logits: Optional[Tensor] = None

        if self.gsd_source == 'online':
            self._init_online_predictor(gsd_cfg)
        else:
            self._load_gsd_table()

    def _init_online_predictor(self, gsd_cfg: Dict) -> None:
        self.gsd_predictor = _GSDRegressorCNNFFTEdge(
            backbone_name=gsd_cfg.get('backbone', 'resnet50'),
            embed_dim=int(gsd_cfg.get('embed_dim', 256)),
            fft_radial_bins=int(gsd_cfg.get('fft_bins', 64)),
            agg=gsd_cfg.get('agg', 'attn'),
            fusion=gsd_cfg.get('fusion', 'gated'))

        ckpt = gsd_cfg.get('ckpt', '')
        if ckpt:
            logger = MMLogger.get_current_instance()
            ckpt_path = Path(ckpt)
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location='cpu')
                if isinstance(state, dict) and 'model' in state:
                    state = state['model']
                if any(k.startswith('module.') for k in state.keys()):
                    state = {k.replace('module.', '', 1): v for k, v in state.items()}
                missing, unexpected = self.gsd_predictor.load_state_dict(
                    state, strict=False)
                logger.info(
                    f'Loaded online GSD predictor ckpt: {ckpt_path}, '
                    f'missing={len(missing)}, unexpected={len(unexpected)}')
            else:
                logger.warning(f'Online GSD ckpt not found: {ckpt_path}')

        if not self.gsd_online_trainable:
            self.gsd_predictor.eval()
            for p in self.gsd_predictor.parameters():
                p.requires_grad = False
        else:
            self.gsd_predictor.train()

    def _load_gsd_table(self) -> None:
        if not self.gsd_enabled or not self.gsd_json:
            return

        logger = MMLogger.get_current_instance()
        json_path = Path(self.gsd_json)
        if not json_path.exists():
            logger.warning(
                f'GSD json not found: {json_path}. Falling back to default_gsd='
                f'{self.default_gsd}.')
            return

        data = json.loads(json_path.read_text(encoding='utf-8'))
        table = {}
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    table[str(k)] = float(v)
                except Exception:
                    continue
        elif isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                key = row.get('img_path', row.get('img'))
                val = row.get('gsd', None)
                if key is None or val is None:
                    continue
                try:
                    table[str(key)] = float(val)
                    table[Path(str(key)).name] = float(val)
                except Exception:
                    continue
        self.gsd_table = table
        logger.info(f'Loaded {len(self.gsd_table)} GSD entries from {json_path}.')

    def _get_batch_gsd(self, batch_data_samples, device, dtype) -> Tensor:
        if self.gsd_source == 'online' and self._cached_batch_gsd is not None:
            gsd = self._cached_batch_gsd.to(device=device, dtype=dtype)
            if self.use_log_gsd:
                gsd = torch.log(torch.clamp(gsd, min=1e-6))
            return gsd

        values = []
        for sample in batch_data_samples:
            img_path = sample.metainfo.get('img_path', '')
            gsd = self.gsd_table.get(img_path, None)
            if gsd is None:
                gsd = self.gsd_table.get(Path(img_path).name, self.default_gsd)
            values.append(float(gsd))
        gsd = torch.tensor(values, device=device, dtype=dtype).unsqueeze(-1)
        if self.use_log_gsd:
            gsd = torch.log(torch.clamp(gsd, min=1e-6))
        return gsd

    def _to_01_rgb(self, batch_inputs: Tensor) -> Tensor:
        """Recover RGB in [0,1] from model input after data_preprocessor."""
        mean = self.data_preprocessor.mean.to(batch_inputs.device).view(1, 3, 1, 1)
        std = self.data_preprocessor.std.to(batch_inputs.device).view(1, 3, 1, 1)
        x = batch_inputs * std + mean
        if mean.max() > 1.5:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    def _extract_fixed_patch_batch(self, x01: Tensor) -> Tensor:
        """Extract fixed local patches: center + four corners (or center-biased)."""
        b, _, h, w = x01.shape
        crop = self.gsd_patch_size
        k = self.gsd_num_patches
        patches = []
        for i in range(b):
            img = x01[i:i + 1]
            pad_h = max(0, crop - h)
            pad_w = max(0, crop - w)
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            _, _, hp, wp = img.shape
            max_x = max(0, wp - crop)
            max_y = max(0, hp - crop)
            cx = max_x // 2
            cy = max_y // 2
            if hp < self.gsd_small_img_threshold or wp < self.gsd_small_img_threshold:
                step = max(1, crop // 8)
                pos = [
                    (cx, cy), (max(0, cx - step), cy), (min(max_x, cx + step), cy),
                    (cx, max(0, cy - step)), (cx, min(max_y, cy + step))
                ]
            else:
                pos = [(0, 0), (max_x, 0), (0, max_y), (max_x, max_y), (cx, cy)]
            pos = pos[:k] + [pos[-1]] * max(0, k - len(pos))
            one = [img[:, :, y:y + crop, x:x + crop] for (x, y) in pos]
            one = torch.cat(one, dim=0)  # (K,3,crop,crop)
            patches.append(one)
        patches = torch.stack(patches, dim=0)  # (B,K,3,crop,crop)

        # ImageNet normalize for gsd predictor
        mean = patches.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = patches.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        patches = (patches - mean) / std
        return patches

    def _update_online_gsd_cache(self, batch_inputs: Tensor) -> None:
        if self.gsd_source != 'online' or self.gsd_predictor is None:
            self._cached_batch_gsd = None
            return
        x01 = self._to_01_rgb(batch_inputs)
        patches = self._extract_fixed_patch_batch(x01)
        gsd = self.gsd_predictor(patches).squeeze(-1).unsqueeze(-1)
        if not self.gsd_online_trainable:
            gsd = gsd.detach()
        if self.use_log_gsd:
            gsd = torch.exp(gsd)
        self._cached_batch_gsd = gsd

    def loss(self, batch_inputs: Tensor, batch_data_samples):
        self._last_query_group_logits = None
        self._update_online_gsd_cache(batch_inputs)
        losses = super().loss(batch_inputs, batch_data_samples)
        if self.query_group_enabled and self._last_query_group_logits is not None:
            group_gsd = self._get_batch_gsd(
                batch_data_samples=batch_data_samples,
                device=self._last_query_group_logits.device,
                dtype=self._last_query_group_logits.dtype)
            losses.update(
                self._query_group_aux_loss(
                    logits=self._last_query_group_logits,
                    batch_data_samples=batch_data_samples,
                    gsd=group_gsd))
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples, rescale: bool = True):
        self._last_query_group_logits = None
        self._update_online_gsd_cache(batch_inputs)
        return super().predict(batch_inputs, batch_data_samples, rescale=rescale)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples=None,
    ):
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            batch_data_samples=batch_data_samples)

        if (not self.gsd_enabled) or batch_data_samples is None:
            return decoder_inputs_dict, head_inputs_dict

        query = decoder_inputs_dict['query']
        gsd = self._get_batch_gsd(
            batch_data_samples=batch_data_samples,
            device=query.device,
            dtype=query.dtype)
        gsd_emb = self.gsd_embed(gsd)
        if self.gsd_modulate == 'film':
            gamma, beta = self.gsd_film(gsd_emb).chunk(2, dim=-1)
            query = query * (1 + self.gsd_scale * gamma.unsqueeze(1)) + \
                self.gsd_scale * beta.unsqueeze(1)
        else:
            query = query + self.gsd_scale * gsd_emb.unsqueeze(1)
        decoder_inputs_dict['query'] = query

        # Always compute grouping logits so the grouping head can be trained
        # with auxiliary supervision, even when training-time grouping is off.
        if self.query_group_enabled:
            feat_sem = memory.mean(dim=1)
            priors = self._extract_density_scale_prior(memory, spatial_shapes)
            self._last_query_group_logits = self.query_group_head(
                torch.cat([feat_sem, gsd_emb, priors], dim=-1))

        if self.query_group_enabled and self._query_group_training_active():
            proxy_scores = self._gt_proxy_query_scores(
                batch_data_samples=batch_data_samples,
                device=query.device,
                dtype=query.dtype,
                gsd=gsd,
                use_gsd=self.query_group_train_use_gsd)
            q_match_per_img, _ = self._scores_to_query_bins(proxy_scores)
            keep_match = int(q_match_per_img.max().item())
            keep_match = self._sync_train_keep_match(
                keep_match=keep_match, device=query.device)
            self._apply_query_grouping(
                decoder_inputs_dict=decoder_inputs_dict,
                head_inputs_dict=head_inputs_dict,
                memory=memory,
                spatial_shapes=spatial_shapes,
                gsd_emb=gsd_emb,
                forced_q_match_per_img=q_match_per_img,
                keep_match_override=keep_match)
        elif self.query_group_enabled and (not self.training):
            self._apply_query_grouping(
                decoder_inputs_dict=decoder_inputs_dict,
                head_inputs_dict=head_inputs_dict,
                memory=memory,
                spatial_shapes=spatial_shapes,
                gsd_emb=gsd_emb)

        return decoder_inputs_dict, head_inputs_dict

    def _extract_density_scale_prior(self, memory: Tensor,
                                     spatial_shapes: Tensor) -> Tensor:
        """Extract lightweight density/scale priors from encoder memory.

        Returns:
            Tensor: Shape (bs, 2), containing [density_proxy, scale_proxy].
        """
        bs = memory.size(0)
        spatial_shapes = spatial_shapes.to(memory.device)
        level_tokens = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        splits = memory.split(level_tokens, dim=1)
        energy = [seg.pow(2).mean(dim=(1, 2)) for seg in splits]
        energy = torch.stack(energy, dim=1)  # (bs, num_levels)

        # density proxy: larger activation variance -> likely denser/complex scene
        density = memory.var(dim=1, unbiased=False).mean(dim=1)

        # scale proxy: coarse-vs-fine energy contrast (positive => more large objs)
        if energy.size(1) >= 2:
            coarse = energy[:, -1]
            fine = energy[:, 0]
            scale = torch.log1p(coarse) - torch.log1p(fine)
        else:
            scale = energy[:, 0]

        prior = torch.stack([density, scale], dim=-1)
        prior = (prior - prior.mean(dim=0, keepdim=True)) / \
                (prior.std(dim=0, keepdim=True) + 1e-6)
        return prior.to(dtype=memory.dtype).view(bs, 2)

    def _choose_query_count(self, query: Tensor, memory: Tensor, gsd_emb: Tensor,
                            spatial_shapes: Tensor,
                            dn_meta: Optional[Dict] = None,
                            logits: Optional[Tensor] = None) -> Tensor:
        feat_sem = memory.mean(dim=1)  # (bs, C)
        priors = self._extract_density_scale_prior(memory, spatial_shapes)
        if logits is None:
            feat = torch.cat([feat_sem, gsd_emb, priors], dim=-1)
            logits = self.query_group_head(feat)
        group_idx = torch.argmax(logits, dim=-1)  # (bs,)
        bins = logits.new_tensor(self.query_group_bins, dtype=torch.long)
        q_target = bins[group_idx].to(dtype=torch.long)

        # Extra heuristic from priors to stabilize early training.
        density_boost = self.query_group_density_alpha * priors[:, 0]
        scale_boost = self.query_group_scale_alpha * (-priors[:, 1])
        boost = density_boost + scale_boost
        up_mask = (boost > 0.8) & (group_idx < len(self.query_group_bins) - 1)
        down_mask = (boost < -0.8) & (group_idx > 0)
        group_idx = torch.where(up_mask, group_idx + 1, group_idx)
        group_idx = torch.where(down_mask, group_idx - 1, group_idx)
        q_target = bins[group_idx].to(dtype=torch.long)

        # Clamp by available matching queries.
        q_total = query.size(1)
        if self.training:
            num_dn = int(dn_meta['num_denoising_queries']) if dn_meta else 0
            max_match = max(1, q_total - num_dn)
            q_target = torch.clamp(q_target, min=1, max=max_match)
        else:
            q_target = torch.clamp(q_target, min=1, max=q_total)
        return q_target

    def _query_group_training_active(self) -> bool:
        epoch = int(getattr(self, 'epoch', 0) or 0)
        if not (self.training and self.query_group_train_enabled
                and epoch >= self.query_group_train_begin_epoch):
            return False
        if self.query_group_train_epoch_ranges is None:
            return True
        for begin, end in self.query_group_train_epoch_ranges:
            if epoch >= begin and (end is None or epoch < end):
                return True
        return False

    def _gt_proxy_query_scores(self,
                               batch_data_samples,
                               device,
                               dtype,
                               gsd: Optional[Tensor] = None,
                               use_gsd: bool = False) -> Tensor:
        """Build a GT-derived proxy score for selecting a query bin."""
        target_scores = []
        eps = 1e-6

        gsd_values = None
        if use_gsd and gsd is not None and self.query_group_gsd_weight != 0:
            gsd_values = gsd.detach().to(device=device, dtype=dtype).view(-1)
            if self.use_log_gsd:
                gsd_values = torch.exp(gsd_values)
            gsd_values = gsd_values.clamp_min(eps)

        for idx, sample in enumerate(batch_data_samples):
            gt_instances = getattr(sample, 'gt_instances', None)
            if gt_instances is None or not hasattr(gt_instances, 'bboxes'):
                target_scores.append(1.0)
                continue

            bboxes = gt_instances.bboxes
            n = int(bboxes.size(0))
            if n == 0:
                target_scores.append(1.0)
                continue

            img_h, img_w = sample.metainfo.get(
                'img_shape', sample.metainfo.get('ori_shape', (1024, 1024)))[:2]
            img_area = float(max(1, img_h * img_w))
            wh = (bboxes[:, 2:4] - bboxes[:, 0:2]).clamp(min=0)
            areas = (wh[:, 0] * wh[:, 1]).float()
            area_ratio = (areas / img_area).clamp(min=eps)
            small_ratio = (area_ratio < 0.001).float().mean().item()
            mean_scale = torch.sqrt(area_ratio).mean().item()
            density = n / (img_area / (1024.0 * 1024.0))
            scale_bonus = max(0.0, self.query_group_scale_ref - mean_scale)

            score = (
                n * self.query_group_query_per_gt
                + self.query_group_small_obj_weight * small_ratio
                + self.query_group_density_weight * density
                + self.query_group_scale_bonus_weight * scale_bonus)

            if gsd_values is not None and idx < gsd_values.numel():
                ref = max(eps, self.query_group_gsd_ref)
                gsd_boost = torch.log(
                    gsd_values.new_tensor(ref) / gsd_values[idx]).clamp(
                        min=-self.query_group_gsd_clip,
                        max=self.query_group_gsd_clip).item()
                score += self.query_group_gsd_weight * gsd_boost

            target_scores.append(float(max(1.0, score)))

        return torch.tensor(target_scores, device=device, dtype=dtype)

    def _scores_to_query_bins(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        bins = scores.new_tensor(self.query_group_bins, dtype=scores.dtype)
        dist_to_bins = torch.abs(scores[:, None] - bins[None, :])
        target_idx = torch.argmin(dist_to_bins, dim=-1)
        q_target = bins[target_idx].to(dtype=torch.long)
        return q_target, target_idx

    def _sync_train_keep_match(self, keep_match: int, device) -> int:
        if (not self.query_group_train_sync_across_ranks
                or not dist.is_available() or not dist.is_initialized()):
            return keep_match
        keep_tensor = torch.tensor([keep_match], device=device, dtype=torch.long)
        dist.all_reduce(keep_tensor, op=dist.ReduceOp.MAX)
        return int(keep_tensor.item())

    def _apply_query_grouping(self,
                              decoder_inputs_dict: Dict,
                              head_inputs_dict: Dict,
                              memory: Tensor,
                              spatial_shapes: Tensor,
                              gsd_emb: Tensor,
                              forced_q_match_per_img: Optional[Tensor] = None,
                              keep_match_override: Optional[int] = None) -> None:
        query = decoder_inputs_dict['query']
        reference_points = decoder_inputs_dict['reference_points']
        dn_mask = decoder_inputs_dict.get('dn_mask', None)
        dn_meta = head_inputs_dict.get('dn_meta', None)
        cached_logits = self._last_query_group_logits

        if forced_q_match_per_img is None:
            q_match_per_img = self._choose_query_count(
                query=query,
                memory=memory,
                gsd_emb=gsd_emb,
                spatial_shapes=spatial_shapes,
                dn_meta=dn_meta,
                logits=cached_logits)
        else:
            q_match_per_img = forced_q_match_per_img.to(
                device=query.device, dtype=torch.long)
        self._last_group_queries = q_match_per_img.detach().cpu()

        # Ensure matching queries are ordered by descending encoder confidence.
        # Upstream RTDETR uses torch.topk(..., sorted=True) for this order, but
        # we keep this fallback reorder for robustness.
        if self.training and 'enc_outputs_class' in head_inputs_dict:
            topk_score = head_inputs_dict['enc_outputs_class']
            rank = topk_score.max(dim=-1).values.argsort(dim=1, descending=True)
            num_match = rank.size(1)
            query_dn = query[:, :-num_match, :]
            query_match = query[:, -num_match:, :]
            ref_dn = reference_points[:, :-num_match, :]
            ref_match = reference_points[:, -num_match:, :]
            query_match = torch.gather(
                query_match, 1, rank.unsqueeze(-1).expand_as(query_match))
            ref_match = torch.gather(
                ref_match, 1, rank.unsqueeze(-1).expand_as(ref_match))
            query = torch.cat([query_dn, query_match], dim=1)
            reference_points = torch.cat([ref_dn, ref_match], dim=1)
            for key in ('enc_outputs_class', 'enc_outputs_coord'):
                if key in head_inputs_dict and head_inputs_dict[key] is not None:
                    value = head_inputs_dict[key]
                    head_inputs_dict[key] = torch.gather(
                        value, 1, rank.unsqueeze(-1).expand_as(value))

        if self.training and dn_meta is not None:
            num_dn = int(dn_meta.get('num_denoising_queries', 0))
            max_match = max(1, query.size(1) - num_dn)
            q_match_per_img = torch.clamp(q_match_per_img, min=1, max=max_match)
            keep_match = int(q_match_per_img.max().item())
            if keep_match_override is not None:
                keep_match = int(max(1, min(keep_match_override, max_match)))
            keep = num_dn + keep_match
            query = query[:, :keep, :]
            reference_points = reference_points[:, :keep, :]

            # Per-image mask over matching part.
            arange = torch.arange(keep_match, device=query.device)[None, :]
            valid_match = arange < q_match_per_img[:, None]
            query_dn = query[:, :num_dn, :]
            query_match = query[:, num_dn:, :] * valid_match.unsqueeze(-1).to(
                query.dtype)
            ref_dn = reference_points[:, :num_dn, :]
            ref_match = reference_points[:, num_dn:, :] * valid_match.unsqueeze(
                -1).to(reference_points.dtype)
            query = torch.cat([query_dn, query_match], dim=1)
            reference_points = torch.cat([ref_dn, ref_match], dim=1)

            decoder_inputs_dict['query'] = query
            decoder_inputs_dict['reference_points'] = reference_points
            if dn_mask is not None:
                decoder_inputs_dict['dn_mask'] = dn_mask[:keep, :keep]
            dn_meta['num_matching_queries'] = keep_match
            for key in ('enc_outputs_class', 'enc_outputs_coord'):
                if key in head_inputs_dict and head_inputs_dict[key] is not None:
                    head_inputs_dict[key] = head_inputs_dict[key][:, :keep_match]
        else:
            max_match = query.size(1)
            q_match_per_img = torch.clamp(q_match_per_img, min=1, max=max_match)
            keep = int(q_match_per_img.max().item())
            if keep_match_override is not None:
                keep = int(max(1, min(keep_match_override, max_match)))
            query = query[:, :keep, :]
            reference_points = reference_points[:, :keep, :]
            arange = torch.arange(keep, device=query.device)[None, :]
            valid_match = arange < q_match_per_img[:, None]
            query = query * valid_match.unsqueeze(-1).to(query.dtype)
            reference_points = reference_points * valid_match.unsqueeze(-1).to(
                reference_points.dtype)
            decoder_inputs_dict['query'] = query
            decoder_inputs_dict['reference_points'] = reference_points

    def _query_group_aux_loss(self,
                              logits: Tensor,
                              batch_data_samples,
                              gsd: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Train grouping head without changing DFINE matching layout."""
        if logits.numel() == 0:
            return {}
        tgt_queries = self._gt_proxy_query_scores(
            batch_data_samples=batch_data_samples,
            device=logits.device,
            dtype=logits.dtype,
            gsd=gsd,
            use_gsd=self.query_group_train_use_gsd)
        _, target_idx = self._scores_to_query_bins(tgt_queries)
        loss_group = F.cross_entropy(logits, target_idx)
        return {'loss_query_group': self.query_group_loss_weight * loss_group}
