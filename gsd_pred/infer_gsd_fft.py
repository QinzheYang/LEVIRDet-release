# infer_gsd_fft_fixed.py
# -*- coding: utf-8 -*-
"""
用训练好的 GSD FFT 模型做推理（自动识别 fusion=concat_mlp 或 gated）：
- 输入单张图：print 预测 gsd
- 输入文件夹：对文件夹内所有图像预测 gsd，保存 results_gsd.json，并输出可视化（直方图 + 排序曲线）

示例：
1) 单张图
python infer_gsd_fft_fixed.py --ckpt /mnt/dataset/wanglubo/yqz-tmp/gsd_fft/best.pt --input /path/to/xxx.jpg --use_log_gsd

2) 文件夹
python infer_gsd_fft_fixed.py --ckpt /mnt/dataset/wanglubo/yqz-tmp/gsd_fft/best.pt --input /path/to/folder --out_dir /path/to/out --use_log_gsd
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm

import matplotlib.pyplot as plt

# -------------------------
# PIL robust
# -------------------------
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# -------------------------
# Patch helpers
# -------------------------
def pad_to_min_size(im: Image.Image, min_hw: int) -> Image.Image:
    w, h = im.size
    pad_w = max(0, min_hw - w)
    pad_h = max(0, min_hw - h)
    if pad_w == 0 and pad_h == 0:
        return im
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return ImageOps.expand(im, border=(left, top, right, bottom), fill=(0, 0, 0))


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[:, None, None]
    return (x - mean) / std


def center_biased_positions(w: int, h: int, crop: int, k: int) -> List[Tuple[int, int]]:
    cx, cy = w // 2, h // 2
    base_x = cx - crop // 2
    base_y = cy - crop // 2
    if k <= 1:
        return [(base_x, base_y)]
    step = max(1, crop // 8)
    offsets = [
        (0, 0), (-step, 0), (step, 0), (0, -step), (0, step),
        (-step, -step), (step, -step), (-step, step), (step, step),
    ]
    pos = []
    for dx, dy in offsets:
        pos.append((base_x + dx, base_y + dy))
        if len(pos) >= k:
            break
    if len(pos) < k:
        pos += [pos[-1]] * (k - len(pos))
    return pos


def fixed_positions(im_w: int, im_h: int, crop: int, k: int, fixed_mode: str, small_thr: int) -> List[Tuple[int, int]]:
    if im_w < small_thr or im_h < small_thr:
        return center_biased_positions(im_w, im_h, crop, k)

    max_x = max(0, im_w - crop)
    max_y = max(0, im_h - crop)
    cx = max_x // 2
    cy = max_y // 2

    if fixed_mode == "center":
        pos = [(cx, cy)]
    else:
        pos = [(0, 0), (max_x, 0), (0, max_y), (max_x, max_y), (cx, cy)]

    if len(pos) >= k:
        pos = pos[:k]
    else:
        pos = pos + [pos[-1]] * (k - len(pos))
    return pos


def image_to_patches_tensor(
    path: Path,
    crop: int,
    k: int,
    fixed_mode: str = "five",
    small_thr: int = 512,
) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    im = pad_to_min_size(im, crop)  # 只 pad 到 crop，避免引入更大黑边
    w, h = im.size

    pos = fixed_positions(w, h, crop, k, fixed_mode=fixed_mode, small_thr=small_thr)

    patches = []
    for x, y in pos:
        patch = im.crop((x, y, x + crop, y + crop))
        t = torchvision.transforms.functional.to_tensor(patch)  # float [0,1]
        t = imagenet_normalize(t)
        patches.append(t)
    return torch.stack(patches, dim=0)  # (K,3,crop,crop)


def list_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda p: p.name.lower())
    return imgs


def save_visualizations(out_dir: Path, results: List[Dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    gsds = [r["gsd"] for r in results if isinstance(r.get("gsd", None), (int, float))]
    if not gsds:
        print("[Warn] no valid predictions for visualization.")
        return

    # histogram
    plt.figure()
    plt.hist(gsds, bins=80)
    plt.xlabel("Predicted GSD")
    plt.ylabel("Count")
    plt.title("GSD prediction histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "gsd_hist.png", dpi=200)
    plt.close()

    # sorted curve
    plt.figure()
    plt.plot(sorted(gsds))
    plt.xlabel("Sorted index")
    plt.ylabel("Predicted GSD")
    plt.title("GSD prediction (sorted)")
    plt.tight_layout()
    plt.savefig(out_dir / "gsd_sorted.png", dpi=200)
    plt.close()


# -------------------------
# Model: CNN + FFT radial + Edge stats (regression)
# -------------------------
def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class RadialFFTSpectrumEncoder(nn.Module):
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
        self.register_buffer("_radial_bin_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_radial_bin_cnt", torch.empty(0, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def _build_radial_bins(self, H: int, W: int, device):
        Wx = W // 2 + 1
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)
        yy = fy[:, None].expand(H, Wx)
        xx = fx[None, :].expand(H, Wx)
        rr = torch.sqrt(xx * xx + yy * yy)
        rmax = rr.max().clamp_min(1e-12)
        rr01 = rr / rmax
        R = self.radial_bins
        bin_idx = torch.clamp((rr01 * (R - 1)).long(), 0, R - 1)
        flat = bin_idx.reshape(-1)
        cnt = torch.bincount(flat, minlength=R).float().clamp_min(1.0)
        self._cache_hw = (H, W)
        self._radial_bin_idx = flat
        self._radial_bin_cnt = cnt

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x_rgb.shape
        device = x_rgb.device
        if self._cache_hw != (H, W) or self._radial_bin_idx.numel() == 0:
            self._build_radial_bins(H, W, device)

        g = rgb_to_gray(x_rgb).squeeze(1)  # (N,H,W)
        Freq = torch.fft.rfft2(g, norm="ortho")
        mag = torch.log1p(torch.abs(Freq))  # (N,H,Wx)

        flat_mag = mag.reshape(N, -1)
        bin_idx = self._radial_bin_idx
        R = self.radial_bins

        spec_sum = torch.zeros((N, R), device=device, dtype=flat_mag.dtype)
        spec_sum.scatter_add_(1, bin_idx[None, :].expand(N, -1), flat_mag)
        spec = spec_sum / self._radial_bin_cnt[None, :].to(spec_sum.dtype)

        z = self.conv(spec.unsqueeze(1))
        z = self.proj(z)
        return z


class EdgeStatEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("lap", lap, persistent=False)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        g = rgb_to_gray(x_rgb)  # (N,1,H,W)
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

        feat = torch.stack([mag_mean, mag_std, mag_p90, mag_p99, lap_var, lap_abs_mean], dim=1)
        return self.mlp(feat)


class GSDRegressor_CNN_FFT_Edge(nn.Module):
    """
    支持两种融合方式：
    - concat_mlp: fuse.* 存在
    - gated: gate.* 存在
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = False,
        embed_dim: int = 256,
        fft_radial_bins: int = 64,
        agg: str = "attn",               # mean / max / attn
        fusion: str = "auto",            # auto / concat_mlp / gated
    ):
        super().__init__()
        assert agg in ("mean", "max", "attn")
        assert fusion in ("auto", "concat_mlp", "gated")
        self.embed_dim = embed_dim
        self.agg = agg
        self.fusion = fusion

        if backbone_name == "resnet18":
            net = torchvision.models.resnet18(weights=None)
            feat_dim = 512
        elif backbone_name == "resnet34":
            net = torchvision.models.resnet34(weights=None)
            feat_dim = 512
        elif backbone_name == "resnet50":
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

        self.fft_enc = RadialFFTSpectrumEncoder(radial_bins=fft_radial_bins, out_dim=embed_dim)
        self.edge_enc = EdgeStatEncoder(out_dim=embed_dim)

        self.fusion = fusion

        if fusion == "concat_mlp":
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.gate = None
        elif fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 3, 128),
                nn.GELU(),
                nn.Linear(128, 3),
            )
            self.fuse = None
        else:
            raise ValueError(f"bad fusion={fusion}")

        if agg == "attn":
            self.attn_score = nn.Linear(embed_dim, 1)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,K,3,H,W) normalized patches
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)

        f = self.cnn(x).flatten(1)
        e_cnn = self.cnn_proj(f)
        e_fft = self.fft_enc(x)
        e_edge = self.edge_enc(x)

        cat = torch.cat([e_cnn, e_fft, e_edge], dim=1)

        if self.fusion == "concat_mlp":
            e = self.fuse(cat)
        else:  # gated
            g = torch.softmax(self.gate(cat), dim=1)
            e = g[:, 0:1] * e_cnn + g[:, 1:2] * e_fft + g[:, 2:3] * e_edge

        e = e.view(B, K, self.embed_dim)

        if self.agg == "mean":
            g_img = e.mean(dim=1)
        elif self.agg == "max":
            g_img = e.max(dim=1).values
        else:
            a = torch.softmax(self.attn_score(e).squeeze(-1), dim=1)  # (B,K)
            g_img = (e * a.unsqueeze(-1)).sum(dim=1)

        return self.head(g_img)


# -------------------------
# Checkpoint loading (auto detect fusion)
# -------------------------
def load_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    backbone: str = "resnet50",
    embed_dim: int = 256,
    fft_bins: int = 64,
    agg: str = "attn",
) -> nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location=device)

    # 取 state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unknown checkpoint format: {type(ckpt)}")

    # 去 module. 前缀
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    fusion = "gated"

    model = GSDRegressor_CNN_FFT_Edge(
        backbone_name=backbone,
        pretrained=False,
        embed_dim=embed_dim,
        fft_radial_bins=fft_bins,
        agg=agg,
        fusion=fusion,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # 这两类缺失/多余如果出现，就提示
    if missing:
        print(f"[Warn] missing keys: {len(missing)} (show first 20): {missing[:20]}")
    if unexpected:
        print(f"[Warn] unexpected keys: {len(unexpected)} (show first 20): {unexpected[:20]}")

    print(f"[Info] fusion detected: {fusion}")
    model.eval()
    return model


# -------------------------
# Inference
# -------------------------
@torch.no_grad()
def infer_one(
    model: nn.Module,
    img_path: Path,
    device: torch.device,
    crop: int,
    k: int,
    fixed_mode: str,
    small_thr: int,
    use_log_gsd: bool,
) -> float:
    x = image_to_patches_tensor(img_path, crop=crop, k=k, fixed_mode=fixed_mode, small_thr=small_thr)
    x = x.unsqueeze(0).to(device)  # (1,K,3,H,W)
    pred = model(x).squeeze().item()
    if use_log_gsd:
        pred = math.exp(pred)
    return float(pred)


def main():
    ap = argparse.ArgumentParser("Infer GSD with trained CNN+FFT+Edge regressor (auto fusion)")
    ap.add_argument("--ckpt", default="/mnt/dataset/wanglubo/yqz-tmp/gsd_fft/best.pt", help="path to best.pt")
    ap.add_argument("--input", required=True, help="image path or folder path")
    ap.add_argument("--out_dir", default="", help="output dir for folder mode; default: <input>_gsd_out")
    ap.add_argument("--out_json", default="results_gsd.json", help="json filename under out_dir for folder mode")

    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--fft_bins", type=int, default=64)
    ap.add_argument("--agg", default="attn", choices=["mean", "max", "attn"])

    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fixed_mode", default="five", choices=["center", "five"])
    ap.add_argument("--small_img_threshold", type=int, default=512)

    ap.add_argument("--use_log_gsd", action="store_true", help="训练回归 log(gsd) 时，推理 exp 回 gsd")
    ap.add_argument("--device", default="cuda", help="cuda / cpu / cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    model = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        device=device,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        fft_bins=args.fft_bins,
        agg=args.agg,
    )

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"input not found: {inp}")

    # single image
    if inp.is_file():
        pred = infer_one(
            model, inp, device,
            crop=args.crop, k=args.k,
            fixed_mode=args.fixed_mode,
            small_thr=args.small_img_threshold,
            use_log_gsd=args.use_log_gsd,
        )
        print(f"{inp} -> predicted_gsd = {pred:.6f}")
        return

    # folder
    out_dir = Path(args.out_dir) if args.out_dir else (inp.parent / (inp.name + "_gsd_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(inp)
    print(f"Found {len(imgs)} images in: {inp}")

    results = []
    for p in tqdm(imgs, desc="Infer", unit="img", mininterval=0.5):
        try:
            gsd_pred = infer_one(
                model, p, device,
                crop=args.crop, k=args.k,
                fixed_mode=args.fixed_mode,
                small_thr=args.small_img_threshold,
                use_log_gsd=args.use_log_gsd,
            )
            results.append({"img": p.name, "gsd": gsd_pred})
        except Exception as e:
            results.append({"img": p.name, "gsd": None, "error": str(e)})

    out_json_path = out_dir / args.out_json
    out_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json_path}")

    save_visualizations(out_dir, results)
    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()