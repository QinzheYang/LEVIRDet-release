# -*- coding: utf-8 -*-
"""Evaluate a GSD regressor extracted from a detector checkpoint.

This script is meant for checking whether the online GSD predictor still
predicts GSD well after it has been fine-tuned inside the detector.

Example:
    python gsd_pred/eval_finetuned_gsd_from_detector.py \
        --index /mnt/user/wanglubo/yqz-tmp/fmow/test_demo/index.json \
        --ckpt work_dirs/xxx/epoch_117.pth \
        --ckpt-type detector \
        --model-impl detector \
        --use-log-gsd \
        --batch-size 64 \
        --num-workers 2 \
        --out-json work_dirs/gsd_eval/epoch117_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF
from tqdm import tqdm

import pyarrow as pa
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsd_pred.train_gsd_resnet_fft_arrow_ddp import (  # noqa: E402
    FMOWArrowDataset,
    GSDRegressor_CNN_FFT_Edge,
)


StateDict = MutableMapping[str, torch.Tensor]


def imagenet_normalize_patch_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[:, None, None]
    return (x - mean) / std


def detector_fixed_patches_tensor(
    im: Image.Image,
    crop: int,
    k: int,
    small_thr: int,
    use_fp16: bool = False,
) -> torch.Tensor:
    """Match DEIMV2GSDGuided._extract_fixed_patch_batch for one image."""
    x01 = TF.to_tensor(im)
    _, h, w = x01.shape
    pad_h = max(0, crop - h)
    pad_w = max(0, crop - w)
    if pad_h > 0 or pad_w > 0:
        x01 = F.pad(x01.unsqueeze(0), (0, pad_w, 0, pad_h), mode="constant", value=0).squeeze(0)

    _, hp, wp = x01.shape
    max_x = max(0, wp - crop)
    max_y = max(0, hp - crop)
    cx = max_x // 2
    cy = max_y // 2
    if hp < small_thr or wp < small_thr:
        step = max(1, crop // 8)
        pos = [
            (cx, cy),
            (max(0, cx - step), cy),
            (min(max_x, cx + step), cy),
            (cx, max(0, cy - step)),
            (cx, min(max_y, cy + step)),
        ]
    else:
        pos = [(0, 0), (max_x, 0), (0, max_y), (max_x, max_y), (cx, cy)]
    pos = pos[:k] + [pos[-1]] * max(0, k - len(pos))

    patches = []
    for x, y in pos:
        patch = x01[:, y:y + crop, x:x + crop]
        patch = imagenet_normalize_patch_tensor(patch)
        if use_fp16:
            patch = patch.half()
        patches.append(patch)
    return torch.stack(patches, dim=0)


class EvalFMOWArrowDataset(FMOWArrowDataset):
    """FMOWArrowDataset with selectable evaluation patch extraction."""

    def __init__(self, *args, patch_impl: str = "standalone", **kwargs):
        super().__init__(*args, **kwargs)
        if patch_impl not in ("standalone", "detector"):
            raise ValueError(f"Unknown patch_impl: {patch_impl}")
        self.patch_impl = patch_impl

    def __getitem__(self, idx: int):
        if self.patch_impl == "standalone":
            return super().__getitem__(idx)

        shard_name, off = self._locate(int(idx))
        tbl = self._cache.get_table(shard_name)

        img_name = tbl["img_name"][off].as_py()
        rel_folder = tbl["rel_folder"][off].as_py()
        gsd = float(tbl["gsd"][off].as_py())
        img_bytes = tbl["img_bytes"][off].as_py()

        buf = img_bytes
        if isinstance(buf, pa.Buffer):
            buf = buf.to_pybytes()
        elif isinstance(buf, memoryview):
            buf = buf.tobytes()
        elif not isinstance(buf, (bytes, bytearray)):
            buf = bytes(buf)

        im = Image.open(io.BytesIO(buf)).convert("RGB")
        x = detector_fixed_patches_tensor(
            im,
            crop=self.crop,
            k=self.k,
            small_thr=self.small_thr,
            use_fp16=self.use_fp16_patches,
        )
        y = math.log(max(gsd, 1e-8)) if self.use_log_gsd else gsd
        y = torch.tensor([y], dtype=torch.float32)
        sample_id = f"{rel_folder}/{img_name}"
        return x, y, sample_id


def strip_known_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def normalize_state_keys(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    return {strip_known_prefixes(str(k)): v for k, v in state.items()}


def select_detector_state(ckpt: Any, source: str) -> StateDict:
    if not isinstance(ckpt, dict):
        raise ValueError("Detector checkpoint must be a dict.")

    candidates = []
    if source == "auto":
        candidates = ["state_dict", "ema_state_dict", "model", "model_state_dict"]
    else:
        candidates = [source]

    for key in candidates:
        state = ckpt.get(key, None)
        if isinstance(state, MutableMapping):
            return normalize_state_keys(state)

    if all(torch.is_tensor(v) for v in ckpt.values()):
        return normalize_state_keys(ckpt)

    raise KeyError(
        f"Could not find a detector state dict. Tried: {', '.join(candidates)}")


def select_standalone_state(ckpt: Any) -> StateDict:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            state = ckpt.get(key, None)
            if isinstance(state, MutableMapping):
                return normalize_state_keys(state)
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return normalize_state_keys(ckpt)
    raise ValueError("Could not find standalone GSD model weights in checkpoint.")


def extract_gsd_predictor_state(state: StateDict) -> StateDict:
    extracted: Dict[str, torch.Tensor] = {}
    marker = "gsd_predictor."
    for raw_key, value in state.items():
        key = strip_known_prefixes(raw_key)
        if marker in key:
            key = key.split(marker, 1)[1]
            extracted[key] = value
    return extracted


def infer_ckpt_type(ckpt: Any, detector_source: str) -> str:
    if isinstance(ckpt, dict):
        try:
            state = select_detector_state(ckpt, detector_source)
            if extract_gsd_predictor_state(state):
                return "detector"
        except Exception:
            pass
        try:
            state = select_standalone_state(ckpt)
            if state and not extract_gsd_predictor_state(state):
                return "standalone"
        except Exception:
            pass
    return "standalone"


def detect_fusion(state: StateDict, fallback: str) -> str:
    if fallback != "auto":
        return fallback
    keys = set(state.keys())
    if any(k.startswith("gate.") for k in keys):
        return "gated"
    if any(k.startswith("fuse.") for k in keys):
        return "concat_mlp"
    return "gated"


def build_model(
    model_impl: str,
    backbone: str,
    embed_dim: int,
    fft_bins: int,
    agg: str,
    fusion: str,
    use_log_gsd: bool,
    device: torch.device,
) -> nn.Module:
    if model_impl == "detector":
        from mmdet.models.detectors.deimv2 import _GSDRegressorCNNFFTEdge

        model = _GSDRegressorCNNFFTEdge(
            backbone_name=backbone,
            embed_dim=embed_dim,
            fft_radial_bins=fft_bins,
            agg=agg,
            fusion=fusion,
        )
    elif model_impl == "standalone":
        model = GSDRegressor_CNN_FFT_Edge(
            backbone_name=backbone,
            pretrained=False,
            embed_dim=embed_dim,
            fft_radial_bins=fft_bins,
            agg=agg,
            fusion=fusion,
            positive_out=(not use_log_gsd),
        )
    else:
        raise ValueError(f"Unknown model implementation: {model_impl}")
    return model.to(device)


def load_gsd_model(
    ckpt_path: Path,
    ckpt_type: str,
    detector_state_source: str,
    model_impl: str,
    backbone: str,
    embed_dim: int,
    fft_bins: int,
    agg: str,
    fusion: str,
    use_log_gsd: bool,
    device: torch.device,
    ckpt_obj: Optional[Any] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = ckpt_obj if ckpt_obj is not None else torch.load(str(ckpt_path), map_location="cpu")
    real_ckpt_type = infer_ckpt_type(ckpt, detector_state_source) if ckpt_type == "auto" else ckpt_type

    if real_ckpt_type == "detector":
        full_state = select_detector_state(ckpt, detector_state_source)
        state = extract_gsd_predictor_state(full_state)
        if not state:
            raise KeyError(f"No gsd_predictor.* weights found in detector ckpt: {ckpt_path}")
        real_model_impl = "detector" if model_impl == "auto" else model_impl
    elif real_ckpt_type == "standalone":
        state = select_standalone_state(ckpt)
        real_model_impl = "standalone" if model_impl == "auto" else model_impl
    else:
        raise ValueError(f"Unknown ckpt type: {real_ckpt_type}")

    state = normalize_state_keys(state)
    real_fusion = detect_fusion(state, fusion)
    model = build_model(
        model_impl=real_model_impl,
        backbone=backbone,
        embed_dim=embed_dim,
        fft_bins=fft_bins,
        agg=agg,
        fusion=real_fusion,
        use_log_gsd=use_log_gsd,
        device=device,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    info = {
        "ckpt": str(ckpt_path),
        "ckpt_type": real_ckpt_type,
        "model_impl": real_model_impl,
        "detector_state_source": detector_state_source,
        "num_loaded_tensors": len(state),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "backbone": backbone,
        "embed_dim": embed_dim,
        "fft_bins": fft_bins,
        "agg": agg,
        "fusion": real_fusion,
        "use_log_gsd": use_log_gsd,
    }
    return model, info


def make_loader(args: argparse.Namespace, k: int) -> DataLoader:
    dataset = EvalFMOWArrowDataset(
        index_json=Path(args.index),
        crop=args.crop,
        k=k,
        train=False,
        train_patch_mode="four_corners",
        fixed_mode=args.fixed_mode,
        use_log_gsd=args.use_log_gsd,
        seed=args.seed,
        small_img_threshold=args.small_img_threshold,
        shard_cache=args.shard_cache,
        use_fp16_patches=args.use_fp16_patches,
        aug=False,
        patch_impl=args.resolved_patch_impl,
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda") and torch.cuda.is_available()),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )


def safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_log_gsd: bool,
    thresholds: Iterable[float],
    amp: bool,
    predictions_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    thresholds = tuple(float(t) for t in thresholds)

    n = 0
    mae_sum = 0.0
    mse_sum = 0.0
    rel_sum = 0.0
    log_sum = 0.0
    hit = {t: 0 for t in thresholds}
    abs_errors: List[torch.Tensor] = []
    rel_errors: List[torch.Tensor] = []

    csv_file = None
    writer = None
    if predictions_csv is not None:
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = predictions_csv.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["sample_id", "target_gsd", "pred_gsd", "abs_error", "rel_error"])

    try:
        for x, y, sample_ids in tqdm(loader, desc="Eval GSD", unit="batch"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(amp and device.type == "cuda"),
            ):
                pred = model(x)
            pred = pred.float()

            if use_log_gsd:
                pred_gsd = torch.exp(pred).clamp_min(1e-8)
                target_gsd = torch.exp(y).clamp_min(1e-8)
            else:
                pred_gsd = pred.clamp_min(1e-8)
                target_gsd = y.clamp_min(1e-8)

            err = (pred_gsd - target_gsd).abs().flatten()
            rel = err / target_gsd.flatten().clamp_min(1e-8)
            log_err = (torch.log(pred_gsd) - torch.log(target_gsd)).abs().flatten()

            batch_n = int(err.numel())
            n += batch_n
            mae_sum += safe_float(err.sum())
            mse_sum += safe_float((err * err).sum())
            rel_sum += safe_float(rel.sum())
            log_sum += safe_float(log_err.sum())
            for t in thresholds:
                hit[t] += int((err < t).sum().item())

            abs_errors.append(err.detach().cpu())
            rel_errors.append(rel.detach().cpu())

            if writer is not None:
                pred_list = pred_gsd.flatten().detach().cpu().tolist()
                target_list = target_gsd.flatten().detach().cpu().tolist()
                err_list = err.detach().cpu().tolist()
                rel_list = rel.detach().cpu().tolist()
                for sid, tgt, prd, ae, re in zip(sample_ids, target_list, pred_list, err_list, rel_list):
                    writer.writerow([sid, f"{tgt:.10g}", f"{prd:.10g}", f"{ae:.10g}", f"{re:.10g}"])
    finally:
        if csv_file is not None:
            csv_file.close()

    if n == 0:
        raise RuntimeError("No samples were evaluated.")

    abs_all = torch.cat(abs_errors)
    rel_all = torch.cat(rel_errors)
    metrics = {
        "num_samples": n,
        "mae": mae_sum / n,
        "rmse": math.sqrt(mse_sum / n),
        "median_abs_error": float(torch.quantile(abs_all, 0.5).item()),
        "p90_abs_error": float(torch.quantile(abs_all, 0.9).item()),
        "mean_relative_error": rel_sum / n,
        "median_relative_error": float(torch.quantile(rel_all, 0.5).item()),
        "mape_percent": 100.0 * rel_sum / n,
        "mean_abs_log_error": log_sum / n,
        "acc_abs_error": {f"lt_{t:g}": hit[t] / n for t in thresholds},
    }
    return metrics


def print_report(label: str, info: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    print(f"\n[{label}]")
    print(f"ckpt: {info['ckpt']}")
    print(
        "loaded: "
        f"ckpt_type={info['ckpt_type']} model_impl={info['model_impl']} "
        f"fusion={info['fusion']} tensors={info['num_loaded_tensors']}")
    print(f"missing={len(info['missing'])} unexpected={len(info['unexpected'])}")
    if info["missing"]:
        print(f"missing first 10: {info['missing'][:10]}")
    if info["unexpected"]:
        print(f"unexpected first 10: {info['unexpected'][:10]}")
    print(
        f"samples={metrics['num_samples']} "
        f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
        f"MedAE={metrics['median_abs_error']:.6f} P90AE={metrics['p90_abs_error']:.6f}")
    print(
        f"MAPE={metrics['mape_percent']:.3f}% "
        f"MedRE={100.0 * metrics['median_relative_error']:.3f}% "
        f"MeanAbsLog={metrics['mean_abs_log_error']:.6f}")
    acc = " ".join(
        f"Acc@{k.replace('lt_', '')}={v * 100:.2f}%"
        for k, v in metrics["acc_abs_error"].items())
    print(acc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone or detector-finetuned GSD predictor on an Arrow index.")
    parser.add_argument("--index", required=True, help="fMoW Arrow index.json for evaluation")
    parser.add_argument("--ckpt", required=True, help="detector epoch_*.pth or standalone GSD .pt/.pth")
    parser.add_argument(
        "--ckpt-type",
        default="auto",
        choices=["auto", "detector", "standalone"],
        help="auto detects detector checkpoints by gsd_predictor.* keys")
    parser.add_argument(
        "--detector-state-source",
        default="auto",
        help="state_dict / ema_state_dict / model / auto for detector checkpoints")
    parser.add_argument(
        "--model-impl",
        default="auto",
        choices=["auto", "detector", "standalone"],
        help=(
            "detector uses the exact GSD module embedded in DEIMV2; "
            "standalone uses the original GSD training module."))
    parser.add_argument(
        "--baseline-ckpt",
        default="",
        help="optional original standalone GSD checkpoint to evaluate with the same loader")

    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--fft-bins", type=int, default=64)
    parser.add_argument("--agg", default="attn", choices=["mean", "max", "attn"])
    parser.add_argument("--fusion", default="auto", choices=["auto", "concat_mlp", "gated"])

    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="patch count. 0 means auto: detector ckpt uses 5, standalone ckpt uses 4.")
    parser.add_argument(
        "--patch-impl",
        default="auto",
        choices=["auto", "detector", "standalone"],
        help="patch extraction implementation. auto follows the selected model implementation.")
    parser.add_argument("--fixed-mode", default="five", choices=["center", "five"])
    parser.add_argument("--small-img-threshold", type=int, default=512)
    parser.add_argument("--use-log-gsd", action="store_true")
    parser.add_argument("--use-fp16-patches", action="store_true")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shard-cache", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--thresholds",
        default="2,1.5,1.2,1,0.5,0.2,0.1,0.05,0.02",
        help="comma separated absolute-error thresholds in GSD units")
    parser.add_argument("--out-json", default="", help="optional metrics json path")
    parser.add_argument("--pred-csv", default="", help="optional per-sample predictions csv path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    index_path = Path(args.index)
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    probe_ckpt = torch.load(str(ckpt_path), map_location="cpu")
    real_ckpt_type = infer_ckpt_type(probe_ckpt, args.detector_state_source) if args.ckpt_type == "auto" else args.ckpt_type
    k = args.k if args.k > 0 else (5 if real_ckpt_type == "detector" else 4)
    resolved_model_impl = (
        ("detector" if real_ckpt_type == "detector" else "standalone")
        if args.model_impl == "auto" else args.model_impl)
    args.resolved_patch_impl = (
        resolved_model_impl if args.patch_impl == "auto" else args.patch_impl)

    print(f"Eval index: {index_path}")
    print(
        f"Device: {device} | k={k} | patch_impl={args.resolved_patch_impl} "
        f"| use_log_gsd={args.use_log_gsd}")
    loader = make_loader(args, k=k)

    model, info = load_gsd_model(
        ckpt_path=ckpt_path,
        ckpt_type=args.ckpt_type,
        detector_state_source=args.detector_state_source,
        model_impl=args.model_impl,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        fft_bins=args.fft_bins,
        agg=args.agg,
        fusion=args.fusion,
        use_log_gsd=args.use_log_gsd,
        device=device,
        ckpt_obj=probe_ckpt,
    )
    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        use_log_gsd=args.use_log_gsd,
        thresholds=thresholds,
        amp=args.amp,
        predictions_csv=Path(args.pred_csv) if args.pred_csv else None,
    )
    print_report("target", info, metrics)

    results = {
        "target": {"load_info": info, "metrics": metrics},
        "loader": {"index": str(index_path), "k": k, "patch_impl": args.resolved_patch_impl},
    }

    if args.baseline_ckpt:
        baseline_path = Path(args.baseline_ckpt)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
        baseline_model, baseline_info = load_gsd_model(
            ckpt_path=baseline_path,
            ckpt_type="standalone",
            detector_state_source=args.detector_state_source,
            model_impl="standalone" if args.model_impl == "auto" else args.model_impl,
            backbone=args.backbone,
            embed_dim=args.embed_dim,
            fft_bins=args.fft_bins,
            agg=args.agg,
            fusion=args.fusion,
            use_log_gsd=args.use_log_gsd,
            device=device,
        )
        baseline_metrics = evaluate(
            model=baseline_model,
            loader=loader,
            device=device,
            use_log_gsd=args.use_log_gsd,
            thresholds=thresholds,
            amp=args.amp,
            predictions_csv=None,
        )
        print_report("baseline", baseline_info, baseline_metrics)
        results["baseline"] = {"load_info": baseline_info, "metrics": baseline_metrics}

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved metrics to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
