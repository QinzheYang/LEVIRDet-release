# train_gsd_resnet_arrow_ddp.py
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision.transforms import functional as TF

import pyarrow as pa
import pyarrow.ipc as ipc

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# -------------------------
# DDP utils
# -------------------------
def ddp_init(backend: str = "nccl"):
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Windows 一般没有 nccl，用 gloo 更稳
        if backend == "nccl" and not dist.is_nccl_available():
            backend = "gloo"

        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
        return True, rank, world_size, local_rank, backend

    return False, 0, 1, 0, backend


def is_main_process(rank: int) -> bool:
    return rank == 0


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_all_reduce_dict(metrics: Dict[str, float], device):
    # 对 dict 里的标量做 all_reduce(sum)
    if not (dist.is_available() and dist.is_initialized()):
        return metrics
    keys = sorted(metrics.keys())
    vals = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    return {k: float(v.item()) for k, v in zip(keys, vals)}


# -------------------------
# logging
# -------------------------
class SimpleLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.f.write(f"[{ts}] {msg}\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# patch helpers
# -------------------------
def corner_positions(w: int, h: int, crop: int) -> List[Tuple[int, int]]:
    max_x = max(0, w - crop)
    max_y = max(0, h - crop)
    return [(0, 0), (max_x, 0), (0, max_y), (max_x, max_y)]


def center_position(w: int, h: int, crop: int) -> Tuple[int, int]:
    max_x = max(0, w - crop)
    max_y = max(0, h - crop)
    return (max_x // 2, max_y // 2)


def center_biased_positions(w: int, h: int, crop: int, k: int) -> List[Tuple[int, int]]:
    """
    小图阈值触发时：中心优先（允许重叠），减少黑边占比。
    允许超界坐标，PIL.crop 会补黑，但中心策略黑边最少。
    """
    cx = w // 2
    cy = h // 2
    base_x = cx - crop // 2
    base_y = cy - crop // 2

    if k <= 1:
        return [(base_x, base_y)]

    step = max(1, crop // 8)
    offsets = [
        (0, 0),
        (-step, 0),
        (step, 0),
        (0, -step),
        (0, step),
        (-step, -step),
        (step, -step),
        (-step, step),
        (step, step),
    ]
    pos = []
    for dx, dy in offsets:
        pos.append((base_x + dx, base_y + dy))
        if len(pos) >= k:
            break
    if len(pos) < k:
        pos += [pos[-1]] * (k - len(pos))
    return pos


def random_positions(w: int, h: int, crop: int, k: int) -> List[Tuple[int, int]]:
    max_x = max(0, w - crop)
    max_y = max(0, h - crop)
    xs = torch.randint(0, max_x + 1, (k,), dtype=torch.int64).tolist()
    ys = torch.randint(0, max_y + 1, (k,), dtype=torch.int64).tolist()
    return list(zip(xs, ys))


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[:, None, None]
    return (x - mean) / std


def patches_from_positions(
    im: Image.Image,
    crop: int,
    positions: List[Tuple[int, int]],
    use_fp16: bool = False,
    rng: Optional[random.Random] = None,
    aug: bool = False,
    flip_p: float = 0.5,
    rot_choices=(0, 90, 180),  # 只含 0/90/180
) -> torch.Tensor:
    patches = []
    for i, (x, y) in enumerate(positions):
        patch = im.crop((x, y, x + crop, y + crop))

        # ----- augmentation on PIL patch -----
        if aug and (rng is not None):
            # H flip
            if rng.random() < flip_p:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
            # V flip
            if rng.random() < flip_p:
                patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
            # rotate
            ang = rng.choice(rot_choices)
            if ang == 90:
                patch = patch.transpose(Image.ROTATE_90)     # 90° CCW
            elif ang == 180:
                patch = patch.transpose(Image.ROTATE_180)

        t = TF.to_tensor(patch)  # float32
        t = imagenet_normalize(t)
        if use_fp16:
            t = t.half()
        patches.append(t)

    return torch.stack(patches, dim=0)


def fixed_patches_tensor(im: Image.Image, crop: int, k: int, mode: str, small_thr: int, use_fp16: bool = False) -> torch.Tensor:
    w, h = im.size
    if w < small_thr or h < small_thr:
        pos = center_biased_positions(w, h, crop, k)
        return patches_from_positions(im, crop, pos, use_fp16=use_fp16)

    max_x = max(0, w - crop)
    max_y = max(0, h - crop)
    cx = max_x // 2
    cy = max_y // 2
    if mode == "center":
        pos = [(cx, cy)]
    else:
        pos = [(0, 0), (max_x, 0), (0, max_y), (max_x, max_y), (cx, cy)]
    if len(pos) >= k:
        pos = pos[:k]
    else:
        pos = pos + [pos[-1]] * (k - len(pos))
    return patches_from_positions(im, crop, pos, use_fp16=use_fp16)


# -------------------------
# Arrow shard reader (LRU)
# -------------------------
@dataclass
class ShardInfo:
    file: str
    rows: int


class ArrowShardCache:
    """
    每个 worker/process 一个 cache：
    - 只缓存少量 shard 的 table，避免频繁打开/解析
    - Arrow IPC 支持 memory-map，读取二进制列很快
    """
    def __init__(self, shards_dir: Path, max_cached: int = 2):
        self.shards_dir = Path(shards_dir)
        self.max_cached = max_cached
        self.cache: "OrderedDict[str, pa.Table]" = OrderedDict()

    def get_table(self, shard_name: str) -> pa.Table:
        if shard_name in self.cache:
            tbl = self.cache.pop(shard_name)
            self.cache[shard_name] = tbl
            return tbl

        shard_path = self.shards_dir / shard_name
        # memory-map
        mm = pa.memory_map(str(shard_path), "r")
        reader = ipc.RecordBatchFileReader(mm)
        tbl = reader.read_all()

        self.cache[shard_name] = tbl
        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return tbl


class FMOWArrowDataset(Dataset):
    def __init__(
        self,
        index_json: Path,
        crop: int = 224,
        k: int = 4,
        train: bool = True,
        train_patch_mode: str = "four_corners",
        fixed_mode: str = "five",
        use_log_gsd: bool = False,
        seed: int = 42,
        small_img_threshold: int = 512,
        shard_cache: int = 2,
        use_fp16_patches: bool = False,  # 减半 CPU->GPU 传输与共享内存压力
        aug: bool = True,
        flip_p: float = 0.5,
    ):
        self.index_json = Path(index_json)
        idx = json.loads(self.index_json.read_text(encoding="utf-8"))
        self.out_dir = Path(idx["out_dir"])
        self.shards_dir = self.out_dir / "shards"
        self.shards: List[ShardInfo] = [ShardInfo(s["file"], int(s["rows"])) for s in idx["shards"]]
        self.total = int(idx["total_rows"])

        self.crop = int(crop)
        self.k = int(k)
        self.train = bool(train)
        self.train_patch_mode = train_patch_mode
        self.fixed_mode = fixed_mode
        self.use_log_gsd = use_log_gsd
        self.seed = int(seed)
        self.small_thr = int(small_img_threshold)
        self.use_fp16_patches = bool(use_fp16_patches)

        # prefix sums for shard lookup
        self.cum = []
        self.aug = bool(aug) and self.train
        self.flip_p = float(flip_p)
        self.rot_choices = (0, 90, 180)
        c = 0
        for s in self.shards:
            c += s.rows
            self.cum.append(c)

        self._epoch = 0
        self._pos_cache: Dict[int, List[Tuple[int, int]]] = {}

        self._cache = ArrowShardCache(self.shards_dir, max_cached=int(shard_cache))

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._pos_cache.clear()

    def __len__(self):
        return self.total

    def _locate(self, idx: int) -> Tuple[str, int]:
        # idx -> (shard_name, offset)
        # 二分查 cum
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard_i = lo
        prev = self.cum[shard_i - 1] if shard_i > 0 else 0
        off = idx - prev
        return self.shards[shard_i].file, off

    def _get_train_positions(self, global_idx: int, w: int, h: int) -> List[Tuple[int, int]]:
        # 以 512 为阈值：小图中心优先
        if w < self.small_thr or h < self.small_thr:
            return center_biased_positions(w, h, self.crop, self.k)

        if self.train_patch_mode == "four_corners":
            pos = corner_positions(w, h, self.crop)
            if self.k <= 4:
                return pos[: self.k]
            cx, cy = center_position(w, h, self.crop)
            return pos + [(cx, cy)] * (self.k - 4)

        if self.train_patch_mode == "cached_random":
            if global_idx in self._pos_cache:
                return self._pos_cache[global_idx]
            rng = random.Random(self.seed + self._epoch * 1000003 + global_idx)
            max_x = max(0, w - self.crop)
            max_y = max(0, h - self.crop)
            pos = [(rng.randint(0, max_x), rng.randint(0, max_y)) for _ in range(self.k)]
            self._pos_cache[global_idx] = pos
            return pos

        return random_positions(w, h, self.crop, self.k)

    def __getitem__(self, idx: int):
        # Arrow 里取一行
        shard_name, off = self._locate(int(idx))
        tbl = self._cache.get_table(shard_name)

        # 列读取（取单元素）
        img_name = tbl["img_name"][off].as_py()
        rel_folder = tbl["rel_folder"][off].as_py()
        gsd = float(tbl["gsd"][off].as_py())
        w = int(tbl["width"][off].as_py())
        h = int(tbl["height"][off].as_py())
        img_bytes = tbl["img_bytes"][off].as_py()  # bytes
        # meta_json = tbl["meta_json"][off].as_py()  # 若你要用也可以读

        # bytes -> PIL
        import io

        buf = img_bytes
        # 有些版本 as_py() 已经是 bytes；有些可能是 pyarrow.Buffer / memoryview
        if isinstance(buf, pa.Buffer):
            buf = buf.to_pybytes()
        elif isinstance(buf, memoryview):
            buf = buf.tobytes()
        elif not isinstance(buf, (bytes, bytearray)):
            buf = bytes(buf)

        im = Image.open(io.BytesIO(buf))
        im = im.convert("RGB")

        if self.train:
            pos = self._get_train_positions(int(idx), im.size[0], im.size[1])

            # 让增强在 DDP/多 worker 下也尽量可复现：由 (seed, epoch, idx) 决定
            rng = random.Random(self.seed + self._epoch * 1000003 + int(idx))

            x = patches_from_positions(
                im, self.crop, pos,
                use_fp16=self.use_fp16_patches,
                rng=rng,
                aug=self.aug,
                flip_p=self.flip_p,
                rot_choices=self.rot_choices
            )
        else:
            x = fixed_patches_tensor(
                im, self.crop, self.k, mode=self.fixed_mode, small_thr=self.small_thr,
                use_fp16=self.use_fp16_patches
            )

        y = gsd
        if self.use_log_gsd:
            y = math.log(max(y, 1e-8))
        y = torch.tensor([y], dtype=torch.float32)

        # 返回一个可追踪的 id
        sample_id = f"{rel_folder}/{img_name}"
        return x, y, sample_id


# -------------------------
# model
# -------------------------
class ResNetPatchRegressor(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True, head_hidden=512, agg="mean", positive=True):
        super().__init__()
        assert agg in ("mean", "max")
        self.agg = agg
        self.positive = positive

        if backbone_name == "resnet18":
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet34":
            net = torchvision.models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 512
        elif backbone_name == "resnet50":
            net = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(backbone_name)

        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        b, k, c, h, w = x.shape
        x = x.view(b * k, c, h, w)
        feat = self.backbone(x).flatten(1)
        feat = feat.view(b, k, -1)
        feat = feat.mean(1) if self.agg == "mean" else feat.max(1).values
        out = self.head(feat)
        if self.positive:
            out = F.softplus(out) + 1e-6
        return out


# -------------------------
# train / eval (DDP)
# -------------------------
@torch.no_grad()
def evaluate_ddp(model, loader, device, use_log_gsd: bool, thresholds, rank: int, desc="Val"):
    model.eval()

    mae_sum = 0.0
    mse_sum = 0.0
    n = 0.0
    hit = {t: 0.0 for t in thresholds}

    it = loader
    if is_main_process(rank):
        it = tqdm(loader, desc=desc, unit="batch", leave=False, mininterval=0.5)

    for x, y, _ in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        if use_log_gsd:
            pred_gsd = torch.exp(pred)
            y_gsd = torch.exp(y)
        else:
            pred_gsd = pred
            y_gsd = y

        err = (pred_gsd - y_gsd).abs()  # (B,1)
        mae_sum += err.sum().item()
        mse_sum += (err ** 2).sum().item()
        bs = y.shape[0]
        n += float(bs)

        err1 = err.squeeze(1)
        for t in thresholds:
            hit[t] += float((err1 < t).sum().item())

        if is_main_process(rank) and n > 0:
            mae_now = mae_sum / n
            rmse_now = math.sqrt(mse_sum / n)
            it.set_postfix(mae=f"{mae_now:.4f}", rmse=f"{rmse_now:.4f}")

    # all_reduce across ranks
    metrics = {"mae_sum": mae_sum, "mse_sum": mse_sum, "n": n}
    for t in thresholds:
        metrics[f"hit_{t}"] = hit[t]
    metrics = ddp_all_reduce_dict(metrics, device=device)

    mae = metrics["mae_sum"] / max(metrics["n"], 1.0)
    rmse = math.sqrt(metrics["mse_sum"] / max(metrics["n"], 1.0))
    acc = {t: metrics[f"hit_{t}"] / max(metrics["n"], 1.0) for t in thresholds}
    return mae, rmse, acc


def train_one_epoch_ddp(model, loader, optimizer, scaler, device, loss_type: str,
                        log_every: int, logger: Optional[SimpleLogger], epoch: int, rank: int):
    model.train()
    running = 0.0
    count = 0.0

    it = loader
    if is_main_process(rank):
        it = tqdm(loader, desc=f"Train E{epoch}", unit="batch", leave=False, mininterval=0.5)

    for step, (x, y, _) in enumerate(it, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y) if loss_type == "smoothl1" else F.mse_loss(pred, y)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = y.shape[0]
        running += loss.item() * bs
        count += bs
        avg = running / max(count, 1.0)

        if is_main_process(rank):
            it.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

            if logger is not None and (step % log_every == 0):
                logger.log(f"epoch={epoch} step={step} loss={loss.item():.6f} avg_loss={avg:.6f}")

    # 返回本 rank 的平均 loss，再 all_reduce
    loss_metrics = {"loss_sum": running, "n": count}
    loss_metrics = ddp_all_reduce_dict(loss_metrics, device=device)
    return loss_metrics["loss_sum"] / max(loss_metrics["n"], 1.0)


def save_checkpoint(path: Path, ckpt: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Bad checkpoint: {path}")
    return ckpt


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def main():
    ap = argparse.ArgumentParser("Train GSD regressor from Arrow shards (DDP)")
    ap.add_argument("--train_index", required=True, help=r"e.g. E:\fmow_arrow\train\index.json")
    ap.add_argument("--val_index", required=True, help=r"e.g. E:\fmow_arrow\test_demo\index.json")

    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--head_hidden", type=int, default=512)
    ap.add_argument("--agg", default="mean", choices=["mean", "max"])

    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fixed_mode", default="five", choices=["center", "five"])
    ap.add_argument("--train_patch_mode", default="four_corners", choices=["random", "four_corners", "cached_random"])
    ap.add_argument("--small_img_threshold", type=int, default=512)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16, help="per-GPU batch size（Windows建议小一点）")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)

    ap.add_argument("--loss", default="smoothl1", choices=["smoothl1", "mse"])
    ap.add_argument("--use_log_gsd", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_dir", default="./checkpoints_gsd_arrow_ddp")
    ap.add_argument("--resume", default="")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--backend", default="nccl", help="ddp backend: nccl or gloo")
    ap.add_argument("--shard_cache", type=int, default=2, help="每个worker缓存多少个arrow shard")
    ap.add_argument("--use_fp16_patches", action="store_true", help="把patch张量转成float16（减内存压力）")

    ap.add_argument("--no_aug", action="store_true", help="disable random flip/rot(90/180) augmentation")
    ap.add_argument("--flip_p", type=float, default=0.5, help="probability for hflip/vflip (train only)")
    args = ap.parse_args()

    ddp, rank, world_size, local_rank, backend = ddp_init(args.backend)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed, rank=rank)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_logger = SimpleLogger(save_dir / "train.log") if is_main_process(rank) else None
    val_logger = SimpleLogger(save_dir / "val.log") if is_main_process(rank) else None
    metrics_jsonl = save_dir / "metrics.jsonl"

    if is_main_process(rank):
        print(f"DDP={ddp} backend={backend} world_size={world_size} local_rank={local_rank}")
        print("Device:", device, flush=True)

    train_ds = FMOWArrowDataset(
        index_json=Path(args.train_index),
        crop=args.crop,
        k=args.k,
        train=True,
        train_patch_mode=args.train_patch_mode,
        fixed_mode=args.fixed_mode,
        use_log_gsd=args.use_log_gsd,
        seed=args.seed,
        small_img_threshold=args.small_img_threshold,
        shard_cache=args.shard_cache,
        use_fp16_patches=args.use_fp16_patches,
        aug=(not args.no_aug),
        flip_p=args.flip_p,
    )
    val_ds = FMOWArrowDataset(
        index_json=Path(args.val_index),
        crop=args.crop,
        k=args.k,
        train=False,
        train_patch_mode="four_corners",
        fixed_mode=args.fixed_mode,
        use_log_gsd=args.use_log_gsd,
        seed=args.seed,
        small_img_threshold=args.small_img_threshold,
        shard_cache=args.shard_cache,
        use_fp16_patches=args.use_fp16_patches,
        aug=False,
        flip_p=args.flip_p,
    )

    if is_main_process(rank):
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = ResNetPatchRegressor(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        head_hidden=args.head_hidden,
        agg=args.agg,
        positive=(not args.use_log_gsd),
    ).to(device)

    if ddp and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    elif ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    thresholds = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
    best_rmse = float("inf")
    start_epoch = 1

    # resume
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), device=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"], strict=True)
        try:
            optimizer.load_state_dict(ckpt.get("optimizer", {}))
        except Exception:
            pass
        try:
            scheduler.load_state_dict(ckpt.get("scheduler", {}))
        except Exception:
            pass
        if scaler is not None and ckpt.get("scaler", None) is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass

        best_rmse = float(ckpt.get("best_rmse", best_rmse))
        last_epoch = int(ckpt.get("epoch", 0))
        start_epoch = last_epoch + 1

        ddp_barrier()
        # resume check（各 rank 都跑一遍，最后 all_reduce）
        mae0, rmse0, acc0 = evaluate_ddp(model, val_loader, device, args.use_log_gsd, thresholds, rank, desc="Val (resume check)")
        if is_main_process(rank):
            acc_str0 = " ".join([f"Acc@{t:g}={acc0[t]*100:.2f}%" for t in thresholds])
            msg0 = f"[ResumeCheck] epoch={last_epoch} MAE={mae0:.6f} RMSE={rmse0:.6f} {acc_str0}"
            print(msg0, flush=True)
            if val_logger: val_logger.log(msg0)
            append_jsonl(metrics_jsonl, {"type":"resume_check","epoch":last_epoch,"mae":mae0,"rmse":rmse0,"acc":acc0})

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)

        lr_now = optimizer.param_groups[0]["lr"]
        if is_main_process(rank):
            print(f"\nEpoch {epoch}/{args.epochs} | lr={lr_now:.3e}", flush=True)
            if train_logger: train_logger.log(f"EpochStart epoch={epoch} lr={lr_now:.8f}")

        train_loss = train_one_epoch_ddp(
            model, train_loader, optimizer, scaler, device,
            loss_type=args.loss, log_every=args.log_every,
            logger=train_logger, epoch=epoch, rank=rank
        )

        mae, rmse, acc = evaluate_ddp(model, val_loader, device, args.use_log_gsd, thresholds, rank, desc=f"Val E{epoch}")

        if is_main_process(rank):
            acc_str = " ".join([f"Acc@{t:g}={acc[t]*100:.2f}%" for t in thresholds])
            msg = f"epoch={epoch} train_loss={train_loss:.6f} val_mae={mae:.6f} val_rmse={rmse:.6f} {acc_str}"
            print(msg, flush=True)
            if val_logger: val_logger.log(msg)
            append_jsonl(metrics_jsonl, {"type":"epoch","epoch":epoch,"lr":lr_now,"train_loss":train_loss,"val_mae":mae,"val_rmse":rmse,"acc":acc})

        scheduler.step()

        # save (rank0)
        if is_main_process(rank):
            ckpt = {
                "epoch": epoch,
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_rmse": best_rmse,
                "args": vars(args),
            }
            save_checkpoint(save_dir / "last.pt", ckpt)
            if rmse < best_rmse:
                best_rmse = rmse
                ckpt["best_rmse"] = best_rmse
                save_checkpoint(save_dir / "best.pt", ckpt)
                if val_logger: val_logger.log(f"SavedBest epoch={epoch} best_rmse={best_rmse:.6f}")
                print(f"Saved best.pt (best_rmse={best_rmse:.6f})", flush=True)

        ddp_barrier()

    if is_main_process(rank):
        print("\nTraining finished.", flush=True)
        print(f"Best RMSE: {best_rmse:.6f}", flush=True)
        print(f"Checkpoints in: {save_dir}", flush=True)
        if train_logger: train_logger.close()
        if val_logger: val_logger.close()

    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
