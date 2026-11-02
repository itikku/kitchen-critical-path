"""Train two classifiers on cached frame embeddings:

- `frame_linear`: a linear layer on one frame (baseline, no time context);
- `temporal_head`: a small 1D-conv network over an 8-frame window (~4.3 s) centered on the frame.

Model selection (configuration and number of epochs) uses the validation people (s4) only;
the test splits T1 and T2 are evaluated once, after selection."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from kcp.perception.data import Split, Video, assign_split
from kcp.perception.labels import CLASSES

WINDOW = 8
SEED = 0
# Feature set -> cache folder. Run 2 used mnv3_full; Phase 2a.2 adds the two cropped sets.
FEATURE_DIRS = {
    "mnv3_full": "mobilenetv3_large_100",
    "mnv3_crop": "mnv3_crop",
    "s3d_crop": "s3d_crop",
    "s3d_crop16": "s3d_crop16",  # Phase 2a.3: a clip every 16 frames
    "s3d_wrist": "s3d_wrist",  # Phase 2a.3: crop around the wrists
}


@dataclass(frozen=True)
class TrainConfig:
    model: str
    epochs: int = 10
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-2
    hidden: int = 256
    dropout: float = 0.2
    class_weight_power: float = 0.5  # weight = (1 / class frequency) ** power, mean-normalized
    center: bool = False  # subtract each video's running-mean embedding (causal; works live)
    features: str = "mnv3_full"

    @property
    def name(self) -> str:
        base = self.model + ("_centered" if self.center else "")
        return base if self.features == "mnv3_full" else f"{self.features}__{base}"


# Run 2 configurations, fixed before any run-2 result was seen. The temporal head gets stronger
# regularization because run 1 overfit (train loss 0.05, test macro-F1 below the frame baseline).
CONFIGS: tuple[TrainConfig, ...] = (
    TrainConfig(model="frame_linear"),
    TrainConfig(model="frame_linear", center=True),
    TrainConfig(model="temporal_head", dropout=0.5, weight_decay=0.05),
    TrainConfig(model="temporal_head", dropout=0.5, weight_decay=0.05, center=True),
)

# Phase 2a.2 configurations (IMPLEMENTATION_PLAN.md), fixed before any 2a.2 result: for each
# feature set, the frame baseline and run 2's selected configuration.
CONFIGS_2A2: tuple[TrainConfig, ...] = tuple(
    cfg
    for features in ("mnv3_full", "mnv3_crop", "s3d_crop")
    for cfg in (
        TrainConfig(model="frame_linear", features=features),
        TrainConfig(
            model="temporal_head", dropout=0.5, weight_decay=0.05, center=True, features=features
        ),
    )
)


def _pair(features: str) -> tuple[TrainConfig, TrainConfig]:
    return (
        TrainConfig(model="frame_linear", features=features),
        TrainConfig(
            model="temporal_head", dropout=0.5, weight_decay=0.05, center=True, features=features
        ),
    )


# Phase 2a.3: the same two configurations on each new feature set.
CONFIGS_2A3: tuple[TrainConfig, ...] = (*_pair("s3d_crop16"), *_pair("s3d_wrist"))


@dataclass
class SplitData:
    emb: np.ndarray  # (N, D) float16, all frames of all videos in the split
    y: np.ndarray  # (N,) int64
    video_start: np.ndarray  # (N,) first global index of the frame's video
    video_end: np.ndarray  # (N,) last global index of the frame's video
    keys: list[str]
    frame: np.ndarray  # (N,) source video frame index of each step (15 fps)


def video_of(key: str) -> Video:
    person, camera, recipe = key.split("_", 2)
    return Video(path=Path(key), person=person, camera=camera, recipe=recipe)


def load_split(cache_dir: Path, split: Split, features: str = "mnv3_full") -> SplitData:
    files = sorted((cache_dir / FEATURE_DIRS[features]).glob("*.npz"))
    files = [
        f for f in files if not f.stem.endswith(".tmp") and assign_split(video_of(f.stem)) == split
    ]
    embs, ys, starts, ends, keys, frames = [], [], [], [], [], []
    offset = 0
    for f in files:
        d = np.load(f)
        n = len(d["y"])
        if n == 0:
            continue
        embs.append(d["emb"])
        ys.append(d["y"].astype(np.int64))
        starts.append(np.full(n, offset))
        ends.append(np.full(n, offset + n - 1))
        keys.append(f.stem)
        frames.append(d["frame"].astype(np.int64))
        offset += n
    return SplitData(
        emb=np.concatenate(embs),
        y=np.concatenate(ys),
        video_start=np.concatenate(starts),
        video_end=np.concatenate(ends),
        keys=keys,
        frame=np.concatenate(frames),
    )


def centered(data: SplitData) -> SplitData:
    """Subtract, from each frame, the mean embedding of its video up to and including that frame.
    This removes most of the kitchen's static appearance and keeps what changes."""
    out = np.empty_like(data.emb)  # float16; only one video at a time is held in float32
    for start in np.unique(data.video_start):
        end = int(data.video_end[start]) + 1
        seg = data.emb[start:end].astype(np.float32)
        running = np.cumsum(seg, axis=0) / np.arange(1, len(seg) + 1)[:, None]
        out[start:end] = seg - running
    return SplitData(out, data.y, data.video_start, data.video_end, data.keys, data.frame)


def feature_stats(emb: np.ndarray, chunk: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and std, computed in chunks to keep memory low."""
    total = np.zeros(emb.shape[1])
    total_sq = np.zeros(emb.shape[1])
    for i in range(0, len(emb), chunk):
        part = emb[i : i + chunk].astype(np.float64)
        total += part.sum(0)
        total_sq += (part**2).sum(0)
    mean = total / len(emb)
    std = np.sqrt(np.maximum(total_sq / len(emb) - mean**2, 0)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def prepare(data: SplitData, cfg: TrainConfig) -> SplitData:
    return centered(data) if cfg.center else data


@dataclass
class DeviceSplit:
    """A split held on the GPU: standardized fp16 embeddings, so windows are gathered there."""

    emb: torch.Tensor  # (N, D) float16
    start: torch.Tensor  # (N,) int64
    end: torch.Tensor  # (N,) int64
    y: torch.Tensor  # (N,) int64


def to_device(data: SplitData, mean: np.ndarray, std: np.ndarray) -> DeviceSplit:
    dev = device()
    chunks = [
        torch.from_numpy((data.emb[i : i + 50_000].astype(np.float32) - mean) / std).half().to(dev)
        for i in range(0, len(data.y), 50_000)
    ]
    return DeviceSplit(
        emb=torch.cat(chunks),
        start=torch.from_numpy(data.video_start).to(dev),
        end=torch.from_numpy(data.video_end).to(dev),
        y=torch.from_numpy(data.y).to(dev),
    )


def windows(ds: DeviceSplit, idx: torch.Tensor) -> torch.Tensor:
    """(B, WINDOW, D) windows centered on each frame; edges repeat the first/last frame."""
    offsets = torch.arange(WINDOW, device=idx.device) - WINDOW // 2
    g = idx[:, None] + offsets[None, :]
    g = torch.minimum(torch.maximum(g, ds.start[idx][:, None]), ds.end[idx][:, None])
    return ds.emb[g].float()


class FrameLinear(nn.Module):
    def __init__(self, dim: int, classes: int, cfg: TrainConfig) -> None:
        super().__init__()
        self.out = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(dim, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D); use the center frame
        logits: torch.Tensor = self.out(x[:, WINDOW // 2])
        return logits


class TemporalHead(nn.Module):
    def __init__(self, dim: int, classes: int, cfg: TrainConfig) -> None:
        super().__init__()
        h = cfg.hidden
        self.proj = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(dim, h))
        self.conv = nn.Sequential(
            nn.Conv1d(h, h, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(h, h, 3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.out = nn.Linear(h, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        z = self.proj(x).transpose(1, 2)
        logits: torch.Tensor = self.out(self.conv(z).mean(dim=2))
        return logits


MODELS: dict[str, type[nn.Module]] = {"frame_linear": FrameLinear, "temporal_head": TemporalHead}


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(cfg: TrainConfig, dim: int) -> nn.Module:
    return MODELS[cfg.model](dim, len(CLASSES), cfg)


def predict_proba(
    model: nn.Module, data: SplitData, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """(N, classes) softmax probabilities."""
    ds = to_device(data, mean, std)
    model.eval()
    probs = []
    with torch.inference_mode():
        for start in range(0, len(data.y), 4096):
            idx = torch.arange(start, min(start + 4096, len(data.y)), device=ds.emb.device)
            probs.append(model(windows(ds, idx)).softmax(1))
    out: np.ndarray = torch.cat(probs).cpu().numpy()
    return out


def predict(model: nn.Module, data: SplitData, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out: np.ndarray = predict_proba(model, data, mean, std).argmax(1)
    return out


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    labels = list(range(len(CLASSES)))
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def train_one(cfg: TrainConfig, train: SplitData, val: SplitData, out_dir: Path) -> None:
    """Train for cfg.epochs; keep the epoch with the best validation macro-F1."""
    train, val = prepare(train, cfg), prepare(val, cfg)
    torch.manual_seed(SEED)
    dev = device()
    dim = train.emb.shape[1]
    mean, std = feature_stats(train.emb)
    freq = np.bincount(train.y, minlength=len(CLASSES)) / len(train.y)
    weight = (1.0 / np.maximum(freq, 1e-6)) ** cfg.class_weight_power
    weight = weight / weight.mean()
    model = build_model(cfg, dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32, device=dev))
    ds = to_device(train, mean, std)
    gen = torch.Generator().manual_seed(SEED)
    log = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(cfg.epochs):
        model.train()
        t0, steps = time.perf_counter(), 0
        total = torch.zeros((), device=dev)
        order = torch.randperm(len(train.y), generator=gen).to(dev)
        for start in range(0, len(order), cfg.batch):
            idx = order[start : start + cfg.batch]
            loss = loss_fn(model(windows(ds, idx)), ds.y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach()  # no per-step .item(): it forces a GPU sync
            steps += 1
        sched.step()
        mean_loss = total.item() / steps
        val_f1 = macro_f1(val.y, predict(model, val, mean, std))
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        seconds = time.perf_counter() - t0
        log.append(
            {"epoch": epoch + 1, "loss": mean_loss, "val_macro_f1": val_f1, "seconds": seconds}
        )
        print(
            f"{cfg.name} epoch {epoch + 1}/{cfg.epochs} loss {mean_loss:.4f} "
            f"val macro-F1 {val_f1:.3f} ({seconds:.0f}s)",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "mean": mean, "std": std, "dim": dim},
        out_dir / "model.pt",
    )
    params = sum(p.numel() for p in model.parameters())
    meta = {
        "config": asdict(cfg),
        "features": cfg.features,
        "window": WINDOW,
        "seed": SEED,
        "parameters": params,
        "train_frames": int(len(train.y)),
        "train_videos": len(train.keys),
        "val_frames": int(len(val.y)),
        "val_videos": len(val.keys),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "class_weights": dict(zip(CLASSES, map(float, weight), strict=True)),
        "log": log,
    }
    (out_dir / "train.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_model(run_dir: Path) -> tuple[nn.Module, TrainConfig, np.ndarray, np.ndarray]:
    meta = json.loads((run_dir / "train.json").read_text())
    cfg = TrainConfig(**meta["config"])
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    model = build_model(cfg, int(ckpt["dim"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device()), cfg, ckpt["mean"], ckpt["std"]


def train_all(cache_dir: Path, runs_dir: Path, configs: tuple[TrainConfig, ...] = CONFIGS) -> None:
    for features in dict.fromkeys(cfg.features for cfg in configs):
        train = load_split(cache_dir, "train", features)
        val = load_split(cache_dir, "val", features)
        print(
            f"[{features}] train: {len(train.keys)} videos, {len(train.y)} steps; "
            f"val: {len(val.keys)} videos, {len(val.y)} steps",
            flush=True,
        )
        for cfg in (c for c in configs if c.features == features):
            train_one(cfg, train, val, runs_dir / cfg.name)
