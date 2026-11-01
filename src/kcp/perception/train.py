"""Train two classifiers on cached frame embeddings:

- `frame_linear`: a linear layer on one frame (baseline, no time context);
- `temporal_head`: a small 1D-conv network over an 8-frame window (~4.3 s) centered on the frame.

Hyperparameters are fixed in advance; the test splits are never used for tuning."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from kcp.perception.data import Split, Video, assign_split
from kcp.perception.embed import ENCODER
from kcp.perception.labels import CLASSES

WINDOW = 8
SEED = 0


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


@dataclass
class SplitData:
    emb: np.ndarray  # (N, D) float16, all frames of all videos in the split
    y: np.ndarray  # (N,) int64
    video_start: np.ndarray  # (N,) first global index of the frame's video
    video_end: np.ndarray  # (N,) last global index of the frame's video
    keys: list[str]


def video_of(key: str) -> Video:
    person, camera, recipe = key.split("_", 2)
    return Video(path=Path(key), person=person, camera=camera, recipe=recipe)


def load_split(cache_dir: Path, split: Split) -> SplitData:
    files = sorted((cache_dir / ENCODER).glob("*.npz"))
    files = [
        f for f in files if not f.stem.endswith(".tmp") and assign_split(video_of(f.stem)) == split
    ]
    embs, ys, starts, ends, keys = [], [], [], [], []
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
        offset += n
    return SplitData(
        emb=np.concatenate(embs),
        y=np.concatenate(ys),
        video_start=np.concatenate(starts),
        video_end=np.concatenate(ends),
        keys=keys,
    )


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


def predict(model: nn.Module, data: SplitData, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    ds = to_device(data, mean, std)
    model.eval()
    preds = []
    with torch.inference_mode():
        for start in range(0, len(data.y), 4096):
            idx = torch.arange(start, min(start + 4096, len(data.y)), device=ds.emb.device)
            preds.append(model(windows(ds, idx)).argmax(1))
    return torch.cat(preds).cpu().numpy()


def train_one(cfg: TrainConfig, train: SplitData, out_dir: Path) -> None:
    torch.manual_seed(SEED)
    dev = device()
    dim = train.emb.shape[1]
    mean = train.emb.astype(np.float32).mean(0)
    std = train.emb.astype(np.float32).std(0) + 1e-6
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
        log.append({"epoch": epoch + 1, "loss": mean_loss, "seconds": time.perf_counter() - t0})
        print(
            f"{cfg.model} epoch {epoch + 1}/{cfg.epochs} loss {mean_loss:.4f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "mean": mean, "std": std, "dim": dim},
        out_dir / "model.pt",
    )
    params = sum(p.numel() for p in model.parameters())
    meta = {
        "config": asdict(cfg),
        "encoder": ENCODER,
        "window": WINDOW,
        "seed": SEED,
        "parameters": params,
        "train_frames": int(len(train.y)),
        "train_videos": len(train.keys),
        "class_weights": dict(zip(CLASSES, map(float, weight), strict=True)),
        "log": log,
    }
    (out_dir / "train.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_model(run_dir: Path) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    meta = json.loads((run_dir / "train.json").read_text())
    cfg = TrainConfig(**meta["config"])
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    model = build_model(cfg, int(ckpt["dim"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device()), ckpt["mean"], ckpt["std"]


def train_all(cache_dir: Path, runs_dir: Path) -> None:
    train = load_split(cache_dir, "train")
    print(f"train: {len(train.keys)} videos, {len(train.y)} frames", flush=True)
    for name in MODELS:
        train_one(TrainConfig(model=name), train, runs_dir / name)
