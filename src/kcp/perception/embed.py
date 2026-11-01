"""Compute frozen-encoder frame embeddings once per video and cache them on disk."""

from __future__ import annotations

import time
from pathlib import Path

import av
import numpy as np
import timm
import torch

from kcp.perception.data import Video, assign_split, list_videos, read_segments
from kcp.perception.labels import class_index, map_label

ENCODER = "mobilenetv3_large_100"
FRAME_STRIDE = 8  # keep every 8th frame of the 15 fps video (1.875 frames/s)
BATCH = 64
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def frame_labels(annotation: Path) -> np.ndarray:
    """Camera-label index for every frame (0-indexed); -1 where no segment covers the frame."""
    segments = read_segments(annotation)
    labels = np.full(segments[-1].end, -1, dtype=np.int8)
    for seg in segments:
        labels[seg.start - 1 : seg.end] = class_index(map_label(seg.label))
    return labels


def sampled_frames(video: Video, n_labeled: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode the video; return (uint8 NHWC frames, indices) for every FRAME_STRIDE-th frame."""
    frames, indices = [], []
    with av.open(str(video.path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i >= n_labeled:
                break
            if i % FRAME_STRIDE == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                indices.append(i)
    return np.stack(frames), np.array(indices, dtype=np.int32)


def load_encoder(device: torch.device) -> torch.nn.Module:
    model = timm.create_model(ENCODER, pretrained=True, num_classes=0)
    return model.eval().to(device).half()


@torch.inference_mode()
def embed_frames(model: torch.nn.Module, frames: np.ndarray, device: torch.device) -> np.ndarray:
    """Embed at native resolution: the encoder is fully convolutional with global pooling."""
    mean, std = _MEAN.to(device).half(), _STD.to(device).half()
    out = []
    for start in range(0, len(frames), BATCH):
        chunk = frames[start : start + BATCH]
        n = len(chunk)
        if n < BATCH:  # pad to a fixed batch shape: the MPS backend recompiles for each new shape
            pad = np.zeros((BATCH - n, *chunk.shape[1:]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        # Move uint8 to the GPU (4x less data than float32), then normalize there.
        x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).half() / 255
        x = (x - mean) / std
        out.append(model(x)[:n].float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def run(video_root: Path, annotation_root: Path, cache_dir: Path, limit: int | None = None) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = cache_dir / ENCODER
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [v for v in list_videos(video_root) if assign_split(v) is not None][:limit]
    model = load_encoder(device)
    done = n_frames = skipped = 0
    t0 = time.perf_counter()
    for video in videos:
        target = out_dir / f"{video.key}.npz"
        if target.exists():  # resumable: one file per video
            continue
        annotation = video.annotation_path(annotation_root)
        if not annotation.exists():
            skipped += 1
            continue
        labels = frame_labels(annotation)
        frames, indices = sampled_frames(video, len(labels))
        keep = labels[indices] >= 0
        frames, indices = frames[keep], indices[keep]
        emb = embed_frames(model, frames, device)
        tmp = target.with_name(target.stem + ".tmp.npz")
        np.savez(tmp, emb=emb, y=labels[indices], frame=indices)
        tmp.rename(target)
        done += 1
        n_frames += len(indices)
        if done % 50 == 0:
            rate = n_frames / (time.perf_counter() - t0)
            print(
                f"{done}/{len(videos)} videos, {n_frames} frames, {rate:.0f} frames/s", flush=True
            )
    elapsed = time.perf_counter() - t0
    print(
        f"done: {done} new videos, {n_frames} frames, {skipped} skipped (no annotation), "
        f"{elapsed:.0f}s",
        flush=True,
    )
