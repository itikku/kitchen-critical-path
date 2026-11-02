"""Phase 2a.2 features: a motion crop, then MobileNetV3 frames and S3D clips.

Videos stream from the .tar.gz archive into memory, one at a time; nothing is extracted to disk.
Each video is decoded twice from memory:
  pass 1 finds where motion happens and fixes one square crop for the whole video;
  pass 2 crops every needed frame to 224x224 and feeds the two encoders.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import timm
import torch
from PIL import Image
from torchvision.models import video as tv_video

from kcp.perception.data import Video, assign_split, video_from_path
from kcp.perception.embed import frame_labels

SIZE = 224
MNV3_STRIDE = 8  # same frames as run 2 (every 8th frame, 1.875 frames/s)
# One S3D clip every 32 frames (~2.1 s). S3D runs at ~19.5 clips/s on the M1 at any batch size;
# a clip every 16 frames would take ~2.5 h, every 32 frames ~1.2 h.
CLIP_STRIDE = 32
CLIP_OFFSETS = (-7, -5, -3, -1, 1, 3, 5, 7)  # 8 frames at stride 2, centered on the anchor
MIN_CROP = 128
_IMNET = (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
_KINETICS = (torch.tensor([0.43216, 0.394666, 0.37645]), torch.tensor([0.22803, 0.22145, 0.216989]))


def iter_archive(archive: Path) -> Iterator[tuple[Video, bytes]]:
    """Stream (video, file bytes) from the archive in its stored order."""
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".avi"):
                continue
            video = video_from_path(Path(member.name))
            if video is None:
                continue
            f = tar.extractfile(member)
            if f is not None:
                yield video, f.read()


def motion_box(data: bytes, n_frames: int) -> tuple[int, int, int, int]:
    """Square crop (left, top, right, bottom) around where the video moves.
    Motion energy = sum of |frame(t) - frame(t-2)| on a 4x-downsampled grayscale copy."""
    energy: np.ndarray | None = None
    history: list[np.ndarray] = []
    width = height = 0
    with av.open(io.BytesIO(data), mode="r") as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= n_frames:
                break
            gray = frame.to_ndarray(format="gray")
            height, width = gray.shape
            small = gray[::4, ::4].astype(np.float32)
            history.append(small)
            if len(history) > 2:
                prev = history.pop(0)
                diff = np.abs(small - prev)
                energy = diff if energy is None else energy + diff
    if energy is None or energy.sum() == 0:
        side = min(width, height)
        return (
            (width - side) // 2,
            (height - side) // 2,
            (width + side) // 2,
            (height + side) // 2,
        )
    weights = np.maximum(energy - np.median(energy), 0)
    ys, xs = np.mgrid[0 : weights.shape[0], 0 : weights.shape[1]] * 4 + 2
    total = weights.sum()
    cx, cy = (weights * xs).sum() / total, (weights * ys).sum() / total
    sx = np.sqrt((weights * (xs - cx) ** 2).sum() / total)
    sy = np.sqrt((weights * (ys - cy) ** 2).sum() / total)
    side = int(np.clip(4 * max(sx, sy), MIN_CROP, min(width, height)))
    left = int(np.clip(cx - side / 2, 0, width - side))
    top = int(np.clip(cy - side / 2, 0, height - side))
    return (left, top, left + side, top + side)


class Encoders:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.mnv3 = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0)
        self.mnv3 = self.mnv3.eval().to(device).half()
        s3d = tv_video.s3d(weights=tv_video.S3D_Weights.KINETICS400_V1)
        self.s3d_features = s3d.features.eval().to(device).half()  # drop the Kinetics classifier

    @torch.inference_mode()
    def frames(self, batch: np.ndarray) -> np.ndarray:
        """(N, 224, 224, 3) uint8 -> (N, 1280) float16."""
        mean, std = (t.view(1, 3, 1, 1).to(self.device).half() for t in _IMNET)
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).half() / 255
        out: np.ndarray = self.mnv3((x - mean) / std).float().cpu().numpy().astype(np.float16)
        return out

    @torch.inference_mode()
    def clips(self, batch: np.ndarray) -> np.ndarray:
        """(N, 8, 224, 224, 3) uint8 -> (N, 1024) float16 S3D features, pooled over space-time."""
        mean, std = (t.view(1, 3, 1, 1, 1).to(self.device).half() for t in _KINETICS)
        x = torch.from_numpy(batch).to(self.device).permute(0, 4, 1, 2, 3).half() / 255
        feats = self.s3d_features((x - mean) / std).mean(dim=(2, 3, 4))
        out: np.ndarray = feats.float().cpu().numpy().astype(np.float16)
        return out


def video_features(
    enc: Encoders, data: bytes, labels: np.ndarray
) -> tuple[dict[str, np.ndarray], tuple[int, int, int, int]]:
    n = len(labels)
    box = motion_box(data, n)
    mnv3_idx = [i for i in range(0, n, MNV3_STRIDE) if labels[i] >= 0]
    anchors = [a for a in range(0, n, CLIP_STRIDE) if labels[a] >= 0]
    clip_frames = {a: [min(max(a + k, 0), n - 1) for k in CLIP_OFFSETS] for a in anchors}
    needed = set(mnv3_idx) | {i for frames in clip_frames.values() for i in frames}
    crops: dict[int, np.ndarray] = {}
    mnv3_batch: list[np.ndarray] = []
    mnv3_out: list[np.ndarray] = []
    mnv3_done: list[int] = []  # frames actually embedded (a video can end before its labels)
    clip_done: list[int] = []
    clip_batch: list[np.ndarray] = []
    clip_out: list[np.ndarray] = []
    done_anchor = 0
    last = -1
    with av.open(io.BytesIO(data), mode="r") as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= n:
                break
            last = i
            if i in needed:
                img = frame.to_image().crop(box)  # type: ignore[no-untyped-call]
                img = img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                crops[i] = np.asarray(img)
                if labels[i] >= 0 and i % MNV3_STRIDE == 0:
                    mnv3_batch.append(crops[i])
                    mnv3_done.append(i)
            # Emit every clip whose last frame has arrived.
            while done_anchor < len(anchors) and clip_frames[anchors[done_anchor]][-1] <= i:
                a = anchors[done_anchor]
                clip_batch.append(np.stack([crops[j] for j in clip_frames[a]]))
                clip_done.append(a)
                done_anchor += 1
            if len(mnv3_batch) >= 64:
                mnv3_out.append(enc.frames(np.stack(mnv3_batch)))
                mnv3_batch = []
            if len(clip_batch) >= 8:
                clip_out.append(enc.clips(np.stack(clip_batch)))
                clip_batch = []
            # A clip spans 15 frames, so older crops are never needed again.
            for j in [j for j in crops if j < i - 15]:
                del crops[j]
    # The video ended before its annotation: keep clips centered on a real frame, reusing the
    # nearest earlier frame for missing ones; drop clips centered past the end.
    for a in anchors[done_anchor:]:
        if a > last:
            break
        clip_batch.append(
            np.stack([crops[max(k for k in crops if k <= j)] for j in clip_frames[a]])
        )
        clip_done.append(a)
    if mnv3_batch:
        mnv3_out.append(enc.frames(np.stack(mnv3_batch)))
    if clip_batch:
        clip_out.append(enc.clips(np.stack(clip_batch)))
    m_idx, c_idx = np.array(mnv3_done, dtype=np.int32), np.array(clip_done, dtype=np.int32)
    assert sum(len(x) for x in mnv3_out) == len(m_idx)
    assert sum(len(x) for x in clip_out) == len(c_idx)
    return {
        "mnv3_crop": np.concatenate(mnv3_out),
        "mnv3_crop_frame": m_idx,
        "s3d_crop": np.concatenate(clip_out),
        "s3d_crop_frame": c_idx,
    }, box


def run(archive: Path, annotation_root: Path, cache_dir: Path, limit: int | None = None) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    enc = Encoders(device)
    for name in ("mnv3_crop", "s3d_crop"):
        (cache_dir / name).mkdir(parents=True, exist_ok=True)
    boxes_path = cache_dir / "crop_boxes.json"
    boxes: dict[str, list[int]] = json.loads(boxes_path.read_text()) if boxes_path.exists() else {}
    done = frames = clips = 0
    t0 = time.perf_counter()
    for video, data in iter_archive(archive):
        if assign_split(video) is None:
            continue
        annotation = video.annotation_path(annotation_root)
        targets = [cache_dir / name / f"{video.key}.npz" for name in ("mnv3_crop", "s3d_crop")]
        # Same videos as run 2, so only the features differ between the two comparisons.
        in_run2 = (cache_dir / "mobilenetv3_large_100" / f"{video.key}.npz").exists()
        if not in_run2 or not annotation.exists() or all(t.exists() for t in targets):
            continue
        labels = frame_labels(annotation)
        feats, box = video_features(enc, data, labels)
        for name, target in zip(("mnv3_crop", "s3d_crop"), targets, strict=True):
            idx = feats[f"{name}_frame"]
            tmp = target.with_name(target.stem + ".tmp.npz")
            np.savez(tmp, emb=feats[name], y=labels[idx], frame=idx)
            tmp.rename(target)
        boxes[video.key] = list(box)
        done += 1
        frames += len(feats["mnv3_crop_frame"])
        clips += len(feats["s3d_crop_frame"])
        if done % 25 == 0:
            boxes_path.write_text(json.dumps(boxes))
            elapsed = time.perf_counter() - t0
            print(f"{done} videos, {frames} frames, {clips} clips, {elapsed:.0f}s", flush=True)
        if limit is not None and done >= limit:
            break
    boxes_path.write_text(json.dumps(boxes))
    print(
        f"done: {done} videos, {frames} frames, {clips} clips, {time.perf_counter() - t0:.0f}s",
        flush=True,
    )


# ------------------------------------------------------- Phase 2a.3 (IMPLEMENTATION_PLAN.md)

WRIST_FRAMES = 8


def _save(target: Path, emb: np.ndarray, y: np.ndarray, frame: np.ndarray) -> None:
    """Write to a temporary file, then rename: an interruption never leaves a partial file."""
    tmp = target.with_name(target.stem + ".tmp.npz")
    np.savez(tmp, emb=emb, y=y, frame=frame)
    tmp.rename(target)


WRIST_MARGIN = 48
_LEFT_WRIST, _RIGHT_WRIST = 9, 10  # COCO keypoint indices


def clip_features(
    enc: Encoders, data: bytes, n: int, box: tuple[int, int, int, int], anchors: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """S3D features for 8-frame clips centered on `anchors`, cropped to `box`.
    Returns (features, anchors actually covered by decoded frames)."""
    clip_frames = {a: [min(max(a + k, 0), n - 1) for k in CLIP_OFFSETS] for a in anchors}
    needed = {i for frames in clip_frames.values() for i in frames}
    crops: dict[int, np.ndarray] = {}
    batch: list[np.ndarray] = []
    out: list[np.ndarray] = []
    done: list[int] = []
    k = 0
    last = -1
    with av.open(io.BytesIO(data), mode="r") as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= n:
                break
            last = i
            if i in needed:
                img = frame.to_image().crop(box)  # type: ignore[no-untyped-call]
                crops[i] = np.asarray(img.resize((SIZE, SIZE), Image.Resampling.BILINEAR))
            while k < len(anchors) and clip_frames[anchors[k]][-1] <= i:
                batch.append(np.stack([crops[j] for j in clip_frames[anchors[k]]]))
                done.append(anchors[k])
                k += 1
            if len(batch) >= 8:
                out.append(enc.clips(np.stack(batch)))
                batch = []
            for j in [j for j in crops if j < i - 15]:
                del crops[j]
    for a in anchors[k:]:  # the video ended before its annotation
        if a > last:
            break
        batch.append(np.stack([crops[max(j for j in crops if j <= f)] for f in clip_frames[a]]))
        done.append(a)
    if batch:
        out.append(enc.clips(np.stack(batch)))
    feats = np.concatenate(out) if out else np.zeros((0, 1024), dtype=np.float16)
    return feats, np.array(done, dtype=np.int32)


def wrist_box(
    pose: torch.nn.Module, device: torch.device, data: bytes, n: int
) -> tuple[tuple[int, int, int, int] | None, int]:
    """Square crop around the cook's wrists on WRIST_FRAMES evenly spaced frames.
    Returns (box or None if fewer than 3 confident wrist points, number of points)."""
    picks = set(np.linspace(n * 0.1, n * 0.9, WRIST_FRAMES).astype(int).tolist())
    frames, width, height = [], 0, 0
    with av.open(io.BytesIO(data), mode="r") as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= n:
                break
            if i in picks:
                rgb = frame.to_ndarray(format="rgb24")
                height, width = rgb.shape[:2]
                frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device))
    points = []
    with torch.inference_mode():
        for o in pose(frames):
            if len(o["scores"]) and float(o["scores"][0]) > 0.8:
                kp, ks = o["keypoints"][0].cpu().numpy(), o["keypoints_scores"][0].cpu().numpy()
                points += [kp[j, :2] for j in (_LEFT_WRIST, _RIGHT_WRIST) if ks[j] > 2.0]
    if len(points) < 3:
        return None, len(points)
    pts = np.array(points)
    x0, y0 = pts.min(0) - WRIST_MARGIN
    x1, y1 = pts.max(0) + WRIST_MARGIN
    side = int(np.clip(max(x1 - x0, y1 - y0), MIN_CROP, min(width, height)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - side / 2, 0, width - side))
    top = int(np.clip(cy - side / 2, 0, height - side))
    return (left, top, left + side, top + side), len(points)


def run_2a3(archive: Path, annotation_root: Path, cache_dir: Path, wrist: bool = True) -> None:
    """Experiment 1 (denser clips, motion crop) and, if `wrist`, experiment 2 (wrist crop)."""
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    enc = Encoders(device)
    pose = None
    if wrist:
        weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        pose = keypointrcnn_resnet50_fpn(weights=weights).eval().to(device)
    motion = json.loads((cache_dir / "crop_boxes.json").read_text())
    dense_dir, wrist_dir = cache_dir / "s3d_crop16", cache_dir / "s3d_wrist"
    dense_dir.mkdir(exist_ok=True)
    wrist_dir.mkdir(exist_ok=True)
    wrist_log_path = cache_dir / "wrist_boxes.json"
    wrist_log = json.loads(wrist_log_path.read_text()) if wrist_log_path.exists() else {}
    done = 0
    t0 = time.perf_counter()
    for video, data in iter_archive(archive):
        base = cache_dir / "s3d_crop" / f"{video.key}.npz"  # same videos as Phase 2a.2
        dense_t, wrist_t = dense_dir / f"{video.key}.npz", wrist_dir / f"{video.key}.npz"
        if not base.exists() or (dense_t.exists() and (wrist_t.exists() or not wrist)):
            continue
        labels = frame_labels(video.annotation_path(annotation_root))
        n = len(labels)
        box = tuple(motion[video.key])
        old = np.load(base)
        if not dense_t.exists():
            between = [a for a in range(CLIP_STRIDE // 2, n, CLIP_STRIDE) if labels[a] >= 0]
            emb, idx = clip_features(enc, data, n, box, between)
            frames = np.concatenate([old["frame"], idx])
            order = np.argsort(frames)
            emb_all = np.concatenate([old["emb"], emb])[order]
            _save(dense_t, emb=emb_all, y=labels[frames[order]], frame=frames[order])
        if wrist and pose is not None and not wrist_t.exists():
            wbox, n_points = wrist_box(pose, device, data, n)
            used = wbox if wbox is not None else box
            emb, idx = clip_features(enc, data, n, used, [int(a) for a in old["frame"]])
            _save(wrist_t, emb=emb, y=labels[idx], frame=idx)
            wrist_log[video.key] = {"box": list(used), "points": n_points, "fallback": wbox is None}
        done += 1
        if done % 25 == 0:
            wrist_log_path.write_text(json.dumps(wrist_log))
            print(f"{done} videos, {time.perf_counter() - t0:.0f}s", flush=True)
    wrist_log_path.write_text(json.dumps(wrist_log))
    fallbacks = sum(v["fallback"] for v in wrist_log.values())
    print(f"done: {done} videos, wrist fallbacks {fallbacks}, {time.perf_counter() - t0:.0f}s")
