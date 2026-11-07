"""On-device accuracy check: do TensorRT engines on the Jetson give the Mac's answers?

`pack` (on the Mac, needs the train extra) writes real inputs and expected outputs:
  - 16 real S3D clips from one T2 (unseen-camera) video, with the Mac's cached embeddings;
  - every T2 step as head windows, with labels and the Mac's PyTorch probabilities;
  - a head ONNX with a fixed batch equal to the number of T2 steps, for one TensorRT run.

`check` (on the Jetson, needs only NumPy and trtexec) runs the engines on those inputs and reports
encoder cosine similarity and head agreement / macro-F1 for FP32 and FP16.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
N_CLIPS = 16


def pack(out: Path, run_dir: Path, cache_dir: Path, archive: Path, annotation_root: Path) -> None:
    import io

    import av
    import torch
    from PIL import Image

    from kcp.edge.export import _export, _Softmax
    from kcp.perception.data import assign_split
    from kcp.perception.embed import frame_labels
    from kcp.perception.features import CLIP_OFFSETS, SIZE, iter_archive
    from kcp.perception.train import (
        WINDOW,
        DeviceSplit,
        load_model,
        load_split,
        predict_proba,
        prepare,
        to_device,
        windows,
    )

    out.mkdir(parents=True, exist_ok=True)
    # Head: every T2 step, as standardized windows.
    model, cfg, mean, std = load_model(run_dir)
    data = prepare(load_split(cache_dir, "T2", cfg.features), cfg)
    probs = predict_proba(model, data, mean, std)
    ds: DeviceSplit = to_device(data, mean, std)
    idx = torch.arange(len(data.y), device=ds.emb.device)
    x: np.ndarray = windows(ds, idx).cpu().numpy().astype(np.float32)
    x.tofile(out / "head_input.bin")
    np.save(out / "head_labels.npy", data.y)
    np.save(out / "head_mac_probs.npy", probs)
    diff = _export(
        _Softmax(model), torch.from_numpy(x[:1].repeat(len(x), 0)), out / "head_batch.onnx"
    )
    print(f"head: {len(x)} windows, batch-ONNX parity {diff:.2e}", flush=True)

    # Encoder: real clips from the first T2 video in the archive, cropped as in training.
    boxes = json.loads((cache_dir / "crop_boxes.json").read_text())
    for video, raw in iter_archive(archive):
        if assign_split(video) != "T2" or video.key not in boxes:
            continue
        cached = np.load(cache_dir / "s3d_crop" / f"{video.key}.npz")
        anchors = [int(a) for a in cached["frame"][:N_CLIPS]]
        n = len(frame_labels(video.annotation_path(annotation_root)))
        need = {min(max(a + k, 0), n - 1) for a in anchors for k in CLIP_OFFSETS}
        crops = {}
        with av.open(io.BytesIO(raw), mode="r") as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i > max(need):
                    break
                if i in need:
                    img = frame.to_image().crop(tuple(boxes[video.key]))  # type: ignore[no-untyped-call]
                    crops[i] = np.asarray(img.resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        for c, a in enumerate(anchors):
            clip = np.stack([crops[min(max(a + k, 0), n - 1)] for k in CLIP_OFFSETS])  # T,H,W,C
            norm = (clip.astype(np.float32) / 255 - KINETICS_MEAN) / KINETICS_STD
            norm.transpose(3, 0, 1, 2)[None].astype(np.float32).tofile(out / f"clip_{c:02d}.bin")
        np.save(out / "clip_expected.npy", cached["emb"][:N_CLIPS].astype(np.float32))
        (out / "pack.json").write_text(
            json.dumps(
                {
                    "video": video.key,
                    "anchors": anchors,
                    "head_steps": int(len(x)),
                    "window": WINDOW,
                    "dim": int(x.shape[2]),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"encoder: {len(anchors)} clips from {video.key}", flush=True)
        return


def _trt_output(cmd: list[str], out_json: Path) -> np.ndarray:
    subprocess.run(cmd + [f"--exportOutput={out_json}"], check=True, capture_output=True)
    first = json.loads(out_json.read_text())[0]
    return np.array(first["values"], dtype=np.float32)


def macro_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    scores = []
    for c in range(classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores))


def check(pack_dir: Path, model_dir: Path) -> dict[str, object]:
    meta = json.loads((pack_dir / "pack.json").read_text())
    y = np.load(pack_dir / "head_labels.npy")
    mac = np.load(pack_dir / "head_mac_probs.npy")
    expected = np.load(pack_dir / "clip_expected.npy")
    tmp = pack_dir / "trt_out.json"
    result: dict[str, object] = {
        "video": meta["video"],
        "mac_macro_f1": macro_f1(y, mac.argmax(1), mac.shape[1]),
    }
    for prec in ("fp32", "fp16"):
        flag = ["--fp16"] if prec == "fp16" else []
        # Encoder: one run per real clip with the benchmark engine.
        cos = []
        for c in range(len(expected)):
            emb = _trt_output(
                [
                    TRTEXEC,
                    f"--loadEngine={model_dir}/encoder.{prec}.engine",
                    f"--loadInputs=x:{pack_dir}/clip_{c:02d}.bin",
                    "--iterations=1",
                    "--warmUp=0",
                    "--duration=0",
                ],
                tmp,
            )
            e = expected[c]
            cos.append(float(emb @ e / (np.linalg.norm(emb) * np.linalg.norm(e))))
        # Head: all T2 windows in one batched engine.
        probs = _trt_output(
            [
                TRTEXEC,
                f"--onnx={pack_dir}/head_batch.onnx",
                *flag,
                f"--loadInputs=x:{pack_dir}/head_input.bin",
                "--iterations=1",
                "--warmUp=0",
                "--duration=0",
            ],
            tmp,
        )
        probs = probs.reshape(mac.shape)
        result[prec] = {
            "encoder_cosine_min": min(cos),
            "encoder_cosine_mean": float(np.mean(cos)),
            "head_max_abs_prob_diff": float(np.abs(probs - mac).max()),
            "head_argmax_agreement": float((probs.argmax(1) == mac.argmax(1)).mean()),
            "head_macro_f1": macro_f1(y, probs.argmax(1), mac.shape[1]),
        }
        print(prec, json.dumps(result[prec]), flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kcp.edge.parity")
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--out", type=Path, default=Path("data/edge/parity"))
    p_pack.add_argument(
        "--run", type=Path, default=Path("data/runs_2a2/s3d_crop__temporal_head_centered")
    )
    p_check = sub.add_parser("check")
    p_check.add_argument("pack_dir", type=Path)
    p_check.add_argument("model_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(
            args.out,
            args.run,
            Path("data/cache"),
            Path("data/raw/breakfast/BreakfastII_15fps_qvga_sync.tar.gz"),
            Path("data/raw/breakfast/annotations/segmentation_coarse"),
        )
    else:
        print(json.dumps(check(args.pack_dir, args.model_dir), indent=2))


if __name__ == "__main__":
    main()
