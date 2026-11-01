"""Benchmark the exported encoder and head with ONNX Runtime: p50/p95 latency and memory.

Run the same command on the Mac and on the Jetson:

    uv run python -m kcp.edge.bench data/edge/<run> --iterations 200

On the Jetson, ONNX Runtime picks TensorRT, then CUDA, then CPU, whichever its build supports.
(Install the Jetson build of onnxruntime-gpu from NVIDIA's Jetson AI Lab package index.)
NVIDIA's own tool gives a second opinion with TensorRT directly:

    /usr/src/tensorrt/bin/trtexec --onnx=data/edge/<run>/encoder.onnx --fp16
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PREFERRED = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


def _session(path: Path) -> ort.InferenceSession:
    available = ort.get_available_providers()
    providers = [p for p in PREFERRED if p in available]
    return ort.InferenceSession(str(path), providers=providers)


def _time(session: ort.InferenceSession, x: np.ndarray, iterations: int) -> dict[str, float]:
    feed = {session.get_inputs()[0].name: x}
    for _ in range(10):  # warm-up: TensorRT builds its engine on the first runs
        session.run(None, feed)
    times = []
    for _ in range(iterations):
        t = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kcp.edge.bench")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args(argv)
    meta = json.loads((args.model_dir / "head.json").read_text())
    rng = np.random.default_rng(0)
    clip = meta["features"] == "s3d_crop"
    enc_input = rng.random((1, 3, 8, 224, 224) if clip else (1, 3, 224, 224), dtype=np.float32)
    head_input = rng.standard_normal((1, meta["window"], meta["dim"]), dtype=np.float32)
    encoder = _session(args.model_dir / "encoder.onnx")
    head = _session(args.model_dir / "head.onnx")
    result = {
        "machine": platform.platform(),
        "providers": encoder.get_providers(),
        "features": meta["features"],
        "encoder": _time(encoder, enc_input, args.iterations),
        "head": _time(head, head_input, args.iterations),
        # ru_maxrss is bytes on macOS and kilobytes on Linux (the Jetson).
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1e6 if platform.system() == "Darwin" else 1e3),
    }
    step_s = 32 / 15 if clip else 8 / 15
    budget_ms = step_s * 1000
    used = result["encoder"]["p95_ms"] + result["head"]["p95_ms"]
    result["realtime_budget_ms"] = budget_ms
    result["budget_used_p95"] = used / budget_ms
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
