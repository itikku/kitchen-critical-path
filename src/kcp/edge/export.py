"""Export the encoder and the trained head to ONNX, and check that ONNX matches PyTorch.

ONNX is a portable model format. On the Jetson, TensorRT (NVIDIA's inference optimizer) or ONNX
Runtime loads these files; PyTorch is not needed at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from kcp.perception.features import Encoders
from kcp.perception.train import WINDOW, load_model


class _Softmax(nn.Module):
    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(x).softmax(-1)
        return out


class _ClipFeatures(nn.Module):
    """S3D features pooled over space-time, as in Phase 2a.2 training."""

    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.features(x).mean(dim=(2, 3, 4))
        return out


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> float:
    """Export in float32 on the CPU; return the max |ONNX - PyTorch| on a random input."""
    model = model.float().cpu().eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (example,), str(path), input_names=["x"], output_names=["y"], dynamo=True
    )
    with torch.inference_mode():
        expected = model(example).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": example.numpy()})[0]
    return float(np.abs(actual - expected).max())


def export_head(run_dir: Path, out_dir: Path) -> dict[str, object]:
    model, cfg, mean, std = load_model(run_dir)
    dim = int(mean.shape[0])
    example = torch.randn(1, WINDOW, dim)
    diff = _export(_Softmax(model), example, out_dir / "head.onnx")
    np.save(out_dir / "mean.npy", mean)
    np.save(out_dir / "std.npy", std)
    meta = {
        "run": run_dir.name,
        "features": cfg.features,
        "center": cfg.center,
        "dim": dim,
        "window": WINDOW,
        "max_abs_diff": diff,
    }
    (out_dir / "head.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def export_encoder(features: str, out_dir: Path) -> dict[str, object]:
    enc = Encoders(torch.device("cpu"))
    if features == "s3d_crop":
        model: nn.Module = _ClipFeatures(enc.s3d_features)
        example = torch.randn(1, 3, 8, 224, 224)
    else:
        model, example = enc.mnv3, torch.randn(1, 3, 224, 224)
    diff = _export(model, example, out_dir / "encoder.onnx")
    return {"features": features, "input_shape": str(tuple(example.shape)), "max_abs_diff": diff}
