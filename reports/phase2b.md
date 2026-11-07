# Phase 2b — Edge Pipeline on the Jetson

Date: 2026-11-05. Raw numbers: `reports/phase2b_jetson.json`. Mac baseline: `reports/phase2b_mac_benchmark.json`. Device: `docs/jetson_inventory.md`. Design: `IMPLEMENTATION_PLAN.md`, "Phase 2b".

## Questions

1. Is the selected perception model (S3D encoder on motion crops + temporal head) fast enough on a Jetson Orin Nano Super for live use?
2. Does it give the same answers on the Jetson as on the Mac, at full (FP32) and half (FP16) precision?

## Method

- **Model:** the ONNX files from `kcp.edge.export` (`data/edge/s3d_selected/`), copied to the Jetson with checksums verified. No retraining and no change to the model.
- **Runtime:** TensorRT 10.16.2, which JetPack 7.2.1 includes. `trtexec` compiles each ONNX file into an engine for this GPU, once in FP32 and once with FP16 allowed.
- **Speed:** `trtexec`, batch 1, 2 s warm-up, then 20 s of back-to-back runs. Latency includes copying the input to the GPU and the output back. `tegrastats` logged power, GPU load, memory, and temperature every 0.5 s. Power mode: MAXN_SUPER.
- **Accuracy** (`src/kcp/edge/parity.py`): the Mac packed real inputs and its own answers; the Jetson ran them through the engines with NumPy and `trtexec` only.
  - Encoder: 16 real 8-frame clips from one unseen-camera (T2) video (`P04_webcam01_friedegg`), cropped and normalized exactly as in training. Compared with the Mac's cached embeddings by cosine similarity.
  - Head: all 6,179 T2 steps in one batch. Compared with the Mac PyTorch probabilities, and scored with the same 8-class macro-F1.
- **Real-time budget:** one S3D step covers 32 frames at 15 fps, so a new step arrives every 2,133 ms.

## Results

### Speed

| Model | Precision | p50 | p95 | Throughput | Engine |
|---|---|---:|---:|---:|---:|
| Encoder (S3D) | FP32 | 10.94 ms | 11.29 ms | 92 /s | 31 MiB |
| Encoder (S3D) | **FP16** | **7.20 ms** | **7.22 ms** | 143 /s | 15 MiB |
| Head | FP32 | 0.18 ms | 0.19 ms | 5,986 /s | 2 MiB |
| Head | **FP16** | **0.10 ms** | **0.12 ms** | 7,475 /s | 1 MiB |
| Mac CPU, ONNX Runtime (encoder) | FP32 | 64.4 ms | 85.7 ms | — | — |

- **One FP16 step (encoder + head) takes ~7.3 ms at p95: 0.34% of the 2,133 ms budget.** The Mac CPU used 4%.
- FP16 is 1.6× faster than FP32 on the Jetson. The Jetson FP16 encoder is ~12× faster than the Mac CPU at p95.
- p95 is within 0.3% of p50 for the FP16 encoder: the latency is very stable.
- Engine builds took 15–121 s each. They happen once per device and model, not at run time.

### Accuracy

| Check | FP32 | FP16 |
|---|---:|---:|
| Encoder cosine similarity, worst of 16 clips | 0.999996 | 0.999990 |
| Head, largest probability difference vs. Mac | 0.0012 | 0.0062 |
| Head, same top class as Mac (6,179 steps) | 100% | 100% |
| Head macro-F1 on T2 | 0.4167 | 0.4167 |
| Mac PyTorch macro-F1 on T2 | 0.4167 | 0.4167 |

- **Every one of the 6,179 T2 predictions is the same on the Jetson as on the Mac, in both precisions.** The macro-F1 is identical to the reported Phase 2a.2 result (0.417).
- FP16 changes probabilities by at most 0.006. No step was close enough to a tie for that to change the answer.
- The FP32 difference (0.0012) is larger than the Mac's ONNX parity (1e-6). The probable cause: TensorRT uses TF32 for FP32 math on this GPU by default, which keeps fewer digits. We did not test this (it would need a `--noTF32` run); it does not change any prediction.

### Power, memory, and heat (while `trtexec` ran back to back)

| Run | Input power (median) | GPU busy (median) | Max junction temp |
|---|---:|---:|---:|
| Idle (inventory) | 5.2 W | 0% | ~50 °C |
| Encoder FP32 | 18.1 W | 99% | 56.0 °C |
| Encoder FP16 | 17.4 W | 98% | 56.8 °C |
| Head FP32 | 10.1 W | 94% | 55.4 °C |
| Head FP16 | 7.5 W | 50% | 54.1 °C |

- System RAM used never exceeded 4.0 of 7.3 GB, including engine builds. The device has no swap, so this margin matters.
- These are worst-case numbers: the GPU ran the model nonstop. Live, the encoder runs ~7 ms every 2.1 s. **Estimate, not measured:** ~17.4 W × 7.2 ms ≈ 0.13 J per step, so the model adds well under 0.1 W on average above idle.

## What this means

- **Speed is not a risk for this model on this device.** The budget leaves room for ~290 FP16 encoder runs per step. That room can pay for denser clips (Phase 2a.3 uses 16 frames per clip), a second camera, or speech recognition on the same device.
- **FP16 is free accuracy-wise here:** same predictions, 1.6× faster, half the engine size. The deployed pipeline will use FP16.
- **PyTorch is not needed on the device.** The model ships as ONNX; TensorRT compiles it. The on-device check needs only NumPy and `trtexec`, both present in JetPack.
- ONNX Runtime with the GPU was not installed. TensorRT already runs the model directly. Installing it would first need a build verified for JetPack 7.2 / CUDA 13.2 / Python 3.12 on NVIDIA's current pages. It stays optional.

## Limits and what is still open

- `trtexec` times the model only. It does not include video decoding, the motion crop, resizing, or Python overhead. Those run once per step on the CPU and are not yet measured on the device.
- The speed runs use random inputs. Latency does not depend on the input values for this model; the accuracy check uses real inputs.
- The encoder check covers 16 clips from one video. The head check covers every T2 step.
- **Still open from the plan:** run the full streaming pipeline (decode → crop → encoder → streaming head → run detector → interpreter) on a Breakfast video on the Jetson with the network off, and record end-to-end latency per step. This needs a video decoder on the device (PyAV or GStreamer) in a virtual environment.

## Reproduce

```sh
# Mac
uv run python -m kcp.edge.export                 # writes data/edge/s3d_selected/
uv run python -m kcp.edge.parity pack            # writes data/edge/parity/ (~270 MB)
scp -r data/edge/s3d_selected data/edge/parity src/kcp/edge/parity.py jetson:kcp/

# Jetson (build engines, time them, log tegrastats)
for m in encoder head; do for p in fp32 fp16; do
  /usr/src/tensorrt/bin/trtexec --onnx=$m.onnx $([ $p = fp16 ] && echo --fp16) \
    --saveEngine=$m.$p.engine --warmUp=2000 --duration=20 --percentile=50,95,99
done; done
python3 parity.py check parity s3d_selected      # accuracy vs. the Mac
```
