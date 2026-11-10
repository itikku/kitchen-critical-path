# Phase 2b — Edge Pipeline on the Jetson

Date: 2026-11-05; end-to-end run 2026-11-09. Raw numbers: `reports/phase2b_jetson.json` and `reports/phase2b_e2e.json`. Mac baseline: `reports/phase2b_mac_benchmark.json`. Device: `docs/jetson_inventory.md`. Design: `IMPLEMENTATION_PLAN.md`, "Phase 2b".

## Questions

1. Is the selected perception model (S3D encoder on motion crops + temporal head) fast enough on a Jetson Orin Nano Super for live use?
2. Does it give the same answers on the Jetson as on the Mac, at full (FP32) and half (FP16) precision?
3. Does the whole pipeline — video in, planner recommendations out — run on the Jetson in real time, and give the same task events as the Mac?

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

## End-to-end run on the device (2026-11-09)

**Pipeline** (`src/kcp/edge/e2e.py`), all on the Jetson:

```text
video file (OpenCV) -> fixed crop -> 224x224 (Pillow) -> 8-frame clip every 32 frames
  -> S3D encoder (TensorRT FP16) -> streaming head (TensorRT FP16, causal centering)
  -> cut / mix run detectors -> interpreter -> scheduler -> recommendations
```

- **Video:** `P07_webcam01_salat`, a T2 video (people and camera never used in training), 3 min 42 s, 104 steps. Its accuracy (52.9%) is typical of T2 (58.8% overall).
- **Software on the device:** JetPack's own Python 3.12, NumPy, OpenCV 4.8, Pillow, and TensorRT 10.16 Python bindings, plus pydantic for the scheduler (copied as files; nothing installed system-wide). No PyTorch, no ONNX Runtime, no virtual environment, no sudo.
- **Crop:** the same fixed box as in training. With a live camera, the box is set once when the camera is framed.
- **Reference:** the Mac runs the same downstream code (`e2e reference`) on the cached training embeddings with the ONNX head. `e2e compare` checks the two step by step.
- **Two runs:** as fast as possible, and paced at 15 fps like a live camera (`--realtime`), with `tegrastats` logging.

### Results

| Check | Value |
|---|---:|
| Steps with the same top class as the Mac | **104 / 104** |
| Task events the same as the Mac (task, status, time) | **4 / 4** |
| Encoder cosine similarity vs. training embeddings (worst / median) | 0.9990 / 0.99977 |
| Largest probability difference vs. Mac | 0.054 |
| Paced run: wall time for 221.9 s of video | 221.8 s (kept up) |
| Paced run: compute per step, p50 / p95 / max | 90 / 92 / 93 ms (4.3% of the 2,133 ms budget at p95) |
| Fast run: whole video | 6.7 s (33× real time); 80 ms per step at p95 |
| Average input power, paced run vs. idle | **4.66 W vs. 4.59 W (+0.07 W)** |
| Peak process memory | 784 MB; system RAM ≤ 2.9 of 7.3 GB |
| Max junction temperature | 48 °C |
| Python network calls | blocked by the run; 0 sockets open at the end |

**Per-step time in the paced run (p50):** decode 20 ms (32 frames) · crop and resize 22 ms · encoder 46 ms · head 1.5 ms · detectors and planner 0.1 ms.

**What the planner did:** the video has two real cutting periods (66–143 s and 186–211 s). The pipeline reported both:

| Event | Reported at | Dated to | True |
|---|---:|---:|---:|
| started "Cut the firm vegetables" | 94.3 s | 66.1 s | 66.1 s |
| complete "Cut the firm vegetables" (asks to confirm) | 171.1 s | 164.3 s | 143.0 s |
| started "Cut the garlic" | 213.8 s | 185.6 s | 185.6 s |
| complete "Cut the garlic" (asks to confirm) | 220.2 s | 219.7 s | 211.2 s |

The full planner transcript is in `reports/phase2b_e2e_transcript.txt`.

### What this means

- **The whole pipeline runs on the device in real time, with large headroom:** 92 ms per 2.1 s step, and it adds ~0.07 W on average. The earlier estimate ("well under 0.1 W") is now measured.
- **The live path gives the training path's answers.** OpenCV decoding, Pillow resizing, and TensorRT FP16 together move probabilities by up to 0.054 — more than FP16 alone (0.006), probably because OpenCV and PyAV decode and convert colors slightly differently (not tested) — but no step and no event changed.
- **Delay:** a start is reported ~28 s after it begins: 20 s minimum run, plus the head's 3-step look-ahead and the clip's last frames (6.9 s). Ends are dated 9–21 s after the true end and reported 9–28 s after it, mostly from the 16 s smoothing. For a planner that works in minutes, the starts are on time; the late ends cost one confirmation prompt each.
- **Per-step time is 9–13× the `trtexec` model time**, for two reasons: (1) decoding, cropping, resizing, and normalization run on the CPU (~50 ms; the encoder stage above includes ~10 ms of normalization); (2) the GPU lowers its clock between steps: measured alone, one encoder run every 2.1 s takes ~23 ms instead of 9 ms back to back. `sudo jetson_clocks` would fix (2) at a higher idle power. It is not needed: the budget is 23× the cost.

## Limits and what is still open

- `trtexec` times the model only. The end-to-end run above adds decoding, cropping, resizing, and Python.
- The speed runs use random inputs. Latency does not depend on the input values for this model; the accuracy check uses real inputs.
- The encoder check covers 16 clips from one video. The head check covers every T2 step.
- The end-to-end run covers one video (104 steps). The step-level match is stronger evidence than the 4 events, but one video is still a small sample.
- The video is breakfast prep and the recipe is a stir-fry, so the task names are the interpreter's guesses, as designed. No taps or voice input were used, so the planner kept recommending "Press the tofu".
- **Network:** the run blocked Python network calls and held no sockets, but Wi-Fi stayed on (switching it off needs root). A Wi-Fi-off run: `scripts/jetson_offline_run.sh`.
- The input is a file, not a camera. A USB camera (InnoMaker U20CAM-1080P, UVC) is the next input; OpenCV reads cameras through the same interface as files.

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

# End to end (pack: data/edge/e2e/ with video.avi + meta.json; src/kcp and pydantic copied to the Jetson)
PYTHONPATH=src:pylib python3 -m kcp.edge.e2e run e2e edge/s3d_selected \
    graphs/tofu_stir_fry.instant_pot.v0.json e2e/out_rt --realtime     # Jetson
uv run --extra edge python -m kcp.edge.e2e reference data/edge/e2e            # Mac
uv run python -m kcp.edge.e2e compare data/edge/e2e/out_rt data/edge/e2e/reference
```
