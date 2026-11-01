# Phase 2a — First PyTorch Perception Model (summary)

Date: 2026-11-01. Generated tables: `reports/phase2a_results.md` (run 2) and `reports/phase2a_run1.md` (run 1). Raw metrics: `reports/phase2a_metrics.json`.

## Question

Can a small PyTorch model on a frozen image encoder recognize prep-zone actions (cut, peel, pour, stir, transfer, take object, background, other) for **people it never saw**, and from a **camera position it never saw**?

## Data and pipeline

- Breakfast Actions (CC BY 4.0): 1,389 videos, 52 people, 18 kitchens, 320×240 at 15 fps. Sampled every 8th frame (1.875 frames/s): **345,175 frames**.
- Frozen ImageNet MobileNetV3-Large, native resolution, fp16 on the Mac GPU (~135 frames/s on an M1). Embeddings computed once (892 MB cache).
- Two PyTorch heads: `frame_linear` (one frame, 10k parameters) and `temporal_head` (1D convolutions over an 8-frame, ~4.3 s window, 724k parameters).
- Optional **causal per-video centering**: subtract the running mean embedding of the video so far. This removes the static look of the kitchen. It uses only past frames, so it works live on the Jetson.
- Class-weighted loss (inverse square-root frequency). Macro-F1 over 8 classes is the main metric, because 54% of frames are `other`.

## Splits

| Split | People | Camera IDs | Videos | Frames | Use |
|---|---|---|---:|---:|---|
| train | s2–s3 (P16–P41) | all except `webcam01` | 549 | 141,708 | training |
| val | s4 (P42–P54) | all except `webcam01` | 320 | 84,399 | choose configuration and epoch |
| T1 | s1 (P03–P15) | all except `webcam01` | 155 | 38,788 | test: new people |
| T2 | s1 (P03–P15) | `webcam01` only | 97 | 24,569 | test: new people + unseen camera ID |

## Results (run 2)

Four configurations were fixed before training. The validation score chose the winner; T1 and T2 were evaluated once, after the choice.

| Model | Val macro-F1 | T1 macro-F1 | T2 macro-F1 | T2 accuracy |
|---|---:|---:|---:|---:|
| majority class (`other`) | — | 0.088 | 0.087 | 53% |
| `frame_linear` | 0.211 | 0.173 | 0.108 | 14% |
| `frame_linear_centered` | 0.183 | 0.176 | 0.157 | 21% |
| `temporal_head` | 0.223 | 0.192 | 0.117 | 21% |
| **`temporal_head_centered`** (selected) | **0.234** | **0.202** | **0.200** | 37% |

## What the results show

1. **Time context plus centering transfers to a new camera.** The selected model loses almost nothing from T1 to T2 (0.202 → 0.200). The frame baseline loses 38% (0.173 → 0.108). Centering removes the kitchen's appearance, which differs most between cameras.
2. **Absolute accuracy is low.** Macro-F1 0.20 is 2.3× the majority baseline, but it is not good enough to drive the planner by itself.
3. **`cut` is weak, and it matters most.** Cut F1 is 0.13 (T1) and 0.16 (T2) for the selected model, below the frame baseline (0.25 and 0.20). `transfer` and `peel` are near zero.
4. **The model still overfits.** Validation peaks at epoch 1–4, while training loss keeps falling.

## Likely cause

The frozen encoder sees the **whole kitchen** at 320×240. Hands and a knife fill a small part of the frame, and the encoder was trained on still images, not motion. Most published Breakfast results use video-pretrained features (for example I3D), which encode motion.

## Next steps (Phase 2a.2 — for approval, not started)

1. **Crop to the work area.** The planned camera frames the prep zone tightly; Breakfast's wide shots do not. Crop Breakfast frames around the work surface to match.
2. **Use a motion-aware, video-pretrained encoder,** and benchmark it on the Jetson before choosing it.
3. **Match the labels to what the planner needs:** "cutting vs. not cutting" and "mixing (pour or stir) vs. not." Report those binary scores.
4. **Measure at the event level.** The planner needs "a cutting run of 30+ seconds happened," not a label for every frame. Smooth predictions over time and score detected events.

## Run 1 (kept for the record)

Run 1 (`reports/phase2a_run1.md`) had two flaws, found after it ran:
- The held-out camera, `webcam02`, exists only for 2 test people and in no training video. That T2 had 15 videos, and holding the camera out removed nothing from training. Run 2 chose `webcam01` by coverage alone.
- No validation set, and a fixed 10 epochs. The temporal head overfit (training loss 0.05) and scored below the frame baseline on T1 (0.159 against 0.221).

## Reproduce

```sh
uv sync --extra train
uv run kcp-train embed      # needs the Breakfast videos; see docs/data_provenance.md
uv run kcp-train train
uv run kcp-train evaluate
```
