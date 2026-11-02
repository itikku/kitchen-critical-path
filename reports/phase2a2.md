# Phase 2a.2 — Better Perception (summary)

Date: 2026-11-01. Generated tables: `reports/phase2a2_results.md`. Raw metrics: `reports/phase2a2_metrics.json`. Design (written before any result): `IMPLEMENTATION_PLAN.md`, "Phase 2a.2".

## Question

Run 2's model kept its accuracy on an unseen camera but was weak overall (macro-F1 0.20), and weakest on cutting. Do a **work-area crop** and a **motion-aware video encoder** fix that?

## What changed

| Feature set | Encoder | Input | Steps |
|---|---|---|---|
| `mnv3_full` (run 2) | MobileNetV3-Large, ImageNet, frozen | whole frame, 320×240 | every 8 frames (0.53 s) |
| `mnv3_crop` | same | motion crop → 224×224 | every 8 frames |
| `s3d_crop` | S3D, Kinetics-400, frozen | motion crop → 8-frame clip, 224×224 | every 32 frames (2.1 s) |

- Same 1,121 videos and the same labels for all three sets; only the features differ.
- Motion crop: one square per video around where the image changes most (`docs/figures/motion_crops.png`). It uses no labels.
- For each feature set: the frame baseline and run 2's selected configuration (temporal head + causal centering). Six models; the winner chosen by validation macro-F1 only.
- Compute on the M1: 1 h 43 min to stream, crop, and encode all videos (the S3D GPU step is the limit, ~19.5 clips/s).

## Results

**Reproducibility check:** the two `mnv3_full` models reproduce run 2's numbers exactly (0.173 / 0.108 and 0.202 / 0.200).

### 8-class macro-F1

| Model | Val | T1: new people | T2: + unseen camera |
|---|---:|---:|---:|
| run 2 (`mnv3_full`, temporal + centering) | 0.234 | 0.202 | 0.200 |
| `mnv3_crop`, temporal + centering | 0.239 | 0.218 | 0.207 |
| `s3d_crop`, frame baseline | 0.325 | 0.292 | 0.321 |
| **`s3d_crop`, temporal + centering (selected)** | **0.423** | **0.397** | **0.417** |

### Planner metrics (selected model vs. run 2)

| Metric | Run 2, T1 | **Selected, T1** | Run 2, T2 | **Selected, T2** | Random guess |
|---|---:|---:|---:|---:|---:|
| Cutting AP | 0.144 | **0.384** | 0.156 | **0.418** | ~0.10 |
| Mixing AP (pour or stir) | 0.211 | **0.490** | 0.188 | **0.550** | ~0.15 |
| Cutting-event F1 | 0.08 | **0.35** | 0.15 | **0.43** | — |
| Cutting events found (TP / FP / FN) | 3 / 42 / 27 | 9 / 16 / 18 | 4 / 29 / 16 | 8 / 10 / 11 | — |

## What the results show

1. **The motion-aware encoder is the big win.** S3D roughly doubles macro-F1 (0.20 → 0.40–0.42) and cutting AP (0.14–0.16 → 0.38–0.42, about 4× random). This supports the diagnosis from Phase 2a: still-image features miss what distinguishes actions, which is motion.
2. **It still transfers to the unseen camera.** T2 is as good as T1 (0.417 vs. 0.397).
3. **Cropping alone helps a little** (0.202 → 0.218 on T1). The crops are coarse: they center on the person, because the whole body moves, and are often the full frame height.
4. **Cutting events are better but not solved.** The selected model finds about a third to a half of the true cutting runs (recall 0.33 on T1, 0.42 on T2), with fewer false alarms than run 2 (precision 0.36 and 0.44). The cook confirms each completion, so a missed or false run costs one prompt, not a wrong plan.
5. **Small event counts.** T1 has 27 true cutting runs and T2 has 19, so event F1 moves a lot with a few runs. The S3D frame baseline scores higher on T1 events (0.47) and lower on T2 (0.32); this difference is within that noise. The pre-registered selection rule (validation macro-F1) stands.

## Costs

- S3D steps are 4× sparser (2.1 s) than MobileNetV3 steps. That is fine for a planner that works with runs of 20 s or more.
- The live delay grows: the temporal head needs 3 future steps, so ~6.4 s instead of ~1.6 s.
- Training data per model is smaller in steps (35,629 vs. 141,708), because of the sparser grid.

## Next steps

1. **Jetson (Phase 2b):** benchmark the exported S3D model with TensorRT; check that one clip per 2.1 s fits easily.
2. **Tighter crops:** a hand detector instead of whole-body motion.
3. **More S3D steps:** a clip every 16 frames doubles the training steps; on the Jetson it costs twice the compute.
4. **Kitchen data (Phase 3):** the AI-labeled loop, starting from this model.

## Reproduce

```sh
uv sync --extra train
uv run kcp-train features      # streams the archive; ~1 h 45 min on an M1
uv run kcp-train train-2a2
uv run kcp-train evaluate-2a2
```
