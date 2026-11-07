# Project State — Kitchen Critical Path

Last updated: 2026-11-07

## Current status

- Phase 0: complete except 0.5 (phone placement, user). See the table below.
- **Phase 1: complete.** All acceptance criteria pass (see "Verification").
- **Phase 2a: complete** (`reports/phase2a.md`). Two runs; run 2 is the result of record.
- **Phase 2a.2: complete** (`reports/phase2a2.md`). Selected model: S3D clips on motion crops + temporal head with causal centering.
- **Phase 2b: Mac side complete; Jetson speed and accuracy complete** (`reports/phase2b.md`, 2026-11-05). Open: the full streaming pipeline on a Breakfast video on the Jetson, network off.
- **Phase 2a.3: design and code committed; feature run paused** (the Mac ran out of memory with two heavy jobs; resume one job at a time when the user says the Mac is free).
- Public repo: https://github.com/itikku/kitchen-critical-path (pushed 2026-11-01). `HANDOFF.md` and `docs/LEARNING_GUIDE.md` are private, git-ignored, and were removed from the public history.

## Phase 0 results

| Step | Status | Output |
|---|---|---|
| 0.1 Task graph | Done. Rice times confirmed by the user; other durations are authored heuristics. | `data/graphs/` |
| 0.2 Breakfast labels | Done. 48 coarse labels, 52 persons, 1,712 annotation files. Checksums match. | `docs/breakfast_labels.md` |
| 0.3 Label mapping | Done. 8 camera labels. `wash_greens` and `press_tofu` use user input (no public data). | `docs/label_mapping.md` |
| 0.4 License and citation | Done. CC BY 4.0 (dataset card). | `docs/data_provenance.md` |
| 0.4b 50 Salads | Blocked: official server down; mirror has features only. Substitute: T2. | `docs/data_provenance.md` |
| 0.5 Phone placement | Open (user). Needed before Phase 3 only. | — |
| 0.6 Privacy plan | Done. | `docs/privacy.md` |
| 0.7 Jetson inventory | Done (2026-11-05). JetPack 7.2.1, TensorRT 10.16.2, no PyTorch, no swap, no camera. | `docs/jetson_inventory.md` |
| 0.8 AI-review design | Approved. | `IMPLEMENTATION_PLAN.md` |

## Phase 1 results (`reports/phase1.md`)

- Frozen rice: scheduler 2.1 min faster per session (never slower); sessions with a window miss 96% → 46%.
- Instant Pot: same total time (−0.1 min); rice-freshness misses 60% → 0%; any-miss sessions 78% → 34%.
- All remaining scheduler misses are the tomato (median overshoot 2 min). Structural: the tomato is cut before the ~19-min wok phase. The window was not tuned after the run.
- Product implication: lead the pitch with timing quality and recovery, not minutes saved (Risk 2).

## Phase 2a results (`reports/phase2a.md`)

- 1,389 Breakfast videos, 345,175 frames embedded on the M1 (~40 min).
- Selected by validation: temporal head + causal per-video centering. Macro-F1 (8 classes): T1 0.202, T2 0.200. Frame baseline: 0.173 / 0.108. Majority: 0.088 / 0.087.
- Strength: no loss on the unseen camera ID. Weakness: low absolute accuracy, and `cut` F1 only 0.13–0.16.
- Run 1 flaws (fixed in run 2): held-out camera `webcam02` covered only 2 of 13 test people (15 videos) and removed 323 videos from training; no validation set, so the temporal head overfit.
- Proposed Phase 2a.2: crop to the work area, motion-aware encoder, planner-level binary labels, event-level scoring.

## Phase 2a.2 results (`reports/phase2a2.md`)

- Macro-F1 (8 classes): T1 0.397, T2 0.417 (run 2: 0.202 / 0.200). Cutting AP 0.384 / 0.418 (random ≈ 0.10). Mixing AP 0.490 / 0.550.
- Cutting-event F1: 0.35 / 0.43 (run 2: 0.08 / 0.15). Small counts: 27 and 19 true runs.
- The two `mnv3_full` models reproduce run 2 exactly.
- Motion crops are coarse (they center on the whole person); cropping alone added ~0.01–0.02 macro-F1.

## Phase 2b (Mac side)

- Streaming head = batch evaluation (tested). ONNX = PyTorch within 1e-6 (MobileNetV3 and S3D).
- Mac CPU, ONNX Runtime: S3D encoder 86 ms p95 + head 0.09 ms per 2.1 s step (4% of real time). `reports/phase2b_mac_benchmark.json`.
- Exported model: `data/edge/s3d_selected/` (not in git; recreate with `kcp.edge.export`).

## Phase 2b on the Jetson (`reports/phase2b.md`)

- TensorRT 10.16.2, MAXN_SUPER. Encoder p95: FP32 11.29 ms, FP16 7.22 ms. Head p95: 0.19 / 0.12 ms.
- One FP16 step ≈ 7.3 ms = 0.34% of the 2,133 ms budget; ~12× faster than the Mac CPU.
- Accuracy: all 6,179 T2 predictions match the Mac in FP32 and FP16; macro-F1 0.4167 = Mac. Encoder cosine ≥ 0.99999 on 16 real clips.
- Back-to-back power: 17–18 W (encoder), 5.2 W idle; max junction 56.8 °C; RAM used ≤ 4.0 GB.
- ONNX Runtime GPU not installed (TensorRT runs the model; no build verified for JetPack 7.2).
- Device files: `~/kcp/` on the Jetson (engines, logs, parity pack). Not in git.

## Verification (2026-11-07)

```text
uv run ruff format --check .   -> all files formatted
uv run ruff check .            -> All checks passed
uv run mypy                    -> no issues in 24 source files
uv run pytest                  -> 44 passed
```

## Implementation notes

- The `.avi.labels` files inside the video archive contain two extra labels (`walk_in`, `walk_out`). The pipeline uses the official coarse annotation files instead.
- Breakfast camera IDs do not mean the same position in each kitchen. T2 = camera ID `webcam01` (run 2), never used in training; run 1 used `webcam02`.
- Embedding speed on the M1: ~135 frames/s at native 320×240 (MPS, fp16, fixed batch shape). Embeddings verified identical (cosine 1.0) before and after the speed fix.
- Some Breakfast videos are 320×180; the encoder accepts any size.

## Decisions

- Recipe family: tofu-vegetable stir-fry, two rice variants. Vegetables grouped by cut and cook time. Tofu pressed.
- Camera covers the prep zone only. Stove, wok, sink, and tofu-press events come from user input: voice first, buttons as fallback (Phase 2b/3).
- Public training data: Breakfast Actions (CC BY 4.0).
- Scale principle: no per-kitchen human labels.
- v1 tests: T1 (new people), T2 (new people + unseen camera ID), simulated sessions.
- Phase 3: a 3-dinner human test set, used once to validate the label-free signals.

## Disk use

- Archive `data/raw/breakfast/BreakfastII_15fps_qvga_sync.tar.gz`: 3.7 GB, **keep** (user decision).
- Extracted videos: deleted after the embedding run (2026-11-01).
- Embedding caches: 892 MB (`mobilenetv3_large_100`), 711 MB (`mnv3_crop`), 145 MB (`s3d_crop`). Python environment: ~1 GB. Model runs and ONNX exports (`data/runs*/`, `data/edge/`): small.

## Blockers

- 50 Salads server down. Affects only the cross-dataset test.

## Next step

- Jetson: run the full streaming pipeline (decode → crop → encoder → head → run detector → interpreter) on a Breakfast video with the network off; record end-to-end latency per step. Needs a video decoder in a virtual environment on the device.
- Phase 2a.3 (approved, paused): finish `features-2a3 --no-wrist` (~1 h), then the wrist feasibility gate, then train and evaluate. One heavy job at a time on the Mac.
