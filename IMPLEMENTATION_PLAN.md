# Implementation Plan — Kitchen Critical Path (Phases 0–1)

> **Status:** DRAFT — waiting for user approval. No code exists yet.
> **Scope of this plan:** Phase 0 (specification) and Phase 1 (offline replay + scheduler). Phases 2–5 get their own plans later.
> **Background:** this plan follows a private discovery handoff (not in this repository). "Changes from the handoff" lists where it differs.

## Decisions made on 2026-11-01

| Decision | Choice | Reason |
|---|---|---|
| Recipe family | **Tofu-vegetable stir-fry + rice** | Strong critical-path problem: all prep must finish before the ~8-minute wok phase. Weeknight dinner is a high-pressure, time-boxed situation (meeting friends, a child's bedtime), so the value is easy to explain. No meat, so no food-safety rules. |
| Camera coverage | **Prep zone only** (cutting board + bowls) | The scheduler adds most of its value before the wok starts. Public data covers prep actions. This keeps the camera away from steam and oil. |
| Stove and wok events | **One tap per event** ("wok started", "done") | No public static-camera data has wok actions. A tap costs the user one second. |
| Public training data | **Breakfast Actions** (CC BY 4.0) | Unedited fixed-camera video in 18 kitchens. The most permissive license of the candidates. |
| Data improvement | **AI-reviewed labels for training data** (design below; built in Phases 2–3) | User requirement: improve the training set with own data, reviewed by AI, not by humans. |
| Rice | **Two graph variants:** frozen rice reheated (common), and Instant Pot | The user often cooks rice ahead and freezes it. The two variants have different critical paths, which makes a good comparison. |
| Vegetables | **Groups by cut and cook time**, not by vegetable | The user's vegetables vary: bok choy, Chinese broccoli, carrot, onion, garlic, tomato, broccoli. The scheduler needs groups, not a recipe for each vegetable. |
| Tofu | **Pressed** | The 20-minute press is passive time that the scheduler can fill with prep. |
| Training order | **Phase 2a (public-data training on the Mac) runs in parallel with Phase 1** | Measured on 2026-11-01: the Mac can do it (see "Phase 2a"). It needs only the Phase 0 label mapping. |
| User input | **Voice first, buttons as fallback.** In this plan, a "tap" means any user input event: a short voice command (primary) or a button press on the web page (fallback). Speech recognition and speech output run locally on the Jetson. Built in Phase 2b/3 with the web page. | User requirement (2026-11-01): hands are wet or dirty while cooking. This overrides the handoff's exclusion of voice work. |
| v1 evaluation | **Public data only:** held-out people in Breakfast Actions, plus 50 Salads as a second, unseen dataset. The user's 3-dinner test moves to Phase 3, after v1 is deployed. | User requirement: the first deployed version must not need the user to label anything. |
| Simulation | **Simulate cooking sessions to evaluate the scheduler** (Phase 1). Do not use generated video as test data. | Simulated event logs have exact labels by construction. Generated video does not (see "v1 evaluation without your labels"). |

### Changes from the handoff

1. **Primary dataset:** Breakfast Actions instead of YouCook2. YouCook2 is edited video (cuts, zooms), which a countertop camera never produces.
2. **Personal sessions:** no human labels before v1. After v1, human labels for **3 test sessions only**. All other personal sessions get AI labels (see "AI-review loop"). The handoff proposed 10–15 human-annotated sessions before evaluation.
3. **Recipe:** the handoff suggested vegetarian roasting recipes. The user chose a stir-fry for the reasons above.

## Assumptions (correct any that are wrong)

- The repo is a noncommercial portfolio project. It never commits or redistributes video.
- Your phone on a stand records the personal sessions. No camera decision is needed before Phase 3.
- You reheat frozen rice in the microwave, in about 3 minutes.
- The Instant Pot rice cycle takes about 30 minutes in total (pressure up, cook, natural release).
- Phase 1 runs on the Mac. It does not need the Jetson.
- Python 3.11+, `uv` for environments, `pydantic` for schemas, `pytest`, `ruff`, `mypy`. No other dependencies in Phase 1.

## AI-review loop (design only — built in Phases 2–3)

**Goal:** add your own sessions to the training set without a human labeling each event.

```text
new session (phone video + taps)
  -> student: small PyTorch model proposes events
  -> teacher: Cosmos-Reason2 labels the same clips offline, choosing only labels the task graph allows
  -> automatic checks:
       1. teacher and student agree
       2. the event is legal in the task graph (prerequisites met, order plausible)
       3. the duration is plausible for the task
       4. the event is consistent with nearby taps (taps are free ground truth)
  -> pass: add to the training pool, with provenance
  -> fail: discard (no human queue)
  -> retrain the student
  -> promote the new student ONLY if it does better on the fixed human-labeled test set
```

**Scale principle:** the loop must work in a kitchen where nobody ever labels anything. Mass deployment cannot depend on per-kitchen human labels. Each kitchen improves its own model from AI labels, and the system judges each kitchen's model with label-free signals only:
- teacher–student agreement rate;
- task-graph violation rate (events in an impossible order);
- agreement with taps;
- user correction rate and confirm rate.

The open question is whether these signals track real accuracy. The Phase 3 test set answers it **once**, in one kitchen: compare the label-free signals with the true accuracy from human labels. If they agree, the method can scale with no human labels. If they do not agree, the project reports that honestly. At scale, the equivalent check is a small central audit sample from opted-in kitchens, never a per-kitchen requirement. Fleet-level learning across kitchens (for example, federated learning) is roadmap only, not in this project.

**Promotion gate in v1 (before any user labels exist):** promote a new student only if it does better on the public test sets (held-out Breakfast people + 50 Salads) and the live correction rate does not increase. Risk: these checks cannot detect a mistake that appears only in the user's kitchen. The Phase 3 test set closes this gap.

**The one place a human must label (Phase 3, after v1):** the test set. If AI labels both the training data and the test data, the test cannot detect a mistake that the teacher and the student share. The test set is 3 recorded dinners. For each one, the user watches the video at 4× speed in a small labeling tool (built in Phase 3) and presses a key when each prep task starts and ends. Estimate: about 20 minutes per dinner, about 1 hour in total, done once. The tool does not show AI labels during this step, so they cannot bias the answers. Test sessions never enter the training pool.

**Known risks:**
- The teacher and the student can agree on the same wrong label. Checks 2–4 reduce this risk but do not remove it. The human test set is the only way to detect it.
- Cosmos-Reason2-2B runs on the Orin Nano through llama.cpp, according to NVIDIA's Jetson AI Lab. Its label accuracy for our label set and its speed on our clips are **unknown**. Phase 2 measures both. Fallback: run the same model on the Mac (unverified; check in Phase 2).
- Accept-rule thresholds are set after the first baseline, not before.

**What Phase 1 does for this loop:** every event records its source (`human`, `tap`, `public_dataset`, `student`, `teacher`, `user_correction`), the model version, and the confidence. This metadata is what lets Phases 2–3 filter and audit AI labels.

## Draft task graph v0 (to confirm in Phase 0)

Durations and windows are **authored heuristics**, not learned values.

**Shared tasks (both variants):**

| Task | Needs | Duration | Attention | Observed by |
|---|---|---|---|---|
| `press_tofu` | — | 20 min | passive | tap (no public data) |
| `mix_sauce` | — | 3 min | active | camera |
| `cut_aromatics` (garlic) | — | 4 min | active | camera |
| `cut_firm_veg` (carrot, broccoli, Chinese broccoli stems, onion) | — | 8 min | active | camera |
| `wash_greens` (bok choy, Chinese broccoli leaves) | — | 3 min | active | tap (sink is outside the prep zone) |
| `cut_leafy_veg` (bok choy, Chinese broccoli leaves) | `wash_greens` | 4 min | active | camera |
| `cut_tomato` | — | 2 min | active | camera |
| `cube_tofu` | `press_tofu` | 3 min | active | camera |
| `heat_wok` | all prep tasks | 3 min | passive | tap |
| `sear_tofu` | `heat_wok`, `cube_tofu` | 6 min | active | tap |
| `stir_fry_firm` (aromatics + firm vegetables) | `sear_tofu` | 5 min | active | tap |
| `stir_fry_leafy` (leafy vegetables + tomato) | `stir_fry_firm` | 3 min | active | tap |
| `combine_and_sauce` | `stir_fry_leafy`, `mix_sauce` | 2 min | active | tap |
| `plate` | `combine_and_sauce`, rice task | 2 min | active | tap |

**Rice variant A — frozen rice (`tofu_stir_fry.frozen_rice.v0.json`):**

| Task | Needs | Duration | Attention | Observed by |
|---|---|---|---|---|
| `reheat_rice` | — | 3 min | passive | tap |

**Rice variant B — Instant Pot (`tofu_stir_fry.instant_pot.v0.json`):**

| Task | Needs | Duration | Attention | Observed by |
|---|---|---|---|---|
| `rinse_rice` | — | 2 min | active | tap |
| `cook_rice_instant_pot` | `rinse_rice` | 30 min | passive | tap (start) |

**Quality windows (heuristic):**
- `cut_tomato`: start no earlier than 20 min before `plate` (cut tomato releases juice).
- `reheat_rice`: finish no earlier than 5 min before `plate` (it cools).
- `cook_rice_instant_pot`: finish no earlier than 20 min before `plate`. The Instant Pot keeps rice warm, so this window is wide.

**What the variants show:** with frozen rice, your hands are the bottleneck. About 24 minutes of active prep is longer than the 20-minute tofu press. With the Instant Pot, the 30-minute rice cycle is the longest path, so the first recommendation is "start the rice."

## v1 evaluation without your labels

| Question | v1 test (public or simulated) | What it cannot show |
|---|---|---|
| Does the model recognize prep actions done by people it never saw? | Breakfast Actions: hold out whole people (the dataset's standard splits do this). | Your kitchen, your camera angle. |
| Does it work from a camera position it never saw? | Breakfast T2: new people, filmed from a camera view removed from training. (50 Salads was the plan; its server is down as of 2026-11-01, so T2 replaces it until it returns.) | Stir-fry vegetables specifically. |
| Does the scheduler save time? | Simulated sessions (Phase 1, step 1.7). | Real cooking behavior. |
| Does it help in your kitchen? | Live signal: how often you correct it, and how often you confirm its prompts. | Exact accuracy. A correction rate counts only the mistakes you notice. |

**The v1 claim:** "Tested on people and kitchens it never saw, and on a second dataset. Not yet measured in the developer's own kitchen." The Phase 3 test set later adds the own-kitchen measurement.

**Why not generated video as test data:** a video generator (for example, Cosmos Predict) can show the wrong action, or show it badly. Then the "correct answer" for that clip is wrong, and the test measures the generator, not our model. One later option is label-safe: restyle real public test clips (new lighting, countertop, colors) with Cosmos Transfer. The action stays the same, so the label stays correct. This needs a large data-center GPU, so it is not in v1. It would need its own cost request.

**Scheduling rules (simple and inspectable):**
1. A task is eligible when all its prerequisites are complete.
2. The user does one active task at a time. Passive tasks run in the background.
3. Start eligible passive tasks first (they use waiting time).
4. Defer a task whose quality window has not opened yet.
5. Rank the remaining active tasks by the longest remaining path to `plate` (critical path).
6. If a required state is `unknown`, recommend "confirm X" before tasks that depend on it.

---

## Phase 0 — Specification and feasibility

**Goal:** record every decision Phase 1 needs, and check access and terms. No code.

| # | Step | Verify |
|---|---|---|
| 0.1 | Confirm the task durations and the vegetable groups in the task graph v0. (Rice method, vegetables, and tofu pressing are answered.) | You approve the task tables. |
| 0.2 | Download only the Breakfast Actions **annotation files** (small; no video). Read the real label list. | Label list saved in `docs/breakfast_labels.md`, with file sizes and checksums. |
| 0.3 | Draft the prep-zone label vocabulary and the Breakfast → ours mapping table. List unmapped labels. | `docs/label_mapping.md` exists. Each of our camera labels has ≥1 Breakfast source label, or is marked "no public data". |
| 0.4 | Confirm the Breakfast Actions license and citation on the official page. | License, URL, citation, and access date in `docs/data_provenance.md`. |
| 0.4b | Download only the 50 Salads **annotation files** (5 MB). Confirm its label list, camera view, and license (CC BY-NC-SA 4.0; acceptable for a noncommercial test set). | Label list, camera view, and license added to `docs/data_provenance.md` and `docs/label_mapping.md`. |
| 0.5 | Choose the phone placement. Record one 2-minute test clip of the prep zone. | You confirm the board, bowls, and your hands are visible. The clip stays local. |
| 0.6 | Write the privacy plan: local storage, `.gitignore` for `sessions/`, deletion steps, no upload without approval. | `docs/privacy.md` exists. |
| 0.7 | Jetson inventory (read-only), after the Jetson setup is complete. | Output of the commands below saved in `docs/jetson_inventory.md`. **Blocked** until setup. Phase 1 does not wait for it. |
| 0.8 | Confirm the AI-review design and the 3-session human test set. | You approve. |

**Jetson inventory commands (read-only):**

```sh
cat /etc/nv_tegra_release
nvpmodel -q
free -h
df -h /
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls /dev/video*
```

**Access needed for Phases 0–1 (one consolidated list):**

| Item | Purpose | Cost | Data sent out |
|---|---|---|---|
| Breakfast Actions annotation files (official site) | Label list and adapter tests | Free | None |
| 50 Salads annotation files (Univ. of Dundee site, 5 MB) | Second test set: label list | Free | None |
| PyPI packages: pydantic, pytest, ruff, mypy (via uv) | Phase 1 code | Free | None |

No accounts, paid services, purchases, or uploads are needed for Phases 0–1.

**Phase 0 acceptance criteria:**
1. Task graph v0 approved.
2. Label vocabulary and mapping table drafted from the real Breakfast label list.
3. License and citation recorded.
4. Privacy plan written.
5. Phone placement confirmed with one local test clip.
6. AI-review design and the test-set rule approved.
7. Jetson inventory recorded, or marked "blocked: setup pending".

**Approval gate:** you review the Phase 0 documents before Phase 1 starts.

---

## Phase 1 — Offline replay + scheduler

**Goal:** prove the core value with no ML. Feed an event log to the scheduler and show the recommendations change as the session runs.

### Layout

```text
pyproject.toml
src/kcp/
  schema.py          # Event, Task, TaskGraph (pydantic)
  scheduler.py       # rules 1–6 above; returns ranked tasks + reasons
  replay.py          # steps through an event log, prints the timeline
  cli.py             # `kcp replay`, `kcp breakfast-map`
  adapters/breakfast.py   # Breakfast annotations -> our events + mapping report
data/
  graphs/tofu_stir_fry.frozen_rice.v0.json
  graphs/tofu_stir_fry.instant_pot.v0.json
  label_maps/breakfast.v0.json
tests/
  fixtures/sessions/*.jsonl         # hand-written stir-fry sessions
  fixtures/breakfast_format/*.txt   # synthetic lines in Breakfast's format (no real data)
  expected/*.txt                    # golden replay outputs
  test_schema.py, test_scheduler.py, test_replay.py, test_breakfast_adapter.py
sessions/            # personal data; in .gitignore
```

### Event schema

```json
{
  "event_id": "s004-e012",
  "session_id": "session_004",
  "t_start_s": 58.0,
  "t_end_s": 121.0,
  "task_id": "cut_hard_veg",
  "status": "complete",
  "source": "human",
  "source_version": null,
  "confidence": null,
  "corrects_event_id": null
}
```

`status` is one of `started`, `complete`, `unknown`. `source` is one of `human`, `tap`, `public_dataset`, `student`, `teacher`, `user_correction`.

### Hand-written fixture sessions

| Fixture | What it tests |
|---|---|
| Fixture | Graph variant | What it tests |
|---|---|---|
| `frozen_on_time.jsonl` | frozen rice | The normal path. The tofu press starts first. The rice reheats near the end. |
| `ip_on_time.jsonl` | Instant Pot | The normal path. The rice starts first. |
| `ip_late_rice.jsonl` | Instant Pot | The rice starts 10 min late. The plan changes, and the reason names the delay. |
| `early_wok.jsonl` | frozen rice | A tap says "wok started" before prep is complete. The tool warns about the unmet prerequisites. |
| `unknown_state.jsonl` | frozen rice | A low-confidence event. The tool asks "confirm X?" |
| `user_correction.jsonl` | frozen rice | The user corrects a wrong event. The plan recomputes. |
| `early_tomato.jsonl` | frozen rice | The tomato is cut too early. The tool reports the missed quality window. |

### Steps

| # | Step | Verify |
|---|---|---|
| 1.1 | Create the project skeleton with `uv`, `ruff`, `mypy`, and `pytest`. | `uv run pytest` runs (0 tests). |
| 1.2 | Write the schemas and the task graph JSON. | `test_schema.py`: the graph loads; invalid events fail with clear messages; a graph with a cycle fails. |
| 1.3 | Write the scheduler. | `test_scheduler.py`: one test per scheduling rule 1–6, plus recovery after a delay. |
| 1.4 | Write the replay CLI. | `test_replay.py`: each fixture's output matches its golden file in `tests/expected/`. |
| 1.5 | Write the Breakfast adapter. | `test_breakfast_adapter.py` passes on the synthetic fixtures. A manual run on the real annotation files prints a mapping report (counts per label, unmapped labels). |
| 1.6 | Write a short README section: how to run the replay. | A fresh clone runs the replay with the commands below. |
| 1.7 | Write a session simulator: a simulated cook follows either the scheduler or the plain recipe order. Task durations vary at random (seeded) around the graph values, with occasional delays. Report total time, idle hands time, and quality-window misses for both. | `test_simulate.py`: the same seed gives the same results; a simulated session never breaks a prerequisite. `kcp simulate` prints the comparison for 200 sessions per graph variant. |

### Verification commands

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run kcp replay tests/fixtures/sessions/ip_late_rice.jsonl --graph data/graphs/tofu_stir_fry.instant_pot.v0.json
uv run kcp breakfast-map --annotations <local-path-to-breakfast-annotations>
uv run kcp simulate --graph data/graphs/tofu_stir_fry.frozen_rice.v0.json --sessions 200 --seed 0
```

### Phase 1 acceptance criteria

1. All verification commands pass.
2. The replay shows, for each event: the current state, the recommended next task, and a one-line reason based on the graph.
3. Each fixture produces the behavior in its table row, checked by a golden-output test.
4. Every event carries `source`, `source_version`, and `confidence`.
5. The Breakfast adapter produces schema-valid events and a mapping report from the real annotation files.
6. No personal video or credentials are in the repo.
7. The simulator reports, for each graph variant, total time, idle hands time, and quality-window misses for the scheduler and for the plain recipe order. No target improvement is assumed before the first run.

**Approval gate:** you review the replay output for `ip_late_rice` and `early_wok`.

## Phase 2a — First PyTorch training run on the Mac (in parallel with Phase 1)

PyTorch training is a required part of this project. Phase 2a starts after Phase 0 (it needs the label mapping) and runs at the same time as Phase 1.

**Inputs from Phase 0:** 8 camera labels (`docs/label_mapping.md`); splits T1 and T2; the Breakfast video archive (3.9 GB, downloaded 2026-11-01).

| # | Step | Verify |
|---|---|---|
| 2a.1 | Extract the video archive. Save one sample frame per camera view. Choose the T2 held-out camera ID by coverage (done: `webcam01`; see `docs/label_mapping.md`). | Sample frames in `docs/figures/` (public data, so they may be committed). T2 view recorded in `docs/label_mapping.md`. |
| 2a.2 | Build a frame index: sample each video at 2 fps; give each frame its mapped label. Never split one video between train and test. | `test_index.py`: every frame has one of the 8 labels; no person appears in both train and test; no T2-view frame is in train. |
| 2a.3 | Compute embeddings with a frozen, ImageNet-pretrained **MobileNetV3-Large** (fp16, MPS, batch 64). Save in chunks to `data/cache/`. DINOv2-S is a second run only if time allows. | Cache row count = index row count. Peak memory ≤ 3 GB. |
| 2a.4 | Baseline: a linear classifier on single-frame embeddings (no time context). | Metrics JSON for T1 and T2. |
| 2a.5 | Train the temporal head: 8-frame windows (~4.3 s), label = label of the center frame; class-balanced loss; seeded. Choose the configuration and epoch on validation people (s4) only. | Training loss goes down. Checkpoint, config, and validation log saved. |
| 2a.6 | Evaluate on T1 and T2: macro F1, per-class precision and recall, confusion matrix; compare with the 2a.4 baseline. | `kcp-train evaluate` writes `reports/phase2a_metrics.json` and `reports/phase2a.md`, including failures. |

New dependencies: `torch`, `torchvision`, `timm`, `av`, `numpy`, `scikit-learn`. One download: MobileNetV3-Large ImageNet weights through `timm` (about 20 MB, free).

**Acceptance criteria:**
1. One command rebuilds the index, the cache, the training run, and the report from the raw archive.
2. The report gives T1 and T2 results for both the baseline and the temporal head. No target score is set before the first run.
3. The report states the main limits: fruit-only cutting data, no stir-fry footage, 320×240 video.

Phase 2b (after Phase 1 and the Jetson setup) exports the model, benchmarks it on the Jetson (p50/p95 latency, memory), and connects its events to the scheduler.

### Feasibility evidence (measured 2026-11-01)

Machine: MacBook Pro, Apple M1 (8 GPU cores), 8 GB memory, 134 GB free disk. PyTorch 2.15.0 with the MPS (Mac GPU) backend. Synthetic data and untrained weights, which run at the same speed as trained weights. Benchmark script: scratchpad only for now; it moves into the repo in Phase 2a.

| Step | Measured speed | Full Breakfast set (~77 h, sampled at 2 frames/s ≈ 554k frames) |
|---|---|---|
| Decode video (320×240 H.264, CPU) | ~5,900 frames/s | ~12 min (all 15 fps frames must be decoded) |
| Embeddings: MobileNetV3-Large (4.2M params), fp16 | ~456 frames/s, 1.1 GB GPU memory | ~20 min, one time |
| Embeddings: DINOv2 ViT-S/14 (21.6M params), fp16 | ~68 frames/s, 1.1 GB GPU memory | ~2.3 h, one time |
| Train temporal head (0.5–0.7M params, 16-frame windows) | ~7,600–8,900 windows/s | ~1 min per epoch; ~35 min for 30 epochs |

Cache size: ~0.4 GB (DINOv2, 384-dim) or ~1.4 GB (MobileNetV3, 1280-dim) in fp16.

**Conclusion:** the Mac can run the full first training cycle. The worst case is about 3 hours, run once. The selected subset is smaller than the full set, so the real time is shorter.

**Not yet verified:**
- Breakfast Actions video format. The benchmark assumes 320×240 at 15 fps, from the original paper. Phase 0 confirms it.
- Real decode speed on the actual files, and sustained speed over hours (heat).
- Which encoder gives better labels. Phase 2a compares the two on the same split.
- Speed on the Jetson. Phase 2b measures it.

Fine-tuning the encoder itself (instead of freezing it) is much slower and is not planned.

### Mac safety limits

- **Memory:** the measured peak is 2.2 GB of 8 GB. Use batch size 64 or lower, and fp16.
- **Disk:** use about 10 GB at most (video subset + cache + checkpoints). 134 GB is free. All data stays in one project folder that is easy to delete.
- **Heat and use:** run the embedding step overnight or while the Mac is not in use, at low priority (`nice`). Save the cache in chunks, so a stop loses at most one chunk.
- **Cloud:** not needed for Phase 2a. Reconsider only if the measured real-data speed is more than 3× slower than this benchmark. Any cloud use needs a separate cost and access request.

## Phase 2a.2 — Better perception (approved 2026-11-01)

**Goal:** raise the planner-relevant accuracy (above all `cut`) without changing the test sets. Every choice below is fixed before the first 2a.2 result.

**Two changes, tested separately and together:**
1. **Crop to the work area.** For each video, find where motion happens (sum of frame differences on a small grayscale copy), and crop a square around it. No labels are used, so this is allowed on test videos. On the Jetson, the camera is framed on the prep zone at setup, so no crop is needed there.
2. **A motion-aware encoder:** S3D, pretrained on Kinetics-400 (8.3M parameters; measured ~17 clips/s on the M1 at 16×224×224). We use 8-frame clips (frame stride 2, ~1.07 s) at 224×224, one clip every **32** video frames (~2.1 s), and take the 1024-dim pooled features. (Changed from every 16 frames before any run: S3D measured ~19.5 clips/s at any batch size, so every 16 frames would take ~2.5 h.)

**Feature sets:** `mnv3_full` (existing), `mnv3_crop`, `s3d_crop`, all on **exactly the run-2 videos and labels**, so only the features differ. Decoding streams each video from the archive in memory; no video is extracted to disk.

**Configurations:** for each feature set, `frame_linear` and `temporal_head_centered` (run-2 settings). Selection: best validation macro-F1. T1 and T2 are evaluated once, after selection.

**Planner-level metrics (new; reported for the selected model and the run-2 model):**
- Binary average precision for **cutting** (`cut`) and **mixing** (`pour` or `stir`), from the model's class probabilities.
- **Event-level F1 for cutting runs:** smooth the per-step cut probability with a moving average, threshold it, and keep runs of ≥ 20 s. A predicted run matches a true run when their overlap (IoU) is ≥ 0.25. The smoothing width and threshold are chosen on validation only.

**Acceptance:** one command per step; results in `reports/phase2a2.md`; run 2 stays the comparison point; failures reported.

## Phase 2b — Edge pipeline (approved 2026-11-01)

**On the Mac now:**
1. **Streaming perception:** frames in → encoder → rolling window → causal centering → head → smoothed action events (`ActionEvent`). The same code runs offline on a video file and live on a camera.
2. **Action → task interpreter:** the rule in `docs/label_mapping.md` (credit actions to the recommended task; ask "Done with <task>?"). Unit tests with synthetic action streams.
3. **ONNX export** of the encoder and the head, with a parity test (ONNX output = PyTorch output within tolerance).
4. **Jetson benchmark script:** p50/p95 latency, throughput, and memory with ONNX Runtime and TensorRT.

**On the Jetson (after setup):** run the benchmark and the streaming pipeline on a Breakfast video with the network off. Record the results in `reports/phase2b.md`.

## Voice input design (Phase 2b/3 — not built in Phases 0–2a)

- **Fixed command list, not open conversation:** "yes", "no", "rice started", "tofu pressing", "greens washed", "wok on", "done", "what's next?", and "no, I'm cutting <task>". A small fixed list is more reliable in a noisy kitchen than free speech, and it needs no language model.
- **Recognition:** a small Whisper model, run locally on the Jetson. Its output is matched to the command list. If no command matches, the system asks again and shows the buttons.
- **Speech output:** the system speaks the next task and its prompts ("Start the rice now"; "Done with the firm vegetables?") with a local text-to-speech model.
- **Microphone:** the phone or tablet that shows the web page sends its microphone audio to the Jetson. No audio goes to a cloud service. This rules out the browser's built-in speech recognition, because some browsers send audio to their vendor's servers.
- **Buttons stay on the page** as the fallback.

**Risks to measure (Phase 2b/3):**
1. Kitchen noise (exhaust fan, wok sizzle) lowers recognition accuracy. Measure the command accuracy with the fan on.
2. Browsers allow microphone access only on secure (HTTPS) pages. The Jetson page needs a local certificate.
3. Memory: speech recognition shares the Jetson's 8 GB with the vision model. Measure both together.
4. Whisper speed on the Orin Nano is not yet measured.

## Not in Phases 0–1

These items belong to later phases or are out of the project scope:

- Later phases: ML training and video downloads (Phase 2a), the Jetson deployment (Phase 2b), personal sessions (Phase 3), Cosmos integration (Phases 2–4), and any UI beyond the CLI.
- Out of the project scope: meat recipes, food safety, and appliance control.
- Git commits happen only when you ask.
