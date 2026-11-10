# Kitchen Critical Path

**Mise en place is a compiler.** A critical-path planner for cooking that watches what has actually happened, tracks dependencies and quality windows, and recommends the next best task.

Target hardware: NVIDIA Jetson Orin Nano Super (8 GB). Current status: the full pipeline — video in, planner recommendations out — runs on the Jetson in real time from a recorded video, with the same answers as the Mac. A live camera and the developer's own kitchen come next.

**New to video models?** Read the [visual explainer](https://itikku.github.io/kitchen-critical-path/how_it_works.html): how the model was trained, packaged, and tested on the Jetson, with pictures.

## What it does

Given a structured recipe and the events of a cooking session, the planner tells you what to do next and why:

```text
[17:00] complete 'Cut the leafy vegetables'
        next: Cook the rice in the Instant Pot — Start it now: it runs by itself for 30 min while you do other work.
        hold: Cut the tomato — wait: start when 'plate the food' is ~25 min away (now ~30 min)
        plate ≈ 49:00 · bottleneck: cook the rice in the instant pot (30 min left)
```

It is not a recipe chatbot or a timer. It replans from what actually happened: a late start, a step done out of order, or a correction.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                      # 51 tests
uv run kcp replay tests/fixtures/sessions/ip_late_rice.jsonl \
    --graph data/graphs/tofu_stir_fry.instant_pot.v0.json
uv run kcp simulate --graph data/graphs/tofu_stir_fry.frozen_rice.v0.json --sessions 200
```

## How it works

```text
camera (prep zone) ──► PyTorch action model ──► action events ─┐
voice / button ────────────────────────────────► task events ──┼─► scheduler ──► next task + reason
recipe task graph (dependencies, durations, quality windows) ──┘
```

**Scheduler rules** (`src/kcp/scheduler.py`): a task needs its prerequisites; one active task at a time; start passive tasks (timers, machines) first; hold a task until its quality window opens, just in time for passive tasks; rank the rest by the longest remaining path; ask to confirm a low-confidence state.

**Perception** (`src/kcp/perception/`): each video is cropped to where motion happens; a frozen, Kinetics-pretrained S3D video encoder turns 8-frame clips into embeddings, computed once; a small PyTorch temporal head (0.66M parameters) classifies 8-step windows into 8 prep-zone actions (cut, peel, pour, stir, transfer, take object, background, other). It trains on [Breakfast Actions](https://serre.lab.brown.edu/breakfast-actions-dataset.html) (CC BY 4.0).

**Edge** (`src/kcp/edge/`): a NumPy-only streaming head that matches the offline evaluation (tested), a run detector, an interpreter that turns actions into task events and asks the cook to confirm completions, ONNX export with parity checks, and the end-to-end pipeline (`e2e.py`): OpenCV decoding, TensorRT FP16 for both models, and the planner, with no PyTorch on the device. A live viewer (`viewer.py`) shows the video, the model's scores, and the planner's next step in a browser, from a recorded video or a USB camera.

## Results so far

| Result | Value | Details |
|---|---|---|
| Simulated sessions, frozen-rice variant | 2.1 min faster; window-miss sessions 96% → 46% | `reports/phase1.md` |
| Simulated sessions, Instant Pot variant | same total time; rice-freshness misses 60% → 0% | `reports/phase1.md` |
| Action model (8 classes), unseen people / unseen camera | macro-F1 0.40 / 0.42 (first model 0.20 / 0.20; majority 0.09) | `reports/phase2a2.md` |
| Cutting detection, unseen people / unseen camera | average precision 0.38 / 0.42 (random ≈ 0.10) | `reports/phase2a2.md` |
| ONNX export and Mac CPU latency | matches PyTorch to 1e-6; 86 ms per 2.1 s step (p95) | `reports/phase2b_mac_benchmark.json` |
| Jetson Orin Nano Super, TensorRT FP16, model only | 7.3 ms per 2.1 s step (p95, 0.34% of real time); 100% of 6,179 test predictions match the Mac | `reports/phase2b.md` |
| Jetson, end to end (video → planner), paced at 15 fps | 92 ms per 2.1 s step (p95, 4.3%); +0.07 W over idle; 104/104 steps and 4/4 task events match the Mac | `reports/phase2b.md` |

## Reproduce the perception results

```sh
uv sync --extra train
# Download Breakfast Actions (see docs/data_provenance.md) into data/raw/breakfast/
uv run kcp-train features      # motion crop + S3D clips, streamed from the archive (~1 h 45 min on an M1)
uv run kcp-train train-2a2     # six models over three feature sets
uv run kcp-train evaluate-2a2  # writes reports/phase2a2_results.md
```

## Limits

- Durations and quality windows are authored heuristics, not measurements.
- Public training data shows breakfast prep (fruit, bread), not stir-fry vegetables. The camera detects *that* you are cutting, not *what*; the planner infers the task from the recipe graph.
- The action model finds about a third to a half of real cutting runs (event F1 0.35–0.43); the cook confirms each completion. See `reports/phase2a2.md`.
- Simulated results are not real cooking measurements. No result yet comes from the developer's own kitchen.
- Food safety is out of scope: the system never judges doneness, temperature, or freshness for safety.

## Repository layout

```text
src/kcp/            scheduler, replay, simulator, schemas, CLI
src/kcp/perception/ embeddings, training, evaluation (PyTorch)
src/kcp/adapters/   Breakfast Actions -> action events
data/graphs/        recipe task graphs
docs/               label mapping, data provenance, privacy plan
reports/            generated results
tests/              unit tests, fixtures, golden replays
```

## Data and citation

Breakfast Actions: H. Kuehne, A. B. Arslan, T. Serre, "The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities," CVPR 2014. No video is stored in this repository; the visual explainer shows 26 still frames from the dataset. See `docs/data_provenance.md`.
