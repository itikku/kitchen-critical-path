"""The live edge path must give the same answers as the offline evaluation path."""

import numpy as np
import pytest
import torch

from kcp.edge.interpret import Interpreter
from kcp.edge.runs import RunDetector, RunEvent
from kcp.edge.stream import WINDOW, StreamingHead
from kcp.perception import train as T
from kcp.scheduler import build_state, recommend
from kcp.schema import Event, TaskGraph


def test_stream_window_matches_training() -> None:
    assert WINDOW == T.WINDOW


@pytest.mark.parametrize("center", [False, True])
def test_streaming_matches_batch(center: bool) -> None:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    dim, lengths = 16, [13, 5]  # the second video is shorter than one window
    cfg = T.TrainConfig(model="temporal_head", hidden=32, center=center)
    model = T.build_model(cfg, dim).to(T.device()).eval()
    emb = rng.normal(size=(sum(lengths), dim)).astype(np.float16)
    starts = np.concatenate([np.full(n, sum(lengths[:i])) for i, n in enumerate(lengths)])
    ends = np.concatenate([np.full(n, sum(lengths[: i + 1]) - 1) for i, n in enumerate(lengths)])
    data = T.SplitData(
        emb, np.zeros(len(emb), dtype=np.int64), starts, ends, ["a", "b"], np.arange(len(emb))
    )
    prepared = T.prepare(data, cfg)
    mean, std = T.feature_stats(prepared.emb)
    batch = T.predict_proba(model, prepared, mean, std)

    def head(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out: np.ndarray = model(torch.from_numpy(x).to(T.device())).softmax(1).cpu().numpy()
        return out

    streamed = []
    for i, n in enumerate(lengths):
        s = StreamingHead(head, mean, std, center)
        offset = sum(lengths[:i])
        for row in emb[offset : offset + n]:
            streamed += [p for _, p in s.push(row)]
        streamed += [p for _, p in s.flush()]
    assert np.allclose(np.stack(streamed), batch, atol=5e-3)


def test_run_detector_reports_runs_of_at_least_20_s() -> None:
    det = RunDetector("cut", width_steps=1, threshold=0.5)
    events = []
    for t in range(80):
        p = 0.9 if 10 <= t < 50 else 0.1
        events += det.push(float(t), p)
    assert [(e.kind, e.t_s) for e in events] == [("start", 10.0), ("end", 50.0)]
    short = RunDetector("cut", width_steps=1, threshold=0.5)
    assert sum((short.push(float(t), 0.9 if t < 15 else 0.1) for t in range(30)), []) == []


def test_interpreter_credits_the_recommended_task_and_asks_to_confirm(frozen: TaskGraph) -> None:
    events = [
        Event(
            event_id="u1",
            session_id="s",
            t_start_s=0,
            task_id="press_tofu",
            status="started",
            source="user",
        )
    ]
    interp = Interpreter(frozen, "s", "test-model")

    def step(run: RunEvent, now: float) -> Event:
        state = build_state(frozen, events, now)
        ev = interp.on_run(run, state, recommend(frozen, state, now))
        assert ev is not None
        events.append(ev)
        return ev

    started = step(RunEvent("cut", "start", 60.0, 0.4), 80.0)
    assert (started.task_id, started.status) == ("cut_firm_veg", "started")
    done = step(RunEvent("cut", "end", 540.0, 0.8), 540.0)
    assert done.status == "complete" and done.confidence is not None and done.confidence < 0.6
    rec = recommend(frozen, build_state(frozen, events, 541.0), 541.0)
    assert rec.confirm and "cut the firm vegetables" in rec.confirm[0]


def test_numpy_macro_f1_matches_sklearn() -> None:
    from sklearn.metrics import f1_score

    from kcp.edge.parity import macro_f1

    rng = np.random.default_rng(1)
    y, pred = rng.integers(0, 8, 500), rng.integers(0, 8, 500)
    pred[:5] = 7  # make sure every class can appear
    expected = f1_score(y, pred, labels=list(range(8)), average="macro", zero_division=0)
    assert abs(macro_f1(y, pred, 8) - expected) < 1e-12


def test_e2e_constants_match_training() -> None:
    from kcp.edge import e2e
    from kcp.perception import features as F

    assert (e2e.SIZE, e2e.CLIP_STRIDE, e2e.CLIP_OFFSETS) == (F.SIZE, F.CLIP_STRIDE, F.CLIP_OFFSETS)


@pytest.mark.parametrize("n", [5, 100, 129, 3329])
def test_clip_assembler_uses_the_training_frames(n: int) -> None:
    """Live clips use the same frame indices as the training cache, including at both edges."""
    from kcp.edge.e2e import CLIP_OFFSETS, CLIP_STRIDE, ClipAssembler

    asm = ClipAssembler(lambda f: f)  # a "frame" is its own index, so clips show which frames
    clips = [c for i in range(n) for c in asm.push(i, np.array([i]))] + asm.flush()
    expected = {
        a: [min(max(a + k, 0), n - 1) for k in CLIP_OFFSETS] for a in range(0, n, CLIP_STRIDE)
    }
    assert [a for a, _, _ in clips] == list(expected)
    for a, now, clip in clips:
        assert clip.ravel().tolist() == expected[a]
        assert now == min(a + CLIP_OFFSETS[-1], n - 1)  # answered as soon as its last frame arrives


def test_planner_turns_a_cutting_stretch_into_task_events(frozen: TaskGraph) -> None:
    from kcp.edge.e2e import CLIP_STRIDE, CUT, FPS, Planner

    planner = Planner(frozen, "s", "test-model")
    step_s = CLIP_STRIDE / FPS
    for s in range(40):
        p = np.full(8, 0.01)
        p[CUT] = 0.9 if 5 <= s < 25 else 0.01
        planner.push(s * step_s, p, s * step_s)
    statuses = [(e.task_id.startswith("cut"), e.status) for e in planner.events]
    assert statuses == [(True, "started"), (True, "complete")]
    assert planner.events[1].confidence is not None and planner.events[1].confidence < 0.6
