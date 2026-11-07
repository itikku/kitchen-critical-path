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
