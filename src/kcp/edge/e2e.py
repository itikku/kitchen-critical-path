"""End-to-end edge pipeline: video frames in, planner recommendations out.

    frames -> fixed crop -> 224x224 -> 8-frame clip every 32 frames -> S3D encoder
           -> streaming head -> cut / mix run detectors -> interpreter -> scheduler

`run` (Jetson): decodes a video file with OpenCV and runs both models with TensorRT. It needs
NumPy, OpenCV, Pillow and TensorRT (all in JetPack) plus pydantic for the scheduler; no PyTorch.
`reference` (Mac): the same downstream code on the cached training embeddings, with the ONNX head.
`compare` (Mac): the Jetson run against the reference and the true labels.

The crop box is fixed once per video, as in training. With a live camera, the box is set when the
camera is framed on the prep zone.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kcp.edge.interpret import Interpreter
from kcp.edge.parity import KINETICS_MEAN, KINETICS_STD
from kcp.edge.runs import RunDetector
from kcp.edge.stream import StreamingHead
from kcp.perception.labels import CLASSES
from kcp.replay import replay
from kcp.scheduler import build_state, recommend
from kcp.schema import Event, TaskGraph

SIZE = 224  # must equal kcp.perception.features.SIZE (checked by a test)
CLIP_STRIDE = 32  # must equal kcp.perception.features.CLIP_STRIDE
CLIP_OFFSETS = (-7, -5, -3, -1, 1, 3, 5, 7)  # must equal kcp.perception.features.CLIP_OFFSETS
FPS = 15.0
# Run detector settings: chosen on validation for cutting in Phase 2a.2 (16 s, 0.5). Mixing reuses
# them untuned.
SMOOTH_S, THRESHOLD = 16.0, 0.5
CUT = CLASSES.index("cut")
MIX = (CLASSES.index("pour"), CLASSES.index("stir"))
_RESIDUES = {k % CLIP_STRIDE for k in CLIP_OFFSETS}

Clip = tuple[int, int, np.ndarray]  # (anchor frame, frame that completed the clip, clip)


class ClipAssembler:
    """Frames in, one clip per anchor out, built exactly as the training cache built them.

    Anchor a uses frames a-7 ... a+7 at stride 2. Frames before 0 repeat frame 0; frames after
    the end of the stream repeat the last frame. Only frames that some clip needs are prepared."""

    def __init__(self, prep: Callable[[np.ndarray], np.ndarray]) -> None:
        self.prep = prep
        self.crops: dict[int, np.ndarray] = {}
        self.next_anchor = 0
        self.last = -1
        self.last_frame: np.ndarray | None = None

    @staticmethod
    def needed(i: int) -> bool:
        return i == 0 or i % CLIP_STRIDE in _RESIDUES

    def _clip(self, a: int) -> np.ndarray:
        rows = []
        for k in CLIP_OFFSETS:
            j = min(max(a + k, 0), self.last)
            if j not in self.crops:  # only the last frame, at the end of the stream
                assert j == self.last and self.last_frame is not None
                self.crops[j] = self.prep(self.last_frame)
            rows.append(self.crops[j])
        return np.stack(rows)

    def push(self, i: int, frame: np.ndarray) -> list[Clip]:
        assert i == self.last + 1, "frames must arrive in order"
        self.last, self.last_frame = i, frame
        if self.needed(i):
            self.crops[i] = self.prep(frame)
        out: list[Clip] = []
        while self.next_anchor + CLIP_OFFSETS[-1] <= i:
            out.append((self.next_anchor, i, self._clip(self.next_anchor)))
            self.next_anchor += CLIP_STRIDE
        for j in [j for j in self.crops if j < i - 15]:  # a clip spans 15 frames
            del self.crops[j]
        return out

    def flush(self) -> list[Clip]:
        out: list[Clip] = []
        while self.next_anchor <= self.last:
            out.append((self.next_anchor, self.last, self._clip(self.next_anchor)))
            self.next_anchor += CLIP_STRIDE
        return out


def normalize(clip: np.ndarray) -> np.ndarray:
    """(8, 224, 224, 3) uint8 -> (1, 3, 8, 224, 224) float32, Kinetics mean and std."""
    x = (clip.astype(np.float32) / 255 - KINETICS_MEAN) / KINETICS_STD
    out: np.ndarray = np.ascontiguousarray(x.transpose(3, 0, 1, 2)[None], dtype=np.float32)
    return out


class Planner:
    """Per-step probabilities in, task events and recommendations out."""

    def __init__(self, graph: TaskGraph, session_id: str, model_version: str) -> None:
        width = max(1, round(SMOOTH_S / (CLIP_STRIDE / FPS)))  # same rounding as the evaluation
        self.graph = graph
        self.detectors = {
            "cut": RunDetector("cut", width, THRESHOLD),
            "mix": RunDetector("mix", width, THRESHOLD),
        }
        self.interp = Interpreter(graph, session_id, model_version)
        self.events: list[Event] = []
        self.log: list[str] = []

    def push(self, t_step_s: float, probs: np.ndarray, now_s: float) -> None:
        p = {"cut": float(probs[CUT]), "mix": float(probs[list(MIX)].sum())}
        for action, det in self.detectors.items():
            for run in det.push(t_step_s, p[action]):
                state = build_state(self.graph, self.events, now_s)
                ev = self.interp.on_run(run, state, recommend(self.graph, state, now_s))
                if ev is None:
                    continue
                self.events.append(ev)
                state = build_state(self.graph, self.events, now_s)
                rec = recommend(self.graph, state, now_s)
                name = self.graph.task(ev.task_id).display_name
                nxt = self.graph.task(rec.next_task).display_name if rec.next_task else "-"
                self.log.append(
                    f"[{now_s:6.1f} s] camera: {ev.status} '{name}' at {ev.t_start_s:.1f} s "
                    f"(confidence {ev.confidence}) -> next: {nxt}"
                    + "".join(f" | {c}" for c in rec.confirm)
                )


@dataclass
class Steps:
    """Per-step record of one run: which clip, when its answer was available, and what it was."""

    anchors: list[int] = field(default_factory=list)  # the clip's center frame
    now_frames: list[int] = field(default_factory=list)  # the frame that completed the clip
    emb: list[np.ndarray] = field(default_factory=list)
    probs: list[np.ndarray] = field(default_factory=list)
    times: list[dict[str, float]] = field(default_factory=list)


def run_steps(
    clips: Iterator[Clip],
    embed: Callable[[np.ndarray], np.ndarray],
    head: StreamingHead,
    planner: Planner,
    timer: Callable[[], dict[str, float]] | None = None,
) -> Steps:
    """Drive the encoder, the streaming head and the planner; record per-step outputs and times.
    `embed` turns one clip into one embedding (the reference passes cached embeddings through)."""
    out = Steps()
    probs: dict[int, np.ndarray] = {}

    def answer(answers: list[tuple[int, np.ndarray]]) -> None:
        for step, p in answers:
            probs[step] = p
            planner.push(out.anchors[step] / FPS, p, out.now_frames[-1] / FPS)

    for a, now, clip in clips:
        t0 = time.perf_counter()
        emb = embed(clip)
        t1 = time.perf_counter()
        out.anchors.append(a)
        out.now_frames.append(now)
        out.emb.append(emb)
        answers = head.push(emb)
        t2 = time.perf_counter()
        answer(answers)
        t3 = time.perf_counter()
        step = {"encoder_ms": (t1 - t0) * 1e3, "head_ms": (t2 - t1) * 1e3}
        step["planner_ms"] = (t3 - t2) * 1e3
        if timer is not None:
            step |= timer()
        out.times.append(step)
    answer(head.flush())
    out.probs = [probs[s] for s in range(len(out.anchors))]
    return out


# ------------------------------------------------------------------------------ Jetson side


class TrtModel:
    """One TensorRT engine with a single input "x" and output "y", via the CUDA runtime library."""

    def __init__(self, path: Path) -> None:
        import tensorrt as trt

        lib = Path("/usr/local/cuda/lib64/libcudart.so")
        self.cuda = ctypes.CDLL(str(lib) if lib.exists() else "libcudart.so")
        self.cuda.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.cuda.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            path.read_bytes()
        )
        self.context = self.engine.create_execution_context()
        self.host: dict[str, np.ndarray] = {}
        self.dev: dict[str, ctypes.c_void_p] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.host[name] = np.zeros(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            ptr = ctypes.c_void_p()
            self._ok(
                self.cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(self.host[name].nbytes))
            )
            self.dev[name] = ptr
            self.context.set_tensor_address(name, ptr.value)
        self.stream = ctypes.c_void_p()
        self._ok(self.cuda.cudaStreamCreate(ctypes.byref(self.stream)))

    @staticmethod
    def _ok(code: int) -> None:
        if code != 0:
            raise RuntimeError(f"CUDA error {code}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=self.host["x"].dtype).reshape(self.host["x"].shape)
        out = self.host["y"]
        self._ok(self.cuda.cudaMemcpyAsync(self.dev["x"], x.ctypes.data, x.nbytes, 1, self.stream))
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT execution failed")
        self._ok(
            self.cuda.cudaMemcpyAsync(out.ctypes.data, self.dev["y"], out.nbytes, 2, self.stream)
        )
        self._ok(self.cuda.cudaStreamSynchronize(self.stream))
        return out.astype(np.float32).copy()


def block_network() -> None:
    """Make any Python network call fail, so the run proves it needs no network."""
    import socket

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("network access is blocked during this run")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


def open_sockets() -> int:
    """Socket file descriptors this process holds (Linux)."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        return -1
    n = 0
    for fd in fd_dir.iterdir():
        try:
            n += os.readlink(fd).startswith("socket:")
        except OSError:
            pass
    return n


def run(
    video: Path,
    meta: dict[str, object],
    models: Path,
    graph: TaskGraph,
    out: Path,
    precision: str,
    realtime: bool,
) -> dict[str, object]:
    import cv2
    from PIL import Image

    block_network()
    left, top, right, bottom = (int(v) for v in meta["box"])  # type: ignore[attr-defined]

    def prep(bgr: np.ndarray) -> np.ndarray:
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop((left, top, right, bottom))
        return np.asarray(img.resize((SIZE, SIZE), Image.Resampling.BILINEAR))

    encoder = TrtModel(models / f"encoder.{precision}.engine")
    head_model = TrtModel(models / f"head.{precision}.engine")
    head_meta = json.loads((models / "head.json").read_text())
    for _ in range(5):  # warm up both engines before timing
        encoder(np.zeros((1, 3, 8, SIZE, SIZE), np.float32))
        head_model(np.zeros((1, head_meta["window"], head_meta["dim"]), np.float32))
    head = StreamingHead(
        head_model, np.load(models / "mean.npy"), np.load(models / "std.npy"), head_meta["center"]
    )
    planner = Planner(graph, str(meta["video"]), f"{head_meta['run']}+trt-{precision}")
    acc = {"decode_ms": 0.0, "prep_ms": 0.0}

    def timer() -> dict[str, float]:
        snap = dict(acc)
        acc["decode_ms"] = acc["prep_ms"] = 0.0
        return snap

    def clips() -> Iterator[Clip]:
        cap = cv2.VideoCapture(str(video))
        asm = ClipAssembler(prep)
        start = time.perf_counter()
        i = 0
        while True:
            if realtime:  # pace the file like a 15 fps camera
                time.sleep(max(0.0, start + i / FPS - time.perf_counter()))
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                break
            ready = asm.push(i, frame)
            acc["decode_ms"] += (t1 - t0) * 1e3
            acc["prep_ms"] += (time.perf_counter() - t1) * 1e3
            yield from ready
            i += 1
        cap.release()
        yield from asm.flush()

    t0 = time.perf_counter()
    result = run_steps(clips(), lambda c: encoder(normalize(c))[0], head, planner, timer)
    wall = time.perf_counter() - t0
    totals = np.array([sum(t.values()) for t in result.times])
    summary: dict[str, object] = {
        "video": meta["video"],
        "precision": precision,
        "realtime": realtime,
        "steps": len(result.anchors),
        "video_s": int(meta["frames"]) / FPS,  # type: ignore[call-overload]
        "wall_s": wall,
        "step_budget_ms": CLIP_STRIDE / FPS * 1e3,
        "step_compute_ms": {
            "p50": float(np.percentile(totals, 50)),
            "p95": float(np.percentile(totals, 95)),
            "max": float(totals.max()),
        },
        "stage_ms_p50": {
            k: float(np.median([t[k] for t in result.times])) for k in result.times[0]
        },
        "answer_delay_s": (3 * CLIP_STRIDE + CLIP_OFFSETS[-1]) / FPS,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
        "open_sockets_at_end": open_sockets(),
    }
    save(out, result, planner, graph, summary)
    return summary


# --------------------------------------------------------------------------------- Mac side


def reference(cache: Path, models: Path, graph: TaskGraph, out: Path, key: str) -> None:
    import onnxruntime as ort

    z = np.load(cache / "s3d_crop" / f"{key}.npz")
    session = ort.InferenceSession(str(models / "head.onnx"), providers=["CPUExecutionProvider"])
    meta = json.loads((models / "head.json").read_text())

    def head_fn(x: np.ndarray) -> np.ndarray:
        out_: np.ndarray = session.run(None, {"x": x.astype(np.float32)})[0]
        return out_

    head = StreamingHead(
        head_fn, np.load(models / "mean.npy"), np.load(models / "std.npy"), meta["center"]
    )
    planner = Planner(graph, key, f"{meta['run']}+onnx-cpu")
    frames = [int(f) for f in z["frame"]]
    n = int(json.loads((out.parent / "meta.json").read_text())["frames"])
    clips = (
        (a, min(a + CLIP_OFFSETS[-1], n - 1), e) for a, e in zip(frames, z["emb"], strict=True)
    )
    result = run_steps(clips, lambda x: x, head, planner)
    save(out, result, planner, graph, {"video": key, "y": z["y"].tolist()})


def save(
    out: Path,
    result: Steps,
    planner: Planner,
    graph: TaskGraph,
    summary: dict[str, object],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "steps.npz",
        anchor=np.array(result.anchors),
        now_frame=np.array(result.now_frames),
        emb=np.stack(result.emb).astype(np.float32),
        probs=np.stack(result.probs),
    )
    (out / "events.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in planner.events))
    (out / "live_log.txt").write_text("\n".join(planner.log) + "\n")
    (out / "transcript.txt").write_text(replay(graph, planner.events))
    (out / "times.json").write_text(json.dumps(result.times))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def compare(jetson: Path, ref: Path) -> dict[str, object]:
    j, r = np.load(jetson / "steps.npz"), np.load(ref / "steps.npz")
    y = np.array(json.loads((ref / "summary.json").read_text())["y"])
    assert (j["anchor"] == r["anchor"]).all(), "the two runs used different steps"
    cos = (j["emb"] * r["emb"]).sum(1) / (
        np.linalg.norm(j["emb"], axis=1) * np.linalg.norm(r["emb"], axis=1)
    )
    jp, rp = j["probs"].argmax(1), r["probs"].argmax(1)

    def events(d: Path) -> list[tuple[str, str, float]]:
        lines = (d / "events.jsonl").read_text().splitlines()
        return [(e["task_id"], e["status"], e["t_start_s"]) for e in map(json.loads, lines)]

    return {
        "steps": int(len(y)),
        "encoder_cosine": {"min": float(cos.min()), "median": float(np.median(cos))},
        "head_max_abs_prob_diff": float(np.abs(j["probs"] - r["probs"]).max()),
        "same_top_class": float((jp == rp).mean()),
        "differing_steps": [int(s) for s in np.flatnonzero(jp != rp)],
        "accuracy_jetson": float((jp == y).mean()),
        "accuracy_reference": float((rp == y).mean()),
        "events_jetson": events(jetson),
        "events_reference": events(ref),
        "same_events": events(jetson) == events(ref),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kcp.edge.e2e")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("pack", type=Path, help="folder with video.avi and meta.json")
    p_run.add_argument("models", type=Path, help="folder with the TensorRT engines")
    p_run.add_argument("graph", type=Path)
    p_run.add_argument("out", type=Path)
    p_run.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p_run.add_argument("--realtime", action="store_true", help="pace frames at 15 fps")
    p_ref = sub.add_parser("reference")
    p_ref.add_argument("pack", type=Path)
    p_ref.add_argument("--models", type=Path, default=Path("data/edge/s3d_selected"))
    p_ref.add_argument(
        "--graph", type=Path, default=Path("data/graphs/tofu_stir_fry.instant_pot.v0.json")
    )
    p_ref.add_argument("--cache", type=Path, default=Path("data/cache"))
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("jetson", type=Path)
    p_cmp.add_argument("reference", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        meta = json.loads((args.pack / "meta.json").read_text())
        summary = run(
            args.pack / "video.avi",
            meta,
            args.models,
            TaskGraph.load(args.graph),
            args.out,
            args.precision,
            args.realtime,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "reference":
        meta = json.loads((args.pack / "meta.json").read_text())
        reference(
            args.cache,
            args.models,
            TaskGraph.load(args.graph),
            args.pack / "reference",
            meta["video"],
        )
    else:
        print(json.dumps(compare(args.jetson, args.reference), indent=2))


if __name__ == "__main__":
    main()
