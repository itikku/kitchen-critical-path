"""Live viewer: the Jetson runs the pipeline and serves the video, the model's answer, and the
planner's next step to a browser.

The server listens on 127.0.0.1 only, so nothing is reachable from the network. To watch from the
Mac, forward the port over SSH:

    scripts/viewer_tunnel.sh        # then open http://localhost:8090

Over the USB-C cable (host `jetson-usb`), the stream is smooth; the Jetson's Wi-Fi link was too
weak for video.

Run on the Jetson (a recorded video, looped; a camera later with --source 0):

    PYTHONPATH=src:pylib python3 -m kcp.edge.viewer e2e edge/s3d_selected \\
        graphs/tofu_stir_fry.instant_pot.v0.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from kcp.edge.e2e import (
    FPS,
    ClipAssembler,
    Planner,
    load_engines,
    make_prep,
    new_head,
    normalize,
)
from kcp.perception.labels import CLASSES
from kcp.scheduler import build_state, fmt, recommend
from kcp.schema import TaskGraph

NAMES = {
    "cut": "Cutting",
    "peel": "Peeling",
    "pour": "Pouring",
    "stir": "Stirring",
    "transfer": "Moving food",
    "take_object": "Picking something up",
    "background": "Nothing",
    "other": "Something else",
}


@dataclass
class Shared:
    """What the pipeline thread publishes and the web handlers read."""

    cond: threading.Condition = field(default_factory=threading.Condition)
    jpeg: bytes = b""
    seq: int = 0
    state: dict[str, Any] = field(default_factory=dict)

    def put_frame(self, jpeg: bytes) -> None:
        with self.cond:
            self.jpeg, self.seq = jpeg, self.seq + 1
            self.cond.notify_all()

    def put_state(self, state: dict[str, Any]) -> None:
        with self.cond:
            self.state = state


def snapshot(
    planner: Planner, now_s: float, step_s: float | None, probs: np.ndarray | None, source: str
) -> dict[str, Any]:
    """The page's view of the session at `now_s`: model answer, next step, prompts, events."""
    graph = planner.graph
    state = build_state(graph, planner.events, now_s)
    rec = recommend(graph, state, now_s)
    out: dict[str, Any] = {
        "source": source,
        "now": fmt(now_s),
        "model": None,
        "next": {
            "task": graph.task(rec.next_task).display_name if rec.next_task else None,
            "reason": rec.reason,
        },
        "confirm": rec.confirm,
        "waiting": [[graph.task(t).display_name, why] for t, why in rec.waiting],
        "plate_eta": fmt(rec.plate_eta),
        "events": [
            {
                "t": fmt(e.t_start_s),
                "status": e.status,
                "task": graph.task(e.task_id).display_name,
                "confidence": e.confidence,
            }
            for e in reversed(planner.events)
        ],
    }
    if probs is not None and step_s is not None:
        top = int(np.argmax(probs))
        out["model"] = {
            "at": fmt(step_s),
            "lag_s": round(now_s - step_s, 1),
            "top": NAMES[CLASSES[top]],
            "scores": [
                {"key": c, "name": NAMES[c], "p": round(float(p), 3)}
                for c, p in zip(CLASSES, probs, strict=True)
            ],
        }
    return out


def center_square(width: int, height: int) -> list[int]:
    """The largest centered square: the default crop until the camera's box is set."""
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return [left, top, left + side, top + side]


def run_session(
    source: str, box: list[int] | None, models: Path, graph: TaskGraph, precision: str,
    shared: Shared, engines: tuple[Any, Any, dict[str, Any]], rotate: int = 0,
) -> None:  # fmt: skip
    """One pass through the source: pipeline, plus a frame and a state update for the page."""
    import cv2

    turn = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    encoder, head_model, head_meta = engines
    head = new_head(head_model, models, head_meta)
    planner = Planner(graph, "viewer", f"{head_meta['run']}+trt-{precision}")
    camera = source.isdigit()
    cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2) if camera else cv2.VideoCapture(source)
    if camera:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    asm: ClipAssembler | None = None
    anchors: list[int] = []
    last: tuple[float, np.ndarray] | None = None
    label = f"camera {source}" if camera else Path(source).name
    start = next_t = time.perf_counter()
    i = 0
    while True:
        if not camera:  # a file plays at camera speed
            time.sleep(max(0.0, start + i / FPS - time.perf_counter()))
        ok, frame = cap.read()
        if not ok:
            break
        if camera:  # the camera runs faster than 15 fps: keep one frame per 1/15 s slot
            now = time.perf_counter()
            if now < next_t:
                continue
            next_t = max(next_t + 1 / FPS, now - 1 / FPS)
        if rotate:
            frame = cv2.rotate(frame, turn[rotate])
        if asm is None:
            box = box or center_square(frame.shape[1], frame.shape[0])
            asm = ClipAssembler(make_prep(box))
        assert box is not None
        shown = frame.copy()
        cv2.rectangle(shown, (box[0], box[1]), (box[2] - 1, box[3] - 1), (0, 200, 255), 2)
        # Modest JPEG quality keeps the stream small; the browser scales the image.
        shared.put_frame(cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
        for a, _, clip in asm.push(i, frame):
            anchors.append(a)
            for step, p in head.push(encoder(normalize(clip))[0]):
                planner.push(anchors[step] / FPS, p, i / FPS)
                last = (anchors[step] / FPS, p)
        if i % 8 == 0:  # ~2 page updates per second
            step_s, probs = last if last else (None, None)
            shared.put_state(snapshot(planner, i / FPS, step_s, probs, label))
        i += 1
    cap.release()


PAGE = Path(__file__).with_name("viewer.html")  # the browser page


def _changed(shared: Shared, seen: int) -> bool:
    return shared.seq != seen


def make_handler(shared: Shared) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/state":
                with shared.cond:
                    body = json.dumps(shared.state).encode()
                self._send(body, "application/json")
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with shared.cond:
                            shared.cond.wait_for(partial(_changed, shared, seen), timeout=5)
                            jpeg, seen = shared.jpeg, shared.seq
                        head = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}"
                        self.wfile.write(head.encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_error(404)

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kcp.edge.viewer")
    parser.add_argument("pack", type=Path, help="folder with video.avi and meta.json (crop box)")
    parser.add_argument("models", type=Path, help="folder with the TensorRT engines")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--source", help="video file, or a camera number such as 0")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--once", action="store_true", help="play a file once instead of looping")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--box", help="crop box left,top,right,bottom (camera default: center)")
    args = parser.parse_args(argv)
    meta = json.loads((args.pack / "meta.json").read_text())
    source = args.source or str(args.pack / "video.avi")
    shared = Shared()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(shared))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"serving on 127.0.0.1:{args.port} (source: {source})", flush=True)
    engines = load_engines(args.models, args.precision)  # TensorRT stays on this thread
    graph = TaskGraph.load(args.graph)
    while True:
        box = [int(v) for v in args.box.split(",")] if args.box else None
        if box is None and not source.isdigit():
            box = meta["box"]  # the recorded video's training crop
        run_session(source, box, args.models, graph, args.precision, shared, engines, args.rotate)
        if args.once or source.isdigit():
            break
    server.shutdown()


if __name__ == "__main__":
    main()
