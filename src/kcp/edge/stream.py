"""Streaming perception: one embedding in at a time, class probabilities out.

The temporal head was trained on windows centered on the labeled step (4 steps before, 3 after),
so a live stream answers for step t once step t+3 has arrived. That is a fixed delay of 3 steps:
~1.6 s with MobileNetV3 frames (every 8 video frames) or ~6.4 s with S3D clips (every 32).
At the end of a stream, flush() pads with the last step, exactly as training did at video edges.

This module needs only NumPy, so it can run on the Jetson with an ONNX head.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

WINDOW = 8  # must equal kcp.perception.train.WINDOW (checked by a test)
_BEFORE = WINDOW // 2  # steps before the labeled step in a training window
_AFTER = WINDOW - _BEFORE - 1  # steps after it

Head = Callable[[np.ndarray], np.ndarray]  # (1, WINDOW, D) float32 -> (1, classes) probabilities


class StreamingHead:
    def __init__(self, head: Head, mean: np.ndarray, std: np.ndarray, center: bool) -> None:
        self.head, self.mean, self.std, self.center = head, mean, std, center
        self.count = 0
        self.running_sum: np.ndarray | None = None
        self.buffer: deque[np.ndarray] = deque(maxlen=WINDOW)  # newest standardized steps
        self.next_out = 0  # the next step to answer for

    def _prepare(self, emb: np.ndarray) -> np.ndarray:
        x = emb.astype(np.float32)
        self.count += 1
        self.running_sum = x.copy() if self.running_sum is None else self.running_sum + x
        if self.center:  # causal: subtract the mean of this stream's steps so far
            x = (x - self.running_sum / self.count).astype(np.float32)
        x = x.astype(np.float16).astype(np.float32)  # match the training cache precision
        out: np.ndarray = (x - self.mean) / self.std
        return out

    def _answer(self, step: int) -> tuple[int, np.ndarray]:
        """Window for `step`, clamped to the steps seen so far, like training at video edges."""
        base = self.count - len(self.buffer)  # stream index of buffer[0]
        last = self.count - 1
        rows = [
            self.buffer[min(max(k, 0), last) - base]
            for k in range(step - _BEFORE, step + _AFTER + 1)
        ]
        return step, self.head(np.stack(rows)[None])[0]

    def push(self, emb: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Add one step; return (step, probabilities) for every step that is now answerable."""
        self.buffer.append(self._prepare(emb))
        out = []
        while self.next_out + _AFTER <= self.count - 1:
            out.append(self._answer(self.next_out))
            self.next_out += 1
        return out

    def flush(self) -> list[tuple[int, np.ndarray]]:
        """Answer for the last steps once the stream ends."""
        out = []
        while self.next_out < self.count:
            out.append(self._answer(self.next_out))
            self.next_out += 1
        return out
