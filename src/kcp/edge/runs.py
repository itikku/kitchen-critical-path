"""Live run detection: per-step probabilities in, "cutting started / ended" events out.

Offline evaluation smooths with a centered moving average (it can see the future). Live, the
detector uses a trailing average of the same width, so it reacts a little later. A run is
reported once the smoothed probability has stayed at or above the threshold for MIN_RUN_S,
and it ends at the first step below the threshold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

MIN_RUN_S = 20.0


@dataclass(frozen=True)
class RunEvent:
    action: str  # "cut" or "mix"
    kind: Literal["start", "end"]
    t_s: float  # start events carry the time the run began; end events the time it ended
    confidence: float  # mean smoothed probability over the run so far


class RunDetector:
    def __init__(self, action: str, width_steps: int, threshold: float) -> None:
        self.action, self.threshold = action, threshold
        self.recent: deque[float] = deque(maxlen=max(1, width_steps))
        self.run_start: float | None = None
        self.run_probs: list[float] = []
        self.reported = False

    def push(self, t_s: float, p: float) -> list[RunEvent]:
        self.recent.append(p)
        smoothed = sum(self.recent) / len(self.recent)
        out: list[RunEvent] = []
        if smoothed >= self.threshold:
            if self.run_start is None:
                self.run_start, self.run_probs = t_s, []
            self.run_probs.append(smoothed)
            if not self.reported and t_s - self.run_start >= MIN_RUN_S:
                out.append(RunEvent(self.action, "start", self.run_start, self._conf()))
                self.reported = True
        else:
            if self.run_start is not None and self.reported:
                out.append(RunEvent(self.action, "end", t_s, self._conf()))
            self.run_start, self.reported = None, False
        return out

    def _conf(self) -> float:
        return sum(self.run_probs) / len(self.run_probs) if self.run_probs else 0.0
