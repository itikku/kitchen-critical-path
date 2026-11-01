"""Metrics that match what the planner needs (IMPLEMENTATION_PLAN.md, Phase 2a.2).

- Binary average precision (AP) for "cutting" and "mixing" (pour or stir): how well the model's
  probability ranks those moments above everything else. 1.0 is perfect; the class's share of
  steps is what random guessing scores.
- Event F1 for cutting runs: smooth the cut probability over time, threshold it, keep runs of at
  least MIN_RUN_S seconds, and match them to true cutting runs by overlap (IoU >= MIN_IOU).
  The smoothing width and the threshold are chosen on validation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from kcp.perception.labels import class_index

FPS = 15.0
MIN_RUN_S = 20.0
MIN_IOU = 0.25
CUT = [class_index("cut")]
MIX = [class_index("pour"), class_index("stir")]
WIDTHS_S = (2.0, 4.0, 8.0, 16.0)
# 0.05 added after a validation-only check of the run-2 model picked the grid edge (0.1).
THRESHOLDS = (0.05, *(round(x, 1) for x in np.arange(0.1, 0.91, 0.1)))


def binary_ap(y: np.ndarray, probs: np.ndarray, classes: list[int]) -> float:
    target = np.isin(y, classes)
    return float(average_precision_score(target, probs[:, classes].sum(1)))


def smooth(p: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average; the edges average over the steps that exist."""
    if width <= 1:
        return p
    kernel = np.ones(width)
    num = np.convolve(p, kernel, mode="same")
    den = np.convolve(np.ones_like(p), kernel, mode="same")
    out: np.ndarray = num / den
    return out


def runs(mask: np.ndarray, t: np.ndarray, step_s: float) -> list[tuple[float, float]]:
    """Contiguous True stretches as (start_s, end_s), keeping those of at least MIN_RUN_S."""
    out = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        start, end = float(t[i]), float(t[j]) + step_s
        if end - start >= MIN_RUN_S:
            out.append((start, end))
        i = j + 1
    return out


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(pred: list[tuple[float, float]], true: list[tuple[float, float]]) -> int:
    """Greedy one-to-one matching by IoU; returns the number of matched pairs."""
    pairs = sorted(
        ((iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)), reverse=True
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    tp = 0
    for score, i, j in pairs:
        if score < MIN_IOU:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            tp += 1
    return tp


@dataclass
class EventScore:
    f1: float
    tp: int
    fp: int
    fn: int


def event_score(
    y: np.ndarray,
    p_cut: np.ndarray,
    frame: np.ndarray,
    video_start: np.ndarray,
    width_s: float,
    threshold: float,
) -> EventScore:
    tp = fp = fn = 0
    for start in np.unique(video_start):
        idx = np.flatnonzero(video_start == start)
        t = frame[idx] / FPS
        step_s = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        width = max(1, round(width_s / step_s))
        pred = runs(smooth(p_cut[idx], width) >= threshold, t, step_s)
        true = runs(np.isin(y[idx], CUT), t, step_s)
        m = match(pred, true)
        tp, fp, fn = tp + m, fp + len(pred) - m, fn + len(true) - m
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return EventScore(f1, tp, fp, fn)


def tune_event(
    y: np.ndarray, p_cut: np.ndarray, frame: np.ndarray, video_start: np.ndarray
) -> tuple[float, float, EventScore]:
    """Choose (width_s, threshold) with the best event F1. Call on validation data only."""
    best: tuple[float, float, EventScore] | None = None
    for w in WIDTHS_S:
        for th in THRESHOLDS:
            s = event_score(y, p_cut, frame, video_start, w, th)
            if best is None or s.f1 > best[2].f1:
                best = (w, th, s)
    assert best is not None
    return best
