"""Breakfast Actions coarse annotations -> camera-label action events + a mapping report.

Action events are what the prep-zone camera model will emit in Phase 2b. They are not task events:
turning actions into task progress is the scheduler's job (docs/label_mapping.md)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kcp.perception.data import read_segments
from kcp.perception.labels import CLASSES, map_label

FPS = 15


class ActionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    t_start_s: float
    t_end_s: float
    action: str
    source: str = "public_dataset"
    source_label: str


def action_events(annotation: Path) -> Iterator[ActionEvent]:
    session_id = annotation.stem
    for seg in read_segments(annotation):
        yield ActionEvent(
            session_id=session_id,
            t_start_s=(seg.start - 1) / FPS,
            t_end_s=seg.end / FPS,
            action=map_label(seg.label),
            source_label=seg.label,
        )


def mapping_report(annotation_root: Path, out: Path | None = None) -> str:
    files = sorted(annotation_root.glob("*/*.txt"))
    seconds: dict[str, float] = dict.fromkeys(CLASSES, 0.0)
    sources: dict[str, Counter[str]] = {c: Counter() for c in CLASSES}
    lines_out = []
    for f in files:
        for ev in action_events(f):
            seconds[ev.action] += ev.t_end_s - ev.t_start_s
            sources[ev.action][ev.source_label] += 1
            if out is not None:
                lines_out.append(ev.model_dump_json())
    if out is not None:
        out.write_text("\n".join(lines_out) + "\n")
    rows = [f"Breakfast mapping report: {len(files)} annotation files (all camera views)", ""]
    rows.append(f"{'camera label':14s} {'hours':>6s}  source labels")
    for c in CLASSES:
        top = ", ".join(f"{k}×{v}" for k, v in sources[c].most_common())
        rows.append(f"{c:14s} {seconds[c] / 3600:6.2f}  {top}")
    rows.append("")
    rows.append(
        "Unmapped labels: none (map_label raises on any label outside the 48 coarse labels)."
    )
    return "\n".join(rows) + "\n"
