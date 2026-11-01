"""Evaluate the trained classifiers on T1 (new people) and T2 (new people + unseen camera ID)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from kcp.perception.data import HELDOUT_CAMERA, Split
from kcp.perception.labels import CLASSES
from kcp.perception.train import MODELS, load_model, load_split, predict

SPLITS: tuple[Split, ...] = ("T1", "T2")
LABELS = list(range(len(CLASSES)))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float((y == pred).mean()),
        "per_class": {
            c: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(CLASSES)
        },
        "confusion": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def evaluate_all(cache_dir: Path, runs_dir: Path, reports_dir: Path) -> None:
    results: dict[str, Any] = {"heldout_camera": HELDOUT_CAMERA, "splits": {}}
    train = load_split(cache_dir, "train")
    majority = int(np.argmax(np.bincount(train.y)))
    for split in SPLITS:
        data = load_split(cache_dir, split)
        persons = sorted({k.split("_")[0] for k in data.keys})
        entry: dict[str, Any] = {
            "videos": len(data.keys),
            "frames": int(len(data.y)),
            "persons": persons,
            "models": {"majority_class": metrics(data.y, np.full_like(data.y, majority))},
        }
        for name in MODELS:
            model, mean, std = load_model(runs_dir / name)
            entry["models"][name] = metrics(data.y, predict(model, data, mean, std))
        results["splits"][split] = entry
        print(
            f"{split}: "
            + ", ".join(f"{m} macro-F1 {v['macro_f1']:.3f}" for m, v in entry["models"].items()),
            flush=True,
        )
    results["train"] = {"videos": len(train.keys), "frames": int(len(train.y))}
    for name in MODELS:
        results.setdefault("runs", {})[name] = json.loads(
            (runs_dir / name / "train.json").read_text()
        )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase2a_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (reports_dir / "phase2a.md").write_text(render(results))
    print(f"wrote {reports_dir / 'phase2a.md'}", flush=True)


def render(r: dict[str, Any]) -> str:
    out = ["# Phase 2a — First PyTorch Training Run (generated)", ""]
    out.append(
        f"Train: {r['train']['videos']} videos, {r['train']['frames']:,} frames "
        f"(Breakfast persons s2–s4, camera IDs other than `{r['heldout_camera']}`)."
    )
    for split, e in r["splits"].items():
        out.append(
            f"{split}: {e['videos']} videos, {e['frames']:,} frames, "
            f"{len(e['persons'])} persons ({e['persons'][0]}–{e['persons'][-1]})."
        )
    out += [
        "",
        "## Macro F1 (8 classes)",
        "",
        "| Model | T1: new people | T2: new people + unseen camera ID |",
        "|---|---:|---:|",
    ]
    for m in ["majority_class", *MODELS]:
        cells = [f"{r['splits'][s]['models'][m]['macro_f1']:.3f}" for s in SPLITS]
        out.append(f"| `{m}` | {' | '.join(cells)} |")
    for split in SPLITS:
        out += [
            "",
            f"## Per-class results — {split}, `temporal_head` vs `frame_linear`",
            "",
            "| Class | Support | Temporal P | Temporal R | Temporal F1 | Frame F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        th = r["splits"][split]["models"]["temporal_head"]["per_class"]
        fl = r["splits"][split]["models"]["frame_linear"]["per_class"]
        for c in CLASSES:
            out.append(
                f"| `{c}` | {th[c]['support']:,} | {th[c]['precision']:.2f} | "
                f"{th[c]['recall']:.2f} | {th[c]['f1']:.2f} | {fl[c]['f1']:.2f} |"
            )
        out += [
            "",
            f"Confusion matrix ({split}, `temporal_head`; rows = true, columns = predicted):",
            "",
            "| true \\ pred | " + " | ".join(CLASSES) + " |",
            "|---|" + "---:|" * len(CLASSES),
        ]
        cm = r["splits"][split]["models"]["temporal_head"]["confusion"]
        for c, row in zip(CLASSES, cm, strict=True):
            out.append(f"| `{c}` | " + " | ".join(f"{v:,}" for v in row) + " |")
    return "\n".join(out) + "\n"
