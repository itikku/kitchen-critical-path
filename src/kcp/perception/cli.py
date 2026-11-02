"""`kcp-train`: embed Breakfast videos, train the temporal head, and evaluate it."""

from __future__ import annotations

import argparse
from pathlib import Path

VIDEO_ROOT = Path("data/raw/breakfast/BreakfastII_15fps_qvga_sync")
ANNOTATION_ROOT = Path("data/raw/breakfast/annotations/segmentation_coarse")
ARCHIVE = Path("data/raw/breakfast/BreakfastII_15fps_qvga_sync.tar.gz")
CACHE_DIR = Path("data/cache")
RUNS_DIR = Path("data/runs")
RUNS_DIR_2A2 = Path("data/runs_2a2")
REPORTS_DIR = Path("reports")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kcp-train")
    sub = parser.add_subparsers(dest="command", required=True)
    p_embed = sub.add_parser("embed", help="cache frozen-encoder embeddings for every video")
    p_embed.add_argument("--limit", type=int, default=None, help="embed only the first N videos")
    p_feat = sub.add_parser("features", help="Phase 2a.2: motion crop + MobileNetV3 + S3D features")
    p_feat.add_argument("--limit", type=int, default=None, help="process only N videos")
    sub.add_parser("train", help="train the frame baseline and the temporal head")
    sub.add_parser("evaluate", help="evaluate both models on T1 and T2 and write the report")
    sub.add_parser("train-2a2", help="Phase 2a.2: train on all three feature sets")
    p_f3 = sub.add_parser("features-2a3", help="Phase 2a.3: denser clips and wrist crops")
    p_f3.add_argument("--no-wrist", action="store_true", help="only the denser-clip experiment")
    sub.add_parser("train-2a3", help="Phase 2a.3: train on the new feature sets")
    sub.add_parser("evaluate-2a3", help="Phase 2a.3: evaluate all S3D models")
    sub.add_parser("evaluate-2a2", help="Phase 2a.2: evaluate with planner-level metrics")
    args = parser.parse_args(argv)

    if args.command == "embed":
        from kcp.perception.embed import run

        run(VIDEO_ROOT, ANNOTATION_ROOT, CACHE_DIR, limit=args.limit)
    elif args.command == "features":
        from kcp.perception.features import run as run_features

        run_features(ARCHIVE, ANNOTATION_ROOT, CACHE_DIR, limit=args.limit)
    elif args.command == "train":
        from kcp.perception.train import train_all

        train_all(CACHE_DIR, RUNS_DIR)
    elif args.command == "evaluate":
        from kcp.perception.evaluate import evaluate_all

        evaluate_all(CACHE_DIR, RUNS_DIR, REPORTS_DIR)
    elif args.command == "train-2a2":
        from kcp.perception.train import CONFIGS_2A2, train_all

        train_all(CACHE_DIR, RUNS_DIR_2A2, CONFIGS_2A2)
    elif args.command == "evaluate-2a2":
        from kcp.perception.evaluate import evaluate_2a2

        evaluate_2a2(CACHE_DIR, RUNS_DIR_2A2, REPORTS_DIR)
    elif args.command == "features-2a3":
        from kcp.perception.features import run_2a3

        run_2a3(ARCHIVE, ANNOTATION_ROOT, CACHE_DIR, wrist=not args.no_wrist)
    elif args.command == "train-2a3":
        from kcp.perception.train import CONFIGS_2A3, train_all

        train_all(CACHE_DIR, RUNS_DIR_2A2, CONFIGS_2A3)
    elif args.command == "evaluate-2a3":
        from kcp.perception.evaluate import evaluate_2a2
        from kcp.perception.train import CONFIGS_2A2, CONFIGS_2A3

        s3d = tuple(c for c in (*CONFIGS_2A2, *CONFIGS_2A3) if c.features.startswith("s3d"))
        evaluate_2a2(CACHE_DIR, RUNS_DIR_2A2, REPORTS_DIR, s3d, "phase2a3")


if __name__ == "__main__":
    main()
