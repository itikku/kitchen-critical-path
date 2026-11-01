"""`kcp`: replay sessions, simulate sessions, and map public annotations."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from kcp.schema import TaskGraph, load_events


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kcp")
    sub = parser.add_subparsers(dest="command", required=True)

    p_replay = sub.add_parser("replay", help="replay an event log against a task graph")
    p_replay.add_argument("events", type=Path)
    p_replay.add_argument("--graph", type=Path, required=True)

    p_sim = sub.add_parser("simulate", help="compare the scheduler with the plain recipe order")
    p_sim.add_argument("--graph", type=Path, required=True)
    p_sim.add_argument("--sessions", type=int, default=200)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--json", type=Path, default=None, help="also write the results as JSON")

    p_map = sub.add_parser(
        "breakfast-map", help="map Breakfast coarse annotations to camera labels"
    )
    p_map.add_argument("--annotations", type=Path, required=True)
    p_map.add_argument("--out", type=Path, default=None, help="write action events as JSONL")

    args = parser.parse_args(argv)
    if args.command == "replay":
        from kcp.replay import replay

        print(replay(TaskGraph.load(args.graph), load_events(args.events)), end="")
    elif args.command == "simulate":
        from kcp.simulate import compare, format_comparison

        result = compare(TaskGraph.load(args.graph), args.sessions, args.seed)
        print(format_comparison(result), end="")
        if args.json:
            args.json.write_text(json.dumps(dataclasses.asdict(result), indent=2) + "\n")
    elif args.command == "breakfast-map":
        from kcp.adapters.breakfast import mapping_report

        print(mapping_report(args.annotations, args.out), end="")


if __name__ == "__main__":
    main()
