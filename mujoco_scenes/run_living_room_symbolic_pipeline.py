"""CLI for frozen living-room Phase-1 witness -> validated Phase-2 plan."""

from __future__ import annotations

import argparse
import json

from .living_room_symbolic_planning import run_living_room_symbolic_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-variant-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_living_room_symbolic_pipeline(
        arguments.phase1_variant_dir, arguments.output_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
