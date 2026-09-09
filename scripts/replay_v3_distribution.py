#!/usr/bin/env python3
"""Zero-call replay of a frozen 3 x 32 distribution, with stage deltas.

Replays every archived raw FM contract through the full deterministic pipeline
without contacting the model, then reports stage progression against an optional
earlier snapshot.  This is the regression loop for compiler and grounder work:
the FM contracts are fixed, so any movement is attributable to the deterministic
system rather than to sampling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

STAGES = ["strict_v3_valid", "task_valid", "graph_compiled",
          "grounding_reached", "complete_grounding", "astar_reached", "success"]


def _run(cmd, env=None):
    return subprocess.run(cmd, env=env, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Distribution root containing trial_01 .. trial_NN")
    ap.add_argument("--out", type=Path, required=True,
                    help="Where to write this replay's outputs")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Earlier distribution_analysis.json to diff against")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo), TAMP_FM_SCHEMA_VERSION="3")
    args.out.mkdir(parents=True, exist_ok=True)

    for t in range(1, args.trials + 1):
        trial = f"trial_{t:02d}"
        src = args.root / trial
        if not src.exists():
            continue
        rc = _run([sys.executable, "scripts/evaluate_vlm_functional_tamp.py",
                   "--mode", "vlm", "--spec-source", "raw-replay",
                   "--specification-root", str(src),
                   "--output-root", str(args.out / trial)], env=env)
        print(f"  {trial}: replay rc={rc}")

    _run([sys.executable, "scripts/package_v3_distribution.py",
          "--root", str(args.out), "--trials", str(args.trials)], env=env)
    _run([sys.executable, "scripts/analyze_v3_distribution.py",
          "--root", str(args.out)], env=env)

    current = json.loads((args.out / "distribution_analysis.json").read_text())
    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text())
        print("\n=== STAGE DELTAS (same frozen raws) ===")
        print(f"{'stage':24} {'before':>8} {'after':>8} {'delta':>8}")
        for s in STAGES:
            b, a = base["overall"].get(s, 0), current["overall"].get(s, 0)
            print(f"{s:24} {b:8d} {a:8d} {a-b:+8d}")
        print("\n=== OUTCOME CATEGORY DELTAS ===")
        keys = set(base.get("outcome_categories", {})) | set(current.get("outcome_categories", {}))
        for k in sorted(keys, key=str):
            b = base.get("outcome_categories", {}).get(k, 0)
            a = current.get("outcome_categories", {}).get(k, 0)
            if a != b:
                print(f"{str(k):36} {b:4d} -> {a:4d}  ({a-b:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
