#!/usr/bin/env python3
"""Aggregate the independent repetitions of the live matrix.

Offline scoring only: reads what the runs wrote and computes nothing that could
feed back into a run.  Every repetition that is present contributes, finished or
not, and a repetition is never dropped for scoring badly -- an unfinished one is
counted and named so the denominator stays honest.

Reports per-variant success out of the number of repetitions that actually
produced that trial, per-domain and overall means, the failure-category
breakdown, and a Wilson interval for each rate.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: sane at the 0/n and n/n ends, where normal is not."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4))


def _trial_rows(repeat_dir: Path) -> list[dict]:
    """Every per-trial record this repetition wrote, however it is stored."""
    for name in ("trial_rows.json", "results.json", "summary.json"):
        path = repeat_dir / name
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            rows = data.get("rows") or data.get("trials") or data.get("results") or (
                data if isinstance(data, list) else None)
            if isinstance(rows, list) and rows:
                return rows
    rows = []
    for path in sorted(repeat_dir.glob("*/*/*/result.json")) + sorted(
            repeat_dir.glob("*/*/result.json")):
        try:
            row = json.loads(path.read_text())
        except Exception:
            continue
        row.setdefault("variant", path.parent.parent.name)
        row.setdefault("domain", path.parent.parent.parent.name)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    repeats = sorted(p for p in args.root.glob("repeat_*") if p.is_dir())
    if not repeats:
        raise SystemExit(f"no repeat_* directories under {args.root}")

    per_variant: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    unfinished: list[str] = []
    for repeat_dir in repeats:
        manifest = repeat_dir / "repeat_manifest.json"
        finished = False
        if manifest.is_file():
            try:
                finished = bool(json.loads(manifest.read_text()).get("finished"))
            except Exception:
                finished = False
        if not finished:
            unfinished.append(repeat_dir.name)
        for row in _trial_rows(repeat_dir):
            key = (str(row.get("domain")), str(row.get("variant")))
            per_variant[key].append({**row, "repeat": repeat_dir.name})

    print(f"repetitions present: {len(repeats)}  unfinished: {unfinished or 'none'}")
    print("Unfinished repetitions still contribute every trial they produced; "
          "none is dropped for its outcome.\n")

    def rate(rows, field):
        total = len(rows)
        hits = sum(1 for row in rows if row.get(field))
        return hits, total

    out_rows = []
    print(f"{'domain':12s} {'variant':8s} {'feasible':9s} {'success':>10s} "
          f"{'outcome ok':>11s} {'goal cov':>9s}")
    for (domain, variant), rows in sorted(per_variant.items()):
        s_hits, s_total = rate(rows, "success")
        o_hits, _ = rate(rows, "outcome_correct")
        coverage = [float(row.get("gt_goal_coverage") or 0) for row in rows]
        feasible = any(row.get("feasible") for row in rows)
        low, high = wilson(s_hits, s_total)
        print(f"{domain:12s} {variant:8s} {'yes' if feasible else 'no':9s} "
              f"{s_hits:4d}/{s_total:<5d} {o_hits:4d}/{s_total:<6d} "
              f"{(sum(coverage)/len(coverage) if coverage else 0):9.3f}")
        out_rows.append({
            "domain": domain, "variant": variant, "feasible": feasible,
            "repeats_observed": s_total, "successes": s_hits,
            "success_rate": round(s_hits / s_total, 4) if s_total else 0.0,
            "success_ci_low": low, "success_ci_high": high,
            "outcome_correct": o_hits,
            "mean_gt_goal_coverage": round(sum(coverage) / len(coverage), 4) if coverage else 0.0,
            "false_completions": sum(1 for row in rows if row.get("false_completion")),
        })

    print()
    for scope in ("kitchen", "living_room", "workshop", "ALL"):
        rows = [row for (domain, _), group in per_variant.items() for row in group
                if scope == "ALL" or domain == scope]
        feasible_rows = [row for row in rows if row.get("feasible")]
        s_hits, s_total = rate(feasible_rows, "success")
        o_hits, o_total = rate(rows, "outcome_correct")
        low, high = wilson(s_hits, s_total)
        print(f"{scope:12s} feasible success {s_hits:4d}/{s_total:<5d} "
              f"= {(s_hits/s_total if s_total else 0):.3f} [{low}, {high}]   "
              f"outcome correct {o_hits}/{o_total}   "
              f"false completions {sum(1 for r in rows if r.get('false_completion'))}")

    categories = collections.Counter(
        str(row.get("outcome_category") or row.get("failure_category") or "UNRECORDED")
        for group in per_variant.values() for row in group)
    print("\nfailure / outcome categories")
    for name, count in categories.most_common():
        print(f"  {count:5d}  {name}")

    destination = args.out or (args.root / "AGGREGATE")
    Path(str(destination) + ".json").write_text(json.dumps({
        "repetitions_present": len(repeats), "unfinished_repetitions": unfinished,
        "per_variant": out_rows, "categories": dict(categories),
    }, indent=2) + "\n")
    with open(str(destination) + ".csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {destination}.json and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
