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


# The evaluator's canonical per-trial output, and the names it actually writes.
# Reading anything else, or guessing field names, silently produced empty or
# wrong aggregates -- the failure mode that makes an experiment look finished
# when it is not.
RECORDS_FILENAME = "evaluation_records.json"
FIELD_FEASIBLE = "gt_feasible"
FIELD_SUCCESS = "full_task_satisfied"
FIELD_COVERAGE = "full_task_goal_coverage"
FIELD_OUTCOME = "outcome_correct"
FIELD_FALSE_COMPLETION = "false_completion"

# The 32 benchmark variants, so a missing trial is a hole in the grid rather
# than an absence nobody notices.  Read from the benchmark's own labels.
def expected_variants() -> list[tuple[str, str]]:
    import sys as _sys
    _sys.path.insert(0, str(REPO))
    from mujoco_scenes.final_paper_variant_labels import PREFIXES, VARIANT_LABELS
    grid = []
    for domain, labels in VARIANT_LABELS.items():
        for index in range(1, len(labels) + 1):
            grid.append((domain, f"{PREFIXES[domain]}{index}"))
    return sorted(grid)


def _authoritative_attempt_dir(repeat_dir: Path) -> Path | None:
    """The one attempt directory this repetition declares as its result.

    The runner names it in the repetition's own manifest.  Only that attempt is
    scored.  Reading several attempts and letting the newest win per variant
    looked equivalent but was not: a variant absent from the newest attempt
    silently inherited an older attempt's outcome, so a repetition could be
    scored from a mixture of runs that never existed as one run.  Without a
    manifest, the newest attempt directory alone is used -- still one attempt,
    never a mixture.
    """
    manifest = repeat_dir / "repeat_manifest.json"
    if manifest.is_file():
        try:
            named = json.loads(manifest.read_text()).get("authoritative_attempt")
        except Exception:
            named = None
        if isinstance(named, str) and named:
            candidate = repeat_dir / named
            if candidate.is_dir():
                return candidate
            return None
    attempts = sorted(p for p in repeat_dir.glob("attempt_*") if p.is_dir())
    return attempts[-1] if attempts else None


def _trial_rows(repeat_dir: Path) -> list[dict]:
    """Every per-trial record the authoritative attempt of this repetition wrote.

    Only the evaluator's canonical file, and only from one attempt.
    """
    root = _authoritative_attempt_dir(repeat_dir) or repeat_dir
    path = root / RECORDS_FILENAME
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("domain")), str(row.get("variant")))
        if key in seen:
            continue
        seen.add(key)
        rows.append({**row, "_attempt": root.name})
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

    # The whole intended grid, so a trial that never ran is a hole rather than a
    # silent absence from the denominator.  Experiment completeness and
    # conditional performance on completed calls are reported separately: a
    # missing call is not a semantic failure and must not be counted as one.
    grid = expected_variants()
    intended = len(grid) * len(repeats)
    produced = sum(len(v) for v in per_variant.values())
    holes = []
    for domain, variant in grid:
        got = len(per_variant.get((domain, variant), []))
        if got < len(repeats):
            holes.append((domain, variant, len(repeats) - got))
    print("EXPERIMENT COMPLETENESS")
    print(f"  repetitions present   : {len(repeats)}")
    print(f"  intended trial slots  : {intended}  ({len(grid)} variants x {len(repeats)} repeats)")
    print(f"  completed trials      : {produced}")
    print(f"  missing trials        : {intended - produced}")
    if holes:
        print("  holes (domain, variant, missing):")
        for row in holes:
            print(f"    {row[0]:12s} {row[1]:5s} {row[2]}")
    print()
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
        s_hits, s_total = rate(rows, FIELD_SUCCESS)
        o_hits, _ = rate(rows, FIELD_OUTCOME)
        coverage = [float(row.get(FIELD_COVERAGE) or 0) for row in rows]
        feasible = any(row.get(FIELD_FEASIBLE) for row in rows)
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
            "false_completions": sum(1 for row in rows if row.get(FIELD_FALSE_COMPLETION)),
        })

    print()
    for scope in ("kitchen", "living_room", "workshop", "ALL"):
        rows = [row for (domain, _), group in per_variant.items() for row in group
                if scope == "ALL" or domain == scope]
        feasible_rows = [row for row in rows if row.get(FIELD_FEASIBLE)]
        s_hits, s_total = rate(feasible_rows, FIELD_SUCCESS)
        o_hits, o_total = rate(rows, FIELD_OUTCOME)
        low, high = wilson(s_hits, s_total)
        print(f"{scope:12s} feasible success {s_hits:4d}/{s_total:<5d} "
              f"= {(s_hits/s_total if s_total else 0):.3f} [{low}, {high}]   "
              f"outcome correct {o_hits}/{o_total}   "
              f"false completions {sum(1 for r in rows if r.get(FIELD_FALSE_COMPLETION))}")

    categories = collections.Counter(
        str(row.get("outcome_category") or row.get("failure_category") or "UNRECORDED")
        for group in per_variant.values() for row in group)
    print("\nfailure / outcome categories")
    for name, count in categories.most_common():
        print(f"  {count:5d}  {name}")

    destination = args.out or (args.root / "AGGREGATE")
    Path(str(destination) + ".json").write_text(json.dumps({
        "repetitions_present": len(repeats), "unfinished_repetitions": unfinished,
        "completeness": {"intended_trial_slots": intended, "completed_trials": produced,
                         "missing_trials": intended - produced,
                         "holes": [{"domain": d, "variant": v, "missing": n} for d, v, n in holes]},
        "per_variant": out_rows, "categories": dict(categories),
    }, indent=2) + "\n")
    if out_rows:
        with open(str(destination) + ".csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)
    print(f"\nwrote {destination}.json and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
