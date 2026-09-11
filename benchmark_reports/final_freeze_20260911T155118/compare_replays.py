#!/usr/bin/env python3
"""Stage-by-stage diff of two frozen replays over the same archived responses."""
from __future__ import annotations
import argparse, collections, glob, json, os, sys
from pathlib import Path

STAGES = ["strict_v3_valid", "task_valid", "canonical_graph_emitted",
          "executable_contract_complete", "grounding_reached",
          "complete_grounding", "astar_reached", "success", "outcome_correct"]


def load(path: str) -> dict[tuple[str, str, str], dict]:
    p = Path(path)
    if p.is_dir():
        rows = [json.loads(Path(f).read_text())
                for f in sorted(glob.glob(str(p / "rows" / "*.json")))]
    else:
        data = json.loads(p.read_text())
        rows = data["rows"] if isinstance(data, dict) else data
    return {(r["domain"], r["variant"], r["trial"]): r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    a = ap.parse_args()
    before, after = load(a.before), load(a.after)
    keys = sorted(set(before) & set(after))
    print(f"trials compared: {len(keys)}  (before {len(before)}, after {len(after)})\n")

    print(f"{'stage':34s} {'before':>8s} {'after':>8s} {'delta':>7s}")
    for stage in STAGES:
        b = sum(1 for k in keys if before[k].get(stage))
        f = sum(1 for k in keys if after[k].get(stage))
        mark = "" if f == b else ("  <-- up" if f > b else "  <-- DOWN")
        print(f"{stage:34s} {b:8d} {f:8d} {f-b:+7d}{mark}")

    for flag in ("false_completion",):
        b = sum(1 for k in keys if before[k].get(flag))
        f = sum(1 for k in keys if after[k].get(flag))
        print(f"{flag:34s} {b:8d} {f:8d} {f-b:+7d}")

    astar = [after[k].get("astar_invocations") or 0 for k in keys]
    vlm = [after[k].get("semantic_vlm_requests") for k in keys]
    harness = [k for k in keys if after[k].get("harness_failure")]
    print(f"\nmax A* invocations per trial: {max(astar) if astar else 0}")
    print(f"semantic FM requests (should all be 0 on a frozen replay): "
          f"{sorted({v for v in vlm if v is not None})}")
    print(f"harness failures: {len(harness)} {harness[:5]}")

    print("\nper-domain feasible success and outcome-correct")
    for dom in ("kitchen", "living_room", "workshop"):
        dk = [k for k in keys if k[0] == dom]
        fk = [k for k in dk if after[k].get("feasible")]
        for label, rows_ in (("feasible success", fk), ("outcome correct", dk)):
            field = "success" if label == "feasible success" else "outcome_correct"
            b = sum(1 for k in rows_ if before[k].get(field))
            f = sum(1 for k in rows_ if after[k].get(field))
            print(f"  {dom:12s} {label:18s} {b:3d}/{len(rows_):3d} -> {f:3d}/{len(rows_):3d}")

    print("\nregressions (was better before)")
    any_reg = False
    for k in keys:
        for stage in STAGES:
            if before[k].get(stage) and not after[k].get(stage):
                print(f"  {'/'.join(k):34s} {stage}: True -> False"
                      f"   after_reason={str(after[k].get('failure_reason'))[:60]}")
                any_reg = True
    if not any_reg:
        print("  none")

    print("\nnewly succeeding trials")
    for k in keys:
        if after[k].get("success") and not before[k].get("success"):
            print(f"  {'/'.join(k):34s} cov={after[k].get('gt_goal_coverage')} "
                  f"plan={after[k].get('plan_length')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
