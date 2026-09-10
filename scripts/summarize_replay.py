#!/usr/bin/env python3
"""Print the stage/outcome/quality tables for one enriched replay artifact."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path

STAGES = [
    ("wire_valid", "strict_v3_valid"),
    ("task_valid", "task_valid"),
    ("canonical_graph_emitted", "canonical_graph_emitted"),
    ("executable_contract_complete", "executable_contract_complete"),
    ("grounding_reached", "grounding_reached"),
    ("complete_grounding", "complete_grounding"),
    ("astar_invoked", "astar_reached"),
    ("success", "success"),
]
OUTCOMES = ["TASK_SPECIFICATION_FAILURE", "GRAPH_COMPILATION_FAILURE",
            "OBJECT_DISCOVERY_FAILURE", "FUNCTIONAL_ASSIGNMENT_FAILURE",
            "PLANNING_FAILURE", "SUCCESS", None]
DOMAINS = ["kitchen", "living_room", "workshop"]


def block(rows):
    out = {name: sum(1 for r in rows if r.get(key)) for name, key in STAGES}
    out["trials"] = len(rows)
    out["gt_goals"] = sum(r.get("gt_goals_satisfied", 0) or 0 for r in rows)
    out["gt_goals_total"] = sum(r.get("gt_goals_total", 0) or 0 for r in rows)
    out["feasible_fully_done"] = sum(
        1 for r in rows if r.get("feasible") and r.get("gt_full_task_satisfied"))
    out["false_completions"] = sum(
        1 for r in rows if r.get("success") and not r.get("gt_full_task_satisfied"))
    out["outcome_correct"] = sum(1 for r in rows if r.get("outcome_correct"))
    out["harness_failures"] = sum(
        1 for r in rows if r.get("pipeline_status") == "PIPELINE_EXCEPTION" or r.get("derive_exception"))
    out["fm_calls"] = sum(r.get("semantic_vlm_requests", 0) or 0 for r in rows)
    out["max_astar"] = max([r.get("astar_invocations", 0) or 0 for r in rows] or [0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    rows = json.loads(Path(args.rows).read_text())
    summary = {"label": args.label, "TOTAL": block(rows)}
    for dom in DOMAINS:
        summary[dom] = block([r for r in rows if r["domain"] == dom])
    counts = collections.Counter(r.get("outcome_category") for r in rows)
    summary["outcomes"] = {str(k): counts.get(k, 0) for k in OUTCOMES}
    summary["outcomes_by_domain"] = {
        dom: dict(collections.Counter(
            str(r.get("outcome_category")) for r in rows if r["domain"] == dom))
        for dom in DOMAINS}

    keys = ["trials"] + [n for n, _ in STAGES] + [
        "gt_goals", "feasible_fully_done", "false_completions", "outcome_correct",
        "harness_failures", "fm_calls", "max_astar"]
    width = max(len(k) for k in keys) + 1
    header = f"{'metric':<{width}}" + "".join(f"{d[:8]:>10}" for d in DOMAINS) + f"{'TOTAL':>10}"
    print(f"### {args.label}")
    print(header)
    for k in keys:
        print(f"{k:<{width}}" + "".join(f"{summary[d][k]:>10}" for d in DOMAINS)
              + f"{summary['TOTAL'][k]:>10}")
    print()
    print(f"{'outcome':<32}" + "".join(f"{d[:8]:>10}" for d in DOMAINS) + f"{'TOTAL':>10}")
    for name in OUTCOMES:
        key = str(name)
        total = summary["outcomes"].get(key, 0)
        if not total:
            continue
        print(f"{key:<32}" + "".join(f"{summary['outcomes_by_domain'][d].get(key,0):>10}" for d in DOMAINS)
              + f"{total:>10}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
