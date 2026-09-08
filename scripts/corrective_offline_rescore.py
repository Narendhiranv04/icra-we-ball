#!/usr/bin/env python3
"""Read-only corrective rescoring of archived V1/V2 benchmark responses."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import candidate_plan_valid, full_task_coverage, read_json
from mujoco_scenes.functional_tamp_pipeline.raw_semantic_evaluation import evaluate_raw_semantics
from scripts.evaluate_vlm_functional_tamp import CANONICAL_TASK_INSTRUCTIONS


DATASETS = {
    "previous_v2_development": Path("benchmark_reports/final_post_optimization_32x1_20260908"),
    "posthoc_additional_stress_test": Path("benchmark_reports/held_out_generalization_15x1_20260908"),
    "archived_v1_baseline": Path("benchmark_reports/raw_replay_final_v1"),
}


def _raw_document(run_dir: Path) -> Any:
    graph = read_json(run_dir / "functional_specification.json")
    raw = graph.get("metadata", {}).get("raw_vlm_response")
    if raw is not None:
        return raw
    diagnostics = sorted((run_dir / "fm_diagnostics").glob("fm_call_*.json"))
    if not diagnostics:
        return None
    content = read_json(diagnostics[0]).get("content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return None
    return content


def _records(root: Path) -> list[dict[str, Any]]:
    records = read_json(root / "evaluation_records.json", default=[])
    return records if isinstance(records, list) else []


def _snapshot_state(run_dir: Path) -> tuple[bool | None, list[str]]:
    snapshots = read_json(run_dir / "grounding_snapshots.json", default=[])
    if not isinstance(snapshots, list) or not snapshots:
        return None, []
    grounding = snapshots[0].get("grounding", {})
    complete = bool(grounding.get("complete") or grounding.get("satisfied") or grounding.get("status") == "COMPLETE")
    return complete, [str(row.get("search_state")) for row in snapshots if row.get("search_state")]


def _first_cause(record: dict[str, Any]) -> str | None:
    if record["infrastructure_error"]:
        return "EVAL_INFRA_ERROR"
    if not record["gt_feasible"] or record["full_task_satisfied"]:
        return None
    if not record["fm_complete_task_contract"]:
        return "TASK_SPECIFICATION_FAILURE"
    if not record["required_contract_complete"]:
        return "GRAPH_COMPILATION_FAILURE"
    if not record["final_grounding_complete"]:
        if record["search_exhausted"] and record["regions_inspected"]:
            return "OBJECT_DISCOVERY_FAILURE"
        return "FUNCTIONAL_ASSIGNMENT_FAILURE"
    return "PLANNING_FAILURE"


def score_run(dataset: str, source_root: Path, old: dict[str, Any]) -> dict[str, Any]:
    domain, variant = old["domain"], old["variant"]
    run_dir = source_root / domain / variant / "vlm"
    graph = read_json(run_dir / "functional_specification.json")
    metadata = graph.get("metadata", {})
    trace = metadata.get("canonicalization_trace", {})
    raw = _raw_document(run_dir)
    raw_score = evaluate_raw_semantics(domain, CANONICAL_TASK_INSTRUCTIONS[domain], raw)
    grounding = read_json(run_dir / "graph_grounding_result.json")
    initial_complete, search_states = _snapshot_state(run_dir)
    final_complete = bool(grounding.get("complete") or grounding.get("status") == "COMPLETE")
    regions = list(old.get("regions_inspected", []))
    total, satisfied, coverage, full = full_task_coverage(domain, run_dir)
    terminal = str(old.get("terminal_status", old.get("candidate_plan_status", "")))
    compiler_complete = bool(metadata.get("required_contract_complete", False))
    unresolved = [
        *metadata.get("unresolved_semantics", []), *trace.get("unresolved_roles", []),
        *trace.get("unresolved_required_relations", []), *trace.get("unresolved_required_operations", []),
        *trace.get("disabled_groups", []),
    ]
    required_complete = bool(compiler_complete and metadata.get("canonicalization_status") == "FULL" and not unresolved)
    record = {
        "dataset": dataset, "source_root": str(source_root), "domain": domain, "variant": variant,
        "gt_feasible": bool(old.get("gt_feasible")),
        "structural_json_valid": isinstance(raw, dict), "raw_schema": raw_score["schema"],
        "fm_role_precision": raw_score["role"]["precision"], "fm_role_recall": raw_score["role"]["recall"], "fm_role_f1": raw_score["role"]["f1"],
        "fm_relation_precision": raw_score["relation"]["precision"], "fm_relation_recall": raw_score["relation"]["recall"], "fm_relation_f1": raw_score["relation"]["f1"],
        "fm_operation_precision": raw_score["operation"]["precision"], "fm_operation_recall": raw_score["operation"]["recall"], "fm_operation_f1": raw_score["operation"]["f1"],
        "fm_count_correct": raw_score["count_correct"], "fm_binding_correct": raw_score["binding_correct"],
        "fm_complete_task_contract": raw_score["complete_task_contract"],
        "compiler_roles_interpreted": sum(row.get("status") == "CANONICAL_EXECUTABLE_SEMANTIC" for row in trace.get("roles", [])),
        "compiler_relations_interpreted": sum(row.get("status") == "CANONICAL_EXECUTABLE_SEMANTIC" for row in trace.get("relations", [])),
        "compiler_operations_interpreted": sum(row.get("status") in {"CANONICAL_OPERATION_GROUP", "STATIC_ALREADY_SATISFIED"} for row in trace.get("groups", [])),
        "unresolved_required_semantics": unresolved, "required_contract_complete": required_complete,
        "initial_grounding_complete": initial_complete, "search_state_trace": search_states,
        "search_eligible": required_complete and "SEARCH_RECOVERABLE" in search_states,
        "regions_inspected": regions, "search_exhausted": "SEARCH_EXHAUSTED" in search_states or bool(old.get("search_exhausted")),
        "final_grounding_complete": final_complete,
        "causal_search_recovery": initial_complete is False and bool(regions) and final_complete,
        "astar_invoked": int(old.get("astar_invocations", 0)) > 0,
        "candidate_plan_length": int(old.get("candidate_plan_length", 0)),
        "candidate_plan_valid": candidate_plan_valid(run_dir),
        "goal_groups_satisfied": satisfied, "goal_groups_total": total, "goal_coverage": coverage,
        "full_task_satisfied": full, "terminal_status": terminal,
        "infrastructure_error": terminal == "PIPELINE_EXCEPTION",
    }
    record["first_cause"] = _first_cause(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root or Path(f"benchmark_reports/corrective_pass_offline_rescore_{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    rows = [score_run(name, root, record) for name, root in DATASETS.items() for record in _records(root)]
    (output / "corrective_failure_funnel.json").write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(), "records": rows}, indent=2) + "\n")
    fields = [key for row in rows for key in row if key != "unresolved_required_semantics"]
    fields = list(dict.fromkeys(fields))
    with (output / "corrective_failure_funnel.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    lines = ["# Corrective Offline Failure Funnel", "", f"Generated from {len(rows)} archived runs; source artifacts were read-only.", "", "| Dataset | Domain | Runs | FM role recall | FM relation recall | FM operation recall | Complete raw | Executable contract | Grounding complete | Valid plan | Full task |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for domain in ("kitchen", "living_room", "workshop"):
            subset = [row for row in rows if row["dataset"] == dataset and row["domain"] == domain]
            if not subset: continue
            avg = lambda key: sum(float(row.get(key) or 0.0) for row in subset) / len(subset)
            lines.append(f"| {dataset} | {domain} | {len(subset)} | {avg('fm_role_recall'):.3f} | {avg('fm_relation_recall'):.3f} | {avg('fm_operation_recall'):.3f} | {avg('fm_complete_task_contract'):.3f} | {avg('required_contract_complete'):.3f} | {avg('final_grounding_complete'):.3f} | {avg('candidate_plan_valid'):.3f} | {avg('full_task_satisfied'):.3f} |")
    (output / "corrective_failure_funnel.md").write_text("\n".join(lines) + "\n")
    print(output)
    print(json.dumps({name: sum(row["dataset"] == name for row in rows) for name in DATASETS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
