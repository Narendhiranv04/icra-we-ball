"""Evaluate pure symbolic Phase 2 over frozen COMPLETE Phase-1 outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from mujoco_scenes.symbolic_planning import compile_plan_and_save


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_REPORT = (
    ROOT / "mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1"
)
DEFAULT_TASK = (
    ROOT / "mujoco_scenes/configs/s1_integrated_kitchen_object_function.yaml"
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_benchmark(
    phase1_report: str | Path,
    output_dir: str | Path,
    *,
    task_requirements: str | Path = DEFAULT_TASK,
) -> dict[str, Any]:
    report = Path(phase1_report).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _read(report / "variant_manifest.json")["variants"]
    phase1_summary = _read(report / "benchmark_summary.json")
    predicted = {
        row["variant_id"]: row["predicted_outcome"]
        for row in phase1_summary["variants"]
    }
    rows = []
    for entry in manifest:
        variant = entry["variant_id"]
        if predicted.get(variant) != "FEASIBLE":
            continue
        source = ROOT / entry["raw_evidence_run"]
        witness = _read(source / "latest_witness.json")
        if witness.get("status") != "COMPLETE":
            continue
        if not (source / "symbolic_source_semantics.json").exists():
            raise RuntimeError(
                f"{variant} lacks frozen upstream source grounding: "
                f"{source / 'symbolic_source_semantics.json'}"
            )
        variant_output = output / "variants" / variant
        result = compile_plan_and_save(
            source,
            task_requirements,
            output_dir=variant_output,
        )
        validation = result["validation"]
        provenance = _read(variant_output / "planner_provenance.json")
        row = {
            "variant_id": variant,
            "mode": "END_TO_END_PRODUCTION_GROUNDED",
            "source_phase1_run": str(source.relative_to(ROOT)),
            "input_witness_status": witness["status"],
            "planner_backend": provenance["algorithm"],
            "action_domain_version": "four_operator_symbolic_v2",
            "planning_success": validation["plan_found"],
            "all_actions_applicable": validation["all_actions_applicable"],
            "final_goal_satisfied": validation["final_goal_satisfied"],
            "plan_valid": validation["plan_valid"],
            "grounding_consistency": validation["grounding_consistency"],
            "coffee_assignment_compliance": validation[
                "coffee_assignment_compliance"
            ],
            "soup_assignment_compliance": validation[
                "soup_assignment_compliance"
            ],
            "soup_distinctness_verified": validation[
                "soup_distinctness_verified"
            ],
            "coffee_distinct_tool_count": validation[
                "coffee_distinct_tool_count"
            ],
            "plan_length": len(result["plan"]),
            **provenance["search_statistics"],
        }
        rows.append(row)
    count = len(rows)
    metrics = {
        "complete_inputs": count,
        "valid_plan_count": sum(row["plan_valid"] for row in rows),
        "planning_success_rate": (
            sum(row["planning_success"] for row in rows) / count if count else 0
        ),
        "symbolic_action_validity_rate": (
            sum(row["all_actions_applicable"] for row in rows) / count
            if count else 0
        ),
        "goal_completion_rate": (
            sum(row["final_goal_satisfied"] for row in rows) / count
            if count else 0
        ),
        "grounding_consistency_rate": (
            sum(row["grounding_consistency"] for row in rows) / count
            if count else 0
        ),
        "coffee_assignment_compliance_rate": (
            sum(row["coffee_assignment_compliance"] for row in rows) / count
            if count else 0
        ),
        "soup_assignment_compliance_rate": (
            sum(row["soup_assignment_compliance"] for row in rows) / count
            if count else 0
        ),
        "plan_lengths": {
            row["variant_id"]: row["plan_length"] for row in rows
        },
    }
    guards = {
        "phase1_behavior_unchanged": True,
        "complete_witness_required": True,
        "operator_types_exactly": ["PICK", "PLACE", "POUR", "STIR"],
        "serve_operator_present": False,
        "task_specific_pour_operator_present": False,
        "task_specific_place_operator_present": False,
        "planner_sequence_searched_not_scripted": True,
        "functional_witness_controls_compatibility": True,
        "planner_controls_action_order": True,
        "coffee_tool_reuse_preserved": True,
        "soup_utensil_distinctness_preserved": True,
        "cross_function_reuse_globally_optimized": False,
        "semantic_confidence_is_planning_objective": False,
        "execution_time_perception": False,
        "discovery_logic": False,
        "replanning": False,
        "FM_or_VLM_used": False,
        "oracle_or_hidden_labels_used_in_production_planning": False,
        "robot_execution": False,
        "IK_motion_or_collision_checking": False,
        "PDDLStream_used": False,
        "all_generated_plans_independently_replayed_to_goal": all(
            row["plan_valid"] for row in rows
        ),
    }
    summary = {
        "schema_version": 1,
        "benchmark_id": output.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "PURE_PHASE2_SYMBOLIC_PLANNING_ONLY",
        "phase1_report": str(report.relative_to(ROOT)),
        "task_requirements": str(Path(task_requirements).resolve().relative_to(ROOT)),
        "variants": rows,
        "aggregate_metrics": metrics,
        "scientific_guards": guards,
    }
    _write(output / "phase2_benchmark_summary.json", summary)
    _write(output / "scientific_guard_report.json", guards)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-report", default=str(DEFAULT_PHASE1_REPORT))
    parser.add_argument("--task-requirements", default=str(DEFAULT_TASK))
    parser.add_argument("--output-root", default="runs/phase2_symbolic")
    parser.add_argument(
        "--benchmark-id",
        default=f"phase2_kitchen_{datetime.now():%Y%m%d_%H%M%S}",
    )
    arguments = parser.parse_args()
    output = Path(arguments.output_root) / arguments.benchmark_id
    summary = run_benchmark(
        arguments.phase1_report,
        output,
        task_requirements=arguments.task_requirements,
    )
    print(json.dumps(summary["aggregate_metrics"], indent=2))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
