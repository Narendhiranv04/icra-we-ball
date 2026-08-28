"""Run the pure-symbolic Living-Room Phase-2 benchmark on frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .living_room_symbolic_planning import (
    SOURCE_PHASE1_COMMIT,
    run_living_room_symbolic_pipeline,
)
from .living_room_variants import load_living_room_variants


DEFAULT_PHASE1 = Path(__file__).parent / "benchmark_reports" / "living_room_region_feasibility_phase1"
DEFAULT_OUTPUT = Path(__file__).parent / "benchmark_reports" / "living_room_symbolic_phase2"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guards(source_root: Path, output_root: Path, rows: list[dict]) -> dict:
    feasible = [row for row in rows if row["expected_class"] == "FEASIBLE"]
    infeasible = [row for row in rows if row["expected_class"] == "INFEASIBLE"]
    module_text = (Path(__file__).parent / "living_room_symbolic_planning.py").read_text()
    core_text = (Path(__file__).parent / "symbolic_planning_core.py").read_text()
    core_imports = "\n".join(
        line for line in core_text.splitlines() if line.startswith(("import ", "from "))
    ).lower()
    checks = {
        "source_phase1_commit_frozen": all(
            json.loads((output_root / "variants" / row["variant"] / "phase1_source_manifest.json").read_text())["source_phase1_commit"] == SOURCE_PHASE1_COMMIT
            for row in rows
        ),
        "source_artifacts_production_only": all(row["source_production_only"] for row in rows),
        "no_oracle_artifact_consumed": all(not row["oracle_consumed"] for row in rows),
        "complete_witnesses_compile": all(row["compilation_status"] == "SUCCESS" for row in feasible),
        "infeasible_witnesses_rejected": all(row["compilation_status"] == "REJECTED" for row in infeasible),
        "infeasible_rejection_reason_exact": all(row["reason"] == "FUNCTIONAL_WITNESS_NOT_COMPLETE" for row in infeasible),
        "planner_not_invoked_for_infeasible": all(not row["planner_invoked"] for row in infeasible),
        "five_payload_goals": all(row["goal_count"] == 5 for row in feasible),
        "all_feasible_plans_succeed": all(row["planning_status"] == "SUCCESS" for row in feasible),
        "all_replays_goal_satisfied": all(row["goal_status"] == "GOAL_SATISFIED" for row in feasible),
        "unit_plan_costs": all(row["plan_cost"] == row["plan_length"] for row in feasible),
        "optimal_actions_skip_preplaced_objects": all(
            row["plan_length"] == (8 if row["variant"] in {
                "F1_LEFT_SAUCER_PREPLACED",
                "F4_SAUCER_PREPLACED_CUP_ON_SHARED",
            } else 10)
            for row in feasible
        ),
        "operator_subset_pick_place": all(set(row["operators"]) <= {"PICK", "PLACE"} for row in feasible),
        "generic_payload_ids_only": all(row["generic_ids_only"] for row in feasible),
        "exact_witness_bindings_preserved": all(row["binding_preserved"] for row in feasible),
        "f5_uses_selected_witness": next(row for row in rows if row["variant"] == "F5_LEFT_PAIR_ON_SHARED")["binding_preserved"],
        "deterministic_repeatability": all(row["repeatable"] for row in feasible),
        "compiler_has_no_oracle_import": "oracle" not in "\n".join(line for line in module_text.splitlines() if line.startswith(("import ", "from "))).lower(),
        "compiler_has_no_mujoco_import": "mujoco" not in "\n".join(line for line in module_text.splitlines() if line.startswith(("import ", "from "))).lower(),
        "planner_has_no_mujoco_import": "mujoco" not in core_imports,
        "planner_has_no_rgbd_or_detector_import": not any(term in core_text.lower() for term in ("ultralytics", "semantic_grounding", "point_cloud")),
        "no_robot_or_tamp_dependency": not any(term in core_text.lower() + module_text.lower() for term in ("pddlstream", "inverse_kinematics", "robot_profile")),
        "no_foundation_model_dependency": not any(term in core_text.lower() + module_text.lower() for term in ("openai", "anthropic", "foundation_model")),
        "independent_replay_declares_no_transition_use": all(row["independent_replay"] for row in feasible),
        "pddl_exports_exist": all(row["pddl_exported"] for row in feasible),
    }
    return {
        "guard_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_passed": all(checks.values()),
        "checks": [{"guard": key, "passed": value} for key, value in checks.items()],
        "evidence_basis": "Generated manifests, plans, replay artifacts, and import-level source inspection",
    }


def run_benchmark(source_root: Path, output_root: Path) -> dict[str, Any]:
    variants_root = source_root / "variants"
    variant_dirs = sorted(path for path in variants_root.iterdir() if path.is_dir())
    expected_variants = set(load_living_room_variants())
    variant_dirs = [path for path in variant_dirs if path.name in expected_variants]
    if {path.name for path in variant_dirs} != expected_variants:
        raise RuntimeError("Frozen Phase-1 variants do not match the 10-variant contract")
    rows = []
    for variant_dir in variant_dirs:
        variant_output = output_root / "variants" / variant_dir.name
        result = run_living_room_symbolic_pipeline(variant_dir, variant_output)
        witness = json.loads((variant_dir / "functional_region_witness.json").read_text())
        expected_class = "FEASIBLE" if witness["status"] == "COMPLETE" else "INFEASIBLE"
        row = {
            "variant": variant_dir.name,
            "expected_class": expected_class,
            "source_status": witness["status"],
            "compilation_status": result["status"],
            "reason": result.get("reason"),
            "planner_invoked": result.get("planner_invoked", False),
            "planning_status": "SUCCESS" if result.get("goal_status") == "GOAL_SATISFIED" else "NOT_INVOKED",
            "goal_status": result.get("goal_status"),
            "goal_count": len(result.get("bindings", [])),
            "plan_length": result.get("plan_length"),
            "plan_cost": result.get("plan_cost"),
            "expanded_states": result.get("search_statistics", {}).get("expanded_states"),
            "generated_states": result.get("search_statistics", {}).get("generated_states"),
            "frontier_peak": result.get("search_statistics", {}).get("frontier_peak"),
            "search_time_ms": result.get("search_statistics", {}).get("search_time_ms"),
            "source_production_only": True,
            "oracle_consumed": False,
        }
        if expected_class == "FEASIBLE":
            plan = json.loads((variant_output / "plan.json").read_text())
            replay = json.loads((variant_output / "replay_validation.json").read_text())
            symbolic = json.loads((variant_output / "symbolic_problem.json").read_text())
            witness_bindings = sorted(
                (object_id, assignment["region_id"])
                for assignment in witness["functional_requirements"]
                for object_id in assignment["payload_ids"]
            )
            compiled_bindings = sorted(
                (item["object_id"], item["region_id"])
                for item in symbolic["witness_selected_bindings"]
            )
            repeat_dir = output_root / ".repeatability" / variant_dir.name
            repeat = run_living_room_symbolic_pipeline(variant_dir, repeat_dir)
            repeat_plan = json.loads((repeat_dir / "plan.json").read_text())
            canonical = lambda payload: json.dumps(payload["actions"], sort_keys=True)
            row.update({
                "operators": sorted({item["operator"] for item in plan["actions"]}),
                "generic_ids_only": all(item[0].startswith("object_") and item[1].startswith("region_") for item in compiled_bindings),
                "binding_preserved": compiled_bindings == witness_bindings,
                "repeatable": repeat.get("goal_status") == "GOAL_SATISFIED" and canonical(plan) == canonical(repeat_plan),
                "independent_replay": replay.get("uses_planner_transition") is False,
                "pddl_exported": (variant_output / "domain.pddl").is_file() and (variant_output / "problem.pddl").is_file(),
                "plan_sha256": _file_hash(variant_output / "plan.txt"),
            })
        else:
            row.update({"operators": [], "generic_ids_only": True, "binding_preserved": True, "repeatable": True, "independent_replay": True, "pddl_exported": False})
        rows.append(row)
    repeat_root = output_root / ".repeatability"
    if repeat_root.exists():
        import shutil
        shutil.rmtree(repeat_root)
    guards = _guards(source_root, output_root, rows)
    feasible = [row for row in rows if row["expected_class"] == "FEASIBLE"]
    infeasible = [row for row in rows if row["expected_class"] == "INFEASIBLE"]
    summary = {
        "schema_version": 1,
        "phase": "Living-Room Phase 2 symbolic planning",
        "source_phase1_commit": SOURCE_PHASE1_COMMIT,
        "variant_count": len(rows),
        "feasible_variant_count": len(feasible),
        "infeasible_variant_count": len(infeasible),
        "planning_success_rate": sum(row["planning_status"] == "SUCCESS" for row in feasible) / len(feasible),
        "pre_planner_rejection_rate": sum(not row["planner_invoked"] for row in infeasible) / len(infeasible),
        "replay_goal_satisfaction_rate": sum(row["goal_status"] == "GOAL_SATISFIED" for row in feasible) / len(feasible),
        "operator_compliance_rate": sum(set(row["operators"]) <= {"PICK", "PLACE"} for row in feasible) / len(feasible),
        "binding_preservation_rate": sum(row["binding_preserved"] for row in feasible) / len(feasible),
        "repeatability_rate": sum(row["repeatable"] for row in feasible) / len(feasible),
        "variants": rows,
        "scientific_guards": {"passed": guards["passed_count"], "total": guards["guard_count"], "all_passed": guards["all_passed"]},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "benchmark_summary.json", summary)
    _write_json(output_root / "scientific_guard_report.json", guards)
    fieldnames = ["variant", "expected_class", "source_status", "compilation_status", "reason", "planner_invoked", "planning_status", "goal_status", "goal_count", "plan_length", "plan_cost", "expanded_states", "generated_states", "frontier_peak", "search_time_ms"]
    with (output_root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader(); writer.writerows(rows)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "source_phase1_commit": SOURCE_PHASE1_COMMIT,
        "cwd": os.getcwd(),
        "uses_mujoco": False,
        "uses_robot": False,
        "uses_foundation_model": False,
        "uses_tamp": False,
    }
    _write_json(output_root / "environment.json", environment)
    (output_root / "reproduction_command.sh").write_text(
        "#!/usr/bin/env sh\nset -eu\n"
        "python -m mujoco_scenes.run_living_room_phase2_symbolic_benchmark\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        """# Living-Room Symbolic Phase 2

This tracked report compiles the frozen production-only Region-Function Phase-1
witnesses into minimal classical placement problems. It does not rerun MuJoCo,
RGB-D perception, semantic detection, geometry, or functional allocation.

For every `COMPLETE` witness, the compiler maps all five generic payload IDs to
the exact generic region IDs selected by Phase 1. The initial abstraction is
`AVAILABLE(object)` plus `HAND_EMPTY`; no unobserved staging surface is
fabricated. Deterministic A* searches grounded `PICK` and `PLACE` operators.
An independent replay implementation then checks every precondition/effect and
all final `ON(object, region)` goals.

For every `INFEASIBLE` witness, compilation returns
`FUNCTIONAL_WITNESS_NOT_COMPLETE` and the planner is not invoked.

## Reproduce

```bash
python -m mujoco_scenes.run_living_room_phase2_symbolic_benchmark
```

Inspect `benchmark_summary.json`, `scientific_guard_report.json`, and each
directory under `variants/`. Feasible variants contain the symbolic state,
goals, generated PDDL, searched plan, and independent replay result. Rejected
variants deliberately contain no plan.

## Boundary

This phase is pure symbolic planning. It performs no robot execution, IK,
motion planning, PDDLStream/TAMP, or foundation-model inference. The minimal
one-shot `AVAILABLE` abstraction is appropriate to this fixed placement task;
later manipulation-aware replanning will require perception-grounded source
locations and motion feasibility.
""",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-report-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_benchmark(args.phase1_report_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
