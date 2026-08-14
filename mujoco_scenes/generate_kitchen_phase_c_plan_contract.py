"""Materialize the immutable Phase-C input contract from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .kitchen_pour_stir_manipulation import phase_c_execution_plan


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "runs/integrated_no_pot_clearance_seed19_20260807"
REFERENCE_PHASE2_ROOT = (
    ROOT
    / "mujoco_scenes/benchmark_reports/kitchen_symbolic_phase2/variants/F1_INITIAL_COMPLETE"
)
OUTPUT = ROOT / "runs/phaseC_plan_contract/phaseC_plan_contract.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_contract() -> dict[str, Any]:
    plan_path = FROZEN_ROOT / "plan.json"
    assignments_path = FROZEN_ROOT / "grounded_role_assignments.json"
    registry_path = FROZEN_ROOT / "object_registry.json"
    plan = json.loads(plan_path.read_text())
    registry = json.loads(registry_path.read_text())
    execution_plan = phase_c_execution_plan(plan, registry)
    assignments = json.loads(assignments_path.read_text())

    pours = [row for row in execution_plan if row["action"].upper() == "POUR"]
    stirs = [row for row in execution_plan if row["action"].upper() == "STIR"]
    excluded = [row for row in plan if row not in execution_plan]
    coffee_targets = list(assignments["coffee_targets"])
    soup_targets = list(assignments["soup_targets"])
    return {
        "schema_version": "kitchen_phase_c_plan_contract_v1",
        "authority": "FROZEN_PHASE1_PHASE2_INPUTS",
        "frozen_phase1_root": str(FROZEN_ROOT.relative_to(ROOT)),
        "frozen_phase2_root": str(FROZEN_ROOT.relative_to(ROOT)),
        "committed_symbolic_phase2_reference_root": str(
            REFERENCE_PHASE2_ROOT.relative_to(ROOT)
        ),
        "committed_symbolic_phase2_reference_authoritative": False,
        "files": {
            "plan": {"path": str(plan_path.relative_to(ROOT)), "sha256": _sha256(plan_path)},
            "assignments": {
                "path": str(assignments_path.relative_to(ROOT)),
                "sha256": _sha256(assignments_path),
            },
            "registry": {
                "path": str(registry_path.relative_to(ROOT)),
                "sha256": _sha256(registry_path),
            },
        },
        "plan_length": len(execution_plan),
        "frozen_input_plan_length": len(plan),
        "operator_counts": dict(sorted(Counter(row["action"].upper() for row in execution_plan).items())),
        "ordered_actions": execution_plan,
        "excluded_by_execution_scope": excluded,
        "ordered_pour_actions": pours,
        "ordered_stir_actions": stirs,
        "source_roles": assignments["source_roles"],
        "target_roles": {
            "coffee_targets": coffee_targets,
            "soup_targets": soup_targets,
        },
        "coffee_stirring": assignments["coffee_stirring"],
        "soup_serving": assignments["soup_serving"],
        "execution_scope": "POUR_STIR_TARGETS_ON_COUNTERTOP_OR_BOX_ONLY;CUPBOARD_PICK_PLACE_ONLY",
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_contract(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
