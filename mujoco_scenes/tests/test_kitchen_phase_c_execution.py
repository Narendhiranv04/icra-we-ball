import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_scenes.kitchen_phase_c_execution import KitchenPhaseCExecutionDispatcher
from mujoco_scenes.kitchen_pour_stir_manipulation import (
    EVIDENCE_MODE,
    PhaseCExecutionLedger,
    derive_pour_spec,
    derive_target_opening,
    derive_tool_tip,
)
from mujoco_scenes.run_kitchen_phase_b_freeze_evidence import (
    PRIMARY_FROZEN,
    primary_validation_dispatcher,
)


@pytest.fixture(scope="module")
def frozen_inputs():
    return (
        json.loads((PRIMARY_FROZEN / "object_registry.json").read_text()),
        json.loads((PRIMARY_FROZEN / "plan.json").read_text()),
    )


@pytest.fixture(scope="module")
def dispatchers(frozen_inputs):
    scene, _, _, phase_b = primary_validation_dispatcher()
    registry, plan = frozen_inputs
    return phase_b, KitchenPhaseCExecutionDispatcher(phase_b, registry, plan)


def test_plan_contract_is_exact_and_non_vacuous():
    contract = json.loads(
        Path("runs/phaseC_plan_contract/phaseC_plan_contract.json").read_text()
    )
    assert contract["plan_length"] == 26
    assert contract["operator_counts"]["POUR"] == 6
    assert contract["operator_counts"]["STIR"] == 3
    assert len(contract["ordered_actions"]) == contract["plan_length"]


def test_phase_b_still_rejects_phase_c_operators(dispatchers):
    phase_b, _ = dispatchers
    for operator in ("POUR", "STIR"):
        result = phase_b.execute_phase2_action(
            {"action": operator, "arguments": ["object_0009", "object_0001"]}
        )
        assert not result["success"]
        assert result["status"] == "UNSUPPORTED_PHASE_C_OPERATOR"


def test_phase_c_requires_exact_frozen_pair(dispatchers):
    _, phase_c = dispatchers
    result = phase_c.pour("object_0009", "object_0002", "water")
    assert not result["success"]
    assert result["failure_code"] == "POUR_TARGET_RESOLUTION_FAILED"
    assert not result["symbolic_effects_applied"]


def test_pour_requires_existing_held_source(dispatchers):
    _, phase_c = dispatchers
    result = phase_c.pour("object_0009", "object_0001", "water")
    assert not result["success"]
    assert result["failure_code"] == "POUR_SOURCE_NOT_HELD"
    assert not result["symbolic_effects_applied"]


def test_stir_requires_existing_held_tool(dispatchers):
    _, phase_c = dispatchers
    result = phase_c.stir("object_0015", "object_0001")
    assert not result["success"]
    assert result["failure_code"] == "STIR_TOOL_NOT_HELD"
    assert not result["symbolic_effects_applied"]


def test_opening_geometry_requires_frozen_measurement(dispatchers, frozen_inputs):
    phase_b, _ = dispatchers
    registry, _ = frozen_inputs
    row = copy.deepcopy(registry["objects"]["object_0001"])
    row["geometric_properties"]["opening_width_m"]["value"] = None
    with pytest.raises(ValueError, match="OPENING_GEOMETRY_UNAVAILABLE"):
        derive_target_opening(
            phase_b.scene,
            row,
            "object_0001",
            phase_b.binding_by_id["object_0001"]["physical_backend_body"],
        )


def test_opening_radius_scales_from_each_frozen_target(dispatchers, frozen_inputs):
    phase_b, _ = dispatchers
    registry, _ = frozen_inputs
    radii = []
    for object_id in ("object_0001", "object_0003", "object_0016"):
        opening = derive_target_opening(
            phase_b.scene,
            registry["objects"][object_id],
            object_id,
            phase_b.binding_by_id[object_id]["physical_backend_body"],
        )
        radii.append(opening.opening_half_extents_m)
        assert min(opening.opening_half_extents_m) > opening.safety_margin_m
    assert len(set(radii)) == 3


def test_pour_specs_are_family_level_and_content_scope_is_explicit(dispatchers):
    phase_b, _ = dispatchers
    specs = []
    for object_id in ("object_0009", "object_0010"):
        binding = phase_b.binding_by_id[object_id]
        specs.append(derive_pour_spec(
            phase_b.scene,
            binding["physical_backend_body"],
            binding["grasp_family"],
        ))
    assert {spec.family for spec in specs} == {"KETTLE", "JAR_SOURCE"}
    assert all("object_" not in spec.outlet_provenance for spec in specs)
    assert EVIDENCE_MODE == "KINEMATIC_ACTION_PROXY_NO_FLUID_DYNAMICS"


def test_active_tip_is_geometry_derived_not_backend_named(dispatchers):
    phase_b, _ = dispatchers
    object_id = "object_0015"
    binding = phase_b.binding_by_id[object_id]
    length = phase_b.inventory_by_id[object_id]["observed_dimensions_m"]["length"]
    tip = derive_tool_tip(
        phase_b.scene, binding["physical_backend_body"], float(length)
    )
    assert np.linalg.norm(tip.active_tip_offset_from_gripper_m) == pytest.approx(
        length, rel=1e-6
    )
    assert binding["physical_backend_body"] not in tip.provenance


def test_failed_motion_cannot_commit_ledger_effect(frozen_inputs):
    _, plan = frozen_inputs
    ledger = PhaseCExecutionLedger(plan)
    result = {
        "request": {"action": "POUR", "arguments": ["object_0009", "object_0001", "water"]},
        "success": False,
        "pour_motion_verified": False,
    }
    assert not ledger.commit(2, result)
    assert ledger.summary()["verified_event_count"] == 0


def test_ledger_duplicate_handling_is_deterministic(frozen_inputs):
    _, plan = frozen_inputs
    ledger = PhaseCExecutionLedger(plan)
    result = {
        "request": {"action": "POUR", "arguments": ["object_0009", "object_0001", "water"]},
        "success": True,
        "pour_motion_verified": True,
    }
    assert ledger.commit(2, result)
    assert ledger.commit(2, result)
    changed = {**result, "physical_action_telemetry": {"different": True}}
    assert not ledger.commit(2, changed)
    assert ledger.summary()["verified_event_count"] == 1


def test_phase_c_execution_does_not_write_payload_qpos_or_add_exemptions():
    source = Path("mujoco_scenes/kitchen_phase_c_execution.py").read_text()
    helper = Path("mujoco_scenes/kitchen_pour_stir_manipulation.py").read_text()
    assert "freejoint" not in source + helper
    assert "eq_active[" not in source + helper
    assert "PHASE_B_MOUNT_ALLOWANCES" not in source + helper
    assert "collision_exemption" not in source + helper
