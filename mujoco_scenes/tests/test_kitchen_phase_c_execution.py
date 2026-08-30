import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_scenes.generic_manipulation import classify_held_payload_contact
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
    assert contract["frozen_input_plan_length"] == 26
    assert contract["plan_length"] == 23
    assert contract["operator_counts"]["POUR"] == 4
    assert contract["operator_counts"]["STIR"] == 2
    assert {row["step"] for row in contract["excluded_by_execution_scope"]} == {
        4, 9, 15
    }
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


def test_cupboard_vessel_is_not_a_phase_c_operator_target(dispatchers):
    _, phase_c = dispatchers
    pour = phase_c.pour("object_0009", "object_0016", "water")
    stir = phase_c.stir("object_0015", "object_0016")
    assert pour["failure_code"] == "POUR_TARGET_RESOLUTION_FAILED"
    assert stir["failure_code"] == "STIR_TARGET_RESOLUTION_FAILED"


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


def test_intended_stir_payload_contact_is_operator_scoped():
    held_tool_body = 17
    intended_target_body = 23
    assert classify_held_payload_contact(
        held_tool_body, intended_target_body, frozenset((intended_target_body,))
    ) == "ALLOWED_TASK_CONTACT"
    assert classify_held_payload_contact(
        held_tool_body, intended_target_body, frozenset()
    ) == "INVALID_COLLISION"
    assert classify_held_payload_contact(
        held_tool_body, 29, frozenset((intended_target_body,))
    ) == "INVALID_COLLISION"


def test_stir_orientation_family_preserves_strict_task_axis_geometry():
    normal = np.array((0.0, 0.0, 1.0))
    local_axis = np.array((0.0, 0.0, 1.0))
    candidates = KitchenPhaseCExecutionDispatcher._stir_orientation_family(
        np.eye(3), local_axis, normal, np.array((1.0, 0.0, 0.0))
    )
    task_candidates = [
        row for row in candidates
        if row["provenance"] == "TASK_EQUIVALENT_TOOL_AXIS_FAMILY"
    ]
    assert {row["inclination_deg"] for row in task_candidates} == {
        0.0,
    }
    assert {row["tool_roll_deg"] for row in task_candidates} == {
        0.0, 180.0
    }
    for row in task_candidates:
        axis = row["rotation"] @ local_axis
        expected = np.cos(np.deg2rad(row["inclination_deg"]))
        assert float(np.dot(axis, normal)) == pytest.approx(expected, abs=1e-7)


def test_serving_utensil_can_bind_to_frozen_future_target_slot(dispatchers):
    phase_b, _ = dispatchers
    resolver = phase_b.manipulation.placement_resolver
    target = resolver.prepare_future_serving_relative_destination(
        "object_0015", "object_0018"
    )
    assert target.destination_kind == "OBJECT_RELATIVE_DESTINATION"
    assert target.target_object_id == "object_0018"
    assert target.support_backend == "serving_surface"
    assert target.provenance == (
        "FROZEN_TARGET_FUTURE_SERVING_SLOT_CENTRELINE_CLEAR_RELEASE_V3"
    )
    assert resolver.resolve("object_0015", "object_0018") == target
