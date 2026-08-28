from pathlib import Path
import inspect
import math

import numpy as np
import pytest

from mujoco_scenes.generic_manipulation import classify_held_payload_contact
from mujoco_scenes.kitchen_phase_c_execution import KitchenPhaseCExecutionDispatcher
from mujoco_scenes.kitchen_pour_stir_manipulation import (
    EVIDENCE_MODE,
    PhaseCExecutionLedger,
    derive_pour_spec,
    phase_c_execution_plan,
)
from mujoco_scenes.scene_loader import KitchenScene


PLAN = [
    {"step": 1, "action": "PICK", "arguments": ["kettle"]},
    {"step": 2, "action": "POUR", "arguments": ["kettle", "mug", "water"]},
    {"step": 3, "action": "STIR", "arguments": ["spoon", "mug"]},
    {"step": 4, "action": "STIR", "arguments": ["spoon", "hidden_mug"]},
]
REGISTRY = {
    "objects": {
        "kettle": {"source_region": "countertop"},
        "spoon": {"source_region": "countertop"},
        "mug": {"source_region": "countertop"},
        "hidden_mug": {"source_region": "C1"},
    }
}


def test_phase_c_plan_excludes_operations_on_stored_targets():
    refined = phase_c_execution_plan(PLAN, REGISTRY)
    assert [row["step"] for row in refined] == [1, 2, 3]


def test_phase_c_plan_tracks_target_relocation_before_stir():
    plan = [
        {"step": 1, "action": "PICK", "arguments": ["hidden_mug"]},
        {"step": 2, "action": "PLACE", "arguments": ["hidden_mug", "countertop"]},
        {"step": 3, "action": "STIR", "arguments": ["spoon", "hidden_mug"]},
    ]
    assert [row["step"] for row in phase_c_execution_plan(plan, REGISTRY)] == [1, 2, 3]


def test_phase_c_plan_uses_canonical_table_region_over_raw_initial_evidence():
    registry = {
        "objects": {
            "kettle": {"source_region": "countertop"},
            "mug": {
                "source_region": "countertop",
                "last_evidence_source_region": "INITIAL",
            },
        }
    }
    plan = [
        {"step": 1, "action": "PICK", "arguments": ["kettle"]},
        {"step": 2, "action": "POUR", "arguments": ["kettle", "mug"]},
    ]

    assert phase_c_execution_plan(plan, registry) == plan


def test_phase_c_plan_normalizes_raw_initial_table_region():
    registry = {
        "objects": {
            "kettle": {"source_region": "countertop"},
            "mug": {"last_evidence_source_region": "INITIAL"},
        }
    }
    plan = [
        {"step": 1, "action": "PICK", "arguments": ["kettle"]},
        {"step": 2, "action": "POUR", "arguments": ["kettle", "mug"]},
    ]

    assert phase_c_execution_plan(plan, registry) == plan


def test_phase_c_plan_rejects_malformed_operator_arguments():
    with pytest.raises(ValueError, match="STIR requires"):
        phase_c_execution_plan(
            [{"step": 1, "action": "STIR", "arguments": ["spoon"]}],
            REGISTRY,
        )


def test_phase_c_ledger_rejects_duplicate_steps():
    with pytest.raises(ValueError, match="Duplicate Phase-C plan step"):
        PhaseCExecutionLedger([PLAN[1], {**PLAN[2], "step": 2}])


def test_failed_motion_cannot_commit_effect():
    ledger = PhaseCExecutionLedger(PLAN)
    result = {
        "request": {"action": "POUR", "arguments": ["kettle", "mug", "water"]},
        "success": False,
        "pour_motion_verified": False,
    }
    assert not ledger.commit(2, result)
    assert ledger.summary()["verified_event_count"] == 0


def test_ledger_duplicate_handling_is_deterministic():
    ledger = PhaseCExecutionLedger(PLAN)
    result = {
        "request": {"action": "POUR", "arguments": ["kettle", "mug", "water"]},
        "success": True,
        "pour_motion_verified": True,
    }
    assert ledger.commit(2, result)
    assert ledger.commit(2, result)
    assert not ledger.commit(2, {**result, "physical_action_telemetry": {"changed": True}})
    assert ledger.summary()["verified_event_count"] == 1


def test_phase_c_execution_has_no_payload_teleport_or_collision_exemption():
    source = Path("mujoco_scenes/kitchen_phase_c_execution.py").read_text()
    helper = Path("mujoco_scenes/kitchen_pour_stir_manipulation.py").read_text()
    assert "freejoint" not in source + helper
    assert "eq_active[" not in source + helper
    assert "collision_exemption" not in source + helper


def test_intended_stir_payload_contact_is_operator_scoped():
    tool_body, target_body = 17, 23
    assert classify_held_payload_contact(
        tool_body, target_body, frozenset((target_body,))
    ) == "ALLOWED_TASK_CONTACT"
    assert classify_held_payload_contact(
        tool_body, target_body, frozenset()
    ) == "INVALID_COLLISION"
    assert classify_held_payload_contact(
        tool_body, 29, frozenset((target_body,))
    ) == "INVALID_COLLISION"


def test_stir_orientation_family_preserves_task_axis_geometry():
    normal = np.array((0.0, 0.0, 1.0))
    local_axis = np.array((0.0, 0.0, 1.0))
    candidates = KitchenPhaseCExecutionDispatcher._stir_orientation_family(
        np.eye(3), local_axis, normal, np.array((1.0, 0.0, 0.0))
    )
    task_candidates = [
        row
        for row in candidates
        if row["provenance"] == "TASK_EQUIVALENT_TOOL_AXIS_FAMILY"
    ]
    assert {row["inclination_deg"] for row in task_candidates} == {0.0}
    assert {row["tool_roll_deg"] for row in task_candidates} == {0.0, 180.0}
    for row in task_candidates:
        axis = row["rotation"] @ local_axis
        expected = np.cos(np.deg2rad(row["inclination_deg"]))
        assert float(np.dot(axis, normal)) == pytest.approx(expected, abs=1e-7)


def test_pour_and_stir_are_kinematic_proxies():
    assert EVIDENCE_MODE == "KINEMATIC_ACTION_PROXY_NO_FLUID_DYNAMICS"


def test_pour_keeps_the_grasp_pose_between_consecutive_targets():
    source = inspect.getsource(KitchenPhaseCExecutionDispatcher.pour)
    assert "POUR_GRASP_POSE_HOVER_RECOVERY" in source
    assert "RECOVER_UPRIGHT_POST_PICK_CARRY_ARM" not in source
    assert "RECOVER_COMPACT_NAVIGATION_CARRY_ARM" not in source


def test_jar_and_kettle_pours_are_visibly_tilted():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary", robot="google"
    )
    jar = derive_pour_spec(
        scene, "s1i_compact_coffee_jar", "JAR_SOURCE"
    )
    kettle = derive_pour_spec(scene, "s1i_compact_kettle", "KETTLE")

    assert math.degrees(jar.tilt_candidates_rad[0]) == pytest.approx(55.0)
    assert math.degrees(kettle.tilt_candidates_rad[0]) == pytest.approx(22.5)
