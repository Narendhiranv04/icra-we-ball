import json
from copy import deepcopy

from mujoco_scenes.kitchen_execution_entities import (
    ExecutionCandidate,
    KitchenExecutionEntityResolver,
    SourceKind,
    build_phase_b_inventory,
    source_context,
)
from mujoco_scenes.run_kitchen_phase_b_freeze_evidence import (
    apply_approved_within_region_execution_calibration,
)


def _record(label, xyz, region="countertop"):
    return {
        "source_region": region,
        "first_seen_stage": 0,
        "centroid_world_m": {
            "value": xyz,
            "measurement_cloud_path": "stages/000/evidence/fused.ply",
        },
        "semantics": {"validated": {"canonical_label": label}},
    }


def _measured_record(label, xyz, dimensions, region="B1"):
    row = _record(label, xyz, region)
    row["dimensions_m"] = {
        axis: {"value": value, "status": "MEASURED"}
        for axis, value in zip(("length", "width", "height"), dimensions)
    }
    row["principal_axis_world"] = {"value": [1.0, 0.0, 0.0]}
    return row


def test_approved_region_calibration_matches_semantics_and_dimensions_not_ids():
    frozen = {"objects": {
        "frozen_spoon": _measured_record("spoon", [0.0, 0.0, 0.0], [0.2, 0.04, 0.02]),
        "frozen_bowl": _measured_record("bowl", [1.0, 0.0, 0.0], [0.14, 0.14, 0.07]),
    }}
    current = {"objects": {
        "fresh_left": _measured_record("bowl", [0.0, 0.0, 0.0], [0.141, 0.139, 0.071]),
        "fresh_right": _measured_record("spoon", [1.0, 0.0, 0.0], [0.201, 0.041, 0.02]),
    }}

    calibrated, audit = apply_approved_within_region_execution_calibration(
        deepcopy(frozen), current, region="B1"
    )

    assert calibrated["objects"]["frozen_bowl"]["centroid_world_m"]["value"] == [0.0, 0.0, 0.0]
    assert calibrated["objects"]["frozen_spoon"]["centroid_world_m"]["value"] == [1.0, 0.0, 0.0]
    assert all(
        row["instance_token_used"] is False
        and row["backend_body_name_used"] is False
        for row in (
            calibrated["objects"]["frozen_bowl"]["execution_scene_calibration"],
            calibrated["objects"]["frozen_spoon"]["execution_scene_calibration"],
        )
    )
    assert sum(row["accepted"] for row in audit) == 2


def test_source_context_uses_observation_region_not_backend_name():
    context = source_context("renamed_object", _record("spoon", [0, 0, 0], "C1"))
    assert context.source_kind is SourceKind.CUPBOARD
    assert context.source_container == "C1"
    assert context.required_workspace.value == "left_side"
    assert context.container_must_be_open


def test_inventory_is_driven_by_roles_and_plan_without_backend_binding():
    registry = {
        "scene_name": "scene",
        "objects": {
            "renamed_a": _record("cup", [0, 0, 0]),
            "renamed_b": _record("spoon", [1, 0, 0]),
        },
    }
    assignments = {
        "coffee_targets": ["renamed_a"],
        "coffee_stirring": [{"relation_checks": [{"from_object": "renamed_b"}]}],
    }
    plan = [{"step": 1, "action": "pick", "arguments": ["renamed_b"]}]
    inventory = build_phase_b_inventory(registry, assignments, plan)
    assert {row["generic_object_id"] for row in inventory["objects"]} == {"renamed_a", "renamed_b"}
    assert all(not row["backend_binding_present"] for row in inventory["objects"])
    assert not inventory["planner_received_backend_names"]
    assert inventory["evaluation_instance_tokens_excluded"]
    assert all("instance_token" not in row for row in inventory["objects"])


def test_missing_detector_label_uses_functional_role_family_not_backend_name():
    registry = {
        "scene_name": "scene",
        "objects": {"generic_tool": _record(None, [0, 0, 0], "D1")},
    }
    assignments = {
        "soup_serving": [{
            "tool_object_id": "generic_tool",
            "target_object_id": "generic_bowl",
        }],
    }
    inventory = build_phase_b_inventory(
        registry,
        assignments,
        [{"step": 1, "action": "pick", "arguments": ["generic_tool"]}],
    )
    assert inventory["objects"][0]["semantic_label"] == "utensil"
    assert (
        inventory["objects"][0]["semantic_label_source"]
        == "FROZEN_FUNCTIONAL_ROLE_FALLBACK"
    )
    assert inventory["objects"][0]["originating_functional_role"] == "soup_utensil"
    resolved = KitchenExecutionEntityResolver().resolve(
        inventory,
        [ExecutionCandidate("renamed_backend", "spoon", "UTENSIL", "D1", (0, 0, 0))],
    )
    assert resolved["all_resolved"]


def test_detector_semantic_provenance_remains_explicit():
    registry = {
        "scene_name": "scene",
        "objects": {"generic_tool": _record("spoon", [0, 0, 0])},
    }
    inventory = build_phase_b_inventory(
        registry,
        {},
        [{"step": 1, "action": "pick", "arguments": ["generic_tool"]}],
    )
    row = inventory["objects"][0]
    assert row["semantic_label"] == "spoon"
    assert row["semantic_label_source"] == "OBSERVED_SEMANTIC_DETECTOR"
    assert row["originating_functional_role"] is None


def test_resolution_is_deterministic_one_to_one_and_centroid_based():
    inventory = {
        "scene_name": "scene",
        "objects": [
            {
                "generic_object_id": "arbitrary_1", "semantic_label": "cup",
                "selected_functions": ["coffee_vessel"],
                "observed_centroid_world_m": [0.01, 0, 0],
                "source_context": {"observed_source_region": "countertop"},
            },
            {
                "generic_object_id": "arbitrary_2", "semantic_label": "cup",
                "selected_functions": ["coffee_vessel"],
                "observed_centroid_world_m": [0.99, 0, 0],
                "source_context": {"observed_source_region": "countertop"},
            },
        ],
    }
    candidates = [
        ExecutionCandidate("backend_right", "cup", "VESSEL", "countertop", (1, 0, 0)),
        ExecutionCandidate("backend_left", "cup", "VESSEL", "countertop", (0, 0, 0)),
    ]
    resolver = KitchenExecutionEntityResolver()
    first = resolver.resolve(inventory, candidates)
    second = resolver.resolve(inventory, list(reversed(candidates)))
    assert first == second
    assert first["all_resolved"] and first["one_to_one"]
    assert {row["physical_backend_body"] for row in first["accepted"]} == {"backend_left", "backend_right"}


def test_semantic_source_and_distance_gates_fail_closed():
    inventory = {
        "scene_name": "scene",
        "objects": [{
            "generic_object_id": "o", "semantic_label": "bowl",
            "selected_functions": [], "observed_centroid_world_m": [0, 0, 0],
            "source_context": {"observed_source_region": "C1"},
        }],
    }
    candidates = [
        ExecutionCandidate("wrong_semantic", "spoon", "UTENSIL", "C1", (0, 0, 0)),
        ExecutionCandidate("wrong_region", "bowl", "BOWL", "C2", (0, 0, 0)),
        ExecutionCandidate("too_far", "bowl", "BOWL", "C1", (1, 0, 0)),
    ]
    result = KitchenExecutionEntityResolver().resolve(inventory, candidates)
    assert not result["all_resolved"]
    assert result["unresolved_object_ids"] == ["o"]
