import json

from mujoco_scenes.kitchen_execution_entities import (
    ExecutionCandidate,
    KitchenExecutionEntityResolver,
    SourceKind,
    build_phase_b_inventory,
    source_context,
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
    resolved = KitchenExecutionEntityResolver().resolve(
        inventory,
        [ExecutionCandidate("renamed_backend", "spoon", "UTENSIL", "D1", (0, 0, 0))],
    )
    assert resolved["all_resolved"]


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
