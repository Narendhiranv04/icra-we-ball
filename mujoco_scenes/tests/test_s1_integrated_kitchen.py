from pathlib import Path

from mujoco_scenes.scene_loader import load_all_configs
from mujoco_scenes.task_witness import (
    evaluate_usage_policy_task_witness,
    load_task_requirements,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
TASK_PATH = CONFIG_DIR / "s1_integrated_kitchen_object_function.yaml"
COFFEE = ("coffee_1", "coffee_2", "coffee_3")
SOUP = ("soup_1", "soup_2", "soup_3")
TARGETS = COFFEE + SOUP


def _node(object_id, label):
    return {
        "id": f"object:{object_id}",
        "type": "object",
        "attributes": {
            "object_id": object_id,
            "measurement_cloud_path": (
                f"stages/000_initial/evidence/{object_id}/fused.ply"
            ),
            "semantics": {
                "validated": {
                    "status": "SUPPORTED",
                    "canonical_label": label,
                    "mean_confidence": 0.8,
                    "semantic_record_path": (
                        f"stages/000_initial/semantics/{object_id}/"
                        "semantic_evidence.json"
                    ),
                }
            },
        },
    }


def _unary(object_id, role, status):
    predicate = (
        "OPEN_CAVITY" if role.endswith("container")
        else "ELONGATED_OBJECT"
    )
    return {
        "source": f"object:{object_id}",
        "target": f"role:{role}",
        "relation": "SATISFIES_GEOMETRY",
        "status": status,
        "evidence": {"checks": [{"name": predicate, "status": status}]},
    }


def _relation(tool, target, predicate, status):
    evidence = {
        "status": status,
        "pass_margin_m": 0.02 if status == "TRUE" else -0.02,
        "source_measurement_cloud_path": (
            f"stages/000_initial/evidence/{tool}/fused.ply"
        ),
        "target_measurement_cloud_path": (
            f"stages/000_initial/evidence/{target}/fused.ply"
        ),
    }
    if predicate == "INSERTABLE_IN":
        evidence.update(
            maximum_cross_section_m=0.02,
            clearance_margin_m=0.005,
            opening_width_m=0.045,
        )
    else:
        evidence.update(
            usable_length_m=0.18,
            grip_allowance_m=0.03,
            cavity_depth_m=0.10,
        )
    return {
        "source": f"object:{tool}",
        "target": f"object:{target}",
        "relation": predicate,
        "status": status,
        "evidence": evidence,
    }


COMPATIBILITY = {
    "short": {"coffee_3", "soup_1"},
    "medium": {"coffee_2", "coffee_3", "soup_1", "soup_2"},
    "wide": {"soup_1", "soup_2"},
    "fork": set(TARGETS),
    "marker": set(TARGETS),
    "oversized": set(),
    "partial": {"coffee_2", "coffee_3", "soup_1", "soup_2"},
    "soup_long": {"coffee_3", "soup_1", "soup_2", "soup_3"},
    "near_miss": {"coffee_2", "coffee_3", *SOUP},
    "final": set(TARGETS),
}


def _graph(tools):
    labels = {
        **{target: ("mug" if target == "coffee_2" else "cup") for target in COFFEE},
        **{target: "bowl" for target in SOUP},
        **{tool: ("fork" if tool == "fork" else "marker" if tool == "marker" else "spoon") for tool in tools},
    }
    nodes = [_node(object_id, label) for object_id, label in labels.items()]
    edges = []
    roles = (
        "coffee_container", "soup_container",
        "coffee_stirrer", "soup_eating_utensil",
    )
    for object_id in labels:
        for role in roles:
            is_target_role = role.endswith("container")
            status = "TRUE" if (object_id in TARGETS) == is_target_role else "FALSE"
            edges.append(_unary(object_id, role, status))
    for tool in tools:
        for target in TARGETS:
            status = "TRUE" if target in COMPATIBILITY[tool] else "FALSE"
            edges.extend(
                _relation(tool, target, predicate, status)
                for predicate in ("INSERTABLE_IN", "REACHES_BOTTOM")
            )
    return {
        "stage": 0,
        "pairing": {"strategy": "exhaustive_all_pairs"},
        "nodes": nodes,
        "edges": edges,
    }


def _evaluate(tools, mode="joint-target-specific"):
    return evaluate_usage_policy_task_witness(
        _graph(tools), TASK_PATH, target_assignment_mode=mode
    )


def _groups(result):
    return {
        item["function_group_id"]: item
        for item in result["function_group_evaluations"]
    }


def test_integrated_scene_family_has_three_visible_and_three_stored_targets():
    configs = load_all_configs()
    names = {
        "S1_integrated_kitchen_object_function_primary",
        "S1_integrated_kitchen_object_function_initial_complete",
        "S1_integrated_kitchen_object_function_exhaustion",
    }
    assert names <= configs.keys()
    primary = configs["S1_integrated_kitchen_object_function_primary"]
    assert len(primary.countertop_objects) == 10
    assert sum(map(len, primary.container_contents.values())) == 10
    assert "s1i_final_long_narrow_spoon" not in (
        primary.countertop_objects.values()
    )
    assert "s1i_final_long_narrow_spoon" in primary.container_contents["C1"]
    assert "marker" in primary.countertop_objects.values()

    initial_complete = configs[
        "S1_integrated_kitchen_object_function_initial_complete"
    ]
    assert list(initial_complete.countertop_objects.values()).count(
        "s1i_final_long_narrow_spoon"
    ) == 3

    exhaustion = configs[
        "S1_integrated_kitchen_object_function_exhaustion"
    ]
    assert "s1i_final_long_narrow_spoon" not in {
        *exhaustion.countertop_objects.values(),
        *(
            item
            for contents in exhaustion.container_contents.values()
            for item in contents
        ),
    }


def test_integrated_manual_specification_has_function_scoped_usage():
    task = load_task_requirements(TASK_PATH)
    assert task["goal_instruction"] == (
        "Prepare and serve coffee and soup for two people using the "
        "available kitchenware. Stir both coffees and provide each "
        "soup bowl with a suitable utensil."
    )
    assert task["roles"]["coffee_container"]["count"] == 2
    assert task["roles"]["soup_container"]["count"] == 2
    coffee = task["operation_groups"]["coffee_stirring"]
    soup = task["operation_groups"]["soup_serving"]
    assert not coffee["usage_policy"]["same_tool_must_cover_all_targets"]
    assert coffee["usage_policy"]["selection_preference"] == "minimize_distinct_tools"
    assert soup["usage_policy"]["distinct_within_group"]
    assert not task["cross_group_reuse"]["allowed"]
    soup_labels = {
        preference["canonical_label"]
        for preference in task["roles"]["soup_eating_utensil"][
            "semantic_preferences"
        ]
    }
    assert soup_labels == {"spoon"}


def test_primary_progression_requires_c1_all_target_spoon():
    initial = ["short", "medium", "wide", "fork", "marker"]
    checkpoints = [
        (initial, "INCOMPLETE", "COMPLETE"),
        (initial + ["oversized"], "INCOMPLETE", "COMPLETE"),
        (initial + ["oversized", "partial"], "COMPLETE", "COMPLETE"),
        (initial + ["oversized", "partial", "soup_long"], "COMPLETE", "COMPLETE"),
        (initial + ["oversized", "partial", "soup_long", "near_miss"], "COMPLETE", "COMPLETE"),
    ]
    for tools, expected_global, expected_soup in checkpoints:
        result = _evaluate(tools)
        assert result["status"] == expected_global
        assert _groups(result)["soup_serving"]["status"] == expected_soup
    complete = _evaluate(checkpoints[-1][0] + ["final"])
    assert complete["status"] == "COMPLETE"
    assert _groups(complete)["coffee_stirring"]["status"] == "COMPLETE"
    assert complete["distinct_physical_tool_count"] == 4
    coffee_tools = {
        item["utensil_object_id"]
        for item in complete["operation_assignments"]
        if item["function_group_id"] == "coffee_stirring"
    }
    soup_tools = {
        item["utensil_object_id"]
        for item in complete["operation_assignments"]
        if item["function_group_id"] == "soup_serving"
    }
    assert coffee_tools == {"final", "medium"}
    assert len(soup_tools) == 2
    assert "final" not in soup_tools


def test_partial_count_and_semantics_cannot_control_production():
    tools = ["short", "medium", "wide", "fork", "marker"]
    assert _evaluate(tools, "joint-target-specific")["status"] == "INCOMPLETE"
    assert _evaluate(tools, "semantic-only")["status"] == "INCOMPLETE"
    assert _evaluate(tools, "geometry-only")["status"] == "COMPLETE"
    assert _evaluate(tools, "joint-target-agnostic-count")["status"] == "INCOMPLETE"


def test_two_serving_roster_completes_without_third_target():
    result = _evaluate(
        ["short", "medium", "wide", "fork", "marker", "oversized", "partial", "soup_long", "near_miss"]
    )
    assert result["status"] == "COMPLETE"
    assert _groups(result)["coffee_stirring"]["status"] == "COMPLETE"
