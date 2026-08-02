from copy import deepcopy
import json
from pathlib import Path

import pytest

from mujoco_scenes.task_witness import (
    evaluate_usage_policy_task_witness,
    load_task_requirements,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PRIMARY_TASK = CONFIG_DIR / "ablation2_count_reuse.yaml"
COFFEE_TASK = CONFIG_DIR / "ablation2_coffee_reuse.yaml"
SOUP_TASK = CONFIG_DIR / "ablation2_soup_dedicated.yaml"


def _object(object_id, label, confidence=0.8):
    return {
        "id": f"object:{object_id}",
        "type": "object",
        "attributes": {
            "object_id": object_id,
            "measurement_cloud_path": (
                f"stages/000_initial/evidence/{object_id}/fused.ply"
            ),
            "last_property_source_region": "INITIAL",
            "semantics": {
                "validated": {
                    "status": "SUPPORTED",
                    "canonical_label": label,
                    "mean_confidence": confidence,
                    "semantic_record_path": (
                        f"stages/000_initial/semantics/{object_id}/"
                        "semantic_evidence.json"
                    ),
                }
            },
        },
    }


def _geometry(object_id, role, status="TRUE"):
    return {
        "source": f"object:{object_id}",
        "target": f"role:{role}",
        "relation": "SATISFIES_GEOMETRY",
        "status": status,
        "evidence": {
            "checks": [
                {
                    "name": (
                        "OPEN_CAVITY"
                        if role in {"coffee_cup", "soup_bowl"}
                        else "ELONGATED_OBJECT"
                    ),
                    "status": status,
                }
            ]
        },
    }


def _relation(tool, target, relation, status="TRUE", margin=0.02):
    return {
        "source": f"object:{tool}",
        "target": f"object:{target}",
        "relation": relation,
        "status": status,
        "evidence": {
            "status": status,
            "pass_margin_m": margin,
            "source_measurement_cloud_path": (
                f"stages/000_initial/evidence/{tool}/fused.ply"
            ),
            "target_measurement_cloud_path": (
                f"stages/000_initial/evidence/{target}/fused.ply"
            ),
        },
    }


def _primary_graph(
    *,
    second_fork=False,
    invalid_spoon=False,
    include_initial_fork=False,
):
    labels = {
        "cup_1": "cup",
        "cup_2": "cup",
        "bowl_1": "bowl",
        "bowl_2": "bowl",
        "spoon_1": "spoon",
    }
    if include_initial_fork or second_fork:
        labels["fork_1"] = "fork"
    if invalid_spoon:
        labels["spoon_bad"] = "spoon"
    nodes = [_object(object_id, label) for object_id, label in labels.items()]
    edges = []
    for object_id in labels:
        for role in (
            "coffee_cup",
            "soup_bowl",
            "coffee_stirrer",
            "soup_utensil",
        ):
            if role == "coffee_cup":
                status = "TRUE" if object_id.startswith("cup_") else "FALSE"
            elif role == "soup_bowl":
                status = "TRUE" if object_id.startswith("bowl_") else "FALSE"
            else:
                status = (
                    "TRUE"
                    if object_id.startswith(("spoon_", "fork_"))
                    else "FALSE"
                )
            edges.append(_geometry(object_id, role, status))
    targets = ("cup_1", "cup_2", "bowl_1", "bowl_2")
    for tool in (
        object_id
        for object_id in labels
        if object_id.startswith(("spoon_", "fork_"))
    ):
        for target in targets:
            for relation in ("INSERTABLE_IN", "REACHES_BOTTOM"):
                edges.append(_relation(tool, target, relation))
    if invalid_spoon:
        for target in targets:
            edges.append(
                _relation(
                    "spoon_bad",
                    target,
                    "INSERTABLE_IN",
                    "FALSE",
                    -0.03,
                )
            )
            edges.append(
                _relation("spoon_bad", target, "REACHES_BOTTOM")
            )
    return {"stage": 2 if second_fork else 0, "nodes": nodes, "edges": edges}


def _evaluate(graph, mode="function-aware", task=PRIMARY_TASK):
    return evaluate_usage_policy_task_witness(
        graph,
        task,
        usage_policy_mode=mode,
    )


def _group(result, group_id):
    return next(
        group
        for group in result["function_group_evaluations"]
        if group["function_group_id"] == group_id
    )


def test_existing_ablation1_schema_still_parses_unchanged():
    task = load_task_requirements(CONFIG_DIR / "stir_contents_joint.yaml")
    assert task["_task_schema"] == "JOINT_ROLE_GROUNDING"
    assert "operation_groups" not in task


def test_operation_group_schema_parses_function_scoped_policies():
    task = load_task_requirements(PRIMARY_TASK)
    assert task["_task_schema"] == "JOINT_USAGE_POLICY_GROUNDING"
    assert (
        task["operation_groups"]["coffee_stirring"]["usage_policy"]["mode"]
        == "sequential_reuse_allowed"
    )
    assert (
        task["operation_groups"]["soup_serving"]["usage_policy"]["mode"]
        == "dedicated_per_target"
    )
    assert task["cross_group_reuse"]["allowed"] is True


def test_runtime_task_schema_contains_no_scene_or_region_identifiers():
    serialized = json.dumps(load_task_requirements(PRIMARY_TASK))
    for forbidden in (
        "S1_ablation2",
        '"D1"',
        '"D2"',
        '"C1"',
        '"C2"',
        '"B1"',
        "object_000",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda task: task["operation_groups"]["coffee_stirring"][
                "usage_policy"
            ].update({"mode": "sometimes"}),
            "usage policy",
        ),
        (
            lambda task: task["operation_groups"]["coffee_stirring"].update(
                {"required_target_count": 0}
            ),
            "positive",
        ),
        (
            lambda task: task["operation_groups"]["soup_serving"][
                "usage_policy"
            ].update({"distinct_within_group": False}),
            "contradictory",
        ),
    ],
)
def test_invalid_usage_policy_schema_is_rejected(mutation, match):
    task = load_task_requirements(PRIMARY_TASK)
    task.pop("_task_schema")
    mutation(task)
    with pytest.raises(ValueError, match=match):
        load_task_requirements(task)


def test_one_spoon_reuses_across_two_coffee_targets():
    result = _evaluate(_primary_graph(), task=COFFEE_TASK)
    assert result["status"] == "COMPLETE"
    group = _group(result, "coffee_stirring")
    assert group["counts"]["satisfied_target_slots"] == 2
    assert group["counts"]["distinct_assigned_physical_objects"] == 1
    assert [a["utensil_object_id"] for a in group["selected_assignments"]] == [
        "spoon_1",
        "spoon_1",
    ]
    assert group["selected_assignments"][1]["reused_assignment"] is True


def test_one_fork_can_reuse_across_two_coffee_targets():
    graph = _primary_graph(include_initial_fork=True)
    graph["nodes"] = [
        node
        for node in graph["nodes"]
        if node["id"] != "object:spoon_1"
    ]
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if edge["source"] != "object:spoon_1"
    ]
    result = _evaluate(graph, task=COFFEE_TASK)
    assert result["status"] == "COMPLETE"
    group = _group(result, "coffee_stirring")
    assert [
        assignment["utensil_object_id"]
        for assignment in group["selected_assignments"]
    ] == ["fork_1", "fork_1"]
    assert group["counts"]["distinct_assigned_physical_objects"] == 1


def test_reused_spoon_must_pass_each_target_relation():
    graph = _primary_graph()
    edge = next(
        edge
        for edge in graph["edges"]
        if edge["source"] == "object:spoon_1"
        and edge["target"] == "object:cup_2"
        and edge["relation"] == "INSERTABLE_IN"
    )
    edge["status"] = "FALSE"
    result = _evaluate(graph, task=COFFEE_TASK)
    assert result["status"] == "INCOMPLETE"
    assert _group(result, "coffee_stirring")["counts"][
        "satisfied_target_slots"
    ] == 1


def test_one_spoon_cannot_fill_two_dedicated_soup_slots():
    result = _evaluate(_primary_graph(), task=SOUP_TASK)
    assert result["status"] == "INCOMPLETE"
    group = _group(result, "soup_serving")
    assert group["counts"]["satisfied_target_slots"] == 1
    assert group["counts"]["distinct_assigned_physical_objects"] == 1


def test_spoon_and_fork_fill_two_dedicated_soup_slots():
    result = _evaluate(
        _primary_graph(second_fork=True), task=SOUP_TASK
    )
    assert result["status"] == "COMPLETE"
    assignments = _group(result, "soup_serving")["selected_assignments"]
    assert len({assignment["utensil_object_id"] for assignment in assignments}) == 2
    assert all(assignment["dedicated_assignment"] for assignment in assignments)
    assert [
        (
            assignment["utensil_object_id"],
            assignment["target_object_id"],
        )
        for assignment in assignments
    ] == [("spoon_1", "bowl_1"), ("fork_1", "bowl_2")]


def test_spoon_and_fork_are_eligible_but_invalid_spoon_is_not_counted():
    result = _evaluate(
        _primary_graph(
            invalid_spoon=True,
            include_initial_fork=True,
        )
    )
    group = _group(result, "soup_serving")
    assert group["counts"]["raw_observed_utensils"] == 3
    assert group["counts"]["semantically_eligible_utensils"] == 3
    assert group["counts"]["geometrically_eligible_utensils"] == 3
    assert group["counts"]["functionally_assignable_utensils"] == 2
    assert "fork_1" in group["semantically_eligible_object_ids"]
    assert "fork_1" in group["functionally_assignable_object_ids"]
    assert "spoon_bad" not in group["functionally_assignable_object_ids"]


def test_function_aware_cross_group_reuse_derives_two_physical_spoons():
    result = _evaluate(_primary_graph(second_fork=True))
    assert result["status"] == "COMPLETE"
    assert result["distinct_physical_tool_count"] == 2
    coffee_ids = {
        assignment["utensil_object_id"]
        for assignment in _group(result, "coffee_stirring")[
            "selected_assignments"
        ]
    }
    soup_ids = {
        assignment["utensil_object_id"]
        for assignment in _group(result, "soup_serving")[
            "selected_assignments"
        ]
    }
    assert len(coffee_ids) == 1
    assert len(soup_ids) == 2
    assert coffee_ids <= soup_ids
    assert result["policy_required_distinct_physical_tool_count"] == 2
    assert all(
        assignment["cross_group_reused_assignment"]
        for assignment in result["operation_assignments"]
        if assignment["utensil_object_id"] in coffee_ids
    )


def test_disallowing_cross_group_reuse_requires_an_additional_tool():
    task = load_task_requirements(PRIMARY_TASK)
    task.pop("_task_schema")
    task["cross_group_reuse"]["allowed"] = False
    result = _evaluate(
        _primary_graph(second_fork=True),
        task=load_task_requirements(task),
    )
    assert result["status"] == "INCOMPLETE"
    assert result["policy_required_distinct_physical_tool_count"] == 3


def test_always_reusable_is_diagnostic_initial_false_positive():
    result = _evaluate(_primary_graph(), mode="always-reusable")
    assert result["status"] == "COMPLETE"
    assert result["diagnostic_ablation"] is True
    assert result["distinct_physical_tool_count"] == 1
    assert result["satisfied_target_slot_count"] == 4
    assert result["policy_required_distinct_physical_tool_count"] == 1


def test_always_distinct_is_false_negative_even_with_two_valid_utensils():
    result = _evaluate(
        _primary_graph(second_fork=True), mode="always-distinct"
    )
    assert result["status"] == "INCOMPLETE"
    assert "GLOBAL_DISTINCTNESS_BLOCKS_ASSIGNMENT" in result["reason_codes"]
    assert result["policy_required_distinct_physical_tool_count"] == 4


def test_assignment_provenance_uses_measurement_evidence_paths():
    result = _evaluate(_primary_graph(second_fork=True))
    for assignment in result["operation_assignments"]:
        assert "/evidence/" in assignment["geometry_evidence_path"]
        assert "/evidence/" in assignment["target_geometry_evidence_path"]
        assert "/semantics/" in assignment["semantic_evidence_path"]
        assert "evaluation_stage" in assignment
        assert "source_stage" in assignment
        assert "source_region" in assignment
        for check in assignment["relation_checks"]:
            evidence = check["evidence"]
            assert "/evidence/" in evidence[
                "source_measurement_cloud_path"
            ]
            assert "/evidence/" in evidence[
                "target_measurement_cloud_path"
            ]


def test_unknown_relation_never_creates_candidate_target_edge():
    graph = _primary_graph(second_fork=True)
    for edge in graph["edges"]:
        if (
            edge["source"] == "object:fork_1"
            and edge["target"] in {
                "object:bowl_1",
                "object:bowl_2",
            }
            and edge["relation"] == "REACHES_BOTTOM"
        ):
            edge["status"] = "UNKNOWN"
    result = _evaluate(graph, task=SOUP_TASK)
    assert result["status"] == "INDETERMINATE"
    assert all(
        not (
            assignment["utensil_object_id"] == "fork_1"
            and assignment["target_object_id"] == "bowl_2"
        )
        for assignment in result["operation_assignments"]
    )


def test_extra_unknown_target_does_not_override_validated_target_set():
    graph = _primary_graph()
    unknown = _object("unknown_distractor", "cup")
    unknown["attributes"]["semantics"]["validated"].update(
        {
            "status": "UNKNOWN",
            "canonical_label": None,
            "mean_confidence": None,
        }
    )
    graph["nodes"].append(unknown)
    graph["edges"].extend(
        [
            _geometry("unknown_distractor", "coffee_cup", "TRUE"),
            _geometry("unknown_distractor", "soup_bowl", "TRUE"),
            _geometry("unknown_distractor", "coffee_stirrer", "FALSE"),
            _geometry("unknown_distractor", "soup_utensil", "FALSE"),
        ]
    )
    result = _evaluate(graph, task=SOUP_TASK)
    assert result["status"] == "INCOMPLETE"
    assert _group(result, "soup_serving")["unknown_target_object_ids"] == []


def test_same_instance_is_never_counted_twice_as_distinct():
    graph = _primary_graph()
    graph["nodes"].append(deepcopy(next(
        node for node in graph["nodes"] if node["id"] == "object:spoon_1"
    )))
    result = _evaluate(graph, task=SOUP_TASK)
    assert result["status"] == "INCOMPLETE"
    assert _group(result, "soup_serving")["counts"][
        "distinct_assigned_physical_objects"
    ] == 1
