from copy import deepcopy
from pathlib import Path

import pytest

from mujoco_scenes.task_witness import (
    build_target_compatibility_matrix,
    evaluate_usage_policy_task_witness,
    load_task_requirements,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
TASK = CONFIG_DIR / "ablation3_multi_target.yaml"
MODES = (
    "semantic-only",
    "geometry-only",
    "joint-target-agnostic-count",
    "joint-target-specific",
)
TARGETS = ("cup", "mug", "bowl_shallow", "bowl_deep")


def _object(object_id, label):
    return {
        "id": f"object:{object_id}",
        "type": "object",
        "attributes": {
            "object_id": object_id,
            "measurement_cloud_path": (
                f"stages/000_initial/evidence/{object_id}/fused.ply"
            ),
            "last_property_update_stage": 0,
            "last_property_source_region": "INITIAL",
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


def _geometry(object_id, role, status):
    predicate = (
        "OPEN_CAVITY"
        if role in {"coffee_container", "soup_container"}
        else "ELONGATED_OBJECT"
    )
    return {
        "source": f"object:{object_id}",
        "target": f"role:{role}",
        "relation": "SATISFIES_GEOMETRY",
        "status": status,
        "evidence": {
            "checks": [{"name": predicate, "status": status}]
        },
    }


def _relation(tool, target, relation, status):
    is_insertion = relation == "INSERTABLE_IN"
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
    if is_insertion:
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
        "relation": relation,
        "status": status,
        "evidence": evidence,
    }


def _graph(
    *,
    include_partial=False,
    include_long=False,
    unknown=False,
    pairing_strategy="exhaustive_all_pairs",
):
    labels = {
        "cup": "cup",
        "mug": "mug",
        "bowl_shallow": "bowl",
        "bowl_deep": "bowl",
        "short": "spoon",
        "medium": "spoon",
        "wide": "spoon",
        "fork": "fork",
    }
    if include_partial:
        labels["partial"] = "spoon"
    if include_long:
        labels["long"] = "spoon"
    nodes = [_object(object_id, label) for object_id, label in labels.items()]
    edges = []
    for object_id in labels:
        for role in (
            "coffee_container",
            "soup_container",
            "coffee_stirrer",
            "soup_spoon",
        ):
            if role in {"coffee_container", "soup_container"}:
                status = "TRUE" if object_id in TARGETS else "FALSE"
            else:
                status = "TRUE" if object_id not in TARGETS else "FALSE"
            edges.append(_geometry(object_id, role, status))
    compatibility = {
        "short": {"bowl_shallow"},
        "medium": {"mug", "bowl_shallow"},
        "wide": {"bowl_shallow", "bowl_deep"},
        "fork": set(TARGETS),
        "partial": {"mug", "bowl_shallow"},
        "long": set(TARGETS),
    }
    for tool in (object_id for object_id in labels if object_id not in TARGETS):
        for target in TARGETS:
            status = "TRUE" if target in compatibility[tool] else "FALSE"
            if unknown and tool == "medium" and target == "mug":
                status = "UNKNOWN"
            for relation in ("INSERTABLE_IN", "REACHES_BOTTOM"):
                edges.append(_relation(tool, target, relation, status))
    return {
        "stage": 2 if include_long else 1 if include_partial else 0,
        "pairing": {"strategy": pairing_strategy},
        "nodes": nodes,
        "edges": edges,
    }


def _evaluate(graph, mode="joint-target-specific"):
    return evaluate_usage_policy_task_witness(
        graph,
        TASK,
        target_assignment_mode=mode,
    )


def _group(result, group_id):
    return next(
        group
        for group in result["function_group_evaluations"]
        if group["function_group_id"] == group_id
    )


def test_ablation3_schema_declares_function_level_all_target_reuse():
    task = load_task_requirements(TASK)
    policy = task["operation_groups"]["coffee_stirring"]["usage_policy"]
    assert task["target_assignment_ablation"] is True
    assert policy["same_tool_must_cover_all_targets"] is True
    assert task["operation_groups"]["soup_serving"]["usage_policy"][
        "distinct_within_group"
    ] is True


def test_invalid_same_tool_policy_is_rejected():
    task = load_task_requirements(TASK)
    task["operation_groups"]["soup_serving"]["usage_policy"][
        "same_tool_must_cover_all_targets"
    ] = True
    with pytest.raises(ValueError, match="only valid"):
        load_task_requirements(task)


def test_invalid_target_count_is_rejected():
    task = load_task_requirements(TASK)
    task["operation_groups"]["coffee_stirring"][
        "required_target_count"
    ] = 0
    with pytest.raises(ValueError, match="must be positive"):
        load_task_requirements(task)


def test_malformed_relation_direction_is_rejected():
    task = load_task_requirements(TASK)
    task["operation_groups"]["coffee_stirring"]["tool_role"] = (
        "coffee_container"
    )
    task["operation_groups"]["coffee_stirring"]["target_role"] = (
        "coffee_stirrer"
    )
    task["operation_groups"]["coffee_stirring"][
        "required_target_count"
    ] = 1
    with pytest.raises(ValueError, match="declared directionally"):
        load_task_requirements(task)


def test_two_partial_spoons_do_not_fake_single_reusable_tool():
    result = _evaluate(_graph(include_partial=True))
    assert result["status"] == "INCOMPLETE"
    coffee = _group(result, "coffee_stirring")
    assert coffee["status"] == "INCOMPLETE"
    assert coffee["reason"] == "INSUFFICIENT_VALID_ASSIGNMENTS"


def test_one_tool_covering_both_targets_satisfies_reusable_group():
    result = _evaluate(_graph(include_long=True))
    assert result["status"] == "COMPLETE"
    coffee = _group(result, "coffee_stirring")
    assignments = coffee["selected_assignments"]
    assert len(assignments) == 2
    assert {item["utensil_object_id"] for item in assignments} == {"long"}
    assert {item["target_object_id"] for item in assignments} == {
        "cup",
        "mug",
    }


def test_dedicated_matching_uses_distinct_persistent_ids():
    result = _evaluate(_graph())
    soup = _group(result, "soup_serving")
    assert soup["status"] == "COMPLETE"
    assignments = soup["selected_assignments"]
    assert len({item["utensil_object_id"] for item in assignments}) == 2


def test_unknown_relation_never_becomes_valid_compatibility_edge():
    result = _evaluate(_graph(unknown=True))
    coffee = _group(result, "coffee_stirring")
    edge = next(
        edge
        for edge in coffee["candidate_target_evaluations"]
        if edge["utensil_object_id"] == "medium"
        and edge["target_object_id"] == "mug"
    )
    assert edge["status"] == "UNKNOWN"


def test_complete_compatibility_matrix_preserves_margins_and_paths():
    witness = _evaluate(_graph(include_long=True))
    matrix = build_target_compatibility_matrix(witness)
    assert matrix["same_observation_evidence"] is True
    assert set(matrix["tool_object_ids"]) == {
        "short", "medium", "wide", "fork", "long"
    }
    cell = next(
        cell
        for cell in matrix["cells"]
        if cell["function_group_id"] == "coffee_stirring"
        and cell["tool_object_id"] == "long"
        and cell["target_object_id"] == "cup"
    )
    assert cell["insertable_in_pass_margin_m"] == pytest.approx(0.02)
    assert cell["reaches_bottom_pass_margin_m"] == pytest.approx(0.02)
    assert "/evidence/long/fused.ply" in cell[
        "tool_geometry_evidence_path"
    ]


def test_every_observed_object_is_checked_before_role_binding():
    graph = _graph(include_long=True)
    witness = _evaluate(graph)
    object_ids = {
        node["attributes"]["object_id"]
        for node in graph["nodes"]
    }
    role_count = 4
    group_count = 2
    assert len(witness["candidate_evaluations"]) == (
        len(object_ids) * role_count
    )
    assert witness["selection_policy"]["pairing_scope"] == (
        "ALL_OBSERVED_ORDERED_OBJECT_PAIRS"
    )
    assert witness["selection_policy"]["role_binding_phase"] == (
        "AFTER_PAIRWISE_GEOMETRY"
    )
    matrix = build_target_compatibility_matrix(witness)
    assert set(matrix["observed_object_ids"]) == object_ids
    assert matrix["ordered_distinct_object_pair_count"] == (
        len(object_ids) * (len(object_ids) - 1)
    )
    assert matrix["cell_count"] == (
        group_count * len(object_ids) * (len(object_ids) - 1)
    )
    for group in witness["function_group_evaluations"]:
        pairs = {
            (edge["utensil_object_id"], edge["target_object_id"])
            for edge in group["candidate_target_evaluations"]
        }
        assert pairs == {
            (source, target)
            for source in object_ids
            for target in object_ids
            if source != target
        }


def test_semantic_first_pairing_keeps_all_unary_checks_but_prunes_pairs():
    graph = _graph(
        include_long=True,
        pairing_strategy="semantic_role_scoped",
    )
    witness = _evaluate(graph)
    object_ids = {
        node["attributes"]["object_id"] for node in graph["nodes"]
    }
    assert len(witness["candidate_evaluations"]) == len(object_ids) * 4
    assert witness["selection_policy"]["unary_geometry_scope"] == (
        "ALL_OBSERVED_OBJECTS"
    )
    assert witness["selection_policy"]["pairing_scope"] == (
        "SEMANTIC_ROLE_SCOPED_OBJECT_PAIRS"
    )
    matrix = build_target_compatibility_matrix(witness)
    # Fork is semantically excluded and containers cannot be tool subjects.
    assert not any(
        cell["tool_object_id"] == "fork" for cell in matrix["cells"]
    )
    assert all(
        cell["tool_semantic_status"] == "TRUE"
        and cell["target_semantic_status"] == "TRUE"
        for cell in matrix["cells"]
    )
    assert matrix["function_pair_evaluation_count"] < (
        2 * len(object_ids) * (len(object_ids) - 1)
    )
    assert matrix["pruned_function_pair_count"] > 0


def test_relation_schema_allows_same_role_pairing_for_future_nesting():
    task = load_task_requirements(TASK)
    task["relations"].append(
        {
            "predicate": "NESTABLE_IN",
            "subject_role": "soup_container",
            "object_role": "soup_container",
            "expected": True,
        }
    )
    normalized = load_task_requirements(task)
    relation = normalized["constraints"]["pairwise"][-1]
    assert relation["from_role"] == "soup_container"
    assert relation["to_role"] == "soup_container"


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("semantic-only", "COMPLETE"),
        ("geometry-only", "COMPLETE"),
        ("joint-target-agnostic-count", "COMPLETE"),
        ("joint-target-specific", "INCOMPLETE"),
    ],
)
def test_required_same_evidence_ablation_outcomes(mode, expected):
    result = _evaluate(_graph(), mode)
    assert result["status"] == expected
    assert result["target_assignment_mode"] == mode


def test_geometry_only_evaluates_semantically_rejected_objects_by_geometry():
    result = _evaluate(_graph(), "geometry-only")
    assert result["status"] == "COMPLETE"
    fork_tool_evaluations = [
        item
        for item in result["candidate_evaluations"]
        if item["object_id"] == "fork"
        and item["role"] in {"coffee_stirrer", "soup_spoon"}
    ]
    assert fork_tool_evaluations
    assert all(
        item["semantic_gate_status"] == "FALSE"
        and item["geometry_gate_status"] == "TRUE"
        and item["status"] == "TRUE"
        for item in fork_tool_evaluations
    )


def test_geometry_only_does_not_use_semantics_to_preassign_targets():
    result = _evaluate(_graph(), "geometry-only")
    coffee = _group(result, "coffee_stirring")
    assert set(coffee["eligible_target_object_ids"]) == set(TARGETS)
    assert result["selection_policy"]["candidate_gate"] == (
        "geometry_after_pairing"
    )


def test_target_order_and_object_order_do_not_change_assignment():
    graph = _graph(include_long=True)
    first = _evaluate(graph)
    permuted = deepcopy(graph)
    permuted["nodes"].reverse()
    permuted["edges"].reverse()
    second = _evaluate(permuted)
    assert second["selected_witness"] == first["selected_witness"]
    assert second["operation_assignments"] == first["operation_assignments"]
