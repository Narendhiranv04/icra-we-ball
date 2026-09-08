from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from mujoco_scenes.functional_tamp_pipeline.executability import analyze_executability
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph


ROOT = Path(__file__).resolve().parents[3]
K1_DIAGNOSTIC = (
    ROOT
    / "benchmark_reports/corrective_recovery_final_32x1_20260909T013814IST"
    / "kitchen/K1/vlm/fm_diagnostics/fm_call_001.json"
)


def _coffee_role() -> dict:
    return {
        "id": "coffee_cup",
        "entity_kind": "OBJECT",
        "function": "container for prepared coffee",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["cup"],
        "required_properties": [],
    }


def _serving_role(function: str = "shared serving support destination") -> dict:
    return {
        "id": "destination",
        "entity_kind": "REGION",
        "function": function,
        "required_count": 1,
        "binding_policy": "SHARED",
        "candidate_categories": ["surface"],
        "required_properties": [],
    }


def _document(*roles: dict, with_placement: bool = True) -> dict:
    return {
        "functional_roles": list(roles),
        "functional_relations": ([{
            "subject_role": "coffee_cup",
            "relation": "placed on",
            "object_role": "destination",
        }] if with_placement else []),
        "interaction_groups": ([{
            "id": "place_prepared_coffee",
            "function": "place prepared coffee on destination",
            "tool_role": "coffee_cup",
            "target_role": "destination",
            "required_target_count": 1,
            "usage_policy": "DEDICATED_PER_TARGET",
            "required_relations": [],
        }] if with_placement else []),
        "inspectable_regions": [],
        "inspection_order": [],
    }


def test_archived_k1_serving_destination_is_absorbed_without_disabling_placements():
    raw = json.loads(json.loads(K1_DIAGNOSTIC.read_text())["content"])
    graph = compile_candidate_graph("kitchen", raw["task_summary"], raw)
    trace = graph.metadata["canonicalization_trace"]

    assert not [
        item for item in trace["unresolved_roles"]
        if item.get("raw_role", {}).get("id") == "serving_location"
    ]
    assert not [
        item for item in trace["disabled_groups"]
        if item.get("raw_group", {}).get("target_role") == "serving_location"
    ]
    context = next(
        item for item in trace["context_only_roles"]
        if item["raw_role"]["id"] == "serving_location"
    )
    assert context["canonical_role"] == "dining_table"
    assert context["status"] == "TASK_EXPRESSED_SYSTEM_CONTEXT"
    absorbed = [
        item for item in trace["groups"]
        if item.get("raw_group", {}).get("target_role") == "serving_location"
    ]
    assert len(absorbed) == 2
    assert {item["status"] for item in absorbed} == {"ABSORBED_INTO_PLANNER_CONTEXT"}
    assert {item["provenance"] for item in absorbed} == {"TASK_EXPRESSED_SYSTEM_CONTEXT"}
    statuses = analyze_executability(graph)
    assert not any(
        item["status"] == "UNINSTANTIABLE_MISSING_ROLE"
        and item.get("id") in {"serving_location", "4", "5"}
        for item in statuses
    )


def test_explicit_serving_support_and_placement_use_registered_planner_context():
    graph = compile_candidate_graph(
        "kitchen", "prepare coffee", _document(_coffee_role(), _serving_role())
    )
    trace = graph.metadata["canonicalization_trace"]

    assert len(trace["context_only_roles"]) == 1
    context = trace["context_only_roles"][0]
    assert context["raw_role"]["id"] == "destination"
    assert context["canonical_role"] == "dining_table"
    assert context["status"] == "TASK_EXPRESSED_SYSTEM_CONTEXT"
    assert context["rule"] == "TASK_EXPRESSED_SYSTEM_CONTEXT"
    assert trace["groups"][0]["status"] == "ABSORBED_INTO_PLANNER_CONTEXT"
    assert trace["relations"][0]["status"] == "ABSORBED_INTO_PLANNER_CONTEXT"
    assert "destination" not in graph.nodes


def test_missing_serving_destination_is_not_synthesized():
    graph = compile_candidate_graph(
        "kitchen", "serve prepared coffee", _document(_coffee_role(), with_placement=False)
    )
    trace = graph.metadata["canonicalization_trace"]

    assert not trace["context_only_roles"]
    assert not trace["groups"]
    assert "dining_table" not in graph.nodes


def test_unrelated_region_is_not_absorbed_as_serving_destination():
    raw = _document(
        _coffee_role(),
        _serving_role("storage area for inventory inspection"),
    )
    graph = compile_candidate_graph("kitchen", "inspect stored items", deepcopy(raw))
    trace = graph.metadata["canonicalization_trace"]

    assert not any(
        item.get("canonical_role") == "dining_table"
        for item in trace["context_only_roles"]
    )
    assert trace["unresolved_roles"][0]["raw_role"]["id"] == "destination"
    assert trace["disabled_groups"][0]["status"] == "UNINSTANTIABLE_MISSING_ROLE"
