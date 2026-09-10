from __future__ import annotations

import json
from pathlib import Path

from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.functional_tamp_pipeline.structural_sanitizer import sanitize_functional_graph


ROOT = Path(__file__).resolve().parents[3]
K1_DIAGNOSTIC = (
    ROOT
    / "benchmark_reports/corrective_recovery_final_32x1_20260909T013814IST"
    / "kitchen/K1/vlm/fm_diagnostics/fm_call_001.json"
)


def _role(role_id: str, function: str) -> dict:
    return {
        "id": role_id,
        "entity_kind": "OBJECT",
        "function": function,
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": [],
        "required_properties": [],
    }


def _doc(roles: list[dict], relations: list[dict], operations: list[dict]) -> dict:
    return {
        "functional_roles": roles,
        "functional_relations": relations,
        "interaction_groups": operations,
        "inspectable_regions": [],
        "inspection_order": [],
    }


def _contains(subject: str, object_: str) -> dict:
    return {"subject_role": subject, "relation": "contains", "object_role": object_}


def _transfer(source: str, target: str) -> dict:
    return {
        "id": "transfer",
        "function": "transfer material into receiving container",
        "tool_role": source,
        "target_role": target,
        "required_target_count": 1,
        "usage_policy": "DEDICATED_PER_TARGET",
        "required_relations": [],
    }


def test_declared_endpoint_state_relation_is_effect_of_explicit_transfer():
    raw = _doc(
        [_role("source", "source of coffee"), _role("carrier", "container for coffee")],
        [_contains("carrier", "source")],
        [_transfer("source", "carrier")],
    )
    graph = compile_candidate_graph("kitchen", "transfer material", raw)

    assert len(graph.operation_groups) == 1
    assert graph.operation_groups[0].function == "POUR"
    assert not graph.relations
    assert len(graph.task_effect_relations) == 1
    effect = graph.task_effect_relations[0]
    assert effect.category == "TASK_EFFECT_SEMANTICS"
    assert effect.object_is_literal is False
    # The effect is attributed to the operation as *compiled*, not to the raw
    # wire group.  V3 participants are unordered and the group they end up in
    # has since been through id normalization, joint typing, slot completion,
    # context elision and composite lowering, so the raw layout is not what the
    # runtime executes; corroborating against it made the attribution depend on
    # the order the model happened to list its participants in.
    assert effect.source_operation_id == graph.operation_groups[0].id
    assert not graph.metadata["canonicalization_trace"]["unresolved_required_relations"]
    corroborated = [
        row for row in graph.metadata["canonicalization_trace"]["relations"]
        if row.get("corroborated_by_compiled_operation")
    ]
    assert corroborated, "the operation that achieves the effect must be recorded"


def test_a_stated_end_state_no_operation_achieves_is_not_an_effect():
    """The adversarial counterpart: an end state needs an operation to achieve it.

    The same containment sentence, with the transfer removed.  Nothing the task
    does brings the state about, so it is not the task's effect -- and it is
    not quietly dropped either: it goes back through the relation path and
    blocks, because the FM did state it.
    """
    raw = _doc(
        [_role("source", "source of coffee"), _role("carrier", "container for coffee")],
        [_contains("carrier", "source")],
        [],
    )
    graph = compile_candidate_graph("kitchen", "the carrier holds the coffee", raw)
    assert not graph.task_effect_relations, "no operation achieves it, so it is no effect"
    trace = graph.metadata["canonicalization_trace"]
    recorded = (trace["unresolved_required_relations"] + trace["relations"]
                + trace.get("constraints_outside_the_runtime_task", []))
    assert any("contains" in str(row) for row in recorded), "it must still be accounted for"


def test_an_end_state_is_not_credited_to_an_unrelated_operation():
    """Corroboration is per operation, not "some operation exists"."""
    raw = _doc(
        [_role("source", "source of coffee"), _role("carrier", "container for coffee"),
         _role("stirrer", "implement used to stir the coffee")],
        [_contains("carrier", "source")],
        [{"id": "stir", "function": "stir", "tool_role": "stirrer",
          "target_role": "carrier", "required_relations": [], "context_relations": []}],
    )
    graph = compile_candidate_graph("kitchen", "stir the coffee", raw)
    achieved = {e.source_operation_id for e in graph.task_effect_relations}
    assert "stir" not in achieved, "stirring does not put the coffee in the cup"


def test_undeclared_non_groundable_content_is_preserved_as_effect_literal():
    raw = _doc(
        [_role("carrier", "container for soup")],
        [_contains("carrier", "content_token")],
        [],
    )
    sanitized = sanitize_functional_graph(raw)
    graph = compile_candidate_graph("kitchen", "container initially holds contents", raw)

    assert sanitized.semantically_incomplete is False
    assert {item["code"] for item in sanitized.repairs} == {"PRESERVED_TASK_EFFECT_LITERAL"}
    assert "content_token" not in graph.nodes
    assert not graph.relations
    assert graph.task_effect_relations[0].object_value == "content_token"
    assert graph.task_effect_relations[0].object_is_literal is True
    assert graph.online_executable_contract_complete is True
    round_trip = type(graph).from_dict(graph.to_dict())
    assert round_trip.task_effect_relations == graph.task_effect_relations


def test_undeclared_operation_participant_still_fails_closed():
    raw = _doc(
        [_role("carrier", "container for coffee")],
        [_contains("carrier", "missing_source")],
        [_transfer("missing_source", "carrier")],
    )
    sanitized = sanitize_functional_graph(raw)
    graph = compile_candidate_graph("kitchen", "transfer missing material", raw)

    assert sanitized.semantically_incomplete is True
    assert {item["code"] for item in sanitized.repairs} == {
        "DANGLING_RELATION_REFERENCE", "INVALID_GROUP_REFERENCE"
    }
    assert not graph.task_effect_relations
    assert not graph.operation_groups
    assert graph.online_executable_contract_complete is False


def test_non_effect_dangling_relation_still_fails_closed():
    raw = _doc(
        [_role("carrier", "container for coffee")],
        [{"subject_role": "carrier", "relation": "compatible with", "object_role": "missing"}],
        [],
    )
    sanitized = sanitize_functional_graph(raw)
    graph = compile_candidate_graph("kitchen", "check compatibility", raw)

    assert sanitized.semantically_incomplete is True
    assert sanitized.repairs[0]["code"] == "DANGLING_RELATION_REFERENCE"
    assert not graph.task_effect_relations
    assert graph.online_executable_contract_complete is False


def test_observed_undeclared_endpoint_is_not_downgraded_to_literal():
    raw = _doc(
        [_role("carrier", "container for soup")],
        [_contains("carrier", "observed_content")],
        [],
    )
    raw["observation_guidance"] = {
        "visible_candidates_per_role": {
            "observed_content": [{"label": "package", "visual_description": "visible package"}]
        }
    }

    sanitized = sanitize_functional_graph(raw)

    assert sanitized.semantically_incomplete is True
    assert sanitized.repairs[0]["code"] == "DANGLING_RELATION_REFERENCE"


def test_state_relation_never_creates_an_operation():
    raw = _doc(
        [_role("source", "source of coffee"), _role("carrier", "container for coffee")],
        [_contains("carrier", "source")],
        [],
    )
    graph = compile_candidate_graph("kitchen", "describe contents", raw)

    assert not graph.operation_groups
    assert not graph.task_effect_relations


def test_archived_k1_contains_edges_are_effects_without_synthesizing_soup():
    raw = json.loads(json.loads(K1_DIAGNOSTIC.read_text())["content"])
    graph = compile_candidate_graph("kitchen", raw["task_summary"], raw)
    trace = graph.metadata["canonicalization_trace"]

    contains_effects = [
        effect for effect in graph.task_effect_relations
        if effect.predicate == "CONTAINS"
    ]
    assert len(contains_effects) == 3
    assert {effect.raw_object for effect in contains_effects} == {
        "coffee", "water_source", "soup"
    }
    assert next(effect for effect in contains_effects if effect.raw_object == "soup").object_is_literal
    assert "soup" not in graph.nodes
    assert not [
        item for item in trace["unresolved_required_relations"]
        if item["raw_phrase"] == "contains"
    ]
    assert not [
        item for item in graph.metadata["structural_sanitizer"]["repairs"]
        if item["code"] == "DANGLING_RELATION_REFERENCE"
    ]
    assert len(graph.operation_groups) == 4
    assert any(
        item.get("status") == "TASK_EXPRESSED_SYSTEM_CONTEXT"
        and item.get("raw_role", {}).get("id") == "serving_location"
        for item in trace["context_only_roles"]
    )
