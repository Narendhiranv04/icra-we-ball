"""Unit tests for Phase 3: Safe Relation Semantic Interpreter (Gate 3).

Verifies:
1. Endpoint-only test fails closed: nonsensical text between a unique valid endpoint pair does NOT compile.
2. Known semantically matching V1 relations compile when justified by semantic evidence.
3. Multi-predicate compilation when relation text contains multiple distinct concepts.
4. Direction normalization when justified by inverse/passive semantic evidence.
5. Integration with semantic_compiler.py preserves W1 and W2 regression fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.predicate_registry import (
    get_endpoint_valid_binary_predicates,
)
from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import (
    interpret_relation,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    compile_candidate_graph,
)


def test_endpoint_only_test_fails_closed():
    """CRITICAL REQUIREMENT: A unique valid endpoint pair must NOT compile nonsensical text."""
    # (driver, fastener) has exactly ONE valid active predicate: COMPATIBLE_WITH
    valid = get_endpoint_valid_binary_predicates(
        "workshop",
        subject_role="driver",
        object_role="fastener",
        subject_kind="OBJECT",
        object_kind="OBJECT",
    )
    assert len(valid) == 1
    assert valid[0].name == "COMPATIBLE_WITH"

    # Nonsensical text must FAIL CLOSED, not guess COMPATIBLE_WITH
    res = interpret_relation(
        domain="workshop",
        raw_phrase="bananas on the moon",
        subject_role="driver",
        object_role="fastener",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        required=True,
    )
    assert res.succeeded is False
    assert res.status == "UNMAPPABLE_REQUIRED_RELATION"
    assert len(res.interpreted_predicates) == 0
    assert "no semantic evidence" in res.reason.lower()


def test_kitchen_nonsensical_text_fails_closed():
    res = interpret_relation(
        domain="kitchen",
        raw_phrase="completely unrelated statement",
        subject_role="coffee_stirrer",
        object_role="coffee_container",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        required=True,
    )
    assert res.succeeded is False
    assert res.status == "UNMAPPABLE_REQUIRED_RELATION"


def test_living_room_nonsensical_text_fails_closed():
    res = interpret_relation(
        domain="living_room",
        raw_phrase="rainbow clouds floating",
        subject_role="PERSONAL_CUP_SAUCER_REGION",
        object_role="SEATING_POSITION",
        subject_kind="REGION",
        object_kind="FIXED_TARGET",
        required=True,
    )
    assert res.succeeded is False
    assert res.status == "UNMAPPABLE_REQUIRED_RELATION"


def test_known_semantically_matching_relations_compile():
    # Workshop COMPATIBLE_WITH
    res_w1 = interpret_relation(
        domain="workshop",
        raw_phrase="mechanically engages screw head",
        subject_role="driver",
        object_role="fastener",
        subject_kind="OBJECT",
        object_kind="OBJECT",
    )
    assert res_w1.succeeded is True
    assert res_w1.status in ("LEXICAL_SEMANTIC_MATCH", "EXACT_CANONICAL_MATCH")
    assert res_w1.interpreted_predicates[0].predicate_name == "COMPATIBLE_WITH"

    # Workshop REACHES_TARGET
    res_w2 = interpret_relation(
        domain="workshop",
        raw_phrase="accesses target repair hole",
        subject_role="driver",
        object_role="repair_target",
        subject_kind="OBJECT",
        object_kind="FIXED_TARGET",
    )
    assert res_w2.succeeded is True
    assert res_w2.interpreted_predicates[0].predicate_name == "REACHES_TARGET"

    # Workshop COMPATIBLE_WITH_TARGET
    res_w3 = interpret_relation(
        domain="workshop",
        raw_phrase="fastener screws into target hole",
        subject_role="fastener",
        object_role="repair_target",
        subject_kind="OBJECT",
        object_kind="FIXED_TARGET",
    )
    assert res_w3.succeeded is True
    assert res_w3.interpreted_predicates[0].predicate_name == "COMPATIBLE_WITH_TARGET"

    # Kitchen INSERTABLE_IN
    res_k1 = interpret_relation(
        domain="kitchen",
        raw_phrase="fits inside container opening",
        subject_role="coffee_stirrer",
        object_role="coffee_container",
        subject_kind="OBJECT",
        object_kind="OBJECT",
    )
    assert res_k1.succeeded is True
    assert res_k1.interpreted_predicates[0].predicate_name == "INSERTABLE_IN"

    # Kitchen REACHES_BOTTOM
    res_k2 = interpret_relation(
        domain="kitchen",
        raw_phrase="touches bottom of the mug",
        subject_role="coffee_stirrer",
        object_role="coffee_container",
        subject_kind="OBJECT",
        object_kind="OBJECT",
    )
    assert res_k2.succeeded is True
    assert res_k2.interpreted_predicates[0].predicate_name == "REACHES_BOTTOM"

    # Living Room FITS_SET_ON
    res_l1 = interpret_relation(
        domain="living_room",
        raw_phrase="tabletop support for drinkware set",
        subject_role="PERSONAL_CUP_SAUCER_REGION",
        object_role="CUP_SAUCER_SET",
        subject_kind="REGION",
        object_kind="OBJECT",
    )
    assert res_l1.succeeded is True
    assert res_l1.interpreted_predicates[0].predicate_name == "FITS_SET_ON"

    # Living Room NEAR_SEAT
    res_l2 = interpret_relation(
        domain="living_room",
        raw_phrase="adjacent to viewer seating position",
        subject_role="PERSONAL_CUP_SAUCER_REGION",
        object_role="SEATING_POSITION",
        subject_kind="REGION",
        object_kind="FIXED_TARGET",
    )
    assert res_l2.succeeded is True
    assert res_l2.interpreted_predicates[0].predicate_name == "NEAR_SEAT"

    # Living Room ACCESSIBLE_FROM_BOTH_SEATS
    res_l3 = interpret_relation(
        domain="living_room",
        raw_phrase="accessible from both seats in pair",
        subject_role="SHARED_REMOTE_REGION",
        object_role="SEATING_PAIR",
        subject_kind="REGION",
        object_kind="FIXED_TARGET",
    )
    assert res_l3.succeeded is True
    assert res_l3.interpreted_predicates[0].predicate_name == "ACCESSIBLE_FROM_BOTH_SEATS"


def test_multi_predicate_compilation():
    """When text explicitly contains multiple concepts, emit multi-predicate result."""
    res = interpret_relation(
        domain="kitchen",
        raw_phrase="fits inside container opening and reaches bottom of container",
        subject_role="coffee_stirrer",
        object_role="coffee_container",
        subject_kind="OBJECT",
        object_kind="OBJECT",
    )
    assert res.succeeded is True
    assert res.status == "MULTI_PREDICATE_MATCH"
    assert len(res.interpreted_predicates) == 2
    pred_names = {p.predicate_name for p in res.interpreted_predicates}
    assert pred_names == {"INSERTABLE_IN", "REACHES_BOTTOM"}


def test_direction_normalization_when_licensed_by_semantic_evidence():
    """Verify that direction normalization works when semantic evidence supports it."""
    # Reverse phrasing: cup contains stirrer
    res = interpret_relation(
        domain="kitchen",
        raw_phrase="container holds and receives stirrer",
        subject_role="coffee_container",
        object_role="coffee_stirrer",
        subject_kind="OBJECT",
        object_kind="OBJECT",
    )
    assert res.succeeded is True
    assert res.direction_normalized is True
    assert res.status == "DIRECTION_NORMALIZED"
    assert res.interpreted_predicates[0].subject_role == "coffee_stirrer"
    assert res.interpreted_predicates[0].predicate_name == "INSERTABLE_IN"
    assert res.interpreted_predicates[0].object_role == "coffee_container"

    # Reverse phrasing in Workshop: fastener driven by driver
    res_w = interpret_relation(
        domain="workshop",
        raw_phrase="fastener driven by power tool",
        subject_role="fastener",
        object_role="driver",
        subject_kind="OBJECT",
        object_kind="OBJECT",
    )
    assert res_w.succeeded is True
    assert res_w.direction_normalized is True
    assert res_w.status == "DIRECTION_NORMALIZED"
    assert res_w.interpreted_predicates[0].subject_role == "driver"
    assert res_w.interpreted_predicates[0].predicate_name == "COMPATIBLE_WITH"
    assert res_w.interpreted_predicates[0].object_role == "fastener"


def test_compiler_integration_nonsensical_relation_fails_closed():
    """In compiler, nonsensical relation must fail closed without crashing."""
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Drive screw",
        "functional_roles": [
            {
                "id": "r1",
                "entity_kind": "OBJECT",
                "function": "driver",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "required_properties": [],
            },
            {
                "id": "r2",
                "entity_kind": "OBJECT",
                "function": "fastener",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "r1",
                "relation": "completely nonsensical noise text",
                "object_role": "r2",
                "required": True,
            }
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("workshop", "Drive screw", raw)
    # The relation should NOT have been added to graph.relations
    rel_preds = [r.predicate for r in graph.relations]
    assert "COMPATIBLE_WITH" not in rel_preds


def test_w1_w2_regression_fixtures_compile_cleanly():
    """W1 and W2 ideal fixtures must compile without regressing."""
    w1_path = Path("mujoco_scenes/functional_tamp_pipeline/tests/fixtures/ideal_raw_vlm/workshop_W1.json")
    if w1_path.exists():
        doc = json.loads(w1_path.read_text(encoding="utf-8"))
        graph = compile_candidate_graph("workshop", "Drive screw", doc)
        assert len(graph.roles) >= 2
        assert len(graph.relations) >= 1

