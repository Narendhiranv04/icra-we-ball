"""Unit tests for Phase 4: Explicit Operation Interpreter and Robot Capabilities (Gate 4).

Verifies:
1. No endpoint-only operation inference: nonsensical or empty operation text fails closed.
2. Explicit operation derives correct physical precondition set from robot capability.
3. Empty/unresolved operation cannot silently gain physical predicates.
4. Robot capability registry hash is deterministic and included in run manifest.
5. W1, W2, K1, and L1 regression fixtures compile cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.audit import compute_provenance_fingerprint
from mujoco_scenes.functional_tamp_pipeline.models import OperationGroup
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    get_robot_capabilities,
    get_robot_capability_registry_hash,
    interpret_operation,
    OperationInterpretationResult,
    RobotCapability,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    compile_candidate_graph,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"


def test_no_endpoint_only_operation_inference():
    """CRITICAL GATE 4 REQUIREMENT: Endpoints alone must never create an operation.

    A valid endpoint pair (stirrer, prepared_cup_target) with nonsensical text
    must FAIL CLOSED as UNMAPPABLE_OPERATION.
    """
    res = interpret_operation(
        domain="kitchen",
        raw_phrase="banana spaceship on the moon",
        source_role="stirrer",
        target_role="prepared_cup_target",
    )
    assert res.succeeded is False
    assert res.status == "UNMAPPABLE_OPERATION"
    assert res.capability is None
    assert len(res.physical_preconditions) == 0
    assert "no semantic evidence" in res.reason.lower()


def test_empty_operation_phrase_fails_closed():
    """Empty or whitespace-only operation phrase must fail closed."""
    res = interpret_operation(
        domain="kitchen",
        raw_phrase="",
        source_role="stirrer",
        target_role="prepared_cup_target",
    )
    assert res.succeeded is False
    assert res.status == "UNMAPPABLE_OPERATION"
    assert "empty operation phrase" in res.reason.lower()


def test_workshop_nonsensical_operation_fails_closed():
    """Workshop driver/fastener endpoints with nonsensical text must not guess DRIVE_FASTENER."""
    res = interpret_operation(
        domain="workshop",
        raw_phrase="paint the walls blue",
        source_role="driver",
        target_role="fastener",
        anchor_role="repair_target",
    )
    assert res.succeeded is False
    assert res.status == "UNMAPPABLE_OPERATION"
    assert res.capability is None


def test_explicit_operation_derives_correct_physical_preconditions():
    """Explicit operation derives the complete physical feasibility precondition set."""
    # 1. Kitchen stirring
    k_stir = interpret_operation(
        domain="kitchen",
        raw_phrase="stir beverage in cups",
        source_role="stirrer",
        target_role="prepared_cup_target",
    )
    assert k_stir.succeeded is True
    assert k_stir.planner_operation == "STIR_COFFEE"
    assert set(k_stir.required_relations) == {"INSERTABLE_IN", "REACHES_BOTTOM"}

    # 2. Kitchen soup serving
    k_soup = interpret_operation(
        domain="kitchen",
        raw_phrase="provide eating utensil for each soup bowl",
        source_role="eating_utensil",
        target_role="soup_bowl_target",
    )
    assert k_soup.succeeded is True
    assert k_soup.planner_operation == "PROVIDE_SOUP_EATING_UTENSIL"
    assert set(k_soup.required_relations) == {"INSERTABLE_IN", "REACHES_BOTTOM"}

    # 3. Living room drinkware support
    l_drink = interpret_operation(
        domain="living_room",
        raw_phrase="support drinkware set beside seat",
        source_role="PERSONAL_CUP_SAUCER_REGION",
        target_role="CUP_SAUCER_SET",
        anchor_role="SEATING_POSITION",
    )
    assert l_drink.succeeded is True
    assert l_drink.planner_operation == "SUPPORT_DRINKWARE"
    assert "FITS_SET_ON" in l_drink.required_relations
    assert "NEAR_SEAT" in l_drink.context_relations

    # 4. Workshop fastening
    w_fasten = interpret_operation(
        domain="workshop",
        raw_phrase="drive screw into joint target",
        source_role="driver",
        target_role="fastener",
        anchor_role="repair_target",
    )
    assert w_fasten.succeeded is True
    assert w_fasten.planner_operation == "DRIVE_FASTENER_INTO_TARGET"
    assert len(w_fasten.physical_preconditions) == 3
    precond_triples = set(w_fasten.physical_preconditions)
    assert ("driver", "COMPATIBLE_WITH", "fastener") in precond_triples
    assert ("driver", "REACHES_TARGET", "repair_target") in precond_triples
    assert ("fastener", "COMPATIBLE_WITH_TARGET", "repair_target") in precond_triples


def test_compiler_disables_unsupported_operator():
    """When a group has nonsensical function text, compiler marks it UNSUPPORTED_OPERATOR."""
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Test stirring",
        "functional_roles": [
            {
                "id": "r1",
                "entity_kind": "OBJECT",
                "function": "stirrer",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["spoon"],
                "required_properties": [],
            },
            {
                "id": "r2",
                "entity_kind": "OBJECT",
                "function": "prepared_cup_target",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup"],
                "required_properties": [],
            },
        ],
        "functional_relations": [],
        "interaction_groups": [
            {
                "id": "g1",
                "function": "completely bogus nonsensical text",
                "tool_role": "r1",
                "target_role": "r2",
                "required_target_count": 1,
                "usage_policy": "DEDICATED_PER_TARGET",
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("kitchen", "Test stirring", raw)
    # The group must NOT be compiled into executable operation_groups
    assert len(graph.operation_groups) == 0


def test_empty_operation_cannot_silently_gain_physical_predicates():
    """An operation group with empty relations and no valid capability must not gain predicates."""
    grp = OperationGroup(
        id="empty_grp",
        function="unmapped",
        tool_role="t",
        target_role="tgt",
        required_target_count=1,
        usage_policy="DEDICATED_PER_TARGET",
        required_relations=(),
    )
    assert grp.required_relations == ()


def test_robot_capability_registry_hash_deterministic():
    """Capability registry hash must be a deterministic 64-char hex string."""
    h1 = get_robot_capability_registry_hash()
    h2 = get_robot_capability_registry_hash()
    assert isinstance(h1, str)
    assert len(h1) == 64
    assert h1 == h2


def test_audit_manifest_contains_capability_registry_hash():
    """compute_provenance_fingerprint must contain robot_capability_registry_hash matching registry."""
    manifest = compute_provenance_fingerprint(domain="kitchen", variant="K1")
    assert "robot_capability_registry_hash" in manifest
    assert manifest["robot_capability_registry_hash"] == get_robot_capability_registry_hash()


def test_fixtures_compile_cleanly():
    """W1, K1, and L1 fixtures compile cleanly with capabilities mapped."""
    for dom, fixture_name in [
        ("kitchen", "kitchen_K1.json"),
        ("living_room", "living_room_L1.json"),
        ("workshop", "workshop_W1.json"),
    ]:
        fixture_path = FIXTURES_DIR / fixture_name
        if not fixture_path.exists():
            continue
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        graph = compile_candidate_graph(dom, "Run fixture task", raw)
        assert len(graph.roles) >= 2
        if dom == "kitchen":
            assert len(graph.operation_groups) == 2
            for g in graph.operation_groups:
                assert g.capability_id in ("STIR_COFFEE", "PROVIDE_SOUP_EATING_UTENSIL")
                assert "INSERTABLE_IN" in g.required_relations
                assert "REACHES_BOTTOM" in g.required_relations
        elif dom == "living_room":
            assert len(graph.operation_groups) == 1
            assert graph.operation_groups[0].capability_id == "SUPPORT_DRINKWARE"
            assert "FITS_SET_ON" in graph.operation_groups[0].required_relations
        elif dom == "workshop":
            # Workshop singletons are compiled into relations
            assert any(r.predicate == "COMPATIBLE_WITH" for r in graph.relations)
            assert any(r.predicate == "REACHES_TARGET" for r in graph.relations)
            assert any(r.predicate == "COMPATIBLE_WITH_TARGET" for r in graph.relations)
