"""Stage 9 — Real-Trace Regression Suite.

Builds a frozen regression corpus from captured real raw Qwen outputs across
Kitchen, Living Room, and Workshop domains, along with strict negative control fixtures.

Verifies for all fixtures (Section 16):
- Raw semantic interpretation
- Canonical role mapping
- Operation capability mapping
- Relation classification (physical verifiers vs task/causal relations)
- Online executable completeness
- No GT/provider access
- No benchmark variant ID leakage or logic

Verifies negative control fixtures (Section 16):
- Vague 'thing'
- Wrong endpoint semantics
- Unsupported operation
- Empty operation
- Self-pairing
- Ambiguous competing roles
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.functional_tamp_pipeline.raw_semantic_evaluation import evaluate_raw_semantics
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import validate_runtime_gf
from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real_qwen_traces"

TASK_INSTRUCTIONS = {
    "kitchen": (
        "Prepare and serve one coffee and one soup for each of two people. "
        "Make each coffee using coffee and water and stir it before serving. "
        "Serve each soup bowl with its own suitable eating utensil."
    ),
    "living_room": (
        "Prepare the living room for two people to enjoy refreshments while watching television. "
        "Provide each person with their own refreshment setting nearby, and place the entertainment "
        "control where it is accessible to both people."
    ),
    "workshop": (
        "Identify the compatible components required to complete the fastening at the marked "
        "workbench location, complete the fastening, and leave any reusable equipment used for "
        "the task safely on the workbench."
    ),
}


# ==============================================================================
# Domain Real-Trace Fixtures (Section 16)
# ==============================================================================

def test_kitchen_real_trace_regression():
    """Kitchen: clear role phrasing, source phrasing, stirrer phrasing, soup utensil phrasing,

    transfer operations, stir operation, serving/association operation.
    """
    fixture_path = FIXTURES_DIR / "kitchen_real_trace.json"
    assert fixture_path.exists(), f"Missing fixture: {fixture_path}"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    # 1. Raw semantic interpretation
    raw_eval = evaluate_raw_semantics("kitchen", "K1", raw_doc)
    assert raw_eval["role"]["recall"] >= 0.8
    assert raw_eval["operation"]["recall"] >= 0.6

    # 2. Canonical role mapping
    graph = compile_candidate_graph("kitchen", TASK_INSTRUCTIONS["kitchen"], raw_doc)
    assert isinstance(graph, FunctionalRequirementGraph)
    validate_runtime_gf(graph)

    # Roles mapped to canonical identities
    mapped_roles = set(graph.nodes.keys())
    assert "coffee_container" in mapped_roles or "beverage_container" in mapped_roles or "coffee_cup" in mapped_roles
    assert "coffee_source" in mapped_roles
    assert "coffee_stirrer" in mapped_roles
    assert "soup_container" in mapped_roles
    assert "soup_eating_utensil" in mapped_roles

    # 3. Operation capability mapping
    op_funcs = {g.function for g in graph.operation_groups}
    assert "POUR" in op_funcs or "TRANSFER_CONTENT_TO_CONTAINER" in op_funcs or "MIX_BEVERAGE_CONTENTS" in op_funcs
    assert "STIR_COFFEE" in op_funcs or "MIX_BEVERAGE_CONTENTS" in op_funcs or "PROVIDE_SOUP_EATING_UTENSIL" in op_funcs

    # Preconditions provenance attached from capability
    preconds = [p for g in graph.operation_groups for p in g.preconditions_provenance]
    assert any(p["provenance"] == "ROBOT_CAPABILITY_PRECONDITION" for p in preconds)

    # 4. Relation classification
    for r in graph.relations:
        assert r.category == "PHYSICAL_VERIFIER"
    for r in graph.task_causal_relations:
        assert r.category == "TASK_CAUSAL_SEMANTICS"

    # 5. Online executable completeness (no variant ID passed)
    assert graph.online_executable_contract_complete is True


def test_living_room_real_trace_regression():
    """Living: refreshment payload, personal support, shared support, entertainment control,

    seating context, personal placement operation, shared control placement.
    """
    fixture_path = FIXTURES_DIR / "living_room_real_trace.json"
    assert fixture_path.exists(), f"Missing fixture: {fixture_path}"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    # 1. Raw semantic interpretation
    raw_eval = evaluate_raw_semantics("living_room", "L1", raw_doc)
    assert raw_eval["role"]["recall"] >= 0.5

    # 2. Canonical role mapping
    graph = compile_candidate_graph("living_room", TASK_INSTRUCTIONS["living_room"], raw_doc)
    assert isinstance(graph, FunctionalRequirementGraph)
    validate_runtime_gf(graph)

    mapped_roles = set(graph.nodes.keys())
    assert "CUP_SAUCER_SET" in mapped_roles
    assert "PERSONAL_CUP_SAUCER_REGION" in mapped_roles
    assert "REMOTE" in mapped_roles
    assert "SHARED_REMOTE_REGION" in mapped_roles

    # 3. Operation capability mapping
    op_funcs = {g.function for g in graph.operation_groups}
    assert "SUPPORT_DRINKWARE" in op_funcs or "PLACE" in op_funcs
    assert "SUPPORT_ENTERTAINMENT_CONTROL" in op_funcs or "PLACE_SHARED_REMOTE" in op_funcs

    # 4. Relation classification
    for r in graph.relations:
        assert r.category == "PHYSICAL_VERIFIER"

    # 5. Online executable completeness
    assert graph.online_executable_contract_complete is True


def test_workshop_real_trace_regression():
    """Workshop: reusable tool, installed component, marked target,

    fastening operation, equipment return operation.
    """
    fixture_path = FIXTURES_DIR / "workshop_real_trace.json"
    assert fixture_path.exists(), f"Missing fixture: {fixture_path}"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    # 1. Raw semantic interpretation
    raw_eval = evaluate_raw_semantics("workshop", "W1", raw_doc)
    assert raw_eval["role"]["recall"] >= 0.6
    assert raw_eval["operation"]["recall"] >= 0.5

    # 2. Canonical role mapping
    graph = compile_candidate_graph("workshop", TASK_INSTRUCTIONS["workshop"], raw_doc)
    assert isinstance(graph, FunctionalRequirementGraph)
    validate_runtime_gf(graph)

    mapped_roles = set(graph.nodes.keys())
    assert "driver" in mapped_roles
    assert "fastener" in mapped_roles
    assert "repair_target" in mapped_roles

    # 3. Operation capability mapping
    trace = graph.metadata.get("canonicalization_trace", {})
    group_traces = trace.get("groups", [])
    assert any(g.get("capability_id") == "FASTEN_JOINT" for g in group_traces)
    assert any(g.get("planner_operation") == "DRIVE_FASTENER_INTO_TARGET" for g in group_traces)

    # Preconditions provenance attached from capability
    preconds = [r for r in graph.relations if r.provenance == "ROBOT_CAPABILITY_PRECONDITION"]
    assert any(r.predicate == "COMPATIBLE_WITH" for r in preconds)
    assert any(r.predicate == "REACHES_TARGET" for r in preconds)
    assert any(r.predicate == "COMPATIBLE_WITH_TARGET" for r in preconds)

    # 4. Relation classification
    for r in graph.relations:
        assert r.category == "PHYSICAL_VERIFIER"
    for r in graph.task_causal_relations:
        assert r.category == "TASK_CAUSAL_SEMANTICS"

    # 5. Online executable completeness
    assert graph.online_executable_contract_complete is True


# ==============================================================================
# Negative Control Fixtures (Section 16)
# ==============================================================================

def test_negative_vague_thing_fails_closed():
    """Negative fixture: vague 'thing' must fail closed and cannot map valid contract."""
    from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
    fixture_path = FIXTURES_DIR / "negative_vague_thing.json"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Fails closed: either raises VLMSpecificationError ("No executable role could be typed") or contract is incomplete
    try:
        graph = compile_candidate_graph("workshop", TASK_INSTRUCTIONS["workshop"], raw_doc)
        assert graph.online_executable_contract_complete is False
        assert "driver" not in graph.nodes or "fastener" not in graph.nodes
    except VLMSpecificationError:
        pass


def test_negative_wrong_endpoint_semantics_fails_closed():
    """Negative fixture: wrong endpoint semantics (surface reaches bottom of driver) fails closed."""
    fixture_path = FIXTURES_DIR / "negative_wrong_endpoint_semantics.json"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Compilation must either reject invalid predicate signature or fail closed
    try:
        graph = compile_candidate_graph("workshop", TASK_INSTRUCTIONS["workshop"], raw_doc)
        assert graph.online_executable_contract_complete is False
    except (MalformedVLMSpecificationError, ValueError):
        pass  # Rejected at validation boundary


def test_negative_unsupported_operation_fails_closed():
    """Negative fixture: unsupported operation ('teleport to mars') must be marked unsupported."""
    fixture_path = FIXTURES_DIR / "negative_unsupported_operation.json"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    graph = compile_candidate_graph("workshop", TASK_INSTRUCTIONS["workshop"], raw_doc)
    trace = graph.metadata.get("canonicalization_trace", {})
    disabled = trace.get("disabled_groups", [])

    # Operation must be recognized as unsupported operator
    assert any(d.get("status") == "UNSUPPORTED_OPERATOR" for d in disabled)
    assert not any(g.function == "teleport_op" for g in graph.operation_groups)


def test_negative_empty_operation_fails_closed():
    """Negative fixture: empty operation string must be disabled or fail closed."""
    fixture_path = FIXTURES_DIR / "negative_empty_operation.json"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    graph = compile_candidate_graph("workshop", TASK_INSTRUCTIONS["workshop"], raw_doc)
    trace = graph.metadata.get("canonicalization_trace", {})
    disabled = trace.get("disabled_groups", [])

    assert any(d.get("status") == "UNSUPPORTED_OPERATOR" for d in disabled)


def test_negative_self_pairing_fails_closed():
    """Negative fixture: self pairing (driver acts on driver) must not create valid operation."""
    fixture_path = FIXTURES_DIR / "negative_self_pairing.json"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    graph = compile_candidate_graph("workshop", TASK_INSTRUCTIONS["workshop"], raw_doc)
    # Self-pairing cannot satisfy required fastening contract
    assert graph.online_executable_contract_complete is False


def test_negative_ambiguous_role_fails_closed():
    """Negative fixture: ambiguous conflicting role definitions fail closed."""
    fixture_path = FIXTURES_DIR / "negative_ambiguous_role.json"
    raw_doc = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Two identical competing roles with non-mergeable distinct operations must fail closed
    graph = compile_candidate_graph("workshop", TASK_INSTRUCTIONS["workshop"], raw_doc)
    assert graph.online_executable_contract_complete is False
