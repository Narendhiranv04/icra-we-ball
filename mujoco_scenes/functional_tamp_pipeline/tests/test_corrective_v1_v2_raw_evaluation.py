"""Corrective-pass regressions for schema-aware raw FM evaluation."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

from mujoco_scenes.functional_tamp_pipeline.evaluation_contract_adapter import extract_evaluation_contract
from mujoco_scenes.functional_tamp_pipeline.raw_semantic_evaluation import evaluate_raw_semantics


TASK = "Identify compatible components required to complete fastening at the marked target, complete it, and return reusable equipment safely."


def _roles():
    return [
        {"id": "tool", "entity_kind": "OBJECT", "function": "driving tool", "candidate_categories": ["screwdriver"], "required_count": 1, "binding_policy": "DISTINCT"},
        {"id": "part", "entity_kind": "OBJECT", "function": "fastening component", "candidate_categories": ["screw"], "required_count": 1, "binding_policy": "DISTINCT"},
        {"id": "target", "entity_kind": "FIXED_TARGET", "function": "marked repair target hole", "candidate_categories": [], "required_count": 1, "binding_policy": "DISTINCT"},
    ]


def _relations():
    return [
        {"subject_role": "tool", "relation": "compatible with", "object_role": "part", "required": True},
        {"subject_role": "tool", "relation": "reaches target", "object_role": "target", "required": True},
        {"subject_role": "part", "relation": "compatible with target", "object_role": "target", "required": True},
    ]


def _v1():
    return {"status": "SUPPORTED", "functional_roles": _roles(), "functional_relations": _relations(), "interaction_groups": []}


def _v2():
    return {"status": "SUPPORTED", "task_contract": {"functional_roles": _roles(), "functional_relations": _relations(), "operation_pairings": []}, "observation_guidance": {}}


def test_v1_raw_contract_extraction():
    contract = extract_evaluation_contract(_v1())
    assert contract.schema == "V1" and len(contract.roles) == 3 and len(contract.relations) == 3


def test_v2_raw_contract_extraction():
    contract = extract_evaluation_contract(_v2())
    assert contract.schema == "V2" and len(contract.roles) == 3 and len(contract.relations) == 3


def test_v2_roles_are_read_from_task_contract():
    assert len(extract_evaluation_contract(_v2()).roles) == 3


def test_v2_relations_are_read_from_task_contract():
    assert len(extract_evaluation_contract(_v2()).relations) == 3


def test_v2_operations_are_read_from_operation_pairings():
    raw = _v2()
    raw["task_contract"]["operation_pairings"] = [{"id": "op", "operation": "fasten", "source_role": "tool", "target_role": "part", "operation_count": 1, "reuse_policy": "REUSABLE_ACROSS_TARGETS"}]
    operation = extract_evaluation_contract(raw).operations[0]
    assert operation.schema == "V2_OPERATION_PAIRING" and operation.source_role == "tool"


def test_nonempty_v2_contract_cannot_score_zero_due_only_to_nesting():
    metrics = evaluate_raw_semantics("workshop", TASK, _v2())
    assert metrics["role"]["recall"] == 1.0
    assert metrics["relation"]["recall"] == 1.0


def test_equivalent_v1_v2_role_meaning_scores_equally():
    assert evaluate_raw_semantics("workshop", TASK, _v1())["role"] == evaluate_raw_semantics("workshop", TASK, _v2())["role"]


def test_equivalent_v1_v2_relation_meaning_scores_equally():
    assert evaluate_raw_semantics("workshop", TASK, _v1())["relation"] == evaluate_raw_semantics("workshop", TASK, _v2())["relation"]


def test_equivalent_v1_v2_operation_meaning_scores_equally():
    assert evaluate_raw_semantics("workshop", TASK, _v1())["operation"] == evaluate_raw_semantics("workshop", TASK, _v2())["operation"]


def test_raw_evaluator_does_not_import_production_semantic_compiler():
    source = Path("mujoco_scenes/functional_tamp_pipeline/raw_semantic_evaluation.py").read_text()
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("semantic_compiler" not in ast.unparse(node) for node in imports)


def test_missing_v2_role_is_detected():
    raw = _v2(); raw["task_contract"]["functional_roles"].pop()
    assert evaluate_raw_semantics("workshop", TASK, raw)["role"]["recall"] < 1.0


def test_missing_v2_relation_is_detected():
    raw = _v2(); raw["task_contract"]["functional_relations"].pop()
    assert evaluate_raw_semantics("workshop", TASK, raw)["relation"]["recall"] < 1.0


def test_missing_v2_operation_is_detected():
    # Kitchen has required reference operations; an empty V2 list must not pass.
    raw = {"task_contract": {"functional_roles": [], "functional_relations": [], "operation_pairings": []}}
    assert evaluate_raw_semantics("kitchen", "prepare servings", raw)["operation"]["recall"] == 0.0


def test_wrong_count_is_detected():
    raw = _v2(); raw["task_contract"]["functional_roles"][0]["required_count"] = 2
    assert evaluate_raw_semantics("workshop", TASK, raw)["count_correct"] is False


def test_wrong_binding_policy_is_detected():
    raw = _v2(); raw["task_contract"]["functional_roles"][0]["binding_policy"] = "SHARED"
    assert evaluate_raw_semantics("workshop", TASK, raw)["binding_correct"] is False
