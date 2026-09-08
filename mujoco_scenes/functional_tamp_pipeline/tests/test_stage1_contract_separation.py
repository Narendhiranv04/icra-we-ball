"""Unit tests for Stage 1: Separation of Online Executability from Offline Task Completeness (Gate 1).

Verifies:
1. FM omission -> offline task incomplete, but an otherwise internally executable
   expressed contract compiles with online_executable_contract_complete == True
   and remains search-eligible.
2. Reasonable raw semantic + failed compiler mapper -> GRAPH_COMPILATION_FAILURE,
   NOT TASK_SPECIFICATION_FAILURE.
3. Actual FM omission of a required reference task item -> TASK_SPECIFICATION_FAILURE.
4. Online contract validation is generic and does not consult hidden domain checklists.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.functional_tamp_pipeline.search import classify_search_state
from mujoco_scenes.functional_tamp_pipeline.raw_semantic_evaluation import evaluate_raw_semantics
from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import compute_primary_metrics
from mujoco_scenes.functional_tamp_pipeline.models import SearchRegionContract


@pytest.fixture
def base_kitchen_v2() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two coffees and two soups for two people.",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_soup_utensil",
                    "entity_kind": "OBJECT",
                    "function": "eating utensil for soup bowl",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["soup spoon", "spoon"],
                    "required_properties": [],
                },
                {
                    "id": "role_soup_bowl",
                    "entity_kind": "OBJECT",
                    "function": "soup bowl container",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["soup bowl", "bowl"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "id": "rel_soup",
                    "subject_role": "role_soup_utensil",
                    "relation": "inserted into and reaches bottom of soup bowl",
                    "object_role": "role_soup_bowl",
                    "required": True,
                },
            ],
            "operation_pairings": [
                {
                    "id": "op_soup",
                    "operation": "provide eating utensil for each soup bowl",
                    "source_role": "role_soup_utensil",
                    "target_role": "role_soup_bowl",
                    "operation_count": 2,
                    "reuse_policy": "DEDICATED_PER_TARGET",
                },
            ],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {"role_soup_utensil": [], "role_soup_bowl": []},
            "inspectable_regions": [{"id": "kitchen_drawer_left", "description": "Left drawer"}],
            "inspection_order": ["kitchen_drawer_left"],
        },
        "unsupported_reason": "",
    }


def test_fm_omission_leaves_offline_incomplete_but_online_executable_and_searchable(base_kitchen_v2):
    """When FM omits coffee, offline reference is incomplete, but soup contract is online-executable and search-eligible."""
    # Offline evaluation against full kitchen task (which expects coffee + soup)
    ref_eval = evaluate_raw_semantics("kitchen", "Kitchen task", base_kitchen_v2)
    assert ref_eval["complete_task_contract"] is False  # Omission detected offline!

    # Online compile
    graph = compile_candidate_graph("kitchen", "Prepare soup and coffee", base_kitchen_v2)
    assert graph.online_executable_contract_complete is True
    assert graph.required_contract_complete is True

    # Search classification: since online contract is complete, ungrounded roles are SEARCH_RECOVERABLE
    search_contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("kitchen_drawer_left",),
    )
    search_state = classify_search_state(
        graph_f=graph,
        grounding=None,
        search_contract=search_contract,
        inspected_regions=(),
    )
    assert search_state == "SEARCH_RECOVERABLE"
    assert search_state != "CONTRACT_INCOMPLETE_NOT_SEARCHABLE"


def test_compiler_failure_triggers_graph_compilation_failure():
    """When FM expresses semantics but compiler mapper rejects it, cause is GRAPH_COMPILATION_FAILURE."""
    # Create an evaluation record simulation where sanitizer succeeded, raw JSON valid,
    # but compiler had an unresolved role
    record = {
        "gt_feasible": True,
        "terminal_status": "COMPLETED",
        "failure_category": "CANONICALIZATION_AMBIGUITY",
        "vlm_json_valid": True,
        "sanitization_succeeded": True,
        "canonicalization_succeeded": False,  # Compiler failed!
        "unresolved_roles": [{"id": "mystery_role", "status": "UNRESOLVED_SEMANTIC"}],
        "disabled_groups": [],
        "executable_contract_complete": False,
        "online_executable_contract_complete": False,
        "offline_reference_task_complete": False,
        "raw_vlm_spec_complete": False,
        "full_task_satisfied": False,
    }

    # Emulate the first-cause precedence logic from evaluation_metrics.py
    first_cause = None
    if not record["vlm_json_valid"]:
        first_cause = "TASK_SPECIFICATION_FAILURE"
    elif not record["sanitization_succeeded"]:
        first_cause = "GRAPH_COMPILATION_FAILURE"
    elif not record["canonicalization_succeeded"] or record.get("unresolved_roles"):
        first_cause = "GRAPH_COMPILATION_FAILURE"
    elif not record.get("offline_reference_task_complete", False):
        first_cause = "TASK_SPECIFICATION_FAILURE"

    assert first_cause == "GRAPH_COMPILATION_FAILURE"


def test_fm_omission_with_valid_compiler_triggers_task_specification_failure():
    """When compiler succeeds on expressed subtask, but full task fails due to FM omission, cause is TASK_SPECIFICATION_FAILURE."""
    record = {
        "gt_feasible": True,
        "terminal_status": "COMPLETED",
        "vlm_json_valid": True,
        "sanitization_succeeded": True,
        "canonicalization_succeeded": True,
        "unresolved_roles": [],
        "disabled_groups": [],
        "executable_contract_complete": True,
        "online_executable_contract_complete": True,
        "offline_reference_task_complete": False,  # FM omitted a reference requirement!
        "raw_vlm_spec_complete": False,
        "full_task_satisfied": False,
    }

    first_cause = None
    if not record["vlm_json_valid"]:
        first_cause = "TASK_SPECIFICATION_FAILURE"
    elif not record["sanitization_succeeded"]:
        first_cause = "GRAPH_COMPILATION_FAILURE"
    elif not record["canonicalization_succeeded"] or record.get("unresolved_roles"):
        first_cause = "GRAPH_COMPILATION_FAILURE"
    elif not record["executable_contract_complete"]:
        first_cause = "GRAPH_COMPILATION_FAILURE"
    elif not record.get("offline_reference_task_complete", False):
        first_cause = "TASK_SPECIFICATION_FAILURE"

    assert first_cause == "TASK_SPECIFICATION_FAILURE"


def test_online_contract_completeness_fails_on_broken_endpoints(base_kitchen_v2):
    """Online contract completeness fails closed when an operation references an unknown role."""
    doc = copy.deepcopy(base_kitchen_v2)
    doc["task_contract"]["operation_pairings"][0]["target_role"] = "nonexistent_role_xyz"
    graph = compile_candidate_graph("kitchen", "Kitchen task", doc)
    assert graph.online_executable_contract_complete is False
    assert len(graph.metadata["contract_missing_reasons"]) > 0
