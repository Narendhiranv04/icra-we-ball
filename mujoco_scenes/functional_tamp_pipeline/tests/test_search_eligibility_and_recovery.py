"""Unit tests for Phase 8: Search Eligibility and Causal Search Recovery (Gate 8).

Verifies:
1. classify_search_state classifies all 5 canonical states:
   - CONTRACT_INCOMPLETE_NOT_SEARCHABLE
   - SATISFIED
   - SEARCH_EXHAUSTED
   - SEARCH_RECOVERABLE
   - GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE
2. Contract-incomplete Kitchen fixture performs zero pointless search.
3. Complete-contract fixture with invalid visible candidate can still search for alternative.
4. roles_in_missing_or_failed_bindings properly implicates:
   - ungrounded roles
   - under-cardinality roles
   - endpoints of FALSE / unsatisfied relations
   - endpoints of UNKNOWN relations
   - members of failed operation groups
5. Grounding snapshots are recorded before and after each search step.
6. Causal search recovery metric strictly requires initial=False, inspected>0, final=True.
"""

from __future__ import annotations

import copy
from typing import Any
import pytest

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    FunctionalRelation,
    OperationGroup,
    GraphGroundingResult,
    SearchRegionContract,
)
from mujoco_scenes.functional_tamp_pipeline.search import (
    classify_search_state,
    compute_causal_search_recovery,
    roles_in_missing_or_failed_bindings,
    search_until_satisfied,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph


def _build_test_graph(
    domain: str = "kitchen",
    contract_complete: bool = True,
    with_roles: bool = True,
) -> FunctionalRequirementGraph:
    nodes = {}
    if with_roles:
        nodes = {
            "coffee_stirrer": FunctionalRole(
                name="coffee_stirrer",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("spoon", "stirrer"),
            ),
            "coffee_container": FunctionalRole(
                name="coffee_container",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("cup", "mug"),
            ),
        }
    relations = (
        FunctionalRelation(
            subject_role="coffee_stirrer",
            predicate="REACHES_BOTTOM",
            object_role="coffee_container",
            expected=True,
        ),
    ) if with_roles else ()
    groups = (
        OperationGroup(
            id="op_stir",
            function="stir beverage in cups",
            tool_role="coffee_stirrer",
            target_role="coffee_container",
            required_target_count=1,
            usage_policy="SEQUENTIAL_REUSE_ALLOWED",
            required_relations=("REACHES_BOTTOM",),
        ),
    ) if with_roles else ()

    return FunctionalRequirementGraph(
        domain=domain,
        task_instruction="Prepare beverage",
        nodes=nodes,
        relations=relations,
        operation_groups=groups,
        candidate_regions=("D1", "D2", "C2", "B1", "C1"),
        region_ranking=("D1", "D2", "C2", "B1", "C1"),
        source="V2_PROMPT_COMPILER",
        metadata={
            "required_contract_complete": contract_complete,
            "raw_schema_version": 2,
        },
    )


class MockSearchDomain:
    def __init__(
        self,
        satisfaction_sequence: list[GraphGroundingResult],
    ):
        self.satisfaction_sequence = list(satisfaction_sequence)
        self.observe_initial_calls = 0
        self.observe_after_open_calls: list[str] = []
        self.opened_regions: list[str] = []

    def observe_initial(self) -> None:
        self.observe_initial_calls += 1

    def evaluate_satisfaction(self, search_exhausted: bool = False) -> GraphGroundingResult:
        if self.satisfaction_sequence:
            return self.satisfaction_sequence.pop(0)
        return GraphGroundingResult(status="INCOMPLETE", complete=False)

    def open_region(self, region: str) -> dict[str, Any]:
        self.opened_regions.append(region)
        return {"success": True, "region": region}

    def observe_after_open(self, region: str) -> None:
        self.observe_after_open_calls.append(region)


def test_classify_search_state_contract_incomplete():
    """A graph with no functional roles has nothing to look for."""
    graph = _build_test_graph(contract_complete=False)
    graph.nodes.clear()
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(status="INCOMPLETE", complete=False)

    state = classify_search_state(graph, grounding, contract, inspected_regions=())
    assert state == "CONTRACT_INCOMPLETE_NOT_SEARCHABLE"


def test_classify_search_state_satisfied():
    """Complete grounding is classified as SATISFIED."""
    graph = _build_test_graph(contract_complete=True)
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"coffee_stirrer": "spoon_1", "coffee_container": "cup_1"},
    )

    state = classify_search_state(graph, grounding, contract, inspected_regions=())
    assert state == "SATISFIED"


def test_classify_search_state_exhausted():
    """When all candidate regions have been inspected, search is SEARCH_EXHAUSTED."""
    graph = _build_test_graph(contract_complete=True)
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("coffee_stirrer",),
    )

    state = classify_search_state(
        graph, grounding, contract, inspected_regions=("D1", "D2", "C2", "B1", "C1")
    )
    assert state == "SEARCH_EXHAUSTED"


def test_classify_search_state_search_recoverable():
    """Uninspected regions + searchable ungrounded role = SEARCH_RECOVERABLE."""
    graph = _build_test_graph(contract_complete=True)
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("coffee_stirrer",),
    )

    state = classify_search_state(
        graph, grounding, contract, inspected_regions=()
    )
    assert state == "SEARCH_RECOVERABLE"


def test_classify_search_state_grounding_failure_not_search_recoverable():
    """When no implicated role is searchable in storage regions, state is NOT_SEARCH_RECOVERABLE."""
    nodes = {
        "robot_arm": FunctionalRole(
            name="robot_arm",
            entity_kind="ROBOT",
            count=1,
            semantic_categories=(),
        )
    }
    graph = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Arm control",
        nodes=nodes,
        candidate_regions=("D1",),
        region_ranking=("D1",),
        source="TEST",
        metadata={"required_contract_complete": True},
    )
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1",),
        source="POLICY",
    )
    grounding = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("robot_arm",),
    )

    state = classify_search_state(graph, grounding, contract, inspected_regions=())
    assert state == "GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE"


def test_contract_incomplete_graph_still_searches_for_a_missing_object():
    """A partially represented contract may still be completed by looking.

    Whether every FM semantic was representable says nothing about whether the
    missing object is in a drawer. A role that is unbound, groundable, and
    implicated in the failure is exactly what search exists to recover, so the
    scene is inspected rather than written off.
    """
    graph = _build_test_graph(domain="kitchen", contract_complete=False)
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="POLICY",
    )

    # Mock domain whose initial evaluation is INCOMPLETE
    initial_res = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("coffee_stirrer",),
    )
    domain = MockSearchDomain([initial_res])

    result, inspected = search_until_satisfied(
        domain,
        graph,
        search_contract=contract,
    )

    # Initial observation was called, and the scene was actually inspected.
    assert domain.observe_initial_calls == 1
    assert len(inspected) >= 1
    assert domain.opened_regions, 'a searchable missing role must trigger inspection'
    assert inspected, 'inspected regions must be reported'
    assert result.complete is False
    # The scene was searched to exhaustion and still could not supply the role,
    # which is a discovery outcome rather than a refusal to look.
    assert result.evidence.get("search_state") == "SEARCH_EXHAUSTED"


def test_complete_contract_fixture_with_invalid_visible_candidate_searches_for_alternative():
    """Gate 8 criterion: complete-contract fixture with invalid visible candidate can still search for alternative.
    
    A visible candidate exists for 'coffee_stirrer' (spoon_1), but fails geometry (REACHES_BOTTOM == False).
    Search must NOT terminate immediately; it must be SEARCH_RECOVERABLE and search D1,
    where a valid spoon_2 is found.
    """
    graph = _build_test_graph(domain="kitchen", contract_complete=True)
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="POLICY",
    )

    # 1. Initial evaluation: spoon_1 is visible on counter but fails REACHES_BOTTOM
    initial_res = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        assignment={"coffee_stirrer": "spoon_1", "coffee_container": "cup_1"},
        unsatisfied_relations=(
            {
                "subject_role": "coffee_stirrer",
                "subject_id": "spoon_1",
                "predicate": "REACHES_BOTTOM",
                "object_role": "coffee_container",
                "object_id": "cup_1",
                "status": "FALSE",
            },
        ),
    )

    # 2. After D1 is opened, valid spoon_2 is found and satisfies all relations
    after_drawer_res = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"coffee_stirrer": "spoon_2", "coffee_container": "cup_1"},
    )

    # Verify classification of initial state: MUST be SEARCH_RECOVERABLE despite having a candidate
    initial_search_state = classify_search_state(
        graph, initial_res, contract, inspected_regions=()
    )
    assert initial_search_state == "SEARCH_RECOVERABLE"

    domain = MockSearchDomain([initial_res, after_drawer_res])

    result, inspected = search_until_satisfied(
        domain,
        graph,
        search_contract=contract,
    )

    # Valid recovery: D1 was opened, alternative was found, search stopped at D1
    assert domain.opened_regions == ["D1"]
    assert inspected == ("D1",)
    assert result.complete is True
    assert result.assignment == {"coffee_stirrer": "spoon_2", "coffee_container": "cup_1"}
    assert result.evidence.get("causal_search_recovery") is True


def test_roles_in_missing_or_failed_bindings_implications():
    """roles_in_missing_or_failed_bindings properly captures all failure modes."""
    graph = _build_test_graph(contract_complete=True)

    # 1. Missing roles
    res1 = GraphGroundingResult(status="INCOMPLETE", complete=False, missing_roles=("coffee_container",))
    assert "coffee_container" in roles_in_missing_or_failed_bindings(graph, res1)

    # 2. Under-cardinality in assignment
    res2 = GraphGroundingResult(status="INCOMPLETE", complete=False, assignment={"coffee_stirrer": []})
    assert "coffee_stirrer" in roles_in_missing_or_failed_bindings(graph, res2)

    # 3. Unsatisfied relations
    res3 = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        unsatisfied_relations=(
            {"subject_role": "coffee_stirrer", "object_role": "coffee_container", "predicate": "REACHES_BOTTOM"},
        ),
    )
    implicated = roles_in_missing_or_failed_bindings(graph, res3)
    assert "coffee_stirrer" in implicated
    assert "coffee_container" in implicated

    # 4. Unresolved relations (UNKNOWN in evidence)
    res4 = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        evidence={"unresolved_relations": [{"subject_role": "coffee_stirrer", "predicate": "REACHES_BOTTOM"}]},
    )
    assert "coffee_stirrer" in roles_in_missing_or_failed_bindings(graph, res4)

    # 5. Failed operation group bindings
    res5 = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        operation_bindings={"op_stir": []},  # required count is 1, found 0
    )
    implicated5 = roles_in_missing_or_failed_bindings(graph, res5)
    assert "coffee_stirrer" in implicated5
    assert "coffee_container" in implicated5


def test_causal_search_recovery_metric_truth_table():
    """Gate 8 criterion: causal search recovery strictly requires initial=False, inspected>0, final=True."""
    # Succeeded recovery
    assert compute_causal_search_recovery(
        initial_grounding_complete=False,
        inspected_regions=("D1",),
        final_grounding_complete=True,
    ) is True

    # Initial was already complete -> Not a search recovery
    assert compute_causal_search_recovery(
        initial_grounding_complete=True,
        inspected_regions=("D1",),
        final_grounding_complete=True,
    ) is False

    # Zero regions inspected -> Not a search recovery
    assert compute_causal_search_recovery(
        initial_grounding_complete=False,
        inspected_regions=(),
        final_grounding_complete=True,
    ) is False

    # Final still incomplete -> Search failed to recover
    assert compute_causal_search_recovery(
        initial_grounding_complete=False,
        inspected_regions=("D1",),
        final_grounding_complete=False,
    ) is False


def test_grounding_snapshots_recorded():
    """Initial and after-search grounding snapshots are recorded in evidence."""
    graph = _build_test_graph(contract_complete=True)
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="POLICY",
    )

    initial_res = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("coffee_stirrer",),
    )
    after_drawer = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"coffee_stirrer": "spoon_1", "coffee_container": "cup_1"},
    )
    domain = MockSearchDomain([initial_res, after_drawer])

    result, inspected = search_until_satisfied(
        domain,
        graph,
        search_contract=contract,
    )

    snapshots = result.evidence.get("grounding_snapshots")
    assert snapshots is not None
    assert len(snapshots) == 2

    # Snapshot 0: initial
    assert snapshots[0]["stage"] == "initial"
    assert snapshots[0]["inspected_regions"] == []
    assert snapshots[0]["search_state"] == "SEARCH_RECOVERABLE"
    assert snapshots[0]["grounding"]["complete"] is False

    # Snapshot 1: after_D1
    assert snapshots[1]["stage"] == "after_D1"
    assert snapshots[1]["inspected_regions"] == ["D1"]
    assert snapshots[1]["search_state"] == "SATISFIED"
    assert snapshots[1]["grounding"]["complete"] is True


def test_kitchen_run_to_plan_contract_incomplete_zero_search(tmp_path, monkeypatch):
    """Kitchen run_to_plan sets sequence to () when contract is incomplete, performing zero search."""
    from pathlib import Path
    from mujoco_scenes.functional_tamp_pipeline.domains import kitchen

    spec = _build_test_graph(domain="kitchen", contract_complete=False)
    captured_sequence = None

    class FakeSession:
        events_path = tmp_path / "events.jsonl"
        run_dir = tmp_path
        registry = {"objects": {}}
    FakeSession.events_path.write_text("")

    def fake_run_sequential_inspection(scene, sequence, **kwargs):
        nonlocal captured_sequence
        captured_sequence = sequence
        return FakeSession()

    monkeypatch.setattr(kitchen, "run_sequential_inspection", fake_run_sequential_inspection)
    monkeypatch.setattr(kitchen, "scene_for_variant", lambda v: None)
    from mujoco_scenes.functional_tamp_pipeline import grounding
    monkeypatch.setattr(grounding, "ground_graph", lambda spec, go, ctx: GraphGroundingResult(status="INCOMPLETE", complete=False))

    result = kitchen.run_to_plan(
        variant_label="K1",
        internal_variant="variant_01",
        mode="gt",
        specification=spec,
        output_dir=tmp_path,
        scene=None,
    )

    assert captured_sequence == ()
    assert result.inspected_regions == ()
    assert result.functional_spec_complete is False
    assert (tmp_path / "grounding_snapshots.json").exists()

