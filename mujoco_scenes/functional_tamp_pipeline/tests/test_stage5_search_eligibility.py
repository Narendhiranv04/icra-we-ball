"""Unit tests for Stage 5: Fix Search Eligibility and Causal Recovery.

Verifies:
1. Search state machine classifies fine-grained states (Section 12.1):
   - CONTRACT_UNEXECUTABLE
   - GROUNDING_COMPLETE
   - NO_CANDIDATE_SEARCHABLE
   - ONLY_FALSE_CANDIDATES_SEARCHABLE
   - ONLY_UNKNOWN_CANDIDATES_SEARCHABLE
   - NO_VALID_JOINT_ASSIGNMENT_SEARCHABLE
   - SEARCH_EXHAUSTED
   - GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE
2. Backward compatibility of classify_search_state(detailed=False).
3. is_searchable_state(), is_complete_state(), and is_unexecutable_state() predicates.
4. Evidence-driven inspection eligibility (Section 12.2, 12.3):
   - Incomplete contract fails closed to unexecutable.
   - Search proceeds when online contract is complete and requirements are ungrounded due to candidate evidence.
5. Online contract vs offline reference decoupling (Section 12.4):
   - Search is not blocked when online contract is complete, even if offline reference contract is incomplete.
6. Causal search recovery metric (Section 12.5):
   - Strictly requires initial_grounding_complete == False, inspected > 0, final_grounding_complete == True.
7. search_until_satisfied records both coarse and fine search states in snapshots.
"""

from __future__ import annotations

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
    CONTRACT_UNEXECUTABLE,
    CONTRACT_INCOMPLETE_NOT_SEARCHABLE,
    GROUNDING_COMPLETE,
    SATISFIED,
    SEARCH_EXHAUSTED,
    GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE,
    NO_CANDIDATE_SEARCHABLE,
    ONLY_FALSE_CANDIDATES_SEARCHABLE,
    ONLY_UNKNOWN_CANDIDATES_SEARCHABLE,
    NO_VALID_JOINT_ASSIGNMENT_SEARCHABLE,
    SEARCH_RECOVERABLE,
    SEARCHABLE_STATES,
    COMPLETE_STATES,
    UNEXECUTABLE_STATES,
    is_searchable_state,
    is_complete_state,
    is_unexecutable_state,
    classify_search_state,
    classify_fine_search_state,
    compute_causal_search_recovery,
    search_until_satisfied,
)


def _build_test_graph(
    *,
    online_complete: bool = True,
    offline_complete: bool = True,
    searchable_roles: bool = True,
) -> FunctionalRequirementGraph:
    if searchable_roles:
        nodes = {
            "fastener": FunctionalRole(
                name="fastener",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("screw", "bolt"),
            ),
            "driver": FunctionalRole(
                name="driver",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("screwdriver", "wrench"),
            ),
        }
        relations = (
            FunctionalRelation(
                subject_role="driver",
                predicate="TOOL_ENGAGES_FASTENER",
                object_role="fastener",
                expected=True,
            ),
        )
        groups = (
            OperationGroup(
                id="op_fasten",
                function="fasten bolt",
                tool_role="driver",
                target_role="fastener",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=("TOOL_ENGAGES_FASTENER",),
            ),
        )
    else:
        nodes = {
            "robot_arm": FunctionalRole(
                name="robot_arm",
                entity_kind="ROBOT",
                count=1,
                semantic_categories=(),
            ),
        }
        relations = ()
        groups = ()

    return FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Fasten component",
        nodes=nodes,
        relations=relations,
        operation_groups=groups,
        candidate_regions=("TOOL_DRAWER", "PARTS_BIN", "SIDE_CABINET"),
        region_ranking=("TOOL_DRAWER", "PARTS_BIN", "SIDE_CABINET"),
        source="V2_PROMPT_COMPILER",
        metadata={
            "online_executable_contract_complete": online_complete,
            "required_contract_complete": offline_complete,
        },
    )


class MockSearchDomain:
    def __init__(self, satisfaction_sequence: list[GraphGroundingResult]):
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


def test_search_predicates():
    """Verify is_searchable_state, is_complete_state, and is_unexecutable_state."""
    for s in SEARCHABLE_STATES:
        assert is_searchable_state(s) is True
        assert is_complete_state(s) is False
        assert is_unexecutable_state(s) is False

    for s in COMPLETE_STATES:
        assert is_searchable_state(s) is False
        assert is_complete_state(s) is True
        assert is_unexecutable_state(s) is False

    for s in UNEXECUTABLE_STATES:
        assert is_searchable_state(s) is False
        assert is_complete_state(s) is False
        assert is_unexecutable_state(s) is True

    assert is_searchable_state(SEARCH_EXHAUSTED) is False
    assert is_searchable_state(GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE) is False
    assert is_searchable_state(None) is False


def test_classify_contract_unexecutable():
    """When online executable contract is incomplete, classification returns unexecutable."""
    graph = _build_test_graph(online_complete=False)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(status="INCOMPLETE", complete=False)

    # Detailed classification
    assert classify_fine_search_state(graph, grounding, contract) == CONTRACT_UNEXECUTABLE
    assert classify_search_state(graph, grounding, contract, detailed=True) == CONTRACT_UNEXECUTABLE

    # Backward-compatible coarse classification
    assert classify_search_state(graph, grounding, contract, detailed=False) == CONTRACT_INCOMPLETE_NOT_SEARCHABLE


def test_classify_grounding_complete():
    """When grounding is satisfied, classification returns grounding complete."""
    graph = _build_test_graph(online_complete=True)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"fastener": "screw_1", "driver": "driver_1"},
    )

    assert classify_fine_search_state(graph, grounding, contract) == GROUNDING_COMPLETE
    assert classify_search_state(graph, grounding, contract, detailed=True) == GROUNDING_COMPLETE
    assert classify_search_state(graph, grounding, contract, detailed=False) == SATISFIED


def test_classify_search_exhausted():
    """When all candidate regions are inspected, classification returns SEARCH_EXHAUSTED."""
    graph = _build_test_graph(online_complete=True)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(status="INCOMPLETE", complete=False, missing_roles=("fastener",))

    state = classify_fine_search_state(
        graph, grounding, contract, inspected_regions=("TOOL_DRAWER", "PARTS_BIN")
    )
    assert state == SEARCH_EXHAUSTED
    assert classify_search_state(
        graph, grounding, contract, inspected_regions=("TOOL_DRAWER", "PARTS_BIN")
    ) == SEARCH_EXHAUSTED


def test_classify_no_candidate_searchable():
    """Zero observed candidates for a required searchable role yields NO_CANDIDATE_SEARCHABLE."""
    graph = _build_test_graph(online_complete=True)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    # Fastener is completely missing from observed candidates
    grounding = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("fastener",),
        evidence={"candidate_evaluations": {}},
    )

    fine = classify_fine_search_state(graph, grounding, contract, inspected_regions=())
    assert fine == NO_CANDIDATE_SEARCHABLE
    assert is_searchable_state(fine) is True

    # Coarse classification maps to SEARCH_RECOVERABLE
    assert classify_search_state(graph, grounding, contract) == SEARCH_RECOVERABLE


def test_classify_only_false_candidates_searchable():
    """Observed candidates all evaluated to FALSE yields ONLY_FALSE_CANDIDATES_SEARCHABLE."""
    graph = _build_test_graph(online_complete=True)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    # Driver candidate was observed, but evaluated to FALSE (wrong tool type)
    grounding = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("driver",),
        evidence={
            "candidate_evaluations": {
                "driver:hammer_1": {"status": "FALSE", "passed": False},
                "driver:pliers_1": {"status": "FALSE", "passed": False},
            },
        },
    )

    fine = classify_fine_search_state(graph, grounding, contract, inspected_regions=())
    assert fine == ONLY_FALSE_CANDIDATES_SEARCHABLE
    assert is_searchable_state(fine) is True
    assert classify_search_state(graph, grounding, contract) == SEARCH_RECOVERABLE


def test_classify_only_unknown_candidates_searchable():
    """Observed candidates evaluated to UNKNOWN yields ONLY_UNKNOWN_CANDIDATES_SEARCHABLE."""
    graph = _build_test_graph(online_complete=True)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    # Driver candidate is occluded / lack of camera views
    grounding = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("driver",),
        evidence={
            "candidate_evaluations": {
                "driver:unknown_tool_1": {"status": "UNKNOWN", "passed": None},
            },
            "unresolved_relations": [
                {"subject_role": "driver", "predicate": "TOOL_ENGAGES_FASTENER", "status": "UNKNOWN"}
            ],
        },
    )

    fine = classify_fine_search_state(graph, grounding, contract, inspected_regions=())
    assert fine == ONLY_UNKNOWN_CANDIDATES_SEARCHABLE
    assert is_searchable_state(fine) is True
    assert classify_search_state(graph, grounding, contract) == SEARCH_RECOVERABLE


def test_classify_no_valid_joint_assignment_searchable():
    """Candidates individually exist with TRUE, but cannot be jointly assigned."""
    graph = _build_test_graph(online_complete=True)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    # Driver and fastener are TRUE individually, but the relation between them evaluated to FALSE
    grounding = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        assignment=None,
        missing_roles=(),
        unsatisfied_relations=(
            {
                "subject_role": "driver",
                "predicate": "TOOL_ENGAGES_FASTENER",
                "object_role": "fastener",
                "status": "FALSE",
            },
        ),
        unresolved_constraints=("INSUFFICIENT_SCENE_OBJECTS_FOR_ROLES",),
        evidence={
            "candidate_evaluations": {
                "driver:hex_key_1": {"status": "TRUE", "passed": True},
                "fastener:phillips_screw_1": {"status": "TRUE", "passed": True},
            },
        },
    )

    fine = classify_fine_search_state(graph, grounding, contract, inspected_regions=())
    assert fine == NO_VALID_JOINT_ASSIGNMENT_SEARCHABLE
    assert is_searchable_state(fine) is True
    assert classify_search_state(graph, grounding, contract) == SEARCH_RECOVERABLE


def test_classify_grounding_failure_not_search_recoverable():
    """Non-searchable implicated role yields GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE."""
    graph = _build_test_graph(online_complete=True, searchable_roles=False)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(status="INCOMPLETE", complete=False, missing_roles=("robot_arm",))

    state = classify_fine_search_state(graph, grounding, contract)
    assert state == GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE
    assert classify_search_state(graph, grounding, contract) == GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE


def test_online_contract_not_blocked_by_offline_reference():
    """Section 12.4: Online executable contract is complete, but offline reference is incomplete.

    Search MUST NOT be blocked.
    """
    graph = _build_test_graph(online_complete=True, offline_complete=False)
    assert graph.online_executable_contract_complete is True
    assert graph.required_contract_complete is False

    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )
    grounding = GraphGroundingResult(status="INCOMPLETE", complete=False, missing_roles=("fastener",))

    state = classify_search_state(graph, grounding, contract)
    assert state == SEARCH_RECOVERABLE
    fine = classify_fine_search_state(graph, grounding, contract)
    assert fine == NO_CANDIDATE_SEARCHABLE
    assert is_searchable_state(state) is True


def test_causal_search_recovery_metric_strictness():
    """Section 12.5: Causal search recovery requires initial=False, inspected>0, final=True."""
    # Succeeded search
    assert compute_causal_search_recovery(False, ["TOOL_DRAWER"], True) is True
    assert compute_causal_search_recovery(False, ["TOOL_DRAWER", "PARTS_BIN"], True) is True

    # Initial was already complete -> not a search recovery
    assert compute_causal_search_recovery(True, ["TOOL_DRAWER"], True) is False

    # Zero regions inspected -> not a search recovery
    assert compute_causal_search_recovery(False, [], True) is False

    # Final grounding remained incomplete -> failed search
    assert compute_causal_search_recovery(False, ["TOOL_DRAWER"], False) is False


def test_search_until_satisfied_snapshots_and_fine_states():
    """search_until_satisfied records coarse and fine search states in snapshots and evidence."""
    graph = _build_test_graph(online_complete=True)
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("TOOL_DRAWER", "PARTS_BIN"),
        source="POLICY",
    )

    # Initial: missing fastener (NO_CANDIDATE_SEARCHABLE)
    initial_res = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("fastener",),
        evidence={"candidate_evaluations": {}},
    )
    # After TOOL_DRAWER: driver found, fastener missing (NO_CANDIDATE_SEARCHABLE)
    after_drawer_res = GraphGroundingResult(
        status="INCOMPLETE",
        complete=False,
        missing_roles=("fastener",),
        evidence={"candidate_evaluations": {"driver:driver_1": {"status": "TRUE"}}},
    )
    # After PARTS_BIN: both found and complete (GROUNDING_COMPLETE)
    after_bin_res = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"fastener": "screw_1", "driver": "driver_1"},
    )

    domain = MockSearchDomain([initial_res, after_drawer_res, after_bin_res])
    result, inspected = search_until_satisfied(
        domain, graph, search_contract=contract, emit=lambda _: None
    )

    assert result.complete is True
    assert inspected == ("TOOL_DRAWER", "PARTS_BIN")
    assert result.evidence.get("causal_search_recovery") is True

    snapshots = result.evidence.get("grounding_snapshots", [])
    assert len(snapshots) >= 3

    # Initial snapshot
    assert snapshots[0]["search_state"] == SEARCH_RECOVERABLE
    assert snapshots[0]["fine_search_state"] == NO_CANDIDATE_SEARCHABLE

    # Snapshot after TOOL_DRAWER
    assert snapshots[1]["search_state"] == SEARCH_RECOVERABLE
    assert snapshots[1]["fine_search_state"] == NO_CANDIDATE_SEARCHABLE

    # Final snapshot after PARTS_BIN
    assert snapshots[2]["search_state"] in COMPLETE_STATES
    assert snapshots[2]["fine_search_state"] == GROUNDING_COMPLETE
