"""Grounding may prune impossible branches, and nothing else.

The joint assignment used to be built as a full Cartesian product of per-role
candidate combinations and only then checked, so every violation was discovered
after the whole tuple existed.  Pruning has to leave the logical result alone:
the same assignment must be chosen, an unsettled question must stay unsettled,
and UNKNOWN must keep its status as evidence that has not been resolved rather
than evidence against.

Separately, a graph whose compile blockers no further observation can settle
must not have assignment hypotheses enumerated for it at all -- the reading
cannot change what is reported, and on the frozen distribution it cost minutes
per trial.

No reference graph, expected plan or benchmark variant appears here.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mujoco_scenes.functional_tamp_pipeline.grounding import (
    JOINT_ASSIGNMENT_PREFIX_BUDGET,
    PROVISIONAL_TYPE_HYPOTHESIS_BUDGET,
    ground_graph,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    classify_non_scene_resolvable_blockers,
)


# ---------------------------------------------------------------------------
# What observation can and cannot fix
# ---------------------------------------------------------------------------


def test_an_operation_with_no_capability_is_not_a_scene_problem():
    blockers = classify_non_scene_resolvable_blockers(
        {"unresolved_required_operations": [{"id": "fill_soup"}]}, [object()], [], 2)
    assert blockers == ["REQUIRED_OPERATION_HAS_NO_RUNTIME_CAPABILITY"]


def test_losing_every_expressed_operation_is_not_a_scene_problem():
    blockers = classify_non_scene_resolvable_blockers({}, [], [], 3)
    assert blockers == ["NO_EXPRESSED_PHYSICAL_OPERATION_SURVIVED_COMPILATION"]


def test_an_uninterpretable_required_relation_is_not_a_scene_problem():
    blockers = classify_non_scene_resolvable_blockers({}, [object()], [{"raw_phrase": "used_for"}], 1)
    assert blockers == ["REQUIRED_RELATION_HAS_NO_CANONICAL_INTERPRETATION"]


def test_a_graph_the_runtime_can_execute_has_no_such_blocker():
    assert classify_non_scene_resolvable_blockers({}, [object()], [], 1) == []


def test_a_provisional_role_type_is_left_to_search():
    """An undecided canonical type is exactly what more observation can settle."""
    assert classify_non_scene_resolvable_blockers(
        {"unresolved_roles": [{"code": "AMBIGUOUS_ROLE_MAPPING"}]}, [object()], [], 1) == []


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _scene(count: int = 6) -> ObservedSceneGraph:
    graph_o = ObservedSceneGraph()
    for index in range(1, count + 1):
        graph_o.add_node(ObservedNode(instance_id=f"object_{index:04d}", entity_kind="OBJECT",
                                      canonical_category="cup"))
    return graph_o


def _provisional_graph(metadata: dict | None = None) -> FunctionalRequirementGraph:
    return FunctionalRequirementGraph(
        domain="kitchen", task_instruction="two coffees",
        nodes={
            "fm_role__thing": FunctionalRole(
                name="fm_role__thing", entity_kind="OBJECT", count=1,
                binding_policy="REUSABLE", semantic_categories=("cup",),
                canonical_role_candidates=("coffee_container", "coffee_source",
                                           "coffee_stirrer", "water_source")),
        },
        metadata=dict(metadata or {}),
    )


def test_a_non_resolvable_graph_has_no_hypotheses_enumerated():
    result = ground_graph(
        _provisional_graph({"non_scene_resolvable_blockers":
                            ["NO_EXPRESSED_PHYSICAL_OPERATION_SURVIVED_COMPILATION"]}),
        _scene(), {"search_exhausted": True})
    assert not result.complete
    assert result.evidence["type_hypotheses_evaluated"] == 0
    assert result.unresolved_constraints == ("NO_EXPRESSED_PHYSICAL_OPERATION_SURVIVED_COMPILATION",)


def test_a_scene_resolvable_provisional_graph_still_gets_enumerated():
    result = ground_graph(_provisional_graph(), _scene(), {"search_exhausted": True})
    assert result.evidence.get("type_hypotheses_evaluated", 0) > 0 or result.complete


def test_the_budgets_are_generous_enough_to_be_backstops():
    assert PROVISIONAL_TYPE_HYPOTHESIS_BUDGET >= 1024
    assert JOINT_ASSIGNMENT_PREFIX_BUDGET >= 100_000


# ---------------------------------------------------------------------------
# Pruning preserves the result, including which assignment is chosen
# ---------------------------------------------------------------------------


def _pair_graph(policy: str = "DISTINCT") -> FunctionalRequirementGraph:
    return FunctionalRequirementGraph(
        domain="kitchen", task_instruction="two coffees",
        nodes={
            "coffee_container": FunctionalRole(
                name="coffee_container", entity_kind="OBJECT", count=2,
                binding_policy="DISTINCT", semantic_categories=("cup",)),
            "coffee_stirrer": FunctionalRole(
                name="coffee_stirrer", entity_kind="OBJECT", count=2,
                binding_policy=policy, semantic_categories=("spoon",)),
        },
        operation_groups=(
            OperationGroup(
                id="stir", function="STIR_COFFEE", tool_role="coffee_stirrer",
                target_role="coffee_container", required_target_count=2,
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("INSERTABLE_IN",),
                capability_id="STIR_COFFEE",
                physical_preconditions=(("coffee_stirrer", "INSERTABLE_IN", "coffee_container"),),
            ),
        ),
    )


def _pair_scene(*, insertable: str = "ALL") -> ObservedSceneGraph:
    graph_o = ObservedSceneGraph()
    cups = [f"cup_{i}" for i in (1, 2, 3)]
    spoons = [f"spoon_{i}" for i in (1, 2, 3)]
    for cup in cups:
        graph_o.add_node(ObservedNode(instance_id=cup, entity_kind="OBJECT", canonical_category="cup"))
    for spoon in spoons:
        graph_o.add_node(ObservedNode(instance_id=spoon, entity_kind="OBJECT", canonical_category="spoon"))
    for spoon in spoons:
        for cup in cups:
            status = {"ALL": "TRUE", "NONE": "FALSE", "UNKNOWN": "UNKNOWN"}[insertable]
            graph_o.add_relation(ObservedRelation(subject_id=spoon, predicate="INSERTABLE_IN",
                                                  object_id=cup, status=status))
    return graph_o


def test_the_first_admissible_assignment_is_still_the_one_reported():
    """Pruning must not change which of several valid assignments is chosen.

    The roles are visited in sorted order and each role's combinations in
    combination order, so the reported witness is the lexicographically first
    admissible one and stays stable as the search gets cheaper.
    """
    result = ground_graph(_pair_graph(), _pair_scene(), {"search_exhausted": True})
    assert result.complete, result.unresolved_constraints
    assert sorted(result.assignment["coffee_container"]) == ["cup_1", "cup_2"]
    assert sorted(result.assignment["coffee_stirrer"]) == ["spoon_1", "spoon_2"]


def test_a_definitively_impossible_graph_is_still_refused():
    result = ground_graph(_pair_graph(), _pair_scene(insertable="NONE"), {"search_exhausted": True})
    assert not result.complete
    assert result.status == "INFEASIBLE"


def test_unknown_evidence_stays_unsettled_rather_than_becoming_a_refusal():
    """UNKNOWN never prunes: an unproven relation is not a disproven one."""
    result = ground_graph(_pair_graph(), _pair_scene(insertable="UNKNOWN"),
                          {"search_exhausted": True})
    assert not result.complete
    assert result.status == "INCOMPLETE"
    assert result.failure_kind == "FUNCTIONAL_ASSIGNMENT_FAILURE"


def test_the_work_the_search_did_is_recorded():
    result = ground_graph(_pair_graph(), _pair_scene(insertable="NONE"), {"search_exhausted": True})
    assert result.evidence["joint_assignment_prefixes_examined"] > 0
    assert "joint_assignment_search_budget_exhausted" not in result.evidence


def test_pruning_a_prefix_does_not_hide_why_it_failed():
    result = ground_graph(_pair_graph(), _pair_scene(insertable="NONE"), {"search_exhausted": True})
    diagnostics = result.evidence.get("pruned_prefix_diagnostics") or result.evidence.get(
        "unsatisfied_relations") or []
    assert diagnostics, "a refusal must say which constraint refused"


@pytest.mark.parametrize("policy", ["DISTINCT", "REUSABLE", "SHARED"])
def test_every_binding_policy_still_grounds(policy):
    result = ground_graph(_pair_graph(policy), _pair_scene(), {"search_exhausted": True})
    assert result.complete, (policy, result.unresolved_constraints)
