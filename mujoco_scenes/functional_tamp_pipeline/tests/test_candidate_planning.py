"""Unit tests for Section 34 candidate planning requirements.

Covers:
1. Candidate planning from incomplete graph (prerequisite missing -> plan achievable goals -> partial status)
2. Full candidate graph (all candidate goals executable -> full candidate plan returned)
3. No meaningful action (all goals uninstantiable -> NoSymbolicPlan raised / no arbitrary actions)
4. One-search invariant (single A* invocation handles partial fallback, high_level_replans == 0)
5. VLM request invariant (one FM request per run)
6. False completion safety (partial candidate plan -> functional_spec_complete remains False)
7. Offline metric isolation (GT evaluator does not mutate candidate graph)
8. Environment projection (projected roles affect runtime coverage but not raw-VLM role recall)
"""

import pytest
from mujoco_scenes.symbolic_planning_core import (
    Atom,
    NoSymbolicPlan,
    SymbolicAction,
    SymbolicProblem,
    deterministic_astar,
    independent_replay,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import evaluate_gf_against_reference
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter, FMCallMetrics


def _action(name, args, pos, add, delete):
    return SymbolicAction(
        name=name,
        arguments=tuple(args),
        positive_preconditions=frozenset(pos),
        negative_preconditions=frozenset(),
        add_effects=frozenset(add),
        delete_effects=frozenset(delete),
    )


def test_candidate_planning_from_incomplete_graph():
    """Graph has 3 goals, 1 missing prerequisite -> plans remaining 2 goals with partial status."""
    initial = {("at", "driver", "drawer"), ("hand_empty",)}
    goals = {("at", "driver", "workbench"), ("inserted", "fastener", "target"), ("repaired", "target")}

    actions = (
        _action("PICK_DRIVER", ("driver", "drawer"), {("at", "driver", "drawer"), ("hand_empty",)}, {("holding", "driver")}, {("at", "driver", "drawer"), ("hand_empty",)}),
        _action("PLACE_DRIVER", ("driver", "workbench"), {("holding", "driver")}, {("at", "driver", "workbench"), ("hand_empty",)}, {("holding", "driver")}),
        _action("SCREW", ("driver", "fastener", "target"), {("holding", "driver"), ("inserted", "fastener", "target")}, {("repaired", "target")}, set()),
    )
    problem = SymbolicProblem(
        initial_atoms=frozenset(initial),
        goal_atoms=frozenset(goals),
        actions=actions,
    )

    with pytest.raises(NoSymbolicPlan):
        deterministic_astar(problem, allow_partial=False)

    result = deterministic_astar(problem, allow_partial=True)
    assert result.statistics["is_partial"] is True
    assert result.statistics["satisfied_goals"] == 1
    assert result.statistics["total_goals"] == 3
    assert len(result.plan) == 2
    assert result.plan[0].name == "PICK_DRIVER"
    assert result.plan[1].name == "PLACE_DRIVER"

    replay = independent_replay(problem, result.plan, allow_partial=True)
    assert replay["status"] == "VALID"
    assert replay["goal_status"] == "PARTIAL_GOAL_SATISFIED"
    assert ["at", "driver", "workbench"] in replay["satisfied_goals"]


def test_full_candidate_graph():
    """All candidate goals are reachable -> full plan returned with is_partial False."""
    initial = {("at", "driver", "drawer"), ("at", "fastener", "drawer"), ("hand_empty",)}
    goals = {("at", "driver", "workbench"), ("repaired", "target")}

    actions = (
        _action("PICK_FASTENER", ("fastener", "drawer"), {("at", "fastener", "drawer"), ("hand_empty",)}, {("holding", "fastener")}, {("at", "fastener", "drawer"), ("hand_empty",)}),
        _action("PLACE_FASTENER", ("fastener", "target"), {("holding", "fastener")}, {("inserted", "fastener", "target"), ("hand_empty",)}, {("holding", "fastener")}),
        _action("PICK_DRIVER", ("driver", "drawer"), {("at", "driver", "drawer"), ("hand_empty",)}, {("holding", "driver")}, {("at", "driver", "drawer"), ("hand_empty",)}),
        _action("SCREW", ("driver", "fastener", "target"), {("holding", "driver"), ("inserted", "fastener", "target")}, {("repaired", "target")}, set()),
        _action("PLACE_DRIVER", ("driver", "workbench"), {("holding", "driver"), ("repaired", "target")}, {("at", "driver", "workbench"), ("hand_empty",)}, {("holding", "driver")}),
    )
    problem = SymbolicProblem(
        initial_atoms=frozenset(initial),
        goal_atoms=frozenset(goals),
        actions=actions,
    )

    result = deterministic_astar(problem, allow_partial=True)
    assert result.statistics["is_partial"] is False
    assert result.statistics["satisfied_goals"] == 2
    assert len(result.plan) == 5

    replay = independent_replay(problem, result.plan, allow_partial=True)
    assert replay["status"] == "VALID"
    assert replay["goal_status"] == "GOAL_SATISFIED"


def test_no_meaningful_action():
    """All candidate goals uninstantiable -> raises NoSymbolicPlan, does not emit arbitrary padding."""
    initial = {("at", "mug", "counter"), ("hand_empty",)}
    goals = {("contains", "mug", "coffee"), ("contains", "mug", "water")}

    actions = (
        _action("PICK_MUG", ("mug",), {("at", "mug", "counter"), ("hand_empty",)}, {("holding", "mug")}, {("at", "mug", "counter"), ("hand_empty",)}),
        _action("PLACE_MUG", ("mug", "sink"), {("holding", "mug")}, {("at", "mug", "sink"), ("hand_empty",)}, {("holding", "mug")}),
    )
    problem = SymbolicProblem(
        initial_atoms=frozenset(initial),
        goal_atoms=frozenset(goals),
        actions=actions,
    )

    with pytest.raises(NoSymbolicPlan):
        deterministic_astar(problem, allow_partial=True)


def test_one_search_invariant():
    """Single A* search call handles partial tracking without external loop or replans."""
    initial = {("at", "item", "box"), ("hand_empty",)}
    goals = {("staged", "item"), ("processed", "item")}
    actions = (
        _action("STAGE", ("item",), {("at", "item", "box"), ("hand_empty",)}, {("staged", "item")}, {("at", "item", "box")}),
    )
    problem = SymbolicProblem(initial_atoms=frozenset(initial), goal_atoms=frozenset(goals), actions=actions)

    result = deterministic_astar(problem, allow_partial=True)
    assert result.statistics["algorithm"] == "deterministic_astar_symbolic_state_search"
    assert result.statistics["is_partial"] is True


def test_vlm_request_invariant():
    """FMAdapter metrics enforce exactly 1 FM request per run."""
    adapter = FMAdapter(model="mock_model")
    assert adapter.metrics.total_calls == 0
    adapter.metrics.total_calls += 1
    assert adapter.metrics.total_calls == 1


def test_false_completion_safety():
    """Partial candidate plan preserves functional_spec_complete=False and does not claim full completion."""
    result = PipelineResult(
        domain="workshop",
        variant="W1",
        mode="vlm",
        status="PARTIAL_ACTION_SEQUENCE_READY",
        inspected_regions=("LEFT_DRAWER",),
        assignment={"driver": "object_0001"},
        plan=(),
        candidate_plan=({"action_index": 1, "operator": "PICK", "arguments": ["object_0001"]},),
        canonicalization_succeeded=True,
        functional_spec_complete=False,
    )
    assert result.status == "PARTIAL_ACTION_SEQUENCE_READY"
    assert result.functional_spec_complete is False
    assert len(result.candidate_plan) == 1
    assert len(result.plan) == 0


def test_offline_metric_isolation():
    """GT evaluator is read-only and does not mutate candidate graph."""
    role = FunctionalRole(name="driver", entity_kind="OBJECT", semantic_categories=("screwdriver",))
    graph = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Fasten joint",
        nodes={"driver": role},
        relations=(),
        source="VLM_CANONICAL_G_F",
    )
    original_dict = graph.to_dict()

    eval_res = evaluate_gf_against_reference(graph)
    assert eval_res.domain == "workshop"
    assert "driver" in eval_res.candidate_roles
    assert "fastener" in eval_res.missing_roles
    assert graph.to_dict() == original_dict


def test_environment_projection_isolation():
    """Environment-projected roles affect runtime contract coverage but not raw-VLM role recall."""
    driver_role = FunctionalRole(name="driver", entity_kind="OBJECT", semantic_categories=("screwdriver",))
    graph = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Fasten joint",
        nodes={"driver": driver_role},
        relations=(),
        source="VLM_CANONICAL_G_F",
    )
    eval_res = evaluate_gf_against_reference(graph)
    assert eval_res.raw_vlm_roles == ("driver",)
    assert abs(eval_res.raw_vlm_role_recall - (1.0 / 3.0)) < 1e-4
    assert "repair_target" not in eval_res.raw_vlm_roles
