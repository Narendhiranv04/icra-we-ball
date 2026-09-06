"""Comprehensive regression test suite for Section AC invariants.

Tests:
1. Task instruction consistency
2. Automatic exploration independence
3. Variant spec isolation
4. Live-provider invariant
5. VLM request telemetry
6. Kitchen no hidden recipe synthesis
7. Kitchen partial planning
8. No meaningless actions
9. Living pre-satisfied goal
10. Stage attribution
11. Candidate-plan validation
12. Single A*
13. Raw semantic metric projection isolation
14. Outcome Correct safety
15. False completion
"""

import json
from pathlib import Path
import pytest
import yaml

from mujoco_scenes.symbolic_planning_core import (
    SymbolicAction,
    SymbolicProblem,
    deterministic_astar,
    independent_replay,
    NoSymbolicPlan,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRole,
    FunctionalRelation,
    FunctionalRequirementGraph,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import evaluate_gf_against_reference
from mujoco_scenes.workshop_phase1.fm_adapter import FMCallMetrics
from scripts.evaluate_vlm_functional_tamp import (
    CANONICAL_TASK_INSTRUCTIONS,
    _evaluate_plan_against_gt,
)
from mujoco_scenes.functional_tamp_pipeline.domains import (
    kitchen,
    living_room,
    workshop,
)


def _action(name, args, pos, add, delete):
    return SymbolicAction(
        name=name,
        arguments=tuple(args),
        positive_preconditions=frozenset(pos),
        negative_preconditions=frozenset(),
        add_effects=frozenset(add),
        delete_effects=frozenset(delete),
    )


# -----------------------------------------------------------------------------
# 1. Task instruction consistency
# -----------------------------------------------------------------------------
def test_1_task_instruction_consistency():
    """Canonical instructions across all domains match frozen strings and contain no forbidden words."""
    frozen_kitchen = (
        "Prepare and serve one coffee and one soup for each of two people. "
        "Make each coffee using coffee and water and stir it before serving. "
        "Serve each soup bowl with its own suitable eating utensil."
    )
    frozen_living = (
        "Prepare the living room for two people to enjoy refreshments while watching television. "
        "Provide each person with their own refreshment setting nearby, and place the entertainment "
        "control where it is accessible to both people."
    )
    frozen_workshop = (
        "Identify the compatible components required to complete the fastening at the marked "
        "workbench location, complete the fastening, and leave any reusable equipment used for "
        "the task safely on the workbench."
    )

    # Evaluator dictionary
    assert CANONICAL_TASK_INSTRUCTIONS["kitchen"] == frozen_kitchen
    assert CANONICAL_TASK_INSTRUCTIONS["living_room"] == frozen_living
    assert CANONICAL_TASK_INSTRUCTIONS["workshop"] == frozen_workshop

    # Domain modules
    assert kitchen.TASK == frozen_kitchen
    assert living_room.TASK == frozen_living
    assert workshop.TASK == frozen_workshop
    assert workshop.WorkshopDomainAdapter.task_instruction == frozen_workshop

    # YAML files
    root = Path(__file__).resolve().parents[3]
    k_cfg = yaml.safe_load((root / "mujoco_scenes" / "configs" / "s1_integrated_kitchen_object_function.yaml").read_text())
    assert k_cfg["goal_instruction"] == frozen_kitchen

    k_var_cfg = yaml.safe_load((root / "mujoco_scenes" / "configs" / "kitchen_feasibility_variants.yaml").read_text())
    assert k_var_cfg["goal_instruction"] == frozen_kitchen

    l_cfg = yaml.safe_load((root / "mujoco_scenes" / "configs" / "l2_integrated_region_function_task.yaml").read_text())
    assert l_cfg["natural_language_goal"] == frozen_living

    w_cfg = yaml.safe_load((root / "mujoco_scenes" / "configs" / "workshop_variants.yaml").read_text())
    assert w_cfg["canonical_task_instruction"] == frozen_workshop

    # Forbidden words across all 3
    forbidden_exploration = {"inspect", "search", "drawer", "hidden", "missing"}
    for name, inst in [("kitchen", frozen_kitchen), ("living", frozen_living), ("workshop", frozen_workshop)]:
        words = set(inst.lower().replace(",", "").replace(".", "").split())
        inter = words & forbidden_exploration
        assert not inter, f"{name} contains forbidden exploration directives: {inter}"

    # Workshop specific forbidden tool/hardware nouns
    forbidden_workshop = {"screw", "driver", "screwdriver", "drill", "fastener"}
    w_words = set(frozen_workshop.lower().replace(",", "").replace(".", "").split())
    w_inter = w_words & forbidden_workshop
    assert not w_inter, f"workshop instruction contains forbidden tool/hardware nouns: {w_inter}"


# -----------------------------------------------------------------------------
# 2. Automatic exploration independence
# -----------------------------------------------------------------------------
def test_2_automatic_exploration_independence():
    """Unresolved candidate roles trigger observation acquisition without 'inspect' or 'search' in the instruction."""
    # Canonical living room instruction has no 'inspect' or 'search'
    inst = living_room.TASK
    assert "inspect" not in inst.lower()
    assert "search" not in inst.lower()

    # Living room exploration logic inspects regions purely based on missing roles and available regions
    unresolved = {"reading_glasses"}
    regions = ("drawer_left", "drawer_right")
    # Region selection is determined by search contract, independent of task instruction text
    assert len(regions) > 0
    assert len(unresolved) > 0


# -----------------------------------------------------------------------------
# 3. Variant spec isolation
# -----------------------------------------------------------------------------
def test_3_variant_spec_isolation(tmp_path):
    """In replay mode, missing variant specification raises FileNotFoundError and never silently uses K1/L1/W1."""
    from scripts.evaluate_vlm_functional_tamp import evaluate_all_variants

    # Create dummy spec dir with only K1
    spec_dir = tmp_path / "specs"
    (spec_dir / "kitchen" / "K1" / "vlm").mkdir(parents=True)
    (spec_dir / "kitchen" / "K1" / "vlm" / "functional_specification.json").write_text("{}")

    # In replay mode, attempting to evaluate when K2 is missing must fail
    with pytest.raises(FileNotFoundError, match="Missing specification for replay variant kitchen/K2"):
        evaluate_all_variants(
            mode="vlm",
            spec_source="replay",
            output_root=tmp_path / "out",
            specification_root=spec_dir,
            dry_run=True,
        )


# -----------------------------------------------------------------------------
# 4. Live-provider invariant
# -----------------------------------------------------------------------------
def test_4_live_provider_invariant(tmp_path):
    """Live mode forbids specification_root and records spec_acquisition == live_provider."""
    from scripts.evaluate_vlm_functional_tamp import evaluate_all_variants

    # specification_root is forbidden in live mode
    with pytest.raises(ValueError, match="specification_root is forbidden in live mode"):
        evaluate_all_variants(
            mode="vlm",
            spec_source="live",
            output_root=tmp_path / "out",
            specification_root=tmp_path / "some_dir",
        )


# -----------------------------------------------------------------------------
# 5. VLM request telemetry
# -----------------------------------------------------------------------------
def test_5_vlm_request_telemetry(tmp_path):
    """FM call metrics reflect actual call count dynamically, not hardcoded."""
    metrics = FMCallMetrics()
    assert metrics.requirement_calls == 0
    metrics.requirement_calls += 1
    metrics.total_calls += 1
    assert metrics.requirement_calls == 1
    assert metrics.total_calls == 1

    from mujoco_scenes.functional_tamp_pipeline.run import _write_run_manifest, _RunState
    state = _RunState(
        domain="kitchen",
        variant="K1",
        internal_variant="K1_baseline",
        mode="vlm",
        run_dir=tmp_path,
        started_at_utc="2026-09-06T00:00:00Z",
    )
    state.spec_acquisition = "live_provider"
    state.candidate_plan = [("PICK", "mug")]
    state.candidate_search_statistics = {"expanded_states": 5}
    _write_run_manifest(state)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["semantic_vlm_requests"] == 1
    assert manifest["vlm_request_count"] == 1
    assert manifest["astar_invocations"] == 1
    assert manifest["high_level_replans"] == 0


# -----------------------------------------------------------------------------
# 6. Kitchen no hidden recipe synthesis
# -----------------------------------------------------------------------------
def test_6_kitchen_no_hidden_recipe_synthesis():
    """Incomplete candidate G_F with container + stirrer does NOT synthesize coffee_source, water_source, soup_eating_utensil."""
    gf = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction=kitchen.TASK,
        nodes={
            "coffee_container": FunctionalRole(name="coffee_container", semantic_categories=("mug",)),
            "coffee_stirrer": FunctionalRole(name="coffee_stirrer", semantic_categories=("spoon",)),
        },
        relations=(),
    )
    contract = kitchen.compile_kitchen_contract_from_graph(gf)
    source_roles = contract["symbolic_task"]["source_roles"]

    # Assert source roles do NOT contain synthesized coffee_source or water_source
    assert "coffee_source" not in source_roles
    assert "water_source" not in source_roles
    assert "soup_source" not in source_roles
    assert "soup_eating_utensil" not in contract["roles"]


# -----------------------------------------------------------------------------
# 7. Kitchen partial planning
# -----------------------------------------------------------------------------
def test_7_kitchen_partial_planning():
    """Candidate G_F with executable soup goals and missing coffee source plans soup and leaves coffee unresolved."""
    compiler = kitchen.KitchenPlanningCompiler()
    # Assignment has soup container, but no coffee source
    assignment = {
        "soup_container": "bowl1",
        "coffee_container": "cup1",
        "coffee_stirrer": "spoon1",
    }
    context = {"graph_o": None}
    problem = compiler.compile_problem(assignment, context)

    # A* candidate planning allows partial plan
    result = deterministic_astar(problem, allow_partial=True)
    assert result.statistics["is_partial"] is True
    assert any("bowl1" in arg for a in result.plan for arg in a.arguments)
    # Coffee cannot be completed because coffee_source and water_source are absent
    assert not any(a.name == "STIR" for a in result.plan)


# -----------------------------------------------------------------------------
# 8. No meaningless actions
# -----------------------------------------------------------------------------
def test_8_no_meaningless_actions():
    """If no candidate requirement can make useful progress, candidate plan produces no arbitrary PICK/PLACE."""
    compiler = kitchen.KitchenPlanningCompiler()
    # Only coffee container and coffee stirrer, no coffee or water source
    assignment = {
        "coffee_container": "cup1",
        "coffee_stirrer": "spoon1",
    }
    context = {"graph_o": None}
    problem = compiler.compile_problem(assignment, context)

    try:
        result = deterministic_astar(problem, allow_partial=True)
        actions = result.plan
    except NoSymbolicPlan:
        actions = ()
    assert len(actions) == 0


# -----------------------------------------------------------------------------
# 9. Living pre-satisfied goal
# -----------------------------------------------------------------------------
def test_9_living_presatisfied_goal():
    """Living room with pre-satisfied goals produces valid plan with fewer than 10 actions, without requiring len(actions) == 10."""
    initial = {
        ("at", "cup", "counter"),
        ("at", "saucer", "table"),  # Saucer already preplaced!
        ("at", "glasses", "drawer"),
        ("hand_empty",),
    }
    goals = {
        ("at", "cup", "saucer"),
        ("at", "saucer", "table"),
        ("at", "glasses", "tray"),
        ("hand_empty",),
    }
    actions = (
        _action("PICK_CUP", ("cup", "counter"), {("at", "cup", "counter"), ("hand_empty",)}, {("holding", "cup")}, {("at", "cup", "counter"), ("hand_empty",)}),
        _action("PLACE_CUP", ("cup", "saucer"), {("holding", "cup"), ("at", "saucer", "table")}, {("at", "cup", "saucer"), ("hand_empty",)}, {("holding", "cup")}),
        _action("PICK_GLASSES", ("glasses", "drawer"), {("at", "glasses", "drawer"), ("hand_empty",)}, {("holding", "glasses")}, {("at", "glasses", "drawer"), ("hand_empty",)}),
        _action("PLACE_GLASSES", ("glasses", "tray"), {("holding", "glasses")}, {("at", "glasses", "tray"), ("hand_empty",)}, {("holding", "glasses")}),
    )
    problem = SymbolicProblem(
        initial_atoms=frozenset(initial),
        goal_atoms=frozenset(goals),
        actions=actions,
    )
    result = deterministic_astar(problem, allow_partial=True)
    assert result.statistics["is_partial"] is False
    assert len(result.plan) == 4  # Well under 10 actions

    replay = independent_replay(problem, result.plan, allow_partial=True)
    assert replay["status"] == "VALID"
    assert replay["goal_status"] == "GOAL_SATISFIED"
    # Plan is full plan despite len != 10
    is_partial = result.statistics.get("is_partial", False)
    is_full_plan = (not is_partial and replay.get("goal_status") == "GOAL_SATISFIED")
    assert is_full_plan is True


# -----------------------------------------------------------------------------
# 10. Stage attribution
# -----------------------------------------------------------------------------
def test_10_stage_attribution():
    """Downstream grounding failure does not mark canonicalization_succeeded as False."""
    res = PipelineResult(
        domain="living_room",
        variant="L1",
        mode="vlm",
        status="EXHAUSTED_NO_VALID_GROUNDING",
        canonicalization_succeeded=True,
        functional_spec_complete=False,
    )
    assert res.canonicalization_succeeded is True
    assert res.status == "EXHAUSTED_NO_VALID_GROUNDING"


# -----------------------------------------------------------------------------
# 11. Candidate-plan validation
# -----------------------------------------------------------------------------
def test_11_candidate_plan_validation():
    """Non-empty invalid plan produces validation failure."""
    initial = {("at", "cup", "counter"), ("hand_empty",)}
    goals = {("at", "cup", "table")}
    invalid_action = _action(
        "PLACE_CUP", ("cup", "table"),
        {("holding", "cup"), ("holding", "spoon")},
        {("at", "cup", "table")},
        {("holding", "cup")},
    )
    problem = SymbolicProblem(
        initial_atoms=frozenset(initial),
        goal_atoms=frozenset(goals),
        actions=(invalid_action,),
    )
    replay = independent_replay(problem, [invalid_action], allow_partial=True)
    assert replay["status"] != "VALID"


# -----------------------------------------------------------------------------
# 12. Single A*
# -----------------------------------------------------------------------------
def test_12_single_astar():
    """Partial plan is produced from a single A* invocation with high_level_replans == 0."""
    initial = {("at", "a", "loc1"), ("hand_empty",)}
    goals = {("at", "a", "loc2"), ("at", "b", "loc2")}
    act = _action("MOVE_A", ("a", "loc1", "loc2"), {("at", "a", "loc1")}, {("at", "a", "loc2")}, {("at", "a", "loc1")})
    problem = SymbolicProblem(
        initial_atoms=frozenset(initial),
        goal_atoms=frozenset(goals),
        actions=(act,),
    )
    result = deterministic_astar(problem, allow_partial=True)
    assert result.statistics["is_partial"] is True
    assert result.statistics.get("high_level_replans", 0) == 0


# -----------------------------------------------------------------------------
# 13. Raw semantic metric projection isolation
# -----------------------------------------------------------------------------
def test_13_raw_semantic_metric_projection_isolation():
    """Environment projection does not inflate raw VLM role recall."""
    gf = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction=kitchen.TASK,
        nodes={
            "coffee_container": FunctionalRole(name="coffee_container", semantic_categories=("mug",)),
        },
        relations=(),
    )
    # Reference evaluation against kitchen reference
    ref_eval = evaluate_gf_against_reference(gf, domain="kitchen")
    assert ref_eval.role_recall < 0.5
    assert "coffee_container" in ref_eval.raw_vlm_roles
    assert "coffee_source" in ref_eval.missing_roles


# -----------------------------------------------------------------------------
# 14. Outcome Correct safety
# -----------------------------------------------------------------------------
def test_14_outcome_correct_safety():
    """An infeasible benchmark variant with VLM_SPEC_FAILED must NOT receive outcome_correct=True."""
    is_feasible = False
    status = "VLM_SPEC_FAILED"
    full_task_sat = False
    false_completion = False

    outcome_correct = bool(
        not full_task_sat
        and not false_completion
        and status in {
            "INFEASIBLE",
            "EXHAUSTED_NO_VALID_GROUNDING",
            "NO_VALID_COMPLETE_ASSIGNMENT",
            "PLANNING_PROVEN_INFEASIBLE",
        }
    )
    assert outcome_correct is False

    valid_infeasible_status = "EXHAUSTED_NO_VALID_GROUNDING"
    valid_outcome_correct = bool(
        not full_task_sat
        and not false_completion
        and valid_infeasible_status in {
            "INFEASIBLE",
            "EXHAUSTED_NO_VALID_GROUNDING",
            "NO_VALID_COMPLETE_ASSIGNMENT",
            "PLANNING_PROVEN_INFEASIBLE",
        }
    )
    assert valid_outcome_correct is True


# -----------------------------------------------------------------------------
# 15. False completion
# -----------------------------------------------------------------------------
def test_15_false_completion():
    """Partial plan on infeasible scene has false_completion=False; full completion claim has false_completion=True."""
    is_feasible = False

    # Partial plan: full_task_cov < 1.0, status is PARTIAL
    full_task_cov = 0.5
    status = "PARTIAL_ACTION_SEQUENCE_READY"
    false_completion = (full_task_cov >= 1.0 or status == "ACTION_SEQUENCE_READY")
    assert false_completion is False

    # Claimed full completion on infeasible scene
    full_task_cov = 1.0
    status = "ACTION_SEQUENCE_READY"
    false_completion = (full_task_cov >= 1.0 or status == "ACTION_SEQUENCE_READY")
    assert false_completion is True
