"""Gate 9 Verification Tests: Planning Validation, Independent Replay, and Grounding Audit.

Verifies:
1. W1 produces its known full valid plan with complete audit verification.
2. Incomplete and invalid grounding can never be attributed to PLANNING_FAILURE.
3. Candidate plan independent replay enforces deterministic state transitions.
4. Domain planning compilers receive and inspect operation bindings.
5. Plan grounding audit covers all plan-used task objects fail-closed.
"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalSpecification, FunctionalRole, FunctionalRelation, OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedSceneGraph, ObservedNode, ObservedRelation,
)
from mujoco_scenes.functional_tamp_pipeline.planning import (
    plan_with_common_astar, PlannedSequence,
)
from mujoco_scenes.functional_tamp_pipeline.domains.workshop import (
    WorkshopPlanningCompiler, SURFACE, TARGET,
)
from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import (
    KitchenPlanningCompiler,
)
from mujoco_scenes.functional_tamp_pipeline.audit import (
    audit_plan_grounding,
)
from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
    enrich_record, candidate_plan_valid,
)
from mujoco_scenes.symbolic_planning_core import (
    SymbolicAction, SymbolicProblem, deterministic_astar, independent_replay,
)
from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline
from mujoco_scenes.functional_tamp_pipeline.tests.test_p3i_full_ideal_convergence import (
    MockFMAdapter,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"
TASK_WORKSHOP = (
    "Identify the compatible components required to complete the fastening at the "
    "marked workbench location, complete the fastening, and leave any reusable "
    "equipment used for the task safely on the workbench."
)


def test_w1_produces_known_full_valid_plan(tmp_path: Path):
    """Gate 9: W1 produces its known full valid plan with clean audit pass."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    if not w1_file.exists():
        pytest.skip(f"W1 fixture missing: {w1_file}")

    raw = json.loads(w1_file.read_text(encoding="utf-8"))
    adapter = MockFMAdapter(raw)

    out_dir = tmp_path / "w1_run"
    with patch("mujoco_scenes.workshop_phase1.fm_adapter.FMAdapter", return_value=adapter), \
         patch("mujoco_scenes.workshop_phase1.requirements.FMAdapter", return_value=adapter), \
         patch("mujoco_scenes.environment_vlm_requirements.FMAdapter", return_value=adapter):

        res = run_pipeline(
            domain="workshop",
            variant="W1",
            mode="vlm",
            output_root=out_dir,
            search_order="auto",
        )

    assert res.status == "ACTION_SEQUENCE_READY"
    assert res.plan is not None and len(res.plan) >= 5
    operators = [a["operator"] for a in res.plan]
    assert "PICK" in operators
    assert "PLACE" in operators
    assert "SCREW" in operators

    run_dir = out_dir / "workshop" / "W1" / "vlm"
    audit_file = run_dir / "plan_grounding_audit.json"
    assert audit_file.exists()
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    assert audit["grounding_complete"] is True
    assert audit["all_assignment_nodes_observed"] is True
    assert audit["all_required_relations_true"] is True
    assert audit["plan_uses_only_grounded_task_objects"] is True
    assert audit["plan_replay_valid"] is True
    assert len(audit["violations"]) == 0

    action_plan_file = run_dir / "action_plan.json"
    assert action_plan_file.exists()
    plan_data = json.loads(action_plan_file.read_text(encoding="utf-8"))
    assert plan_data["validation"]["status"] == "VALID"
    assert plan_data["validation"]["goal_status"] == "GOAL_SATISFIED"


def test_candidate_plan_independent_replay_validation():
    """Gate 9: Candidate plan independent replay enforces step-by-step valid transitions."""
    initial = frozenset([("hand_empty",), ("at", "fastener_01", "bench")])
    goals = frozenset([("inserted", "fastener_01", "target")])
    actions = (
        SymbolicAction(
            name="PICK",
            arguments=("fastener_01", "bench"),
            positive_preconditions=frozenset([("hand_empty",), ("at", "fastener_01", "bench")]),
            negative_preconditions=frozenset(),
            add_effects=frozenset([("holding", "fastener_01")]),
            delete_effects=frozenset([("hand_empty",), ("at", "fastener_01", "bench")]),
            cost=1,
        ),
        SymbolicAction(
            name="PLACE",
            arguments=("fastener_01", "target"),
            positive_preconditions=frozenset([("holding", "fastener_01")]),
            negative_preconditions=frozenset(),
            add_effects=frozenset([("hand_empty",), ("inserted", "fastener_01", "target")]),
            delete_effects=frozenset([("holding", "fastener_01")]),
            cost=1,
        ),
    )
    problem = SymbolicProblem(initial_atoms=initial, goal_atoms=goals, actions=actions)

    search = deterministic_astar(problem)
    assert len(search.plan) == 2

    # Valid independent replay
    val = independent_replay(problem, search.plan)
    assert val["status"] == "VALID"
    assert val["goal_status"] == "GOAL_SATISFIED"
    assert len(val["steps"]) == 2
    assert val["steps"][0]["status"] == "VALID"
    assert val["steps"][1]["status"] == "VALID"

    # Invalid sequence: skipping the PICK
    invalid_plan = (actions[1],)
    val_invalid = independent_replay(problem, invalid_plan)
    assert val_invalid["status"] == "INVALID"
    assert val_invalid["failed_step"] == 0
    assert "missing_positive_preconditions" in val_invalid["steps"][0]["failure"]


def test_incomplete_grounding_cannot_be_labeled_planning_failure(tmp_path: Path):
    """Gate 9: Incomplete grounding must never be labeled PLANNING_FAILURE."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    raw_w1 = json.loads(w1_file.read_text(encoding="utf-8")) if w1_file.exists() else {}

    run_dir = tmp_path / "mock_incomplete_run"
    run_dir.mkdir(parents=True)

    grounding = {
        "status": "INCOMPLETE",
        "complete": False,
        "assignment": {"driver": "screwdriver_01"},
        "missing_roles": ["fastener"],
        "unsatisfied_relations": [],
        "evidence": {},
    }
    (run_dir / "graph_grounding_result.json").write_text(json.dumps(grounding))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": TASK_WORKSHOP,
        "nodes": {
            "driver": {"name": "driver"},
            "fastener": {"name": "fastener"},
            "repair_target": {"name": "repair_target", "entity_kind": "FIXED_TARGET"},
        },
        "relations": [],
        "operation_groups": [],
        "metadata": {
            "canonicalization_status": "FULL",
            "required_contract_complete": True,
            "raw_role_to_canonical": {"driver": "driver", "fastener": "fastener"},
            "raw_vlm_response": raw_w1,
        },
    }))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "provider_model": "mock",
        "git_commit": "abc",
        "terminal_status": "PARTIAL_GROUNDING",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "regions_available": True,
    }))
    (run_dir / "plan_grounding_audit.json").write_text(json.dumps({
        "grounding_complete": False,
        "all_assignment_nodes_observed": True,
        "all_required_relations_true": True,
        "plan_uses_only_grounded_task_objects": True,
        "violations": ["Grounding result is not complete"],
    }))

    row = {
        "domain": "workshop",
        "variant": "W1",
        "gt_feasible": True,
        "requires_observation_recovery": False,
        "runtime_contract_complete": True,
        "canonicalization_succeeded": True,
        "regions_available": ["TOOL_CABINET"],
        "regions_inspected": [],
        "astar_invocations": 0,
        "candidate_plan_length": 0,
        "candidate_plan_status": "NO_PLAN",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "full_task_goal_coverage": 0.0,
        "full_task_satisfied": False,
        "outcome_correct": False,
        "false_completion": False,
        "search_exhausted": False,
    }

    record = enrich_record(row, run_dir, TASK_WORKSHOP)
    assert record["first_cause_category"] != "PLANNING_FAILURE"
    assert record["first_cause_category"] in ("FUNCTIONAL_ASSIGNMENT_FAILURE", "OBJECT_DISCOVERY_FAILURE")


def test_invalid_grounding_cannot_be_labeled_planning_failure(tmp_path: Path):
    """Gate 9: Invalid grounding evidence (unobserved nodes / false relations) cannot be PLANNING_FAILURE."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    raw_w1 = json.loads(w1_file.read_text(encoding="utf-8")) if w1_file.exists() else {}

    run_dir = tmp_path / "mock_invalid_grounding_run"
    run_dir.mkdir(parents=True)

    grounding = {
        "status": "COMPLETE",
        "complete": True,
        "assignment": {"driver": "screwdriver_01", "fastener": "screw_ghost"},
        "missing_roles": [],
        "unsatisfied_relations": [],
        "evidence": {},
    }
    (run_dir / "graph_grounding_result.json").write_text(json.dumps(grounding))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": TASK_WORKSHOP,
        "nodes": {
            "driver": {"name": "driver"},
            "fastener": {"name": "fastener"},
            "repair_target": {"name": "repair_target", "entity_kind": "FIXED_TARGET"},
        },
        "relations": [],
        "operation_groups": [],
        "metadata": {
            "canonicalization_status": "FULL",
            "required_contract_complete": True,
            "raw_role_to_canonical": {"driver": "driver", "fastener": "fastener"},
            "raw_vlm_response": raw_w1,
        },
    }))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "provider_model": "mock",
        "git_commit": "abc",
        "terminal_status": "ACTION_SEQUENCE_READY",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "regions_available": True,
        "astar_invocations": 1,
    }))
    # Audit explicitly notes unobserved node
    (run_dir / "plan_grounding_audit.json").write_text(json.dumps({
        "grounding_complete": True,
        "all_assignment_nodes_observed": False,
        "all_required_relations_true": True,
        "plan_uses_only_grounded_task_objects": False,
        "violations": ["Assigned object 'screw_ghost' not found in observed scene graph G_O"],
    }))

    row = {
        "domain": "workshop",
        "variant": "W1",
        "gt_feasible": True,
        "requires_observation_recovery": False,
        "runtime_contract_complete": True,
        "canonicalization_succeeded": True,
        "regions_available": ["TOOL_CABINET"],
        "regions_inspected": [],
        "astar_invocations": 1,
        "candidate_plan_length": 0,
        "candidate_plan_status": "NO_PLAN",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "full_task_goal_coverage": 0.0,
        "full_task_satisfied": False,
        "outcome_correct": False,
        "false_completion": False,
        "search_exhausted": False,
    }

    record = enrich_record(row, run_dir, TASK_WORKSHOP)
    assert record["first_cause_category"] != "PLANNING_FAILURE"
    assert record["first_cause_category"] == "FUNCTIONAL_ASSIGNMENT_FAILURE"
    assert record["failure_category"] == "INVALID_GROUNDING_EVIDENCE"


def test_true_planning_failure_attribution(tmp_path: Path):
    """Gate 9: PLANNING_FAILURE is only assigned when grounding is complete/valid but plan fails."""
    w1_file = FIXTURES_DIR / "workshop_W1.json"
    raw_w1 = json.loads(w1_file.read_text(encoding="utf-8")) if w1_file.exists() else {}

    run_dir = tmp_path / "mock_true_planning_failure_run"
    run_dir.mkdir(parents=True)

    grounding = {
        "status": "COMPLETE",
        "complete": True,
        "assignment": {"driver": "screwdriver_01", "fastener": "screw_01"},
        "missing_roles": [],
        "unsatisfied_relations": [],
        "evidence": {},
    }
    (run_dir / "graph_grounding_result.json").write_text(json.dumps(grounding))
    (run_dir / "functional_specification.json").write_text(json.dumps({
        "domain": "workshop",
        "task_instruction": TASK_WORKSHOP,
        "nodes": {
            "driver": {"name": "driver"},
            "fastener": {"name": "fastener"},
            "repair_target": {"name": "repair_target", "entity_kind": "FIXED_TARGET"},
        },
        "relations": [],
        "operation_groups": [],
        "metadata": {
            "canonicalization_status": "FULL",
            "required_contract_complete": True,
            "raw_role_to_canonical": {"driver": "driver", "fastener": "fastener"},
            "raw_vlm_response": raw_w1,
        },
    }))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "provider_model": "mock",
        "git_commit": "abc",
        "terminal_status": "NO_PLAN",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "regions_available": True,
        "astar_invocations": 1,
    }))
    # Valid grounding audit, but plan validation failed or no plan produced
    (run_dir / "plan_grounding_audit.json").write_text(json.dumps({
        "grounding_complete": True,
        "all_assignment_nodes_observed": True,
        "all_required_relations_true": True,
        "plan_uses_only_grounded_task_objects": True,
        "preparation_accessibility_valid": True,
        "violations": [],
    }))

    row = {
        "domain": "workshop",
        "variant": "W1",
        "gt_feasible": True,
        "requires_observation_recovery": False,
        "runtime_contract_complete": True,
        "canonicalization_succeeded": True,
        "regions_available": ["TOOL_CABINET"],
        "regions_inspected": [],
        "astar_invocations": 1,
        "candidate_plan_length": 0,
        "candidate_plan_status": "NO_PLAN",
        "full_task_goal_count": 3,
        "full_task_goal_satisfied_count": 0,
        "full_task_goal_coverage": 0.0,
        "full_task_satisfied": False,
        "outcome_correct": False,
        "false_completion": False,
        "search_exhausted": False,
    }

    record = enrich_record(row, run_dir, TASK_WORKSHOP)
    assert record["first_cause_category"] == "PLANNING_FAILURE"


def test_domain_planning_compilers_receive_operation_bindings():
    """Gate 9: Domain compilers inspect operation bindings from planning context."""
    compiler = WorkshopPlanningCompiler()
    spec = FunctionalSpecification(
        domain="workshop",
        task_instruction="Repair",
        nodes={"driver": FunctionalRole(name="driver"), "fastener": FunctionalRole(name="fastener")},
        relations=(
            FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener"),
            FunctionalRelation(subject_role="fastener", predicate="COMPATIBLE_WITH_TARGET", object_role="repair_target"),
            FunctionalRelation(subject_role="driver", predicate="REACHES_TARGET", object_role="repair_target"),
        ),
        operation_groups=(
            OperationGroup(
                id="fastening",
                function="Fasten screw",
                tool_role="driver",
                target_role="fastener",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=("COMPATIBLE_WITH",),
            ),
        ),
        source="VLM",
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="driver_1", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="fastener_1", entity_kind="OBJECT"))
    graph_o.add_relation(ObservedRelation(predicate="COMPATIBLE_WITH", subject_id="driver_1", object_id="fastener_1", status="TRUE"))
    graph_o.add_relation(ObservedRelation(predicate="COMPATIBLE_WITH_TARGET", subject_id="fastener_1", object_id=TARGET, status="TRUE"))
    graph_o.add_relation(ObservedRelation(predicate="REACHES_TARGET", subject_id="driver_1", object_id=TARGET, status="TRUE"))

    assignment = {"driver": "driver_1", "fastener": "fastener_1"}

    # Context with operation_bindings marked FALSE
    ctx_failed = {
        "specification": spec,
        "graph_o": graph_o,
        "operation_bindings": {
            "fastening": [{"tool_id": "driver_1", "target_id": "fastener_1", "status": "FALSE"}]
        },
    }
    prob_failed = compiler.compile_problem(assignment, ctx_failed)
    action_names_failed = [a.name for a in prob_failed.actions]
    assert "SCREW" not in action_names_failed, "SCREW action must be suppressed when operation binding fails"

    # Context with operation_bindings marked TRUE
    ctx_ok = {
        "specification": spec,
        "graph_o": graph_o,
        "operation_bindings": {
            "fastening": [{"tool_id": "driver_1", "target_id": "fastener_1", "status": "TRUE"}]
        },
    }
    prob_ok = compiler.compile_problem(assignment, ctx_ok)
    action_names_ok = [a.name for a in prob_ok.actions]
    assert "SCREW" in action_names_ok, "SCREW action must be present when operation binding is TRUE"


def test_plan_grounding_audit_covers_all_plan_used_task_objects():
    """Gate 9: Grounding audit catches ungrounded or unassigned objects used in plan."""
    spec = FunctionalSpecification(
        domain="kitchen",
        task_instruction="Prepare coffee",
        nodes={"coffee_container": FunctionalRole(name="coffee_container")},
        source="VLM",
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="mug_01", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="sponge_unassigned", entity_kind="OBJECT"))

    ground_result = MagicMock()
    ground_result.complete = True
    ground_result.assignment = {"coffee_container": "mug_01"}
    ground_result.operation_bindings = {}

    # Plan correctly uses mug_01 and countertop
    valid_plan = [
        {"operator": "PICK", "arguments": ["mug_01", "countertop"]},
        {"operator": "PLACE", "arguments": ["mug_01", "dining_table"]},
    ]
    audit_valid = audit_plan_grounding(spec, graph_o, ground_result, valid_plan, home_region="countertop")
    assert audit_valid["plan_uses_only_grounded_task_objects"] is True
    assert len(audit_valid["violations"]) == 0

    # Plan uses sponge_unassigned which is an unassigned OBJECT
    invalid_plan = [
        {"operator": "PICK", "arguments": ["sponge_unassigned", "countertop"]},
    ]
    audit_invalid = audit_plan_grounding(spec, graph_o, ground_result, invalid_plan, home_region="countertop")
    assert audit_invalid["plan_uses_only_grounded_task_objects"] is False
    assert audit_invalid["plan_replay_valid"] is False
    assert any("sponge_unassigned" in v for v in audit_invalid["violations"])
