"""Unit tests for Stage 6: Planner Action Provenance and Goal Compilation.

Verifies:
1. Roles without explicit operation do not create task operation goals (Section 13.1, 13.2, 13.4).
2. Explicit operation enables corresponding planner realization (POUR, STIR, SCREW, RETURN).
3. Planner still generates valid low-level primitives (PICK, PLACE, POUR, STIR, SCREW).
4. One A* only sequencing with independent symbolic replay validation (Section 13.5).
5. State transitions validated through independent replay.
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
)
from mujoco_scenes.functional_tamp_pipeline.planning import (
    plan_with_common_astar,
    PlannedSequence,
)
from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import KitchenPlanningCompiler
from mujoco_scenes.functional_tamp_pipeline.domains.workshop import WorkshopPlanningCompiler
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedSceneGraph,
    ObservedNode,
    ObservedRelation,
)


def test_kitchen_roles_only_synthesizes_no_operation_actions_or_goals():
    """Gate 6: Kitchen roles without explicit operation do NOT generate POUR or STIR actions or goals."""
    compiler = KitchenPlanningCompiler()
    spec = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Prepare beverage",
        nodes={
            "coffee_source": FunctionalRole(name="coffee_source", entity_kind="OBJECT", semantic_categories=("coffee",)),
            "coffee_container": FunctionalRole(name="coffee_container", entity_kind="OBJECT", semantic_categories=("cup",)),
            "coffee_stirrer": FunctionalRole(name="coffee_stirrer", entity_kind="OBJECT", semantic_categories=("spoon",)),
        },
        relations=(),
        operation_groups=(),  # ZERO explicit operations
        source="VLM",
        metadata={"online_executable_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="coffee_jar_1", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="spoon_1", entity_kind="OBJECT"))

    assignment = {
        "coffee_source": "coffee_jar_1",
        "coffee_container": "cup_1",
        "coffee_stirrer": "spoon_1",
    }
    context = {
        "specification": spec,
        "graph_o": graph_o,
        "operation_bindings": {},
    }

    problem = compiler.compile_problem(assignment, context)
    action_names = {a.name for a in problem.actions}

    # POUR and STIR must NOT be generated from role pairs alone
    assert "POUR" not in action_names, "POUR action must not be synthesized from role pairs alone"
    assert "STIR" not in action_names, "STIR action must not be synthesized from role pairs alone"

    # Goal atoms must NOT contain transfer or stirring goals
    assert ("contains", "cup_1", "coffee") not in problem.goal_atoms
    assert ("contains", "cup_1", "water") not in problem.goal_atoms
    assert ("stirred", "cup_1") not in problem.goal_atoms


def test_kitchen_explicit_transfer_enables_pour_action_and_goal():
    """Gate 6: Explicit TRANSFER_CONTENT_TO_CONTAINER operation enables POUR action and goal."""
    compiler = KitchenPlanningCompiler()
    spec = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Pour coffee into cup",
        nodes={
            "coffee_source": FunctionalRole(name="coffee_source", entity_kind="OBJECT", semantic_categories=("coffee",)),
            "coffee_container": FunctionalRole(name="coffee_container", entity_kind="OBJECT", semantic_categories=("cup",)),
        },
        relations=(),
        operation_groups=(
            OperationGroup(
                id="op_pour_coffee",
                function="pour coffee into cup",
                tool_role="coffee_source",
                target_role="coffee_container",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=(),
                capability_id="TRANSFER_CONTENT_TO_CONTAINER",
            ),
        ),
        source="VLM",
        metadata={"online_executable_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="coffee_jar_1", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT"))

    assignment = {
        "coffee_source": "coffee_jar_1",
        "coffee_container": "cup_1",
    }
    context = {
        "specification": spec,
        "graph_o": graph_o,
        "operation_bindings": {
            "op_pour_coffee": [{"tool_id": "coffee_jar_1", "target_id": "cup_1", "status": "TRUE"}],
        },
    }

    problem = compiler.compile_problem(assignment, context)
    action_names = {a.name for a in problem.actions}

    assert "POUR" in action_names, "POUR action must be enabled by explicit transfer operation"
    assert ("contains", "cup_1", "coffee") in problem.goal_atoms

    # Plan with common A* and verify independent replay
    planned = plan_with_common_astar(compiler, assignment, context)
    assert planned.validation["status"] == "VALID"
    operators = [a["operator"] for a in planned.actions]
    assert "POUR" in operators
    assert "PICK" in operators


def test_kitchen_explicit_stir_enables_stir_action_and_goal():
    """Gate 6: Explicit MIX_BEVERAGE_CONTENTS operation enables STIR action and goal."""
    compiler = KitchenPlanningCompiler()
    spec = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Stir beverage in cup",
        nodes={
            "coffee_stirrer": FunctionalRole(name="coffee_stirrer", entity_kind="OBJECT", semantic_categories=("spoon",)),
            "coffee_container": FunctionalRole(name="coffee_container", entity_kind="OBJECT", semantic_categories=("cup",)),
        },
        relations=(),
        operation_groups=(
            OperationGroup(
                id="op_stir_beverage",
                function="stir beverage in cup",
                tool_role="coffee_stirrer",
                target_role="coffee_container",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=(),
                capability_id="MIX_BEVERAGE_CONTENTS",
            ),
        ),
        source="VLM",
        metadata={"online_executable_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="spoon_1", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT"))

    assignment = {
        "coffee_stirrer": "spoon_1",
        "coffee_container": "cup_1",
    }
    context = {
        "specification": spec,
        "graph_o": graph_o,
        "operation_bindings": {
            "op_stir_beverage": [{"tool_id": "spoon_1", "target_id": "cup_1", "status": "TRUE"}],
        },
    }

    problem = compiler.compile_problem(assignment, context)
    action_names = {a.name for a in problem.actions}

    assert "STIR" in action_names, "STIR action must be enabled by explicit stir operation"
    assert ("stirred", "cup_1") in problem.goal_atoms

    # Plan with common A* and verify independent replay
    planned = plan_with_common_astar(compiler, assignment, context)
    assert planned.validation["status"] == "VALID"
    operators = [a["operator"] for a in planned.actions]
    assert "STIR" in operators


def test_workshop_roles_only_synthesizes_no_fastening_or_return_goals():
    """Gate 6: Workshop roles without explicit operation do NOT create SCREW action or return goals."""
    compiler = WorkshopPlanningCompiler()
    spec = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Stage parts",
        nodes={
            "driver": FunctionalRole(name="driver", entity_kind="OBJECT", semantic_categories=("screwdriver",)),
            "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", semantic_categories=("screw",)),
        },
        relations=(),
        operation_groups=(),  # ZERO explicit operations
        source="VLM",
        metadata={"online_executable_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="screwdriver_1", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="bolt_1", entity_kind="OBJECT"))

    assignment = {
        "driver": "screwdriver_1",
        "fastener": "bolt_1",
    }
    context = {
        "specification": spec,
        "graph_o": graph_o,
        "target_joint": "joint_target",
        "work_surface": "workbench",
        "operation_bindings": {},
    }

    problem = compiler.compile_problem(assignment, context)
    action_names = {a.name for a in problem.actions}

    assert "SCREW" not in action_names, "SCREW action must not be synthesized without explicit operation"
    assert ("repaired", "joint_target") not in problem.goal_atoms
    assert ("at", "screwdriver_1", "workbench") not in problem.goal_atoms


def test_workshop_explicit_fasten_enables_screw_and_valid_primitives():
    """Gate 6: Explicit FASTEN_JOINT operation enables SCREW and generates valid low-level plan."""
    compiler = WorkshopPlanningCompiler()
    spec = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Fasten joint and return tool",
        nodes={
            "driver": FunctionalRole(name="driver", entity_kind="OBJECT", semantic_categories=("screwdriver",)),
            "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", semantic_categories=("screw",)),
        },
        relations=(
            FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener"),
            FunctionalRelation(subject_role="fastener", predicate="COMPATIBLE_WITH_TARGET", object_role="repair_target"),
            FunctionalRelation(subject_role="driver", predicate="REACHES_TARGET", object_role="repair_target"),
        ),
        operation_groups=(
            OperationGroup(
                id="op_fasten",
                function="fasten bolt",
                tool_role="driver",
                target_role="fastener",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=("COMPATIBLE_WITH",),
                capability_id="FASTEN_JOINT",
            ),
            OperationGroup(
                id="op_return",
                function="return driver to workbench",
                tool_role="driver",
                target_role="workbench",
                required_target_count=1,
                usage_policy="SEQUENTIAL_REUSE_ALLOWED",
                required_relations=(),
                capability_id="RETURN_REUSABLE_ITEM_TO_SUPPORT",
            ),
        ),
        source="VLM",
        metadata={"online_executable_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="screwdriver_1", entity_kind="OBJECT"))
    graph_o.add_node(ObservedNode(instance_id="bolt_1", entity_kind="OBJECT"))
    graph_o.add_relation(ObservedRelation(predicate="COMPATIBLE_WITH", subject_id="screwdriver_1", object_id="bolt_1", status="TRUE"))
    graph_o.add_relation(ObservedRelation(predicate="COMPATIBLE_WITH_TARGET", subject_id="bolt_1", object_id="joint_target", status="TRUE"))
    graph_o.add_relation(ObservedRelation(predicate="REACHES_TARGET", subject_id="screwdriver_1", object_id="joint_target", status="TRUE"))

    assignment = {
        "driver": "screwdriver_1",
        "fastener": "bolt_1",
    }
    context = {
        "specification": spec,
        "graph_o": graph_o,
        "target_joint": "joint_target",
        "work_surface": "workbench",
        "operation_bindings": {
            "op_fasten": [{"tool_id": "screwdriver_1", "target_id": "bolt_1", "status": "TRUE"}],
            "op_return": [{"tool_id": "screwdriver_1", "target_id": "workbench", "status": "TRUE"}],
        },
    }

    problem = compiler.compile_problem(assignment, context)
    action_names = {a.name for a in problem.actions}

    assert "SCREW" in action_names
    assert "PICK" in action_names
    assert "PLACE" in action_names
    assert ("repaired", "joint_target") in problem.goal_atoms
    assert ("at", "screwdriver_1", "workbench") in problem.goal_atoms

    # Exactly one A* search and independent replay
    planned = plan_with_common_astar(compiler, assignment, context)
    assert planned.validation["status"] == "VALID"
    operators = [a["operator"] for a in planned.actions]
    assert operators == ["PICK", "PLACE", "PICK", "SCREW", "PLACE"]
