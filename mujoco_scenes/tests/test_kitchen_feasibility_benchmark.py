import inspect
from pathlib import Path

from mujoco_scenes.kitchen_feasibility_oracle import (
    evaluate_all_oracle_variants,
    load_feasibility_benchmark_config,
)
from mujoco_scenes.run_kitchen_feasibility_benchmark import (
    predicted_feasibility_from_witness,
    run_predicted_variant,
)
from mujoco_scenes.scene_loader import load_all_configs
from mujoco_scenes.task_witness import evaluate_usage_policy_task_witness


TASK = Path(__file__).resolve().parents[1] / "configs" / (
    "s1_integrated_kitchen_object_function.yaml"
)
COFFEE = ("coffee_1", "coffee_2", "coffee_3")
SOUP = ("soup_1", "soup_2", "soup_3")


def _node(object_id, label):
    return {
        "id": f"object:{object_id}",
        "type": "object",
        "attributes": {
            "object_id": object_id,
            "semantics": {"validated": {
                "status": "SUPPORTED",
                "canonical_label": label,
                "mean_confidence": 0.8,
            }},
        },
    }


def _unary(object_id, role, status):
    return {
        "source": f"object:{object_id}",
        "target": f"role:{role}",
        "relation": "SATISFIES_GEOMETRY",
        "status": status,
        "evidence": {"checks": [{"status": status}]},
    }


def _relation(tool, target, predicate, status):
    return {
        "source": f"object:{tool}",
        "target": f"object:{target}",
        "relation": predicate,
        "status": status,
        "evidence": {"pass_margin_m": 0.01 if status == "TRUE" else -0.01},
    }


def _assignment_graph(coffee_edges):
    coffee_tools = sorted(coffee_edges)
    soup_tools = ["soup_tool_1", "soup_tool_2", "soup_tool_3"]
    tools = coffee_tools + soup_tools
    labels = {
        **{target: "cup" for target in COFFEE},
        **{target: "bowl" for target in SOUP},
        **{tool: "spoon" for tool in tools},
    }
    nodes = [_node(object_id, label) for object_id, label in labels.items()]
    edges = []
    roles = (
        "coffee_container", "soup_container",
        "coffee_stirrer", "soup_eating_utensil",
    )
    for object_id in labels:
        for role in roles:
            target_role = role.endswith("container")
            status = "TRUE" if (object_id in COFFEE + SOUP) == target_role else "FALSE"
            edges.append(_unary(object_id, role, status))
    for tool in tools:
        for target in COFFEE + SOUP:
            valid = (
                target in coffee_edges.get(tool, set())
                or (
                    tool.startswith("soup_tool_")
                    and target == SOUP[int(tool[-1]) - 1]
                )
            )
            for predicate in ("INSERTABLE_IN", "REACHES_BOTTOM"):
                edges.append(_relation(
                    tool, target, predicate, "TRUE" if valid else "FALSE"
                ))
    return {"stage": 0, "nodes": nodes, "edges": edges}


def _coffee_group(result):
    return next(
        group for group in result["function_group_evaluations"]
        if group["function_group_id"] == "coffee_stirring"
    )


def _evaluate(coffee_edges):
    return evaluate_usage_policy_task_witness(
        _assignment_graph(coffee_edges), TASK
    )


def test_coffee_one_tool_cover_is_preferred_over_multi_tool_alternatives():
    result = _evaluate({
        "universal": set(COFFEE),
        "a": {COFFEE[0], COFFEE[1]},
        "b": {COFFEE[2]},
    })
    group = _coffee_group(result)
    assert result["status"] == "COMPLETE"
    assert group["minimum_distinct_tool_count"] == 1
    assert {a["utensil_object_id"] for a in group["selected_assignments"]} == {
        "universal"
    }


def test_coffee_two_tool_collective_cover_is_feasible_and_optimal():
    result = _evaluate({
        "a": {COFFEE[0], COFFEE[1]},
        "b": {COFFEE[2]},
        "c": {COFFEE[0]},
    })
    group = _coffee_group(result)
    assert result["status"] == "COMPLETE"
    assert group["minimum_distinct_tool_count"] == 2
    assert len({a["utensil_object_id"] for a in group["selected_assignments"]}) == 2


def test_coffee_three_tool_collective_cover_is_feasible():
    result = _evaluate({
        "a": {COFFEE[0]}, "b": {COFFEE[1]}, "c": {COFFEE[2]},
    })
    group = _coffee_group(result)
    assert result["status"] == "COMPLETE"
    assert group["minimum_distinct_tool_count"] == 3


def test_uncovered_coffee_target_is_infeasible():
    result = _evaluate({"a": {COFFEE[0]}, "b": {COFFEE[1]}})
    assert result["status"] == "INCOMPLETE"
    assert not _coffee_group(result)["complete_target_coverage_exists"]


def test_oracle_curated_variants_and_stage_labels_match_intent():
    results = evaluate_all_oracle_variants()
    assert results["F0_REUSE_ONE"]["oracle_coffee_minimum_unique_tools"] == 1
    assert results["F2_DISTRIBUTED_COFFEE_TWO"]["oracle_coffee_minimum_unique_tools"] == 2
    assert results["F3_DISTRIBUTED_COFFEE_THREE"]["oracle_coffee_minimum_unique_tools"] == 3
    assert results["F1_INITIAL_COMPLETE"]["oracle_earliest_feasible_stage"] == "INITIAL"
    assert results["F4_EARLY_RELOCATION"]["oracle_earliest_feasible_stage"] == "D2"
    assert results["F5_LATE_RELOCATION"]["oracle_earliest_feasible_stage"] == "C1"
    assert results["I3_ONLY_TWO_SOUP_TOOLS"]["oracle_soup_matching_size"] == 2
    assert results["I4_SOUP_MATCHING_TRAP"]["oracle_failure_reason"] == "NO_COMPLETE_SOUP_MATCHING"


def test_every_variant_has_byte_identical_goal_instruction():
    benchmark = load_feasibility_benchmark_config()
    scenes = load_all_configs()
    assert all(
        scenes[variant["scene_name"]].goal == benchmark["goal_instruction"]
        for variant in benchmark["variants"].values()
    )


def test_terminal_mapping_is_binary_and_early_complete_has_no_inspections():
    incomplete = predicted_feasibility_from_witness(
        "fixture", "goal", {"status": "INDETERMINATE", "stage": 5},
        inspection_count=5,
    )
    complete = predicted_feasibility_from_witness(
        "fixture", "goal", {
            "status": "COMPLETE", "stage": 0,
            "function_group_evaluations": [],
        }, inspection_count=0,
    )
    assert incomplete["terminal_outcome"] == "INFEASIBLE"
    assert complete["terminal_outcome"] == "FEASIBLE"
    assert complete["completion_stage"] == "INITIAL"
    assert complete["inspection_count"] == 0


def test_predicted_runner_has_no_oracle_or_action_planner_dependency():
    signature = inspect.signature(run_predicted_variant)
    assert "oracle" not in signature.parameters
    source = Path(inspect.getsourcefile(run_predicted_variant)).read_text()
    assert "symbolic_planning" not in source
    assert "domain.pddl" not in source
    assert "problem.pddl" not in source
