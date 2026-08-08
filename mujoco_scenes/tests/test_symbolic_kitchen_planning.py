import json
from pathlib import Path

import pytest

from mujoco_scenes.symbolic_planning import (
    GroundAction,
    KitchenSymbolicProblem,
    SymbolicCompilationError,
    compile_observed_symbolic_state,
    compile_plan_and_save,
    plan_symbolic_task,
    render_domain_pddl,
    validate_symbolic_plan,
)


TASK = Path(__file__).resolve().parents[1] / "configs" / (
    "s1_integrated_kitchen_object_function.yaml"
)


def _record(object_id, label, region="INITIAL", stage=0, **extra):
    value = {
        "object_id": object_id,
        "first_seen_stage": stage,
        "last_seen_stage": stage,
        "last_evidence_stage": stage,
        "last_evidence_source_region": region,
        "source_region": "PRIVILEGED_WRONG_REGION",
        "oracle_source_region": "PRIVILEGED_WRONG_REGION",
        "measurement_cloud_path": f"stages/{stage:03d}/evidence/{object_id}/fused.ply",
        "semantics": {"validated": {
            "status": "SUPPORTED",
            "canonical_label": label,
            "mean_confidence": 0.8,
        }},
    }
    value.update(extra)
    return value


def _assignment(group, function, tool, target, valid=True):
    status = "TRUE" if valid else "FALSE"
    return {
        "function_group_id": group,
        "function": function,
        "tool_role": "coffee_stirrer" if group == "coffee_stirring" else "soup_eating_utensil",
        "target_role": "coffee_container" if group == "coffee_stirring" else "soup_container",
        "utensil_object_id": tool,
        "target_object_id": target,
        "assignment_status": status,
        "pair_geometry_status": status,
        "relation_checks": [
            {"relation": name, "status": status, "evidence": {"pass_margin_m": 0.01}}
            for name in ("INSERTABLE_IN", "REACHES_BOTTOM")
        ],
    }


def _make_run(tmp_path, prefix="object"):
    ids = {
        "coffee": [f"{prefix}_coffee_{i}" for i in range(3)],
        "soup": [f"{prefix}_soup_{i}" for i in range(3)],
        "stir": f"{prefix}_stir",
        "soup_tools": [f"{prefix}_soup_tool_{i}" for i in range(3)],
        "coffee_source": f"{prefix}_coffee_source",
        "water_source": f"{prefix}_water_source",
        "soup_source": f"{prefix}_soup_source",
    }
    records = {}
    for object_id in ids["coffee"]:
        records[object_id] = _record(object_id, "cup")
    for object_id in ids["soup"]:
        records[object_id] = _record(object_id, "bowl")
    records[ids["stir"]] = _record(ids["stir"], "spoon", "C1", 5)
    for index, object_id in enumerate(ids["soup_tools"]):
        records[object_id] = _record(object_id, "spoon", ("D2", "C2", "C1")[index], index + 2)
    records[ids["coffee_source"]] = _record(ids["coffee_source"], "coffee_source")
    records[ids["water_source"]] = _record(ids["water_source"], "kettle")
    records[ids["soup_source"]] = _record(ids["soup_source"], "soup_source")
    assignments = [
        _assignment("coffee_stirring", "STIR_COFFEE", ids["stir"], target)
        for target in ids["coffee"]
    ] + [
        _assignment("soup_serving", "PROVIDE_SOUP_EATING_UTENSIL", tool, target)
        for tool, target in zip(ids["soup_tools"], ids["soup"])
    ]
    witness = {
        "status": "COMPLETE",
        "stage": 5,
        "selected_witness": {
            "coffee_container": ids["coffee"],
            "soup_container": ids["soup"],
            "coffee_stirrer": [ids["stir"]],
            "soup_eating_utensil": ids["soup_tools"],
        },
        "operation_assignments": assignments,
    }
    graph = {"nodes": [
        {"type": "region", "attributes": {"region_id": region, "open": region != "C2", "inspected": True}}
        for region in ("countertop", "D1", "D2", "C2", "B1", "C1")
    ], "edges": []}
    (tmp_path / "object_registry.json").write_text(json.dumps({"objects": records}))
    (tmp_path / "observed_graph.json").write_text(json.dumps(graph))
    (tmp_path / "latest_witness.json").write_text(json.dumps(witness))
    (tmp_path / "events.jsonl").write_text("\n".join(
        json.dumps({"stage": i + 1, "event": "REGION_OPENED", "region_id": region})
        for i, region in enumerate(("D1", "D2", "C2", "B1", "C1"))
    ) + "\n")
    return ids, records, witness


def test_unobserved_object_is_excluded_and_location_uses_evidence(tmp_path):
    ids, records, _ = _make_run(tmp_path)
    # Hidden/uninspected objects are absent from the observed registry even if
    # an oracle evaluation fixture knows their physical name/location.
    (tmp_path / "object_registry.json").write_text(json.dumps({"objects": records}))
    compiled = compile_observed_symbolic_state(tmp_path, TASK)
    assert "hidden_never_seen" not in compiled["objects"]
    assert compiled["objects"][ids["stir"]]["location"]["region_id"] == "C1"
    assert compiled["objects"][ids["coffee_source"]]["location"]["region_id"] == "countertop"
    assert compiled["location_basis"].endswith("EVIDENCE_ONLY")


def test_missing_stage_local_location_is_rejected_even_with_oracle(tmp_path):
    ids, records, _ = _make_run(tmp_path)
    del records[ids["stir"]]["last_evidence_source_region"]
    records[ids["stir"]]["source_region"] = "C1"
    (tmp_path / "object_registry.json").write_text(json.dumps({"objects": records}))
    with pytest.raises(SymbolicCompilationError, match="stage-local location"):
        compile_observed_symbolic_state(tmp_path, TASK)


def test_compilation_is_generic_across_persistent_id_names(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir(); second.mkdir()
    _make_run(first, "alpha")
    _make_run(second, "renumbered")
    a = compile_observed_symbolic_state(first, TASK)
    b = compile_observed_symbolic_state(second, TASK)
    assert len(a["objects"]) == len(b["objects"])
    assert len(a["capabilities"]["can_stir"]) == len(b["capabilities"]["can_stir"]) == 3
    assert len(a["capabilities"]["assigned_soup_utensil"]) == 3


def test_coffee_reuse_soup_distinctness_and_plan_validation(tmp_path):
    ids, _, _ = _make_run(tmp_path)
    result = compile_plan_and_save(tmp_path, TASK)
    compiled = result["compiled"]
    assert {tool for tool, _ in map(tuple, compiled["capabilities"]["can_stir"])} == {ids["stir"]}
    assert len({tool for tool, _ in map(tuple, compiled["capabilities"]["assigned_soup_utensil"])}) == 3
    assert result["validation"]["valid"]
    assert result["validation"]["all_goals_satisfied"]
    assert result["validation"]["coffee_reuse_verified"]
    assert result["validation"]["soup_distinctness_verified"]
    assert compiled["role_assignments"]["source_roles"].keys() == {
        "coffee_source", "water_source"
    }
    assert {
        tuple(item)
        for item in compiled["capabilities"]["initial_target_contents"]
    } == {(target, "soup") for target in ids["soup"]}
    assert not any(
        step["action"] == "pour" and step["arguments"][-1] == "soup"
        for step in result["plan"]
    )
    assert (tmp_path / "domain.pddl").exists()
    assert (tmp_path / "problem.pddl").exists()
    assert (tmp_path / "planner_provenance.json").exists()
    assert (tmp_path / "scientific_validation.json").exists()
    provenance = json.loads((tmp_path / "planner_provenance.json").read_text())
    assert provenance["planner_entry_point"].endswith("plan_symbolic_task")
    assert provenance["plan_renderer_role"].startswith("serialization_only")
    domain = (tmp_path / "domain.pddl").read_text()
    assert domain.count("(:action ") == 4
    assert all(f"(:action {name}" in domain for name in ("pick", "place", "pour", "stir"))
    assert all(token not in domain for token in ("serve_", "place_serving", "(:action open", "(:action close"))
    assert "PLAN VALID" in (tmp_path / "combined_action_sequence.txt").read_text()
    problem_text = (tmp_path / "problem.pddl").read_text()
    assert all(
        f"(contains {target} soup)" in problem_text
        for target in ids["soup"]
    )


def test_symbolic_compiler_accepts_collective_multi_tool_coffee_cover(tmp_path):
    ids, records, witness = _make_run(tmp_path)
    second_tool = "object_second_stirrer"
    records[second_tool] = _record(second_tool, "spoon", "D2", 2)
    witness["operation_assignments"][2] = _assignment(
        "coffee_stirring", "STIR_COFFEE",
        second_tool, ids["coffee"][2],
    )
    witness["selected_witness"]["coffee_stirrer"] = [
        ids["stir"], second_tool,
    ]
    (tmp_path / "object_registry.json").write_text(
        json.dumps({"objects": records})
    )
    (tmp_path / "latest_witness.json").write_text(json.dumps(witness))

    result = compile_plan_and_save(tmp_path, TASK)

    assert result["validation"]["valid"]
    assert result["validation"]["all_goals_satisfied"]
    assert not result["validation"]["coffee_reuse_verified"]
    assert {tool for tool, _ in map(
        tuple, result["compiled"]["capabilities"]["can_stir"]
    )} == {ids["stir"], second_tool}


def test_failed_pair_geometry_never_becomes_capability(tmp_path):
    ids, _, witness = _make_run(tmp_path)
    witness["operation_assignments"][0] = _assignment(
        "coffee_stirring", "STIR_COFFEE", ids["stir"], ids["coffee"][0], valid=False
    )
    (tmp_path / "latest_witness.json").write_text(json.dumps(witness))
    with pytest.raises(SymbolicCompilationError, match="do not cover"):
        compile_observed_symbolic_state(tmp_path, TASK)


def test_no_robot_or_fm_runtime_dependency_in_symbolic_boundary():
    source = Path(__file__).resolve().parents[1].joinpath("symbolic_planning.py").read_text()
    assert "import mujoco" not in source
    assert "openai" not in source.lower()
    assert "anthropic" not in source.lower()
    runner = Path(__file__).resolve().parents[1].joinpath("run_kitchen_symbolic_pipeline.py").read_text()
    assert "KitchenScene" not in runner
    assert "ground_symbolic_sources" not in runner
    assert "semantic_detector" not in runner


def _problem(tmp_path):
    _make_run(tmp_path)
    return KitchenSymbolicProblem(compile_observed_symbolic_state(tmp_path, TASK))


def test_noncomplete_witness_is_rejected(tmp_path):
    _, _, witness = _make_run(tmp_path)
    witness["status"] = "INDETERMINATE"
    (tmp_path / "latest_witness.json").write_text(json.dumps(witness))
    with pytest.raises(SymbolicCompilationError, match="not COMPLETE"):
        compile_observed_symbolic_state(tmp_path, TASK)


def test_oracle_source_binding_payload_is_rejected(tmp_path):
    _make_run(tmp_path)
    (tmp_path / "symbolic_source_semantics.json").write_text(json.dumps({
        "inference_basis": "ORACLE_GROUND_TRUTH",
        "objects": {},
    }))
    with pytest.raises(SymbolicCompilationError, match="frozen RGB"):
        compile_observed_symbolic_state(tmp_path, TASK)


def test_exactly_four_generic_operator_types():
    assert KitchenSymbolicProblem.OPERATOR_TYPES == {"pick", "place", "pour", "stir"}
    domain = render_domain_pddl()
    assert domain.count("(:action ") == 4
    assert "(:action serve" not in domain.lower()
    assert "pour_coffee" not in domain.lower()
    assert "place_soup" not in domain.lower()


def test_pick_and_generic_place_preconditions_and_effects(tmp_path):
    problem = _problem(tmp_path)
    source = next(iter(sorted(problem.source_contents)))
    picked = problem.apply(problem.initial, GroundAction("pick", (source,)))
    assert picked.held == source and source not in picked.location_map()
    with pytest.raises(ValueError, match="PICK"):
        problem.apply(picked, GroundAction("pick", (source,)))
    placed = problem.apply(picked, GroundAction("place", (source, problem.home)))
    assert placed.held is None and placed.location_map()[source] == problem.home
    with pytest.raises(ValueError, match="PLACE"):
        problem.apply(problem.initial, GroundAction("place", (source, problem.home)))


def test_one_place_operator_handles_all_three_destination_contexts(tmp_path):
    ids, _, _ = _make_run(tmp_path)
    result = compile_plan_and_save(tmp_path, TASK)
    placements = {
        tuple(step["arguments"])
        for step in result["plan"] if step["action"] == "place"
    }
    problem = KitchenSymbolicProblem(result["compiled"])
    assert any(destination == problem.home for _, destination in placements)
    assert any(destination == problem.serving_destination for _, destination in placements)
    assert all((tool, target) in placements for tool, target in problem.soup_assignments)
    assert {step["action"] for step in result["plan"]} <= problem.OPERATOR_TYPES


def test_generic_pour_derives_content_from_source(tmp_path):
    problem = _problem(tmp_path)
    by_content = {content: source for source, content in problem.source_contents.items()}
    target = sorted(problem.coffee_targets)[0]
    state = problem.apply(problem.initial, GroundAction("pick", (by_content["coffee"],)))
    state = problem.apply(state, GroundAction("pour", (by_content["coffee"], target)))
    assert (target, "coffee") in state.contents
    assert (target, "water") not in state.contents
    with pytest.raises(ValueError, match="POUR"):
        problem.apply(state, GroundAction("pour", (by_content["coffee"], sorted(problem.soup_targets)[0])))


def test_stir_requires_binding_and_both_contents(tmp_path):
    problem = _problem(tmp_path)
    tool, target = sorted(problem.can_stir)[0]
    state = problem.apply(problem.initial, GroundAction("pick", (tool,)))
    with pytest.raises(ValueError, match="STIR"):
        problem.apply(state, GroundAction("stir", (tool, target)))
    ready = state.__class__(
        locations=state.locations,
        held=state.held,
        contents=state.contents | {(target, "coffee"), (target, "water")},
        stirred=state.stirred,
    )
    stirred = problem.apply(ready, GroundAction("stir", (tool, target)))
    assert target in stirred.stirred
    incompatible = sorted(problem.soup_targets)[0]
    with pytest.raises(ValueError, match="STIR"):
        problem.apply(ready, GroundAction("stir", (tool, incompatible)))


def test_validator_rejects_corrupted_order_and_binding(tmp_path):
    problem = _problem(tmp_path)
    tool, target = sorted(problem.can_stir)[0]
    invalid = validate_symbolic_plan(problem, [GroundAction("stir", (tool, target))])
    assert not invalid["plan_valid"]
    assert invalid["failed_step"] == 1
    assert f"holding({tool})" in invalid["failed_preconditions"]


def test_planning_is_deterministic_and_goal_omission_is_detected(tmp_path):
    problem = _problem(tmp_path)
    first = plan_symbolic_task(problem)
    second = plan_symbolic_task(problem)
    assert first == second
    valid = validate_symbolic_plan(problem, first)
    assert valid["plan_valid"]
    omitted = validate_symbolic_plan(problem, first[:-2])
    assert omitted["all_actions_applicable"]
    assert not omitted["final_goal_satisfied"]


def test_planner_reports_no_plan_for_impossible_symbolic_problem(tmp_path):
    problem = _problem(tmp_path)
    problem.source_contents = {}
    with pytest.raises(SymbolicCompilationError, match="no valid plan"):
        plan_symbolic_task(problem)


@pytest.mark.parametrize("tool_count", [1, 2, 3])
def test_planner_supports_one_two_or_three_coffee_tools(tmp_path, tool_count):
    ids, records, witness = _make_run(tmp_path)
    tools = [ids["stir"]]
    for index in range(1, tool_count):
        tool = f"object_extra_stir_{index}"
        tools.append(tool)
        records[tool] = _record(tool, "spoon", "D1", 1)
    coffee_assignments = []
    for index, target in enumerate(ids["coffee"]):
        coffee_assignments.append(_assignment(
            "coffee_stirring", "STIR_COFFEE",
            tools[min(index, tool_count - 1)], target,
        ))
    witness["operation_assignments"] = coffee_assignments + [
        item for item in witness["operation_assignments"]
        if item["function_group_id"] == "soup_serving"
    ]
    witness["selected_witness"]["coffee_stirrer"] = tools
    (tmp_path / "object_registry.json").write_text(json.dumps({"objects": records}))
    (tmp_path / "latest_witness.json").write_text(json.dumps(witness))
    result = compile_plan_and_save(tmp_path, TASK)
    assert result["validation"]["plan_valid"]
    assert result["validation"]["coffee_distinct_tool_count"] == tool_count
    assert {step["action"] for step in result["plan"]} <= KitchenSymbolicProblem.OPERATOR_TYPES
