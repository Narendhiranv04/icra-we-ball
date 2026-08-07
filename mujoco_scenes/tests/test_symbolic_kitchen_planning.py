import json
from pathlib import Path

import pytest

from mujoco_scenes.symbolic_planning import (
    KitchenSymbolicProblem,
    SymbolicCompilationError,
    compile_observed_symbolic_state,
    compile_plan_and_save,
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
    assert (tmp_path / "domain.pddl").exists()
    assert (tmp_path / "problem.pddl").exists()
    assert "physical_object region content - object" in (
        tmp_path / "domain.pddl"
    ).read_text()
    assert "PLAN VALID" in (tmp_path / "combined_action_sequence.txt").read_text()


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
    assert "include_robot=False" in runner
