import json
from pathlib import Path
import shutil

import pytest

from mujoco_scenes.living_room_symbolic_planning import (
    LivingRoomCompilationError,
    compile_living_room_problem,
    run_living_room_symbolic_pipeline,
)
from mujoco_scenes.symbolic_planning_core import deterministic_astar, independent_replay
from mujoco_scenes.living_room_variants import load_living_room_variants


ROOT = Path(__file__).parents[1] / "benchmark_reports" / "living_room_region_feasibility_phase1" / "variants"
pytestmark = pytest.mark.skipif(
    not (ROOT / "F0_ALL_OBJECTS_IN_STAGING" / "functional_region_witness.json").is_file(),
    reason="generated Living-Room Phase-1 benchmark evidence is unavailable",
)
VARIANTS = load_living_room_variants()
FEASIBLE = [name for name, spec in VARIANTS.items() if spec["intended_outcome"] == "FEASIBLE"]
INFEASIBLE = [name for name, spec in VARIANTS.items() if spec["intended_outcome"] == "INFEASIBLE"]


def _copy_variant(tmp_path, name="F0_ALL_OBJECTS_IN_STAGING"):
    target = tmp_path / name
    target.mkdir()
    for source in (ROOT / name).glob("*.json"):
        shutil.copy2(source, target / source.name)
    return target


@pytest.mark.parametrize("variant", FEASIBLE)
def test_complete_variants_compile_plan_replay_deterministically(variant):
    compiled = compile_living_room_problem(ROOT / variant)
    assert len(compiled.symbolic["objects"]) == 5
    assert len(compiled.problem.goal_atoms) == 5
    assert all(item.startswith("object_") for item in compiled.symbolic["objects"])
    assert all(item.startswith("region_") for item in compiled.symbolic["regions"])
    goals = [(atom[1], atom[2]) for atom in compiled.symbolic["goal_atoms"]]
    assert len({object_id for object_id, _ in goals}) == 5
    first = deterministic_astar(compiled.problem)
    second = deterministic_astar(compiled.problem)
    assert first.plan == second.plan
    assert len(first.plan) == (
        8 if variant in {"F1_LEFT_SAUCER_PREPLACED", "F4_SAUCER_PREPLACED_CUP_ON_SHARED"}
        else 10
    )
    assert {action.name for action in first.plan} == {"pick", "place"}
    assert independent_replay(compiled.problem, first.plan)["goal_status"] == "GOAL_SATISFIED"


@pytest.mark.parametrize("variant", INFEASIBLE)
def test_infeasible_variants_reject_before_planning(tmp_path, variant):
    output = tmp_path / "output"
    result = run_living_room_symbolic_pipeline(ROOT / variant, output)
    assert result["status"] == "REJECTED"
    assert result["reason"] == "FUNCTIONAL_WITNESS_NOT_COMPLETE"
    assert result["planner_invoked"] is False
    assert not (output / "plan.json").exists()
    assert (output / "phase1_source_manifest.json").exists()


def test_bindings_are_exact_witness_selection_including_f5():
    for variant in ("F0_ALL_OBJECTS_IN_STAGING", "F5_LEFT_PAIR_ON_SHARED"):
        witness = json.loads((ROOT / variant / "functional_region_witness.json").read_text())
        expected = sorted(
            (object_id, assignment["region_id"])
            for assignment in witness["functional_requirements"]
            for object_id in assignment["payload_ids"]
        )
        compiled = compile_living_room_problem(ROOT / variant)
        actual = sorted(
            (item["object_id"], item["region_id"])
            for item in compiled.symbolic["witness_selected_bindings"]
        )
        assert actual == expected


@pytest.mark.parametrize("corruption,reason", [
    ("unknown", "SELECTED_EDGE_NOT_TRUE"),
    ("duplicate", "DUPLICATE_PAYLOAD_BINDING"),
    ("missing", "INVALID_PAYLOAD_GROUP"),
])
def test_corrupt_complete_witness_is_rejected(tmp_path, corruption, reason):
    variant = _copy_variant(tmp_path)
    witness_path = variant / "functional_region_witness.json"
    assignments_path = variant / "region_assignments.json"
    witness = json.loads(witness_path.read_text())
    if corruption == "unknown":
        witness["functional_requirements"][0]["selected_compatibility_evidence"]["compatibility_status"] = "UNKNOWN"
    elif corruption == "duplicate":
        witness["functional_requirements"][1]["payload_ids"][0] = witness["functional_requirements"][0]["payload_ids"][0]
    else:
        witness["functional_requirements"][0]["payload_ids"].pop()
    witness_path.write_text(json.dumps(witness))
    assignments_path.write_text(json.dumps({"assignments": witness["functional_requirements"]}))
    with pytest.raises(LivingRoomCompilationError) as caught:
        compile_living_room_problem(variant)
    assert caught.value.result["reason"] == reason


def test_compiler_does_not_read_oracle(monkeypatch):
    original = Path.read_text
    def guarded(self, *args, **kwargs):
        assert "oracle" not in self.name.lower()
        assert "comparison" not in self.name.lower()
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    compile_living_room_problem(ROOT / "F0_ALL_OBJECTS_IN_STAGING")


def test_initial_state_tracks_observed_preplacement_without_fabrication():
    compiled = compile_living_room_problem(ROOT / "F0_ALL_OBJECTS_IN_STAGING")
    assert sum(atom[0] == "available" for atom in compiled.problem.initial_atoms) == 5
    assert not any(atom[0] == "on" for atom in compiled.problem.initial_atoms)
    preplaced = compile_living_room_problem(ROOT / "F1_LEFT_SAUCER_PREPLACED")
    assert sum(atom[0] == "on" for atom in preplaced.problem.initial_atoms) == 1
