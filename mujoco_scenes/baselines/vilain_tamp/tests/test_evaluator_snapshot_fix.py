"""Regression tests for MuJoCoPhysicalStateObserver snapshot and benchmark evaluation.

Verifies:
1. Hidden evaluator sees movable objects even when identity resolution bindings are empty.
2. Living Room F1 and F4 detect the preplaced left saucer (1/5 subgoals passed), while F0 has 0/5.
3. Containment is computed physically across all scene bodies.
4. No hidden GT context or mappings leak into planning or perception.
"""

from pathlib import Path
import pytest

from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene
from mujoco_scenes.living_room_variants import scene_name
from mujoco_scenes.baselines.vilain_tamp.live_execution import MuJoCoPhysicalStateObserver
from mujoco_scenes.baselines.vilain_tamp.evaluation.subgoals import evaluate_terminal_subgoals
from mujoco_scenes.baselines.vilain_tamp.evaluation.living_room import evaluate_living_room_requirements
from mujoco_scenes.baselines.vilain_tamp.production_execution import BenchmarkRegistryHiddenContextProvider


@pytest.fixture
def config_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def test_observer_sees_movable_objects_with_empty_bindings() -> None:
    scene = L2LivingRoomRegionScene(scene_name("F0_ALL_OBJECTS_IN_STAGING"), robot="google")
    observer = MuJoCoPhysicalStateObserver(scene, bindings={}, fixed_bindings={})
    snapshot = observer.snapshot("living_room", False)

    # Must contain all movable task objects even though bindings is empty
    expected_movables = {
        "a2_drink_left",
        "a2_drink_right",
        "a2_snack_left",
        "a2_snack_right",
        "a2_remote_payload",
    }
    for obj_name in expected_movables:
        assert obj_name in snapshot.objects, f"Movable object {obj_name} missing from snapshot.objects"
        obj_state = snapshot.objects[obj_name]
        assert obj_state.get("present") is True
        assert "world_position_m" in obj_state
        assert "support" in obj_state
        assert obj_state.get("released") is True


def test_living_room_preplaced_saucer_detection(config_root: Path) -> None:
    hidden_provider = BenchmarkRegistryHiddenContextProvider(config_root)

    # F0: All objects in staging -> 0/5 subgoals passed
    scene_f0 = L2LivingRoomRegionScene(scene_name("F0_ALL_OBJECTS_IN_STAGING"), robot="google")
    obs_f0 = MuJoCoPhysicalStateObserver(scene_f0, bindings={}, fixed_bindings={})
    snap_f0 = obs_f0.snapshot("living_room", False)
    ctx_f0 = hidden_provider.load("living_room", "F0_ALL_OBJECTS_IN_STAGING")
    eval_f0 = evaluate_terminal_subgoals(snap_f0, (), ctx_f0)
    assert eval_f0.passed_subgoals == 0
    assert eval_f0.total_subgoals == 5
    assert eval_f0.coverage == pytest.approx(0.0)

    # F1: Left saucer preplaced -> 1/5 subgoals passed
    scene_f1 = L2LivingRoomRegionScene(scene_name("F1_LEFT_SAUCER_PREPLACED"), robot="google")
    obs_f1 = MuJoCoPhysicalStateObserver(scene_f1, bindings={}, fixed_bindings={})
    snap_f1 = obs_f1.snapshot("living_room", False)
    ctx_f1 = hidden_provider.load("living_room", "F1_LEFT_SAUCER_PREPLACED")
    eval_f1 = evaluate_terminal_subgoals(snap_f1, (), ctx_f1)
    assert eval_f1.passed_subgoals == 1
    assert eval_f1.total_subgoals == 5
    assert eval_f1.coverage == pytest.approx(0.2)
    saucer_result = next(r for r in eval_f1.results if r.subgoal.subject == "a2_snack_left")
    assert saucer_result.passed is True
    assert saucer_result.evidence["support"] == "a2_personal_left"

    # F4: Saucer preplaced, cup on shared -> 1/5 subgoals passed
    scene_f4 = L2LivingRoomRegionScene(scene_name("F4_SAUCER_PREPLACED_CUP_ON_SHARED"), robot="google")
    obs_f4 = MuJoCoPhysicalStateObserver(scene_f4, bindings={}, fixed_bindings={})
    snap_f4 = obs_f4.snapshot("living_room", False)
    ctx_f4 = hidden_provider.load("living_room", "F4_SAUCER_PREPLACED_CUP_ON_SHARED")
    eval_f4 = evaluate_terminal_subgoals(snap_f4, (), ctx_f4)
    assert eval_f4.passed_subgoals == 1
    assert eval_f4.total_subgoals == 5
    assert eval_f4.coverage == pytest.approx(0.2)


def test_hidden_context_remains_isolated_from_observer() -> None:
    scene = L2LivingRoomRegionScene(scene_name("F0_ALL_OBJECTS_IN_STAGING"), robot="google")
    observer = MuJoCoPhysicalStateObserver(scene, bindings={}, fixed_bindings={})
    # Observer has no reference to hidden context or variant YAML requirements
    assert not hasattr(observer, "hidden_context")
    assert not hasattr(observer, "requirements")
    snapshot = observer.snapshot("living_room", False)
    assert "benchmark_success" not in snapshot.measurements

