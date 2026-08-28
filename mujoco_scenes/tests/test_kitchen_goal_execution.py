import json
from pathlib import Path

import pytest

from mujoco_scenes import kitchen_execution_bundle as bundle_module
from mujoco_scenes.kitchen_execution_bundle import (
    KitchenExecutionBundleError,
    build_kitchen_execution_bundle,
)
from mujoco_scenes.run_kitchen_goal_execution import (
    GoalContractError,
    initial_observation_images,
    validate_goal_contract,
)


def test_goal_contract_requires_exact_scope_and_fm_functions():
    goal = "Prepare coffee for three people"
    task = {
        "goal_instruction": goal,
        "execution_goal_contract": {
            "required_functions": ["can_hold_liquid", "can_stir"]
        },
    }
    response = {
        "decomposition": {
            "status": "DECOMPOSED",
            "functional_requirements": [
                {"function": "can_hold_liquid"},
                {"function": "can_stir"},
            ],
        }
    }
    assert validate_goal_contract(goal, task, response)["status"] == "DECOMPOSED"
    with pytest.raises(GoalContractError, match="does not match"):
        validate_goal_contract("Prepare one coffee", task, response)
    response["decomposition"]["functional_requirements"].pop()
    with pytest.raises(GoalContractError, match="can_stir"):
        validate_goal_contract(goal, task, response)


def test_initial_observation_images_are_camera_labelled(tmp_path):
    camera = tmp_path / "stages/000_initial/cameras/front/rgb.png"
    camera.parent.mkdir(parents=True)
    camera.write_bytes(b"png")
    assert initial_observation_images(tmp_path) == [("front", camera)]


def test_execution_bundle_rejects_incomplete_witness_before_scene_creation(
    tmp_path,
):
    for name, payload in {
        "object_registry.json": {"objects": {}},
        "observed_graph.json": {"nodes": []},
        "latest_witness.json": {
            "status": "INDETERMINATE",
            "reason_codes": ["REQUIRED_EVIDENCE_UNKNOWN"],
        },
    }.items():
        (tmp_path / name).write_text(json.dumps(payload))
    with pytest.raises(KitchenExecutionBundleError, match="not COMPLETE"):
        build_kitchen_execution_bundle(
            tmp_path,
            output_dir=tmp_path / "execution",
            scene_factory=lambda *args, **kwargs: pytest.fail("scene created"),
        )


def test_execution_bundle_rejects_non_object_artifact(tmp_path):
    (tmp_path / "object_registry.json").write_text("[]", encoding="utf-8")
    (tmp_path / "observed_graph.json").write_text("{}", encoding="utf-8")
    (tmp_path / "latest_witness.json").write_text(
        '{"status": "COMPLETE"}', encoding="utf-8"
    )
    with pytest.raises(KitchenExecutionBundleError, match="JSON object"):
        build_kitchen_execution_bundle(
            tmp_path,
            output_dir=tmp_path / "execution",
            scene_factory=lambda *args, **kwargs: pytest.fail("scene created"),
        )


def test_execution_bundle_writes_planner_and_resolution_boundary(
    tmp_path, monkeypatch
):
    phase1 = tmp_path / "phase1"
    output = tmp_path / "execution"
    phase1.mkdir()
    registry = {
        "scene_name": "test_scene",
        "objects": {"object_1": {"object_id": "object_1"}},
    }
    witness = {"status": "COMPLETE"}
    for name, payload in {
        "object_registry.json": registry,
        "observed_graph.json": {"nodes": []},
        "latest_witness.json": witness,
    }.items():
        (phase1 / name).write_text(json.dumps(payload))
    plan = [{"step": 1, "action": "pick", "arguments": ["object_1"]}]
    symbolic = {
        "compiled": {"role_assignments": {"coffee_targets": ["object_1"]}},
        "plan": plan,
    }
    inventory = {
        "scene_name": "test_scene",
        "objects": [
            {
                "generic_object_id": "object_1",
                "source_context": {"source_container": "D1"},
            }
        ],
    }
    monkeypatch.setattr(
        bundle_module,
        "compile_plan_and_save",
        lambda *args, **kwargs: symbolic,
    )
    monkeypatch.setattr(
        bundle_module,
        "build_phase_b_inventory",
        lambda *args, **kwargs: inventory,
    )

    class Resolver:
        def candidates_from_scene(self, scene, *, observed_regions):
            assert observed_regions == {"D1"}
            return ["candidate"]

        def resolve(self, received_inventory, candidates):
            assert received_inventory is inventory
            assert candidates == ["candidate"]
            return {
                "all_resolved": True,
                "one_to_one": True,
                "unresolved_object_ids": [],
                "accepted": [],
            }

    fake_scene = object()
    result = build_kitchen_execution_bundle(
        phase1,
        output_dir=output,
        task_requirements=Path("task.yaml"),
        scene_factory=lambda *args, **kwargs: fake_scene,
        resolver=Resolver(),
    )
    assert result.scene is fake_scene
    assert json.loads((output / "planner_output.json").read_text()) == plan
    manifest = json.loads((output / "execution_bundle_manifest.json").read_text())
    assert manifest["planner_received_backend_names"] is False
    assert manifest["execution_resolution_all_resolved"] is True
