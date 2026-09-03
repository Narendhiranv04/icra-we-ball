from __future__ import annotations

import json

import pytest
import yaml

from baseline_common.models import Action

from mujoco_scenes.final_paper_variant_labels import variant_mapping
from vlm_tamp_baseline.models import Subgoal
from vlm_tamp_baseline.planner import VLMTAMPPlanner, VLMTAMPPlannerConfig
from vlm_tamp_baseline.pddlstream_refiner import PDDLStreamProtocol
from vlm_tamp_baseline.run_workshop import build_parser
from vlm_tamp_baseline.workshop_refiner import WorkshopPDDLStreamRefiner
from vlm_tamp_baseline.workshop_runtime import (
    DEFAULT_EXPECTED_ROOT,
    VARIANT_CONFIG,
    ExpectedGT,
    WorkshopPlanningRuntime,
    WorkshopSymbolicExecutor,
    compare_workshop_actions,
)


def _runtime(tmp_path, variant: str = "W1"):
    return WorkshopPlanningRuntime(variant, tmp_path, camera_count=1, image_width=320, image_height=240)


def test_workshop_closed_observation_hides_variant_contents(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        observation = runtime.observe_state()
        assert runtime.variant == "W1"
        assert observation.object_ids == {"object_0001"}
        assert {region.label for region in observation.regions} == {
            "left drawer", "right drawer", "tool cabinet", "main workbench"
        }
        workbench = next(
            region for region in observation.regions if region.label == "main workbench"
        )
        assert workbench.state == "open"
        assert workbench.inspected
        private = json.loads((tmp_path / "_private_evaluation" / "variant_state.json").read_text())
        assert "storage_contents" in private
        assert "storage_contents" not in observation.as_annotated_prompt_dict()
    finally:
        runtime.close()


@pytest.mark.parametrize("camera_count", (1, 3, 5))
def test_workshop_images_annotate_all_visible_regions(tmp_path, camera_count):
    runtime = WorkshopPlanningRuntime("W1", tmp_path, camera_count=camera_count, image_width=640, image_height=480)
    try:
        runtime.images()
        manifest = json.loads((tmp_path / "observations" / "initial" / "annotations.json").read_text())
        region_labels = {
            row["semantic_label"]
            for camera in manifest["cameras"].values()
            for row in camera["regions"]
        }
        assert region_labels == {"left drawer", "right drawer", "tool cabinet", "main workbench"}
    finally:
        runtime.close()


def test_workshop_inspection_reveals_only_its_storage_contents(tmp_path):
    runtime = _runtime(tmp_path)
    executor = WorkshopSymbolicExecutor(runtime)
    try:
        assert executor.execute(Action("INSPECT", {"region_id": "region_0001"})).success
        observation = runtime.observe_state()
        assert observation.object_ids == {"object_0001", "object_0002", "object_0004"}
        assert next(region for region in observation.regions if region.region_id == "region_0001").inspected
    finally:
        runtime.close()


def test_workshop_inspection_refreshes_visual_evidence(tmp_path):
    runtime = WorkshopPlanningRuntime("W1", tmp_path, camera_count=5, image_width=640, image_height=480)
    executor = WorkshopSymbolicExecutor(runtime)
    try:
        runtime.images()
        assert executor.execute(Action("INSPECT", {"region_id": "region_0001"})).success
        runtime.images()
        initial = json.loads((tmp_path / "observations" / "initial" / "annotations.json").read_text())
        refreshed = json.loads((tmp_path / "observations" / "revision_001" / "annotations.json").read_text())
        initial_labels = {
            row["semantic_label"]
            for camera in initial["cameras"].values()
            for row in camera["objects"]
        }
        refreshed_labels = {
            row["semantic_label"]
            for camera in refreshed["cameras"].values()
            for row in camera["objects"]
        }
        assert initial_labels == {"frame joint"}
        assert {"manual screwdriver", "screw"} <= refreshed_labels
    finally:
        runtime.close()


def test_workshop_annotation_labels_do_not_overlap(tmp_path):
    runtime = WorkshopPlanningRuntime("W1", tmp_path, camera_count=5, image_width=640, image_height=480)
    executor = WorkshopSymbolicExecutor(runtime)
    try:
        assert executor.execute(Action("INSPECT", {"region_id": "region_0001"})).success
        runtime.images()
        manifest = json.loads((tmp_path / "observations" / "revision_001" / "annotations.json").read_text())
        for camera in manifest["cameras"].values():
            labels = [
                tuple(row["label_bbox_xyxy"])
                for category in ("objects", "regions")
                for row in camera[category]
            ]
            for index, left in enumerate(labels):
                for right in labels[index + 1:]:
                    assert (
                        left[2] < right[0]
                        or right[2] < left[0]
                        or left[3] < right[1]
                        or right[3] < left[1]
                    )
    finally:
        runtime.close()


def test_workshop_pddlstream_refines_visible_repair_tuple(tmp_path):
    runtime = _runtime(tmp_path)
    executor = WorkshopSymbolicExecutor(runtime)
    try:
        assert executor.execute(Action("INSPECT", {"region_id": "region_0001"})).success
        result = WorkshopPDDLStreamRefiner(
            runtime, protocol=PDDLStreamProtocol(timeout_seconds=10.0)
        ).refine(
            Subgoal("FASTENED", {"tool_id": "object_0002", "fastener_id": "object_0004", "target_id": "object_0001"}),
            runtime.observe_state(),
        )
        assert result.success
        assert [action.skill for action in result.actions] == ["PICK", "INSERT", "PICK", "FASTEN"]
        for action in result.actions:
            assert executor.execute(action).success
        assert not runtime.goal_verifier()
        assert executor.execute(
            Action("PLACE", {"object_id": "object_0002", "region_id": "region_0004"})
        ).success
        assert runtime.goal_verifier()
    finally:
        runtime.close()


def test_workshop_pddlstream_rejects_hammer_as_fastener_or_driver(tmp_path):
    runtime = _runtime(tmp_path)
    executor = WorkshopSymbolicExecutor(runtime)
    refiner = WorkshopPDDLStreamRefiner(runtime, protocol=PDDLStreamProtocol(timeout_seconds=10.0))
    try:
        assert executor.execute(Action("INSPECT", {"region_id": "region_0001"})).success
        assert executor.execute(Action("INSPECT", {"region_id": "region_0003"})).success
        observation = runtime.observe_state()
        hammer = "object_0005"
        screw = "object_0004"
        target = "object_0001"
        assert not refiner.refine(
            Subgoal("INSERTED", {"fastener_id": hammer, "target_id": target}), observation
        ).success
        assert not refiner.refine(
            Subgoal("FASTENED", {"tool_id": hammer, "fastener_id": screw, "target_id": target}), observation
        ).success
    finally:
        runtime.close()


def test_workshop_expected_gt_agrees_with_public_variant_layout():
    config = yaml.safe_load(VARIANT_CONFIG.read_text())
    for label, internal in variant_mapping("workshop").items():
        expected = ExpectedGT.load(DEFAULT_EXPECTED_ROOT, label)
        contents = config["variants"][internal]["storage_contents"]
        assert expected.internal_variant == internal
        assert expected.intended_outcome == config["variants"][internal]["intended_outcome"]
        for action in expected.actions:
            if action["operator"] == "PICK":
                item, region = action["arguments"]
                if region in contents:
                    assert item in contents[region]
                else:
                    assert region == "MAIN_WORKBENCH_ZONE"
                    assert item in {item for values in contents.values() for item in values}
        inspected = [
            action["arguments"][0]
            for action in expected.actions
            if action["operator"] == "INSPECT_STORAGE"
        ]
        discovered: set[str] = set()
        required_inspections = []
        for region in config["search_order"]:
            required_inspections.append(region)
            discovered.update(contents[region])
            if {
                "workshop_medium_phillips_screw",
            } <= discovered and any(
                item in discovered
                for item in ("workshop_long_phillips_driver", "workshop_power_driver")
            ):
                break
        assert inspected == required_inspections
        if expected.intended_outcome == "FEASIBLE":
            driven = [action for action in expected.actions if action["operator"] == "DRIVE_FASTENER"]
            assert len(driven) == 1
            driver, screw, _target = driven[0]["arguments"]
            all_items = {item for values in contents.values() for item in values}
            assert driver in all_items
            assert screw in all_items
        else:
            assert any(action["operator"] == "TERMINATE_INFEASIBLE" for action in expected.actions)


def test_workshop_all_variants_have_the_expected_symbolic_feasibility(tmp_path):
    for label in variant_mapping("workshop"):
        runtime = _runtime(tmp_path / label, label)
        executor = WorkshopSymbolicExecutor(runtime)
        try:
            for region_id in sorted(runtime.storage_region_ids):
                assert executor.execute(Action("INSPECT", {"region_id": region_id})).success
            observation = runtime.observe_state()
            drivers = sorted(object_id for object_id in observation.object_ids if runtime.is_driver(object_id))
            screws = sorted(object_id for object_id in observation.object_ids if runtime.is_fastener(object_id))
            expected_feasible = runtime.expected.intended_outcome == "FEASIBLE"
            assert bool(drivers and screws) is expected_feasible
            if not expected_feasible:
                continue
            result = WorkshopPDDLStreamRefiner(
                runtime, protocol=PDDLStreamProtocol(timeout_seconds=10.0)
            ).refine(
                Subgoal(
                    "FASTENED",
                    {"tool_id": drivers[0], "fastener_id": screws[0], "target_id": runtime.target_object_id},
                ),
                observation,
            )
            assert result.success
            for action in result.actions:
                assert executor.execute(action).success
            assert executor.execute(
                Action("PLACE", {"object_id": drivers[0], "region_id": "region_0004"})
            ).success
            assert runtime.goal_verifier()
        finally:
            runtime.close()


def test_workshop_infeasibility_requires_complete_observation(tmp_path):
    runtime = _runtime(tmp_path, "W9")
    executor = WorkshopSymbolicExecutor(runtime)
    try:
        assert not runtime.infeasibility_proven()
        for region_id in sorted(runtime.storage_region_ids):
            assert executor.execute(Action("INSPECT", {"region_id": region_id})).success
        assert runtime.infeasibility_proven()
    finally:
        runtime.close()


def test_workshop_vlm_prompt_exposes_no_hidden_contents(tmp_path):
    class Transport:
        def __init__(self):
            self.payloads = []
            self.responses = [
                {"status": "STEPS", "steps": ["Inspect the left drawer."]},
                {"status": "SUBGOALS", "subgoals": [{"predicate": "INSPECTED", "arguments": {"region_id": "region_0001"}}]},
            ]

        def complete(self, payload):
            self.payloads.append(payload)
            return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}

    runtime = _runtime(tmp_path)
    try:
        observation, images = runtime.observe()
        transport = Transport()
        result = VLMTAMPPlanner(VLMTAMPPlannerConfig(model="test"), transport=transport).plan(
            runtime.goal, observation, images
        )
        assert result.plan.subgoals[0].predicate == "INSPECTED"
        prompt_text = json.dumps(transport.payloads[0])
        assert "workshop_long_phillips_driver" not in prompt_text
        assert "workshop_medium_phillips_screw" not in prompt_text
        assert "left drawer" in prompt_text
    finally:
        runtime.close()


def test_workshop_replan_exposes_only_inspected_visual_contents(tmp_path):
    class Transport:
        def __init__(self):
            self.payloads = []
            self.responses = [
                {"status": "STEPS", "steps": ["Inspect the left drawer."]},
                {"status": "SUBGOALS", "subgoals": [{"predicate": "INSPECTED", "arguments": {"region_id": "region_0001"}}]},
                {"status": "STEPS", "steps": ["Fasten the visible screw with the visible driver."]},
                {"status": "SUBGOALS", "subgoals": [{"predicate": "FASTENED", "arguments": {"tool_id": "object_0002", "fastener_id": "object_0004", "target_id": "object_0001"}}]},
            ]

        def complete(self, payload):
            self.payloads.append(payload)
            return {"choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}]}

    runtime = _runtime(tmp_path)
    executor = WorkshopSymbolicExecutor(runtime)
    transport = Transport()
    planner = VLMTAMPPlanner(VLMTAMPPlannerConfig(model="test"), transport=transport)
    try:
        initial, initial_images = runtime.observe()
        assert planner.plan(runtime.goal, initial, initial_images).plan.subgoals[0].predicate == "INSPECTED"
        assert executor.execute(Action("INSPECT", {"region_id": "region_0001"})).success
        refreshed, refreshed_images = runtime.observe()
        assert planner.plan(runtime.goal, refreshed, refreshed_images).plan.subgoals[0].predicate == "FASTENED"
        initial_prompt = json.dumps(transport.payloads[0])
        refreshed_prompt = json.dumps(transport.payloads[2])
        assert "object_0002" not in initial_prompt and "object_0004" not in initial_prompt
        assert "manual screwdriver" in refreshed_prompt and "object_0004" in refreshed_prompt
    finally:
        runtime.close()


def test_workshop_comparison_removes_execution_only_detail():
    expected = [
        {"operator": "MOVE_TO", "arguments": ["LEFT_DRAWER"]},
        {"operator": "OPEN_STORAGE", "arguments": ["LEFT_DRAWER"]},
        {"operator": "INSPECT_STORAGE", "arguments": ["LEFT_DRAWER"]},
        {"operator": "CLOSE_STORAGE", "arguments": ["LEFT_DRAWER"]},
        {"operator": "INSERT_FASTENER", "arguments": ["screw", "target"]},
    ]
    predicted = [
        {"operator": "INSPECT", "arguments": ["LEFT_DRAWER"]},
        {"operator": "INSERT", "arguments": ["screw", "target"]},
    ]
    comparison = compare_workshop_actions(predicted, expected)
    assert comparison["shared_task_vocabulary"]["exact_sequence_match"]
    assert not comparison["raw_execution_vocabulary"]["exact_sequence_match"]


def test_workshop_comparison_drops_temporary_workbench_staging():
    expected = [
        {"operator": "INSPECT_STORAGE", "arguments": ["LEFT_DRAWER"]},
        {"operator": "PICK", "arguments": ["driver", "LEFT_DRAWER"]},
        {"operator": "PLACE_ON_SURFACE", "arguments": ["driver", "MAIN_WORKBENCH_ZONE"]},
        {"operator": "PICK", "arguments": ["screw", "LEFT_DRAWER"]},
        {"operator": "PLACE_ON_SURFACE", "arguments": ["screw", "MAIN_WORKBENCH_ZONE"]},
        {"operator": "PICK", "arguments": ["screw", "MAIN_WORKBENCH_ZONE"]},
        {"operator": "INSERT_FASTENER", "arguments": ["screw", "target"]},
        {"operator": "PICK", "arguments": ["driver", "MAIN_WORKBENCH_ZONE"]},
        {"operator": "DRIVE_FASTENER", "arguments": ["driver", "screw", "target"]},
        {"operator": "PLACE_ON_SURFACE", "arguments": ["driver", "MAIN_WORKBENCH_ZONE"]},
    ]
    predicted = [
        {"operator": "INSPECT", "arguments": ["LEFT_DRAWER"]},
        {"operator": "PICK", "arguments": ["screw"]},
        {"operator": "INSERT", "arguments": ["screw", "target"]},
        {"operator": "PICK", "arguments": ["driver"]},
        {"operator": "FASTEN", "arguments": ["driver", "screw", "target"]},
        {"operator": "PLACE", "arguments": ["driver", "MAIN_WORKBENCH_ZONE"]},
    ]
    assert compare_workshop_actions(predicted, expected)["shared_task_vocabulary"]["exact_sequence_match"]


def test_workshop_cli_defaults_to_one_initial_model_call():
    args = build_parser().parse_args(["--variant", "W1", "--output-dir", "run"])
    assert args.max_model_calls == 1
    assert args.camera_count == 5
