from __future__ import annotations

from pathlib import Path
import json

from baseline_common.models import Action, Entity, Observation

from vlm_tamp_baseline.living_room_runtime import (
    LivingRoomPlanningRuntime,
    LivingRoomSymbolicExecutor,
    canonical_actions,
    compare_action_sequences,
)
from vlm_tamp_baseline.models import Subgoal
from vlm_tamp_baseline.pddlstream_refiner import (
    LivingRoomGeometryOracle,
    PDDLStreamSubgoalRefiner,
)
from vlm_tamp_baseline.run_living_room import build_parser


def _runtime(tmp_path: Path, variant: str = "L1") -> LivingRoomPlanningRuntime:
    return LivingRoomPlanningRuntime(
        variant,
        tmp_path,
        image_width=160,
        image_height=90,
    )


def test_living_room_runtime_resolves_short_variant_and_private_goal(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        observation = runtime.observe_state()
        assert runtime.variant == "L1"
        assert observation.scene == "living_room"
        assert len(observation.entities) == 5
        assert len(observation.regions) == 4
        assert not runtime.goal_verifier()
        assert {item.label for item in observation.entities} == {
            "cup_1", "cup_2", "saucer_1", "saucer_2", "tv_remote"
        }
        annotated = observation.as_annotated_prompt_dict()["semantic_annotations"]
        assert {row["alias"] for row in annotated["objects"]} == {
            "cup_1", "cup_2", "saucer_1", "saucer_2", "tv_remote"
        }
        assert runtime.expected.intended_outcome == "FEASIBLE"
    finally:
        runtime.close()


def test_living_room_images_annotate_visible_destination_regions(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        images = runtime.images()
        annotations = json.loads(
            (tmp_path / "observations" / "initial" / "annotations.json").read_text()
        )
        assert len(images) == 5
        visible_regions = {
            row["id"]
            for camera in annotations["cameras"].values()
            for row in camera["regions"]
        }
        assert set(runtime.region_roles) <= visible_regions
        visible_aliases = {
            row["semantic_label"]
            for camera in annotations["cameras"].values()
            for row in camera["objects"]
        }
        assert {"cup 1", "cup 2", "saucer 1", "saucer 2", "tv remote"} <= visible_aliases
    finally:
        runtime.close()


def test_missing_region_variant_resolves_without_backend_probe_failure(tmp_path):
    runtime = _runtime(tmp_path, "L7")
    try:
        assert runtime.expected.intended_outcome == "INFEASIBLE"
        assert "SHARED_TABLE" not in runtime.region_roles.values()
        assert not runtime.goal_verifier()
    finally:
        runtime.close()


def test_symbolic_executor_completes_functional_goal_without_mujoco_motion(tmp_path):
    runtime = _runtime(tmp_path)
    executor = LivingRoomSymbolicExecutor(runtime)
    try:
        targets = {
            "object_0001": "region_0001",
            "object_0002": "region_0001",
            "object_0003": "region_0002",
            "object_0004": "region_0003",
            "object_0005": "region_0003",
        }
        initial_time = float(runtime.scene.data.time)
        for object_id, region_id in targets.items():
            assert executor.execute(Action("PICK", {"object_id": object_id})).success
            assert executor.execute(
                Action("PLACE", {"object_id": object_id, "region_id": region_id})
            ).success
        assert runtime.goal_verifier()
        assert float(runtime.scene.data.time) == initial_time
    finally:
        runtime.close()


def test_gt_sequence_comparison_reports_ordered_overlap():
    expected = [
        {"operator": "PICK", "arguments": ["a"]},
        {"operator": "PLACE", "arguments": ["a", "r"]},
    ]
    predicted = [
        {"operator": "PICK", "arguments": ["a"]},
        {"operator": "PICK", "arguments": ["b"]},
        {"operator": "PLACE", "arguments": ["a", "r"]},
    ]
    comparison = compare_action_sequences(predicted, expected)
    assert comparison["lcs_action_count"] == 2
    assert comparison["ordered_recall"] == 1.0
    assert not comparison["exact_sequence_match"]


def test_pddlstream_refines_living_room_placement_from_measured_geometry(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        refiner = PDDLStreamSubgoalRefiner(
            runtime.inventory,
            LivingRoomGeometryOracle(runtime.inventory, runtime.region_registry),
        )
        result = refiner.refine(
            Subgoal(
                "PLACED",
                {"object_id": "object_0001", "region_id": "region_0001"},
            ),
            runtime.observe_state(),
        )
        assert result.success
        assert [action.skill for action in result.actions] == ["PICK", "PLACE"]
    finally:
        runtime.close()


def test_canonical_actions_excludes_failed_and_non_gt_actions():
    history = (
        {
            "action": {"skill": "PICK", "arguments": {"object_id": "a"}},
            "success": True,
        },
        {
            "action": {"skill": "INSPECT", "arguments": {"region_id": "r"}},
            "success": True,
        },
        {
            "action": {
                "skill": "PLACE",
                "arguments": {"object_id": "a", "region_id": "r"},
            },
            "success": False,
        },
    )
    assert canonical_actions(history) == [
        {"operator": "PICK", "arguments": ["a"]}
    ]


def test_living_room_cli_defaults_to_planning_only_inputs():
    arguments = build_parser().parse_args(
        ["--variant", "L1", "--output-dir", "run"]
    )
    # Thinking is enabled by default, so the token budget is the profile's,
    # and both baselines default to the same decoding condition.
    assert arguments.max_tokens == 24576
    assert arguments.decoding == "model-native"
    assert arguments.max_model_calls == 1
    assert arguments.max_total_actions == 40


def test_catalogued_but_unrefinable_predicate_is_a_typed_failure():
    """The living-room catalogue advertises CLEANED; PDDLStream cannot encode it.

    A real VLM can therefore emit it, and that must surface as a feedable
    ``unsupported_subgoal`` failure rather than crashing the episode.
    """
    from vlm_tamp_baseline.catalog import load_catalog, scene_subgoals
    from vlm_tamp_baseline.pddlstream_refiner import (
        PDDLStreamSubgoalRefiner,
        REFINABLE_PREDICATES,
    )

    advertised = set(scene_subgoals(load_catalog(), "living_room"))
    unrefinable = advertised - REFINABLE_PREDICATES
    assert "CLEANED" in unrefinable, "catalogue no longer advertises CLEANED"

    refiner = PDDLStreamSubgoalRefiner.__new__(PDDLStreamSubgoalRefiner)
    subgoal = Subgoal("CLEANED", {"tool_id": "object_0001", "target_id": "object_0002"})
    observation = Observation(
        "living_room", 0,
        (Entity("object_0001", "object", "cloth", {}),
         Entity("object_0002", "object", "table", {})),
        (), {"held_object": None}, False,
    )

    result = refiner.refine(subgoal, observation)

    assert not result.success
    assert result.failure is not None
    assert result.failure.code == "unsupported_subgoal"
    assert result.actions == ()
