import json

import mujoco
import numpy as np

from baseline_common.models import Entity, Observation
from mujoco_scenes.baseline_kitchen_runtime import (
    BaselineKitchenRuntime,
    KitchenEffectLedger,
    KitchenGoalContract,
    PUBLIC_REGIONS,
    STATIC_REGION_GEOMS,
    _public_region_reference,
    goal_contract_from_expected_actions,
)
from mujoco_scenes.kitchen_execution_policy import (
    KitchenWorkspace,
    WORKSPACE_DESTINATIONS,
)
from mujoco_scenes.mobile_motion import physical_location


def contract():
    return KitchenGoalContract(
        ("poured(kettle,cup)", "placed(cup,serving_area)"),
        ("cup",),
        (("bowl", "spoon"),),
        {"spoon_a": "spoon", "spoon_b": "spoon"},
    )


def test_goal_contract_accepts_functionally_equivalent_spoon_instances():
    ledger = KitchenEffectLedger(contract())
    ledger.accept(("poured(kettle,cup)",))
    ledger.accept(("stirred(spoon_b,cup)",))
    ledger.accept(("placed(spoon_a,bowl)",))
    ledger.accept(("placed(cup,serving_area)",))
    assert ledger.goal_satisfied


def test_expected_actions_compile_only_private_terminal_relations(tmp_path):
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({
        "intended_outcome": "FEASIBLE",
        "actions": [
            {"operator": "PICK", "arguments": ["jar"]},
            {"operator": "POUR", "arguments": ["jar", "cup"]},
            {"operator": "PLACE", "arguments": ["jar", "countertop"]},
            {"operator": "STIR", "arguments": ["spoon", "cup"]},
            {"operator": "PLACE", "arguments": ["cup", "serving_area"]},
        ],
    }))

    compiled = goal_contract_from_expected_actions(
        path,
        generic_for_backend={"jar": "object_1", "cup": "object_2", "spoon": "object_3"},
        object_labels={"object_1": "jar", "object_2": "cup", "object_3": "spoon"},
    )

    assert compiled.required_effects == (
        "poured(object_1,object_2)",
        "placed(object_2,serving_area)",
    )
    assert compiled.stir_targets == ("object_2",)


def test_moving_an_object_replaces_its_previous_place_effect():
    ledger = KitchenEffectLedger(contract())
    ledger.accept(("placed(spoon_a,bowl)",))
    ledger.accept(("placed(spoon_a,countertop)",))
    assert "placed(spoon_a,bowl)" not in ledger.effects


def test_placing_an_object_clears_its_stale_holding_effect():
    ledger = KitchenEffectLedger(contract())
    ledger.accept(("holding(spoon_a)",))
    ledger.accept(("placed(spoon_a,countertop)",))
    assert "holding(spoon_a)" not in ledger.effects


def test_effects_are_exposed_as_observed_entity_facts():
    ledger = KitchenEffectLedger(contract())
    ledger.accept(("poured(kettle,cup)", "stirred(spoon_a,cup)"))
    observation = Observation(
        "kitchen",
        1,
        (Entity("cup", "object", "cup"),),
        (),
        {},
    )
    updated = ledger.augment(observation)
    assert updated.entities[0].facts["poured_from"] == ["kettle"]
    assert updated.entities[0].facts["stirred_with"] == ["spoon_a"]


def test_pddl_workspace_destinations_resolve_to_mobile_physical_poses():
    assert {
        workspace.value: physical_location(destination)
        for workspace, destination in WORKSPACE_DESTINATIONS.items()
    } == {
        "home": "home",
        "left_side": "cupboard1",
        "right_side": "right_side",
    }


def test_private_table_provenance_uses_the_public_countertop_region():
    assert _public_region_reference("INITIAL") == "countertop"
    assert _public_region_reference("TABLE") == "countertop"
    assert _public_region_reference("TABLETOP") == "countertop"
    assert _public_region_reference("C1") == "C1"


def test_every_public_region_has_annotation_geometry():
    assert set(STATIC_REGION_GEOMS) == {
        region.region_id for region in PUBLIC_REGIONS
    }


def test_shared_textualization_hides_semantic_names_and_functions():
    observation = Observation(
        "kitchen",
        1,
        (
            Entity(
                "object_0001",
                "object",
                "mug",
                {
                    "source_region": "countertop",
                    "semantic_provenance": "YOLO_WORLD",
                    "selected_functions": ["can_hold_liquid"],
                },
            ),
        ),
        (),
        {"holding": None},
    )

    state = observation.as_semantic_neutral_prompt_dict()

    assert state["visible_objects"] == [
        {"id": "object_0001", "facts": {"source_region": "countertop"}}
    ]
    assert "mug" not in str(state)
    assert "YOLO" not in str(state)
    assert "can_hold_liquid" not in str(state)


def test_annotation_uses_segmentation_and_semantic_labels():
    runtime = BaselineKitchenRuntime.__new__(BaselineKitchenRuntime)
    runtime.contract = KitchenGoalContract((), (), (), {"object_0001": "mug"})
    runtime.region_geom_ids = {
        "C1": frozenset((5,)),
        "countertop": frozenset((11,)),
        "serving_area": frozenset((9,)),
    }
    runtime.object_geom_ids = {"object_0001": frozenset((7,))}
    runtime._latest_visible_object_ids = frozenset(("object_0001",))
    frame = np.full((80, 120, 3), 255, dtype=np.uint8)
    segmentation = np.full((80, 120, 2), -1, dtype=np.int32)
    segmentation[5:30, 5:45, 0] = 5
    segmentation[5:30, 5:45, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    segmentation[35:70, 55:100, 0] = 7
    segmentation[35:70, 55:100, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    segmentation[55:75, 4:50, 0] = 9
    segmentation[55:75, 4:50, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    for row in range(30, 35):
        inset = row - 30
        segmentation[row, inset : 120 - inset, 0] = 11
        segmentation[row, inset : 120 - inset, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)

    _image, manifest = runtime._annotate_frame(frame, segmentation)

    assert manifest == {
        "objects": [
            {
                "id": "object_0001",
                "semantic_label": "mug",
                "bbox_xyxy": [55, 35, 99, 69],
                "pixel_count": 1575,
            }
        ],
        "regions": [
            {
                "id": "C1",
                "bbox_xyxy": [5, 5, 44, 29],
                "pixel_count": 1000,
            },
            {
                "id": "countertop",
                "bbox_xyxy": [0, 30, 119, 34],
                "pixel_count": 580,
            },
            {
                "id": "serving_area",
                "bbox_xyxy": [4, 55, 49, 74],
                "pixel_count": 920,
            },
        ],
    }
    assert manifest["objects"][0]["semantic_label"] == "mug"
    # Static surface annotations follow the segmentation instead of drawing the
    # entire axis-aligned bounding rectangle.
    assert _image.getpixel((0, 34)) == (255, 255, 255)
    # The object label is placed outside the object box, leaving its pixels visible.
    assert _image.getpixel((60, 40)) == (255, 255, 255)
