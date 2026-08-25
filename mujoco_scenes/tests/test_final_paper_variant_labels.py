from mujoco_scenes.final_paper_variant_labels import (
    VARIANT_LABELS,
    paper_variant_label,
    resolve_variant_name,
    variant_mapping,
)
from mujoco_scenes.run_kitchen_ground_truth_execution import discover_variant_names
from mujoco_scenes.run_living_room_execution import EXPECTED_VARIANTS
from mujoco_scenes.workshop_ground_truth_planner import load_variant_specs


def test_short_labels_follow_active_execution_order():
    active = {
        "kitchen": tuple(discover_variant_names()),
        "living_room": tuple(EXPECTED_VARIANTS),
        "workshop": tuple(load_variant_specs()),
    }
    expected_sizes = {"kitchen": 12, "living_room": 10, "workshop": 10}
    for environment, variants in active.items():
        assert VARIANT_LABELS[environment] == variants
        assert len(variant_mapping(environment)) == expected_sizes[environment]


def test_short_labels_are_case_insensitive_and_reversible():
    examples = {
        "kitchen": ("K1", "F0_ALL_VISIBLE"),
        "living_room": ("L10", "I3_NO_TABLES"),
        "workshop": ("W9", "I0_NO_DRIVER"),
    }
    for environment, (label, internal) in examples.items():
        assert resolve_variant_name(environment, label.lower()) == internal
        assert resolve_variant_name(environment, label) == internal
        assert resolve_variant_name(environment, internal) == internal
        assert paper_variant_label(environment, internal) == label
