"""Focused structural tests for the plan-only canonical pipeline."""

from pathlib import Path

import yaml

from mujoco_scenes.final_paper_variant_labels import (
    PREFIXES, VARIANT_LABELS, paper_variant_label, resolve_variant_name,
)
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.planning import plan_with_common_astar
from mujoco_scenes.functional_tamp_pipeline.spec_provider import (
    FunctionalSpecProvider, provider_for_mode,
)
from mujoco_scenes.functional_tamp_pipeline.domains.workshop import WorkshopPlanningCompiler
from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import KitchenPlanningCompiler
from mujoco_scenes.living_room_variants import load_living_room_variants
from mujoco_scenes.workshop_scene import WORKSHOP_VARIANTS_CONFIG


def test_both_modes_share_the_provider_boundary() -> None:
    assert isinstance(provider_for_mode("gt"), FunctionalSpecProvider)
    assert isinstance(provider_for_mode("vlm"), FunctionalSpecProvider)


def test_gt_specs_have_no_variant_solution_input() -> None:
    provider = GTSpecProvider()
    for domain in ("workshop", "kitchen", "living_room"):
        specification = provider.provide(domain, "task")
        assert specification.source == "GT_FUNCTIONAL_SPEC_ONLY"
        assert specification.roles
        serialized = str(specification.to_dict()).lower()
        assert "expected_solution" not in serialized
        assert "hidden" not in serialized


def test_workshop_plan_uses_common_astar_and_excludes_search_open() -> None:
    assignment = {
        "driver": "workshop_long_phillips_driver",
        "fastener": "workshop_medium_phillips_screw",
        "driver_source": "RIGHT_DRAWER",
        "fastener_source": "LEFT_DRAWER",
        "work_surface": "MAIN_WORKBENCH_ZONE",
        "target_joint": "workshop_frame_joint",
    }
    result = plan_with_common_astar(
        WorkshopPlanningCompiler(), assignment,
        {"opened_regions": ("LEFT_DRAWER", "RIGHT_DRAWER")},
    )
    assert result.search.statistics["algorithm"] == "deterministic_astar_symbolic_state_search"
    assert [row["operator"] for row in result.actions] == [
        "PICK", "PLACE", "PICK", "SCREW", "PLACE",
    ]
    assert all(row["operator"] != "OPEN" for row in result.actions)


def test_kitchen_compiles_observed_witness_into_common_astar() -> None:
    objects = ("cup", "bowl", "coffee_source", "water_source", "soup_source", "spoon", "fork")
    compiled = {
        "role_assignments": {"coffee_targets": ["cup"], "soup_targets": ["bowl"]},
        "capabilities": {
            "source_contains": [
                ["coffee_source", "coffee"], ["water_source", "water"],
                ["soup_source", "soup"],
            ],
            "initial_target_contents": [],
            "can_stir": [["spoon", "cup"]],
            "assigned_soup_utensil": [["fork", "bowl"]],
        },
        "requirements": {
            "home_region": "countertop", "serving_destination": "serving_area",
        },
        "objects": {
            obj: {"location": {"region_id": "countertop"}} for obj in objects
        },
    }
    result = plan_with_common_astar(
        KitchenPlanningCompiler(), compiled["role_assignments"],
        {"compiled_observed_state": compiled},
    )
    assert result.search.statistics["algorithm"] == "deterministic_astar_symbolic_state_search"
    assert result.validation["goal_status"] == "GOAL_SATISFIED"
    assert {row["operator"] for row in result.actions} == {
        "PICK", "PLACE", "POUR", "STIR",
    }
    assert all(row["operator"] != "OPEN" for row in result.actions)


def test_new_pipeline_does_not_import_oracle_solution_generators() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in root.rglob("*.py")
        if "tests" not in path.parts
    )
    for forbidden in (
        "solve_gt_assignment", "generate_gt_plan", "EXPECTED_GT_ACTIONS",
        "expected_inspection_regions", "privileged_validate_variant_feasibility",
    ):
        assert forbidden not in source


def test_every_paper_variant_resolves_to_a_backing_scene_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    kitchen = yaml.safe_load(
        (root / "configs" / "kitchen_feasibility_variants.yaml").read_text(
            encoding="utf-8"
        )
    )["variants"]
    workshop = yaml.safe_load(
        WORKSHOP_VARIANTS_CONFIG.read_text(encoding="utf-8")
    )["variants"]
    living_room = load_living_room_variants()
    backing = {
        "kitchen": kitchen,
        "workshop": workshop,
        "living_room": living_room,
    }
    assert {domain: len(variants) for domain, variants in VARIANT_LABELS.items()} == {
        "kitchen": 12, "workshop": 10, "living_room": 10,
    }
    for domain, variants in VARIANT_LABELS.items():
        for index, internal in enumerate(variants, start=1):
            short = f"{PREFIXES[domain]}{index}"
            assert resolve_variant_name(domain, short) == internal
            assert resolve_variant_name(domain, short.lower()) == internal
            assert paper_variant_label(domain, internal) == short
            assert internal in backing[domain]


def test_runner_stops_at_action_sequence_and_has_no_plan_execution_call() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run.py").read_text(encoding="utf-8")
    models = (root / "models.py").read_text(encoding="utf-8")
    assert "ACTION_SEQUENCE_READY" in runner
    assert "execute_plan(" not in runner
    assert "execute_sequence(" not in runner
    assert "execution_result" not in models
