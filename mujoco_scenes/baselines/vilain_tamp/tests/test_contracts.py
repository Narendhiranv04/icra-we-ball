from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.config import (
    BaselineConfig,
    Domain,
    ModelCondition,
    ObservationMode,
)
from mujoco_scenes.baselines.vilain_tamp.contracts import (
    ObjectEstimate,
    ObjectEstimateStatus,
    SymbolicAction,
    SymbolicPlan,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_symbolic_action_preserves_neutral_four_field_shape() -> None:
    action = SymbolicAction(
        action_index=1,
        action_instance_id="vilain_00_001_pick_from",
        operator="PICK-FROM",
        arguments=("mug_1", "counter"),
    )
    assert action.to_dict() == {
        "action_index": 1,
        "action_instance_id": "vilain_00_001_pick_from",
        "operator": "PICK-FROM",
        "arguments": ["mug_1", "counter"],
    }
    with pytest.raises(FrozenInstanceError):
        action.operator = "PLACE-ON"  # type: ignore[misc]


def test_nested_contracts_are_json_friendly() -> None:
    action = SymbolicAction(1, "vilain_00_001_pick_from", "PICK-FROM", ("mug_1", "counter"))
    plan = SymbolicPlan(
        attempt_index=0,
        planner_name="Fast Downward",
        planner_version="24.06",
        search_configuration="lama-first",
        actions=(action,),
        plan_cost=1.0,
        planner_time_seconds=0.25,
        raw_plan_artifacts=("attempts/00/sas_plan",),
        plan_sha256="abc123",
    )
    assert plan.to_dict()["actions"] == [action.to_dict()]


def test_object_estimate_contains_only_baseline_evidence() -> None:
    estimate = ObjectEstimate(
        object_id="mug_1",
        label="mug",
        pddl_type="vessel",
        description="white ceramic mug",
        detections=({"camera": "front", "xyxy": [1, 2, 3, 4], "confidence": 0.9},),
        estimated_centroid_m=(0.1, 0.2, 0.3),
        centroid_covariance=None,
        observation_stage_ids=("000_initial",),
        status=ObjectEstimateStatus.OBSERVED,
    )
    serialized = estimate.to_dict()
    assert serialized["status"] == "OBSERVED"
    assert set(serialized) == {
        "object_id",
        "label",
        "pddl_type",
        "description",
        "detections",
        "estimated_centroid_m",
        "centroid_covariance",
        "observation_stage_ids",
        "status",
    }


def test_paper_faithful_configuration_is_frozen_and_validated() -> None:
    config = BaselineConfig.from_yaml(CONFIG_ROOT / "paper_faithful.yaml")
    assert config.domain is Domain.KITCHEN
    assert config.observation_mode is ObservationMode.INITIAL_ONLY
    assert config.model_condition is ModelCondition.PAPER_FAITHFUL
    assert config.object_estimator_model == "Qwen2.5-VL-7B-Instruct"
    assert config.reasoning_model == "gpt-4o-2024-08-06"
    assert config.symbolic_planner == "Fast Downward"
    assert config.search_configuration == "lama-first"
    assert config.timeouts.symbolic_seconds == 200
    assert config.max_cp_corrections == 3
    with pytest.raises(FrozenInstanceError):
        config.max_cp_corrections = 2  # type: ignore[misc]


def test_configuration_rejects_invalid_boundary_values() -> None:
    valid = {
        "domain": "workshop",
        "observation_mode": "fixed_full_inspection",
        "model_condition": "model_matched",
        "max_cp_corrections": 3,
        "timeouts": {
            "symbolic_seconds": 200,
            "model_seconds": 120,
            "refinement_seconds": 200,
        },
        "output_root": "runs/vilain_tamp",
        "external_tools": {"fast_downward": None, "val": None},
        "object_estimator_model": "independent-model",
        "reasoning_model": "independent-model",
        "symbolic_planner": "Fast Downward",
        "search_configuration": "lama-first",
        "independent_model_calls": True,
    }
    with pytest.raises(ValueError, match="max_cp_corrections"):
        BaselineConfig.from_mapping({**valid, "max_cp_corrections": 4})
    with pytest.raises(ValueError, match="symbolic_seconds"):
        BaselineConfig.from_mapping(
            {**valid, "timeouts": {**valid["timeouts"], "symbolic_seconds": 0}}
        )
    with pytest.raises(ValueError, match="output_root"):
        BaselineConfig.from_mapping({**valid, "output_root": " "})
    with pytest.raises(ValueError, match="must be an absolute path"):
        BaselineConfig.from_mapping(
            {**valid, "external_tools": {"fast_downward": "relative/path", "val": None}}
        )
