import json
from pathlib import Path

import pytest
import yaml

from mujoco_scenes.evaluate_usage_policy_run import (
    evaluate_saved_usage_policy_run,
)


MODES = ("always-reusable", "always-distinct", "function-aware")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mode_result(mode, stage):
    complete = (
        mode == "always-reusable"
        or (mode == "function-aware" and stage >= 2)
    )
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "distinct_physical_tool_count": (
            1 if mode == "always-reusable" else 2 if stage >= 2 else 1
        ),
        "policy_required_distinct_physical_tool_count": {
            "always-reusable": 1,
            "always-distinct": 4,
            "function-aware": 2,
        }[mode],
        "satisfied_target_slot_count": 4 if complete else 3,
        "required_target_slot_count": 4,
        "operation_assignments": [],
        "function_group_evaluations": [],
    }


def _saved_run(tmp_path):
    run = tmp_path / "run"
    sequence = ["D1", "D2", "C2", "B1", "C1"]
    _write_json(
        run / "run_config.json",
        {
            "scene_name": "S1_ablation2_count_reuse_primary",
            "inspection_sequence": sequence,
        },
    )
    stages = []
    regions = ["INITIAL", *sequence]
    for stage, region in enumerate(regions):
        modes = {mode: _mode_result(mode, stage) for mode in MODES}
        stages.append({"stage": stage, "region_id": region, "modes": modes})
        _write_json(
            run
            / "stages"
            / f"{stage:03d}_{'initial' if stage == 0 else 'after_' + region}"
            / "policy_mode_comparison.json",
            {
                "same_observation_evidence": True,
                "measurement_cloud_paths": [],
                "semantic_record_paths": [],
                "modes": modes,
            },
        )
    _write_json(
        run / "policy_ablation_summary.json",
        {
            "task_id": "prepare_two_coffees_and_soups",
            "shared_observation_evidence": True,
            "stages": stages,
        },
    )
    evaluation = tmp_path / "evaluation.yaml"
    evaluation.write_text(
        yaml.safe_dump(
            {
                "purpose": "OFFLINE_EVALUATION_ONLY",
                "runtime_import_forbidden": True,
                "scene_name": "S1_ablation2_count_reuse_primary",
                "task_id": "prepare_two_coffees_and_soups",
                "expected": {
                    "always-reusable": {
                        "status": "COMPLETE",
                        "completion_stage": 0,
                    },
                    "always-distinct": {
                        "status": "EXHAUSTED",
                        "policy_required_distinct_physical_tool_count": 4,
                    },
                    "function-aware": {
                        "status": "COMPLETE",
                        "completion_stage": 2,
                        "completion_region": "D2",
                        "distinct_physical_tool_count": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return run, evaluation


def test_offline_policy_evaluation_uses_one_shared_saved_run(tmp_path):
    run, evaluation = _saved_run(tmp_path)
    report = evaluate_saved_usage_policy_run(
        run, evaluation_config=evaluation
    )
    assert report["all_expected_results_matched"] is True
    assert report["policy_modes_share_identical_observation_evidence"] is True
    assert report["perception_was_not_rerun"] is True
    assert report["modes"]["always-reusable"]["completion_stage"] == 0
    assert report["modes"]["function-aware"]["completion_stage"] == 2
    assert report["modes"]["always-distinct"]["actual_status"] == "EXHAUSTED"


def test_offline_policy_evaluation_rejects_unshared_evidence(tmp_path):
    run, evaluation = _saved_run(tmp_path)
    stage = next((run / "stages").iterdir())
    comparison = json.loads(
        (stage / "policy_mode_comparison.json").read_text()
    )
    comparison["same_observation_evidence"] = False
    _write_json(stage / "policy_mode_comparison.json", comparison)
    with pytest.raises(ValueError, match="same-evidence"):
        evaluate_saved_usage_policy_run(
            run, evaluation_config=evaluation
        )
