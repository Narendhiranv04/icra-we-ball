import json
from pathlib import Path

import pytest
import yaml

from mujoco_scenes.evaluate_target_assignment_run import (
    ASSIGNMENT_MODES,
    evaluate_saved_target_assignment_run,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mode(mode: str, stage: int):
    complete = mode != "joint-target-specific" or stage >= 2
    label = "fork" if mode == "geometry-only" else "spoon"
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "selected_witness": None,
        "operation_assignments": [
            {"semantic": {"canonical_label": label}}
        ] if complete else [],
        "function_group_evaluations": [],
    }


def _saved_run(tmp_path: Path):
    run = tmp_path / "run"
    _write_json(
        run / "run_config.json",
        {"scene_name": "S1_ablation3_multi_target_primary"},
    )
    summary_stages = []
    for stage, region in enumerate(("INITIAL", "D1", "D2")):
        modes = {mode: _mode(mode, stage) for mode in ASSIGNMENT_MODES}
        name = f"{stage:03d}_{'initial' if stage == 0 else 'after_' + region}"
        _write_json(
            run / "stages" / name / "assignment_mode_comparison.json",
            {
                "same_observation_evidence": True,
                "perception_rerun_per_mode": False,
                "measurement_cloud_paths": [],
                "semantic_record_paths": [],
                "modes": modes,
            },
        )
        summary_stages.append(
            {"stage": stage, "region_id": region, "modes": modes}
        )
    _write_json(
        run / "assignment_ablation_summary.json",
        {
            "task_id": "prepare_coffee_and_soup_multi_target",
            "stages": summary_stages,
        },
    )
    evaluation = tmp_path / "evaluation.yaml"
    evaluation.write_text(
        yaml.safe_dump(
            {
                "purpose": "OFFLINE_EVALUATION_ONLY",
                "runtime_import_forbidden": True,
                "scene_name": "S1_ablation3_multi_target_primary",
                "task_id": "prepare_coffee_and_soup_multi_target",
                "expected": {
                    "semantic-only": {
                        "status": "COMPLETE",
                        "completion_stage": 0,
                        "completion_region": "INITIAL",
                        "intentionally_incorrect": True,
                    },
                    "geometry-only": {
                        "status": "COMPLETE",
                        "completion_stage": 0,
                        "completion_region": "INITIAL",
                        "intentionally_incorrect": True,
                        "required_selected_tool_label": "fork",
                    },
                    "joint-target-agnostic-count": {
                        "status": "COMPLETE",
                        "completion_stage": 0,
                        "completion_region": "INITIAL",
                        "intentionally_incorrect": True,
                    },
                    "joint-target-specific": {
                        "status": "COMPLETE",
                        "completion_stage": 2,
                        "completion_region": "D2",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return run, evaluation


def test_target_assignment_offline_modes_share_saved_evidence(tmp_path):
    run, evaluation = _saved_run(tmp_path)
    report = evaluate_saved_target_assignment_run(
        run, evaluation_config=evaluation
    )
    assert report["all_expected_results_matched"] is True
    assert report["modes_share_identical_observation_evidence"] is True
    assert report["perception_was_not_rerun"] is True
    assert report["modes"]["geometry-only"]["selected_tool_labels"] == [
        "fork"
    ]
    assert report["modes"]["joint-target-specific"][
        "completion_stage"
    ] == 2


def test_target_assignment_offline_rejects_per_mode_perception(tmp_path):
    run, evaluation = _saved_run(tmp_path)
    stage = run / "stages" / "000_initial" / "assignment_mode_comparison.json"
    payload = json.loads(stage.read_text(encoding="utf-8"))
    payload["perception_rerun_per_mode"] = True
    _write_json(stage, payload)
    with pytest.raises(ValueError, match="shared observation"):
        evaluate_saved_target_assignment_run(
            run, evaluation_config=evaluation
        )
