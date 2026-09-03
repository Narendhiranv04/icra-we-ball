import json

import pytest

from baseline_common.summarize_plan_gt_batch import (
    _apply_planned_placements,
    _episode_metrics,
    _living_room_placement_metrics,
    main,
)


def _snapshot(placements):
    regions = [
        {"id": "left", "label": "personal_table_left"},
        {"id": "shared", "label": "shared_table"},
        {"id": "right", "label": "personal_table_right"},
        {"id": "staging", "label": "staging_area"},
    ]
    entities = [
        {
            "id": f"object_{index}",
            "kind": "object",
            "label": label,
            "facts": {"region_id": region},
        }
        for index, (label, region) in enumerate(placements, 1)
    ]
    return {"known_regions": regions, "visible_entities": entities}


def test_living_room_placement_metrics_score_complete_goal():
    snapshot = _snapshot(
        [
            ("cup_1", "left"),
            ("saucer_1", "left"),
            ("cup_2", "right"),
            ("saucer_2", "right"),
            ("tv_remote", "shared"),
        ]
    )
    assert _living_room_placement_metrics(snapshot) == (1.0, 1.0)


def test_living_room_placement_metrics_penalize_duplicate_and_missing_pairs():
    snapshot = _snapshot(
        [
            ("cup_1", "left"),
            ("saucer_1", "left"),
            ("cup_2", "left"),
            ("saucer_2", "staging"),
            ("tv_remote", "shared"),
        ]
    )
    correctness, coverage = _living_room_placement_metrics(snapshot)
    assert correctness == 3 / 4
    assert coverage == 3 / 5


def test_apply_planned_placements_supports_planning_only_owl_output():
    snapshot = _snapshot([("cup_1", "staging")])
    payload = {
        "result": {
            "actions": [
                {"operator": "PICK", "arguments": ["object_1"]},
                {"operator": "PLACE", "arguments": ["object_1", "left"]},
            ]
        }
    }
    _apply_planned_placements(snapshot, payload)
    assert snapshot["visible_entities"][0]["facts"]["region_id"] == "left"


def test_workshop_summary_does_not_score_preexecution_goal_snapshot(tmp_path):
    run = tmp_path / "trial"
    private = run / "_private_evaluation"
    private.mkdir(parents=True)
    (private / "latest_observation.json").write_text('{"goal_satisfied": true}')
    metrics = _episode_metrics(
        run / "episode_result.json",
        {
            "environment": "workshop",
            "gt_comparison": {"expected_outcome": "FEASIBLE", "predicted_outcome": "FEASIBLE"},
        },
    )
    assert metrics["goal_complete"] is None


def test_legacy_vlm_artifact_infers_two_raw_requests(tmp_path):
    metrics = _episode_metrics(
        tmp_path / "episode_result.json",
        {
            "baseline": "vlm_tamp",
            "environment": "workshop",
            "result": {"model_calls": 1},
            "gt_comparison": {
                "expected_outcome": "INFEASIBLE",
                "predicted_outcome": "UNRESOLVED",
            },
        },
    )
    assert metrics["planning_rounds"] == 1.0
    assert metrics["raw_vlm_requests"] == 2.0


def test_summary_reports_missing_episode_results_as_failed_trials(tmp_path, monkeypatch):
    output = tmp_path / "missing_episode"
    output.mkdir()
    (tmp_path / "batch_summary.json").write_text(json.dumps({
        "runs": [{
            "method": "vlm_tamp",
            "environment": "workshop",
            "variant": "W1",
            "camera_count": 5,
            "seed": 0,
            "return_code": 1,
            "output_dir": str(output),
        }]
    }))
    report = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_plan_gt_batch", str(tmp_path), "--output-dir", str(report)],
    )
    main()
    row = json.loads((report / "table4.json").read_text())["rows"][0]
    assert row["requested_trials"] == 1
    assert row["completed_trials"] == 0
    assert row["failed_trials"] == 1


def test_summary_matches_moved_episode_by_trial_identity(tmp_path, monkeypatch):
    run = tmp_path / "moved" / "seed_000"
    private = run / "_private_evaluation"
    private.mkdir(parents=True)
    (private / "latest_observation.json").write_text('{"goal_satisfied": false}')
    (run / "episode_result.json").write_text(json.dumps({
        "baseline": "vlm_tamp",
        "environment": "workshop",
        "variant": "W1",
        "seed": 0,
        "camera_count": 5,
        "result": {"model_calls": 1},
        "gt_comparison": {
            "expected_outcome": "FEASIBLE",
            "predicted_outcome": "UNRESOLVED",
        },
    }))
    (tmp_path / "batch_summary.json").write_text(json.dumps({"runs": [{
        "method": "vlm_tamp",
        "environment": "workshop",
        "variant": "W1",
        "camera_count": 5,
        "seed": 0,
        "output_dir": "/old/location/that/no/longer/exists",
    }]}))
    report = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_plan_gt_batch", str(tmp_path), "--output-dir", str(report)],
    )

    main()

    row = json.loads((report / "table4.json").read_text())["rows"][0]
    assert row["requested_trials"] == 1
    assert row["failed_trials"] == 0


def test_physical_episode_is_not_scored_as_a_planning_trial(tmp_path, monkeypatch):
    run = tmp_path / "physical" / "seed_000"
    run.mkdir(parents=True)
    (run / "episode_result.json").write_text(json.dumps({
        "baseline": "vlm_tamp",
        "environment": "kitchen",
        "variant": "K1",
        "seed": 0,
        "camera_count": 5,
        "physical_execution": True,
        "planning_rounds": 3,
        "raw_vlm_requests": 6,
        "result": {"status": "MODEL_CALL_BUDGET_EXHAUSTED"},
    }))
    report = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_plan_gt_batch", str(tmp_path), "--output-dir", str(report)],
    )

    with pytest.raises(SystemExit):
        main()

    assert not (report / "table4.json").exists()


def test_unknown_baseline_name_is_not_reported_as_vlm_tamp(tmp_path, monkeypatch):
    run = tmp_path / "unknown" / "seed_000"
    run.mkdir(parents=True)
    (run / "episode_result.json").write_text(json.dumps({
        "baseline": "some_other_method",
        "environment": "kitchen",
        "variant": "K1",
        "seed": 0,
        "camera_count": 5,
        "gt_comparison": {
            "expected_outcome": "FEASIBLE",
            "predicted_outcome": "FEASIBLE",
        },
    }))
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_plan_gt_batch", str(tmp_path),
            "--output-dir", str(tmp_path / "report"),
        ],
    )

    with pytest.raises(SystemExit, match="some_other_method"):
        main()


def test_other_experiment_class_rows_do_not_abort_the_planning_table(tmp_path, monkeypatch):
    """A discovery batch under the same root must not kill Table 4 generation."""
    (tmp_path / "disco").mkdir()
    (tmp_path / "disco" / "batch_summary.json").write_text(json.dumps({"runs": [{
        "method": "discovery_replanning",
        "environment": "living_room",
        "variant": "L1",
        "camera_count": 5,
        "seed": 0,
        "output_dir": str(tmp_path / "disco" / "L1"),
    }]}))
    run = tmp_path / "plan" / "seed_000"
    run.mkdir(parents=True)
    (run / "episode_result.json").write_text(json.dumps({
        "baseline": "vlm_tamp",
        "environment": "living_room",
        "variant": "L1",
        "seed": 0,
        "camera_count": 5,
        "gt_comparison": {
            "expected_outcome": "FEASIBLE",
            "predicted_outcome": "FEASIBLE",
        },
    }))
    report = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_plan_gt_batch", str(tmp_path), "--output-dir", str(report)],
    )

    main()

    rows = json.loads((report / "table4.json").read_text())["rows"]
    assert len(rows) == 1
    assert rows[0]["method"] == "VLM-TAMP"
    assert rows[0]["completed_trials"] == 1
    assert rows[0]["failed_trials"] == 0
