"""The live harness must agree with the evaluator, and lose nothing.

These are the checks that would have caught the defects this file was written
for: an aggregator reading files the evaluator never writes, field names it
never uses, retries overwriting an earlier attempt's raw responses, and missing
trials quietly leaving the denominator.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
AGGREGATOR = REPO / "scripts" / "aggregate_live_repeats.py"
RUNNER = REPO / "scripts" / "run_live_repeat_experiment.py"


def _record(domain, variant, *, feasible, satisfied, coverage, outcome, false_completion=False):
    """One row shaped exactly as the evaluator writes it."""
    return {
        "domain": domain, "variant": variant,
        "gt_feasible": feasible,
        "full_task_satisfied": satisfied,
        "full_task_goal_coverage": coverage,
        "outcome_correct": outcome,
        "false_completion": false_completion,
    }


def _repeat(root: Path, name: str, records, *, finished=True, attempt=None):
    directory = root / name / attempt if attempt else root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation_records.json").write_text(json.dumps(records))
    (root / name / "repeat_manifest.json").write_text(
        json.dumps({"repeat_index": 1, "finished": finished}))
    return directory


def _aggregate(root: Path):
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR), "--root", str(root)],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    return json.loads((root / "AGGREGATE.json").read_text()), result.stdout


def test_aggregator_reads_the_evaluator_s_own_file_and_fields(tmp_path):
    """The exact filename and field names the evaluator writes, end to end."""
    _repeat(tmp_path, "repeat_01", [
        _record("kitchen", "K1", feasible=True, satisfied=True, coverage=1.0, outcome=True),
        _record("workshop", "W10", feasible=False, satisfied=False, coverage=0.0, outcome=True),
    ])
    _repeat(tmp_path, "repeat_02", [
        _record("kitchen", "K1", feasible=True, satisfied=False, coverage=0.5, outcome=False),
        _record("workshop", "W10", feasible=False, satisfied=False, coverage=0.0, outcome=True),
    ])
    data, _ = _aggregate(tmp_path)
    by_variant = {(row["domain"], row["variant"]): row for row in data["per_variant"]}

    kitchen = by_variant[("kitchen", "K1")]
    assert kitchen["feasible"] is True
    assert (kitchen["successes"], kitchen["repeats_observed"]) == (1, 2)
    assert kitchen["mean_gt_goal_coverage"] == 0.75
    assert kitchen["outcome_correct"] == 1

    infeasible = by_variant[("workshop", "W10")]
    assert infeasible["feasible"] is False
    assert infeasible["successes"] == 0
    assert infeasible["outcome_correct"] == 2
    assert infeasible["false_completions"] == 0


def test_a_file_the_evaluator_never_writes_is_not_read(tmp_path):
    """Guards the original defect: reading results.json and finding nothing."""
    directory = tmp_path / "repeat_01"
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps(
        [_record("kitchen", "K1", feasible=True, satisfied=True, coverage=1.0, outcome=True)]))
    (directory / "repeat_manifest.json").write_text(json.dumps({"finished": True}))
    data, _ = _aggregate(tmp_path)
    assert data["per_variant"] == []
    assert data["completeness"]["completed_trials"] == 0


def test_the_whole_grid_is_materialised_so_missing_trials_are_visible(tmp_path):
    """A trial that never ran is a hole, not an absence from the denominator."""
    _repeat(tmp_path, "repeat_01", [
        _record("kitchen", "K1", feasible=True, satisfied=True, coverage=1.0, outcome=True)])
    data, text = _aggregate(tmp_path)
    completeness = data["completeness"]
    assert completeness["intended_trial_slots"] == 32
    assert completeness["completed_trials"] == 1
    assert completeness["missing_trials"] == 31
    assert len(completeness["holes"]) == 31
    assert "EXPERIMENT COMPLETENESS" in text


def test_a_later_attempt_supersedes_an_earlier_one_without_erasing_it(tmp_path):
    """Immutable retries: both attempts survive, the later one is authoritative."""
    _repeat(tmp_path, "repeat_01", [
        _record("kitchen", "K1", feasible=True, satisfied=False, coverage=0.0, outcome=False)],
        attempt="attempt_01")
    _repeat(tmp_path, "repeat_01", [
        _record("kitchen", "K1", feasible=True, satisfied=True, coverage=1.0, outcome=True)],
        attempt="attempt_02")
    assert (tmp_path / "repeat_01" / "attempt_01" / "evaluation_records.json").is_file()
    data, _ = _aggregate(tmp_path)
    kitchen = next(row for row in data["per_variant"] if row["variant"] == "K1")
    assert (kitchen["successes"], kitchen["repeats_observed"]) == (1, 1)


def test_unfinished_repetitions_still_contribute_their_trials(tmp_path):
    """A repetition is never dropped for its outcome, only reported as unfinished."""
    _repeat(tmp_path, "repeat_01", [
        _record("kitchen", "K1", feasible=True, satisfied=False, coverage=0.25, outcome=False)],
        finished=False)
    data, _ = _aggregate(tmp_path)
    assert data["unfinished_repetitions"] == ["repeat_01"]
    assert data["completeness"]["completed_trials"] == 1


def test_runner_requires_an_exact_model_and_never_falls_back():
    """No silently benchmarking whatever the endpoint happens to serve."""
    source = RUNNER.read_text()
    assert "ids[0]" not in source, "a fallback to an arbitrary served model is present"
    assert "--model" in source, "the runner must take an explicit required model"
