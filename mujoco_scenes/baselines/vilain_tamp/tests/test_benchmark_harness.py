import json
from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.benchmark_harness import (
    PROTOCOLS,
    REPEATS,
    _semantic_status,
    authoritative_variants,
    build_schedule,
    prepare_manifest,
)


def _config_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def test_authoritative_matrix_has_every_registry_variant_and_repeat() -> None:
    variants = authoritative_variants(_config_root())
    assert {domain: len(rows) for domain, rows in variants.items()} == {
        "kitchen": 12,
        "living_room": 10,
        "workshop": 10,
    }
    schedule = build_schedule(_config_root())
    assert len(schedule) == 32 * len(PROTOCOLS) * REPEATS == 320
    assert len({item.run_id for item in schedule}) == len(schedule)
    assert len({item.seed for item in schedule}) == len(schedule)


def test_schedule_manifest_is_immutable_and_records_frozen_runtime(tmp_path: Path) -> None:
    schedule = prepare_manifest(tmp_path, source_commit="a" * 40, config_root=_config_root())
    first = (tmp_path / "schedule_manifest.json").read_bytes()
    prepare_manifest(tmp_path, source_commit="a" * 40, config_root=_config_root())
    assert (tmp_path / "schedule_manifest.json").read_bytes() == first
    payload = json.loads(first)
    assert payload["runtime"]["repeat_count"] == 5
    assert payload["runtime"]["cp_limit"] == 3
    assert payload["model"]["served_revision"] == "unverified"
    assert len(schedule) == 320
    with pytest.raises(RuntimeError, match="immutable schedule"):
        prepare_manifest(tmp_path, source_commit="b" * 40, config_root=_config_root())


def test_terminal_status_keeps_generated_and_hidden_outcomes_distinct() -> None:
    result = {
        "run_status": "SUCCESS",
        "generated_goal_status": "SUCCESS",
        "benchmark_status": "FAILURE",
        "metrics": {
            "generated_goal_satisfied": True,
            "actual_benchmark_success": False,
        },
    }
    assert _semantic_status(result) == "SUCCESS"
    assert result["metrics"]["generated_goal_satisfied"] is True
    assert result["metrics"]["actual_benchmark_success"] is False
