from __future__ import annotations

import json

import pytest

from baseline_common.physical_benchmark import write_execution_result


def test_writes_shared_physical_execution_artifact(tmp_path):
    result = write_execution_result(
        tmp_path,
        scene="kitchen",
        method="vlm_tamp",
        protocol="native",
        variant="K1",
        camera_count=5,
        seed=3,
        success=True,
        executed_actions=12,
        model_calls=2,
        raw_vlm_requests=4,
        replans=1,
        planning_latency_s=1.25,
        elapsed_seconds=7.5,
        terminal_status="GOAL_COMPLETE",
    )

    saved = json.loads((tmp_path / "benchmark_execution_result.json").read_text())
    assert result == saved
    assert saved["physical_execution"] is True
    assert saved["raw_vlm_requests"] == 4


def test_rejects_planning_only_camera_count(tmp_path):
    with pytest.raises(ValueError, match="camera_count"):
        write_execution_result(
            tmp_path,
            scene="kitchen",
            method="owl_tamp",
            protocol="native",
            variant="K1",
            camera_count=2,
            seed=0,
            success=False,
            executed_actions=0,
            model_calls=1,
            raw_vlm_requests=1,
            replans=0,
            planning_latency_s=0.0,
            elapsed_seconds=0.0,
            terminal_status="NO_PLAN",
        )
