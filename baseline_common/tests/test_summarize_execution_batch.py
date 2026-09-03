from __future__ import annotations

import json

from baseline_common.summarize_execution_batch import main


def test_execution_summary_keeps_protocols_separate(tmp_path, monkeypatch):
    for protocol, success, calls in (("native", True, 3), ("single_call", False, 1)):
        run = tmp_path / protocol
        run.mkdir()
        (run / "discovery_replanning_result.json").write_text(json.dumps({
            "scene": "living_room",
            "method": "discovery_replanning",
            "protocol": protocol,
            "camera_count": 5,
            "success": success,
            "executed_actions": 2,
            "model_calls": calls,
            "raw_vlm_requests": calls,
            "replans": calls - 1,
            "planning_latency_s": 1.0,
            "elapsed_seconds": 2.0,
        }))
    output = tmp_path / "summary"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_execution_batch", str(tmp_path), "--output-dir", str(output)],
    )

    main()

    rows = json.loads((output / "execution_summary.json").read_text())["rows"]
    assert {row["protocol"] for row in rows} == {"native", "single_call"}
    assert {row["success_percent"] for row in rows} == {0.0, 100.0}
