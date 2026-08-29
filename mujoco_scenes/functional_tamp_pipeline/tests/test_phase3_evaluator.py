"""Unit tests for Phase 3.3 evaluator orchestration, command construction, and aggregation."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluate_functional_tamp_variants import (
    DOMAINS,
    EXPECTED,
    aggregate_random_trials,
    build_runner_command,
    compute_file_sha256,
    evaluate_variant,
    load_grounding_audit,
    load_grounding_info,
    load_plan_validation,
    load_run_manifest,
    main,
    parse_random_seeds,
)


# 1. Command Construction Tests
def test_build_runner_command_default():
    cmd = build_runner_command(
        domain="kitchen",
        variant="K1",
        mode="gt",
        runner_output_root="/tmp/out",
    )
    assert cmd == [
        sys.executable,
        "-m", "mujoco_scenes.functional_tamp_pipeline.run",
        "--domain", "kitchen",
        "--variant", "K1",
        "--mode", "gt",
        "--dry-run",
        "--output-root", "/tmp/out",
    ]
    assert "--search-order" not in cmd
    assert "--search-seed" not in cmd
    assert "--specification-json" not in cmd
    assert "--visualize" not in cmd


def test_build_runner_command_gt_oracle():
    cmd = build_runner_command(
        domain="kitchen",
        variant="K1",
        mode="gt",
        runner_output_root="/tmp/out",
        search_order="oracle",
    )
    assert "--search-order" in cmd
    idx = cmd.index("--search-order")
    assert cmd[idx + 1] == "oracle"
    assert "--search-seed" not in cmd


def test_build_runner_command_vlm_provider():
    cmd = build_runner_command(
        domain="workshop",
        variant="W1",
        mode="vlm",
        runner_output_root="/tmp/out",
        search_order="provider",
    )
    assert "--mode" in cmd
    assert cmd[cmd.index("--mode") + 1] == "vlm"
    assert "--search-order" in cmd
    assert cmd[cmd.index("--search-order") + 1] == "provider"


def test_build_runner_command_vlm_random_replay():
    cmd = build_runner_command(
        domain="kitchen",
        variant="K2",
        mode="vlm",
        runner_output_root="/tmp/out/trials/random/seed_000",
        search_order="random",
        search_seed=0,
        specification_json="/tmp/saved_spec/spec.json",
    )
    assert "--mode" in cmd
    assert cmd[cmd.index("--mode") + 1] == "vlm"
    assert "--search-order" in cmd
    assert cmd[cmd.index("--search-order") + 1] == "random"
    assert "--search-seed" in cmd
    assert cmd[cmd.index("--search-seed") + 1] == "0"
    assert "--specification-json" in cmd
    assert cmd[cmd.index("--specification-json") + 1] == "/tmp/saved_spec/spec.json"


def test_build_runner_command_seed_9():
    cmd = build_runner_command(
        domain="workshop",
        variant="W2",
        mode="gt",
        runner_output_root="/tmp/out/trials/random/seed_009",
        search_order="random",
        search_seed=9,
    )
    assert "--search-seed" in cmd
    assert cmd[cmd.index("--search-seed") + 1] == "9"


# 2. Random Seed Parsing Tests
def test_parse_random_seeds_valid():
    seeds = parse_random_seeds("0,1,2,3,4,5,6,7,8,9")
    assert seeds == list(range(10))


def test_parse_random_seeds_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate seed"):
        parse_random_seeds("0,1,2,2,3")


def test_parse_random_seeds_rejects_negative():
    with pytest.raises(ValueError, match="Negative seed"):
        parse_random_seeds("0,1,-3")


def test_parse_random_seeds_rejects_invalid_int():
    with pytest.raises(ValueError, match="Invalid integer"):
        parse_random_seeds("0,foo,2")


def test_parse_random_seeds_rejects_empty():
    with pytest.raises(ValueError, match="Empty --random-seeds"):
        parse_random_seeds(" , ")


# 3. Preflight Rejection Tests
def test_preflight_vlm_oracle_rejected(tmp_path: Path):
    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--mode", "vlm",
        "--search-order", "oracle",
    ]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_preflight_living_room_oracle_rejected(tmp_path: Path):
    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "living_room",
        "--search-order", "oracle",
    ]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_preflight_living_room_random_rejected(tmp_path: Path):
    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "living_room",
        "--search-order", "random",
        "--search-seed", "0",
    ]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_preflight_random_without_seeds_rejected(tmp_path: Path):
    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "kitchen",
        "--search-order", "random",
    ]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_preflight_non_random_with_seed_rejected(tmp_path: Path):
    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "kitchen",
        "--search-order", "oracle",
        "--search-seed", "0",
    ]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_preflight_both_seed_and_seeds_rejected(tmp_path: Path):
    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "kitchen",
        "--search-order", "random",
        "--search-seed", "0",
        "--random-seeds", "0,1,2",
    ]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_preflight_vlm_multi_seed_without_spec_root_rejected(tmp_path: Path):
    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "kitchen",
        "--mode", "vlm",
        "--search-order", "random",
        "--random-seeds", "0,1,2",
    ]
    with patch("sys.argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_preflight_missing_replay_spec_aborts_before_subprocess(tmp_path: Path):
    spec_root = tmp_path / "saved_specs"
    spec_root.mkdir()
    # K1 exists, K2 is missing
    k1_dir = spec_root / "kitchen" / "K1" / "gt"
    k1_dir.mkdir(parents=True)
    (k1_dir / "functional_specification.json").write_text("{}", encoding="utf-8")

    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "kitchen",
        "--variants", "K1,K2",
        "--specification-root", str(spec_root),
    ]

    with patch("sys.argv", test_args), \
         patch("subprocess.run") as mock_subproc, \
         pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    # Exactly ZERO subprocesses must be launched
    mock_subproc.assert_not_called()


# 4. Output Isolation Test
def test_random_trial_output_isolation(tmp_path: Path):
    output_root = tmp_path / "eval_out"
    seeds = [0, 1, 9]

    run_dirs = []
    for s in seeds:
        trial_root = str(output_root / "trials" / "random" / f"seed_{s:03d}")
        run_dir = os.path.join(trial_root, "kitchen", "K2", "gt")
        run_dirs.append(run_dir)

    # Assert distinct output roots and run directories
    assert len(set(run_dirs)) == 3
    assert "seed_000" in run_dirs[0]
    assert "seed_001" in run_dirs[1]
    assert "seed_009" in run_dirs[2]


# 5. Manifest Loading and Provenance Test
def test_manifest_provenance_loaded_in_result(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)

    result_json = run_dir / "result.json"
    result_json.write_text(json.dumps({
        "status": "ACTION_SEQUENCE_READY",
        "inspected_regions": ["C2"],
        "plan": [{"operator": "PICK"}],
    }), encoding="utf-8")

    manifest_json = run_dir / "run_manifest.json"
    manifest_json.write_text(json.dumps({
        "spec_mode": "gt",
        "spec_acquisition": "GT_SPEC_PROVIDER",
        "specification_sha256": "abcdef123456",
        "search_order_source_effective": "oracle",
        "search_seed_effective": None,
        "provider_region_ranking": ["C2", "B1"],
        "region_order_used": ["C2", "B1"],
        "exploration_actuation": "robot_physical",
    }), encoding="utf-8")

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0)
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), search_order="oracle")

    assert res["search_order_effective"] == "oracle"
    assert res["provider_region_ranking"] == ["C2", "B1"]
    assert res["region_order_used"] == ["C2", "B1"]
    assert res["exploration_actuation"] == "robot_physical"
    assert res["spec_acquisition"] == "GT_SPEC_PROVIDER"
    assert res["specification_sha256"] == "abcdef123456"
    assert res["n_open"] == 1


# 6. Replay Hash Pairing Test
def test_replay_hash_pairing_match_and_mismatch(tmp_path: Path):
    spec_file = tmp_path / "source_spec.json"
    spec_file.write_text(json.dumps({"domain": "kitchen", "task": "test"}), encoding="utf-8")
    expected_sha = compute_file_sha256(str(spec_file))

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")

    # Case A: Matching hash
    (run_dir / "run_manifest.json").write_text(json.dumps({"specification_sha256": expected_sha}), encoding="utf-8")
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_match = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res_match["specification_hash_match"] is True
    assert res_match["source_specification_sha256"] == expected_sha
    assert res_match["match"] == "YES"

    # Case B: Mismatched hash
    (run_dir / "run_manifest.json").write_text(json.dumps({"specification_sha256": "different_hash_value"}), encoding="utf-8")
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_mismatch = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res_mismatch["specification_hash_match"] is False
    assert res_mismatch["actual_status"] == "ERROR_SPEC_HASH_MISMATCH"
    assert res_mismatch["match"] == "NO"


# 7. Random Multi-Seed Aggregation Test
def test_random_aggregation_statistics(tmp_path: Path):
    # Construct 4 synthetic trial rows for (workshop, W1)
    synthetic_rows = [
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES",
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "specification_hash_match": True,
        },
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES",
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 2, "inspected_regions": ["DRAWER_LEFT", "TOOL_CABINET"], "runtime_sec": 2.0,
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 1, "specification_hash_match": True,
        },
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES",
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 3, "inspected_regions": ["DRAWER_RIGHT", "DRAWER_LEFT", "TOOL_CABINET"], "runtime_sec": 3.0,
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 2, "specification_hash_match": True,
        },
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES",
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 2, "inspected_regions": ["DRAWER_LEFT", "TOOL_CABINET"], "runtime_sec": 2.0,
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 3, "specification_hash_match": True,
        },
    ]

    out_dir = str(tmp_path / "agg_out")
    os.makedirs(out_dir)
    summary_data, json_p, csv_p = aggregate_random_trials(synthetic_rows, out_dir)

    agg = summary_data["aggregates"][0]
    assert agg["domain"] == "workshop"
    assert agg["variant"] == "W1"
    assert agg["n_trials"] == 4
    assert agg["n_valid_trials"] == 4
    assert agg["expected_match_rate"] == 1.0
    assert agg["grounding_complete_rate"] == 1.0

    # Inspection counts: [1, 2, 3, 2] -> mean = 2.0, median = 2.0, min = 1, max = 3
    # Population std: pstdev([1, 2, 3, 2]) = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2 + (2-2)^2)/4) = sqrt(2/4) = sqrt(0.5) ≈ 0.7071
    assert agg["inspection_count_mean"] == pytest.approx(2.0, 1e-4)
    assert agg["inspection_count_median"] == 2.0
    assert agg["inspection_count_min"] == 1
    assert agg["inspection_count_max"] == 3
    assert agg["inspection_count_std"] == pytest.approx(0.7071, 1e-4)

    # Runtime: [1.0, 2.0, 3.0, 2.0] -> mean = 2.0
    assert agg["runtime_mean"] == pytest.approx(2.0, 1e-4)
    assert agg["plan_length_mean"] == pytest.approx(2.0, 1e-4)
    assert agg["replay_valid_rate"] == 1.0
    assert agg["specification_hash_all_match"] is True

    # P(complete by k)
    p_k = agg["p_grounding_complete_by_k"]
    # k=0: 0/4 = 0.0; k=1: 1/4 = 0.25; k=2: 3/4 = 0.75; k=3: 4/4 = 1.0; k=4: 4/4 = 1.0
    assert p_k["0"] == 0.0
    assert p_k["1"] == 0.25
    assert p_k["2"] == 0.75
    assert p_k["3"] == 1.0
    assert p_k["4"] == 1.0

    assert os.path.isfile(json_p)
    assert os.path.isfile(csv_p)


# 8. CSV Backward Compatibility Test
def test_csv_structure_backward_compatibility(tmp_path: Path):
    out_root = tmp_path / "csv_test"

    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(out_root),
        "--domains", "kitchen",
        "--variants", "K1",
    ]

    mock_result = {
        "domain": "kitchen", "variant": "K1", "expected_status": "ACTION_SEQUENCE_READY",
        "actual_status": "ACTION_SEQUENCE_READY", "match": "YES", "return_code": 0,
        "completed": True, "runtime_sec": 1.23, "inspected_regions": ["C2"], "inspection_count": 1,
        "n_open": 1, "plan_length": 2, "combined_high_level_count": 3, "grounding_status": "COMPLETE",
        "grounding_complete": True, "grounding_audit_valid": True, "plan_replay_valid": True,
        "accessibility_valid": True, "failure_reason": None, "result_json_path": "/tmp/res.json",
        "graph_grounding_path": None, "audit_path": None, "spec_mode": "gt", "condition_label": "gt_oracle",
        "search_order_requested": "auto", "search_order_effective": "oracle", "search_seed_requested": None,
        "search_seed_effective": None, "provider_region_ranking": ["C2"], "region_order_used": ["C2"],
        "exploration_actuation": "robot_physical", "spec_acquisition": "GT_SPEC_PROVIDER",
        "specification_sha256": "abc", "source_specification_path": None, "source_specification_sha256": None,
        "specification_hash_match": None, "run_manifest_path": "/tmp/man.json", "trial_id": "kitchen_K1_gt",
        "trial_output_root": str(out_root), "run_dir": str(out_root / "kitchen" / "K1" / "gt"),
    }

    with patch("sys.argv", test_args), \
         patch("scripts.evaluate_functional_tamp_variants.evaluate_variant", return_value=mock_result):
        main()

    csv_path = out_root / "summary.csv"
    assert csv_path.is_file()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)

    # Assert legacy headers are first and in exact order
    legacy_expected = [
        "Domain", "Variant", "Expected", "Actual", "Match", "ReturnCode", "Completed",
        "Inspections", "InspectionSequence", "PlanLen", "CombinedCount", "Grounding",
        "GroundingComplete", "GroundingAuditValid", "PlanReplayValid", "AccessValid", "Runtime", "FailureReason"
    ]
    assert headers[:len(legacy_expected)] == legacy_expected

    # Assert additive Phase 3.3 headers are present
    assert "Mode" in headers
    assert "Condition" in headers
    assert "SearchOrderRequested" in headers
    assert "SearchOrderEffective" in headers
    assert "NOpen" in headers
    assert "SpecificationSHA256" in headers
    assert "SpecificationHashMatch" in headers
    assert "TrialID" in headers
