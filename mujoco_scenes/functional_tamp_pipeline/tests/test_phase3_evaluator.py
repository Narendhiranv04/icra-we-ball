"""Comprehensive unit tests for Phase 3.3 and 3.3.1 evaluator orchestration, provenance, and aggregation."""

import csv
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


# 2. Strict Random Seed Parsing Tests
def test_parse_random_seeds_valid():
    seeds = parse_random_seeds("0,1,2,3,4,5,6,7,8,9")
    assert seeds == list(range(10))

    seeds_with_spaces = parse_random_seeds("0, 1, 2")
    assert seeds_with_spaces == [0, 1, 2]


@pytest.mark.parametrize("bad_input", ["0,,2", "0,", ",0", "0, ,2", ""])
def test_parse_random_seeds_strict_rejects_empty_tokens(bad_input):
    with pytest.raises(ValueError, match="Empty"):
        parse_random_seeds(bad_input)


def test_parse_random_seeds_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate seed"):
        parse_random_seeds("0,1,2,2,3")


def test_parse_random_seeds_rejects_negative():
    with pytest.raises(ValueError, match="Negative seed"):
        parse_random_seeds("0,1,-3")


def test_parse_random_seeds_rejects_invalid_int():
    with pytest.raises(ValueError, match="Invalid integer"):
        parse_random_seeds("0,foo,2")


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

    assert len(set(run_dirs)) == 3
    assert "seed_000" in run_dirs[0]
    assert "seed_001" in run_dirs[1]
    assert "seed_009" in run_dirs[2]


# 5. Manifest Provenance & Consistency Test
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
        "spec_acquisition": "live_provider",
        "specification_sha256": "abcdef123456",
        "search_order_source_requested": "oracle",
        "search_order_source_effective": "oracle",
        "search_seed_requested": None,
        "search_seed_effective": None,
        "provider_region_ranking": ["C2", "B1"],
        "region_order_used": ["C2", "B1"],
        "exploration_actuation": "direct_sim_articulation",
        "terminal_status": "ACTION_SEQUENCE_READY",
    }), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), search_order="oracle")

    assert res["search_order_effective"] == "oracle"
    assert res["provider_region_ranking"] == ["C2", "B1"]
    assert res["region_order_used"] == ["C2", "B1"]
    assert res["exploration_actuation"] == "direct_sim_articulation"
    assert res["spec_acquisition"] == "live_provider"
    assert res["specification_sha256"] == "abcdef123456"
    assert res["provenance_match"] is True
    assert res["evaluation_valid"] is True
    assert res["valid_match"] == "YES"
    assert res["n_open"] == 1


# 6. Source Hash Pre-Subprocess and Verified Pairing
def test_replay_verified_pairing(tmp_path: Path):
    spec_file = tmp_path / "source_spec.json"
    spec_file.write_text(json.dumps({"domain": "kitchen", "task": "test"}), encoding="utf-8")
    expected_sha = compute_file_sha256(str(spec_file))

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "spec_mode": "gt",
        "search_order_source_requested": "auto",
        "search_order_source_effective": "oracle",
        "specification_sha256": expected_sha,
        "terminal_status": "ACTION_SEQUENCE_READY",
    }), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res["pairing_status"] == "VERIFIED"
    assert res["specification_hash_match"] is True
    assert res["source_specification_unchanged"] is True
    assert res["evaluation_valid"] is True
    assert res["valid_match"] == "YES"
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


# 7. Source Mutation Test
def test_source_mutation_detected_after_subprocess(tmp_path: Path):
    spec_file = tmp_path / "source_spec.json"
    spec_file.write_text(json.dumps({"version": 1}), encoding="utf-8")
    initial_sha = compute_file_sha256(str(spec_file))

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "spec_mode": "gt",
        "search_order_source_requested": "auto",
        "search_order_source_effective": "oracle",
        "specification_sha256": initial_sha,
        "terminal_status": "ACTION_SEQUENCE_READY",
    }), encoding="utf-8")

    def mutating_subprocess(*args, **kwargs):
        # Mutate the source file during run
        spec_file.write_text(json.dumps({"version": 2}), encoding="utf-8")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=mutating_subprocess):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res["source_specification_unchanged"] is False
    assert res["pairing_status"] == "SOURCE_CHANGED"
    assert res["evaluation_valid"] is False
    assert res["valid_match"] == "N/A"
    # Raw pipeline status must NOT be rewritten
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"
    assert res["match"] == "YES"


# 8. Manifest Missing / Hash Missing / Hash Mismatch Tests
def test_manifest_missing_invalidates_pairing(tmp_path: Path):
    spec_file = tmp_path / "source_spec.json"
    spec_file.write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res["pairing_status"] == "MANIFEST_MISSING"
    assert res["evaluation_valid"] is False
    assert res["valid_match"] == "N/A"
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


def test_manifest_hash_missing_invalidates_pairing(tmp_path: Path):
    spec_file = tmp_path / "source_spec.json"
    spec_file.write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"spec_mode": "gt"}), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res["pairing_status"] == "MANIFEST_HASH_MISSING"
    assert res["evaluation_valid"] is False
    assert res["valid_match"] == "N/A"
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


def test_replay_hash_mismatch_invalidates_pairing(tmp_path: Path):
    spec_file = tmp_path / "source_spec.json"
    spec_file.write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "spec_mode": "gt",
        "search_order_source_requested": "auto",
        "specification_sha256": "different_hash",
        "terminal_status": "ACTION_SEQUENCE_READY",
    }), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res["pairing_status"] == "HASH_MISMATCH"
    assert res["specification_hash_match"] is False
    assert res["evaluation_valid"] is False
    assert res["valid_match"] == "N/A"
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


# 9. Provenance Mismatch Tests
def test_provenance_mismatch_detection(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "vlm"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    # Manifest reports seed 1, but evaluator requested seed 0
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "spec_mode": "vlm",
        "search_order_source_requested": "random",
        "search_seed_requested": 1,
        "terminal_status": "ACTION_SEQUENCE_READY",
    }), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), mode="vlm", search_order="random", search_seed=0)

    assert res["provenance_match"] is False
    assert len(res["provenance_mismatches"]) == 1
    assert "search_seed: evaluator=0 != manifest=1" in res["provenance_mismatches"][0]
    assert res["evaluation_valid"] is False
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


def test_manifest_terminal_status_mismatch(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "spec_mode": "gt",
        "search_order_source_requested": "auto",
        "terminal_status": "INFEASIBLE",
    }), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"))

    assert res["provenance_match"] is False
    assert res["evaluation_valid"] is False
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


# 10. Dynamic K Range and Aggregation Tests
def test_dynamic_k_range_kitchen_5_regions(tmp_path: Path):
    synthetic_rows = [
        {
            "domain": "kitchen", "variant": "K1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 2, "inspected_regions": ["C2", "B1"], "runtime_sec": 1.0,
            "region_order_used": ["C2", "B1", "DRAWER", "FRIDGE", "CABINET"], # 5 regions
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
    ]

    out_dir = str(tmp_path / "agg_k")
    os.makedirs(out_dir)
    summary_data, _, _ = aggregate_random_trials(synthetic_rows, out_dir)

    agg = summary_data["aggregates"][0]
    p_k = agg["p_grounding_complete_by_k"]
    # Keys must be exactly "0", "1", "2", "3", "4", "5"
    assert set(p_k.keys()) == {"0", "1", "2", "3", "4", "5"}
    assert "6" not in p_k


def test_dynamic_k_range_workshop_3_regions(tmp_path: Path):
    synthetic_rows = [
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["TOOL_CABINET", "DRAWER_LEFT", "DRAWER_RIGHT"], # 3 regions
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
    ]

    out_dir = str(tmp_path / "agg_w")
    os.makedirs(out_dir)
    summary_data, _, _ = aggregate_random_trials(synthetic_rows, out_dir)

    agg = summary_data["aggregates"][0]
    p_k = agg["p_grounding_complete_by_k"]
    # Keys must be exactly "0", "1", "2", "3"
    assert set(p_k.keys()) == {"0", "1", "2", "3"}
    assert "4" not in p_k


def test_mixed_aggregate_csv_dynamic_columns(tmp_path: Path):
    synthetic_rows = [
        # Kitchen variant with 5 regions
        {
            "domain": "kitchen", "variant": "K1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 2, "inspected_regions": ["C2", "B1"], "runtime_sec": 1.0,
            "region_order_used": ["C2", "B1", "C3", "B2", "D1"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
        # Workshop variant with 3 regions
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["TOOL_CABINET", "DRAWER_LEFT", "DRAWER_RIGHT"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
    ]

    out_dir = str(tmp_path / "agg_mixed")
    os.makedirs(out_dir)
    _, _, csv_p = aggregate_random_trials(synthetic_rows, out_dir)

    with open(csv_p, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    # Headers must include PCompleteByK5
    assert "PCompleteByK5" in headers
    assert "PCompleteByK6" not in headers

    k_idx_4 = headers.index("PCompleteByK4")
    k_idx_5 = headers.index("PCompleteByK5")

    kitchen_row = [r for r in rows if r[0] == "kitchen"][0]
    workshop_row = [r for r in rows if r[0] == "workshop"][0]

    assert kitchen_row[k_idx_4] != "N/A"
    assert kitchen_row[k_idx_5] != "N/A"
    assert workshop_row[k_idx_4] == "N/A"
    assert workshop_row[k_idx_5] == "N/A"


def test_invalid_trial_excluded_from_numeric_stats(tmp_path: Path):
    synthetic_rows = [
        # Seed 0: valid, inspection_count = 1
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["TOOL_CABINET", "DRAWER_LEFT", "DRAWER_RIGHT"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "VERIFIED",
        },
        # Seed 1: invalid (e.g. hash mismatch), inspection_count = 99
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": False,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 99, "inspected_regions": ["DRAWER_RIGHT"], "runtime_sec": 9.0,
            "region_order_used": ["TOOL_CABINET", "DRAWER_LEFT", "DRAWER_RIGHT"],
            "plan_length": 99, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 1, "pairing_status": "HASH_MISMATCH",
        },
    ]

    out_dir = str(tmp_path / "agg_inv")
    os.makedirs(out_dir)
    summary_data, _, _ = aggregate_random_trials(synthetic_rows, out_dir)

    agg = summary_data["aggregates"][0]
    assert agg["n_trials"] == 2
    assert agg["n_valid_trials"] == 1
    assert agg["evaluation_invalid_count"] == 1

    # Mean inspection count over valid trials must be 1.0 (not (1+99)/2 = 50.0)
    assert agg["inspection_count_mean"] == 1.0
    assert agg["plan_length_mean"] == 2.0

    # P(complete by k) denominator remains n_trials=2
    # For k=1: 1 valid trial completed <= 1 -> 1 / 2 = 0.5
    assert agg["p_grounding_complete_by_k"]["1"] == 0.5


# 11. Summary & CSV Backward Compatibility Test
def test_summary_and_csv_structure_backward_compatibility(tmp_path: Path):
    out_root = tmp_path / "csv_test"

    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(out_root),
        "--domains", "kitchen",
        "--variants", "K1",
    ]

    mock_result = {
        "domain": "kitchen", "variant": "K1", "expected_status": "ACTION_SEQUENCE_READY",
        "actual_status": "ACTION_SEQUENCE_READY", "match": "YES", "valid_match": "YES", "return_code": 0,
        "completed": True, "runtime_sec": 1.23, "inspected_regions": ["C2"], "inspection_count": 1,
        "n_open": 1, "plan_length": 2, "combined_high_level_count": 3, "grounding_status": "COMPLETE",
        "grounding_complete": True, "grounding_audit_valid": True, "plan_replay_valid": True,
        "accessibility_valid": True, "failure_reason": None, "result_json_path": "/tmp/res.json",
        "graph_grounding_path": None, "audit_path": None, "spec_mode": "gt", "condition_label": "gt_oracle",
        "search_order_requested": "auto", "search_order_effective": "oracle", "search_seed_requested": None,
        "search_seed_effective": None, "evaluator_mode_requested": "gt", "evaluator_search_order_requested": "auto",
        "evaluator_search_seed_requested": None, "manifest_spec_mode": "gt", "manifest_search_order_requested": "auto",
        "manifest_search_order_effective": "oracle", "manifest_search_seed_requested": None,
        "manifest_search_seed_effective": None, "manifest_terminal_status": "ACTION_SEQUENCE_READY",
        "provenance_match": True, "provenance_mismatches": [], "provider_region_ranking": ["C2"],
        "region_order_used": ["C2"], "exploration_actuation": "direct_sim_articulation",
        "spec_acquisition": "live_provider", "specification_sha256": "abc", "source_specification_path": None,
        "source_specification_sha256": None, "source_specification_sha256_after": None,
        "source_specification_unchanged": None, "pairing_status": "NOT_APPLICABLE",
        "specification_hash_match": None, "evaluation_valid": True, "evaluation_failure_reason": None,
        "run_manifest_path": "/tmp/man.json", "trial_id": "kitchen_K1_gt", "trial_output_root": str(out_root),
        "run_dir": str(out_root / "kitchen" / "K1" / "gt"),
    }

    with patch("sys.argv", test_args), \
         patch("scripts.evaluate_functional_tamp_variants.evaluate_variant", return_value=mock_result):
        main()

    # Verify summary.json
    with open(out_root / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "attempted_count" in summary
    assert "completed_count" in summary
    assert "matching_count" in summary
    assert "exact_match_rate" in summary
    assert "scientifically_valid_count" in summary
    assert "scientifically_valid_rate" in summary
    assert summary["scientifically_valid_count"] == 1
    assert summary["scientifically_valid_rate"] == 1.0

    # Verify summary.csv
    with open(out_root / "summary.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

    legacy_expected = [
        "Domain", "Variant", "Expected", "Actual", "Match", "ReturnCode", "Completed",
        "Inspections", "InspectionSequence", "PlanLen", "CombinedCount", "Grounding",
        "GroundingComplete", "GroundingAuditValid", "PlanReplayValid", "AccessValid", "Runtime", "FailureReason"
    ]
    assert headers[:len(legacy_expected)] == legacy_expected
    assert "Mode" in headers
    assert "NOpen" in headers
    assert "EvaluationValid" in headers
    assert "ValidMatch" in headers
    assert "PairingStatus" in headers
    assert "ProvenanceMatch" in headers
