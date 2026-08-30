"""Comprehensive unit tests for Phase 3.3, 3.3.1, and 3.3.2 evaluator orchestration, provenance, and aggregation."""

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
    REQUIRED_MANIFEST_FIELDS,
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


# Helper to build a complete canonical manifest dictionary
def make_valid_manifest(
    *,
    spec_mode="gt",
    spec_acquisition="live_provider",
    search_order_source_requested="auto",
    search_order_source_effective="oracle",
    search_seed_requested=None,
    search_seed_effective=None,
    terminal_status="ACTION_SEQUENCE_READY",
    exploration_actuation="direct_sim_articulation",
    specification_sha256="abc123sha",
    provider_region_ranking=None,
    region_order_used=None,
):
    return {
        "spec_mode": spec_mode,
        "spec_acquisition": spec_acquisition,
        "search_order_source_requested": search_order_source_requested,
        "search_order_source_effective": search_order_source_effective,
        "search_seed_requested": search_seed_requested,
        "search_seed_effective": search_seed_effective,
        "terminal_status": terminal_status,
        "exploration_actuation": exploration_actuation,
        "specification_sha256": specification_sha256,
        "provider_region_ranking": provider_region_ranking or ["C2", "B1"],
        "region_order_used": region_order_used or ["C2", "B1"],
    }


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


def test_preflight_hash_failure_aborts_before_subprocess(tmp_path: Path):
    spec_root = tmp_path / "saved_specs"
    spec_root.mkdir()
    k1_dir = spec_root / "kitchen" / "K1" / "gt"
    k1_dir.mkdir(parents=True)
    spec_file = k1_dir / "functional_specification.json"
    spec_file.write_text("{}", encoding="utf-8")

    test_args = [
        "evaluate_functional_tamp_variants.py",
        "--output-root", str(tmp_path / "out"),
        "--domains", "kitchen",
        "--variants", "K1",
        "--specification-root", str(spec_root),
    ]

    with patch("sys.argv", test_args), \
         patch("scripts.evaluate_functional_tamp_variants.compute_file_sha256", side_effect=PermissionError("Cannot read")), \
         patch("subprocess.run") as mock_subproc, \
         pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
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


# 5. Non-replay Missing Manifest Invalidates Trial (Issue #1)
def test_non_replay_missing_manifest_invalidates_evaluation(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)

    result_json = run_dir / "result.json"
    result_json.write_text(json.dumps({
        "status": "ACTION_SEQUENCE_READY",
        "inspected_regions": ["C2"],
        "plan": [{"operator": "PICK"}],
    }), encoding="utf-8")
    # run_manifest.json is absent!

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"))

    # Raw pipeline science must remain intact
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"
    assert res["completed"] is True
    assert res["match"] == "YES"

    # Evaluator validity must be marked invalid
    assert res["provenance_match"] is False
    assert res["evaluation_valid"] is False
    assert res["valid_match"] == "N/A"
    assert "run manifest missing or unreadable" in res["evaluation_failure_reason"]


# 6. Spec Acquisition Validation Tests (Issue #2)
def test_spec_acquisition_provenance_live_and_replay_valid(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")

    # Case A: Live provider run
    manifest = make_valid_manifest(spec_acquisition="live_provider")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_live = evaluate_variant("kitchen", "K1", str(tmp_path / "out"))

    assert res_live["expected_spec_acquisition"] == "live_provider"
    assert res_live["manifest_spec_acquisition"] == "live_provider"
    assert res_live["provenance_match"] is True
    assert res_live["evaluation_valid"] is True

    # Case B: Replayed provider output run
    spec_file = tmp_path / "spec.json"
    spec_file.write_text("{}", encoding="utf-8")
    sha_val = compute_file_sha256(str(spec_file))

    manifest_replay = make_valid_manifest(
        spec_acquisition="replayed_provider_output",
        specification_sha256=sha_val,
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest_replay), encoding="utf-8")
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_replay = evaluate_variant(
            "kitchen", "K1", str(tmp_path / "out"),
            specification_json=str(spec_file),
            source_specification_sha256=sha_val,
        )

    assert res_replay["expected_spec_acquisition"] == "replayed_provider_output"
    assert res_replay["manifest_spec_acquisition"] == "replayed_provider_output"
    assert res_replay["provenance_match"] is True
    assert res_replay["evaluation_valid"] is True
    assert res_replay["pairing_status"] == "VERIFIED"


def test_spec_acquisition_mismatch_invalidates_evaluation(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")

    # Replay requested, but manifest says live_provider
    spec_file = tmp_path / "spec.json"
    spec_file.write_text("{}", encoding="utf-8")
    sha_val = compute_file_sha256(str(spec_file))

    manifest = make_valid_manifest(spec_acquisition="live_provider", specification_sha256=sha_val)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant(
            "kitchen", "K1", str(tmp_path / "out"),
            specification_json=str(spec_file),
            source_specification_sha256=sha_val,
        )

    assert res["provenance_match"] is False
    assert res["evaluation_valid"] is False
    assert any("spec_acquisition" in m for m in res["provenance_mismatches"])


def test_spec_acquisition_reverse_mismatch_live_requested_manifest_replay(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")

    # Live run requested (specification_json=None), but manifest says replayed_provider_output
    manifest = make_valid_manifest(spec_acquisition="replayed_provider_output")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"))

    assert res["provenance_match"] is False
    assert res["evaluation_valid"] is False
    assert any("spec_acquisition" in m for m in res["provenance_mismatches"])


# 7. Seed Provenance Comparison with None (Issue #3)
def test_seed_provenance_comparison_with_none(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")

    # Evaluator seed=None, manifest seed=7 -> mismatch!
    manifest_mismatch = make_valid_manifest(search_seed_requested=7)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest_mismatch), encoding="utf-8")
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_mismatch = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), search_seed=None)

    assert res_mismatch["provenance_match"] is False
    assert res_mismatch["evaluation_valid"] is False
    assert any("search_seed" in m for m in res_mismatch["provenance_mismatches"])

    # Evaluator seed=None, manifest seed=None -> match!
    manifest_match = make_valid_manifest(search_seed_requested=None)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest_match), encoding="utf-8")
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_match = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), search_seed=None)

    assert res_match["provenance_match"] is True
    assert res_match["evaluation_valid"] is True


# 8. Post-run Hash Read Error vs Source Changed & Direct Pre-Hash Failure (Issue #4)
def test_direct_evaluate_variant_pre_hash_failure_blocks_subprocess(tmp_path: Path):
    nonexistent_file = str(tmp_path / "does_not_exist.json")
    with patch("subprocess.run") as mock_subproc:
        with pytest.raises(RuntimeError, match="Failed to read/hash replay specification before subprocess"):
            evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=nonexistent_file)
        mock_subproc.assert_not_called()


def test_post_run_hash_read_failure_classified_as_error_after(tmp_path: Path):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text("{}", encoding="utf-8")
    initial_sha = compute_file_sha256(str(spec_file))

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    manifest = make_valid_manifest(spec_acquisition="replayed_provider_output", specification_sha256=initial_sha)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def deleting_subprocess(*args, **kwargs):
        # File is deleted or unreadable post-run
        os.remove(str(spec_file))
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=deleting_subprocess):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res["pairing_status"] == "SOURCE_HASH_ERROR_AFTER"
    assert res["source_specification_unchanged"] is None
    assert res["evaluation_valid"] is False
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


def test_source_changed_when_hashes_differ(tmp_path: Path):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text("{\"v\": 1}", encoding="utf-8")
    initial_sha = compute_file_sha256(str(spec_file))

    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    manifest = make_valid_manifest(spec_acquisition="replayed_provider_output", specification_sha256=initial_sha)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def mutating_subprocess(*args, **kwargs):
        spec_file.write_text("{\"v\": 2}", encoding="utf-8")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=mutating_subprocess):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"), specification_json=str(spec_file))

    assert res["pairing_status"] == "SOURCE_CHANGED"
    assert res["source_specification_unchanged"] is False
    assert res["evaluation_valid"] is False
    assert res["actual_status"] == "ACTION_SEQUENCE_READY"


def test_manifest_missing_required_field_invalidates_trial(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")
    manifest = make_valid_manifest()
    del manifest["terminal_status"] # Missing critical field!
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res = evaluate_variant("kitchen", "K1", str(tmp_path / "out"))

    assert res["provenance_match"] is False
    assert res["evaluation_valid"] is False
    assert any("manifest missing field: terminal_status" in m for m in res["provenance_mismatches"])


def test_manifest_empty_field_invalidates_trial(tmp_path: Path):
    run_dir = tmp_path / "out" / "kitchen" / "K1" / "gt"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "ACTION_SEQUENCE_READY", "inspected_regions": []}), encoding="utf-8")

    # Case A: specification_sha256 is None
    manifest_a = make_valid_manifest()
    manifest_a["specification_sha256"] = None
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest_a), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_a = evaluate_variant("kitchen", "K1", str(tmp_path / "out"))

    assert res_a["provenance_match"] is False
    assert res_a["evaluation_valid"] is False
    assert any("manifest empty field: specification_sha256" in m for m in res_a["provenance_mismatches"])

    # Case B: terminal_status is empty string ""
    manifest_b = make_valid_manifest()
    manifest_b["terminal_status"] = "   "
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest_b), encoding="utf-8")

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        res_b = evaluate_variant("kitchen", "K1", str(tmp_path / "out"))

    assert res_b["provenance_match"] is False
    assert res_b["evaluation_valid"] is False
    assert any("manifest empty field: terminal_status" in m for m in res_b["provenance_mismatches"])


# 9. Dynamic K Range with Canonical Region IDs (Issue #5 & #6)
def test_dynamic_k_range_kitchen_canonical_ids(tmp_path: Path):
    # Canonical Kitchen IDs: ["D1", "D2", "C2", "B1", "C1"] (5 candidate regions)
    synthetic_rows = [
        {
            "domain": "kitchen", "variant": "K1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 2, "inspected_regions": ["C2", "B1"], "runtime_sec": 1.0,
            "region_order_used": ["D1", "D2", "C2", "B1", "C1"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
    ]

    out_dir = str(tmp_path / "agg_k_canonical")
    os.makedirs(out_dir)
    summary_data, _, _ = aggregate_random_trials(synthetic_rows, out_dir)

    agg = summary_data["aggregates"][0]
    p_k = agg["p_grounding_complete_by_k"]
    # Keys must be exactly "0", "1", "2", "3", "4", "5"
    assert set(p_k.keys()) == {"0", "1", "2", "3", "4", "5"}
    assert "6" not in p_k


def test_dynamic_k_range_workshop_canonical_ids(tmp_path: Path):
    # Canonical Workshop IDs: ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"] (3 candidate regions)
    synthetic_rows = [
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
    ]

    out_dir = str(tmp_path / "agg_w_canonical")
    os.makedirs(out_dir)
    summary_data, _, _ = aggregate_random_trials(synthetic_rows, out_dir)

    agg = summary_data["aggregates"][0]
    p_k = agg["p_grounding_complete_by_k"]
    # Keys must be exactly "0", "1", "2", "3"
    assert set(p_k.keys()) == {"0", "1", "2", "3"}
    assert "4" not in p_k


def test_mixed_aggregate_csv_dynamic_columns_and_canonical_ids(tmp_path: Path):
    synthetic_rows = [
        # Kitchen variant with 5 canonical regions
        {
            "domain": "kitchen", "variant": "K1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 2, "inspected_regions": ["C2", "B1"], "runtime_sec": 1.0,
            "region_order_used": ["D1", "D2", "C2", "B1", "C1"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
        # Workshop variant with 3 canonical regions
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE",
        },
    ]

    out_dir = str(tmp_path / "agg_mixed_canonical")
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


# 10. Aggregation with Zero Valid Trials vs Mixed Trials (Issue #5)
def test_zero_valid_trials_reports_none_and_na(tmp_path: Path):
    synthetic_rows = [
        # Trial 0: invalid
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": False,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 5, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
            "plan_length": 20, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "HASH_MISMATCH",
        },
        # Trial 1: invalid
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": False,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 8, "inspected_regions": ["LEFT_DRAWER"], "runtime_sec": 3.0,
            "region_order_used": ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
            "plan_length": 30, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 1, "pairing_status": "HASH_MISMATCH",
        },
    ]

    out_dir = str(tmp_path / "agg_zero_valid")
    os.makedirs(out_dir)
    summary_data, json_p, csv_p = aggregate_random_trials(synthetic_rows, out_dir)

    agg = summary_data["aggregates"][0]
    assert agg["n_trials"] == 2
    assert agg["n_valid_trials"] == 0
    assert agg["evaluation_invalid_count"] == 2

    # Scientific valid-only stats must be None (not 0.0)
    assert agg["inspection_count_mean"] is None
    assert agg["inspection_count_std"] is None
    assert agg["inspection_count_median"] is None
    assert agg["inspection_count_min"] is None
    assert agg["inspection_count_max"] is None
    assert agg["plan_length_mean"] is None
    assert agg["plan_length_std"] is None
    assert agg["runtime_mean"] is None
    assert agg["runtime_std"] is None

    # Attempt-level runtime statistics are computed over all rows
    assert agg["runtime_attempt_mean"] == 2.0
    assert agg["runtime_attempt_std"] == 1.0

    # CSV verification: valid-only numeric columns must be "N/A"
    with open(csv_p, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)

    assert row[headers.index("InspectionCountMean")] == "N/A"
    assert row[headers.index("PlanLengthMean")] == "N/A"
    assert row[headers.index("RuntimeMean")] == "N/A"
    assert row[headers.index("RuntimeAttemptMean")] == "2.0000"
    assert row[headers.index("SpecificationHashAllMatch")] == "False"


def test_non_replay_aggregate_csv_specification_hash_all_match_na(tmp_path: Path):
    synthetic_rows = [
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "NOT_APPLICABLE", # Non-replay!
        },
    ]

    out_dir = str(tmp_path / "agg_non_replay")
    os.makedirs(out_dir)
    summary_data, json_p, csv_p = aggregate_random_trials(synthetic_rows, out_dir)

    assert summary_data["aggregates"][0]["specification_hash_all_match"] is None

    with open(csv_p, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)

    # Must be "N/A", not "None"
    assert row[headers.index("SpecificationHashAllMatch")] == "N/A"


def test_invalid_trial_excluded_from_numeric_stats(tmp_path: Path):
    synthetic_rows = [
        # Seed 0: valid, inspection_count = 1
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": True,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 1, "inspected_regions": ["TOOL_CABINET"], "runtime_sec": 1.0,
            "region_order_used": ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
            "plan_length": 2, "plan_replay_valid": True, "grounding_complete": True,
            "search_seed_requested": 0, "pairing_status": "VERIFIED",
        },
        # Seed 1: invalid (e.g. hash mismatch), inspection_count = 99
        {
            "domain": "workshop", "variant": "W1", "completed": True, "match": "YES", "evaluation_valid": False,
            "actual_status": "ACTION_SEQUENCE_READY", "expected_status": "ACTION_SEQUENCE_READY",
            "inspection_count": 99, "inspected_regions": ["RIGHT_DRAWER"], "runtime_sec": 9.0,
            "region_order_used": ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"],
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
        "evaluator_search_seed_requested": None, "expected_spec_acquisition": "live_provider",
        "manifest_spec_mode": "gt", "manifest_spec_acquisition": "live_provider",
        "manifest_search_order_requested": "auto", "manifest_search_order_effective": "oracle",
        "manifest_search_seed_requested": None, "manifest_search_seed_effective": None,
        "manifest_terminal_status": "ACTION_SEQUENCE_READY", "provenance_match": True, "provenance_mismatches": [],
        "provider_region_ranking": ["C2"], "region_order_used": ["C2"],
        "exploration_actuation": "direct_sim_articulation", "spec_acquisition": "live_provider",
        "specification_sha256": "abc", "source_specification_path": None, "source_specification_sha256": None,
        "source_specification_sha256_after": None, "source_specification_unchanged": None,
        "pairing_status": "NOT_APPLICABLE", "specification_hash_match": None, "evaluation_valid": True,
        "evaluation_failure_reason": None, "run_manifest_path": "/tmp/man.json", "trial_id": "kitchen_K1_gt",
        "trial_output_root": str(out_root), "run_dir": str(out_root / "kitchen" / "K1" / "gt"),
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
    assert "ExpectedSpecAcquisition" in headers
    assert "ManifestSpecAcquisition" in headers
    assert "ProvenanceMatch" in headers
