"""Unit tests for offline Phase 3.4 VLM replay validator."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
)
from scripts.validate_phase3_vlm_replay import compute_file_sha256, validate_vlm_replay


def make_test_graph(
    domain: str = "kitchen",
    candidate_regions: tuple[str, ...] = ("D1", "D2", "C2", "B1", "C1"),
    region_ranking: tuple[str, ...] = ("C2", "B1", "D1", "D2", "C1"),
) -> FunctionalRequirementGraph:
    nodes = {
        "sponge": FunctionalRole(
            name="sponge",
            entity_kind="OBJECT",
            count=1,
            semantic_categories=("sponge",),
        )
    }
    return FunctionalRequirementGraph(
        domain=domain,
        task_instruction="wipe table",
        nodes=nodes,
        relations=(),
        detector_vocabulary=("sponge",),
        candidate_regions=candidate_regions,
        region_ranking=region_ranking,
        source="VLM_FUNCTIONAL_SPEC",
    )


def setup_pair_dirs(
    tmp_path: Path,
    *,
    live_graph: FunctionalRequirementGraph | None = None,
    replay_graph: FunctionalRequirementGraph | None = None,
    live_manifest_extra: dict | None = None,
    replay_manifest_extra: dict | None = None,
    live_result_extra: dict | None = None,
    replay_result_extra: dict | None = None,
    live_ggr_extra: dict | None = None,
    replay_ggr_extra: dict | None = None,
) -> tuple[Path, Path]:
    live_dir = tmp_path / "live_run"
    live_dir.mkdir(parents=True)
    replay_dir = tmp_path / "replay_run"
    replay_dir.mkdir(parents=True)

    g_live = live_graph or make_test_graph()
    g_replay = replay_graph or make_test_graph()

    live_spec_file = live_dir / "functional_specification.json"
    live_spec_file.write_text(json.dumps(g_live.to_dict()), encoding="utf-8")
    live_sha = compute_file_sha256(live_spec_file)

    replay_spec_file = replay_dir / "functional_specification.json"
    replay_spec_file.write_text(json.dumps(g_replay.to_dict()), encoding="utf-8")
    replay_sha = compute_file_sha256(replay_spec_file)

    # Live Manifest & Result
    l_man = {
        "domain": g_live.domain,
        "variant": "K2",
        "spec_mode": "vlm",
        "spec_acquisition": "live_provider",
        "search_order_source_requested": "provider",
        "search_order_source_effective": "provider",
        "search_seed_requested": None,
        "search_seed_effective": None,
        "terminal_status": "ACTION_SEQUENCE_READY",
        "exploration_actuation": "direct_sim_articulation",
        "specification_sha256": live_sha,
        "provider_region_ranking": list(g_live.region_ranking),
        "region_order_used": list(g_live.region_ranking),
    }
    if live_manifest_extra:
        l_man.update(live_manifest_extra)
    (live_dir / "run_manifest.json").write_text(json.dumps(l_man), encoding="utf-8")

    l_res = {
        "status": "ACTION_SEQUENCE_READY",
        "inspected_regions": ["C2"],
        "plan": [{"operator": "PICK"}],
    }
    if live_result_extra:
        l_res.update(live_result_extra)
    (live_dir / "result.json").write_text(json.dumps(l_res), encoding="utf-8")

    # Replay Manifest & Result
    r_man = {
        "domain": g_replay.domain,
        "variant": "K2",
        "spec_mode": "vlm",
        "spec_acquisition": "replayed_provider_output",
        "search_order_source_requested": "provider",
        "search_order_source_effective": "provider",
        "search_seed_requested": None,
        "search_seed_effective": None,
        "terminal_status": "ACTION_SEQUENCE_READY",
        "exploration_actuation": "direct_sim_articulation",
        "specification_sha256": replay_sha,
        "provider_region_ranking": list(g_replay.region_ranking),
        "region_order_used": list(g_replay.region_ranking),
    }
    if replay_manifest_extra:
        r_man.update(replay_manifest_extra)
    (replay_dir / "run_manifest.json").write_text(json.dumps(r_man), encoding="utf-8")

    r_res = {
        "status": "ACTION_SEQUENCE_READY",
        "inspected_regions": ["C2"],
        "plan": [{"operator": "PICK"}],
    }
    if replay_result_extra:
        r_res.update(replay_result_extra)
    (replay_dir / "result.json").write_text(json.dumps(r_res), encoding="utf-8")

    if live_ggr_extra is not None:
        (live_dir / "graph_grounding_result.json").write_text(json.dumps(live_ggr_extra), encoding="utf-8")
    if replay_ggr_extra is not None:
        (replay_dir / "graph_grounding_result.json").write_text(json.dumps(replay_ggr_extra), encoding="utf-8")

    return live_dir, replay_dir


# A. Live provider + exact provider replay -> PASS
def test_validator_live_provider_exact_replay_pass(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(tmp_path)
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is True
    assert details["success"] is True
    assert details["checks"]["exact_gf_sha"] == "PASS"
    assert details["checks"]["structural_gf_identity"] == "PASS"
    assert details["checks"]["deterministic_provider_replay"] == "PASS"


# B. Same G_F SHA mismatch -> FAIL
def test_validator_sha_mismatch_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(tmp_path)
    # Tamper with replay manifest SHA
    r_man = json.loads((replay_dir / "run_manifest.json").read_text(encoding="utf-8"))
    r_man["specification_sha256"] = "tampered_sha_123"
    (replay_dir / "run_manifest.json").write_text(json.dumps(r_man), encoding="utf-8")

    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["exact_gf_sha"] == "FAIL"


# C. Structural graph mismatch -> FAIL
def test_validator_structural_graph_mismatch_fail(tmp_path: Path):
    g_live = make_test_graph(domain="kitchen")
    g_replay = make_test_graph(domain="workshop")
    live_dir, replay_dir = setup_pair_dirs(tmp_path, live_graph=g_live, replay_graph=g_replay)

    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["exact_gf_sha"] == "FAIL"
    assert details["checks"]["structural_gf_identity"] == "FAIL"


# D. Replay says live_provider -> FAIL
def test_validator_replay_says_live_provider_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={"spec_acquisition": "live_provider"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["provenance"] == "FAIL"


# E. Random seed mismatch -> FAIL
def test_validator_random_seed_mismatch_fail(tmp_path: Path):
    g = make_test_graph()
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 0,
            "region_order_used": ["D1", "D2", "C2", "B1", "C1"],
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=1, # Expected 1, manifest had 0!
    )
    assert ok is False
    assert details["checks"]["provenance"] == "FAIL"


# F. Random order wrong candidate set -> FAIL
def test_validator_random_order_wrong_candidates_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 0,
            "region_order_used": ["D1", "WRONG_REGION", "C2", "B1", "C1"],
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=0,
    )
    assert ok is False
    assert details["checks"]["candidate_permutation"] == "FAIL"


# G. Random order contains duplicate -> FAIL
def test_validator_random_order_duplicate_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 0,
            "region_order_used": ["D1", "D1", "C2", "B1", "C1"],
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=0,
    )
    assert ok is False
    assert details["checks"]["candidate_permutation"] == "FAIL"


# H. Random order coincidentally equal to provider order -> PASS
def test_validator_random_order_coincidentally_equal_pass(tmp_path: Path):
    g = make_test_graph()
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 0,
            "region_order_used": list(g.region_ranking), # Coincidentally equals provider ranking!
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=0,
    )
    assert ok is True
    assert details["checks"]["candidate_permutation"] == "PASS"


# I. Provider replay assignment differs -> FAIL
def test_validator_provider_replay_assignment_differs_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_ggr_extra={"assignment": {"sponge": "sponge_1"}},
        replay_ggr_extra={"assignment": {"sponge": "sponge_2"}},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"


# J. Provider replay plan differs -> FAIL
def test_validator_provider_replay_plan_differs_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_result_extra={"plan": [{"operator": "PICK"}]},
        replay_result_extra={"plan": [{"operator": "PLACE"}]},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"


# K. Terminal status disagrees with manifest -> FAIL
def test_validator_terminal_status_disagrees_with_manifest_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_manifest_extra={"terminal_status": "INFEASIBLE"},
        live_result_extra={"status": "ACTION_SEQUENCE_READY"}, # Disagrees with manifest!
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["terminal_result_consistency"] == "FAIL"
