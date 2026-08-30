"""Comprehensive unit tests for offline Phase 3.4 / 3.4.1 / 3.4.2 VLM replay validator."""

from __future__ import annotations

import json
from pathlib import Path
import random
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
    source: str = "VLM_FUNCTIONAL_SPEC",
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
        source=source,
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
    include_default_ggr: bool = True,
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
        "spec_provider_source": g_live.source,
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
        "spec_provider_source": g_replay.source,
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

    # Default GGR artifacts for READY runs
    if include_default_ggr:
        default_ggr = {
            "status": "COMPLETE",
            "complete": True,
            "assignment": {"sponge": "sponge_obj_1"},
            "missing_roles": [],
        }
        (live_dir / "graph_grounding_result.json").write_text(
            json.dumps(live_ggr_extra if live_ggr_extra is not None else default_ggr),
            encoding="utf-8",
        )
        (replay_dir / "graph_grounding_result.json").write_text(
            json.dumps(replay_ggr_extra if replay_ggr_extra is not None else default_ggr),
            encoding="utf-8",
        )
    else:
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
    assert details["checks"]["vlm_source"] == "PASS"
    assert details["checks"]["identity"] == "PASS"
    assert details["checks"]["provider_order_used"] == "PASS"
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


# H. Random order coincidentally equal to provider order -> PASS if produced by seed
def test_validator_random_order_coincidentally_equal_pass(tmp_path: Path):
    g = make_test_graph()
    base = list(g.candidate_regions)
    rng = random.Random(42)
    rng.shuffle(base)
    g_matched = make_test_graph(candidate_regions=g.candidate_regions, region_ranking=tuple(base))

    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g_matched,
        replay_graph=g_matched,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 42,
            "search_seed_effective": 42,
            "region_order_used": base,
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=42,
    )
    assert ok is True
    assert details["checks"]["live_provider_baseline"] == "PASS"
    assert details["checks"]["candidate_permutation"] == "PASS"
    assert details["checks"]["random_seed_order"] == "PASS"


# I. Provider replay assignment differs -> FAIL
def test_validator_provider_replay_assignment_differs_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_ggr_extra={"status": "COMPLETE", "complete": True, "assignment": {"sponge": "sponge_1"}},
        replay_ggr_extra={"status": "COMPLETE", "complete": True, "assignment": {"sponge": "sponge_2"}},
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


# 12. Domain Mismatch -> FAIL (Issue #1)
def test_validator_domain_mismatch_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_manifest_extra={"domain": "kitchen"},
        replay_manifest_extra={"domain": "workshop"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["identity"] == "FAIL"
    assert any("domain mismatch" in f for f in details["failures"])


# 13. Variant Mismatch -> FAIL (Issue #2)
def test_validator_variant_mismatch_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_manifest_extra={"variant": "K1"},
        replay_manifest_extra={"variant": "K2"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["identity"] == "FAIL"
    assert any("variant mismatch" in f for f in details["failures"])


# 14. Graph Domain Mismatch -> FAIL (Section 7)
def test_validator_graph_domain_mismatch_fail(tmp_path: Path):
    g = make_test_graph(domain="kitchen")
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g,
        replay_graph=g,
        live_manifest_extra={"domain": "workshop"}, # Manifest says workshop, but graph is kitchen
        replay_manifest_extra={"domain": "workshop"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["structural_gf_identity"] == "FAIL"
    assert any("graph domain" in f for f in details["failures"])


# 15. VLM Source Tests (Issue #3)
def test_validator_vlm_source_live_not_vlm_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_manifest_extra={"spec_provider_source": "GT_FUNCTIONAL_SPEC"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["vlm_source"] == "FAIL"


def test_validator_vlm_source_replay_not_vlm_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={"spec_provider_source": "GT_FUNCTIONAL_SPEC"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["vlm_source"] == "FAIL"


def test_validator_vlm_source_graph_mismatch_fail(tmp_path: Path):
    g_gt = make_test_graph(source="GT_FUNCTIONAL_SPEC")
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g_gt,
        replay_graph=g_gt,
        live_manifest_extra={"spec_provider_source": "VLM_FUNCTIONAL_SPEC"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir)
    assert ok is False
    assert details["checks"]["vlm_source"] == "FAIL"


# 16. Provider Replay Used Order Mismatch -> FAIL (Issue #4)
def test_validator_provider_replay_used_order_mismatch_fail(tmp_path: Path):
    g = make_test_graph()
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g,
        replay_graph=g,
        replay_manifest_extra={
            "region_order_used": ["D2", "D1", "C2", "B1", "C1"], # Differs from ranking ("C2", "B1", "D1", "D2", "C1")!
        },
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["provider_order_used"] == "FAIL"


# 17. READY Provider Replay Missing GGR or Parse Error -> FAIL (Issue #5)
def test_validator_ready_provider_replay_missing_ggr_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(tmp_path, include_default_ggr=False)
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"
    assert any("missing" in f for f in details["failures"])


def test_validator_ready_provider_replay_ggr_parse_error_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(tmp_path)
    (replay_dir / "graph_grounding_result.json").write_text("invalid_json{{{", encoding="utf-8")
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"
    assert any("failed to read grounding artifact" in f for f in details["failures"])


# 18. INFEASIBLE Provider Replay Determinism -> PASS & Status Mismatch -> FAIL
def test_validator_infeasible_provider_replay_determinism_pass(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_manifest_extra={"terminal_status": "INFEASIBLE"},
        replay_manifest_extra={"terminal_status": "INFEASIBLE"},
        live_result_extra={"status": "INFEASIBLE", "plan": []},
        replay_result_extra={"status": "INFEASIBLE", "plan": []},
        live_ggr_extra={"status": "INFEASIBLE", "complete": False, "missing_roles": ["sponge"]},
        replay_ggr_extra={"status": "INFEASIBLE", "complete": False, "missing_roles": ["sponge"]},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is True
    assert details["checks"]["deterministic_provider_replay"] == "PASS"


def test_validator_provider_replay_status_mismatch_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_manifest_extra={"terminal_status": "ACTION_SEQUENCE_READY"},
        replay_manifest_extra={"terminal_status": "INFEASIBLE"},
        live_result_extra={"status": "ACTION_SEQUENCE_READY"},
        replay_result_extra={"status": "INFEASIBLE"},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"


# 19. Random Seed Tests (Section 19)
def test_validator_random_seed_none_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": None,
            "search_seed_effective": None,
        },
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="random")
    assert ok is False
    assert details["checks"]["provenance"] == "FAIL"


def test_validator_random_seed_req_eff_mismatch_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 1,
        },
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="random")
    assert ok is False
    assert details["checks"]["provenance"] == "FAIL"


def test_validator_random_seed_negative_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": -1,
            "search_seed_effective": -1,
        },
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="random")
    assert ok is False
    assert details["checks"]["provenance"] == "FAIL"


def test_validator_random_seed_bool_rejected_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": True,
            "search_seed_effective": True,
        },
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="random")
    assert ok is False
    assert details["checks"]["provenance"] == "FAIL"


def test_validator_random_order_not_matching_seed_permutation_fail(tmp_path: Path):
    g = make_test_graph()
    # Compute permutation for seed 0
    base0 = list(g.candidate_regions)
    rng0 = random.Random(0)
    rng0.shuffle(base0)

    # Compute permutation for seed 1
    base1 = list(g.candidate_regions)
    rng1 = random.Random(1)
    rng1.shuffle(base1)

    # Replay manifest records seed 0, but uses permutation from seed 1
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g,
        replay_graph=g,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 0,
            "region_order_used": base1,
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=0,
    )
    assert ok is False
    assert details["checks"]["random_seed_order"] == "FAIL"


# Pass 3.4.2 New Tests
# 20. Issue #1: Random Replay Live Baseline Check
def test_validator_random_replay_live_baseline_not_provider_fail(tmp_path: Path):
    g = make_test_graph()
    base0 = list(g.candidate_regions)
    rng0 = random.Random(0)
    rng0.shuffle(base0)

    # Live run was random instead of provider!
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g,
        replay_graph=g,
        live_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "region_order_used": base0,
        },
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 0,
            "region_order_used": base0,
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=0,
    )
    assert ok is False
    assert details["checks"]["live_provider_baseline"] == "FAIL"


def test_validator_random_replay_live_baseline_provider_pass(tmp_path: Path):
    g = make_test_graph()
    base0 = list(g.candidate_regions)
    rng0 = random.Random(0)
    rng0.shuffle(base0)

    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g,
        replay_graph=g,
        replay_manifest_extra={
            "search_order_source_requested": "random",
            "search_order_source_effective": "random",
            "search_seed_requested": 0,
            "search_seed_effective": 0,
            "region_order_used": base0,
        },
    )
    ok, details = validate_vlm_replay(
        live_dir,
        replay_dir,
        expect_replay_search="random",
        expect_seed=0,
    )
    assert ok is True
    assert details["checks"]["live_provider_baseline"] == "PASS"


# 21. Issue #2: Gating deterministic_provider_replay by upstream failures
def test_validator_provider_replay_gated_by_upstream_order_failure(tmp_path: Path):
    g = make_test_graph()
    # Replay used order differs, but everything downstream matches
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_graph=g,
        replay_graph=g,
        replay_manifest_extra={
            "region_order_used": ["D2", "D1", "C2", "B1", "C1"], # Mismatch!
        },
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["provider_order_used"] == "FAIL"
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"


# 22. Issue #3: READY GGR assignment and completeness checks
def test_validator_ready_ggr_assignment_missing_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_ggr_extra={"status": "COMPLETE", "complete": True, "assignment": None},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"


def test_validator_ready_ggr_assignment_empty_dict_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_ggr_extra={"status": "COMPLETE", "complete": True, "assignment": {}},
        replay_ggr_extra={"status": "COMPLETE", "complete": True, "assignment": {}},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"


def test_validator_ready_ggr_complete_false_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_ggr_extra={"status": "COMPLETE", "complete": False, "assignment": {"sponge": "sponge_1"}},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"


def test_validator_ready_ggr_status_not_complete_fail(tmp_path: Path):
    live_dir, replay_dir = setup_pair_dirs(
        tmp_path,
        live_ggr_extra={"status": "INCOMPLETE", "complete": True, "assignment": {"sponge": "sponge_1"}},
    )
    ok, details = validate_vlm_replay(live_dir, replay_dir, expect_replay_search="provider")
    assert ok is False
    assert details["checks"]["deterministic_provider_replay"] == "FAIL"
