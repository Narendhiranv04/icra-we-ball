#!/usr/bin/env python3
"""
Offline artifact validator for Phase 3.4 / 3.4.1 / 3.4.2 VLM and exact-G_F replay integration.
Verifies cryptographic SHA identity, structural G_F validity, domain/variant identity,
VLM source provenance, provider ranking fidelity, provider used order, random seed validity,
exact random permutation reproduction, live provider baseline, and gated deterministic provider replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(file_path: Path | str) -> dict[str, Any]:
    """Load JSON file or raise FileNotFoundError / ValueError."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_vlm_replay(
    live_run_dir: Path | str,
    replay_run_dir: Path | str,
    *,
    expect_replay_search: str = "provider",
    expect_seed: int | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a live VLM provider run directory against a replay run directory.
    Returns (success_bool, details_dict).
    """
    live_dir = Path(live_run_dir)
    replay_dir = Path(replay_run_dir)

    failures: List[str] = []
    checks: Dict[str, str] = {}

    # 1. Existence of required files
    for name, rdir in [("live", live_dir), ("replay", replay_dir)]:
        for artifact in ["run_manifest.json", "functional_specification.json", "result.json"]:
            art_path = rdir / artifact
            if not art_path.is_file():
                failures.append(f"{name} artifact missing: {art_path}")

    if failures:
        return False, {
            "success": False,
            "failures": failures,
            "checks": checks,
        }

    live_manifest = load_json(live_dir / "run_manifest.json")
    live_spec_dict = load_json(live_dir / "functional_specification.json")
    live_result = load_json(live_dir / "result.json")

    replay_manifest = load_json(replay_dir / "run_manifest.json")
    replay_spec_dict = load_json(replay_dir / "functional_specification.json")
    replay_result = load_json(replay_dir / "result.json")

    live_domain = live_manifest.get("domain")
    replay_domain = replay_manifest.get("domain")
    live_variant = live_manifest.get("variant")
    replay_variant = replay_manifest.get("variant")

    # Domain and Variant Identity
    id_ok = True
    if not isinstance(live_domain, str) or not live_domain.strip():
        failures.append(f"live manifest missing or empty domain: {live_domain!r}")
        id_ok = False
    if not isinstance(replay_domain, str) or not replay_domain.strip():
        failures.append(f"replay manifest missing or empty domain: {replay_domain!r}")
        id_ok = False
    if live_domain != replay_domain or not live_domain:
        failures.append(f"domain mismatch: live='{live_domain}', replay='{replay_domain}'")
        id_ok = False

    if not isinstance(live_variant, str) or not live_variant.strip():
        failures.append(f"live manifest missing or empty variant: {live_variant!r}")
        id_ok = False
    if not isinstance(replay_variant, str) or not replay_variant.strip():
        failures.append(f"replay manifest missing or empty variant: {replay_variant!r}")
        id_ok = False
    if live_variant != replay_variant or not live_variant:
        failures.append(f"variant mismatch: live='{live_variant}', replay='{replay_variant}'")
        id_ok = False

    domain = live_domain if (live_domain and live_domain == replay_domain) else "unknown"
    variant = live_variant if (live_variant and live_variant == replay_variant) else "unknown"
    checks["identity"] = "PASS" if id_ok else "FAIL"

    # 2. SHA-256 Cryptographic Identity
    live_file_sha = compute_file_sha256(live_dir / "functional_specification.json")
    replay_file_sha = compute_file_sha256(replay_dir / "functional_specification.json")
    live_man_sha = live_manifest.get("specification_sha256")
    replay_man_sha = replay_manifest.get("specification_sha256")

    sha_checks_ok = (
        live_file_sha == replay_file_sha == live_man_sha == replay_man_sha
        and live_file_sha is not None
    )
    if sha_checks_ok:
        checks["exact_gf_sha"] = "PASS"
    else:
        checks["exact_gf_sha"] = "FAIL"
        failures.append(
            f"G_F SHA mismatch: live_file={live_file_sha}, replay_file={replay_file_sha}, "
            f"live_manifest={live_man_sha}, replay_manifest={replay_man_sha}"
        )

    # 3. Structural G_F Validation & Graph Domain Consistency
    live_graph: Optional[FunctionalRequirementGraph] = None
    replay_graph: Optional[FunctionalRequirementGraph] = None
    try:
        live_graph = FunctionalRequirementGraph.from_dict(live_spec_dict)
        live_graph.validate()
        replay_graph = FunctionalRequirementGraph.from_dict(replay_spec_dict)
        replay_graph.validate()

        struct_ok = True
        if live_graph.to_dict() != replay_graph.to_dict():
            failures.append("live_graph.to_dict() != replay_graph.to_dict()")
            struct_ok = False

        if id_ok:
            if live_graph.domain != live_domain:
                failures.append(f"live graph domain '{live_graph.domain}' != live manifest domain '{live_domain}'")
                struct_ok = False
            if replay_graph.domain != replay_domain:
                failures.append(f"replay graph domain '{replay_graph.domain}' != replay manifest domain '{replay_domain}'")
                struct_ok = False

        checks["structural_gf_identity"] = "PASS" if struct_ok else "FAIL"
    except Exception as e:
        checks["structural_gf_identity"] = "FAIL"
        failures.append(f"G_F graph validation exception: {e}")
        live_graph = None
        replay_graph = None

    # 4. VLM Source Validation
    vlm_source_ok = True
    live_man_src = live_manifest.get("spec_provider_source")
    replay_man_src = replay_manifest.get("spec_provider_source")

    if not isinstance(live_man_src, str) or not live_man_src.startswith("VLM"):
        failures.append(f"live manifest spec_provider_source '{live_man_src}' does not start with 'VLM'")
        vlm_source_ok = False
    if not isinstance(replay_man_src, str) or not replay_man_src.startswith("VLM"):
        failures.append(f"replay manifest spec_provider_source '{replay_man_src}' does not start with 'VLM'")
        vlm_source_ok = False

    if live_graph is not None:
        if not isinstance(live_graph.source, str) or not live_graph.source.startswith("VLM"):
            failures.append(f"live graph source '{live_graph.source}' does not start with 'VLM'")
            vlm_source_ok = False
        if live_man_src != live_graph.source:
            failures.append(f"live manifest source '{live_man_src}' != live graph source '{live_graph.source}'")
            vlm_source_ok = False

    if replay_graph is not None:
        if not isinstance(replay_graph.source, str) or not replay_graph.source.startswith("VLM"):
            failures.append(f"replay graph source '{replay_graph.source}' does not start with 'VLM'")
            vlm_source_ok = False
        if replay_man_src != replay_graph.source:
            failures.append(f"replay manifest source '{replay_man_src}' != replay graph source '{replay_graph.source}'")
            vlm_source_ok = False

    checks["vlm_source"] = "PASS" if vlm_source_ok else "FAIL"

    # 5. Manifest Provenance Validation
    prov_ok = True
    if live_manifest.get("spec_mode") != "vlm":
        failures.append(f"live spec_mode={live_manifest.get('spec_mode')} (expected 'vlm')")
        prov_ok = False
    if live_manifest.get("spec_acquisition") != "live_provider":
        failures.append(f"live spec_acquisition={live_manifest.get('spec_acquisition')} (expected 'live_provider')")
        prov_ok = False

    if replay_manifest.get("spec_mode") != "vlm":
        failures.append(f"replay spec_mode={replay_manifest.get('spec_mode')} (expected 'vlm')")
        prov_ok = False
    if replay_manifest.get("spec_acquisition") != "replayed_provider_output":
        failures.append(f"replay spec_acquisition={replay_manifest.get('spec_acquisition')} (expected 'replayed_provider_output')")
        prov_ok = False

    # Effective search order checks
    if domain == "living_room":
        if live_manifest.get("search_order_source_effective") != "not_applicable":
            failures.append(f"living room live search={live_manifest.get('search_order_source_effective')} (expected 'not_applicable')")
            prov_ok = False
        if replay_manifest.get("search_order_source_effective") != "not_applicable":
            failures.append(f"living room replay search={replay_manifest.get('search_order_source_effective')} (expected 'not_applicable')")
            prov_ok = False
    elif expect_replay_search == "provider":
        if live_manifest.get("search_order_source_effective") != "provider":
            failures.append(f"live search={live_manifest.get('search_order_source_effective')} (expected 'provider')")
            prov_ok = False
        if replay_manifest.get("search_order_source_effective") != "provider":
            failures.append(f"replay search={replay_manifest.get('search_order_source_effective')} (expected 'provider')")
            prov_ok = False
    elif expect_replay_search == "random":
        if replay_manifest.get("search_order_source_effective") != "random":
            failures.append(f"replay search={replay_manifest.get('search_order_source_effective')} (expected 'random')")
            prov_ok = False

        seed_req = replay_manifest.get("search_seed_requested")
        seed_eff = replay_manifest.get("search_seed_effective")
        seed_valid = (
            isinstance(seed_req, int)
            and not isinstance(seed_req, bool)
            and isinstance(seed_eff, int)
            and not isinstance(seed_eff, bool)
            and seed_req >= 0
            and seed_eff >= 0
            and seed_req == seed_eff
        )
        if not seed_valid:
            failures.append(
                f"invalid random seed in replay manifest: requested={seed_req} ({type(seed_req).__name__}), "
                f"effective={seed_eff} ({type(seed_eff).__name__})"
            )
            prov_ok = False

        if expect_seed is not None:
            if seed_req != expect_seed or seed_eff != expect_seed:
                failures.append(
                    f"replay seed requested={seed_req}, effective={seed_eff} (expected {expect_seed})"
                )
                prov_ok = False

    checks["provenance"] = "PASS" if prov_ok else "FAIL"

    # 6. Provider Ranking Fidelity
    ranking_ok = True
    if live_graph is not None:
        if list(live_manifest.get("provider_region_ranking", [])) != list(live_graph.region_ranking):
            failures.append("live provider_region_ranking != live_graph.region_ranking")
            ranking_ok = False
    if replay_graph is not None:
        if list(replay_manifest.get("provider_region_ranking", [])) != list(replay_graph.region_ranking):
            failures.append("replay provider_region_ranking != replay_graph.region_ranking")
            ranking_ok = False

    checks["provider_ranking_preserved"] = "PASS" if ranking_ok else "FAIL"

    # 7. Provider Used Order (for provider replay)
    if expect_replay_search == "provider" and domain != "living_room":
        prov_order_ok = True
        if live_graph is not None:
            if list(live_manifest.get("region_order_used", [])) != list(live_graph.region_ranking):
                failures.append("live region_order_used != live_graph.region_ranking")
                prov_order_ok = False
        if replay_graph is not None:
            if list(replay_manifest.get("region_order_used", [])) != list(replay_graph.region_ranking):
                failures.append("replay region_order_used != replay_graph.region_ranking")
                prov_order_ok = False
        checks["provider_order_used"] = "PASS" if prov_order_ok else "FAIL"
    else:
        checks["provider_order_used"] = "N/A"

    # 8. Live Provider Baseline (for random replay)
    if expect_replay_search == "random" and domain != "living_room":
        base_prov_ok = True
        if live_manifest.get("search_order_source_effective") != "provider":
            failures.append(f"live baseline search_order_source_effective='{live_manifest.get('search_order_source_effective')}' (expected 'provider')")
            base_prov_ok = False
        if live_graph is not None:
            if list(live_manifest.get("provider_region_ranking", [])) != list(live_graph.region_ranking):
                failures.append("live baseline provider_region_ranking != live_graph.region_ranking")
                base_prov_ok = False
            if list(live_manifest.get("region_order_used", [])) != list(live_graph.region_ranking):
                failures.append("live baseline region_order_used != live_graph.region_ranking")
                base_prov_ok = False
        checks["live_provider_baseline"] = "PASS" if base_prov_ok else "FAIL"
    else:
        checks["live_provider_baseline"] = "N/A"

    # 9. Candidate Permutation & Seed-Order Check (for random replay)
    if expect_replay_search == "random" and replay_graph is not None and domain != "living_room":
        used_order = list(replay_manifest.get("region_order_used", []))
        candidate_regions = list(replay_graph.candidate_regions)
        if set(used_order) == set(candidate_regions) and len(used_order) == len(set(used_order)) == len(candidate_regions):
            checks["candidate_permutation"] = "PASS"
        else:
            checks["candidate_permutation"] = "FAIL"
            failures.append(f"random region_order_used {used_order} is not a valid permutation of candidates {candidate_regions}")

        # Exact random permutation verification from recorded seed
        seed_eff = replay_manifest.get("search_seed_effective")
        if isinstance(seed_eff, int) and not isinstance(seed_eff, bool) and seed_eff >= 0:
            base = list(replay_graph.candidate_regions)
            rng = random.Random(seed_eff)
            rng.shuffle(base)
            if used_order == base:
                checks["random_seed_order"] = "PASS"
            else:
                checks["random_seed_order"] = "FAIL"
                failures.append(f"random region_order_used {used_order} does not match expected permutation {base} for seed {seed_eff}")
        else:
            checks["random_seed_order"] = "FAIL"
    else:
        checks["candidate_permutation"] = "N/A"
        checks["random_seed_order"] = "N/A"

    # 10. Terminal / Result Consistency
    term_ok = True
    if live_manifest.get("terminal_status") != live_result.get("status"):
        failures.append(f"live manifest terminal_status={live_manifest.get('terminal_status')} != result.json status={live_result.get('status')}")
        term_ok = False
    if replay_manifest.get("terminal_status") != replay_result.get("status"):
        failures.append(f"replay manifest terminal_status={replay_manifest.get('terminal_status')} != result.json status={replay_result.get('status')}")
        term_ok = False

    checks["terminal_result_consistency"] = "PASS" if term_ok else "FAIL"

    # 11. Deterministic Provider Replay Check (Strictly Gated by Prerequisite Checks)
    if expect_replay_search == "provider":
        det_ok = True

        # Check prerequisite checks
        prereq_checks = [
            "identity",
            "exact_gf_sha",
            "structural_gf_identity",
            "vlm_source",
            "provenance",
            "provider_ranking_preserved",
            "provider_order_used",
            "terminal_result_consistency",
        ]
        prereq_ok = all(checks.get(c) in ("PASS", "N/A") for c in prereq_checks)
        if not prereq_ok:
            det_ok = False
            failures.append("deterministic provider replay prerequisites failed")

        if live_result.get("status") != replay_result.get("status"):
            failures.append(f"status differs between live and provider replay: {live_result.get('status')} != {replay_result.get('status')}")
            det_ok = False
        if live_result.get("inspected_regions") != replay_result.get("inspected_regions"):
            failures.append(f"inspected_regions differs between live and provider replay: {live_result.get('inspected_regions')} != {replay_result.get('inspected_regions')}")
            det_ok = False
        if live_result.get("plan") != replay_result.get("plan"):
            failures.append("final plan differs between live and provider replay")
            det_ok = False

        live_ggr_file = live_dir / "graph_grounding_result.json"
        replay_ggr_file = replay_dir / "graph_grounding_result.json"

        # If terminal status is ACTION_SEQUENCE_READY, require GGR artifacts and valid complete matching assignment
        if live_result.get("status") == "ACTION_SEQUENCE_READY":
            if not live_ggr_file.is_file():
                failures.append(f"live run ACTION_SEQUENCE_READY missing {live_ggr_file}")
                det_ok = False
            if not replay_ggr_file.is_file():
                failures.append(f"replay run ACTION_SEQUENCE_READY missing {replay_ggr_file}")
                det_ok = False

            if live_ggr_file.is_file() and replay_ggr_file.is_file():
                try:
                    live_ggr = load_json(live_ggr_file)
                    replay_ggr = load_json(replay_ggr_file)

                    # Require complete is True
                    if live_ggr.get("complete") is not True:
                        failures.append(f"live GGR complete={live_ggr.get('complete')!r} (expected True)")
                        det_ok = False
                    if replay_ggr.get("complete") is not True:
                        failures.append(f"replay GGR complete={replay_ggr.get('complete')!r} (expected True)")
                        det_ok = False

                    # Require success status is COMPLETE
                    if live_ggr.get("status") != "COMPLETE":
                        failures.append(f"live GGR status={live_ggr.get('status')!r} (expected 'COMPLETE')")
                        det_ok = False
                    if replay_ggr.get("status") != "COMPLETE":
                        failures.append(f"replay GGR status={replay_ggr.get('status')!r} (expected 'COMPLETE')")
                        det_ok = False

                    # Require non-empty assignment dict
                    live_assign = live_ggr.get("assignment")
                    replay_assign = replay_ggr.get("assignment")
                    if not isinstance(live_assign, dict) or not live_assign:
                        failures.append(f"live GGR assignment is invalid or empty: {live_assign!r}")
                        det_ok = False
                    if not isinstance(replay_assign, dict) or not replay_assign:
                        failures.append(f"replay GGR assignment is invalid or empty: {replay_assign!r}")
                        det_ok = False

                    if isinstance(live_assign, dict) and isinstance(replay_assign, dict) and live_assign and replay_assign:
                        if live_assign != replay_assign:
                            failures.append(f"grounding assignment differs between live ({live_assign}) and provider replay ({replay_assign})")
                            det_ok = False
                except Exception as e:
                    failures.append(f"failed to read grounding artifact: {e}")
                    det_ok = False
        elif live_ggr_file.is_file() and replay_ggr_file.is_file():
            try:
                live_ggr = load_json(live_ggr_file)
                replay_ggr = load_json(replay_ggr_file)
                for field in ["status", "complete", "missing_roles"]:
                    if live_ggr.get(field) != replay_ggr.get(field):
                        failures.append(f"grounding {field} differs between live and provider replay: {live_ggr.get(field)} != {replay_ggr.get(field)}")
                        det_ok = False
            except Exception as e:
                failures.append(f"failed to read grounding artifact: {e}")
                det_ok = False

        checks["deterministic_provider_replay"] = "PASS" if det_ok else "FAIL"
    else:
        checks["deterministic_provider_replay"] = "N/A"

    overall_success = (len(failures) == 0)

    details = {
        "domain": domain,
        "variant": variant,
        "success": overall_success,
        "live": {
            "dir": str(live_dir),
            "spec_acquisition": live_manifest.get("spec_acquisition"),
            "specification_sha256": live_manifest.get("specification_sha256"),
            "search_effective": live_manifest.get("search_order_source_effective"),
            "provider_ranking": live_manifest.get("provider_region_ranking", []),
            "region_order_used": live_manifest.get("region_order_used", []),
            "terminal_status": live_manifest.get("terminal_status"),
            "n_open": len(live_result.get("inspected_regions", [])),
            "plan_length": len(live_result.get("plan", [])),
        },
        "replay": {
            "dir": str(replay_dir),
            "spec_acquisition": replay_manifest.get("spec_acquisition"),
            "specification_sha256": replay_manifest.get("specification_sha256"),
            "search_effective": replay_manifest.get("search_order_source_effective"),
            "search_seed_requested": replay_manifest.get("search_seed_requested"),
            "search_seed_effective": replay_manifest.get("search_seed_effective"),
            "provider_ranking": replay_manifest.get("provider_region_ranking", []),
            "region_order_used": replay_manifest.get("region_order_used", []),
            "terminal_status": replay_manifest.get("terminal_status"),
            "n_open": len(replay_result.get("inspected_regions", [])),
            "plan_length": len(replay_result.get("plan", [])),
        },
        "checks": checks,
        "failures": failures,
    }

    return overall_success, details


def print_validation_report(details: Dict[str, Any]) -> None:
    """Print human-readable compact validation summary."""
    live = details.get("live", {})
    replay = details.get("replay", {})
    checks = details.get("checks", {})
    failures = details.get("failures", [])

    print("=" * 60)
    print("PHASE 3.4 VLM REPLAY VALIDATION")
    print("=" * 60)
    print(f"Domain:  {details.get('domain')}")
    print(f"Variant: {details.get('variant')}")
    print("\nLive:")
    print(f"  spec acquisition:  {live.get('spec_acquisition')}")
    print(f"  spec SHA:          {live.get('specification_sha256')}")
    print(f"  effective search:  {live.get('search_effective')}")
    print(f"  provider ranking:  {live.get('provider_ranking')}")
    print(f"  used order:        {live.get('region_order_used')}")
    print(f"  terminal status:   {live.get('terminal_status')}")
    print(f"  N_open:            {live.get('n_open')}")
    print(f"  plan length:       {live.get('plan_length')}")

    print("\nReplay:")
    print(f"  spec acquisition:  {replay.get('spec_acquisition')}")
    print(f"  spec SHA:          {replay.get('specification_sha256')}")
    print(f"  effective search:  {replay.get('search_effective')}")
    print(f"  seed:              {replay.get('search_seed_requested')}")
    print(f"  provider ranking:  {replay.get('provider_ranking')}")
    print(f"  used order:        {replay.get('region_order_used')}")
    print(f"  terminal status:   {replay.get('terminal_status')}")
    print(f"  N_open:            {replay.get('n_open')}")
    print(f"  plan length:       {replay.get('plan_length')}")

    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {k:<30}: {v}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")

    print("\nFINAL:")
    print("  PASS" if details.get("success") else "  FAIL")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline artifact validator for Phase 3.4 VLM and exact-G_F replay integration."
    )
    parser.add_argument("--live-run", type=str, required=True, help="Path to live VLM run directory")
    parser.add_argument("--replay-run", type=str, required=True, help="Path to replayed G_F run directory")
    parser.add_argument(
        "--expect-replay-search",
        type=str,
        choices=["provider", "random", "not_applicable"],
        default="provider",
        help="Expected effective search policy in replay run (default: provider)",
    )
    parser.add_argument(
        "--expect-seed",
        type=int,
        default=None,
        help="Expected deterministic random seed in replay run (if random)",
    )
    parser.add_argument(
        "--output-summary-json",
        type=str,
        default=None,
        help="Optional path to save JSON validation summary",
    )
    args = parser.parse_args()

    success, details = validate_vlm_replay(
        args.live_run,
        args.replay_run,
        expect_replay_search=args.expect_replay_search,
        expect_seed=args.expect_seed,
    )

    print_validation_report(details)

    if args.output_summary_json:
        with open(args.output_summary_json, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
