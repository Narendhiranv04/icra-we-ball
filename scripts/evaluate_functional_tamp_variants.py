#!/usr/bin/env python3
"""
Evaluate functional TAMP variants across benchmark domains (Kitchen, Living Room, Workshop).
Provides clean execution, return code checking, artifact inspection, plan validation,
grounding audits, structured JSON/CSV logging, and summary reporting.

Extended in Pass 3.3 / 3.3.1 for controlled search experiments:
- GT Oracle, Provider (FM-guided when VLM), Seeded Random
- Multi-seed random trial output isolation
- Exact G_F replay and pre-run/post-run SHA-256 pairing verification
- Authoritative manifest provenance consistency checking
- Summary aggregation for multi-seed random trials with dynamic P(complete by k)
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

DOMAINS: Dict[str, List[str]] = {
    "kitchen": [f"K{i}" for i in range(1, 13)],
    "living_room": [f"L{i}" for i in range(1, 11)],
    "workshop": [f"W{i}" for i in range(1, 11)],
}

EXPECTED: Dict[str, Dict[str, str]] = {
    "kitchen": {v: "ACTION_SEQUENCE_READY" if int(v[1:]) <= 6 else "INFEASIBLE" for v in DOMAINS["kitchen"]},
    "living_room": {v: "ACTION_SEQUENCE_READY" if int(v[1:]) <= 6 else "INFEASIBLE" for v in DOMAINS["living_room"]},
    "workshop": {v: "ACTION_SEQUENCE_READY" if int(v[1:]) <= 8 else "INFEASIBLE" for v in DOMAINS["workshop"]},
}


def load_grounding_info(run_dir: str) -> Tuple[str, Optional[bool]]:
    """
    Extract grounding status and completeness from graph_grounding_result.json or satisfaction.json.
    Returns (grounding_status, grounding_complete).
    """
    ggr_file = os.path.join(run_dir, "graph_grounding_result.json")
    if os.path.exists(ggr_file):
        try:
            with open(ggr_file, "r") as f:
                ggr = json.load(f)
                status = ggr.get("status", "UNKNOWN")
                complete = ggr.get("complete")
                if complete is None:
                    complete = (status == "COMPLETE")
                return status, bool(complete)
        except Exception:
            pass

    sat_file = os.path.join(run_dir, "satisfaction.json")
    if os.path.exists(sat_file):
        try:
            with open(sat_file, "r") as f:
                sat = json.load(f)
                status = sat.get("status", "UNKNOWN")
                satisfied = sat.get("satisfied")
                if satisfied is None:
                    satisfied = (status == "COMPLETE")
                return status, bool(satisfied)
        except Exception:
            pass

    return "UNKNOWN", None


def load_plan_validation(run_dir: str) -> Optional[bool]:
    """
    Search for action plan or replay validation artifacts and extract independent symbolic replay validation status.
    Supports both nested validation (Style A: data['validation']['status']) and direct replay artifacts (Style B: data['status']).
    Returns True if validator confirmed replay status == 'VALID', False if 'INVALID', None if absent/NA.
    """
    candidates = [
        os.path.join(run_dir, "action_sequence", "action_plan.json"),
        os.path.join(run_dir, "action_plan.json"),
        os.path.join(run_dir, "action_sequence", "replay_validation.json"),
        os.path.join(run_dir, "replay_validation.json"),
        os.path.join(run_dir, "action_sequence", "plan.json"),
        os.path.join(run_dir, "plan.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r") as f:
                    data = json.load(f)
                    # Style A: nested planner validation object
                    if isinstance(data.get("validation"), dict):
                        val_status = data["validation"].get("status")
                        if val_status in ("VALID", "INVALID"):
                            return val_status == "VALID"
                    # Style B: direct replay validation artifact
                    val_status = data.get("status")
                    if val_status in ("VALID", "INVALID"):
                        return val_status == "VALID"
            except Exception:
                pass
    return None


def load_grounding_audit(run_dir: str) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Load plan_grounding_audit.json if present.
    Returns (grounding_audit_valid, accessibility_valid).
    """
    audit_file = os.path.join(run_dir, "plan_grounding_audit.json")
    if not os.path.exists(audit_file):
        return None, None
    try:
        with open(audit_file, "r") as f:
            audit = json.load(f)
            complete = bool(audit.get("grounding_complete", False))
            nodes_obs = bool(audit.get("all_assignment_nodes_observed", False))
            rels_true = bool(audit.get("all_required_relations_true", False))
            plan_obj_grounded = bool(audit.get("plan_uses_only_grounded_task_objects", False))
            prep_access = bool(audit.get("preparation_accessibility_valid", False))
            violations = audit.get("violations", [])

            audit_valid = (
                complete
                and nodes_obs
                and rels_true
                and plan_obj_grounded
                and prep_access
                and len(violations) == 0
            )
            return audit_valid, prep_access
    except Exception:
        return False, False


def load_run_manifest(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load run_manifest.json from run directory if present."""
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_random_seeds(seeds_arg: Optional[str]) -> Optional[List[int]]:
    """
    Parse comma-separated seed list string into a list of non-negative integers.
    Strictly rejects empty tokens (e.g. '0,,2', '0,', ',0', '0, ,2'), negatives, duplicates, and non-integers.
    """
    if seeds_arg is None:
        return None
    raw_tokens = seeds_arg.split(",")
    if not raw_tokens:
        raise ValueError("Empty --random-seeds argument.")
    seeds: List[int] = []
    seen = set()
    for token in raw_tokens:
        stripped = token.strip()
        if not stripped:
            raise ValueError(f"Empty token in --random-seeds: '{seeds_arg}'")
        try:
            val = int(stripped)
        except ValueError:
            raise ValueError(f"Invalid integer in --random-seeds: '{stripped}'")
        if val < 0:
            raise ValueError(f"Negative seed not permitted in --random-seeds: {val}")
        if val in seen:
            raise ValueError(f"Duplicate seed found in --random-seeds: {val}")
        seen.add(val)
        seeds.append(val)
    return seeds


def build_runner_command(
    *,
    domain: str,
    variant: str,
    mode: str = "gt",
    runner_output_root: str,
    search_order: Optional[str] = None,
    search_seed: Optional[int] = None,
    specification_json: Optional[str] = None,
) -> List[str]:
    """
    Construct command line arguments for the canonical pipeline subprocess.
    Omits optional flags when default to preserve exact legacy command stability.
    """
    cmd = [
        sys.executable,
        "-m", "mujoco_scenes.functional_tamp_pipeline.run",
        "--domain", domain,
        "--variant", variant,
        "--mode", mode,
        "--dry-run",
        "--output-root", runner_output_root,
    ]
    if search_order is not None and search_order != "auto":
        cmd.extend(["--search-order", search_order])
    if search_seed is not None:
        cmd.extend(["--search-seed", str(search_seed)])
    if specification_json is not None:
        cmd.extend(["--specification-json", specification_json])
    return cmd


def evaluate_variant(
    domain: str,
    variant: str,
    output_root: str,
    *,
    mode: str = "gt",
    search_order: Optional[str] = None,
    search_seed: Optional[int] = None,
    specification_json: Optional[str] = None,
    source_specification_sha256: Optional[str] = None,
    trial_output_root: Optional[str] = None,
    trial_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Launch canonical runner for one domain variant trial, record runtime and return code,
    and parse resulting execution, grounding, and manifest artifacts.
    """
    expected = EXPECTED[domain][variant]
    actual_runner_output_root = trial_output_root if trial_output_root is not None else output_root

    # Compute source specification hash before subprocess launch if not precomputed
    source_spec_sha256_before = source_specification_sha256
    source_hash_error_before = False
    if specification_json is not None and source_spec_sha256_before is None:
        try:
            source_spec_sha256_before = compute_file_sha256(specification_json)
        except Exception:
            source_hash_error_before = True

    cmd = build_runner_command(
        domain=domain,
        variant=variant,
        mode=mode,
        runner_output_root=actual_runner_output_root,
        search_order=search_order,
        search_seed=search_seed,
        specification_json=specification_json,
    )

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."},
    )
    runtime = time.time() - t0

    run_dir = os.path.join(actual_runner_output_root, domain, variant, mode)
    res_file = os.path.join(run_dir, "result.json")

    actual = "ERROR_NO_RESULT"
    failure_reason: Optional[str] = None
    inspected_regions: List[str] = []
    plan_len = 0
    is_completed = False

    if proc.returncode != 0:
        actual = "CRASH"
        failure_reason = f"return_code={proc.returncode}"
    elif not os.path.exists(res_file):
        actual = "ERROR_NO_RESULT"
        failure_reason = "result.json missing"
    else:
        try:
            with open(res_file, "r") as f:
                data = json.load(f)
                actual = data.get("status", "UNKNOWN")
                failure_reason = data.get("failure_reason")
                inspected_regions = data.get("inspected_regions", [])
                plan_len = len(data.get("plan", []))
                is_completed = True
        except Exception as e:
            actual = "ERROR_INVALID_RESULT"
            failure_reason = str(e)

    # Manifest provenance extraction
    manifest = load_run_manifest(run_dir)
    manifest_path = os.path.join(run_dir, "run_manifest.json") if manifest is not None else None

    evaluator_mode_requested = mode
    evaluator_search_order_requested = search_order or "auto"
    evaluator_search_seed_requested = search_seed

    manifest_spec_mode = manifest.get("spec_mode") if manifest else None
    manifest_search_order_requested = manifest.get("search_order_source_requested") if manifest else None
    manifest_search_order_effective = manifest.get("search_order_source_effective") if manifest else None
    manifest_search_seed_requested = manifest.get("search_seed_requested") if manifest else None
    manifest_search_seed_effective = manifest.get("search_seed_effective") if manifest else None
    manifest_terminal_status = manifest.get("terminal_status") if manifest else None

    provider_region_ranking: List[str] = list(manifest.get("provider_region_ranking", [])) if manifest else []
    region_order_used: List[str] = list(manifest.get("region_order_used", [])) if manifest else []
    exploration_actuation = manifest.get("exploration_actuation", "unknown") if manifest else "unknown"
    spec_acquisition = manifest.get("spec_acquisition", "unknown") if manifest else "unknown"
    manifest_spec_sha256 = manifest.get("specification_sha256") if manifest else None

    # Provenance consistency checks
    provenance_mismatches: List[str] = []
    if manifest is not None:
        if manifest_spec_mode != evaluator_mode_requested:
            provenance_mismatches.append(f"spec_mode: evaluator='{evaluator_mode_requested}' != manifest='{manifest_spec_mode}'")
        if manifest_search_order_requested != evaluator_search_order_requested:
            provenance_mismatches.append(f"search_order_requested: evaluator='{evaluator_search_order_requested}' != manifest='{manifest_search_order_requested}'")
        if evaluator_search_seed_requested is not None and manifest_search_seed_requested != evaluator_search_seed_requested:
            provenance_mismatches.append(f"search_seed: evaluator={evaluator_search_seed_requested} != manifest={manifest_search_seed_requested}")
        if is_completed and actual in {"ACTION_SEQUENCE_READY", "INFEASIBLE"} and manifest_terminal_status != actual:
            provenance_mismatches.append(f"terminal_status: result.json='{actual}' != manifest='{manifest_terminal_status}'")
        provenance_match = (len(provenance_mismatches) == 0)
    else:
        provenance_match = None

    # Replay SHA verification and source immutability check
    source_spec_sha256_after = None
    source_spec_unchanged = None
    pairing_status = "NOT_APPLICABLE"
    spec_hash_match = None

    if specification_json is not None:
        if source_hash_error_before or source_spec_sha256_before is None:
            pairing_status = "SOURCE_HASH_ERROR"
            spec_hash_match = False
        else:
            try:
                source_spec_sha256_after = compute_file_sha256(specification_json)
                source_spec_unchanged = (source_spec_sha256_before == source_spec_sha256_after)
            except Exception:
                source_spec_unchanged = False

            if source_spec_unchanged is False:
                pairing_status = "SOURCE_CHANGED"
                spec_hash_match = False
            elif manifest is None:
                pairing_status = "MANIFEST_MISSING"
                spec_hash_match = False
            elif manifest_spec_sha256 is None:
                pairing_status = "MANIFEST_HASH_MISSING"
                spec_hash_match = False
            elif source_spec_sha256_before != manifest_spec_sha256:
                pairing_status = "HASH_MISMATCH"
                spec_hash_match = False
            else:
                pairing_status = "VERIFIED"
                spec_hash_match = True

    # Evaluation validity determination (scientific condition validity)
    evaluation_valid = True
    evaluation_failure_reasons: List[str] = []

    if not is_completed:
        evaluation_valid = False
        evaluation_failure_reasons.append(f"pipeline did not complete ({failure_reason or actual})")

    if provenance_match is False:
        evaluation_valid = False
        evaluation_failure_reasons.extend(provenance_mismatches)

    if specification_json is not None:
        if pairing_status != "VERIFIED":
            evaluation_valid = False
            evaluation_failure_reasons.append(f"pairing_status: {pairing_status}")

    evaluation_failure_reason = "; ".join(evaluation_failure_reasons) if evaluation_failure_reasons else None

    # Match and valid_match calculations
    match = "YES" if actual == expected else "NO"
    valid_match = ("YES" if actual == expected else "NO") if evaluation_valid else "N/A"

    inspection_count = len(inspected_regions)
    combined_count = inspection_count + plan_len

    # Grounding status and completeness
    grounding_status, grounding_complete = load_grounding_info(run_dir)

    # Independent plan replay validation
    plan_replay_valid = load_plan_validation(run_dir)

    # Plan grounding audit and accessibility validation
    grounding_audit_valid, accessibility_valid = load_grounding_audit(run_dir)

    ggr_path = os.path.join(run_dir, "graph_grounding_result.json")
    audit_path = os.path.join(run_dir, "plan_grounding_audit.json")

    # Effective search order resolution for compatibility field
    search_order_effective = manifest_search_order_effective
    if search_order_effective is None and domain == "living_room":
        search_order_effective = "not_applicable"

    # Condition label derivation
    if domain == "living_room":
        condition_label = f"{mode}_living_room"
    elif evaluator_search_order_requested == "random" or search_order_effective == "random":
        seed_str = f"{evaluator_search_seed_requested:03d}" if evaluator_search_seed_requested is not None else "none"
        condition_label = f"{mode}_random_seed_{seed_str}"
    elif (search_order_effective or evaluator_search_order_requested) == "oracle":
        condition_label = f"{mode}_oracle"
    elif (search_order_effective or evaluator_search_order_requested) == "provider":
        condition_label = f"{mode}_provider"
    else:
        condition_label = f"{mode}_{evaluator_search_order_requested}"

    return {
        "domain": domain,
        "variant": variant,
        "expected_status": expected,
        "actual_status": actual,
        "match": match,
        "valid_match": valid_match,
        "evaluation_valid": evaluation_valid,
        "evaluation_failure_reason": evaluation_failure_reason,
        "return_code": proc.returncode,
        "completed": is_completed,
        "runtime_sec": runtime,
        "inspected_regions": inspected_regions,
        "inspection_count": inspection_count,
        "n_open": inspection_count,
        "plan_length": plan_len,
        "combined_high_level_count": combined_count,
        "grounding_status": grounding_status,
        "grounding_complete": grounding_complete,
        "grounding_audit_valid": grounding_audit_valid,
        "plan_replay_valid": plan_replay_valid,
        "accessibility_valid": accessibility_valid,
        "failure_reason": failure_reason,
        "result_json_path": res_file,
        "graph_grounding_path": ggr_path if os.path.exists(ggr_path) else None,
        "audit_path": audit_path if os.path.exists(audit_path) else None,
        "spec_mode": manifest_spec_mode or mode,
        "condition_label": condition_label,
        "search_order_requested": manifest_search_order_requested or evaluator_search_order_requested,
        "search_order_effective": search_order_effective,
        "search_seed_requested": manifest_search_seed_requested if manifest_search_seed_requested is not None else evaluator_search_seed_requested,
        "search_seed_effective": manifest_search_seed_effective,
        "evaluator_mode_requested": evaluator_mode_requested,
        "evaluator_search_order_requested": evaluator_search_order_requested,
        "evaluator_search_seed_requested": evaluator_search_seed_requested,
        "manifest_spec_mode": manifest_spec_mode,
        "manifest_search_order_requested": manifest_search_order_requested,
        "manifest_search_order_effective": manifest_search_order_effective,
        "manifest_search_seed_requested": manifest_search_seed_requested,
        "manifest_search_seed_effective": manifest_search_seed_effective,
        "manifest_terminal_status": manifest_terminal_status,
        "provenance_match": provenance_match,
        "provenance_mismatches": provenance_mismatches,
        "provider_region_ranking": provider_region_ranking,
        "region_order_used": region_order_used,
        "exploration_actuation": exploration_actuation,
        "spec_acquisition": spec_acquisition,
        "specification_sha256": manifest_spec_sha256,
        "source_specification_path": specification_json,
        "source_specification_sha256": source_spec_sha256_before,
        "source_specification_sha256_after": source_spec_sha256_after,
        "source_specification_unchanged": source_spec_unchanged,
        "pairing_status": pairing_status,
        "specification_hash_match": spec_hash_match,
        "run_manifest_path": manifest_path,
        "trial_id": trial_id or f"{domain}_{variant}_{mode}",
        "trial_output_root": actual_runner_output_root,
        "run_dir": run_dir,
    }


def aggregate_random_trials(
    results: Sequence[Dict[str, Any]],
    output_root: str,
) -> Tuple[Dict[str, Any], str, str]:
    """
    Aggregate repeated random seed trials per (domain, variant).
    Computes descriptive statistics (mean, population std, median, min, max) and dynamic P(grounding complete by k).
    Produces random_aggregate.json and random_aggregate.csv at output_root with dynamic global max k.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in results:
        key = (r["domain"], r["variant"])
        grouped.setdefault(key, []).append(r)

    aggregates_by_variant = []
    global_max_k = 0

    for (dom, var), group_rows in sorted(grouped.items()):
        n_trials = len(group_rows)
        seeds = [r["search_seed_requested"] for r in group_rows if r.get("search_seed_requested") is not None]
        valid_rows = [r for r in group_rows if r.get("evaluation_valid") is True]
        n_valid = len(valid_rows)
        n_invalid = n_trials - n_valid
        expected_status = EXPECTED[dom][var]

        status_counts = {}
        for r in group_rows:
            st = r.get("actual_status", "UNKNOWN")
            status_counts[st] = status_counts.get(st, 0) + 1

        pairing_counts = {}
        for r in group_rows:
            ps = r.get("pairing_status", "NOT_APPLICABLE")
            pairing_counts[ps] = pairing_counts.get(ps, 0) + 1

        prov_failures = sum(1 for r in group_rows if r.get("provenance_match") is False)

        match_count_all = sum(1 for r in valid_rows if r.get("match") == "YES")
        match_rate_all = (match_count_all / n_trials) if n_trials > 0 else 0.0
        match_rate_valid = (match_count_all / n_valid) if n_valid > 0 else None

        gr_complete_count = sum(1 for r in valid_rows if r.get("grounding_complete") is True)
        gr_complete_rate = (gr_complete_count / n_trials) if n_trials > 0 else 0.0
        gr_complete_rate_valid = (gr_complete_count / n_valid) if n_valid > 0 else None

        # Inspection count statistics over valid completed trials
        if valid_rows:
            insp_counts = [r["inspection_count"] for r in valid_rows]
            insp_mean = statistics.mean(insp_counts)
            insp_std = statistics.pstdev(insp_counts) if len(insp_counts) > 1 else 0.0
            insp_median = statistics.median(insp_counts)
            insp_min = min(insp_counts)
            insp_max = max(insp_counts)
        else:
            insp_mean, insp_std, insp_median, insp_min, insp_max = 0.0, 0.0, 0.0, 0, 0

        # Runtime statistics over valid completed trials (or all rows if none valid)
        target_runtime_rows = valid_rows if valid_rows else group_rows
        runtimes = [r["runtime_sec"] for r in target_runtime_rows]
        rt_mean = statistics.mean(runtimes) if runtimes else 0.0
        rt_std = statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0

        # Plan length statistics over valid completed trials
        if valid_rows:
            plan_lens = [r["plan_length"] for r in valid_rows]
            pl_mean = statistics.mean(plan_lens)
            pl_std = statistics.pstdev(plan_lens) if len(plan_lens) > 1 else 0.0
        else:
            pl_mean, pl_std = 0.0, 0.0

        # Replay valid rate among applicable feasible tasks
        feasible_valid = [r for r in valid_rows if r.get("expected_status") == "ACTION_SEQUENCE_READY"]
        if feasible_valid:
            replay_valid_count = sum(1 for r in feasible_valid if r.get("plan_replay_valid") is True)
            replay_valid_rate = replay_valid_count / len(feasible_valid)
        else:
            replay_valid_rate = None

        # Specification hash all match
        has_replay = any(r.get("pairing_status") != "NOT_APPLICABLE" for r in group_rows)
        if has_replay:
            spec_hash_all_match = all(r.get("pairing_status") == "VERIFIED" for r in group_rows)
        else:
            spec_hash_all_match = None

        # Dynamic k range definition based on region_order_used length (fallback to inspected_regions)
        candidate_lens = [len(r.get("region_order_used", [])) for r in group_rows if r.get("region_order_used")]
        if not candidate_lens:
            candidate_lens = [len(r.get("inspected_regions", [])) for r in group_rows]
        variant_max_k = max(candidate_lens) if candidate_lens else 0
        global_max_k = max(global_max_k, variant_max_k)

        p_grounding_complete_by_k = {}
        for k in range(variant_max_k + 1):
            succ = sum(1 for r in valid_rows if r.get("grounding_complete") is True and r["inspection_count"] <= k)
            p_k = (succ / n_trials) if n_trials > 0 else 0.0
            p_grounding_complete_by_k[str(k)] = round(p_k, 4)

        aggregates_by_variant.append({
            "domain": dom,
            "variant": var,
            "n_trials": n_trials,
            "seeds": seeds,
            "n_valid_trials": n_valid,
            "evaluation_invalid_count": n_invalid,
            "expected_status": expected_status,
            "actual_status_counts": status_counts,
            "pairing_status_counts": pairing_counts,
            "provenance_failure_count": prov_failures,
            "expected_match_rate": round(match_rate_all, 4),
            "expected_match_rate_all_attempts": round(match_rate_all, 4),
            "expected_match_rate_valid_trials": round(match_rate_valid, 4) if match_rate_valid is not None else None,
            "grounding_complete_rate": round(gr_complete_rate, 4),
            "grounding_complete_rate_valid_trials": round(gr_complete_rate_valid, 4) if gr_complete_rate_valid is not None else None,
            "inspection_count_mean": round(insp_mean, 4),
            "inspection_count_std": round(insp_std, 4),
            "inspection_count_median": insp_median,
            "inspection_count_min": insp_min,
            "inspection_count_max": insp_max,
            "runtime_mean": round(rt_mean, 4),
            "runtime_std": round(rt_std, 4),
            "plan_length_mean": round(pl_mean, 4),
            "plan_length_std": round(pl_std, 4),
            "replay_valid_rate": round(replay_valid_rate, 4) if replay_valid_rate is not None else None,
            "specification_hash_all_match": spec_hash_all_match,
            "variant_max_k": variant_max_k,
            "p_grounding_complete_by_k": p_grounding_complete_by_k,
        })

    aggregate_summary = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "std_metric": "population_standard_deviation (statistics.pstdev)",
        "global_max_k": global_max_k,
        "aggregates": aggregates_by_variant,
    }

    json_path = os.path.join(output_root, "random_aggregate.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)

    # Dynamic CSV generation up to global_max_k
    csv_path = os.path.join(output_root, "random_aggregate.csv")
    csv_headers = [
        "Domain",
        "Variant",
        "NTrials",
        "NValidTrials",
        "EvaluationInvalidCount",
        "ExpectedStatus",
        "ExpectedMatchRate",
        "ExpectedMatchRateAllAttempts",
        "ExpectedMatchRateValidTrials",
        "GroundingCompleteRate",
        "GroundingCompleteRateValidTrials",
        "InspectionCountMean",
        "InspectionCountStd",
        "InspectionCountMedian",
        "InspectionCountMin",
        "InspectionCountMax",
        "RuntimeMean",
        "RuntimeStd",
        "PlanLengthMean",
        "PlanLengthStd",
        "ReplayValidRate",
        "PairingStatusCounts",
        "ProvenanceFailureCount",
        "SpecificationHashAllMatch",
    ]
    for k in range(global_max_k + 1):
        csv_headers.append(f"PCompleteByK{k}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for agg in aggregates_by_variant:
            p_k = agg["p_grounding_complete_by_k"]
            row = [
                agg["domain"],
                agg["variant"],
                agg["n_trials"],
                agg["n_valid_trials"],
                agg["evaluation_invalid_count"],
                agg["expected_status"],
                f"{agg['expected_match_rate']:.4f}",
                f"{agg['expected_match_rate_all_attempts']:.4f}",
                f"{agg['expected_match_rate_valid_trials']:.4f}" if agg["expected_match_rate_valid_trials"] is not None else "N/A",
                f"{agg['grounding_complete_rate']:.4f}",
                f"{agg['grounding_complete_rate_valid_trials']:.4f}" if agg["grounding_complete_rate_valid_trials"] is not None else "N/A",
                f"{agg['inspection_count_mean']:.4f}",
                f"{agg['inspection_count_std']:.4f}",
                f"{agg['inspection_count_median']}",
                agg["inspection_count_min"],
                agg["inspection_count_max"],
                f"{agg['runtime_mean']:.4f}",
                f"{agg['runtime_std']:.4f}",
                f"{agg['plan_length_mean']:.4f}",
                f"{agg['plan_length_std']:.4f}",
                f"{agg['replay_valid_rate']:.4f}" if agg["replay_valid_rate"] is not None else "N/A",
                json.dumps(agg["pairing_status_counts"]),
                agg["provenance_failure_count"],
                str(agg["specification_hash_all_match"]),
            ]
            for k in range(global_max_k + 1):
                row.append(str(p_k.get(str(k), "N/A")))
            writer.writerow(row)

    return aggregate_summary, json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate functional TAMP variants across benchmark domains under controlled search conditions."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=f"/tmp/pass2b41_sweep_{int(time.time())}",
        help="Output root directory for results (must not exist beforehand)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated list of variant names to evaluate (e.g. 'K1,K2,K7,L1,L7,W1,W9'). If omitted, runs all 32.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated list of domains to evaluate (e.g. 'kitchen,workshop'). If omitted, runs all domains.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gt", "vlm"],
        default="gt",
        help="Specification acquisition mode: 'gt' (ground truth specification) or 'vlm' (visual language model specification). Default: gt.",
    )
    parser.add_argument(
        "--search-order",
        type=str,
        choices=["auto", "oracle", "provider", "random"],
        default=None,
        help="Search order policy: 'oracle' (privileged GT reference), 'provider' (FM-guided when VLM), 'random' (seeded permutation), or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=None,
        help="Deterministic non-negative random seed for single-trial random search (--search-order random).",
    )
    parser.add_argument(
        "--random-seeds",
        type=str,
        default=None,
        help="Comma-separated list of non-negative integers for multi-seed random evaluation (e.g. '0,1,2,3,4,5,6,7,8,9').",
    )
    parser.add_argument(
        "--specification-root",
        type=str,
        default=None,
        help="Path to directory containing previously saved functional_specification.json files to replay for controlled comparison.",
    )
    args = parser.parse_args()

    # Preflight Check 1: Output root non-existence
    if os.path.exists(args.output_root):
        print(f"Error: Output root {args.output_root} already exists. Please delete it or use a fresh path.", file=sys.stderr)
        sys.exit(1)

    # Normalize search order
    search_order = args.search_order or "auto"

    # Preflight Check 2: Random seed arguments parsing & mutual exclusivity
    if args.search_seed is not None and args.random_seeds is not None:
        print("Error: Cannot supply both --search-seed and --random-seeds.", file=sys.stderr)
        sys.exit(1)

    if args.search_seed is not None and args.search_seed < 0:
        print(f"Error: Negative seed not permitted: {args.search_seed}", file=sys.stderr)
        sys.exit(1)

    parsed_seeds: Optional[List[int]] = None
    if args.random_seeds is not None:
        try:
            parsed_seeds = parse_random_seeds(args.random_seeds)
        except ValueError as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)

    # Preflight Check 3: Search order vs seed requirements
    if search_order == "random":
        if args.search_seed is None and parsed_seeds is None:
            print("Error: --search-order random requires either --search-seed INT or --random-seeds CSV.", file=sys.stderr)
            sys.exit(1)
    else:
        if args.search_seed is not None or parsed_seeds is not None:
            print(f"Error: Search seed(s) specified but --search-order is '{search_order}' (expected 'random').", file=sys.stderr)
            sys.exit(1)

    # Preflight Check 4: Mode & Search compatibility
    if args.mode == "vlm" and search_order == "oracle":
        print("Error: Oracle search is a privileged GT reference and cannot be run with mode=vlm.", file=sys.stderr)
        sys.exit(1)

    # Preflight Check 5: Multi-seed VLM random without specification-root
    if args.mode == "vlm" and search_order == "random" and parsed_seeds is not None and len(parsed_seeds) > 1 and args.specification_root is None:
        print(
            "Error: Multi-seed VLM random evaluation requires --specification-root so all seeds reuse the same saved G_F.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Filter domains / variants
    selected_domains = list(DOMAINS.keys())
    if args.domains:
        selected_domains = [d.strip() for d in args.domains.split(",") if d.strip() in DOMAINS]

    filter_variants = None
    if args.variants:
        filter_variants = {v.strip() for v in args.variants.split(",") if v.strip()}

    variant_queue: List[Tuple[str, str]] = []
    for domain in selected_domains:
        for variant in DOMAINS[domain]:
            if filter_variants is None or variant in filter_variants:
                variant_queue.append((domain, variant))

    if not variant_queue:
        print("Error: No valid domain variants matched selection filters.", file=sys.stderr)
        sys.exit(1)

    # Preflight Check 6: Living room compatibility
    if any(d == "living_room" for d, _ in variant_queue) and search_order in {"oracle", "random"}:
        print(
            f"Error: Living Room domain has no region search loop; '{search_order}' search is not applicable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Preflight Check 7: Replay specification preflight across ALL queued variants with SHA-256 precomputation
    spec_info_by_variant: Dict[Tuple[str, str], Dict[str, str]] = {}
    if args.specification_root:
        missing_specs = []
        hashing_failures = []
        for d, v in variant_queue:
            spec_candidate = os.path.join(args.specification_root, d, v, args.mode, "functional_specification.json")
            if not os.path.isfile(spec_candidate):
                missing_specs.append((d, v, spec_candidate))
            else:
                try:
                    sha_val = compute_file_sha256(spec_candidate)
                    spec_info_by_variant[(d, v)] = {
                        "path": spec_candidate,
                        "sha256": sha_val,
                    }
                except Exception as e:
                    hashing_failures.append((d, v, spec_candidate, str(e)))

        if missing_specs or hashing_failures:
            if missing_specs:
                print(
                    f"Error: Preflight failed. Missing {len(missing_specs)} replayed specification file(s) in {args.specification_root}:",
                    file=sys.stderr,
                )
                for d, v, path in missing_specs[:5]:
                    print(f"  - [{d} {v}]: {path}", file=sys.stderr)
                if len(missing_specs) > 5:
                    print(f"  ... and {len(missing_specs) - 5} more.", file=sys.stderr)
            if hashing_failures:
                print(
                    f"Error: Preflight failed. Unable to hash {len(hashing_failures)} replayed specification file(s):",
                    file=sys.stderr,
                )
                for d, v, path, err in hashing_failures[:5]:
                    print(f"  - [{d} {v}]: {path} ({err})", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_root, exist_ok=False)

    # Build sequence of trial runs
    seeds_list: List[Optional[int]] = [args.search_seed] if args.search_seed is not None else (parsed_seeds if parsed_seeds is not None else [None])

    results: List[Dict[str, Any]] = []
    attempted_count = 0
    completed_count = 0
    matching_count = 0
    valid_count = 0

    print(
        f"{'Domain':<12} {'Variant':<8} {'Expected':<22} {'Actual':<22} {'Match':<6} "
        f"{'Inspections':<12} {'PlanLen':<8} {'Combined':<9} {'Grounding':<10} "
        f"{'GroundingAudit':<15} {'ReplayValid':<12} {'AccessValid':<12} {'Runtime':<8} {'Condition'}"
    )
    print("-" * 175)

    for domain, variant in variant_queue:
        spec_info = spec_info_by_variant.get((domain, variant))
        spec_json_path = spec_info["path"] if spec_info else None
        spec_sha256_pre = spec_info["sha256"] if spec_info else None

        for seed_val in seeds_list:
            attempted_count += 1

            if search_order == "random" and seed_val is not None:
                trial_output_root = os.path.join(args.output_root, "trials", "random", f"seed_{seed_val:03d}")
                trial_id = f"{domain}_{variant}_{args.mode}_random_seed_{seed_val:03d}"
            else:
                trial_output_root = args.output_root
                trial_id = f"{domain}_{variant}_{args.mode}"

            res = evaluate_variant(
                domain,
                variant,
                args.output_root,
                mode=args.mode,
                search_order=search_order,
                search_seed=seed_val,
                specification_json=spec_json_path,
                source_specification_sha256=spec_sha256_pre,
                trial_output_root=trial_output_root,
                trial_id=trial_id,
            )
            results.append(res)

            if res["completed"]:
                completed_count += 1
            if res["match"] == "YES":
                matching_count += 1
            if res.get("evaluation_valid") is True:
                valid_count += 1

            audit_str = "N/A" if res["grounding_audit_valid"] is None else ("VALID" if res["grounding_audit_valid"] else "INVALID")
            replay_str = "N/A" if res["plan_replay_valid"] is None else ("VALID" if res["plan_replay_valid"] else "INVALID")
            access_str = "N/A" if res["accessibility_valid"] is None else ("VALID" if res["accessibility_valid"] else "INVALID")

            print(
                f"{res['domain']:<12} {res['variant']:<8} {res['expected_status']:<22} {res['actual_status']:<22} {res['match']:<6} "
                f"{res['inspection_count']:<12} {res['plan_length']:<8} {res['combined_high_level_count']:<9} {res['grounding_status']:<10} "
                f"{audit_str:<15} {replay_str:<12} {access_str:<12} {res['runtime_sec']:<8.2f} {res['condition_label']}"
            )

    # Multi-seed random aggregation if applicable
    random_aggregate_json_path = None
    if search_order == "random":
        _, random_aggregate_json_path, _ = aggregate_random_trials(results, args.output_root)

    # Write top-level summary.json
    summary_data = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": args.mode,
        "requested_search_order": search_order,
        "requested_search_seeds": [s for s in seeds_list if s is not None],
        "specification_root": args.specification_root,
        "random_aggregate_path": random_aggregate_json_path,
        "attempted_count": attempted_count,
        "completed_count": completed_count,
        "matching_count": matching_count,
        "scientifically_valid_count": valid_count,
        "exact_match_rate": (matching_count / attempted_count) if attempted_count > 0 else 0.0,
        "scientifically_valid_rate": (valid_count / attempted_count) if attempted_count > 0 else 0.0,
        "results": results,
    }
    with open(os.path.join(args.output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Write summary.csv preserving all legacy headers, appending Phase 3.3 and 3.3.1 columns
    csv_file = os.path.join(args.output_root, "summary.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            # Legacy headers
            "Domain",
            "Variant",
            "Expected",
            "Actual",
            "Match",
            "ReturnCode",
            "Completed",
            "Inspections",
            "InspectionSequence",
            "PlanLen",
            "CombinedCount",
            "Grounding",
            "GroundingComplete",
            "GroundingAuditValid",
            "PlanReplayValid",
            "AccessValid",
            "Runtime",
            "FailureReason",
            # Phase 3.3 additive headers
            "Mode",
            "Condition",
            "SearchOrderRequested",
            "SearchOrderEffective",
            "SearchSeedRequested",
            "SearchSeedEffective",
            "NOpen",
            "ProviderRegionRanking",
            "RegionOrderUsed",
            "ExplorationActuation",
            "SpecAcquisition",
            "SpecificationSHA256",
            "SourceSpecificationPath",
            "SourceSpecificationSHA256",
            "SpecificationHashMatch",
            "RunManifest",
            "TrialID",
            "TrialOutputRoot",
            "RunDir",
            # Phase 3.3.1 additive provenance headers
            "EvaluationValid",
            "EvaluationFailureReason",
            "ValidMatch",
            "PairingStatus",
            "SourceSpecificationSHA256After",
            "SourceSpecificationUnchanged",
            "EvaluatorModeRequested",
            "EvaluatorSearchOrderRequested",
            "EvaluatorSearchSeedRequested",
            "ManifestSpecMode",
            "ManifestSearchOrderRequested",
            "ManifestSearchOrderEffective",
            "ManifestSearchSeedRequested",
            "ManifestSearchSeedEffective",
            "ManifestTerminalStatus",
            "ProvenanceMatch",
            "ProvenanceMismatches",
        ])
        for r in results:
            audit_val = "N/A" if r["grounding_audit_valid"] is None else str(r["grounding_audit_valid"])
            replay_val = "N/A" if r["plan_replay_valid"] is None else str(r["plan_replay_valid"])
            access_val = "N/A" if r["accessibility_valid"] is None else str(r["accessibility_valid"])
            gr_complete = "N/A" if r["grounding_complete"] is None else str(r["grounding_complete"])
            writer.writerow([
                r["domain"],
                r["variant"],
                r["expected_status"],
                r["actual_status"],
                r["match"],
                r["return_code"],
                r["completed"],
                r["inspection_count"],
                ";".join(r["inspected_regions"]),
                r["plan_length"],
                r["combined_high_level_count"],
                r["grounding_status"],
                gr_complete,
                audit_val,
                replay_val,
                access_val,
                f"{r['runtime_sec']:.2f}",
                r["failure_reason"] or "",
                r["spec_mode"],
                r["condition_label"],
                r["search_order_requested"],
                r["search_order_effective"] or "",
                "" if r["search_seed_requested"] is None else str(r["search_seed_requested"]),
                "" if r["search_seed_effective"] is None else str(r["search_seed_effective"]),
                r["n_open"],
                ";".join(r.get("provider_region_ranking", [])),
                ";".join(r.get("region_order_used", [])),
                r.get("exploration_actuation", ""),
                r.get("spec_acquisition", ""),
                r.get("specification_sha256") or "",
                r.get("source_specification_path") or "",
                r.get("source_specification_sha256") or "",
                "" if r.get("specification_hash_match") is None else str(r["specification_hash_match"]),
                r.get("run_manifest_path") or "",
                r.get("trial_id", ""),
                r.get("trial_output_root", ""),
                r.get("run_dir", ""),
                str(r.get("evaluation_valid", True)),
                r.get("evaluation_failure_reason") or "",
                r.get("valid_match", "N/A"),
                r.get("pairing_status", "NOT_APPLICABLE"),
                r.get("source_specification_sha256_after") or "",
                "" if r.get("source_specification_unchanged") is None else str(r["source_specification_unchanged"]),
                r.get("evaluator_mode_requested", ""),
                r.get("evaluator_search_order_requested", ""),
                "" if r.get("evaluator_search_seed_requested") is None else str(r["evaluator_search_seed_requested"]),
                r.get("manifest_spec_mode") or "",
                r.get("manifest_search_order_requested") or "",
                r.get("manifest_search_order_effective") or "",
                "" if r.get("manifest_search_seed_requested") is None else str(r["manifest_search_seed_requested"]),
                "" if r.get("manifest_search_seed_effective") is None else str(r["manifest_search_seed_effective"]),
                r.get("manifest_terminal_status") or "",
                "" if r.get("provenance_match") is None else str(r["provenance_match"]),
                ";".join(r.get("provenance_mismatches", [])),
            ])

    total_variants_queued = len(variant_queue) * len(seeds_list)
    print("-" * 175)
    print(f"Attempted: {attempted_count} / {total_variants_queued}")
    print(f"Completed: {completed_count} / {total_variants_queued}")
    print(f"Match: {matching_count} / {attempted_count}")
    print(f"Scientifically Valid: {valid_count} / {attempted_count}")


if __name__ == "__main__":
    main()
