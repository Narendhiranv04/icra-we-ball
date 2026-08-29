#!/usr/bin/env python3
"""
Evaluate functional TAMP variants across benchmark domains (Kitchen, Living Room, Workshop).
Provides clean execution, return code checking, artifact inspection, plan validation,
grounding audits, structured JSON/CSV logging, and summary reporting.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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


def evaluate_variant(domain: str, variant: str, output_root: str) -> Dict[str, Any]:
    """
    Launch canonical runner for one domain variant, record runtime and return code,
    and parse resulting execution and grounding artifacts.
    """
    expected = EXPECTED[domain][variant]
    cmd = [
        sys.executable,
        "-m", "mujoco_scenes.functional_tamp_pipeline.run",
        "--domain", domain,
        "--variant", variant,
        "--mode", "gt",
        "--dry-run",
        "--output-root", output_root,
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."},
    )
    runtime = time.time() - t0

    run_dir = os.path.join(output_root, domain, variant, "gt")
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

    match = "YES" if actual == expected else "NO"
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

    return {
        "domain": domain,
        "variant": variant,
        "expected_status": expected,
        "actual_status": actual,
        "match": match,
        "return_code": proc.returncode,
        "completed": is_completed,
        "runtime_sec": runtime,
        "inspected_regions": inspected_regions,
        "inspection_count": inspection_count,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate functional TAMP variants across benchmark domains.")
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
    args = parser.parse_args()

    if os.path.exists(args.output_root):
        print(f"Error: Output root {args.output_root} already exists. Please delete it or use a fresh path.")
        sys.exit(1)

    os.makedirs(args.output_root, exist_ok=False)

    # Filter domains / variants if specified
    selected_domains = list(DOMAINS.keys())
    if args.domains:
        selected_domains = [d.strip() for d in args.domains.split(",") if d.strip() in DOMAINS]

    filter_variants = None
    if args.variants:
        filter_variants = {v.strip() for v in args.variants.split(",") if v.strip()}

    total_target_variants = 0
    variant_queue: List[Tuple[str, str]] = []
    for domain in selected_domains:
        for variant in DOMAINS[domain]:
            if filter_variants is None or variant in filter_variants:
                variant_queue.append((domain, variant))
                total_target_variants += 1

    results: List[Dict[str, Any]] = []
    attempted_count = 0
    completed_count = 0
    matching_count = 0

    print(
        f"{'Domain':<12} {'Variant':<8} {'Expected':<22} {'Actual':<22} {'Match':<6} "
        f"{'Inspections':<12} {'PlanLen':<8} {'Combined':<9} {'Grounding':<10} "
        f"{'GroundingAudit':<15} {'ReplayValid':<12} {'AccessValid':<12} {'Runtime':<8} {'FailureReason'}"
    )
    print("-" * 165)

    for domain, variant in variant_queue:
        attempted_count += 1
        res = evaluate_variant(domain, variant, args.output_root)
        results.append(res)

        if res["completed"]:
            completed_count += 1
        if res["match"] == "YES":
            matching_count += 1

        audit_str = "N/A" if res["grounding_audit_valid"] is None else ("VALID" if res["grounding_audit_valid"] else "INVALID")
        replay_str = "N/A" if res["plan_replay_valid"] is None else ("VALID" if res["plan_replay_valid"] else "INVALID")
        access_str = "N/A" if res["accessibility_valid"] is None else ("VALID" if res["accessibility_valid"] else "INVALID")
        fail_str = str(res["failure_reason"]) if res["failure_reason"] is not None else "None"
        if len(fail_str) > 40:
            fail_str = fail_str[:37] + "..."

        print(
            f"{res['domain']:<12} {res['variant']:<8} {res['expected_status']:<22} {res['actual_status']:<22} {res['match']:<6} "
            f"{res['inspection_count']:<12} {res['plan_length']:<8} {res['combined_high_level_count']:<9} {res['grounding_status']:<10} "
            f"{audit_str:<15} {replay_str:<12} {access_str:<12} {res['runtime_sec']:<8.2f} {fail_str}"
        )

    # Write summary.json
    summary_data = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attempted_count": attempted_count,
        "completed_count": completed_count,
        "matching_count": matching_count,
        "exact_match_rate": (matching_count / attempted_count) if attempted_count > 0 else 0.0,
        "results": results,
    }
    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    # Write summary.csv with standard csv writer
    csv_file = os.path.join(args.output_root, "summary.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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
            ])

    print("-" * 165)
    print(f"Attempted: {attempted_count} / {total_target_variants}")
    print(f"Completed: {completed_count} / {total_target_variants}")
    print(f"Match: {matching_count} / {attempted_count}")


if __name__ == "__main__":
    main()
