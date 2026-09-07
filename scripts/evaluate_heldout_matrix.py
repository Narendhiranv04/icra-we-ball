#!/usr/bin/env python3
"""Phase 14: Held-Out Generalization Matrix Evaluator.

Evaluates 15 genuinely unseen variants (5 Kitchen, 5 Living Room, 5 Workshop;
9 feasible, 6 infeasible) using the frozen V2 prompt, schema, and runtime ontology.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph, PipelineResult
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import evaluate_gf_against_reference
from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline
from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
    candidate_plan_valid,
    enrich_record,
    format_rate,
    full_task_coverage,
    write_detailed_report,
)

DOMAINS = {
    "kitchen": [f"HK{i}" for i in range(1, 6)],
    "living_room": [f"HL{i}" for i in range(1, 6)],
    "workshop": [f"HW{i}" for i in range(1, 6)],
}

FEASIBLE_VARIANTS = {
    "kitchen": ["HK1", "HK2", "HK3"],
    "living_room": ["HL1", "HL2", "HL3"],
    "workshop": ["HW1", "HW2", "HW3"],
}

RECOVERY_VARIANTS = {
    "kitchen": ["HK1", "HK2", "HK3"],
    "living_room": [],
    "workshop": ["HW1", "HW2", "HW3"],
}

CANONICAL_TASK_INSTRUCTIONS = {
    "kitchen": "Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil.",
    "living_room": "Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people.",
    "workshop": "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench.",
}


def _evaluate_plan_against_gt(
    domain: str,
    variant: str,
    plan_actions: tuple[dict[str, Any], ...],
    run_dir: Path,
) -> Tuple[int, int, float, bool]:
    return full_task_coverage(domain, run_dir)


def evaluate_heldout_variants(
    *,
    mode: str = "vlm",
    spec_source: str = "live",
    output_root: Path,
    dry_run: bool = True,
    resume: bool = False,
    variants: Optional[str] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    if spec_source == "live" and not resume and output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Refusing to overwrite an existing live matrix: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    variant_filter = set(v.strip() for v in variants.split(",")) if variants else None

    records: List[Dict[str, Any]] = []
    if resume:
        records_path = output_root / "evaluation_records.json"
        if not records_path.exists():
            raise ValueError(f"Cannot resume without {records_path}")
        records = json.loads(records_path.read_text(encoding="utf-8"))
        if any(r.get("terminal_status") == "PIPELINE_EXCEPTION" for r in records):
            raise ValueError("Remove/archive trailing failed record before resuming")
    completed = {(r["domain"], r["variant"]) for r in records}

    total_variants = sum(len(v) for v in DOMAINS.values())
    total_feasible = sum(len(v) for v in FEASIBLE_VARIANTS.values())
    total_infeasible = total_variants - total_feasible
    print(f"Evaluating Held-Out Generalization Matrix: {total_variants} variants ({total_feasible} feasible, {total_infeasible} infeasible)...")

    for domain, all_domain_variants in DOMAINS.items():
        variants_to_run = [v for v in all_domain_variants if variant_filter is None or v in variant_filter]
        if not variants_to_run:
            continue

        print(f"\n=== Running Domain: {domain.upper()} ({len(variants_to_run)} held-out variants) ===")
        for variant in variants_to_run:
            if (domain, variant) in completed:
                print(f"  [{variant}] preserved from prior segment")
                continue
            is_feasible = variant in FEASIBLE_VARIANTS[domain]
            is_recovery = variant in RECOVERY_VARIANTS[domain]

            t0 = time.perf_counter()
            try:
                result = run_pipeline(
                    domain=domain,
                    variant=variant,
                    mode=mode,
                    output_root=output_root,
                    dry_run=dry_run,
                )
                pipeline_res = result
            except Exception as exc:
                print(f"  [{variant}] PIPELINE RUN EXCEPTION: {exc}")
                pipeline_res = PipelineResult(
                    domain=domain, variant=variant, mode=mode,
                    status="PIPELINE_EXCEPTION", failure_reason=str(exc),
                )
            runtime_sec = time.perf_counter() - t0

            run_dir = output_root / domain / variant / mode
            spec_file = run_dir / "functional_specification.json"

            if spec_file.exists():
                try:
                    graph_dict = json.loads(spec_file.read_text(encoding="utf-8"))
                    graph = FunctionalRequirementGraph.from_dict(graph_dict)
                    ref_eval = evaluate_gf_against_reference(graph, pipeline_result=pipeline_res)
                except Exception as exc:
                    print(f"  [{variant}] GF EVAL EXCEPTION: {exc}")
                    ref_eval = None
            else:
                ref_eval = None

            candidate_plan = pipeline_res.candidate_plan or ()
            full_gt_goals, sat_gt_goals, full_task_cov, full_task_sat = _evaluate_plan_against_gt(
                domain, variant, candidate_plan, run_dir
            )

            false_completion = bool(not is_feasible and full_task_sat)
            cand_stats = pipeline_res.candidate_search_statistics or pipeline_res.search_statistics or {}
            cand_total = cand_stats.get("total_goals", 0)
            cand_sat = cand_stats.get("satisfied_goals", 0)
            cand_cov = (cand_sat / cand_total) if cand_total > 0 else 0.0

            if is_feasible:
                outcome_correct = bool(full_task_sat)
            else:
                outcome_correct = bool(
                    not full_task_sat
                    and not false_completion
                    and pipeline_res.status in {
                        "INFEASIBLE",
                        "EXHAUSTED_NO_VALID_GROUNDING",
                        "NO_VALID_COMPLETE_ASSIGNMENT",
                        "PLANNING_PROVEN_INFEASIBLE",
                        "NO_MEANINGFUL_CANDIDATE_PLAN",
                        "VLM_SPEC_FAILED",
                    }
                )

            manifest: Dict[str, Any] = {}
            man_p = run_dir / "run_manifest.json"
            if man_p.exists():
                try:
                    manifest = json.loads(man_p.read_text(encoding="utf-8"))
                except Exception:
                    pass

            vlm_diag: Dict[str, Any] = {}
            diag_p = run_dir / "fm_diagnostics" / "fm_call_001.json"
            if diag_p.exists():
                try:
                    vlm_diag = json.loads(diag_p.read_text(encoding="utf-8"))
                except Exception:
                    pass

            spec_acq = manifest.get("spec_acquisition", "live_provider")
            spec_inp = manifest.get("specification_input")
            sem_vlm_req = manifest.get("semantic_vlm_requests", 1 if mode == "vlm" else 0)
            vlm_req_count = manifest.get("vlm_request_count", sem_vlm_req)
            astar_invs = manifest.get("astar_invocations", 1 if (candidate_plan or pipeline_res.search_statistics) else 0)
            replans = manifest.get("high_level_replans", 0)

            search_contract_data = manifest.get("search_contract", {})
            regions_avail = list(search_contract_data.get("regions", [])) if search_contract_data else []
            if not regions_avail:
                if domain == "workshop":
                    regions_avail = ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"]
                elif domain == "kitchen":
                    regions_avail = ["D1", "D2", "C2", "B1", "C1"]
                else:
                    regions_avail = []
            insp_order_src = manifest.get("search_order_source_effective", "deterministic_system")
            insp_order_used = manifest.get("region_order_used", list(pipeline_res.inspected_regions))
            search_exh = (len(pipeline_res.inspected_regions) >= len(regions_avail)) if regions_avail else False

            row: Dict[str, Any] = {
                "domain": domain,
                "variant": variant,
                "mode": mode,
                "gt_feasible": is_feasible,
                "requires_observation_recovery": is_recovery,
                "spec_acquisition": spec_acq,
                "specification_input": spec_inp,
                "semantic_vlm_requests": sem_vlm_req,
                "vlm_request_count": vlm_req_count,
                "raw_vlm_spec_complete": bool(ref_eval.reference_complete if ref_eval else False),
                "runtime_contract_complete": bool(ref_eval.runtime_contract_role_coverage >= 1.0 if ref_eval else False),
                "canonicalization_succeeded": bool(pipeline_res.canonicalization_succeeded),
                "expressed_role_count": len(ref_eval.raw_vlm_roles if ref_eval else []),
                "missing_reference_roles": list(ref_eval.missing_roles if ref_eval else []),
                "environment_projected_roles": list(ref_eval.environment_projected_roles if ref_eval else []),
                "regions_available": regions_avail,
                "regions_inspected": list(pipeline_res.inspected_regions),
                "inspection_order_source": insp_order_src,
                "inspection_order_used": insp_order_used,
                "search_exhausted": search_exh,
                "search_required_by_offline_benchmark": is_recovery,
                "candidate_grounding_eligible": bool(ref_eval.candidate_grounding_eligible if ref_eval else False),
                "candidate_grounding_succeeded": bool(ref_eval.candidate_grounding_succeeded if ref_eval else False),
                "candidate_goal_count": cand_total,
                "candidate_goal_satisfied_count": cand_sat,
                "candidate_goal_coverage": cand_cov,
                "uninstantiable_goal_count": max(0, cand_total - cand_sat),
                "uninstantiable_reasons": [pipeline_res.failure_reason] if pipeline_res.failure_reason else [],
                "candidate_plan_status": pipeline_res.status,
                "candidate_plan_length": len(candidate_plan),
                "candidate_plan_valid": candidate_plan_valid(run_dir),
                "full_task_goal_count": full_gt_goals,
                "full_task_goal_satisfied_count": sat_gt_goals,
                "full_task_goal_coverage": full_task_cov,
                "full_task_satisfied": full_task_sat,
                "false_completion": false_completion,
                "outcome_correct": outcome_correct,
                "astar_invocations": astar_invs,
                "high_level_replans": replans,
                "astar_expanded": cand_stats.get("expanded_states", 0),
                "astar_generated": cand_stats.get("generated_states", 0),
                "runtime_seconds": runtime_sec,
                "pipeline_runtime_sec": round(runtime_sec, 2),
                "terminal_status": pipeline_res.status,
                "failure_reason": pipeline_res.failure_reason,
            }

            row = enrich_record(row, run_dir, CANONICAL_TASK_INSTRUCTIONS[domain])
            records.append(row)
            completed.add((domain, variant))

            (output_root / "evaluation_records.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )

            status_glyph = "CORRECT" if row.get("outcome_correct") else "INCORRECT"
            fc = row.get("first_cause_category")
            print(f"  [{variant}] status={pipeline_res.status} outcome={status_glyph} (runtime={runtime_sec:.1f}s, first_cause={fc})")

    # Output records CSV and JSON
    (output_root / "evaluation_records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    if records:
        with (output_root / "evaluation_records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    # Compute aggregate metrics
    n_total = len(records)
    outcome_correct_count = sum(1 for r in records if r.get("outcome_correct"))
    outcome_correct_pct = (outcome_correct_count / n_total) * 100 if n_total else 0.0

    feasible_rows = [r for r in records if r["gt_feasible"]]
    infeasible_rows = [r for r in records if not r["gt_feasible"]]
    feasible_success_count = sum(1 for r in feasible_rows if r.get("full_task_satisfied"))
    feasible_success_pct = (feasible_success_count / len(feasible_rows)) * 100 if feasible_rows else 0.0

    recovery_rows = [r for r in records if r["requires_observation_recovery"]]
    recovery_success_count = sum(1 for r in recovery_rows if r.get("full_task_satisfied"))
    recovery_success_pct = (recovery_success_count / len(recovery_rows)) * 100 if recovery_rows else 0.0

    goal_cov_pct = (sum(r.get("full_task_goal_coverage", 0.0) for r in records) / n_total) * 100 if n_total else 0.0
    false_completion_count = sum(1 for r in records if r.get("false_completion"))
    false_completion_pct = (false_completion_count / n_total) * 100 if n_total else 0.0

    vlm_req_mean = sum(r.get("semantic_vlm_requests", 0) for r in records) / n_total if n_total else 0.0
    replans_mean = sum(r.get("high_level_replans", 0) for r in records) / n_total if n_total else 0.0

    primary_metrics = {
        "outcome_correct": outcome_correct_pct,
        "feasible_success": feasible_success_pct,
        "feasibility_recovery": recovery_success_pct,
        "goal_coverage": goal_cov_pct,
        "false_completion": false_completion_pct,
        "vlm_requests": vlm_req_mean,
        "high_level_replans": replans_mean,
    }

    # Main Paper Table (Section 39)
    table_md = f"""# Section 39: Main Paper Table (Held-Out Generalization Matrix)

| Method | Outcome Correct ↑ | Feasible-task Success ↑ | Feasibility Recovery ↑ | Goal Coverage ↑ | False Completion ↓ | VLM Requests ↓ | Replans ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ours (FM-Grounding)** | **{outcome_correct_pct:.1f}%** | **{feasible_success_pct:.1f}%** | **{recovery_success_pct:.1f}%** | **{goal_cov_pct:.1f}%** | **{false_completion_pct:.1f}%** | **{vlm_req_mean:.2f}** | **{replans_mean:.2f}** |
"""
    (output_root / "main_paper_table.md").write_text(table_md, encoding="utf-8")

    # Generate pipeline diagnostic table and failure analysis via standard reporter
    write_detailed_report(output_root, records, live=False)

    # Invariants verification (Section AJ / Gate 14)
    errors: List[str] = []
    if variant_filter is None:
        if len(records) != 15:
            errors.append(f"Expected 15 variants, got {len(records)}")
        if sum(1 for r in records if r["gt_feasible"]) != 9:
            errors.append(f"Expected 9 feasible variants, got {sum(1 for r in records if r['gt_feasible'])}")
        if sum(1 for r in records if not r["gt_feasible"]) != 6:
            errors.append(f"Expected 6 infeasible variants, got {sum(1 for r in records if not r['gt_feasible'])}")
        if sum(1 for r in records if r["requires_observation_recovery"]) != 6:
            errors.append(f"Expected 6 recovery variants, got {sum(1 for r in records if r['requires_observation_recovery'])}")

    if spec_source == "live":
        for r in records:
            if r.get("semantic_vlm_requests") != 1:
                errors.append(f"{r['variant']}: semantic_vlm_requests={r.get('semantic_vlm_requests')} != 1")
            if r.get("high_level_replans") != 0:
                errors.append(f"{r['variant']}: high_level_replans={r.get('high_level_replans')} != 0")
            if r.get("spec_acquisition") != "live_provider":
                errors.append(f"{r['variant']}: spec_acquisition={r.get('spec_acquisition')} != 'live_provider'")
            if r.get("specification_input") is not None:
                errors.append(f"{r['variant']}: specification_input is not None")
        if len({r.get("prompt_hash") for r in records}) != 1:
            errors.append("Mixed prompt hashes")
        if len({r.get("model") for r in records}) != 1:
            errors.append("Mixed models")

    invariants_payload = {
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "variant_count": len(records),
        "feasible_count": sum(1 for r in records if r["gt_feasible"]),
        "infeasible_count": sum(1 for r in records if not r["gt_feasible"]),
        "vlm_requests_per_variant": vlm_req_mean,
        "replans_per_variant": replans_mean,
    }
    (output_root / "invariants.json").write_text(json.dumps(invariants_payload, indent=2) + "\n", encoding="utf-8")

    diag_md_path = output_root / "pipeline_diagnostic_table.md"
    diag_md = diag_md_path.read_text(encoding="utf-8") if diag_md_path.exists() else ""

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "spec_source": spec_source,
        "total_variants": n_total,
        "feasible_variants": len(feasible_rows),
        "infeasible_variants": len(infeasible_rows),
        "primary_metrics": primary_metrics,
        "invariants": invariants_payload,
    }
    (output_root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== HELDOUT EVALUATION COMPLETE ===")
    print(table_md)
    print(diag_md)
    print("Invariants:", json.dumps(invariants_payload, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("vlm", "gt"), default="vlm")
    parser.add_argument("--spec-source", choices=("live", "replay"), default="live")
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_reports/held_out_generalization_15x1_20260908"))
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variants", type=str, default=None)
    args = parser.parse_args()

    evaluate_heldout_variants(
        mode=args.mode,
        spec_source=args.spec_source,
        output_root=args.output_root,
        dry_run=args.dry_run,
        resume=args.resume,
        variants=args.variants,
    )


if __name__ == "__main__":
    main()
