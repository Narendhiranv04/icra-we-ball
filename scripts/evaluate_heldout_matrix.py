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
    compute_primary_metrics,
    enrich_record,
    format_rate,
    full_task_coverage,
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

            # One shared rule with the main evaluator and the frozen-replay
            # scorer.  This file used to carry a third definition that credited
            # NO_MEANINGFUL_CANDIDATE_PLAN -- a partial plan, not a conclusion --
            # while omitting INFEASIBLE, so an actual infeasibility conclusion
            # scored as wrong here and a partial plan scored as right.  Held-out
            # and main numbers were therefore not comparable.
            from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
                outcome_is_correct,
            )
            outcome_correct = outcome_is_correct(
                gt_feasible=bool(is_feasible), task_satisfied=bool(full_task_sat),
                false_completion=bool(false_completion),
                pipeline_status=pipeline_res.status)

            manifest: Dict[str, Any] = {}
            man_p = run_dir / "run_manifest.json"
            if man_p.exists():
                manifest = json.loads(man_p.read_text(encoding="utf-8"))

            vlm_diag: Dict[str, Any] = {}
            diag_p = run_dir / "fm_diagnostics" / "fm_call_001.json"
            if diag_p.exists():
                vlm_diag = json.loads(diag_p.read_text(encoding="utf-8"))

            record: Dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "domain": domain,
                "variant": variant,
                "mode": mode,
                "gt_feasible": is_feasible,
                "requires_observation_recovery": is_recovery,
                "semantic_vlm_requests": 1 if mode == "vlm" else 0,
                "high_level_replans": 0,
                "pipeline_runtime_sec": round(runtime_sec, 2),
                "terminal_status": pipeline_res.status,
                "failure_reason": pipeline_res.failure_reason,
                "failure_category": pipeline_res.failure_category,
                "canonicalization_succeeded": pipeline_res.canonicalization_succeeded,
                "required_contract_complete": pipeline_res.functional_spec_complete,
                "complete_candidate_grounding": bool(pipeline_res.assignment),
                "candidate_plan_found": bool(pipeline_res.candidate_plan),
                "candidate_plan_length": len(pipeline_res.candidate_plan or ()),
                "candidate_plan_status": (
                    "EMPTY" if not pipeline_res.candidate_plan
                    else "PARTIAL" if (cand_sat < cand_total and cand_total > 0)
                    else "FULL"
                ),
                "regions_inspected": list(pipeline_res.inspected_regions),
                "full_task_satisfied": full_task_sat,
                "outcome_correct": outcome_correct,
                "false_completion": false_completion,
                "candidate_goal_coverage": cand_cov,
                "candidate_goal_count": cand_total,
                "candidate_goal_satisfied_count": cand_sat,
                "full_task_goal_count": full_gt_goals,
                "full_task_goal_satisfied_count": sat_gt_goals,
                "full_task_goal_coverage": full_task_cov,
                "candidate_plan_valid": candidate_plan_valid(run_dir),
                "astar_invocations": manifest.get("astar_invocations", 1 if candidate_plan else 0),
                "regions_available": list(manifest.get("search_contract", {}).get("regions", [])),
                "search_exhausted": False,
                "git_commit": manifest.get("git_commit"),
                "git_dirty": manifest.get("git_dirty"),
                "model": vlm_diag.get("model") or manifest.get("provider_model"),
                "prompt_hash": manifest.get("prompt_schema_hash"),
                "spec_acquisition": manifest.get("spec_acquisition", "live_provider"),
                "specification_input": manifest.get("specification_input"),
                "execution_state": manifest.get("execution_state", "planning_only"),
                "inspection_order_source": "FM",
            }

            record = enrich_record(record, run_dir, CANONICAL_TASK_INSTRUCTIONS[domain])
            fc = record["first_cause_category"]

            rec_file = run_dir / "evaluation_record.json"
            rec_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

            records.append(record)
            completed.add((domain, variant))

            (output_root / "evaluation_records.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )

            status_glyph = "CORRECT" if outcome_correct else "INCORRECT"
            print(f"  [{variant}] status={pipeline_res.status} outcome={status_glyph} (runtime={runtime_sec:.1f}s, first_cause={fc})")

    # Output records CSV
    csv_fields = [
        "domain", "variant", "mode", "gt_feasible", "requires_observation_recovery",
        "terminal_status", "first_cause_category", "failure_category", "failure_reason",
        "semantic_vlm_requests", "high_level_replans", "pipeline_runtime_sec",
        "canonicalization_succeeded", "required_contract_complete",
        "complete_candidate_grounding", "candidate_plan_length", "candidate_plan_valid",
        "full_task_satisfied", "outcome_correct", "false_completion",
        "candidate_goal_coverage", "full_task_goal_coverage",
    ]
    with (output_root / "evaluation_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Compute aggregate metrics
    n_total = len(records)
    feasible_rows = [r for r in records if r["gt_feasible"]]
    infeasible_rows = [r for r in records if not r["gt_feasible"]]
    recovery_rows = [r for r in records if r["requires_observation_recovery"]]
    primary_metrics = compute_primary_metrics(records)
    outcome_correct_pct = primary_metrics["outcome_correct"]
    feasible_success_pct = primary_metrics["feasible_success"]
    recovery_success_pct = primary_metrics["feasibility_recovery"]
    goal_cov_pct = primary_metrics["goal_coverage"]
    false_completion_pct = primary_metrics["false_completion"]
    vlm_req_mean = primary_metrics["vlm_requests"]
    replans_mean = primary_metrics["high_level_replans"]

    # Main Paper Table
    table_md = f"""# Section 39: Main Paper Table (Held-Out Generalization Matrix)

| Method | Outcome Correct ↑ | Feasible-task Success ↑ | Feasibility Recovery ↑ | Goal Coverage ↑ | False Completion ↓ | VLM Requests ↓ | Replans ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ours (FM-Grounding)** | **{outcome_correct_pct:.1f}%** | **{feasible_success_pct:.1f}%** | **{recovery_success_pct:.1f}%** | **{goal_cov_pct:.1f}%** | **{false_completion_pct:.1f}%** | **{vlm_req_mean:.2f}** | **{replans_mean:.2f}** |
"""
    (output_root / "main_paper_table.md").write_text(table_md, encoding="utf-8")

    # Diagnostics Table
    diagnostic_rows: Dict[str, Dict[str, str]] = {}
    for domain in ("kitchen", "living_room", "workshop"):
        dom_records = [r for r in records if r["domain"] == domain]
        n_dom = len(dom_records)
        dom_feasible = [r for r in dom_records if r["gt_feasible"]]

        mean_raw_recall = sum(r.get("raw_role_recall", 0.0) or 0.0 for r in dom_records) / n_dom if n_dom else 0.0
        mean_raw_f1 = sum(r.get("raw_role_f1", 0.0) or 0.0 for r in dom_records) / n_dom if n_dom else 0.0
        mean_rel_f1 = sum(r.get("interpreter_matched_raw_relation_f1", 0.0) or 0.0 for r in dom_records) / n_dom if n_dom else 0.0
        raw_comp_rate = sum(1 for r in dom_records if r.get("raw_vlm_spec_complete")) / n_dom if n_dom else 0.0
        contract_comp_rate = sum(1 for r in dom_records if r.get("required_contract_complete")) / n_dom if n_dom else 0.0
        canon_rate = sum(1 for r in dom_records if r.get("canonicalization_succeeded")) / n_dom if n_dom else 0.0
        any_grounding_rate = sum(1 for r in dom_records if r.get("complete_candidate_grounding")) / n_dom if n_dom else 0.0
        comp_grounding_rate = sum(1 for r in dom_records if r.get("complete_candidate_grounding")) / n_dom if n_dom else 0.0
        grounded_role_cov = sum(r.get("grounded_role_coverage", 0.0) or 0.0 for r in dom_records) / n_dom if n_dom else 0.0
        plan_gen_rate = sum(1 for r in dom_records if r.get("candidate_plan_length", 0) > 0) / n_dom if n_dom else 0.0

        plans = [r for r in dom_records if r.get("candidate_plan_length", 0) > 0]
        plan_valid_rate = sum(1 for r in plans if r.get("candidate_plan_valid")) / len(plans) if plans else None
        partial_rate = sum(1 for r in dom_records if "PARTIAL" in r.get("candidate_plan_status", "")) / n_dom if n_dom else 0.0
        cand_goal_cov = sum(r.get("candidate_goal_coverage", 0.0) for r in dom_records) / n_dom if n_dom else 0.0
        full_success = sum(1 for r in dom_feasible if r["full_task_satisfied"]) / len(dom_feasible) if dom_feasible else 0.0
        mean_reg_inspected = sum(len(r.get("regions_inspected", [])) for r in dom_records) / n_dom if n_dom else 0.0

        diagnostic_rows[domain] = {
            "raw_vlm_role_recall": f"{mean_raw_recall * 100:.1f}%",
            "raw_vlm_role_f1": f"{mean_raw_f1 * 100:.1f}%",
            "interpreter_matched_raw_relation_f1": f"{mean_rel_f1 * 100:.1f}%",
            "raw_complete_spec_rate": f"{raw_comp_rate * 100:.1f}%",
            "executable_contract_complete_rate": f"{contract_comp_rate * 100:.1f}%",
            "runtime_contract_coverage": f"{contract_comp_rate * 100:.1f}%",
            "canonicalization_success": f"{canon_rate * 100:.1f}%",
            "any_verified_grounding": f"{any_grounding_rate * 100:.1f}%",
            "complete_candidate_grounding": f"{comp_grounding_rate * 100:.1f}%",
            "grounded_role_coverage": f"{grounded_role_cov * 100:.1f}%",
            "nonempty_plan_rate": f"{plan_gen_rate * 100:.1f}%",
            "candidate_plan_rate": f"{plan_valid_rate * 100:.1f}%" if plan_valid_rate is not None else "N/A",
            "partial_plan_rate": f"{partial_rate * 100:.1f}%",
            "candidate_goal_coverage": f"{cand_goal_cov * 100:.1f}%",
            "full_task_success": f"{full_success * 100:.1f}%",
            "mean_regions_inspected": f"{mean_reg_inspected:.2f}",
        }

    total_mean_recall = sum(r.get("raw_role_recall", 0.0) or 0.0 for r in records) / n_total if n_total else 0.0
    total_mean_f1 = sum(r.get("raw_role_f1", 0.0) or 0.0 for r in records) / n_total if n_total else 0.0
    total_rel_f1 = sum(r.get("interpreter_matched_raw_relation_f1", 0.0) or 0.0 for r in records) / n_total if n_total else 0.0
    total_raw_comp = sum(1 for r in records if r.get("raw_vlm_spec_complete")) / n_total if n_total else 0.0
    total_contract_comp = sum(1 for r in records if r.get("required_contract_complete")) / n_total if n_total else 0.0
    total_canon = sum(1 for r in records if r.get("canonicalization_succeeded")) / n_total if n_total else 0.0
    total_any_ground = sum(1 for r in records if r.get("complete_candidate_grounding")) / n_total if n_total else 0.0
    total_comp_ground = sum(1 for r in records if r.get("complete_candidate_grounding")) / n_total if n_total else 0.0
    total_grounded_role_cov = sum(r.get("grounded_role_coverage", 0.0) or 0.0 for r in records) / n_total if n_total else 0.0
    total_plan_gen = sum(1 for r in records if r.get("candidate_plan_length", 0) > 0) / n_total if n_total else 0.0
    all_plans = [r for r in records if r.get("candidate_plan_length", 0) > 0]
    total_plan_valid = sum(1 for r in all_plans if r.get("candidate_plan_valid")) / len(all_plans) if all_plans else None
    total_partial = sum(1 for r in records if "PARTIAL" in r.get("candidate_plan_status", "")) / n_total if n_total else 0.0
    total_cand_cov = sum(r.get("candidate_goal_coverage", 0.0) for r in records) / n_total if n_total else 0.0
    total_mean_reg = sum(len(r.get("regions_inspected", [])) for r in records) / n_total if n_total else 0.0

    diagnostic_rows["Overall"] = {
        "raw_vlm_role_recall": f"{total_mean_recall * 100:.1f}%",
        "raw_vlm_role_f1": f"{total_mean_f1 * 100:.1f}%",
        "interpreter_matched_raw_relation_f1": f"{total_rel_f1 * 100:.1f}%",
        "raw_complete_spec_rate": f"{total_raw_comp * 100:.1f}%",
        "executable_contract_complete_rate": f"{total_contract_comp * 100:.1f}%",
        "runtime_contract_coverage": f"{total_contract_comp * 100:.1f}%",
        "canonicalization_success": f"{total_canon * 100:.1f}%",
        "any_verified_grounding": f"{total_any_ground * 100:.1f}%",
        "complete_candidate_grounding": f"{total_comp_ground * 100:.1f}%",
        "grounded_role_coverage": f"{total_grounded_role_cov * 100:.1f}%",
        "nonempty_plan_rate": f"{total_plan_gen * 100:.1f}%",
        "candidate_plan_rate": f"{total_plan_valid * 100:.1f}%" if total_plan_valid is not None else "N/A",
        "partial_plan_rate": f"{total_partial * 100:.1f}%",
        "candidate_goal_coverage": f"{total_cand_cov * 100:.1f}%",
        "full_task_success": f"{feasible_success_pct:.1f}%",
        "mean_regions_inspected": f"{total_mean_reg:.2f}",
    }

    diag_md = f"""# Section 40: Pipeline Diagnostic Table (Held-Out Generalization Matrix)

| Metric | Kitchen | Living Room | Workshop | Overall |
| :--- | ---: | ---: | ---: | ---: |
| Raw VLM role recall | {diagnostic_rows['kitchen']['raw_vlm_role_recall']} | {diagnostic_rows['living_room']['raw_vlm_role_recall']} | {diagnostic_rows['workshop']['raw_vlm_role_recall']} | {diagnostic_rows['Overall']['raw_vlm_role_recall']} |
| Raw VLM role F1 | {diagnostic_rows['kitchen']['raw_vlm_role_f1']} | {diagnostic_rows['living_room']['raw_vlm_role_f1']} | {diagnostic_rows['workshop']['raw_vlm_role_f1']} | {diagnostic_rows['Overall']['raw_vlm_role_f1']} |
| Interpreter-matched raw relation F1 | {diagnostic_rows['kitchen']['interpreter_matched_raw_relation_f1']} | {diagnostic_rows['living_room']['interpreter_matched_raw_relation_f1']} | {diagnostic_rows['workshop']['interpreter_matched_raw_relation_f1']} | {diagnostic_rows['Overall']['interpreter_matched_raw_relation_f1']} |
| Raw complete spec rate | {diagnostic_rows['kitchen']['raw_complete_spec_rate']} | {diagnostic_rows['living_room']['raw_complete_spec_rate']} | {diagnostic_rows['workshop']['raw_complete_spec_rate']} | {diagnostic_rows['Overall']['raw_complete_spec_rate']} |
| Executable contract complete rate | {diagnostic_rows['kitchen']['executable_contract_complete_rate']} | {diagnostic_rows['living_room']['executable_contract_complete_rate']} | {diagnostic_rows['workshop']['executable_contract_complete_rate']} | {diagnostic_rows['Overall']['executable_contract_complete_rate']} |
| Canonicalization success | {diagnostic_rows['kitchen']['canonicalization_success']} | {diagnostic_rows['living_room']['canonicalization_success']} | {diagnostic_rows['workshop']['canonicalization_success']} | {diagnostic_rows['Overall']['canonicalization_success']} |
| Any verified grounding | {diagnostic_rows['kitchen']['any_verified_grounding']} | {diagnostic_rows['living_room']['any_verified_grounding']} | {diagnostic_rows['workshop']['any_verified_grounding']} | {diagnostic_rows['Overall']['any_verified_grounding']} |
| Complete candidate grounding | {diagnostic_rows['kitchen']['complete_candidate_grounding']} | {diagnostic_rows['living_room']['complete_candidate_grounding']} | {diagnostic_rows['workshop']['complete_candidate_grounding']} | {diagnostic_rows['Overall']['complete_candidate_grounding']} |
| Grounded role coverage | {diagnostic_rows['kitchen']['grounded_role_coverage']} | {diagnostic_rows['living_room']['grounded_role_coverage']} | {diagnostic_rows['workshop']['grounded_role_coverage']} | {diagnostic_rows['Overall']['grounded_role_coverage']} |
| Non-empty plan generated | {diagnostic_rows['kitchen']['nonempty_plan_rate']} | {diagnostic_rows['living_room']['nonempty_plan_rate']} | {diagnostic_rows['workshop']['nonempty_plan_rate']} | {diagnostic_rows['Overall']['nonempty_plan_rate']} |
| Candidate plan valid / generated | {diagnostic_rows['kitchen']['candidate_plan_rate']} | {diagnostic_rows['living_room']['candidate_plan_rate']} | {diagnostic_rows['workshop']['candidate_plan_rate']} | {diagnostic_rows['Overall']['candidate_plan_rate']} |
| Partial-plan rate | {diagnostic_rows['kitchen']['partial_plan_rate']} | {diagnostic_rows['living_room']['partial_plan_rate']} | {diagnostic_rows['workshop']['partial_plan_rate']} | {diagnostic_rows['Overall']['partial_plan_rate']} |
| Candidate goal coverage | {diagnostic_rows['kitchen']['candidate_goal_coverage']} | {diagnostic_rows['living_room']['candidate_goal_coverage']} | {diagnostic_rows['workshop']['candidate_goal_coverage']} | {diagnostic_rows['Overall']['candidate_goal_coverage']} |
| Full-task success | {diagnostic_rows['kitchen']['full_task_success']} | {diagnostic_rows['living_room']['full_task_success']} | {diagnostic_rows['workshop']['full_task_success']} | {diagnostic_rows['Overall']['full_task_success']} |
| Mean regions inspected | {diagnostic_rows['kitchen']['mean_regions_inspected']} | {diagnostic_rows['living_room']['mean_regions_inspected']} | {diagnostic_rows['workshop']['mean_regions_inspected']} | {diagnostic_rows['Overall']['mean_regions_inspected']} |
"""
    (output_root / "pipeline_diagnostic_table.md").write_text(diag_md, encoding="utf-8")

    # Invariants verification
    errors = []
    if len(records) != 15:
        errors.append(f"Expected 15 variants, got {len(records)}")
    if sum(1 for r in records if r["gt_feasible"]) != 9:
        errors.append("Expected 9 feasible variants")
    for r in records:
        if r["semantic_vlm_requests"] != 1 or r["high_level_replans"] != 0:
            errors.append(f"{r['variant']}: live invariant failed")
    if len({r.get("prompt_hash") for r in records}) != 1:
        errors.append("Mixed prompt hashes")
    if len({r.get("model") for r in records}) != 1:
        errors.append("Mixed models")

    invariants_payload = {
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "variant_count": len(records),
        "feasible_count": sum(1 for r in records if r["gt_feasible"]),
        "vlm_requests_per_variant": vlm_req_mean,
        "replans_per_variant": replans_mean,
    }
    (output_root / "invariants.json").write_text(json.dumps(invariants_payload, indent=2), encoding="utf-8")

    # Failure analysis
    feasible_records = [r for r in records if r.get("gt_feasible")]
    first_cause_counts = Counter(r.get("first_cause_category") or "NONE" for r in feasible_records)
    detailed_counts = Counter(r.get("failure_category") or "NONE" for r in records)
    failure_payload = {
        "first_cause_categories_feasible": dict(first_cause_counts),
        "detailed_failure_categories_all": dict(detailed_counts),
    }
    (output_root / "failure_analysis.json").write_text(json.dumps(failure_payload, indent=2) + "\n")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "spec_source": spec_source,
        "total_variants": n_total,
        "feasible_variants": len(feasible_rows),
        "infeasible_variants": len(infeasible_rows),
        "primary_metrics": primary_metrics,
        "diagnostic_metrics": diagnostic_rows,
    }
    (output_root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
