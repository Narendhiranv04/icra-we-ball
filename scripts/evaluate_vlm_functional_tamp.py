#!/usr/bin/env python3
"""Final 32-Variant Benchmark Evaluator for VLM-guided Functional-TAMP.

Produces:
1. Per-variant Section 38 structured records (evaluation_records.json and .csv)
2. Section 39 Main Paper Table (Outcome Correct, Feasible Success, Feasibility Recovery,
   Goal Coverage, False Completion, VLM Requests, Replans)
3. Section 40 Pipeline Diagnostic Table (Raw VLM role recall, Runtime contract coverage,
   Canonicalization success, Candidate grounding success, Candidate plan rate,
   Partial plan rate, Candidate goal coverage, Full-task success, Mean regions inspected)
4. Full mathematical metric verification and provenance tracking.
"""

import argparse
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
from mujoco_scenes.symbolic_planning_core import independent_replay

# Authoritative catalogue definitions
DOMAINS = {
    "kitchen": [f"K{i}" for i in range(1, 13)],
    "living_room": [f"L{i}" for i in range(1, 11)],
    "workshop": [f"W{i}" for i in range(1, 11)],
}

FEASIBLE_VARIANTS = {
    "kitchen": [f"K{i}" for i in range(1, 7)],
    "living_room": [f"L{i}" for i in range(1, 7)],
    "workshop": [f"W{i}" for i in range(1, 9)],
}

# Offline recovery set V_recovery:
# GT feasible AND initial observation alone has no complete valid grounding
# AND valid grounding exists after inspecting additional region(s)
RECOVERY_VARIANTS = {
    "kitchen": ["K2", "K3", "K4", "K5", "K6"],  # K1 is all visible on countertop
    "living_room": [],  # All living room regions/objects are in open scene
    "workshop": [f"W{i}" for i in range(1, 9)],  # All workshop tools/fasteners are in closed storage
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
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import full_task_coverage
    return full_task_coverage(domain, run_dir)


def evaluate_all_variants(
    *,
    mode: str = "vlm",
    spec_source: str = "live",
    output_root: Path,
    specification_root: Optional[Path] = None,
    dry_run: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    if spec_source == "live" and not resume and output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Refusing to overwrite an existing live matrix: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    if spec_source == "live":
        if specification_root is not None:
            raise ValueError("specification_root is forbidden in live mode (--spec-source live).")
    elif spec_source in ("replay", "raw-replay"):
        if specification_root is None:
            raise ValueError(f"specification_root is required in {spec_source} mode (--spec-source {spec_source}).")
    else:
        raise ValueError(f"Invalid spec_source: {spec_source!r}. Must be 'live', 'replay', or 'raw-replay'.")

    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import candidate_plan_valid, format_rate
    records: List[Dict[str, Any]] = []
    if resume:
        records_path = output_root / "evaluation_records.json"
        if not records_path.exists():
            raise ValueError(f"Cannot resume without {records_path}")
        records = json.loads(records_path.read_text(encoding="utf-8"))
        if any(r.get("terminal_status") == "PIPELINE_EXCEPTION" for r in records):
            raise ValueError("Remove/archive the trailing failed record before resuming")
    completed = {(r["domain"], r["variant"]) for r in records}

    # Total variant counts
    total_variants = sum(len(v) for v in DOMAINS.values())
    total_feasible = sum(len(v) for v in FEASIBLE_VARIANTS.values())
    total_infeasible = total_variants - total_feasible
    assert total_feasible == 20, f"Expected 20 feasible variants, got {total_feasible}"
    print(f"Evaluating {total_variants} total variants ({total_feasible} feasible, {total_infeasible} infeasible, spec_source={spec_source})...")

    for domain, variants in DOMAINS.items():
        print(f"\n=== Running Domain: {domain.upper()} ({len(variants)} variants) ===")
        for variant in variants:
            if (domain, variant) in completed:
                print(f"  [{variant}] preserved from prior segment")
                continue
            is_feasible = variant in FEASIBLE_VARIANTS[domain]
            is_recovery = variant in RECOVERY_VARIANTS[domain]

            spec_json = None
            if spec_source == "replay":
                cand = specification_root / domain / variant / mode / "functional_specification.json"
                if not cand.exists():
                    raise FileNotFoundError(f"Missing specification for replay variant {domain}/{variant}: {cand}")
                spec_json = cand
            elif spec_source == "raw-replay":
                cand = specification_root / domain / variant / mode / "fm_diagnostics" / "fm_call_001.json"
                if not cand.exists():
                    cand = specification_root / domain / variant / mode / "raw_vlm_response.json"
                if not cand.exists():
                    raise FileNotFoundError(f"Missing raw FM response for variant {domain}/{variant}: {cand}")
                spec_json = cand
            elif spec_source == "live":
                spec_json = None

            t0 = time.perf_counter()
            try:
                result = run_pipeline(
                    domain=domain,
                    variant=variant,
                    mode=mode,
                    specification_json=spec_json,
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

            # Load graph and run offline reference evaluation
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

            # Calculate genuine task goal satisfaction against GT
            candidate_plan = pipeline_res.candidate_plan or ()
            full_gt_goals, sat_gt_goals, full_task_cov, full_task_sat = _evaluate_plan_against_gt(
                domain, variant, candidate_plan, run_dir
            )

            # ACTION_SEQUENCE_READY describes candidate planning, not a claim
            # that the full user task was accomplished.
            false_completion = bool(not is_feasible and full_task_sat)

            # Candidate goal satisfaction
            cand_stats = pipeline_res.candidate_search_statistics or pipeline_res.search_statistics or {}
            cand_total = cand_stats.get("total_goals", 0)
            cand_sat = cand_stats.get("satisfied_goals", 0)
            cand_cov = (cand_sat / cand_total) if cand_total > 0 else 0.0

            # Correctness determination (Section U)
            if is_feasible:
                outcome_correct = bool(full_task_sat)
            else:
                # Legitimate infeasibility conclusion reached after runtime reasoning/search
                outcome_correct = bool(
                    not full_task_sat
                    and not false_completion
                    and pipeline_res.status in {
                        "INFEASIBLE",
                        "EXHAUSTED_NO_VALID_GROUNDING",
                        "NO_VALID_COMPLETE_ASSIGNMENT",
                        "PLANNING_PROVEN_INFEASIBLE",
                    }
                )

            # Telemetry extraction from run manifest
            manifest = {}
            manifest_file = run_dir / "run_manifest.json"
            if manifest_file.exists():
                try:
                    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            spec_acq = manifest.get("spec_acquisition", "live_provider" if spec_json is None else "replayed_provider_output")
            spec_inp = manifest.get("specification_input")
            sem_vlm_req = manifest.get("semantic_vlm_requests", 1 if (mode == "vlm" and spec_acq == "live_provider") else 0)
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

            row = {
                "domain": domain,
                "variant": variant,
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
                "terminal_status": pipeline_res.status,
                "failure_reason": pipeline_res.failure_reason,
            }
            from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import enrich_record
            row = enrich_record(row, run_dir, CANONICAL_TASK_INSTRUCTIONS[domain])
            records.append(row)
            (output_root / "evaluation_records.json").write_text(json.dumps(records, indent=2) + "\n")
            if spec_source == "live" and (pipeline_res.status == "PIPELINE_EXCEPTION" or row["high_level_replans"] > 0):
                (output_root / "invariants.json").write_text(json.dumps({"status": "INVALID", "errors": [f"{variant}: implementation failure; matrix stopped"]}, indent=2))
                raise RuntimeError(f"INVALID MATRIX: {variant}: {pipeline_res.failure_reason}")
            print(f"  [{variant}] outcome_correct={row['outcome_correct']} status={pipeline_res.status} full_task_sat={full_task_sat} cand_plan_len={len(candidate_plan)} runtime={runtime_sec:.2f}s")

    # Save Section 38 records
    rec_json = output_root / "evaluation_records.json"
    rec_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    rec_csv = output_root / "evaluation_records.csv"
    if records:
        with open(rec_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    # Compute Primary 7 Metrics
    n_total = len(records)
    feasible_rows = [r for r in records if r["gt_feasible"]]
    infeasible_rows = [r for r in records if not r["gt_feasible"]]
    recovery_rows = [r for r in records if r["requires_observation_recovery"]]

    # Section AJ Invariants check:
    assert len(records) == 32, f"Expected 32 total variants, got {len(records)}"
    assert len(feasible_rows) == 20, f"Expected 20 feasible variants, got {len(feasible_rows)}"
    assert len(infeasible_rows) == 12, f"Expected 12 infeasible variants, got {len(infeasible_rows)}"
    assert len(recovery_rows) == 13, f"Expected 13 recovery variants, got {len(recovery_rows)}"

    if spec_source == "live":
        for r in records:
            assert r["spec_acquisition"] == "live_provider", f"Variant {r['domain']}/{r['variant']} spec_acquisition={r['spec_acquisition']} != 'live_provider'"
            assert r["specification_input"] is None, f"Variant {r['domain']}/{r['variant']} specification_input={r['specification_input']} is not None"
            assert r["semantic_vlm_requests"] == 1, f"Variant {r['domain']}/{r['variant']} semantic_vlm_requests={r['semantic_vlm_requests']} != 1"
            assert r["high_level_replans"] == 0, f"Variant {r['domain']}/{r['variant']} high_level_replans={r['high_level_replans']} != 0"
    elif spec_source == "raw-replay":
        for r in records:
            assert r["spec_acquisition"] == "archived_raw_provider_response", f"Variant {r['domain']}/{r['variant']} spec_acquisition={r['spec_acquisition']} != 'archived_raw_provider_response'"
            assert r["semantic_vlm_requests"] == 0, f"Variant {r['domain']}/{r['variant']} semantic_vlm_requests={r['semantic_vlm_requests']} != 0"
            assert r["high_level_replans"] == 0, f"Variant {r['domain']}/{r['variant']} high_level_replans={r['high_level_replans']} != 0"

    n_correct = sum(1 for r in records if r["outcome_correct"])
    outcome_correct_pct = (100.0 * n_correct / n_total) if n_total > 0 else 0.0

    n_feasible_success = sum(1 for r in feasible_rows if r["full_task_satisfied"])
    feasible_success_pct = (100.0 * n_feasible_success / len(feasible_rows)) if feasible_rows else 0.0

    n_recovery_success = sum(1 for r in recovery_rows if r["full_task_satisfied"])
    feasibility_recovery_pct = (100.0 * n_recovery_success / len(recovery_rows)) if recovery_rows else 0.0

    mean_goal_cov = (100.0 * sum(r["full_task_goal_coverage"] for r in feasible_rows) / len(feasible_rows)) if feasible_rows else 0.0

    n_false_comp = sum(1 for r in infeasible_rows if r["false_completion"])
    false_comp_pct = (100.0 * n_false_comp / len(infeasible_rows)) if infeasible_rows else 0.0

    mean_vlm_req = sum(r["semantic_vlm_requests"] for r in records) / n_total if n_total > 0 else 0.0
    mean_replans = sum(r["high_level_replans"] for r in records) / n_total if n_total > 0 else 0.0

    primary_metrics = {
        "outcome_correct": outcome_correct_pct,
        "feasible_success": feasible_success_pct,
        "feasibility_recovery": feasibility_recovery_pct,
        "goal_coverage": mean_goal_cov,
        "false_completion": false_comp_pct,
        "vlm_requests": mean_vlm_req,
        "high_level_replans": mean_replans,
    }

    # Generate Section 39 Main Paper Table
    table_md = f"""# Section 39: Main Paper Table

| Method | Outcome Correct ↑ | Feasible-task Success ↑ | Feasibility Recovery ↑ | Goal Coverage ↑ | False Completion ↓ | VLM Requests ↓ | Replans ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ours (FM-Grounding)** | **{outcome_correct_pct:.1f}%** | **{feasible_success_pct:.1f}%** | **{feasibility_recovery_pct:.1f}%** | **{mean_goal_cov:.1f}%** | **{false_comp_pct:.1f}%** | **{mean_vlm_req:.2f}** | **{mean_replans:.2f}** |
"""
    (output_root / "main_paper_table.md").write_text(table_md, encoding="utf-8")

    # Diagnostic Metrics per domain
    diagnostic_rows = {}
    for d in DOMAINS.keys():
        d_recs = [r for r in records if r["domain"] == d]
        d_feas = [r for r in d_recs if r["gt_feasible"]]
        d_total = len(d_recs)

        raw_recalls = [r["raw_role_recall"] for r in d_recs if r.get("raw_role_recall") is not None]
        raw_role_rec = (sum(raw_recalls) / len(raw_recalls)) if raw_recalls else 0.0
        raw_f1s = [r["raw_role_f1"] for r in d_recs if r.get("raw_role_f1") is not None]
        raw_role_f1 = (sum(raw_f1s) / len(raw_f1s)) if raw_f1s else 0.0
        raw_spec_comp = sum(r.get("raw_vlm_spec_complete", False) for r in d_recs) / d_total if d_total else 0.0

        runtime_cov = sum(r["runtime_contract_complete"] for r in d_recs) / d_total if d_total else 0.0
        canon_succ = sum(r["canonicalization_succeeded"] for r in d_recs) / d_total if d_total else 0.0

        eligible_cands = [r for r in d_recs if r["candidate_grounding_eligible"]]
        any_ground = (sum(r["candidate_grounding_succeeded"] for r in eligible_cands) / len(eligible_cands)) if eligible_cands else None
        comp_ground = (sum(r.get("complete_candidate_grounding", False) for r in eligible_cands) / len(eligible_cands)) if eligible_cands else None
        role_cov_ground = (sum(r.get("grounded_expressed_role_coverage", 0.0) for r in eligible_cands) / len(eligible_cands)) if eligible_cands else None

        nonempty_plans = sum(r.get("candidate_plan_length", 0) > 0 for r in d_recs) / d_total if d_total else 0.0
        plan_eligible = [r for r in d_recs if r.get("candidate_plan_length", 0) > 0]
        cand_plan_rate = (sum(r["candidate_plan_valid"] for r in plan_eligible) / len(plan_eligible)) if plan_eligible else None
        partial_rate = sum(1.0 if "PARTIAL" in r["candidate_plan_status"] else 0.0 for r in d_recs) / d_total if d_total else 0.0
        cand_cov = sum(r["candidate_goal_coverage"] for r in d_recs) / d_total if d_total else 0.0
        full_succ = (sum(r["full_task_satisfied"] for r in d_feas) / len(d_feas)) if d_feas else 0.0
        mean_regions = sum(len(r["regions_inspected"]) for r in d_recs) / d_total if d_total else 0.0

        diagnostic_rows[d] = {
            "raw_vlm_role_recall": f"{raw_role_rec * 100:.1f}%",
            "raw_vlm_role_f1": f"{raw_role_f1 * 100:.1f}%",
            "raw_complete_spec_rate": f"{raw_spec_comp * 100:.1f}%",
            "runtime_contract_coverage": f"{runtime_cov * 100:.1f}%",
            "canonicalization_success": f"{canon_succ * 100:.1f}%",
            "any_verified_grounding": format_rate(any_ground),
            "complete_candidate_grounding": format_rate(comp_ground),
            "grounded_role_coverage": format_rate(role_cov_ground),
            "nonempty_plan_rate": f"{nonempty_plans * 100:.1f}%",
            "candidate_plan_rate": format_rate(cand_plan_rate),
            "partial_plan_rate": f"{partial_rate * 100:.1f}%",
            "candidate_goal_coverage": f"{cand_cov * 100:.1f}%",
            "full_task_success": f"{full_succ * 100:.1f}%",
            "mean_regions_inspected": f"{mean_regions:.2f}",
        }

    # Overall Diagnostic row
    all_raw_recalls = [r["raw_role_recall"] for r in records if r.get("raw_role_recall") is not None]
    total_raw_role_rec = (sum(all_raw_recalls) / len(all_raw_recalls)) if all_raw_recalls else 0.0
    all_raw_f1s = [r["raw_role_f1"] for r in records if r.get("raw_role_f1") is not None]
    total_raw_role_f1 = (sum(all_raw_f1s) / len(all_raw_f1s)) if all_raw_f1s else 0.0
    total_raw_spec_comp = sum(r.get("raw_vlm_spec_complete", False) for r in records) / n_total

    total_runtime_cov = sum(r["runtime_contract_complete"] for r in records) / n_total
    total_canon = sum(r["canonicalization_succeeded"] for r in records) / n_total

    all_eligible = [r for r in records if r["candidate_grounding_eligible"]]
    total_any_ground = (sum(r["candidate_grounding_succeeded"] for r in all_eligible) / len(all_eligible)) if all_eligible else None
    total_comp_ground = (sum(r.get("complete_candidate_grounding", False) for r in all_eligible) / len(all_eligible)) if all_eligible else None
    total_role_cov_ground = (sum(r.get("grounded_expressed_role_coverage", 0.0) for r in all_eligible) / len(all_eligible)) if all_eligible else None

    total_nonempty_plans = sum(r.get("candidate_plan_length", 0) > 0 for r in records) / n_total
    all_plan_eligible = [r for r in records if r.get("candidate_plan_length", 0) > 0]
    total_cand_plan = (sum(r["candidate_plan_valid"] for r in all_plan_eligible) / len(all_plan_eligible)) if all_plan_eligible else None
    total_partial = sum(1.0 if "PARTIAL" in r["candidate_plan_status"] else 0.0 for r in records) / n_total
    total_cand_cov = sum(r["candidate_goal_coverage"] for r in records) / n_total
    total_mean_reg = sum(len(r["regions_inspected"]) for r in records) / n_total

    diagnostic_rows["Overall"] = {
        "raw_vlm_role_recall": f"{total_raw_role_rec * 100:.1f}%",
        "raw_vlm_role_f1": f"{total_raw_role_f1 * 100:.1f}%",
        "raw_complete_spec_rate": f"{total_raw_spec_comp * 100:.1f}%",
        "runtime_contract_coverage": f"{total_runtime_cov * 100:.1f}%",
        "canonicalization_success": f"{total_canon * 100:.1f}%",
        "any_verified_grounding": format_rate(total_any_ground),
        "complete_candidate_grounding": format_rate(total_comp_ground),
        "grounded_role_coverage": format_rate(total_role_cov_ground),
        "nonempty_plan_rate": f"{total_nonempty_plans * 100:.1f}%",
        "candidate_plan_rate": format_rate(total_cand_plan),
        "partial_plan_rate": f"{total_partial * 100:.1f}%",
        "candidate_goal_coverage": f"{total_cand_cov * 100:.1f}%",
        "full_task_success": f"{feasible_success_pct:.1f}%",
        "mean_regions_inspected": f"{total_mean_reg:.2f}",
    }

    diag_md = f"""# Section 40: Pipeline Diagnostic Table

| Metric | Kitchen | Living Room | Workshop | Overall |
| :--- | ---: | ---: | ---: | ---: |
| Raw VLM role recall | {diagnostic_rows['kitchen']['raw_vlm_role_recall']} | {diagnostic_rows['living_room']['raw_vlm_role_recall']} | {diagnostic_rows['workshop']['raw_vlm_role_recall']} | {diagnostic_rows['Overall']['raw_vlm_role_recall']} |
| Raw VLM role F1 | {diagnostic_rows['kitchen']['raw_vlm_role_f1']} | {diagnostic_rows['living_room']['raw_vlm_role_f1']} | {diagnostic_rows['workshop']['raw_vlm_role_f1']} | {diagnostic_rows['Overall']['raw_vlm_role_f1']} |
| Raw complete spec rate | {diagnostic_rows['kitchen']['raw_complete_spec_rate']} | {diagnostic_rows['living_room']['raw_complete_spec_rate']} | {diagnostic_rows['workshop']['raw_complete_spec_rate']} | {diagnostic_rows['Overall']['raw_complete_spec_rate']} |
| Runtime contract coverage | {diagnostic_rows['kitchen']['runtime_contract_coverage']} | {diagnostic_rows['living_room']['runtime_contract_coverage']} | {diagnostic_rows['workshop']['runtime_contract_coverage']} | {diagnostic_rows['Overall']['runtime_contract_coverage']} |
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
    print("\n=== EVALUATION COMPLETE ===")
    print(table_md)
    print(diag_md)
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import write_detailed_report
    errors = write_detailed_report(output_root, records, live=spec_source == "live" and not resume)
    if errors:
        raise RuntimeError("INVALID BENCHMARK: " + "; ".join(errors))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("vlm", "gt"), default="vlm")
    parser.add_argument("--spec-source", choices=("live", "replay", "raw-replay"), default="live",
                        help="Specification sourcing mode: 'live' (fresh FM call per variant), 'replay' (canonical graph JSON), or 'raw-replay' (archived raw FM response)")
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_reports/final_vlm_evaluation"))
    parser.add_argument("--specification-root", type=Path, default=None,
                        help="Path to root containing per-variant specifications (required for --spec-source replay or raw-replay, forbidden for live)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true",
                        help="Preserve completed records and run only missing variants in an existing output root")
    args = parser.parse_args()

    evaluate_all_variants(
        mode=args.mode,
        spec_source=args.spec_source,
        output_root=args.output_root,
        specification_root=args.specification_root,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
