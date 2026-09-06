"""Paper-readiness audit, metrics extraction, aggregation, and LaTeX generation for ViLaIn-TAMP."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from collections import Counter, defaultdict

from .benchmark_harness import (
    CONFIG_FILES,
    authoritative_variants,
    authoritative_feasibility,
    authoritative_requirements_count,
)
from .evaluation import (
    CANONICAL_REQUIREMENT_NAMES,
    CANONICAL_SUBGOAL_COUNTS,
    SubgoalCoverageEvaluation,
    canonical_requirements_count,
    canonical_subgoal_count,
    canonical_terminal_subgoals,
    evaluate_terminal_subgoals,
    initial_snapshot_from_config,
)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute 95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% confidence z-score
    p_hat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p_hat + z2 / (2.0 * total)) / denominator
    spread = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))
    low = 0.0 if successes == 0 else max(0.0, centre - spread)
    high = 1.0 if successes == total else min(1.0, centre + spread)
    return low, high


def continuous_stats(values: Sequence[float | int | None]) -> dict[str, Any]:
    """Compute mean, std, median, q25, q75, IQR for continuous measurements."""
    valid = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not valid:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    n = len(valid)
    mean_val = sum(valid) / n
    variance = sum((x - mean_val) ** 2 for x in valid) / n if n > 0 else 0.0
    std_val = math.sqrt(variance)

    sorted_v = sorted(valid)
    def percentile(p: float) -> float:
        idx = p * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_v[lower]
        return sorted_v[lower] * (upper - idx) + sorted_v[upper] * (idx - lower)

    median_val = percentile(0.5)
    q25 = percentile(0.25)
    q75 = percentile(0.75)
    iqr = q75 - q25
    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "min": sorted_v[0],
        "max": sorted_v[-1],
    }


def extract_run_attempts(artifacts_root: Path, run_id: str) -> list[dict[str, Any]]:
    """Extract all plan attempts and their artifacts for a run."""
    attempts_dir = artifacts_root / "attempts"
    if not attempts_dir.exists():
        return []

    records = []
    for att_path in sorted(attempts_dir.glob("*")):
        if not att_path.is_dir():
            continue
        try:
            att_idx = int(att_path.name)
        except ValueError:
            att_idx = -1

        plan_json = att_path / "planner" / "symbolic_plan.json"
        val_json = att_path / "planner" / "plan_validation.json"
        id_json = att_path / "identity" / "identity_resolution.json"
        ref_json = att_path / "refinement.json"
        ao_json = att_path / "attempt_outcome.json"

        has_plan = plan_json.exists()
        actions: list[dict[str, Any]] = []
        plan_sha = None
        plan_cost = None
        if has_plan:
            try:
                with open(plan_json, encoding="utf-8") as pf:
                    pdata = json.load(pf)
                    actions = pdata.get("actions", [])
                    plan_sha = pdata.get("plan_sha256")
                    plan_cost = pdata.get("plan_cost")
            except Exception:
                pass

        val_valid = False
        if val_json.exists():
            try:
                with open(val_json, encoding="utf-8") as vf:
                    val_valid = bool(json.load(vf).get("valid", False))
            except Exception:
                pass

        id_attempted = has_plan or id_json.exists()
        id_success = False
        ref_attempted = False
        ref_success = False
        outcome_success = False
        fail_kind = None
        fail_summary = None

        if ao_json.exists():
            try:
                with open(ao_json, encoding="utf-8") as aof:
                    aodata = json.load(aof)
                    outcome_success = bool(aodata.get("success", False))
                    fail = aodata.get("failure") or {}
                    fail_kind = fail.get("kind")
                    fail_summary = fail.get("summary")
                    if outcome_success or fail_kind in {"REFINEMENT", "EXECUTION"}:
                        id_success = True
                    if outcome_success or fail_kind == "EXECUTION":
                        ref_success = True
                    if id_success:
                        ref_attempted = True
            except Exception:
                pass
        elif ref_json.exists():
            try:
                with open(ref_json, encoding="utf-8") as rf:
                    rdata = json.load(rf)
                    ref_success = bool(rdata.get("success", False) or rdata.get("status") == "SUCCESS")
                    id_success = True
                    ref_attempted = True
            except Exception:
                pass

        att_nonempty_fd = len(actions) > 0 if has_plan else False
        att_nonempty_val = bool(att_nonempty_fd and val_valid)
        att_nonempty_identity = bool(att_nonempty_val and id_success)
        att_nonempty_refine = bool(att_nonempty_identity and ref_success)

        records.append({
            "run_id": run_id,
            "attempt_index": att_idx,
            "problem_sha256": plan_sha,
            "plan_sha256": plan_sha,
            "plan_cost": plan_cost,
            "plan_artifact": str(plan_json.resolve()) if has_plan else None,
            "actions": actions,
            "plan_length": len(actions) if has_plan else None,
            "plan_found": has_plan,
            "nonempty_plan": att_nonempty_fd,
            "val_valid": val_valid,
            "identity_attempted": id_attempted,
            "identity_success": id_success,
            "refinement_attempted": ref_attempted,
            "refinement_success": ref_success,
            "attempt_success": outcome_success,
            "attempt_nonempty_fd": att_nonempty_fd,
            "attempt_nonempty_val": att_nonempty_val,
            "attempt_nonempty_identity": att_nonempty_identity,
            "attempt_nonempty_refine": att_nonempty_refine,
            "failure_kind": fail_kind,
            "failure_summary": fail_summary,
        })
    return records


def audit_run(run_dir: Path, config_root: Path | None = None) -> dict[str, Any]:
    """Audit one run directory completely and extract all paper metrics."""
    tf_path = run_dir / "terminal_status.json"
    if not tf_path.is_file():
        raise FileNotFoundError(f"Missing terminal_status.json in {run_dir}")

    with open(tf_path, encoding="utf-8") as tf:
        tdata = json.load(tf)

    run_id = tdata.get("run_id", run_dir.name)
    domain = tdata.get("domain")
    variant = tdata.get("variant")
    protocol = tdata.get("observation_protocol")
    repeat = tdata.get("repeat_index")
    seed = tdata.get("seed")
    source_commit = tdata.get("source_commit")
    raw_status = tdata.get("terminal_status") or tdata.get("status")
    infra_failure = bool(tdata.get("infrastructure_failure", False) or raw_status == "INFRASTRUCTURE_FAILURE")

    art_dir = run_dir / "artifacts"
    bresult = tdata.get("baseline_result") or {}
    metrics = bresult.get("metrics") or tdata.get("metrics") or {}

    # 1. Observation
    obs_manifest = art_dir / "observations" / "observation_manifest.json"
    observation_success = obs_manifest.is_file()

    # 2. Object estimation & PDDL generation
    interp_gen = art_dir / "interpreter" / "generation_artifacts.json"
    has_obj = False
    has_init = False
    has_goal = False
    if interp_gen.is_file():
        try:
            with open(interp_gen, encoding="utf-8") as f:
                gdata = json.load(f)
                has_obj = bool(gdata.get("object_response_artifact"))
                has_init = bool(gdata.get("initial_fragment_artifact"))
                has_goal = bool(gdata.get("goal_fragment_artifact"))
        except Exception:
            pass

    pddl_init = art_dir / "interpreter" / "problem_initial.pddl"
    pddl_valid = pddl_init.is_file()

    # Attempts
    attempts = extract_run_attempts(art_dir, run_id)
    fd_invoked = pddl_valid and (len(attempts) > 0 or raw_status != "FM_OBJECT_FAILURE")

    plans = [a for a in attempts if a["plan_found"]]
    nonempty_plans = [a for a in attempts if a.get("attempt_nonempty_fd", a.get("nonempty_plan", False))]
    val_valid_plans = [a for a in attempts if a["val_valid"]]

    first_plan_idx = plans[0]["attempt_index"] if plans else None
    first_nonempty_plan_idx = nonempty_plans[0]["attempt_index"] if nonempty_plans else None
    any_plan = len(plans) > 0
    any_nonempty_plan = len(nonempty_plans) > 0
    any_val_valid = len(val_valid_plans) > 0

    # Strict attempt-level progression indicators
    any_plan_fd = any_plan
    any_plan_val = any_val_valid
    nonempty_plan_fd = any_nonempty_plan
    nonempty_plan_val = any(a.get("attempt_nonempty_val", False) for a in attempts)
    nonempty_plan_identity = any(a.get("attempt_nonempty_identity", False) for a in attempts)
    nonempty_plan_refine = any(a.get("attempt_nonempty_refine", False) for a in attempts)

    best_plan_len = None
    if nonempty_plans:
        best_plan_len = min(a["plan_length"] for a in nonempty_plans if a["plan_length"] is not None)
    elif plans:
        best_plan_len = plans[0]["plan_length"]

    final_selected_plan_len = None
    final_plan = art_dir / "final_action_plan.json"
    if final_plan.is_file():
        try:
            with open(final_plan, encoding="utf-8") as f:
                final_selected_plan_len = len(json.load(f).get("actions", []))
        except Exception:
            pass
    elif bresult.get("selected_attempt_index") is not None:
        sel_idx = bresult.get("selected_attempt_index")
        for a in attempts:
            if a["attempt_index"] == sel_idx:
                final_selected_plan_len = a["plan_length"]
                break

    # Identity & Refinement
    identity_attempted = any(a["identity_attempted"] for a in attempts)
    identity_success = any(a["identity_success"] for a in attempts)
    refinement_attempted = any(a["refinement_attempted"] for a in attempts)
    refinement_success = any(a["refinement_success"] for a in attempts)
    execution_projection_available = final_plan.is_file() or any((art_dir / f"attempts/{a['attempt_index']:02d}/execution_projections.json").is_file() for a in attempts)

    # Execution
    exec_trace = art_dir / "execution" / "execution_trace.json"
    exec_attempted = exec_trace.is_file()
    exec_success = False
    exec_result_file = art_dir / "execution" / "execution_result.json"
    if exec_result_file.is_file():
        try:
            with open(exec_result_file, encoding="utf-8") as f:
                exec_success = bool(json.load(f).get("success", False))
        except Exception:
            pass

    # Evaluations
    gge_file = art_dir / "benchmark" / "generated_goal_evaluation.json"
    gge_evaluated = gge_file.is_file()
    gge_satisfied = None
    gge_atoms_passed = 0
    gge_atoms_total = 0
    if gge_evaluated:
        try:
            with open(gge_file, encoding="utf-8") as f:
                gdata = json.load(f)
                gge_satisfied = bool(gdata.get("satisfied", False))
                for c in gdata.get("goal_checks", []):
                    gge_atoms_total += 1
                    if c.get("passed") is True:
                        gge_atoms_passed += 1
        except Exception:
            pass

    bme_file = art_dir / "benchmark" / "benchmark_goal_evaluation.json"
    bme_evaluated = bme_file.is_file()
    actual_task_success = False
    gt_feasible = None
    raw_predicted_infeasible = None
    bme_reqs_passed = 0
    bme_reqs_total = 0
    if bme_evaluated:
        try:
            with open(bme_file, encoding="utf-8") as f:
                bdata = json.load(f)
                actual_task_success = bool(bdata.get("actual_task_success", False))
                gt_feasible = bdata.get("ground_truth_feasibility")
                raw_predicted_infeasible = bdata.get("predicted_infeasible")
                for r in bdata.get("requirement_checks", []):
                    bme_reqs_total += 1
                    if r.get("passed") is True:
                        bme_reqs_passed += 1
        except Exception:
            pass

    # Authoritative feasibility and requirement counts
    if config_root is None:
        config_root = Path(__file__).resolve().parents[3] / "mujoco_scenes" / "configs"
    feas_map = authoritative_feasibility(config_root)
    req_counts = authoritative_requirements_count(config_root)
    if gt_feasible is None and domain and variant:
        dom_feas = feas_map.get(str(domain), {})
        if str(variant) in dom_feas:
            gt_feasible = dom_feas[str(variant)]
        else:
            matching_feas = [v for k, v in dom_feas.items() if k == str(variant) or k.startswith(str(variant) + "_")]
            if matching_feas:
                gt_feasible = matching_feas[0]
    if gt_feasible is None and variant:
        gt_feasible = str(variant).startswith("F")

    if not bme_evaluated and domain and variant:
        dom_reqs = req_counts.get(str(domain), {})
        if str(variant) in dom_reqs:
            bme_reqs_total = dom_reqs[str(variant)]
        else:
            matching_reqs = [v for k, v in dom_reqs.items() if k == str(variant) or k.startswith(str(variant) + "_")]
            if matching_reqs:
                bme_reqs_total = matching_reqs[0]
            else:
                try:
                    bme_reqs_total = canonical_requirements_count(str(domain))
                except Exception:
                    bme_reqs_total = 8 if str(domain) == "kitchen" else 6
        bme_reqs_passed = 0

    # Attempt tracking & Provenance (Part 6.3)
    selected_attempt_idx = bresult.get("selected_attempt_index")
    selected_attempt = None
    if selected_attempt_idx is not None:
        for a in attempts:
            if a["attempt_index"] == selected_attempt_idx:
                selected_attempt = a
                break
    if selected_attempt is None and attempts:
        for a in attempts:
            if a.get("attempt_nonempty_refine"):
                selected_attempt = a
                break
        if selected_attempt is None:
            for a in attempts:
                if a["nonempty_plan"]:
                    selected_attempt = a
                    break
        if selected_attempt is None:
            selected_attempt = attempts[0]

    selected_action_seq_str = "[]"
    if final_plan.is_file():
        try:
            with open(final_plan, encoding="utf-8") as f:
                actions = json.load(f).get("actions", [])
                if actions:
                    selected_action_seq_str = " -> ".join(
                        f"{a.get('operator')}({', '.join(a.get('arguments', []))})" for a in actions
                    )
        except Exception:
            pass
    elif selected_attempt and selected_attempt.get("actions"):
        sel_actions = selected_attempt.get("actions", [])
        if sel_actions:
            selected_action_seq_str = " -> ".join(
                f"{a.get('operator')}({', '.join(a.get('arguments', []))})" for a in sel_actions
            )
    elif plans:
        sel_actions = plans[0].get("actions", [])
        if sel_actions:
            selected_action_seq_str = " -> ".join(
                f"{a.get('operator')}({', '.join(a.get('arguments', []))})" for a in sel_actions
            )

    provenance_consistent = True
    provenance_diagnostics: list[str] = []
    if final_plan.is_file():
        try:
            with open(final_plan, encoding="utf-8") as f:
                fp_data = json.load(f)
                fp_att = fp_data.get("selected_attempt_index")
                fp_sha = fp_data.get("plan_sha256") or fp_data.get("selected_plan_sha256")
                if selected_attempt_idx is not None and fp_att is not None and fp_att != selected_attempt_idx:
                    provenance_consistent = False
                    provenance_diagnostics.append(f"attempt mismatch: final_plan has {fp_att}, baseline has {selected_attempt_idx}")
                att_sha = (selected_attempt.get("plan_sha256") or selected_attempt.get("problem_sha256")) if selected_attempt else None
                if selected_attempt and fp_sha and att_sha and fp_sha != att_sha:
                    provenance_consistent = False
                    provenance_diagnostics.append(f"sha mismatch: final_plan {fp_sha} != attempt {att_sha}")
        except Exception as e:
            provenance_consistent = False
            provenance_diagnostics.append(str(e))

    # Terminal causal category (B2B) & CP diagnostic classification (Part 7)
    cp_file = art_dir / "corrective_planning_result.json"
    cp_terminal_kind = None
    cp_terminal_stage = None
    cp_terminal_summary = None
    cp_diagnostics: Sequence[Any] = ()
    cp_classified_label = None
    if cp_file.is_file():
        try:
            with open(cp_file, encoding="utf-8") as f:
                cp_data = json.load(f)
                tfail = cp_data.get("terminal_failure") or {}
                cp_terminal_kind = tfail.get("kind")
                cp_terminal_summary = tfail.get("summary")
                details = tfail.get("details") or {}
                cp_terminal_stage = details.get("stage")
                cp_diagnostics = details.get("diagnostics") or ()
                cp_status = cp_data.get("status")
                sum_lower = (cp_terminal_summary or "").lower()
                diag_str = " ".join(str(d).lower() for d in cp_diagnostics)
                if "repeats" in sum_lower or "repeated" in sum_lower or cp_status == "REPEATED_REVISION":
                    cp_classified_label = "REPEATED_CORRECTION"
                elif "unknown_fact" in diag_str or "unknown fact" in diag_str:
                    cp_classified_label = "UNKNOWN_FACT_ID"
                elif "malformed" in diag_str or "parse" in diag_str or "json" in diag_str:
                    cp_classified_label = "MALFORMED_CORRECTIVE_RESPONSE"
                elif "inconsistent" in diag_str or "contradict" in diag_str:
                    cp_classified_label = "INCONSISTENT_CORRECTIVE_FACT_SET"
                elif "unreachable" in diag_str:
                    cp_classified_label = "UNREACHABLE_CORRECTED_GOAL"
                elif cp_terminal_kind == "INVALID_CORRECTION":
                    cp_classified_label = "INVALID_CORRECTIVE_SELECTION"
        except Exception:
            pass

    if raw_status == "FM_OBJECT_FAILURE" or not pddl_valid:
        causal_category = "UNRESOLVED_FM_FAILURE"
    elif raw_status in {"SUCCESS", "BENCHMARK_FAILURE"}:
        causal_category = "PLAN_FOUND"
    elif raw_status == "REFINEMENT_FAILURE":
        causal_category = "UNRESOLVED_REFINEMENT_FAILURE"
    elif raw_status == "IDENTITY_FAILURE":
        causal_category = "UNRESOLVED_IDENTITY_FAILURE"
    elif cp_terminal_kind == "INVALID_CORRECTION":
        causal_category = f"UNRESOLVED_{cp_classified_label}" if cp_classified_label else "UNRESOLVED_INVALID_CORRECTIVE_SELECTION"
    elif raw_status == "NO_PLAN" or cp_terminal_kind == "NO_PLAN" or cp_terminal_stage == "NO_PLAN":
        causal_category = "SYMBOLIC_NO_PLAN_AFTER_BOUNDED_CP"
    elif cp_terminal_kind == "ENTITY_RESOLUTION":
        causal_category = "UNRESOLVED_IDENTITY_FAILURE"
    elif cp_terminal_kind == "REFINEMENT":
        causal_category = "UNRESOLVED_REFINEMENT_FAILURE"
    elif any_plan:
        causal_category = "PLAN_FOUND"
    else:
        causal_category = "OTHER_UNRESOLVED"

    # Outcome Correct:
    # For GT-feasible: task actually completed successfully
    # For GT-infeasible: clean symbolic rejection (valid object + valid pddl + NO PLAN under bounded CP + no unresolved identity/refinement/runtime failure)
    if gt_feasible:
        outcome_correct = bool(actual_task_success)
    else:
        outcome_correct = bool(
            has_obj
            and pddl_valid
            and (not any_plan)
            and (causal_category == "SYMBOLIC_NO_PLAN_AFTER_BOUNDED_CP")
            and not infra_failure
        )

    # Strict same-attempt physical plan found
    physical_plan_found = nonempty_plan_refine

    execution_stage_completed = bool(exec_success)
    selected_has_nonempty_refine = (
        selected_attempt.get("attempt_nonempty_refine", False) if selected_attempt else nonempty_plan_refine
    )
    nonempty_plan_execution_completed = bool(
        exec_success and (final_selected_plan_len or 0) > 0 and selected_has_nonempty_refine
    )
    nonempty_plan_exec = nonempty_plan_execution_completed
    task_final = bool(nonempty_plan_exec and actual_task_success)

    declared_completion = bool(exec_success or raw_status in {"SUCCESS", "BENCHMARK_FAILURE"})
    false_completion = bool(declared_completion and not actual_task_success)

    model_calls = metrics.get("model_calls_by_type") or {}
    obj_calls = int(model_calls.get("object_estimation", 1 if has_obj else 0))
    init_calls = int(model_calls.get("initial_state", 1 if has_init else 0))
    goal_calls = int(model_calls.get("goal_state", 1 if has_goal else 0))
    cp_calls = metrics.get("cp_calls")
    if cp_calls is None:
        cp_calls = int(model_calls.get("corrective_planning", len(attempts) - 1 if attempts else 0))
    else:
        cp_calls = int(cp_calls)
    high_level_replans = max(0, min(3, cp_calls))

    raw_vlm_requests = metrics.get("model_call_count")
    if raw_vlm_requests is None:
        raw_vlm_requests = obj_calls + init_calls + goal_calls + cp_calls
    else:
        raw_vlm_requests = int(raw_vlm_requests)

    actions_attempted = 0
    actions_succeeded = 0
    if exec_trace.is_file():
        try:
            with open(exec_trace, encoding="utf-8") as f:
                trace_actions = json.load(f).get("controller_actions", [])
                actions_attempted = len(trace_actions)
                actions_succeeded = sum(1 for a in trace_actions if a.get("success") is True)
        except Exception:
            pass
    else:
        actions_attempted = int(metrics.get("controller_action_count", 0))

    # Initial vs Terminal Goal Coverage from canonical GT manipulation subgoals (Part 2, 3, 5)
    term_sge_file = art_dir / "benchmark" / "terminal_subgoal_evaluation.json"
    init_sge_file = art_dir / "benchmark" / "initial_subgoal_evaluation.json"

    terminal_subgoals_total = canonical_subgoal_count(domain) if domain else 0
    terminal_subgoals_passed = 0
    terminal_subgoal_eval = None

    if term_sge_file.is_file():
        try:
            with open(term_sge_file, encoding="utf-8") as f:
                sge_data = json.load(f)
                terminal_subgoals_total = sge_data.get("total_subgoals", terminal_subgoals_total)
                terminal_subgoals_passed = sge_data.get("passed_subgoals", 0)
                terminal_subgoal_eval = sge_data
        except Exception:
            pass
    elif domain and variant:
        try:
            from .production_execution import BenchmarkRegistryHiddenContextProvider
            ctx_provider = BenchmarkRegistryHiddenContextProvider(config_root)
            hidden_ctx = ctx_provider.load(domain, variant)
            init_snap = initial_snapshot_from_config(domain, variant, config_root)
            subgoal_eval = evaluate_terminal_subgoals(init_snap, (), hidden_ctx)
            terminal_subgoals_total = subgoal_eval.total_subgoals
            terminal_subgoals_passed = subgoal_eval.passed_subgoals
            terminal_subgoal_eval = subgoal_eval.to_dict()
        except Exception:
            pass

    initial_subgoals_total = terminal_subgoals_total
    initial_subgoals_passed = 0
    initial_subgoal_eval = None

    if init_sge_file.is_file():
        try:
            with open(init_sge_file, encoding="utf-8") as f:
                init_sge_data = json.load(f)
                initial_subgoals_total = init_sge_data.get("total_subgoals", initial_subgoals_total)
                initial_subgoals_passed = init_sge_data.get("passed_subgoals", 0)
                initial_subgoal_eval = init_sge_data
        except Exception:
            pass
    elif domain and variant:
        try:
            from .production_execution import BenchmarkRegistryHiddenContextProvider
            ctx_provider = BenchmarkRegistryHiddenContextProvider(config_root)
            hidden_ctx = ctx_provider.load(domain, variant)
            init_snap = initial_snapshot_from_config(domain, variant, config_root)
            init_eval = evaluate_terminal_subgoals(init_snap, (), hidden_ctx)
            initial_subgoals_total = init_eval.total_subgoals
            initial_subgoals_passed = init_eval.passed_subgoals
            initial_subgoal_eval = init_eval.to_dict()
        except Exception:
            initial_subgoals_passed = terminal_subgoals_passed if actions_attempted == 0 else 0

    initial_goal_coverage = (
        (initial_subgoals_passed / initial_subgoals_total)
        if (gt_feasible and initial_subgoals_total > 0)
        else None
    )
    terminal_goal_coverage = (
        (terminal_subgoals_passed / terminal_subgoals_total)
        if (gt_feasible and terminal_subgoals_total > 0)
        else None
    )
    delta_goal_coverage = (
        (terminal_goal_coverage - initial_goal_coverage)
        if (gt_feasible and terminal_goal_coverage is not None and initial_goal_coverage is not None)
        else None
    )

    term_fail_cause = ""
    if raw_status != "SUCCESS":
        if cp_classified_label:
            term_fail_cause = cp_classified_label
        elif cp_terminal_summary:
            term_fail_cause = cp_terminal_summary
        else:
            term_fail_cause = cp_terminal_kind or raw_status

    return {
        "run_id": run_id,
        "domain": domain,
        "variant": variant,
        "observation_protocol": protocol,
        "repeat_index": repeat,
        "seed": seed,
        "source_commit": source_commit,
        "raw_terminal_status": raw_status,
        "terminal_status": raw_status,
        "infrastructure_failure": infra_failure,
        "causal_category": causal_category,
        "cp_terminal_kind": cp_terminal_kind,
        "cp_terminal_stage": cp_terminal_stage,
        "cp_terminal_summary": cp_terminal_summary,
        # Funnel booleans
        "observation_success": observation_success,
        "object_estimation_success": has_obj,
        "initial_state_success": has_init,
        "goal_state_success": has_goal,
        "pddl_valid": pddl_valid,
        "fd_invoked": fd_invoked,
        "symbolic_plan_found": any_plan,
        "symbolic_plan_nonempty": any_nonempty_plan,
        "val_plan_valid": any_val_valid,
        "identity_attempted": identity_attempted,
        "identity_success": identity_success,
        "refinement_attempted": refinement_attempted,
        "refinement_success": refinement_success,
        "execution_projection_available": execution_projection_available,
        "execution_attempted": exec_attempted,
        "execution_success": exec_success,
        "execution_stage_completed": execution_stage_completed,
        "nonempty_plan_execution_completed": nonempty_plan_execution_completed,
        "physical_plan_found": physical_plan_found,
        "declared_completion": declared_completion,
        "false_completion": false_completion,
        "outcome_correct": outcome_correct,
        "generated_goal_evaluated": gge_evaluated,
        "hidden_benchmark_evaluated": bme_evaluated,
        # Strict same-attempt sequence funnel booleans
        "any_plan_fd": any_plan_fd,
        "any_plan_val": any_plan_val,
        "nonempty_plan_fd": nonempty_plan_fd,
        "nonempty_plan_val": nonempty_plan_val,
        "nonempty_plan_identity": nonempty_plan_identity,
        "nonempty_plan_refine": nonempty_plan_refine,
        "nonempty_plan_exec": nonempty_plan_exec,
        "task_final": task_final,
        "provenance_consistent": provenance_consistent,
        "provenance_diagnostics": provenance_diagnostics,
        # Action sequence metrics
        "first_plan_attempt_index": first_plan_idx,
        "first_nonempty_plan_attempt_index": first_nonempty_plan_idx,
        "number_of_symbolic_plans_generated": len(plans),
        "number_of_nonempty_symbolic_plans": len(nonempty_plans),
        "number_of_val_valid_plans": len(val_valid_plans),
        "best_symbolic_plan_length": best_plan_len,
        "final_selected_plan_length": final_selected_plan_len,
        "actual_selected_plan_length": final_selected_plan_len or (plans[0]["plan_length"] if plans else 0),
        "actual_selected_action_sequence": selected_action_seq_str,
        # Evaluations
        "actual_task_success": actual_task_success,
        "hidden_task_success": actual_task_success,
        "ground_truth_feasible": gt_feasible,
        "raw_predicted_infeasible": raw_predicted_infeasible,
        "generated_goal_satisfied": gge_satisfied,
        "generated_goal_atoms_passed": gge_atoms_passed,
        "generated_goal_atoms_total": gge_atoms_total,
        "generated_goal_atom_coverage": (gge_atoms_passed / gge_atoms_total) if gge_atoms_total > 0 else None,
        "benchmark_requirements_passed": bme_reqs_passed,
        "goal_requirements_passed": terminal_subgoals_passed,
        "benchmark_requirements_total": bme_reqs_total,
        "goal_requirements_total": terminal_subgoals_total,
        "benchmark_requirement_coverage": (bme_reqs_passed / bme_reqs_total) if bme_reqs_total > 0 else None,
        "goal_coverage": terminal_goal_coverage,
        "initial_requirements_passed": initial_subgoals_passed if gt_feasible else None,
        "initial_requirements_total": initial_subgoals_total if gt_feasible else None,
        "terminal_requirements_passed": terminal_subgoals_passed if gt_feasible else None,
        "terminal_requirements_total": terminal_subgoals_total if gt_feasible else None,
        "initial_goal_coverage": initial_goal_coverage,
        "terminal_goal_coverage": terminal_goal_coverage,
        "delta_goal_coverage": delta_goal_coverage,
        "initial_subgoals_passed": initial_subgoals_passed,
        "initial_subgoals_total": initial_subgoals_total,
        "terminal_subgoals_passed": terminal_subgoals_passed,
        "terminal_subgoals_total": terminal_subgoals_total,
        "initial_subgoal_eval": initial_subgoal_eval,
        "terminal_subgoal_eval": terminal_subgoal_eval,
        "selected_attempt_index": selected_attempt_idx,
        "selected_plan_length": selected_attempt["plan_length"] if selected_attempt else 0,
        "selected_plan_val_valid": selected_attempt["val_valid"] if selected_attempt else False,
        "selected_plan_identity_success": selected_attempt["identity_success"] if selected_attempt else False,
        "selected_plan_refinement_success": selected_attempt["refinement_success"] if selected_attempt else False,
        # Calls and timings
        "fm_calls": raw_vlm_requests,
        "raw_vlm_requests": raw_vlm_requests,
        "object_calls": obj_calls,
        "initial_state_calls": init_calls,
        "init_calls": init_calls,
        "goal_state_calls": goal_calls,
        "goal_calls": goal_calls,
        "cp_calls": cp_calls,
        "CP_calls": cp_calls,
        "high_level_replans": high_level_replans,
        "inspections": metrics.get("inspected_region_count", 0),
        "controller_actions": metrics.get("controller_action_count", 0),
        "physical_actions_attempted": actions_attempted,
        "physical_actions_succeeded": actions_succeeded,
        "terminal_failure_cause": term_fail_cause,
        "runtime_seconds": metrics.get("end_to_end_seconds", tdata.get("elapsed_seconds")),
        "end_to_end_seconds": metrics.get("end_to_end_seconds", tdata.get("elapsed_seconds")),
        "fm_latency_seconds": metrics.get("fm_latency_seconds"),
        "symbolic_planning_seconds": metrics.get("symbolic_planning_seconds"),
        "refinement_seconds": metrics.get("geometric_refinement_seconds"),
        "execution_seconds": metrics.get("execution_seconds"),
        # Artifact paths
        "pddl_path": str((art_dir / "interpreter" / "problem_initial.pddl").resolve()) if (art_dir / "interpreter" / "problem_initial.pddl").is_file() else "",
        "action_sequence_path": str(final_plan.resolve()) if final_plan.is_file() else (str((art_dir / "attempts/00/planner/symbolic_plan.json").resolve()) if (art_dir / "attempts/00/planner/symbolic_plan.json").is_file() else ""),
        "val_result_path": str((art_dir / "attempts/00/planner/plan_validation.json").resolve()) if (art_dir / "attempts/00/planner/plan_validation.json").is_file() else "",
        "identity_path": str((art_dir / "attempts/00/identity/identity_resolution.json").resolve()) if (art_dir / "attempts/00/identity/identity_resolution.json").is_file() else "",
        "refinement_path": str((art_dir / "attempts/00/refinement.json").resolve()) if (art_dir / "attempts/00/refinement.json").is_file() else "",
        "execution_path": str(exec_trace.resolve()) if exec_trace.is_file() else "",
        "terminal_state_path": str((art_dir / "execution" / "terminal_state.json").resolve()) if (art_dir / "execution" / "terminal_state.json").is_file() else "",
        "benchmark_evaluation_path": str(bme_file.resolve()) if bme_file.is_file() else "",
        "attempts": attempts,
    }


def compute_confusion_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute binary classification metrics with Task Infeasible as Positive class."""
    evaluable = [r for r in rows if r.get("ground_truth_feasible") is not None and r.get("raw_predicted_infeasible") is not None]

    tp = sum(1 for r in evaluable if r["ground_truth_feasible"] is False and r["raw_predicted_infeasible"] is True)
    fp = sum(1 for r in evaluable if r["ground_truth_feasible"] is True and r["raw_predicted_infeasible"] is True)
    tn = sum(1 for r in evaluable if r["ground_truth_feasible"] is True and r["raw_predicted_infeasible"] is False)
    fn = sum(1 for r in evaluable if r["ground_truth_feasible"] is False and r["raw_predicted_infeasible"] is False)

    total = len(evaluable)
    positives = tp + fn  # Actual Infeasible
    negatives = tn + fp  # Actual Feasible

    acc = (tp + tn) / total if total > 0 else None
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec = tp / positives if positives > 0 else None  # Infeasible Recall
    spec = tn / negatives if negatives > 0 else None  # Feasible Recall
    bal_acc = (rec + spec) / 2.0 if (rec is not None and spec is not None) else None
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    false_infeas_rate = fp / negatives if negatives > 0 else None
    false_feas_rate = fn / positives if positives > 0 else None
    coverage = total / len(rows) if rows else 0.0

    acc_low, acc_high = wilson_interval(tp + tn, total) if total > 0 else (0.0, 0.0)

    return {
        "evaluated_runs": total,
        "total_scheduled_runs": len(rows),
        "decision_coverage": coverage,
        "actual_infeasible_count": positives,
        "actual_feasible_count": negatives,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": acc,
        "accuracy_wilson_ci95": [acc_low, acc_high],
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "false_infeasible_rate": false_infeas_rate,
        "false_feasible_rate": false_feas_rate,
    }


def compute_group_aggregates(rows: Sequence[Mapping[str, Any]], group_label: str = "all") -> dict[str, Any]:
    """Compute primary paper metrics over a subset of non-infrastructure runs."""
    n_total = len(rows)
    feasible_rows = [r for r in rows if r.get("ground_truth_feasible") is True]
    n_feasible = len(feasible_rows)

    # 7 Manuscript Main-Table Metrics
    # 1. Outcome Correct (%)
    outcome_correct_count = sum(1 for r in rows if r.get("outcome_correct") is True)
    outcome_correct_rate = outcome_correct_count / n_total if n_total > 0 else 0.0
    outcome_correct_ci = wilson_interval(outcome_correct_count, n_total)

    # 2. Feasible-Task Success (%)
    feasible_task_success_count = sum(1 for r in feasible_rows if r.get("actual_task_success") is True)
    feasible_task_success_rate = feasible_task_success_count / n_feasible if n_feasible > 0 else 0.0
    feasible_task_success_ci = wilson_interval(feasible_task_success_count, n_feasible)

    # 3. Goal Coverage (%) (Micro & Macro over GT-feasible runs)
    goal_reqs_passed_feasible = sum(r.get("goal_requirements_passed", 0) for r in feasible_rows)
    goal_reqs_total_feasible = sum(r.get("goal_requirements_total", 0) for r in feasible_rows)
    goal_coverage_micro = (
        goal_reqs_passed_feasible / goal_reqs_total_feasible if goal_reqs_total_feasible > 0 else 0.0
    )
    goal_cov_macro_list = [
        (r.get("goal_requirements_passed", 0) / r.get("goal_requirements_total"))
        for r in feasible_rows
        if (r.get("goal_requirements_total") or 0) > 0
    ]
    goal_coverage_macro = (
        sum(goal_cov_macro_list) / len(goal_cov_macro_list) if goal_cov_macro_list else 0.0
    )

    # 3b. Sequence-Induced Goal Gain (Δ Goal Coverage)
    initial_goal_reqs_passed_feasible = sum(
        (r.get("initial_requirements_passed") or 0) for r in feasible_rows
    )
    initial_goal_coverage_micro = (
        initial_goal_reqs_passed_feasible / goal_reqs_total_feasible if goal_reqs_total_feasible > 0 else 0.0
    )
    delta_goal_coverage_micro = goal_coverage_micro - initial_goal_coverage_micro

    initial_goal_cov_macro_list = [
        ((r.get("initial_requirements_passed") or 0) / r.get("goal_requirements_total"))
        for r in feasible_rows
        if (r.get("goal_requirements_total") or 0) > 0
    ]
    initial_goal_coverage_macro = (
        sum(initial_goal_cov_macro_list) / len(initial_goal_cov_macro_list) if initial_goal_cov_macro_list else 0.0
    )
    delta_goal_coverage_macro = goal_coverage_macro - initial_goal_coverage_macro

    # 4. False Completion (%)
    declared_completion_count = sum(1 for r in rows if r.get("declared_completion") is True)
    false_completion_count = sum(
        1 for r in rows
        if r.get("false_completion") is True
        or (r.get("declared_completion") is True and not r.get("actual_task_success"))
    )
    false_completion_rate = (
        (false_completion_count / declared_completion_count) if declared_completion_count > 0 else None
    )
    false_completion_ci = (
        wilson_interval(false_completion_count, declared_completion_count)
        if declared_completion_count > 0
        else (0.0, 0.0)
    )

    # 5. Physical Plan Found (%) (GT-feasible only, non-empty only)
    physical_plan_found_count = sum(1 for r in feasible_rows if r.get("physical_plan_found") is True)
    physical_plan_found_rate = physical_plan_found_count / n_feasible if n_feasible > 0 else 0.0
    physical_plan_found_ci = wilson_interval(physical_plan_found_count, n_feasible)

    # 6. Raw VLM Requests
    raw_vlm_requests_stats = continuous_stats([r.get("raw_vlm_requests") for r in rows])

    # 7. High-Level Replans
    high_level_replans_stats = continuous_stats([r.get("high_level_replans") for r in rows])

    # Secondary execution metrics
    nonempty_plan_exec_count = sum(1 for r in rows if r.get("nonempty_plan_execution_completed") is True)
    nonempty_plan_exec_rate = nonempty_plan_exec_count / n_total if n_total > 0 else 0.0
    exec_stage_completed_count = sum(1 for r in rows if r.get("execution_stage_completed") is True)

    task_success_count = sum(1 for r in rows if r.get("actual_task_success") is True)
    task_success_rate = task_success_count / n_total if n_total > 0 else 0.0
    task_success_ci = wilson_interval(task_success_count, n_total)

    action_seq_count = sum(1 for r in rows if r.get("symbolic_plan_found") is True)
    action_seq_rate = action_seq_count / n_total if n_total > 0 else 0.0
    action_seq_ci = wilson_interval(action_seq_count, n_total)

    nonempty_action_seq_count = sum(1 for r in rows if r.get("symbolic_plan_nonempty") is True)
    nonempty_action_seq_rate = nonempty_action_seq_count / n_total if n_total > 0 else 0.0
    nonempty_action_seq_ci = wilson_interval(nonempty_action_seq_count, n_total)

    val_valid_count = sum(1 for r in rows if r.get("val_plan_valid") is True)
    val_valid_rate = val_valid_count / n_total if n_total > 0 else 0.0
    val_valid_ci = wilson_interval(val_valid_count, n_total)

    exec_ready_count = sum(1 for r in rows if r.get("identity_success") is True and r.get("refinement_success") is True)
    exec_ready_rate = exec_ready_count / n_total if n_total > 0 else 0.0
    exec_ready_ci = wilson_interval(exec_ready_count, n_total)

    exec_attempted_count = sum(1 for r in rows if r.get("execution_attempted") is True)
    exec_success_count = sum(1 for r in rows if r.get("execution_success") is True)

    exec_unconditional_rate = exec_success_count / n_total if n_total > 0 else 0.0
    exec_unconditional_ci = wilson_interval(exec_success_count, n_total)

    exec_conditional_rate = exec_success_count / exec_attempted_count if exec_attempted_count > 0 else None
    exec_conditional_ci = wilson_interval(exec_success_count, exec_attempted_count) if exec_attempted_count > 0 else (0.0, 0.0)

    gge_evaluated_rows = [r for r in rows if r.get("generated_goal_evaluated") is True]
    n_gge = len(gge_evaluated_rows)
    gge_satisfied_count = sum(1 for r in gge_evaluated_rows if r.get("generated_goal_satisfied") is True)
    gge_satisfaction_rate = gge_satisfied_count / n_gge if n_gge > 0 else None
    gge_satisfaction_ci = wilson_interval(gge_satisfied_count, n_gge) if n_gge > 0 else (0.0, 0.0)
    gge_coverage = n_gge / n_total if n_total > 0 else 0.0

    total_reqs_passed = sum(r.get("benchmark_requirements_passed", 0) for r in rows)
    total_reqs_checked = sum(r.get("benchmark_requirements_total", 0) for r in rows)
    req_coverage_micro = total_reqs_passed / total_reqs_checked if total_reqs_checked > 0 else 0.0

    total_atoms_passed = sum(r.get("generated_goal_atoms_passed", 0) for r in gge_evaluated_rows)
    total_atoms_checked = sum(r.get("generated_goal_atoms_total", 0) for r in gge_evaluated_rows)
    atom_coverage_micro = total_atoms_passed / total_atoms_checked if total_atoms_checked > 0 else None

    fm_stats = continuous_stats([r.get("fm_calls") for r in rows])
    cp_stats = continuous_stats([r.get("cp_calls") for r in rows])
    plan_len_stats = continuous_stats([r.get("best_symbolic_plan_length") for r in rows if r.get("symbolic_plan_found")])
    time_stats = continuous_stats([r.get("end_to_end_seconds") for r in rows])

    cp_counts = Counter(r.get("cp_calls", 0) for r in rows)
    cp0_count = cp_counts.get(0, 0)
    cp1_count = cp_counts.get(1, 0)
    cp2_count = cp_counts.get(2, 0)
    cp3_count = cp_counts.get(3, 0)

    confusion = compute_confusion_matrix(rows)

    return {
        "group": group_label,
        "total_runs": n_total,
        "feasible_runs": n_feasible,
        # 7 Manuscript Main-Table Metrics
        "outcome_correct_count": outcome_correct_count,
        "outcome_correct_rate": outcome_correct_rate,
        "outcome_correct_ci95": list(outcome_correct_ci),
        "feasible_task_success_count": feasible_task_success_count,
        "feasible_task_success_rate": feasible_task_success_rate,
        "feasible_task_success_ci95": list(feasible_task_success_ci),
        "goal_requirements_passed_feasible": goal_reqs_passed_feasible,
        "goal_requirements_total_feasible": goal_reqs_total_feasible,
        "goal_coverage_micro": goal_coverage_micro,
        "goal_coverage_macro": goal_coverage_macro,
        "initial_goal_requirements_passed_feasible": initial_goal_reqs_passed_feasible,
        "initial_goal_coverage_micro": initial_goal_coverage_micro,
        "initial_goal_coverage_macro": initial_goal_coverage_macro,
        "delta_goal_coverage_micro": delta_goal_coverage_micro,
        "delta_goal_coverage_macro": delta_goal_coverage_macro,
        "declared_completion_count": declared_completion_count,
        "false_completion_count": false_completion_count,
        "false_completion_rate": false_completion_rate,
        "false_completion_ci95": list(false_completion_ci),
        "physical_plan_found_count": physical_plan_found_count,
        "physical_plan_found_rate": physical_plan_found_rate,
        "physical_plan_found_ci95": list(physical_plan_found_ci),
        "raw_vlm_requests": raw_vlm_requests_stats,
        "high_level_replans": high_level_replans_stats,
        # Backwards compatible & secondary metrics
        "task_success_count": task_success_count,
        "task_success_rate": task_success_rate,
        "task_success_ci95": list(task_success_ci),
        "task_success_feasible_count": feasible_task_success_count,
        "task_success_feasible_rate": feasible_task_success_rate,
        "task_success_feasible_ci95": list(feasible_task_success_ci),
        "action_sequence_generation_count": action_seq_count,
        "action_sequence_generation_rate": action_seq_rate,
        "action_sequence_generation_ci95": list(action_seq_ci),
        "nonempty_action_sequence_count": nonempty_action_seq_count,
        "nonempty_action_sequence_rate": nonempty_action_seq_rate,
        "nonempty_action_sequence_ci95": list(nonempty_action_seq_ci),
        "val_valid_plan_count": val_valid_count,
        "val_valid_plan_rate": val_valid_rate,
        "val_valid_plan_ci95": list(val_valid_ci),
        "execution_ready_plan_count": exec_ready_count,
        "execution_ready_plan_rate": exec_ready_rate,
        "execution_ready_plan_ci95": list(exec_ready_ci),
        "execution_attempted_count": exec_attempted_count,
        "execution_success_count": exec_success_count,
        "execution_stage_completed_count": exec_stage_completed_count,
        "nonempty_plan_execution_completed_count": nonempty_plan_exec_count,
        "nonempty_plan_execution_completed_rate": nonempty_plan_exec_rate,
        "physical_execution_rate_unconditional": exec_unconditional_rate,
        "physical_execution_unconditional_ci95": list(exec_unconditional_ci),
        "physical_execution_rate_conditional": exec_conditional_rate,
        "physical_execution_conditional_ci95": list(exec_conditional_ci),
        "generated_goal_evaluated_count": n_gge,
        "generated_goal_evaluation_coverage": gge_coverage,
        "generated_goal_satisfied_count": gge_satisfied_count,
        "generated_goal_satisfaction_rate": gge_satisfaction_rate,
        "generated_goal_satisfaction_ci95": list(gge_satisfaction_ci),
        "benchmark_requirement_coverage_micro": req_coverage_micro,
        "generated_goal_atom_coverage_micro": atom_coverage_micro,
        "fm_calls": fm_stats,
        "cp_calls": cp_stats,
        "plan_length": plan_len_stats,
        "end_to_end_seconds": time_stats,
        "cp_iteration_counts": {
            "cp0": cp0_count,
            "cp1": cp1_count,
            "cp2": cp2_count,
            "cp3": cp3_count,
        },
        "feasibility_confusion": confusion,
    }


def compute_stage_funnel(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute unconditional and sequential stage funnel metrics.

    Maintains diagnostic ANY PLAN counts and the strictly nested same-attempt
    action-sequence funnel:
    NONEMPTY PLAN@FD -> NONEMPTY PLAN@VAL -> NONEMPTY PLAN@IDENTITY -> NONEMPTY PLAN@REFINE -> PLAN@EXEC -> TASK@FINAL.
    All conditional conversion rates in the action-sequence chain are <= 100%.
    """
    n = len(rows)

    def _pred(r: Mapping[str, Any], key: str, fallback_key: str | None = None) -> bool:
        if key in r and r[key] is not None:
            return bool(r[key])
        if fallback_key is not None and fallback_key in r and r[fallback_key] is not None:
            return bool(r[fallback_key])
        return False

    stages = [
        ("1_observation", "Observation", lambda r: _pred(r, "observation_success"), None),
        ("2_object_estimation", "Object estimation", lambda r: _pred(r, "object_estimation_success"), "1_observation"),
        ("3_pddl_valid", "Valid PDDL", lambda r: _pred(r, "pddl_valid"), "2_object_estimation"),
        ("4_any_plan_fd", "ANY PLAN@FD", lambda r: _pred(r, "any_plan_fd", "symbolic_plan_found"), "3_pddl_valid"),
        ("4b_any_plan_val", "ANY PLAN@VAL", lambda r: _pred(r, "any_plan_val", "val_plan_valid"), "4_any_plan_fd"),
        ("5_nonempty_plan_fd", "NONEMPTY PLAN@FD", lambda r: _pred(r, "nonempty_plan_fd", "symbolic_plan_nonempty"), "3_pddl_valid"),
        ("6_nonempty_plan_val", "NONEMPTY PLAN@VAL", lambda r: _pred(r, "nonempty_plan_val", "val_plan_valid") and _pred(r, "nonempty_plan_fd", "symbolic_plan_nonempty"), "5_nonempty_plan_fd"),
        ("7_nonempty_plan_identity", "NONEMPTY PLAN@IDENTITY", lambda r: _pred(r, "nonempty_plan_identity", "identity_success") and _pred(r, "nonempty_plan_val", "val_plan_valid") and _pred(r, "nonempty_plan_fd", "symbolic_plan_nonempty"), "6_nonempty_plan_val"),
        ("8_nonempty_plan_refine", "NONEMPTY PLAN@REFINE", lambda r: _pred(r, "nonempty_plan_refine", "physical_plan_found"), "7_nonempty_plan_identity"),
        ("9_nonempty_exec_success", "NONEMPTY PLAN@EXEC", lambda r: _pred(r, "nonempty_plan_exec", "nonempty_plan_execution_completed"), "8_nonempty_plan_refine"),
        ("10_task_success", "TASK@FINAL", lambda r: _pred(r, "task_final", "actual_task_success") and _pred(r, "nonempty_plan_exec", "nonempty_plan_execution_completed"), "9_nonempty_exec_success"),
    ]

    stage_counts: dict[str, int] = {}
    funnel = []

    for stage_id, name, predicate, prev_id in stages:
        count = sum(1 for r in rows if predicate(r))
        stage_counts[stage_id] = count
        prev_count = stage_counts.get(prev_id, n) if prev_id is not None else n
        if count > prev_count:
            raise ValueError(
                f"Funnel nesting invariant violated: {stage_id} ({name}) count {count} "
                f"exceeds predecessor {prev_id} count {prev_count}"
            )
        unconditional_rate = count / n if n > 0 else 0.0
        conditional_rate = count / prev_count if prev_count > 0 else 0.0
        ci_low, ci_high = wilson_interval(count, n)
        funnel.append({
            "stage_id": stage_id,
            "stage_name": name,
            "count": count,
            "total_runs": n,
            "unconditional_rate": unconditional_rate,
            "unconditional_ci95": [ci_low, ci_high],
            "previous_stage_count": prev_count,
            "conditional_conversion_rate": conditional_rate,
        })
    return funnel


def audit_benchmark_failures(rows: Sequence[Mapping[str, Any]], results_root: Path) -> list[dict[str, Any]]:
    """Audit all 18 BENCHMARK_FAILURE runs individually and classify failed atoms."""
    records = []
    bm_runs = [r for r in rows if r.get("raw_terminal_status") == "BENCHMARK_FAILURE"]

    for r in bm_runs:
        run_id = r["run_id"]
        rdir = results_root / "runs" / run_id / "artifacts"

        # Load initial vs goal facts
        init_facts_file = rdir / "interpreter" / "selected_initial_facts.json"
        goal_facts_file = rdir / "interpreter" / "selected_goal_facts.json"
        init_literals = set()
        goal_literals = set()
        if init_facts_file.is_file():
            with open(init_facts_file, encoding="utf-8") as f:
                init_literals = {fact.get("literal") for fact in json.load(f).get("facts", [])}
        if goal_facts_file.is_file():
            with open(goal_facts_file, encoding="utf-8") as f:
                goal_literals = {fact.get("literal") for fact in json.load(f).get("facts", [])}

        overlap = goal_literals.intersection(init_literals)
        overlap_count = len(overlap)
        goal_count = len(goal_literals)

        # Actions planned / attempted / succeeded
        plan_length = r.get("best_symbolic_plan_length") or 0
        exec_attempted = 0
        exec_succeeded = 0
        exec_trace_file = rdir / "execution" / "execution_trace.json"
        if exec_trace_file.is_file():
            with open(exec_trace_file, encoding="utf-8") as f:
                actions = json.load(f).get("controller_actions", [])
                exec_attempted = len(actions)
                exec_succeeded = sum(1 for a in actions if a.get("success") is True)

        # Failed generated goal atoms
        gge_file = rdir / "benchmark" / "generated_goal_evaluation.json"
        failed_goal_atoms = []
        if gge_file.is_file():
            with open(gge_file, encoding="utf-8") as f:
                for c in json.load(f).get("goal_checks", []):
                    if not c.get("passed"):
                        failed_goal_atoms.append(c.get("atom"))

        # Failed hidden requirements
        bme_file = rdir / "benchmark" / "benchmark_goal_evaluation.json"
        failed_hidden_reqs = []
        if bme_file.is_file():
            with open(bme_file, encoding="utf-8") as f:
                for req in json.load(f).get("requirement_checks", []):
                    if not req.get("passed"):
                        failed_hidden_reqs.append(req.get("name"))

        # Classification
        if overlap_count == goal_count and plan_length == 0:
            classification = "EMPTY_OR_TRIVIAL_PLAN_FROM_HALLUCINATED_INIT"
        elif plan_length == 0:
            classification = "GENUINE_MODEL_STATE_ERROR"
        elif exec_succeeded < exec_attempted:
            classification = "EXECUTION_SEMANTICS_FAILURE"
        else:
            classification = "GENUINE_MODEL_GOAL_ERROR"

        records.append({
            "run_id": run_id,
            "domain": r["domain"],
            "variant": r["variant"],
            "protocol": r["observation_protocol"],
            "repeat": r["repeat_index"],
            "plan_length": plan_length,
            "goal_atom_count": goal_count,
            "initial_goal_overlap_count": overlap_count,
            "execution_actions_planned": plan_length,
            "execution_actions_attempted": exec_attempted,
            "execution_actions_succeeded": exec_succeeded,
            "generated_goal_atoms_passed": r["generated_goal_atoms_passed"],
            "generated_goal_atoms_total": r["generated_goal_atoms_total"],
            "hidden_requirements_passed": r["benchmark_requirements_passed"],
            "hidden_requirements_total": r["benchmark_requirements_total"],
            "failed_generated_goal_atoms": "; ".join(failed_goal_atoms),
            "failed_hidden_requirements": "; ".join(failed_hidden_reqs),
            "causal_classification": classification,
        })
    return records


def audit_identity_failures(rows: Sequence[Mapping[str, Any]], results_root: Path) -> list[dict[str, Any]]:
    """Audit all 39 IDENTITY_FAILURE runs."""
    records = []
    id_runs = [r for r in rows if r.get("raw_terminal_status") == "IDENTITY_FAILURE"]

    for r in id_runs:
        run_id = r["run_id"]
        rdir = results_root / "runs" / run_id / "artifacts"

        sym_obj = None
        cand_count = 0
        cand_bodies = []
        reason = None

        for att in sorted((rdir / "attempts").glob("*")):
            ao_file = att / "attempt_outcome.json"
            if ao_file.is_file():
                with open(ao_file, encoding="utf-8") as f:
                    ao = json.load(f)
                    fail = ao.get("failure") or {}
                    if fail.get("kind") == "ENTITY_RESOLUTION" or fail.get("details", {}).get("stage") == "ENTITY_RESOLUTION":
                        details = fail.get("details") or {}
                        objs = details.get("object_ids", [])
                        sym_obj = objs[0] if objs else None
                        cand_bodies = details.get("candidate_entities", [])
                        cand_count = len(cand_bodies)
                        reason = details.get("reason_code") or fail.get("summary")
                        break

        if reason == "AMBIGUOUS_ENTITY" or cand_count > 1:
            classification = "MULTIPLE_PHYSICAL_CANDIDATES"
        elif cand_count == 0:
            classification = "NO_PHYSICAL_CANDIDATE"
        else:
            classification = "SINGLE_PHYSICAL_CANDIDATE_BUT_UNRESOLVED"

        records.append({
            "run_id": run_id,
            "domain": r["domain"],
            "variant": r["variant"],
            "protocol": r["observation_protocol"],
            "repeat": r["repeat_index"],
            "symbolic_object": sym_obj,
            "pddl_type": "unknown",
            "candidate_count": cand_count,
            "candidate_physical_bodies": "; ".join(cand_bodies),
            "centroid_distance": None,
            "aabb_distance": None,
            "ambiguity_margin": None,
            "failure_reason": reason,
            "observation_stage": r["observation_protocol"],
            "classification": classification,
        })
    return records


def audit_refinement_failures(rows: Sequence[Mapping[str, Any]], results_root: Path) -> list[dict[str, Any]]:
    """Audit all REFINEMENT_FAILURE runs with neutral diagnostic classes."""
    records = []
    ref_runs = [r for r in rows if r.get("raw_terminal_status") == "REFINEMENT_FAILURE"]

    for r in ref_runs:
        run_id = r["run_id"]
        rdir = results_root / "runs" / run_id / "artifacts"

        operator = None
        stage = None
        reason_code = None

        for att in sorted((rdir / "attempts").glob("*")):
            ao_file = att / "attempt_outcome.json"
            if ao_file.is_file():
                with open(ao_file, encoding="utf-8") as f:
                    ao = json.load(f)
                    fail = ao.get("failure") or {}
                    if fail.get("kind") == "REFINEMENT" or fail.get("details", {}).get("stage") in {"IK", "COLLISION", "SKILL_ENVELOPE", "REFINEMENT"}:
                        details = fail.get("details") or {}
                        stage = details.get("stage")
                        rfail = details.get("refinement_failure") or {}
                        operator = rfail.get("operator")
                        reason_code = rfail.get("reason_code") or details.get("reason_code")
                        break

        stage_str = (stage or "").upper()
        if "IK" in stage_str:
            classification = "REFINEMENT_IK_REJECTION"
        elif "COLLISION" in stage_str:
            classification = "REFINEMENT_COLLISION_REJECTION"
        elif "ENVELOPE" in stage_str or "SKILL" in stage_str:
            classification = "REFINEMENT_SKILL_ENVELOPE_REJECTION"
        else:
            classification = "REFINEMENT_OTHER"

        records.append({
            "run_id": run_id,
            "domain": r["domain"],
            "variant": r["variant"],
            "protocol": r["observation_protocol"],
            "repeat": r["repeat_index"],
            "operator": operator,
            "failure_stage": stage,
            "reason_code": reason_code,
            "classification": classification,
        })
    return records


def audit_generated_goals(rows: Sequence[Mapping[str, Any]], results_root: Path) -> list[dict[str, Any]]:
    """Audit every atom checked across all evaluated generated goals."""
    records = []
    gge_runs = [r for r in rows if r.get("generated_goal_evaluated")]

    for r in gge_runs:
        run_id = r["run_id"]
        gge_file = results_root / "runs" / run_id / "artifacts" / "benchmark" / "generated_goal_evaluation.json"
        if not gge_file.is_file():
            continue
        with open(gge_file, encoding="utf-8") as f:
            gdata = json.load(f)
            for check in gdata.get("goal_checks", []):
                records.append({
                    "run_id": run_id,
                    "domain": r["domain"],
                    "variant": r["variant"],
                    "protocol": r["observation_protocol"],
                    "repeat": r["repeat_index"],
                    "attempt_index": gdata.get("attempt_index", 0),
                    "atom": check.get("atom"),
                    "predicate": check.get("predicate"),
                    "arguments": "; ".join(check.get("arguments", [])),
                    "passed": bool(check.get("passed")),
                    "physical_evidence": json.dumps(check.get("physical_evidence", {})),
                    "failure_classification": "EMPTY_PLAN_HALLUCINATED_INIT" if not check.get("passed") else "PASSED",
                })
    return records


def audit_predicted_infeasible_causes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Audit terminal causal failure for every run carrying predicted_infeasible = True."""
    records = []
    for r in rows:
        if r.get("raw_predicted_infeasible") is True:
            records.append({
                "run_id": r["run_id"],
                "domain": r["domain"],
                "variant": r["variant"],
                "protocol": r["observation_protocol"],
                "repeat": r["repeat_index"],
                "ground_truth_feasible": r["ground_truth_feasible"],
                "raw_predicted_infeasible": r["raw_predicted_infeasible"],
                "terminal_causal_failure": r["causal_category"],
                "cp_terminal_kind": r["cp_terminal_kind"],
                "cp_terminal_stage": r["cp_terminal_stage"],
                "summary": r["cp_terminal_summary"],
            })
    return records


def extract_action_sequence_artifacts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate index rows and full action sequence records."""
    index_rows = []
    all_sequences = []

    for r in rows:
        attempts = r.get("attempts", [])

        summary = "NO_PLAN"
        best_plan = None
        for a in attempts:
            if a["nonempty_plan"]:
                best_plan = a
                break
        if best_plan is None and attempts:
            best_plan = next((a for a in attempts if a["plan_found"]), None)

        if best_plan:
            ops = [act.get("operator", "") for act in best_plan.get("actions", [])]
            summary = f"len={len(ops)}: " + " -> ".join(ops[:5])
            if len(ops) > 5:
                summary += f" ... (+{len(ops)-5} more)"

        index_rows.append({
            "run_id": r["run_id"],
            "domain": r["domain"],
            "variant": r["variant"],
            "protocol": r["observation_protocol"],
            "repeat": r["repeat_index"],
            "first_plan_attempt_index": r["first_plan_attempt_index"],
            "any_symbolic_plan_found": r["symbolic_plan_found"],
            "any_nonempty_symbolic_plan_found": r["symbolic_plan_nonempty"],
            "any_val_valid_plan": r["val_plan_valid"],
            "number_of_symbolic_plans_generated": r["number_of_symbolic_plans_generated"],
            "number_of_val_valid_plans": r["number_of_val_valid_plans"],
            "best_symbolic_plan_length": r["best_symbolic_plan_length"],
            "final_selected_plan_length": r["final_selected_plan_length"],
            "selected_action_sequence_summary": summary,
        })

        for a in attempts:
            if a["plan_found"]:
                all_sequences.append({
                    "run_id": r["run_id"],
                    "domain": r["domain"],
                    "variant": r["variant"],
                    "protocol": r["observation_protocol"],
                    "repeat": r["repeat_index"],
                    "attempt_index": a["attempt_index"],
                    "problem_sha256": a["problem_sha256"],
                    "plan_cost": a["plan_cost"],
                    "plan_artifact": a["plan_artifact"],
                    "actions": a["actions"],
                    "plan_length": a["plan_length"],
                    "val_status": "VALID" if a["val_valid"] else "INVALID",
                    "identity_status": "SUCCESS" if a["identity_success"] else "FAILURE" if a["identity_attempted"] else "NOT_ATTEMPTED",
                    "refinement_status": "SUCCESS" if a["refinement_success"] else "FAILURE" if a["refinement_attempted"] else "NOT_ATTEMPTED",
                })

    return index_rows, all_sequences


def extract_representative_action_sequences(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Find the lowest run_id in each domain having a VAL-valid NON-EMPTY symbolic plan."""
    domains = sorted({str(r["domain"]) for r in rows if r.get("domain")})
    result: dict[str, dict[str, Any] | None] = {}
    for d in domains:
        drows = [r for r in rows if str(r.get("domain")) == d]
        drows.sort(key=lambda r: str(r.get("run_id", "")))
        rep = None
        for r in drows:
            attempts = r.get("attempts", [])
            valid_nonempty = [a for a in attempts if a.get("nonempty_plan") and a.get("val_valid")]
            if valid_nonempty:
                first_plan = valid_nonempty[0]
                actions = first_plan.get("actions", [])
                seq_str = " -> ".join(
                    f"{a.get('operator')}({', '.join(a.get('arguments', []))})" for a in actions
                ) if actions else "[]"
                rep = {
                    "run_id": r["run_id"],
                    "domain": d,
                    "variant": r.get("variant"),
                    "attempt_index": first_plan.get("attempt_index"),
                    "plan_length": first_plan.get("plan_length"),
                    "action_sequence_str": seq_str,
                    "actions": actions,
                }
                break
        result[d] = rep
    return result


def generate_manuscript_main_table(
    overall: Mapping[str, Any],
    by_domain: Sequence[Mapping[str, Any]],
    output_dir: Path,
    file_prefix: str = "smoke_",
) -> None:
    """Generate the exact 7 manuscript main-table metrics across benchmark domains."""
    domain_map = {d["group"]: d for d in by_domain}
    kitchen = domain_map.get("kitchen", {})
    living = domain_map.get("living_room", {})
    workshop = domain_map.get("workshop", {})

    def rate_str(cnt: Any, den: Any, val: Any) -> str:
        if val is None or den is None or den == 0:
            return "N/A"
        return f"{val * 100:.1f}\\% ({cnt}/{den})"

    def false_comp_str(grp: Mapping[str, Any]) -> str:
        dec = grp.get("declared_completion_count", 0)
        fc = grp.get("false_completion_count", 0)
        rate = grp.get("false_completion_rate")
        if dec == 0 or rate is None:
            return "N/A"
        return f"{rate * 100:.1f}\\% ({fc}/{dec})"

    def stat_str(stat: Mapping[str, Any]) -> str:
        m = stat.get("mean")
        s = stat.get("std")
        if m is None:
            return "N/A"
        if s is not None and s > 0:
            return f"{m:.2f} $\\pm$ {s:.2f}"
        return f"{m:.2f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{ViLaIn-TAMP-Qwen Baseline Performance Across Benchmark Domains.}",
        r"Evaluated over non-infrastructure runs ($N=" + str(overall.get("total_runs", 0)) + r"$).",
        r"Feasible-task success, Goal coverage, and Physical plan found are evaluated strictly over ground-truth feasible variants ($N_{\text{feasible}}=" + str(overall.get("feasible_runs", 0)) + r"$).",
        r"False completion reports rate over declared-complete runs ($N_{\text{declared}}$). Zero-step plans do not count toward Physical plan found.}",
        r"\label{tab:vilain_qwen_main}",
        r"\begin{tabular}{l|c|ccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Overall} & \textbf{Kitchen} & \textbf{Living Room} & \textbf{Workshop} \\",
        r"\hline",
        f"Total Non-Infrastructure Runs & {overall.get('total_runs', 0)} & {kitchen.get('total_runs', 0)} & {living.get('total_runs', 0)} & {workshop.get('total_runs', 0)} \\\\",
        f"Feasible Variant Runs & {overall.get('feasible_runs', 0)} & {kitchen.get('feasible_runs', 0)} & {living.get('feasible_runs', 0)} & {workshop.get('feasible_runs', 0)} \\\\",
        r"\hline",
        f"Outcome correct (\\%) $\\uparrow$ & "
        + f"{rate_str(overall.get('outcome_correct_count'), overall.get('total_runs'), overall.get('outcome_correct_rate'))} & "
        + f"{rate_str(kitchen.get('outcome_correct_count'), kitchen.get('total_runs'), kitchen.get('outcome_correct_rate'))} & "
        + f"{rate_str(living.get('outcome_correct_count'), living.get('total_runs'), living.get('outcome_correct_rate'))} & "
        + f"{rate_str(workshop.get('outcome_correct_count'), workshop.get('total_runs'), workshop.get('outcome_correct_rate'))} \\\\",
        f"Feasible-task success (\\%) $\\uparrow$ & "
        + f"{rate_str(overall.get('feasible_task_success_count'), overall.get('feasible_runs'), overall.get('feasible_task_success_rate'))} & "
        + f"{rate_str(kitchen.get('feasible_task_success_count'), kitchen.get('feasible_runs'), kitchen.get('feasible_task_success_rate'))} & "
        + f"{rate_str(living.get('feasible_task_success_count'), living.get('feasible_runs'), living.get('feasible_task_success_rate'))} & "
        + f"{rate_str(workshop.get('feasible_task_success_count'), workshop.get('feasible_runs'), workshop.get('feasible_task_success_rate'))} \\\\",
        f"Goal coverage (\\%) $\\uparrow$ & "
        + f"{rate_str(overall.get('goal_requirements_passed_feasible'), overall.get('goal_requirements_total_feasible'), overall.get('goal_coverage_micro'))} & "
        + f"{rate_str(kitchen.get('goal_requirements_passed_feasible'), kitchen.get('goal_requirements_total_feasible'), kitchen.get('goal_coverage_micro'))} & "
        + f"{rate_str(living.get('goal_requirements_passed_feasible'), living.get('goal_requirements_total_feasible'), living.get('goal_coverage_micro'))} & "
        + f"{rate_str(workshop.get('goal_requirements_passed_feasible'), workshop.get('goal_requirements_total_feasible'), workshop.get('goal_coverage_micro'))} \\\\",
        f"False completion (\\%) $\\downarrow$ & "
        + f"{false_comp_str(overall)} & "
        + f"{false_comp_str(kitchen)} & "
        + f"{false_comp_str(living)} & "
        + f"{false_comp_str(workshop)} \\\\",
        f"Physical plan found (\\%) $\\uparrow$ & "
        + f"{rate_str(overall.get('physical_plan_found_count'), overall.get('feasible_runs'), overall.get('physical_plan_found_rate'))} & "
        + f"{rate_str(kitchen.get('physical_plan_found_count'), kitchen.get('feasible_runs'), kitchen.get('physical_plan_found_rate'))} & "
        + f"{rate_str(living.get('physical_plan_found_count'), living.get('feasible_runs'), living.get('physical_plan_found_rate'))} & "
        + f"{rate_str(workshop.get('physical_plan_found_count'), workshop.get('feasible_runs'), workshop.get('physical_plan_found_rate'))} \\\\",
        f"Raw VLM requests $\\downarrow$ & "
        + f"{stat_str(overall.get('raw_vlm_requests', {}))} & "
        + f"{stat_str(kitchen.get('raw_vlm_requests', {}))} & "
        + f"{stat_str(living.get('raw_vlm_requests', {}))} & "
        + f"{stat_str(workshop.get('raw_vlm_requests', {}))} \\\\",
        f"High-level replans (\\#) $\\downarrow$ & "
        + f"{stat_str(overall.get('high_level_replans', {}))} & "
        + f"{stat_str(kitchen.get('high_level_replans', {}))} & "
        + f"{stat_str(living.get('high_level_replans', {}))} & "
        + f"{stat_str(workshop.get('high_level_replans', {}))} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    tex_content = "\n".join(lines) + "\n"
    (output_dir / f"{file_prefix}main_table.tex").write_text(tex_content, encoding="utf-8")

    def pct_str(val: float | None) -> str:
        return f"{val*100:.1f}\\%" if val is not None else "N/A"

    attr_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{ViLaIn-TAMP-Qwen Sequence-Induced Goal Gain Across Benchmark Domains.}",
        r"Evaluated over ground-truth feasible variants ($N_{\text{feasible}}=" + str(overall.get("feasible_runs", 0)) + r"$).",
        r"Initial Goal Coverage reflects requirements satisfied prior to manipulation. Terminal Goal Coverage reflects requirements after execution. $\Delta$ Goal Coverage measures sequence-induced progress.}",
        r"\label{tab:vilain_goal_attribution}",
        r"\begin{tabular}{l|c|ccc}",
        r"\hline",
        r"\textbf{Goal Coverage Metric} & \textbf{Overall} & \textbf{Kitchen} & \textbf{Living Room} & \textbf{Workshop} \\",
        r"\hline",
        f"Feasible Variant Runs & {overall.get('feasible_runs', 0)} & {kitchen.get('feasible_runs', 0)} & {living.get('feasible_runs', 0)} & {workshop.get('feasible_runs', 0)} \\\\",
        r"\hline",
        f"Initial Goal Coverage (\\%) & "
        + f"{rate_str(overall.get('initial_goal_requirements_passed_feasible'), overall.get('goal_requirements_total_feasible'), overall.get('initial_goal_coverage_micro'))} & "
        + f"{rate_str(kitchen.get('initial_goal_requirements_passed_feasible'), kitchen.get('goal_requirements_total_feasible'), kitchen.get('initial_goal_coverage_micro'))} & "
        + f"{rate_str(living.get('initial_goal_requirements_passed_feasible'), living.get('goal_requirements_total_feasible'), living.get('initial_goal_coverage_micro'))} & "
        + f"{rate_str(workshop.get('initial_goal_requirements_passed_feasible'), workshop.get('goal_requirements_total_feasible'), workshop.get('initial_goal_coverage_micro'))} \\\\",
        f"Terminal Goal Coverage (\\%) & "
        + f"{rate_str(overall.get('goal_requirements_passed_feasible'), overall.get('goal_requirements_total_feasible'), overall.get('goal_coverage_micro'))} & "
        + f"{rate_str(kitchen.get('goal_requirements_passed_feasible'), kitchen.get('goal_requirements_total_feasible'), kitchen.get('goal_coverage_micro'))} & "
        + f"{rate_str(living.get('goal_requirements_passed_feasible'), living.get('goal_requirements_total_feasible'), living.get('goal_coverage_micro'))} & "
        + f"{rate_str(workshop.get('goal_requirements_passed_feasible'), workshop.get('goal_requirements_total_feasible'), workshop.get('goal_coverage_micro'))} \\\\",
        f"$\\Delta$ Goal Coverage (\\%) & "
        + f"{pct_str(overall.get('delta_goal_coverage_micro'))} & "
        + f"{pct_str(kitchen.get('delta_goal_coverage_micro'))} & "
        + f"{pct_str(living.get('delta_goal_coverage_micro'))} & "
        + f"{pct_str(workshop.get('delta_goal_coverage_micro'))} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (output_dir / f"{file_prefix}goal_attribution_table.tex").write_text("\n".join(attr_lines) + "\n", encoding="utf-8")

    def format_group_metrics(grp: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "total_runs": grp.get("total_runs"),
            "feasible_runs": grp.get("feasible_runs"),
            "outcome_correct_rate": grp.get("outcome_correct_rate"),
            "outcome_correct_count": grp.get("outcome_correct_count"),
            "feasible_task_success_rate": grp.get("feasible_task_success_rate"),
            "feasible_task_success_count": grp.get("feasible_task_success_count"),
            "goal_coverage_micro": grp.get("goal_coverage_micro"),
            "goal_coverage_macro": grp.get("goal_coverage_macro"),
            "goal_requirements_passed_feasible": grp.get("goal_requirements_passed_feasible"),
            "goal_requirements_total_feasible": grp.get("goal_requirements_total_feasible"),
            "initial_goal_coverage_micro": grp.get("initial_goal_coverage_micro"),
            "initial_goal_coverage_macro": grp.get("initial_goal_coverage_macro"),
            "delta_goal_coverage_micro": grp.get("delta_goal_coverage_micro"),
            "delta_goal_coverage_macro": grp.get("delta_goal_coverage_macro"),
            "initial_goal_requirements_passed_feasible": grp.get("initial_goal_requirements_passed_feasible"),
            "false_completion_rate": grp.get("false_completion_rate"),
            "false_completion_count": grp.get("false_completion_count"),
            "declared_completion_count": grp.get("declared_completion_count"),
            "physical_plan_found_rate": grp.get("physical_plan_found_rate"),
            "physical_plan_found_count": grp.get("physical_plan_found_count"),
            "raw_vlm_requests": grp.get("raw_vlm_requests"),
            "high_level_replans": grp.get("high_level_replans"),
        }

    main_metrics = {
        "overall": format_group_metrics(overall),
        "kitchen": format_group_metrics(kitchen),
        "living_room": format_group_metrics(living),
        "workshop": format_group_metrics(workshop),
    }
    (output_dir / f"{file_prefix}main_table_metrics.json").write_text(
        json.dumps(main_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


def generate_latex_tables(
    overall: Mapping[str, Any],
    by_domain: Sequence[Mapping[str, Any]],
    by_protocol: Sequence[Mapping[str, Any]],
    funnel: Sequence[Mapping[str, Any]],
    confusion: Mapping[str, Any],
    causal_categories: Mapping[str, int],
    output_dir: Path,
) -> None:
    """Generate clean, paper-ready LaTeX tables."""
    domain_map = {d["group"]: d for d in by_domain}
    kitchen = domain_map.get("kitchen", {})
    living = domain_map.get("living_room", {})
    workshop = domain_map.get("workshop", {})

    proto_map = {p["group"]: p for p in by_protocol}
    fixed = proto_map.get("fixed_full_inspection", {})
    initial = proto_map.get("initial_observation_only", {})

    def pct_str(val: float | None) -> str:
        return f"{val*100:.1f}\\%" if val is not None else "N/A"

    def mean_str(stat: Mapping[str, Any]) -> str:
        m = stat.get("mean")
        s = stat.get("std")
        if m is None:
            return "N/A"
        return f"{m:.2f} $\\pm$ {s:.2f}" if s is not None else f"{m:.2f}"

    main_tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{ViLaIn-TAMP-Qwen Baseline Performance Across Benchmark Domains and Observation Protocols.}
All metrics are evaluated over the non-infrastructure runs ($N=""" + str(overall.get("total_runs", 0)) + r"""$, 0 infrastructure failures).
Action Sequence Generation Rate and VAL-valid Plan Rate are evaluated independently of downstream geometric refinement.
Denominators are strictly reported for conditional evaluations.}
\label{tab:vilain_qwen_main_legacy}
\begin{tabular}{l|c|ccc|cc}
\hline
\textbf{Metric} & \textbf{Overall} & \textbf{Kitchen} & \textbf{Living Room} & \textbf{Workshop} & \textbf{Full Inspection} & \textbf{Init Only} \\
\hline
Total Non-Infrastructure Runs & """ + str(overall.get("total_runs", 0)) + r""" & """ + str(kitchen.get("total_runs", 0)) + r""" & """ + str(living.get("total_runs", 0)) + r""" & """ + str(workshop.get("total_runs", 0)) + r""" & """ + str(fixed.get("total_runs", 0)) + r""" & """ + str(initial.get("total_runs", 0)) + r""" \\
Feasible Variant Runs & """ + str(overall.get("feasible_runs", 0)) + r""" & """ + str(kitchen.get("feasible_runs", 0)) + r""" & """ + str(living.get("feasible_runs", 0)) + r""" & """ + str(workshop.get("feasible_runs", 0)) + r""" & """ + str(fixed.get("feasible_runs", 0)) + r""" & """ + str(initial.get("feasible_runs", 0)) + r""" \\
\hline
Outcome Correct Rate & """ + pct_str(overall.get("outcome_correct_rate")) + r""" & """ + pct_str(kitchen.get("outcome_correct_rate")) + r""" & """ + pct_str(living.get("outcome_correct_rate")) + r""" & """ + pct_str(workshop.get("outcome_correct_rate")) + r""" & """ + pct_str(fixed.get("outcome_correct_rate")) + r""" & """ + pct_str(initial.get("outcome_correct_rate")) + r""" \\
Feasible-Task Success Rate & """ + pct_str(overall.get("feasible_task_success_rate")) + r""" & """ + pct_str(kitchen.get("feasible_task_success_rate")) + r""" & """ + pct_str(living.get("feasible_task_success_rate")) + r""" & """ + pct_str(workshop.get("feasible_task_success_rate")) + r""" & """ + pct_str(fixed.get("feasible_task_success_rate")) + r""" & """ + pct_str(initial.get("feasible_task_success_rate")) + r""" \\
Action Sequence Generation Rate (PLAN@FD) & """ + pct_str(overall.get("action_sequence_generation_rate")) + r""" & """ + pct_str(kitchen.get("action_sequence_generation_rate")) + r""" & """ + pct_str(living.get("action_sequence_generation_rate")) + r""" & """ + pct_str(workshop.get("action_sequence_generation_rate")) + r""" & """ + pct_str(fixed.get("action_sequence_generation_rate")) + r""" & """ + pct_str(initial.get("action_sequence_generation_rate")) + r""" \\
Non-Empty Plan Rate & """ + pct_str(overall.get("nonempty_action_sequence_rate")) + r""" & """ + pct_str(kitchen.get("nonempty_action_sequence_rate")) + r""" & """ + pct_str(living.get("nonempty_action_sequence_rate")) + r""" & """ + pct_str(workshop.get("nonempty_action_sequence_rate")) + r""" & """ + pct_str(fixed.get("nonempty_action_sequence_rate")) + r""" & """ + pct_str(initial.get("nonempty_action_sequence_rate")) + r""" \\
VAL-Valid Plan Rate (PLAN@VAL) & """ + pct_str(overall.get("val_valid_plan_rate")) + r""" & """ + pct_str(kitchen.get("val_valid_plan_rate")) + r""" & """ + pct_str(living.get("val_valid_plan_rate")) + r""" & """ + pct_str(workshop.get("val_valid_plan_rate")) + r""" & """ + pct_str(fixed.get("val_valid_plan_rate")) + r""" & """ + pct_str(initial.get("val_valid_plan_rate")) + r""" \\
Physical Plan Found Rate (PLAN@REFINE) & """ + pct_str(overall.get("physical_plan_found_rate")) + r""" & """ + pct_str(kitchen.get("physical_plan_found_rate")) + r""" & """ + pct_str(living.get("physical_plan_found_rate")) + r""" & """ + pct_str(workshop.get("physical_plan_found_rate")) + r""" & """ + pct_str(fixed.get("physical_plan_found_rate")) + r""" & """ + pct_str(initial.get("physical_plan_found_rate")) + r""" \\
Physical Execution Success Rate (Unconditional) & """ + pct_str(overall.get("physical_execution_rate_unconditional")) + r""" & """ + pct_str(kitchen.get("physical_execution_rate_unconditional")) + r""" & """ + pct_str(living.get("physical_execution_rate_unconditional")) + r""" & """ + pct_str(workshop.get("physical_execution_rate_unconditional")) + r""" & """ + pct_str(fixed.get("physical_execution_rate_unconditional")) + r""" & """ + pct_str(initial.get("physical_execution_rate_unconditional")) + r""" \\
Non-Empty Plan Execution Success Rate & """ + pct_str(overall.get("nonempty_plan_execution_completed_rate")) + r""" & """ + pct_str(kitchen.get("nonempty_plan_execution_completed_rate")) + r""" & """ + pct_str(living.get("nonempty_plan_execution_completed_rate")) + r""" & """ + pct_str(workshop.get("nonempty_plan_execution_completed_rate")) + r""" & """ + pct_str(fixed.get("nonempty_plan_execution_completed_rate")) + r""" & """ + pct_str(initial.get("nonempty_plan_execution_completed_rate")) + r""" \\
Goal Coverage (Micro) & """ + pct_str(overall.get("goal_coverage_micro")) + r""" & """ + pct_str(kitchen.get("goal_coverage_micro")) + r""" & """ + pct_str(living.get("goal_coverage_micro")) + r""" & """ + pct_str(workshop.get("goal_coverage_micro")) + r""" & """ + pct_str(fixed.get("goal_coverage_micro")) + r""" & """ + pct_str(initial.get("goal_coverage_micro")) + r""" \\
False Completion Rate & """ + pct_str(overall.get("false_completion_rate")) + r""" & """ + pct_str(kitchen.get("false_completion_rate")) + r""" & """ + pct_str(living.get("false_completion_rate")) + r""" & """ + pct_str(workshop.get("false_completion_rate")) + r""" & """ + pct_str(fixed.get("false_completion_rate")) + r""" & """ + pct_str(initial.get("false_completion_rate")) + r""" \\
\hline
Raw VLM Requests & """ + mean_str(overall.get("raw_vlm_requests", {})) + r""" & """ + mean_str(kitchen.get("raw_vlm_requests", {})) + r""" & """ + mean_str(living.get("raw_vlm_requests", {})) + r""" & """ + mean_str(workshop.get("raw_vlm_requests", {})) + r""" & """ + mean_str(fixed.get("raw_vlm_requests", {})) + r""" & """ + mean_str(initial.get("raw_vlm_requests", {})) + r""" \\
High-Level Replans & """ + mean_str(overall.get("high_level_replans", {})) + r""" & """ + mean_str(kitchen.get("high_level_replans", {})) + r""" & """ + mean_str(living.get("high_level_replans", {})) + r""" & """ + mean_str(workshop.get("high_level_replans", {})) + r""" & """ + mean_str(fixed.get("high_level_replans", {})) + r""" & """ + mean_str(initial.get("high_level_replans", {})) + r""" \\
Average Symbolic Plan Length & """ + mean_str(overall.get("plan_length", {})) + r""" & """ + mean_str(kitchen.get("plan_length", {})) + r""" & """ + mean_str(living.get("plan_length", {})) + r""" & """ + mean_str(workshop.get("plan_length", {})) + r""" & """ + mean_str(fixed.get("plan_length", {})) + r""" & """ + mean_str(initial.get("plan_length", {})) + r""" \\
\hline
\end{tabular}
\end{table*}
"""
    (output_dir / "paper_main_table.tex").write_text(main_tex, encoding="utf-8")

    failure_rows_tex = []
    for s in funnel:
        name = s["stage_name"]
        cnt = s["count"]
        tot = s["total_runs"]
        unc = s["unconditional_rate"] * 100.0
        ci = s["unconditional_ci95"]
        ci_str = f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"
        con = s["conditional_conversion_rate"] * 100.0
        failure_rows_tex.append(f"{name} & {cnt} / {tot} & {unc:.1f}\\% & {ci_str} & {con:.1f}\\% \\\\")

    failure_tex = r"""\begin{table}[t]
\centering
\small
\caption{\textbf{ViLaIn-TAMP-Qwen Stage Funnel and Cumulative Conversion Rates.}
Shows progression across pipeline stages for non-infrastructure runs ($N=""" + str(overall.get("total_runs", 0)) + r"""$).
Conditional conversion denotes transition probability from the preceding stage.}
\label{tab:vilain_qwen_funnel}
\begin{tabular}{l|c|c|c|c}
\hline
\textbf{Pipeline Stage} & \textbf{Count / Total} & \textbf{Unconditional \%} & \textbf{95\% Wilson CI} & \textbf{Conversion Rate} \\
\hline
""" + "\n".join(failure_rows_tex) + r"""
\hline
\end{tabular}
\end{table}
"""
    (output_dir / "paper_failure_table.tex").write_text(failure_tex, encoding="utf-8")

    tot_runs = overall.get("total_runs", 0)
    causal_rows_tex = []
    for cat, cnt in causal_categories.items():
        pct = (cnt / tot_runs * 100.0) if tot_runs > 0 else 0.0
        causal_rows_tex.append(f"\\multicolumn{{3}}{{l|}}{{{cat}}} & {cnt} ({pct:.1f}\\%) \\\\")

    feas_tex = r"""\begin{table}[t]
\centering
\small
\caption{\textbf{ViLaIn-TAMP-Qwen Feasibility Decision Confusion Matrix and Classification Metrics.}
Evaluated with Task Infeasible as the Positive class ($N=""" + str(confusion.get("evaluated_runs", 0)) + r"""$).}
\label{tab:vilain_qwen_feasibility}
\begin{tabular}{l|cc|l}
\hline
\multicolumn{4}{c}{\textbf{Confusion Matrix (Raw Termination Rule)}} \\
\hline
& \textbf{Predicted Infeasible} & \textbf{Predicted Feasible} & \textbf{Total Actual} \\
\hline
\textbf{Actual Infeasible} & """ + str(confusion.get("true_positives")) + r""" (TP) & """ + str(confusion.get("false_negatives")) + r""" (FN) & """ + str(confusion.get("actual_infeasible_count")) + r""" \\
\textbf{Actual Feasible}   & """ + str(confusion.get("false_positives")) + r""" (FP) & """ + str(confusion.get("true_negatives")) + r""" (TN) & """ + str(confusion.get("actual_feasible_count")) + r""" \\
\hline
\textbf{Total Predicted}   & """ + str(confusion.get("true_positives", 0) + confusion.get("false_positives", 0)) + r""" & """ + str(confusion.get("true_negatives", 0) + confusion.get("false_negatives", 0)) + r""" & """ + str(confusion.get("evaluated_runs")) + r""" \\
\hline
\hline
\multicolumn{3}{l|}{\textbf{Metric}} & \textbf{Value [95\% CI]} \\
\hline
\multicolumn{3}{l|}{Decision Coverage} & """ + pct_str(confusion.get("decision_coverage")) + r""" (""" + str(confusion.get("evaluated_runs")) + r"""/""" + str(confusion.get("total_scheduled_runs")) + r""") \\
\multicolumn{3}{l|}{Accuracy} & """ + pct_str(confusion.get("accuracy")) + r""" [""" + f"{confusion.get('accuracy_wilson_ci95', [0,0])[0]*100:.1f}, {confusion.get('accuracy_wilson_ci95', [0,0])[1]*100:.1f}" + r"""] \\
\multicolumn{3}{l|}{Precision (PPV)} & """ + pct_str(confusion.get("precision")) + r""" \\
\multicolumn{3}{l|}{Recall / Infeasible Recall (TPR)} & """ + pct_str(confusion.get("recall")) + r""" \\
\multicolumn{3}{l|}{Specificity / Feasible Recall (TNR)} & """ + pct_str(confusion.get("specificity")) + r""" \\
\multicolumn{3}{l|}{Balanced Accuracy} & """ + pct_str(confusion.get("balanced_accuracy")) + r""" \\
\multicolumn{3}{l|}{F1-Score} & """ + f"{confusion.get('f1_score', 0):.3f}" + r""" \\
\multicolumn{3}{l|}{False Infeasible Rate (FPR)} & """ + pct_str(confusion.get("false_infeasible_rate")) + r""" \\
\multicolumn{3}{l|}{False Feasible Rate (FNR)} & """ + pct_str(confusion.get("false_feasible_rate")) + r""" \\
\hline
\hline
\multicolumn{4}{c}{\textbf{Causal Failure Distribution ($N=""" + str(tot_runs) + r"""$)}} \\
\hline
""" + "\n".join(causal_rows_tex) + r"""
\hline
\end{tabular}
\end{table}
"""
    (output_dir / "paper_feasibility_table.tex").write_text(feas_tex, encoding="utf-8")


def write_csv_dicts(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write list of mappings to a CSV file."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_full_paper_analysis(
    results_root: Path,
    output_root: Path,
    file_prefix: str = "paper_",
) -> dict[str, Any]:
    """Perform read-only paper audit and analysis on a frozen results directory."""
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runs_dir = results_root / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and (d / "terminal_status.json").is_file()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with terminal_status.json found in {runs_dir}")

    # Audit all runs
    audited_rows = [audit_run(d) for d in run_dirs]

    # Save run_metrics.csv and run_metrics.jsonl
    run_metric_rows = []
    for r in audited_rows:
        copy_r = dict(r)
        copy_r.pop("attempts", None)
        run_metric_rows.append(copy_r)

    write_csv_dicts(output_root / f"{file_prefix}run_metrics.csv", run_metric_rows)
    with (output_root / f"{file_prefix}run_metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in run_metric_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # Non-infrastructure runs
    completed_rows = [r for r in audited_rows if not r.get("infrastructure_failure")]

    # Overall aggregates
    overall = compute_group_aggregates(completed_rows, group_label="overall")
    (output_root / f"{file_prefix}overall.json").write_text(json.dumps(overall, indent=2, sort_keys=True), encoding="utf-8")

    # By Domain
    domains = sorted({r["domain"] for r in completed_rows})
    domain_aggregates = [compute_group_aggregates([r for r in completed_rows if r["domain"] == d], group_label=d) for d in domains]
    domain_flat = []
    for da in domain_aggregates:
        domain_flat.append({
            "domain": da["group"],
            "total_runs": da["total_runs"],
            "feasible_runs": da["feasible_runs"],
            "outcome_correct_rate": da["outcome_correct_rate"],
            "outcome_correct_count": da["outcome_correct_count"],
            "feasible_task_success_rate": da["feasible_task_success_rate"],
            "feasible_task_success_count": da["feasible_task_success_count"],
            "goal_coverage_micro": da["goal_coverage_micro"],
            "goal_coverage_macro": da["goal_coverage_macro"],
            "initial_goal_coverage_micro": da["initial_goal_coverage_micro"],
            "terminal_goal_coverage_micro": da["goal_coverage_micro"],
            "delta_goal_coverage_micro": da["delta_goal_coverage_micro"],
            "initial_goal_coverage_macro": da["initial_goal_coverage_macro"],
            "terminal_goal_coverage_macro": da["goal_coverage_macro"],
            "delta_goal_coverage_macro": da["delta_goal_coverage_macro"],
            "false_completion_rate": da["false_completion_rate"],
            "false_completion_count": da["false_completion_count"],
            "declared_completion_count": da["declared_completion_count"],
            "physical_plan_found_rate": da["physical_plan_found_rate"],
            "physical_plan_found_count": da["physical_plan_found_count"],
            "average_raw_vlm_requests": da["raw_vlm_requests"]["mean"],
            "average_high_level_replans": da["high_level_replans"]["mean"],
            "action_sequence_generation_rate": da["action_sequence_generation_rate"],
            "nonempty_action_sequence_rate": da["nonempty_action_sequence_rate"],
            "val_valid_plan_rate": da["val_valid_plan_rate"],
            "execution_ready_plan_rate": da["execution_ready_plan_rate"],
            "physical_execution_unconditional": da["physical_execution_rate_unconditional"],
            "nonempty_plan_execution_completed_rate": da["nonempty_plan_execution_completed_rate"],
            "generated_goal_satisfaction_rate": da["generated_goal_satisfaction_rate"],
            "generated_goal_evaluation_coverage": da["generated_goal_evaluation_coverage"],
            "benchmark_requirement_coverage": da["benchmark_requirement_coverage_micro"],
            "generated_goal_atom_coverage": da["generated_goal_atom_coverage_micro"],
            "feasibility_accuracy": da["feasibility_confusion"]["accuracy"],
            "feasibility_coverage": da["feasibility_confusion"]["decision_coverage"],
            "average_fm_calls": da["fm_calls"]["mean"],
            "average_cp_calls": da["cp_calls"]["mean"],
            "average_plan_length": da["plan_length"]["mean"],
        })
    write_csv_dicts(output_root / f"{file_prefix}by_domain.csv", domain_flat)

    # By Protocol
    protocols = sorted({r["observation_protocol"] for r in completed_rows})
    protocol_aggregates = [compute_group_aggregates([r for r in completed_rows if r["observation_protocol"] == p], group_label=p) for p in protocols]
    protocol_flat = []
    for pa in protocol_aggregates:
        protocol_flat.append({
            "observation_protocol": pa["group"],
            "total_runs": pa["total_runs"],
            "feasible_runs": pa["feasible_runs"],
            "outcome_correct_rate": pa["outcome_correct_rate"],
            "outcome_correct_count": pa["outcome_correct_count"],
            "feasible_task_success_rate": pa["feasible_task_success_rate"],
            "feasible_task_success_count": pa["feasible_task_success_count"],
            "goal_coverage_micro": pa["goal_coverage_micro"],
            "goal_coverage_macro": pa["goal_coverage_macro"],
            "initial_goal_coverage_micro": pa["initial_goal_coverage_micro"],
            "terminal_goal_coverage_micro": pa["goal_coverage_micro"],
            "delta_goal_coverage_micro": pa["delta_goal_coverage_micro"],
            "initial_goal_coverage_macro": pa["initial_goal_coverage_macro"],
            "terminal_goal_coverage_macro": pa["goal_coverage_macro"],
            "delta_goal_coverage_macro": pa["delta_goal_coverage_macro"],
            "false_completion_rate": pa["false_completion_rate"],
            "false_completion_count": pa["false_completion_count"],
            "declared_completion_count": pa["declared_completion_count"],
            "physical_plan_found_rate": pa["physical_plan_found_rate"],
            "physical_plan_found_count": pa["physical_plan_found_count"],
            "average_raw_vlm_requests": pa["raw_vlm_requests"]["mean"],
            "average_high_level_replans": pa["high_level_replans"]["mean"],
            "action_sequence_generation_rate": pa["action_sequence_generation_rate"],
            "nonempty_action_sequence_rate": pa["nonempty_action_sequence_rate"],
            "val_valid_plan_rate": pa["val_valid_plan_rate"],
            "execution_ready_plan_rate": pa["execution_ready_plan_rate"],
            "physical_execution_unconditional": pa["physical_execution_rate_unconditional"],
            "nonempty_plan_execution_completed_rate": pa["nonempty_plan_execution_completed_rate"],
            "generated_goal_satisfaction_rate": pa["generated_goal_satisfaction_rate"],
            "generated_goal_evaluation_coverage": pa["generated_goal_evaluation_coverage"],
            "benchmark_requirement_coverage": pa["benchmark_requirement_coverage_micro"],
            "generated_goal_atom_coverage": pa["generated_goal_atom_coverage_micro"],
            "feasibility_accuracy": pa["feasibility_confusion"]["accuracy"],
            "feasibility_coverage": pa["feasibility_confusion"]["decision_coverage"],
            "average_fm_calls": pa["fm_calls"]["mean"],
            "average_cp_calls": pa["cp_calls"]["mean"],
            "average_plan_length": pa["plan_length"]["mean"],
        })
    write_csv_dicts(output_root / f"{file_prefix}by_protocol.csv", protocol_flat)

    # By Domain x Protocol
    dom_proto_flat = []
    for d in domains:
        for p in protocols:
            sub = [r for r in completed_rows if r["domain"] == d and r["observation_protocol"] == p]
            if not sub:
                continue
            spa = compute_group_aggregates(sub, group_label=f"{d}__{p}")
            dom_proto_flat.append({
                "domain": d,
                "observation_protocol": p,
                "total_runs": spa["total_runs"],
                "feasible_runs": spa["feasible_runs"],
                "outcome_correct_rate": spa["outcome_correct_rate"],
                "outcome_correct_count": spa["outcome_correct_count"],
                "feasible_task_success_rate": spa["feasible_task_success_rate"],
                "feasible_task_success_count": spa["feasible_task_success_count"],
                "goal_coverage_micro": spa["goal_coverage_micro"],
                "goal_coverage_macro": spa["goal_coverage_macro"],
                "initial_goal_coverage_micro": spa["initial_goal_coverage_micro"],
                "terminal_goal_coverage_micro": spa["goal_coverage_micro"],
                "delta_goal_coverage_micro": spa["delta_goal_coverage_micro"],
                "initial_goal_coverage_macro": spa["initial_goal_coverage_macro"],
                "terminal_goal_coverage_macro": spa["goal_coverage_macro"],
                "delta_goal_coverage_macro": spa["delta_goal_coverage_macro"],
                "false_completion_rate": spa["false_completion_rate"],
                "false_completion_count": spa["false_completion_count"],
                "declared_completion_count": spa["declared_completion_count"],
                "physical_plan_found_rate": spa["physical_plan_found_rate"],
                "physical_plan_found_count": spa["physical_plan_found_count"],
                "average_raw_vlm_requests": spa["raw_vlm_requests"]["mean"],
                "average_high_level_replans": spa["high_level_replans"]["mean"],
                "action_sequence_generation_rate": spa["action_sequence_generation_rate"],
                "nonempty_action_sequence_rate": spa["nonempty_action_sequence_rate"],
                "val_valid_plan_rate": spa["val_valid_plan_rate"],
                "execution_ready_plan_rate": spa["execution_ready_plan_rate"],
                "physical_execution_unconditional": spa["physical_execution_rate_unconditional"],
                "nonempty_plan_execution_completed_rate": spa["nonempty_plan_execution_completed_rate"],
                "generated_goal_satisfaction_rate": spa["generated_goal_satisfaction_rate"],
                "generated_goal_evaluation_coverage": spa["generated_goal_evaluation_coverage"],
                "benchmark_requirement_coverage": spa["benchmark_requirement_coverage_micro"],
                "generated_goal_atom_coverage": spa["generated_goal_atom_coverage_micro"],
                "feasibility_accuracy": spa["feasibility_confusion"]["accuracy"],
                "feasibility_coverage": spa["feasibility_confusion"]["decision_coverage"],
                "average_fm_calls": spa["fm_calls"]["mean"],
                "average_cp_calls": spa["cp_calls"]["mean"],
                "average_plan_length": spa["plan_length"]["mean"],
            })
    write_csv_dicts(output_root / f"{file_prefix}by_domain_protocol.csv", dom_proto_flat)

    # By Variant
    variants = sorted({(r["domain"], r["variant"]) for r in completed_rows})
    variant_flat = []
    for dom, var in variants:
        vrows = [r for r in completed_rows if r["domain"] == dom and r["variant"] == var]
        va = compute_group_aggregates(vrows, group_label=f"{dom}__{var}")
        first_r = vrows[0]
        variant_flat.append({
            "domain": dom,
            "variant": var,
            "ground_truth_feasible": first_r.get("ground_truth_feasible"),
            "total_runs": va["total_runs"],
            "terminal_status": first_r.get("terminal_status"),
            "outcome_correct": va["outcome_correct_count"] > 0,
            "outcome_correct_rate": va["outcome_correct_rate"],
            "feasible_task_success": va["feasible_task_success_count"] > 0 if va["feasible_runs"] > 0 else None,
            "feasible_task_success_rate": va["feasible_task_success_rate"] if va["feasible_runs"] > 0 else None,
            "goal_requirements_passed": va["goal_requirements_passed_feasible"] if va["feasible_runs"] > 0 else first_r.get("goal_requirements_passed"),
            "goal_requirements_total": va["goal_requirements_total_feasible"] if va["feasible_runs"] > 0 else first_r.get("goal_requirements_total"),
            "goal_coverage": va["goal_coverage_micro"] if va["feasible_runs"] > 0 else None,
            "initial_goal_requirements_passed": va["initial_goal_requirements_passed_feasible"] if va["feasible_runs"] > 0 else first_r.get("initial_requirements_passed"),
            "initial_goal_coverage": va["initial_goal_coverage_micro"] if va["feasible_runs"] > 0 else None,
            "terminal_goal_coverage": va["goal_coverage_micro"] if va["feasible_runs"] > 0 else None,
            "delta_goal_coverage": va["delta_goal_coverage_micro"] if va["feasible_runs"] > 0 else None,
            "declared_completion": va["declared_completion_count"] > 0,
            "false_completion": va["false_completion_count"] > 0,
            "any_symbolic_plan": va["action_sequence_generation_count"] > 0,
            "any_nonempty_symbolic_plan": va["nonempty_action_sequence_count"] > 0,
            "val_valid_plan": va["val_valid_plan_count"] > 0,
            "identity_success": sum(1 for r in vrows if r.get("identity_success")) > 0,
            "physical_plan_found": va["physical_plan_found_count"] > 0,
            "refinement_success": sum(1 for r in vrows if r.get("refinement_success")) > 0,
            "execution_attempted": va["execution_attempted_count"] > 0,
            "physical_actions_attempted": sum(r.get("physical_actions_attempted", 0) for r in vrows),
            "physical_actions_succeeded": sum(r.get("physical_actions_succeeded", 0) for r in vrows),
            "nonempty_plan_execution_completed": va["nonempty_plan_execution_completed_count"] > 0,
            "hidden_task_success": va["task_success_count"] > 0,
            "raw_vlm_requests": va["raw_vlm_requests"]["mean"],
            "object_calls": sum(r.get("object_calls", 0) for r in vrows),
            "initial_state_calls": sum(r.get("initial_state_calls", 0) for r in vrows),
            "goal_calls": sum(r.get("goal_calls", 0) for r in vrows),
            "cp_calls": sum(r.get("cp_calls", 0) for r in vrows),
            "high_level_replans": va["high_level_replans"]["mean"],
            "runtime_seconds": va["end_to_end_seconds"]["mean"],
            "actual_selected_plan_length": first_r.get("actual_selected_plan_length"),
            "actual_selected_action_sequence": first_r.get("actual_selected_action_sequence"),
            "terminal_failure_cause": first_r.get("terminal_failure_cause"),
        })
    write_csv_dicts(output_root / f"{file_prefix}by_variant.csv", variant_flat)

    # Stage Funnel
    funnel = compute_stage_funnel(completed_rows)
    funnel_flat = []
    for s in funnel:
        funnel_flat.append({
            "stage_id": s["stage_id"],
            "stage_name": s["stage_name"],
            "count": s["count"],
            "total_runs": s["total_runs"],
            "unconditional_rate": s["unconditional_rate"],
            "unconditional_ci95_low": s["unconditional_ci95"][0],
            "unconditional_ci95_high": s["unconditional_ci95"][1],
            "previous_stage_count": s["previous_stage_count"],
            "conditional_conversion_rate": s["conditional_conversion_rate"],
        })
    write_csv_dicts(output_root / f"{file_prefix}stage_funnel.csv", funnel_flat)
    write_csv_dicts(output_root / f"{file_prefix}sequence_funnel.csv", funnel_flat)

    # Feasibility Metrics & Confusion Matrix
    confusion = overall["feasibility_confusion"]
    (output_root / f"{file_prefix}feasibility_confusion.json").write_text(json.dumps(confusion, indent=2, sort_keys=True), encoding="utf-8")
    feas_metrics_flat = [{
        "evaluated_runs": confusion["evaluated_runs"],
        "total_scheduled_runs": confusion["total_scheduled_runs"],
        "decision_coverage": confusion["decision_coverage"],
        "true_positives": confusion["true_positives"],
        "false_positives": confusion["false_positives"],
        "true_negatives": confusion["true_negatives"],
        "false_negatives": confusion["false_negatives"],
        "accuracy": confusion["accuracy"],
        "accuracy_ci95_low": confusion["accuracy_wilson_ci95"][0],
        "accuracy_ci95_high": confusion["accuracy_wilson_ci95"][1],
        "precision": confusion["precision"],
        "recall": confusion["recall"],
        "specificity": confusion["specificity"],
        "balanced_accuracy": confusion["balanced_accuracy"],
        "f1_score": confusion["f1_score"],
        "false_infeasible_rate": confusion["false_infeasible_rate"],
        "false_feasible_rate": confusion["false_feasible_rate"],
    }]
    write_csv_dicts(output_root / f"{file_prefix}feasibility_metrics.csv", feas_metrics_flat)

    # Causal failure breakdown
    causal_counts = Counter(r["causal_category"] for r in completed_rows)
    causal_flat = [{"causal_category": cat, "count": cnt, "percentage": cnt / len(completed_rows) if completed_rows else 0.0} for cat, cnt in causal_counts.most_common()]
    write_csv_dicts(output_root / f"{file_prefix}failure_breakdown.csv", causal_flat)

    # Audits
    bm_audit = audit_benchmark_failures(completed_rows, results_root)
    write_csv_dicts(output_root / f"{file_prefix}benchmark_failure_audit.csv", bm_audit)

    id_audit = audit_identity_failures(completed_rows, results_root)
    write_csv_dicts(output_root / f"{file_prefix}identity_failures.csv", id_audit)

    ref_audit = audit_refinement_failures(completed_rows, results_root)
    write_csv_dicts(output_root / f"{file_prefix}refinement_failures.csv", ref_audit)

    gge_audit = audit_generated_goals(completed_rows, results_root)
    write_csv_dicts(output_root / f"{file_prefix}generated_goal_audit.csv", gge_audit)

    pred_infeas_causes = audit_predicted_infeasible_causes(completed_rows)
    write_csv_dicts(output_root / f"{file_prefix}predicted_infeasible_cause.csv", pred_infeas_causes)

    # Action sequence extraction
    act_index, act_sequences = extract_action_sequence_artifacts(completed_rows)
    write_csv_dicts(output_root / f"{file_prefix}action_sequence_index.csv", act_index)
    with (output_root / f"{file_prefix}action_sequences.jsonl").open("w", encoding="utf-8") as f:
        for seq in act_sequences:
            f.write(json.dumps(seq, sort_keys=True) + "\n")

    # Representative action sequences
    rep_sequences = extract_representative_action_sequences(audited_rows)
    (output_root / f"{file_prefix}representative_sequences.json").write_text(
        json.dumps(rep_sequences, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Subgoal definitions JSON (Part 11)
    from .evaluation.subgoals import CANONICAL_SUBGOAL_COUNTS
    subgoal_definitions = {
        "kitchen": {
            "canonical_subgoal_count": CANONICAL_SUBGOAL_COUNTS["kitchen"],
            "description": "12 canonical terminal manipulation outcomes: 8 coffee subgoals (on serving support, water delivered, coffee delivered, stirred for 2 vessels) and 4 soup subgoals (on serving support, soup utensil contained for 2 vessels).",
            "categories": {
                "coffee_vessel_placement": 2,
                "coffee_water_delivery": 2,
                "coffee_powder_delivery": 2,
                "coffee_stirred": 2,
                "soup_vessel_placement": 2,
                "soup_utensil_contained": 2,
            },
            "excluded_passive_structural": [
                "two distinct coffee vessels exist",
                "two distinct soup vessels exist",
                "coffee and soup vessel groups disjoint",
                "no required object held",
            ],
        },
        "living_room": {
            "canonical_subgoal_count": CANONICAL_SUBGOAL_COUNTS["living_room"],
            "description": "5 canonical terminal manipulation placement subgoals: left cup ON left table, left saucer ON left table, right cup ON right table, right saucer ON right table, remote ON shared coffee table.",
            "subgoals": [
                "(cup_left, ON, personal_table_left)",
                "(saucer_left, ON, personal_table_left)",
                "(cup_right, ON, personal_table_right)",
                "(saucer_right, ON, personal_table_right)",
                "(remote, ON, coffee_table_shared)",
            ],
            "excluded_passive_structural": [
                "required payloads present",
                "required supports present",
                "no payload held",
            ],
        },
        "workshop": {
            "canonical_subgoal_count": CANONICAL_SUBGOAL_COUNTS["workshop"],
            "description": "3 canonical terminal manipulation completion subgoals: compatible screw inserted in target, joint fastened / fully driven, selected driver placed safely on main workbench.",
            "subgoals": [
                "(compatible_screw, INSERTED_IN, repair_joint)",
                "(repair_joint, FASTENED, true)",
                "(selected_driver, ON, main_workbench)",
            ],
            "excluded_passive_structural": [
                "driver/screw compatibility checks",
                "procedural first-driver constraint",
                "no object held",
            ],
        },
    }
    (output_root / f"{file_prefix}subgoal_definitions.json").write_text(
        json.dumps(subgoal_definitions, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Initial and terminal subgoal evaluations jsonl (Part 11)
    with (output_root / f"{file_prefix}initial_subgoal_evaluations.jsonl").open("w", encoding="utf-8") as f:
        for r in completed_rows:
            f.write(json.dumps({
                "run_id": r.get("run_id"),
                "domain": r.get("domain"),
                "variant": r.get("variant"),
                "observation_protocol": r.get("observation_protocol"),
                "repeat_index": r.get("repeat_index"),
                "ground_truth_feasible": r.get("ground_truth_feasible"),
                "initial_subgoals_passed": r.get("initial_subgoals_passed"),
                "initial_subgoals_total": r.get("initial_subgoals_total"),
                "initial_goal_coverage": r.get("initial_goal_coverage"),
                "initial_subgoal_eval": r.get("initial_subgoal_eval"),
            }, sort_keys=True) + "\n")

    with (output_root / f"{file_prefix}terminal_subgoal_evaluations.jsonl").open("w", encoding="utf-8") as f:
        for r in completed_rows:
            f.write(json.dumps({
                "run_id": r.get("run_id"),
                "domain": r.get("domain"),
                "variant": r.get("variant"),
                "observation_protocol": r.get("observation_protocol"),
                "repeat_index": r.get("repeat_index"),
                "ground_truth_feasible": r.get("ground_truth_feasible"),
                "terminal_subgoals_passed": r.get("terminal_subgoals_passed"),
                "terminal_subgoals_total": r.get("terminal_subgoals_total"),
                "terminal_goal_coverage": r.get("terminal_goal_coverage"),
                "delta_goal_coverage": r.get("delta_goal_coverage"),
                "terminal_subgoal_eval": r.get("terminal_subgoal_eval"),
            }, sort_keys=True) + "\n")

    # Metric definitions JSON
    definitions = {
        "outcome_correct": "Proportion of runs with correct outcome: actual task success for GT-feasible tasks, or clean symbolic rejection under bounded CP without unresolved execution/refinement/identity failures for GT-infeasible tasks.",
        "feasible_task_success": "Proportion of GT-feasible runs where the hidden benchmark goal was physically achieved (denominator: scheduled GT-feasible runs).",
        "goal_coverage": "Micro-average fraction of canonical GT terminal manipulation subgoals satisfied in the physical state across all scheduled GT-feasible runs (excludes passive structural prerequisites).",
        "initial_goal_coverage": "Micro-average fraction of canonical GT terminal manipulation subgoals satisfied at initial physical state across all scheduled GT-feasible runs.",
        "delta_goal_coverage": "Change in goal coverage from initial state to terminal state (TerminalGoalCoverage - InitialGoalCoverage).",
        "false_completion": "Proportion of declared-complete runs where hidden benchmark evaluation failed (denominator: runs where runtime declared completion).",
        "physical_plan_found": "Proportion of GT-feasible runs yielding at least one non-empty, VAL-valid, identity-resolved, and refinement-passed physical action plan (zero-step plans excluded).",
        "raw_vlm_requests": "Total foundation model calls made per run across object estimation, initial state, goal state, and corrective planning.",
        "high_level_replans": "Number of corrective planning replans invoked (bounded in [0, 3]).",
        "task_success_rate": "Proportion of runs achieving the hidden benchmark goal in physical simulation.",
        "action_sequence_generation_rate": "Proportion of runs where Fast Downward generated at least one parseable symbolic action sequence.",
        "val_valid_plan_rate": "Proportion of runs where VAL validated at least one symbolic plan.",
        "execution_ready_plan_rate": "Proportion of runs producing a plan that succeeded in both entity resolution and geometric refinement.",
        "physical_execution_unconditional": "Proportion of runs where physical simulation executed the complete plan without failure.",
        "nonempty_plan_execution_completed_rate": "Proportion of runs where physical simulation executed a non-empty plan without failure.",
        "generated_goal_satisfaction_rate": "Proportion of evaluated generated goals physically satisfied at the terminal state.",
        "generated_goal_evaluation_coverage": "Proportion of runs where generated goal evaluation was executed.",
        "benchmark_requirement_coverage_micro": "Fraction of total hidden benchmark requirements satisfied in the physical state across all runs.",
        "generated_goal_atom_coverage_micro": "Fraction of total generated PDDL goal atoms physically satisfied across all evaluated runs.",
        "feasibility_accuracy": "Binary accuracy of predicted infeasibility against ground truth on covered decisions.",
        "feasibility_decision_coverage": "Proportion of runs with a definitive feasibility evaluation.",
    }
    (output_root / f"{file_prefix}metric_definitions.json").write_text(json.dumps(definitions, indent=2, sort_keys=True), encoding="utf-8")

    try:
        import subprocess
        eval_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        eval_commit = "unknown"
    beh_commit = audited_rows[0].get("source_commit") if audited_rows else "2c3c402c25cbd87ac0e3873231d971c854524f11"

    # Audit summary
    audit_summary = {
        "results_root": str(results_root),
        "total_scheduled_runs": len(audited_rows),
        "completed_non_infrastructure_runs": len(completed_rows),
        "infrastructure_failures": len(audited_rows) - len(completed_rows),
        "source_commit": beh_commit,
        "behavior_commit": beh_commit,
        "evaluation_commit": eval_commit,
        "runs_with_symbolic_plan": sum(1 for r in completed_rows if r["symbolic_plan_found"]),
        "runs_with_nonempty_plan": sum(1 for r in completed_rows if r["symbolic_plan_nonempty"]),
        "runs_with_val_valid_plan": sum(1 for r in completed_rows if r["val_plan_valid"]),
        "runs_with_physical_plan_found": sum(1 for r in completed_rows if r["physical_plan_found"]),
        "runs_with_refinement_success": sum(1 for r in completed_rows if r["refinement_success"]),
        "runs_with_execution_success": sum(1 for r in completed_rows if r["execution_success"]),
        "runs_with_nonempty_execution_success": sum(1 for r in completed_rows if r["nonempty_plan_execution_completed"]),
        "runs_with_benchmark_success": sum(1 for r in completed_rows if r["actual_task_success"]),
        "outcome_correct_runs": sum(1 for r in completed_rows if r["outcome_correct"]),
        "declared_completion_runs": sum(1 for r in completed_rows if r["declared_completion"]),
        "false_completion_runs": sum(1 for r in completed_rows if r["false_completion"]),
        "generated_goals_evaluated": len(bm_audit),
        "generated_goals_satisfied": sum(1 for r in completed_rows if r.get("generated_goal_satisfied") is True),
        "causal_failure_distribution": dict(causal_counts),
        "representative_sequences": rep_sequences,
    }
    (output_root / f"{file_prefix}audit.json").write_text(json.dumps(audit_summary, indent=2, sort_keys=True), encoding="utf-8")

    # Generate LaTeX main table
    generate_manuscript_main_table(
        overall=overall,
        by_domain=domain_aggregates,
        output_dir=output_root,
        file_prefix=file_prefix,
    )

    # Generate full legacy tables as well
    generate_latex_tables(
        overall=overall,
        by_domain=domain_aggregates,
        by_protocol=protocol_aggregates,
        funnel=funnel,
        confusion=confusion,
        causal_categories=causal_counts,
        output_dir=output_root,
    )

    return overall
