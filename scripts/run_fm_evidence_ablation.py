#!/usr/bin/env python3
"""Sweep the seven evidence masks over the FM-authored functional graph.

A shadow of the live benchmark, in two phases, changing no pipeline code:

  Phase 1 (once per trial x variant)
      archived FM response -> G_F -> the unmodified run_pipeline
      -> persists observed_scene_graph.json and the unablated result

  Phase 2 (per trial x variant x condition)
      masked G_F + that same observed graph -> ground_graph -> plan -> score

Masking is applied at the *grounding* boundary, which is where the GT ablation
applied it too.  The earlier and more obvious design -- feed a masked G_F back
through run_pipeline's replay path -- cannot work and should not: that path
first calls the task-interface validator, whose job is to reject a malformed FM
contract, and an ablated graph is malformed by exactly that standard ("operation
group has empty required_relations").  Routing ablations through it would have
measured the validator rather than the grounder, and every mask that removed a
channel would have failed identically for a reason unrelated to evidence.

Makes no FM call.  Reads archived responses, so the whole sweep is deterministic
and repeatable, and running it cannot disturb or compete with a live run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("TAMP_FM_SCHEMA_VERSION", "3")

from mujoco_scenes.evaluation_outcome import (  # noqa: E402
    completion_claimed, is_false_completion, outcome_is_correct, trial_is_scorable,
)
from mujoco_scenes.fm_evidence_ablation import (  # noqa: E402
    COMPONENT_MASKS, CONDITION_LABELS, CONDITION_ORDER,
)

DOMAINS = {
    "kitchen": [f"K{i}" for i in range(1, 13)],
    "living_room": [f"L{i}" for i in range(1, 11)],
    "workshop": [f"W{i}" for i in range(1, 11)],
}
FEASIBLE = {
    "kitchen": {f"K{i}" for i in range(1, 7)},
    "living_room": {f"L{i}" for i in range(1, 7)},
    "workshop": {f"W{i}" for i in range(1, 9)},
}
TASKS = {
    "kitchen": "Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil.",
    "living_room": "Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people.",
    "workshop": "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench.",
}


def _archived_response(archive: Path, trial: str, domain: str, variant: str) -> Path | None:
    for candidate in (
        archive / trial / domain / variant / "vlm" / "fm_diagnostics" / "fm_call_001.json",
        archive / trial / domain / variant / "vlm" / "raw_vlm_response.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _full_task_satisfied(run_dir: Path, domain: str, variant: str) -> bool:
    """Reuse the benchmark's own independent evaluation; never re-derive it."""
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
        full_task_coverage,
    )
    try:
        # (satisfied_count, goal_count, coverage, full_task_satisfied)
        return bool(full_task_coverage(domain, run_dir)[3])
    except Exception:
        return False


def _full_task_satisfied(run_dir: Path, domain: str) -> tuple[bool, float]:
    """Reuse the benchmark's own independent evaluation; never re-derive it."""
    from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import (
        full_task_coverage,
    )
    try:
        satisfied_count, goal_count, coverage, satisfied = full_task_coverage(
            domain, run_dir)
        return bool(satisfied), float(coverage)
    except Exception:
        return False, 0.0


def evaluate_one(*, domain: str, variant: str, trial: str, condition: str,
                 raw_path: Path, output_root: Path) -> dict[str, Any]:
    """One full pipeline run with `condition`'s evidence mask in force."""
    from mujoco_scenes.fm_ablation_shadow import evidence_masked
    from mujoco_scenes.functional_tamp_pipeline.models import PipelineResult
    from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline

    run_root = output_root / trial / condition
    run_dir = run_root / domain / variant / "vlm"
    row: dict[str, Any] = {
        "domain": domain, "variant": variant, "trial": trial, "condition": condition,
        "enabled_evidence_components": list(COMPONENT_MASKS[condition]),
        "gt_feasible": variant in FEASIBLE[domain],
        "archived_response": str(raw_path),
    }
    started = time.perf_counter()
    with evidence_masked(condition) as stats:
        try:
            result = run_pipeline(domain=domain, variant=variant, mode="vlm",
                                  specification_json=raw_path,
                                  output_root=run_root, dry_run=True)
        except Exception as exc:
            result = PipelineResult(domain=domain, variant=variant, mode="vlm",
                                    status="PIPELINE_EXCEPTION",
                                    failure_reason=f"{type(exc).__name__}: {exc}")

    # A shadow that silently failed to patch would report the unablated pipeline
    # seven times over and look like a clean result, so the mask having been
    # applied is recorded per row rather than assumed.
    row["condition_label"] = CONDITION_LABELS[condition]
    row["mask_applied"] = bool(stats and stats.get("groundings"))
    row["groundings_masked"] = int(stats.get("groundings", 0)) if stats else 0
    row["observed_evidence_after_mask"] = (stats or {}).get("evidence_after_mask")

    satisfied, coverage = _full_task_satisfied(run_dir, domain)
    row.update(
        pipeline_status=result.status,
        failure_reason=result.failure_reason,
        failure_category=getattr(result, "failure_category", None),
        canonicalization_succeeded=getattr(result, "canonicalization_succeeded", None),
        runtime_contract_complete=getattr(result, "functional_spec_complete", None),
        candidate_plan_length=len(result.candidate_plan or ()),
        gt_full_task_satisfied=satisfied,
        full_task_coverage=coverage,
        completion_claimed=completion_claimed(result.status),
        false_completion=is_false_completion(pipeline_status=result.status,
                                             gt_full_task_satisfied=satisfied),
        outcome_correct=outcome_is_correct(
            gt_feasible=row["gt_feasible"], gt_full_task_satisfied=satisfied,
            pipeline_status=result.status,
            runtime_contract_complete=getattr(result, "functional_spec_complete", None)),
        runtime_seconds=round(time.perf_counter() - started, 3),
    )
    return row


def compare_to_full(rows: list[dict[str, Any]]) -> None:
    """Attach agreement with the `full` condition on the same trial and variant."""
    reference = {(r["trial"], r["domain"], r["variant"]): r
                 for r in rows if r["condition"] == "full"}
    for row in rows:
        base = reference.get((row["trial"], row["domain"], row["variant"]))
        if base is None:
            continue
        row["full_reference_status"] = base.get("pipeline_status")
        row["status_matches_full"] = row.get("pipeline_status") == base.get("pipeline_status")
        row["outcome_matches_full"] = row.get("outcome_correct") == base.get("outcome_correct")


def summarize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """End-to-end pipeline outcomes per condition, scored by the live scorer."""
    rows = list(rows)
    out = []
    for domain in (*DOMAINS, "all"):
        source = rows if domain == "all" else [r for r in rows if r["domain"] == domain]
        for condition in CONDITION_ORDER:
            group = [r for r in source if r["condition"] == condition]
            if not group:
                continue
            scorable = [r for r in group if trial_is_scorable(r.get("pipeline_status"))]
            feasible = [r for r in scorable if r["gt_feasible"]]
            infeasible = [r for r in scorable if not r["gt_feasible"]]

            def pct(subset, key):
                subset = [r for r in subset if r.get(key) is not None]
                return (round(100 * sum(bool(r[key]) for r in subset) / len(subset), 2)
                        if subset else None)
            out.append({
                "domain": domain, "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "enabled_evidence_components": "+".join(COMPONENT_MASKS[condition]),
                "trials": len(group),
                "mask_applied_all": all(r.get("mask_applied") for r in group),
                "feasible_trials": len(feasible),
                "infeasible_trials": len(infeasible),
                "contract_complete_pct": pct(group, "runtime_contract_complete"),
                "feasible_success_pct": pct(feasible, "outcome_correct"),
                "infeasible_rejection_pct": pct(infeasible, "outcome_correct"),
                "overall_correct_pct": pct(scorable, "outcome_correct"),
                "completion_claimed_pct": pct(feasible, "completion_claimed"),
                "status_matches_full_pct": pct(group, "status_matches_full"),
                "mean_full_task_coverage": (
                    round(sum(r.get("full_task_coverage") or 0 for r in feasible)
                          / len(feasible), 4) if feasible else None),
                "false_completions": sum(bool(r.get("false_completion")) for r in group),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=REPO / "benchmark_reports/v3_qwen_distribution_3x32_20260910T053937",
                        help="Root holding trial_NN/<domain>/<variant>/vlm archived FM responses")
    parser.add_argument("--trials", default="trial_01,trial_02,trial_03")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--variants", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    conditions = (CONDITION_ORDER if args.conditions == "all"
                  else tuple(c.strip() for c in args.conditions.split(",") if c.strip()))
    unknown = set(conditions) - set(COMPONENT_MASKS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())
    trials = tuple(t.strip() for t in args.trials.split(",") if t.strip())
    wanted = ({v.strip() for v in args.variants.split(",")} if args.variants else None)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for trial in trials:
        for domain in domains:
            for variant in DOMAINS[domain]:
                if wanted and variant not in wanted:
                    continue
                raw = _archived_response(args.archive, trial, domain, variant)
                if raw is None:
                    missing.append(f"{trial}/{domain}/{variant}")
                    continue
                for condition in conditions:
                    row = evaluate_one(domain=domain, variant=variant, trial=trial,
                                       condition=condition, raw_path=raw,
                                       output_root=args.output_root)
                    rows.append(row)
                    print(f"[fm-ablation] {trial} {domain}/{variant} {condition:14s} "
                          f"status={row.get('pipeline_status'):32s} "
                          f"correct={str(row.get('outcome_correct')):5s} "
                          f"masked={row.get('mask_applied')}", flush=True)
                (args.output_root / "results.json").write_text(
                    json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")

    compare_to_full(rows)
    summary = summarize(rows)
    (args.output_root / "results.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_root / "summary.json").write_text(json.dumps({
        "kind": "FM_EVIDENCE_ABLATION",
        "archive": str(args.archive),
        "trials": list(trials),
        "conditions": list(conditions),
        "evaluation_count": len(rows),
        "missing_archived_responses": missing,
        "rows": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if missing:
        print(f"\n{len(missing)} archived response(s) missing: {missing[:5]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
