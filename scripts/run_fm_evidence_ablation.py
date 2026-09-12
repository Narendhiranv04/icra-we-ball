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
    COMPONENT_MASKS, evidence_channels_present, mask_specification,
    specification_from_archived_response,
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


def _observed_graph_for(*, domain: str, variant: str, trial: str,
                        raw_path: Path, output_root: Path):
    """Phase 1: one unmodified pipeline run, reused by all seven conditions.

    The observed scene graph is a property of the scene, not of the mask, so
    building it once is both correct and what makes the seven conditions
    genuinely comparable: they differ only in the evidence the grounder is
    allowed to read from the *same* observation.
    """
    from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedSceneGraph
    from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline

    base_root = output_root / trial / "_phase1"
    graph_path = base_root / domain / variant / "vlm" / "observed_scene_graph.json"
    if not graph_path.is_file():
        try:
            run_pipeline(domain=domain, variant=variant, mode="vlm",
                         specification_json=raw_path, output_root=base_root,
                         dry_run=True)
        except Exception:
            pass  # a failed run can still have written the observation
    if not graph_path.is_file():
        return None
    return ObservedSceneGraph.from_dict(json.loads(graph_path.read_text()))


def _validate_roles(specification, result, graph) -> dict[str, Any]:
    """Score chosen objects against FULL evidence, whatever the mask allowed.

    This is the point of the ablation: a condition may ground happily on partial
    evidence and still have chosen the wrong object.  Scoring against the mask
    would make every condition trivially correct.
    """
    from mujoco_scenes.functional_tamp_pipeline.grounding import (
        check_semantic_role_compatibility,
    )
    total = selected = confirmed = refuted = unknown = 0
    details = []
    assignment = result.assignment or {}
    for name, role in specification.nodes.items():
        ids = assignment.get(name)
        ids = [] if ids is None else ([ids] if isinstance(ids, str) else list(ids))
        for index in range(role.minimum_count):
            total += 1
            object_id = ids[index] if index < len(ids) else None
            node = graph.get_node(object_id) if object_id else None
            if node is None:
                sem = "FALSE"
            else:
                selected += 1
                sem = (check_semantic_role_compatibility(node, role.semantic_categories)[0]
                       if role.semantic_categories else "TRUE")
            confirmed += sem == "TRUE"
            refuted += sem == "FALSE"
            unknown += sem == "UNKNOWN"
            details.append({"role": name, "slot_index": index,
                            "selected_object_id": object_id, "semantic_status": sem})
    # Real perception returns UNKNOWN where an oracle returned TRUE, so the GT
    # ablation's "every slot must be TRUE" would mark the unablated pipeline
    # itself as failing.  UNKNOWN is therefore reported as its own quantity and
    # the success test is "nothing was refuted", never "UNKNOWN counts as TRUE":
    # a slot the evidence contradicts still fails.
    return {"role_slots_total": total, "role_slots_selected": selected,
            "role_slots_gt_confirmed": confirmed,
            "role_slots_gt_refuted": refuted,
            "role_slots_gt_unknown": unknown,
            "role_slots_gt_valid": confirmed,
            "role_slot_gt_valid_pct": round(100 * confirmed / total, 2) if total else None,
            "exact_role_gt_success": bool(
                result.complete and total and selected == total and refuted == 0),
            "exact_role_gt_confirmed_success": bool(
                result.complete and total and selected == total == confirmed),
            "per_role_gt_validation": details}


def _validate_bindings(specification, result, graph) -> dict[str, Any]:
    """Score chosen pairings against the FULL binary evidence in the scene."""
    total = selected = valid = 0
    details = []
    for group in specification.operation_groups:
        bindings = list((result.operation_bindings or {}).get(group.id, []))
        for index in range(group.required_target_count):
            total += 1
            binding = bindings[index] if index < len(bindings) else None
            checks = []
            if binding:
                selected += 1
                tool_id = binding.get("target_id") if False else binding.get("tool_id")
                target_id = binding.get("target_id")
                context_id = (binding.get("context") or {}).get(group.context_role)
                # physical_preconditions overrides required_relations inside
                # the grounder.  Checking only required_relations left every
                # FM group that declares preconditions with an empty check
                # list, which then scored as invalid for having nothing to
                # verify -- the validator failing, not the binding.
                if group.physical_preconditions:
                    roles = {group.tool_role: tool_id, group.target_role: target_id}
                    if group.context_role and context_id is not None:
                        roles[group.context_role] = context_id
                    triples = [(roles[s_role], predicate, roles[o_role])
                               for s_role, predicate, o_role in group.physical_preconditions
                               if s_role in roles and o_role in roles]
                else:
                    triples = [(tool_id, predicate, target_id)
                               for predicate in group.required_relations]
                    if group.context_role and context_id is not None:
                        triples += [(tool_id, predicate, context_id)
                                    for predicate in group.context_relations]
                for subject, predicate, obj in triples:
                    relation = graph.get_relation(predicate, subject, obj)
                    checks.append({"predicate": predicate, "subject_id": subject,
                                   "object_id": obj,
                                   "status": relation.status if relation else "UNKNOWN"})
            # A group that declares no binary evidence at all cannot be refuted
            # by binary evidence; it is vacuously satisfied, exactly as the
            # grounder treats it (no FALSE and no UNKNOWN means TRUE).
            ok = bool(binding) and not any(c["status"] == "FALSE" for c in checks)
            refuted_here = any(c["status"] == "FALSE" for c in checks)
            valid += ok
            details.append({"group": group.id, "binding_index": index,
                            "binding": binding, "relation_checks": checks, "gt_valid": ok})
    return {"operation_bindings_total": total, "operation_bindings_selected": selected,
            "operation_bindings_gt_valid": valid,
            "operation_binding_gt_valid_pct": round(100 * valid / total, 2) if total else None,
            "exact_operation_binding_gt_success": bool(
                result.complete and total and selected == total == valid),
            "per_binding_gt_validation": details}


def _norm_assignment(value) -> dict[str, list[str]]:
    out = {}
    for role, ids in sorted((value or {}).items()):
        ids = [] if ids is None else ([ids] if isinstance(ids, str) else list(ids))
        out[role] = sorted(str(x) for x in ids)
    return out


def _norm_bindings(value) -> dict[str, list[str]]:
    out = {}
    for group, bindings in sorted((value or {}).items()):
        out[group] = sorted(json.dumps({
            "tool_id": str(b.get("tool_id")), "target_id": str(b.get("target_id")),
            "context": {str(k): str(v) for k, v in sorted((b.get("context") or {}).items())},
        }, sort_keys=True) for b in bindings)
    return out


def evaluate_one(*, domain: str, variant: str, trial: str, condition: str,
                 raw_path: Path, output_root: Path, graph) -> dict[str, Any]:
    from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
    from mujoco_scenes.fm_evidence_ablation import write_masked_specification

    row: dict[str, Any] = {
        "domain": domain, "variant": variant, "trial": trial, "condition": condition,
        "enabled_evidence_components": list(COMPONENT_MASKS[condition]),
        "gt_feasible": variant in FEASIBLE[domain],
        "archived_response": str(raw_path),
    }
    started = time.perf_counter()
    try:
        specification = specification_from_archived_response(raw_path, domain, TASKS[domain])
    except Exception as exc:
        # The FM response itself was unusable.  Identical across all seven
        # masks, so it is recorded rather than dropped: dropping it would let
        # different conditions be compared over different variant sets.
        row.update(fm_spec_available=False, grounding_status="FM_SPEC_UNUSABLE",
                   grounding_complete=False,
                   failure_reason=f"{type(exc).__name__}: {exc}",
                   runtime_seconds=round(time.perf_counter() - started, 3))
        return row
    row["fm_spec_available"] = True
    row["fm_channels_present"] = evidence_channels_present(specification)

    masked = mask_specification(specification, condition)
    row["masked_channels_present"] = evidence_channels_present(masked)
    write_masked_specification(
        specification, condition,
        output_root / trial / condition / "_masked_specs" / domain / f"{variant}.json")

    try:
        result = ground_graph(masked, graph, {"search_exhausted": True})
    except Exception as exc:
        row.update(grounding_status="GROUNDING_EXCEPTION", grounding_complete=False,
                   failure_reason=f"{type(exc).__name__}: {exc}",
                   runtime_seconds=round(time.perf_counter() - started, 3))
        return row

    row.update(grounding_status=result.status,
               grounding_complete=bool(result.complete),
               selected_assignment=result.assignment,
               selected_operation_bindings={k: list(v) for k, v
                                            in (result.operation_bindings or {}).items()},
               **_validate_roles(masked, result, graph),
               **_validate_bindings(specification, result, graph))
    row["infeasible_rejected"] = (not row["gt_feasible"]) and not result.complete
    row["false_grounding"] = (not row["gt_feasible"]) and bool(result.complete)
    row["runtime_seconds"] = round(time.perf_counter() - started, 3)
    return row


def compare_to_full(rows: list[dict[str, Any]]) -> None:
    """Attach agreement with the `full` condition on the same trial and variant."""
    reference = {(r["trial"], r["domain"], r["variant"]): r
                 for r in rows if r["condition"] == "full"}
    for row in rows:
        base = reference.get((row["trial"], row["domain"], row["variant"]))
        if base is None or not row.get("fm_spec_available"):
            continue
        row["exact_role_match_to_full"] = (
            _norm_assignment(row.get("selected_assignment"))
            == _norm_assignment(base.get("selected_assignment")))
        row["exact_operation_binding_match_to_full"] = (
            _norm_bindings(row.get("selected_operation_bindings"))
            == _norm_bindings(base.get("selected_operation_bindings")))
        row["outcome_matches_full"] = (
            row.get("grounding_complete") == base.get("grounding_complete"))


def summarize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror the GT ablation's columns so the two tables read side by side."""
    rows = list(rows)
    out = []
    for domain in (*DOMAINS, "all"):
        source = rows if domain == "all" else [r for r in rows if r["domain"] == domain]
        for condition in COMPONENT_MASKS:
            group = [r for r in source if r["condition"] == condition]
            if not group:
                continue
            usable = [r for r in group if r.get("fm_spec_available")]
            feasible = [r for r in usable if r["gt_feasible"]]
            infeasible = [r for r in usable if not r["gt_feasible"]]

            def pct(subset, key):
                subset = [r for r in subset if r.get(key) is not None]
                return (round(100 * sum(bool(r[key]) for r in subset) / len(subset), 2)
                        if subset else None)

            def ratio(subset, num, den):
                total = sum(r.get(den) or 0 for r in subset)
                return (round(100 * sum(r.get(num) or 0 for r in subset) / total, 2)
                        if total else None)
            out.append({
                "domain": domain, "condition": condition,
                "enabled_evidence_components": "+".join(COMPONENT_MASKS[condition]),
                "trials": len(group),
                "fm_spec_usable": len(usable),
                "feasible_trials": len(feasible),
                "infeasible_trials": len(infeasible),
                "grounding_completion_pct": pct(feasible, "grounding_complete"),
                "role_slot_gt_valid_pct": ratio(feasible, "role_slots_gt_valid", "role_slots_total"),
                "role_gt_success_pct": pct(feasible, "exact_role_gt_success"),
                "role_match_full_pct": pct(feasible, "exact_role_match_to_full"),
                "operation_binding_gt_valid_pct": ratio(
                    feasible, "operation_bindings_gt_valid", "operation_bindings_total"),
                "pair_gt_success_pct": pct(feasible, "exact_operation_binding_gt_success"),
                "pair_match_full_pct": pct(feasible, "exact_operation_binding_match_to_full"),
                "infeasible_rejection_pct": pct(infeasible, "infeasible_rejected"),
                "false_grounding_count": sum(bool(r.get("false_grounding")) for r in infeasible),
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

    conditions = (tuple(COMPONENT_MASKS) if args.conditions == "all"
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
                graph = _observed_graph_for(domain=domain, variant=variant, trial=trial,
                                            raw_path=raw, output_root=args.output_root)
                if graph is None:
                    missing.append(f"{trial}/{domain}/{variant} (no observed graph)")
                    continue
                for condition in conditions:
                    row = evaluate_one(domain=domain, variant=variant, trial=trial,
                                       condition=condition, raw_path=raw,
                                       output_root=args.output_root, graph=graph)
                    rows.append(row)
                    print(f"[fm-ablation] {trial} {domain}/{variant} {condition:14s} "
                          f"status={row.get('grounding_status')} "
                          f"complete={row.get('grounding_complete')} "
                          f"role_gt={row.get('exact_role_gt_success')} "
                          f"pair_gt={row.get('exact_operation_binding_gt_success')}",
                          flush=True)
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
