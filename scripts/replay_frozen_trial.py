#!/usr/bin/env python3
"""Replay exactly one frozen raw FM contract through the deterministic pipeline.

Zero FM calls: the specification comes from the archived raw response on disk.
Emits one JSON row describing every stage the pipeline reached.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, os.environ.get("TAMP_REPO", str(Path(__file__).resolve().parents[1])))

# Before anything else in the process.  Seeding lazily, when the detector is
# constructed, leaves everything that runs earlier unseeded -- including the
# inspection controller choosing where to look.  Two replays of the same
# archived contract disagreed on how many views supported an object (three
# against two), which is what a differently-posed capture looks like, not what a
# differently-scored image looks like.
from mujoco_scenes.determinism import enable_deterministic_inference  # noqa: E402
_DETERMINISM = enable_deterministic_inference()

CANONICAL_TASK_INSTRUCTIONS = {
    "kitchen": "Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil.",
    "living_room": "Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people.",
    "workshop": "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench.",
}
FEASIBLE = {
    "kitchen": {f"K{i}" for i in range(1, 7)},
    "living_room": {f"L{i}" for i in range(1, 7)},
    "workshop": {f"W{i}" for i in range(1, 9)},
}


def derive(raw, domain):
    """Recompute the deterministic front half (wire -> canonical -> compile)."""
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
        normalize_v3_live_document, validate_v3_live_contract,
        convert_v3_to_canonical_document)
    instruction = CANONICAL_TASK_INSTRUCTIONS.get(domain, "")
    out = {
        "strict_v3_valid": False, "strict_v3_error": None,
        "task_valid": False, "task_error": None,
        "graph_compiled": False, "compile_error": None,
        "canonical_roles": [], "unresolved_roles": [], "n_relations_compiled": 0,
        "n_operation_groups": 0, "n_disabled_operations": 0,
        "disabled_operation_codes": [], "unresolved_role_codes": [],
        "contract_complete": False, "contract_missing_reasons": None,
        "canonical_graph_emitted": False,
        "executable_contract_complete": False,
        "executable_contract_missing_reasons": None,
        "non_scene_resolvable_blockers": [],
        "canonicalization_status": None, "n_provisional_ops": 0,
        "n_task_causal": 0, "n_unresolved_required_relations": 0,
        "n_unresolved_required_operations": 0,
        "n_canonical_relations_in_doc": 0, "n_canonical_groups_in_doc": 0,
    }
    if raw is None:
        out["strict_v3_error"] = "NO_PARSEABLE_FM_CONTENT"
        return out
    try:
        normalized, _trace = normalize_v3_live_document(raw, domain=domain, task_instruction=instruction)
        validate_v3_live_contract(normalized, domain=None)
        out["strict_v3_valid"] = True
    except Exception as exc:
        out["strict_v3_error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        canonical = convert_v3_to_canonical_document(normalized, domain=domain, task_instruction=instruction)
        out["task_valid"] = True
        out["n_canonical_relations_in_doc"] = len(canonical.get("functional_relations", []))
        out["n_canonical_groups_in_doc"] = len(canonical.get("interaction_groups", []))
    except Exception as exc:
        out["task_error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
        graph = compile_candidate_graph(domain, instruction, canonical)
        meta = graph.metadata or {}
        trace = meta.get("canonicalization_trace", {}) or {}
        out["graph_compiled"] = bool(graph.nodes)
        out["canonical_roles"] = sorted(graph.nodes)
        out["unresolved_roles"] = [
            (r.get("code") or r.get("status")) for r in trace.get("unresolved_roles", [])]
        out["unresolved_role_codes"] = sorted({str(x) for x in out["unresolved_roles"]})
        out["n_relations_compiled"] = len(graph.relations or ())
        out["n_operation_groups"] = len(graph.operation_groups or ())
        out["n_disabled_operations"] = len(trace.get("disabled_groups", []))
        out["disabled_operation_codes"] = sorted({str(r.get("status")) for r in trace.get("disabled_groups", [])})
        out["n_provisional_ops"] = len(graph.provisional_operation_constraints or ())
        out["n_task_causal"] = len(graph.task_causal_relations or ())
        out["n_unresolved_required_relations"] = len(trace.get("unresolved_required_relations", []))
        out["n_unresolved_required_operations"] = len(trace.get("unresolved_required_operations", []))
        out["contract_complete"] = bool(meta.get("required_contract_complete"))
        out["contract_missing_reasons"] = [str(x)[:300] for x in (meta.get("contract_missing_reasons") or [])]
        # "canonical graph emitted" and "executable contract complete" are two
        # different facts and were previously reported under one name.
        out["canonical_graph_emitted"] = bool(graph.nodes)
        out["executable_contract_complete"] = bool(
            meta.get("online_executable_contract_complete", meta.get("required_contract_complete")))
        out["executable_contract_missing_reasons"] = [
            str(x)[:300] for x in (meta.get("executable_contract_missing_reasons") or [])]
        out["non_scene_resolvable_blockers"] = sorted(
            {str(x) for x in (meta.get("non_scene_resolvable_blockers") or ())})
        out["canonicalization_status"] = meta.get("canonicalization_status")
        out["role_counts"] = {n: [graph.nodes[n].minimum_count, graph.nodes[n].maximum_count,
                                  graph.nodes[n].binding_policy] for n in sorted(graph.nodes)}
        out["operation_summary"] = [
            {"id": g.id, "function": g.function, "tool": g.tool_role, "target": g.target_role,
             "ctx": g.context_role, "count": g.required_target_count, "policy": g.usage_policy}
            for g in (graph.operation_groups or ())]
    except Exception as exc:
        out["compile_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--raw", required=True, help="archived fm_call_001.json / raw_v3.json")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--row-out", required=True)
    args = ap.parse_args()

    os.environ["TAMP_FM_SCHEMA_VERSION"] = "3"
    raw_path = Path(args.raw)
    raw = None
    if raw_path.exists():
        try:
            data = json.loads(raw_path.read_text())
            raw = json.loads(data["content"]) if isinstance(data.get("content"), str) else data
        except Exception:
            raw = None

    row = {"domain": args.domain, "variant": args.variant, "trial": args.trial,
           "feasible": args.variant in FEASIBLE[args.domain]}
    t0 = time.perf_counter()
    try:
        row.update(derive(raw, args.domain))
    except Exception as exc:
        row["derive_exception"] = f"{type(exc).__name__}: {exc}"
        row.setdefault("strict_v3_valid", False)

    from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline
    from mujoco_scenes.functional_tamp_pipeline.models import PipelineResult
    out_root = Path(args.output_root)
    run_dir = out_root / args.domain / args.variant / "vlm"
    try:
        res = run_pipeline(domain=args.domain, variant=args.variant, mode="vlm",
                           specification_json=raw_path if raw_path.exists() else None,
                           output_root=out_root, dry_run=True)
    except Exception as exc:
        traceback.print_exc()
        res = PipelineResult(domain=args.domain, variant=args.variant, mode="vlm",
                             status="PIPELINE_EXCEPTION", failure_reason=str(exc))
    row["runtime_sec"] = round(time.perf_counter() - t0, 2)

    def rj(name, default=None):
        p = run_dir / name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text())
        except Exception:
            return default

    grounding = rj("graph_grounding_result.json", {}) or {}
    manifest = rj("run_manifest.json", {}) or {}
    plan = list(res.candidate_plan or ())
    # Each domain writes its independent replay validation in its own place; a
    # missing record must not read as an unvalidated plan.
    validation = (rj("action_sequence/replay_validation.json")
                  or (rj("action_sequence/action_plan.json", {}) or {}).get("validation")
                  or (rj("action_plan.json", {}) or {}).get("validation") or {})
    row.update({
        "pipeline_status": res.status,
        "outcome_category": res.outcome_category,
        "failure_category": res.failure_category,
        "failure_reason": str(res.failure_reason or "")[:600],
        "inspected_regions": list(res.inspected_regions or ()),
        "n_inspected_regions": len(res.inspected_regions or ()),
        "grounding_reached": bool(res.status not in (None, "VLM_SPEC_FAILED", "PIPELINE_EXCEPTION")),
        "grounding_status": grounding.get("status"),
        "grounding_satisfied": bool(grounding.get("satisfied")),
        "grounding_complete": bool(grounding.get("complete")),
        "grounding_assignment": grounding.get("assignment") or {},
        "grounding_missing": [str(x)[:200] for x in (grounding.get("missing_requirements") or [])],
        "grounding_failure_kind": grounding.get("failure_kind"),
        "complete_grounding": bool(res.functional_spec_complete),
        "grounding_sec": (manifest.get("logical_grounding_seconds")
                          or (grounding.get("evidence") or {}).get("logical_grounding_seconds")),
        "astar_invocations": manifest.get("astar_invocations", 0),
        "astar_reached": bool(plan) or res.status == "ACTION_SEQUENCE_READY",
        "plan_length": len(plan),
        "symbolic_validation": validation.get("status") or validation.get("valid"),
        "symbolic_goal_status": validation.get("goal_status"),
        "success": res.status == "ACTION_SEQUENCE_READY",
        "semantic_vlm_requests": manifest.get("semantic_vlm_requests", 0),
    })
    try:
        from mujoco_scenes.functional_tamp_pipeline.evaluation_metrics import full_task_coverage
        tot, sat, cov, full = full_task_coverage(args.domain, run_dir)
        row.update({"gt_goals_total": tot, "gt_goals_satisfied": sat,
                    "gt_goal_coverage": round(cov, 4), "gt_full_task_satisfied": bool(full)})
    except Exception as exc:
        row["gt_eval_error"] = f"{type(exc).__name__}: {exc}"
        row["gt_full_task_satisfied"] = False
    # One authoritative rule, shared with the live evaluator and the held-out
    # matrix.  See mujoco_scenes/evaluation_outcome.py for the four disagreeing
    # definitions this replaced and why each was wrong.
    from mujoco_scenes.evaluation_outcome import is_false_completion, outcome_is_correct
    row["false_completion"] = is_false_completion(
        pipeline_status=row.get("pipeline_status"),
        gt_full_task_satisfied=bool(row.get("gt_full_task_satisfied")))
    row["outcome_correct"] = outcome_is_correct(
        gt_feasible=bool(row["feasible"]),
        gt_full_task_satisfied=bool(row.get("gt_full_task_satisfied")),
        pipeline_status=row.get("pipeline_status"),
        runtime_contract_complete=row.get("executable_contract_complete"))
    Path(args.row_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.row_out).write_text(json.dumps(row, indent=2, default=str) + "\n")
    _prune_bulk_artifacts(run_dir)
    return 0


KEEP = {
    "result.json", "run_manifest.json", "functional_specification.json",
    "functional_requirement_graph.json", "graph_grounding_result.json",
    "satisfaction.json", "observed_scene_graph.json", "action_plan.json",
    "plan.json", "replay_validation.json", "plan_grounding_audit.json",
    "evaluation_record.json", "structural_sanitization.json",
    "grounding_snapshots.json", "symbolic_problem.json",
    "physical_relation_verification_trace.json", "detection_diagnostics.json",
    "canonical_grounding_witness.json", "grounding_summary.json",
}


def _prune_bulk_artifacts(run_dir: Path) -> None:
    """Keep the JSON evidence replay analysis reads; drop renders and per-stage dumps.

    Disk here is scarce and a single trial writes tens of megabytes of camera
    renders and point clouds that no downstream analysis consumes.
    """
    import shutil
    for child in sorted(run_dir.rglob("*")):
        if child.is_file() and child.name not in KEEP:
            try:
                child.unlink()
            except OSError:
                pass
    for child in sorted(run_dir.rglob("*"), key=lambda p: -len(p.parts)):
        if child.is_dir() and not any(child.iterdir()):
            try:
                child.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
