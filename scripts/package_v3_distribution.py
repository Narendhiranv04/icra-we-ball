#!/usr/bin/env python3
"""Package a 3 x 32 V3 development distribution into the frozen per-trial layout.

Reads the raw evaluator output (``<root>/trial_NN/<domain>/<variant>/vlm/``) and
writes ``<root>/<domain>/<variant>/trial_NN/`` holding the frozen raw FM contract
plus every artifact that can be re-derived from it deterministically.

The raw contract is copied verbatim and never rewritten: it is the frozen
development datum.  Everything else is recomputed, so that later zero-call
replays can be compared against the same starting point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _raw_contract(vlm_dir: Path):
    """Return (raw_document, fm_call_record) for the single semantic call."""
    calls = sorted((vlm_dir / "fm_diagnostics").glob("fm_call_*.json"))
    if not calls:
        return None, None
    record = _load(calls[0])
    if not record:
        return None, None
    try:
        return json.loads(record.get("content") or ""), record
    except Exception:
        return None, record


CANONICAL_TASK_INSTRUCTIONS = {
    "kitchen": "Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil.",
    "living_room": "Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people.",
    "workshop": "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench.",
}


def _derive(raw, domain):
    """Recompute normalization / compilation / accounting from a frozen raw.

    Mirrors the online path: same domain-aware normalization, same canonical
    conversion, same compiler, and the compiler's own completeness verdict.
    """
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
        normalize_and_validate_v3_contract, convert_v3_to_canonical_document,
    )
    instruction = CANONICAL_TASK_INSTRUCTIONS.get(domain, "")
    out = {
        "strict_v3_valid": False, "strict_v3_error": None,
        "normalized": None, "normalization_trace": None,
        "canonical_document": None, "task_valid": False, "task_error": None,
        "semantic_accounting": None,
        "compiled": False, "compile_error": None, "compiler_ran": False,
        "contract_missing_reasons": None, "canonicalization_status": None,
        "compiled_graph_summary": None, "semantic_hypotheses": None,
        "unresolved_semantics": None, "constraint_interpretation": None,
    }
    if raw is None:
        out["strict_v3_error"] = "NO_PARSEABLE_FM_CONTENT"
        return out
    try:
        normalized, trace = normalize_and_validate_v3_contract(
            raw, domain=domain, task_instruction=instruction)
        out.update(strict_v3_valid=True, normalized=normalized,
                   normalization_trace=trace)
    except Exception as exc:
        out["strict_v3_error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        canonical = convert_v3_to_canonical_document(normalized, domain=domain)
        out.update(task_valid=True, canonical_document=canonical)
    except Exception as exc:
        out["task_error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
        graph = compile_candidate_graph(domain, instruction, canonical)
        meta = getattr(graph, "metadata", {}) or {}
        nodes = getattr(graph, "nodes", {}) or {}
        out["compiler_ran"] = True
        out["compiled"] = bool(meta.get("required_contract_complete"))
        out["contract_missing_reasons"] = meta.get("contract_missing_reasons")
        out["canonicalization_status"] = meta.get("canonicalization_status")
        out["semantic_accounting"] = meta.get("fm_semantic_accounting")
        out["semantic_hypotheses"] = meta.get("role_type_hypotheses")
        out["unresolved_semantics"] = meta.get("unresolved_semantics")
        out["constraint_interpretation"] = meta.get("functional_constraint_interpretation")
        out["compiled_graph_summary"] = {
            "role_count": len(nodes),
            "roles": sorted(nodes),
            "relation_count": len(getattr(graph, "relations", ()) or ()),
            "operation_group_count": len(getattr(graph, "operation_groups", ()) or ()),
            "required_contract_complete": out["compiled"],
            "canonicalization_status": out["canonicalization_status"],
            "contract_missing_reasons": out["contract_missing_reasons"],
        }
    except Exception as exc:
        out["compile_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _stage_row(domain, variant, trial, raw, derived, result):
    result = result or {}
    outcome = result.get("outcome_category")
    status = result.get("status")
    grounding = result.get("graph_grounding_result") or {}
    plan = result.get("candidate_plan") or []
    return {
        "domain": domain, "variant": variant, "trial": trial,
        "feasible": variant in FEASIBLE[domain],
        "fm_status": (raw or {}).get("status"),
        "strict_v3_valid": derived["strict_v3_valid"],
        "strict_v3_error": derived["strict_v3_error"],
        "task_valid": derived["task_valid"],
        "task_error": derived["task_error"],
        "graph_compiled": derived["compiled"],
        "compile_error": derived["compile_error"],
        "pipeline_status": status,
        "outcome_category": outcome,
        "grounding_reached": bool(result.get("grounding_reached", grounding or status not in
                                             (None, "VLM_SPEC_FAILED"))),
        "complete_grounding": bool(result.get("functional_spec_complete")),
        "astar_reached": bool(plan) or status == "ACTION_SEQUENCE_READY",
        "plan_length": len(plan),
        "success": status == "ACTION_SEQUENCE_READY",
        "failure_reason": str(result.get("failure_reason") or "")[:400],
        "n_roles": len(((raw or {}).get("task_contract") or {}).get("functional_roles", [])),
        "n_relations": len(((raw or {}).get("task_contract") or {}).get("functional_relations", [])),
        "n_operations": len(((raw or {}).get("task_contract") or {}).get("operation_pairings", [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for t in range(1, args.trials + 1):
        trial = f"trial_{t:02d}"
        trial_root = args.root / trial
        if not trial_root.exists():
            print(f"  (missing {trial}, skipping)")
            continue
        for domain, variants in DOMAINS.items():
            for variant in variants:
                src = trial_root / domain / variant / "vlm"
                if not src.exists():
                    continue
                dst = args.root / domain / variant / trial
                dst.mkdir(parents=True, exist_ok=True)

                raw, call = _raw_contract(src)
                if raw is not None:
                    _write(dst / "raw_v3.json", raw)
                if call is not None:
                    _write(dst / "fm_call_record.json", {
                        k: v for k, v in call.items() if k != "content"
                    })
                derived = _derive(raw, domain)
                if derived["normalized"] is not None:
                    _write(dst / "normalized_v3.json", derived["normalized"])
                    _write(dst / "normalization_trace.json", derived["normalization_trace"])
                if derived["canonical_document"] is not None:
                    _write(dst / "functional_requirement_graph.json", derived["canonical_document"])
                if derived["semantic_accounting"] is not None:
                    _write(dst / "semantic_accounting.json", derived["semantic_accounting"])
                if derived["semantic_hypotheses"] is not None:
                    _write(dst / "semantic_hypotheses.json", derived["semantic_hypotheses"])
                if derived["compiled_graph_summary"] is not None:
                    _write(dst / "compiled_graph_summary.json", derived["compiled_graph_summary"])

                for name in ("result.json", "run_manifest.json", "evaluation_record.json",
                             "graph_grounding_result.json", "grounding_summary.json",
                             "structural_sanitization.json", "candidate_replay_validation.json",
                             "executability_analysis.json"):
                    if (src / name).exists():
                        shutil.copy2(src / name, dst / ("final_result.json" if name == "result.json" else name))
                if (src / "action_sequence").exists():
                    shutil.copytree(src / "action_sequence", dst / "action_sequence", dirs_exist_ok=True)

                row = _stage_row(domain, variant, trial, raw, derived, _load(src / "result.json"))
                _write(dst / "stage_progression.json", row)
                rows.append(row)

    _write(args.root / "stage_progression_all.json", rows)
    print(f"packaged {len(rows)} trials -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
