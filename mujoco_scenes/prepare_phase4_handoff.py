"""Deterministic generator and reconstructor for Phase-3 GT handoffs.

Reconstructs the immutable Phase-3 planning artifacts required for Phase-4
execution from tracked minimal scene observation fixtures.

Usage:
    python -m mujoco_scenes.prepare_phase4_handoff --domain kitchen --variant K4 --mode gt
    python -m mujoco_scenes.prepare_phase4_handoff --domain kitchen --variant all --mode gt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from .final_paper_variant_labels import paper_variant_label, resolve_variant_name
from .functional_tamp_pipeline.audit import audit_plan_grounding
from .functional_tamp_pipeline.domains.kitchen import (
    TASK as KITCHEN_TASK,
    KitchenPlanningCompiler,
    build_canonical_kitchen_witness,
    compile_kitchen_contract_from_graph,
    compile_observed_symbolic_state,
)
from .functional_tamp_pipeline.grounding import ground_graph
from .functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from .functional_tamp_pipeline.planning import plan_with_common_astar
from .functional_tamp_pipeline.scene_graph import ObservedSceneGraph
from .functional_tamp_pipeline.search_contract import freeze_search_region_contract
from .phase4_execution import Phase3Handoff, load_phase3_handoff


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "functional_tamp_pipeline"
DEFAULT_FIXTURE_ROOT = ROOT / "mujoco_scenes" / "fixtures" / "phase3_gt"

KITCHEN_VARIANTS = tuple(f"K{i}" for i in range(1, 7))


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def prepare_kitchen_gt_handoff(
    variant: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
) -> Phase3Handoff:
    """Deterministically compile and write the Phase-3 GT handoff for Kitchen."""
    v_norm = variant.strip().upper()
    internal_variant = resolve_variant_name("kitchen", v_norm)
    paper_label = paper_variant_label("kitchen", internal_variant)

    fixture_dir = fixture_root / "kitchen" / paper_label
    if not fixture_dir.is_dir():
        raise FileNotFoundError(
            f"Missing tracked GT fixture for kitchen/{paper_label} at {fixture_dir}"
        )

    run_dir = output_root / "kitchen" / paper_label / "gt"
    run_dir.mkdir(parents=True, exist_ok=True)
    p1_dir = run_dir / "observed_search" / "phase1"
    p1_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy observation evidence from tracked fixture
    fixture_p1 = fixture_dir / "observed_search" / "phase1"
    shutil.copy2(fixture_dir / "observed_scene_graph.json", run_dir / "observed_scene_graph.json")
    shutil.copy2(fixture_p1 / "object_registry.json", p1_dir / "object_registry.json")
    shutil.copy2(fixture_p1 / "observed_graph.json", p1_dir / "observed_graph.json")
    shutil.copy2(fixture_p1 / "latest_witness.json", p1_dir / "latest_witness.json")
    if (fixture_p1 / "events.jsonl").is_file():
        shutil.copy2(fixture_p1 / "events.jsonl", p1_dir / "events.jsonl")

    # 2. Functional specification from authoritative GT provider
    spec_provider = GTSpecProvider()
    spec = spec_provider.provide("kitchen", KITCHEN_TASK)
    spec_dict = spec.to_dict()
    _write_json(run_dir / "functional_specification.json", spec_dict)
    _write_json(run_dir / "functional_requirement_graph.json", spec_dict)

    # 3. Grounding against observed scene graph
    go_dict = json.loads((run_dir / "observed_scene_graph.json").read_text(encoding="utf-8"))
    graph_o = ObservedSceneGraph.from_dict(go_dict)
    ground_result = ground_graph(spec, graph_o, {"search_exhausted": True})
    if not ground_result.complete:
        raise RuntimeError(f"Grounding failed for kitchen/{paper_label}: {ground_result.missing_roles}")
    _write_json(run_dir / "graph_grounding_result.json", ground_result.to_dict())

    # 4. Canonical witness & contract compilation
    contract = compile_kitchen_contract_from_graph(spec)
    witness_payload = build_canonical_kitchen_witness(spec, ground_result, graph_o)
    _write_json(run_dir / "canonical_grounding_witness.json", witness_payload)

    # 5. Symbolic state compilation & A* planning with KitchenPlanningCompiler
    compiled = compile_observed_symbolic_state(p1_dir, contract)
    assignments = ground_result.assignment
    planned = plan_with_common_astar(
        KitchenPlanningCompiler(),
        assignments,
        {"compiled_observed_state": compiled},
    )

    plan_dir = run_dir / "action_sequence"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_data = {
        "planner": planned.search.statistics,
        "actions": list(planned.actions),
        "validation": planned.validation if isinstance(planned.validation, dict) else (
            planned.validation.to_dict() if planned.validation else {}
        ),
        "exploratory_open_actions_excluded": True,
    }
    _write_json(plan_dir / "action_plan.json", plan_data)

    # 6. Plan grounding audit
    home_reg = contract.get("symbolic_task", {}).get("home_region", "countertop")
    audit = audit_plan_grounding(spec, graph_o, ground_result, planned.actions, home_region=home_reg)
    _write_json(run_dir / "plan_grounding_audit.json", audit)

    # 7. Inspected regions from inspection events
    opened: list[str] = []
    events_file = p1_dir / "events.jsonl"
    if events_file.is_file():
        for line in events_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    ev = json.loads(line)
                    if ev.get("event") == "REGION_OPENED":
                        opened.append(str(ev["region_id"]))
                except Exception:
                    pass

    # 8. Result artifact
    result_data = {
        "domain": "kitchen",
        "variant": paper_label,
        "mode": "gt",
        "status": "ACTION_SEQUENCE_READY",
        "inspected_regions": opened,
        "assignment": ground_result.assignment,
        "plan": planned.actions,
        "search_statistics": planned.search.statistics,
        "failure_reason": None,
    }
    _write_json(run_dir / "result.json", result_data)

    # 9. Search contract and run manifest
    search_contract = freeze_search_region_contract(
        spec, domain="kitchen", mode="gt", variant=paper_label
    )
    spec_sha = _compute_sha256(run_dir / "functional_specification.json")
    manifest = {
        "schema_version": 2,
        "domain": "kitchen",
        "variant": paper_label,
        "internal_variant": internal_variant,
        "spec_mode": "gt",
        "terminal_status": "ACTION_SEQUENCE_READY",
        "execution_state": "planning_only",
        "spec_provider_source": "GT_SPEC_PROVIDER",
        "specification_sha256": spec_sha,
        "artifacts": {
            "manifest": "run_manifest.json",
            "result": "result.json",
            "grounding": "graph_grounding_result.json",
            "final_plan": "action_sequence/action_plan.json",
            "plan_grounding_audit": "plan_grounding_audit.json",
            "observed_graph": "observed_scene_graph.json",
            "functional_specification": "functional_specification.json",
        },
        "search_contract": search_contract.to_dict(),
        "search_order_source_effective": "gt_system",
        "search_seed_effective": search_contract.search_seed,
        "region_order_used": list(search_contract.canonical_region_ids),
        "pipeline_runtime_seconds": planned.search.statistics.get("search_time_ms", 0) / 1000.0,
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    # 10. Fail-closed validation with load_phase3_handoff
    handoff = load_phase3_handoff(run_dir)
    return handoff


def prepare_phase4_handoff(
    domain: str,
    variant: str,
    mode: str = "gt",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
) -> list[Phase3Handoff]:
    """Prepare Phase-3 handoff for one or more variants."""
    domain = domain.lower()
    if domain != "kitchen":
        raise NotImplementedError(
            f"Phase-3 handoff preparation currently supported for domain 'kitchen', got '{domain}'"
        )
    if mode != "gt":
        raise ValueError(f"Handoff preparation currently supported for mode='gt', got '{mode}'")

    if variant.lower() in ("all", "*"):
        targets = list(KITCHEN_VARIANTS)
    elif "," in variant:
        targets = [v.strip().upper() for v in variant.split(",") if v.strip()]
    else:
        targets = [variant.strip().upper()]

    handoffs = []
    for v in targets:
        handoff = prepare_kitchen_gt_handoff(
            variant=v,
            output_root=output_root,
            fixture_root=fixture_root,
        )
        handoffs.append(handoff)
        print(
            f"[Phase-3 GT Handoff Ready] {domain.upper()} {handoff.variant} -> "
            f"{len(handoff.actions)} actions at {handoff.run_dir}",
            flush=True,
        )
    return handoffs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically generate/reconstruct Phase-3 GT handoffs from tracked inputs"
    )
    parser.add_argument("--domain", choices=("kitchen",), default="kitchen")
    parser.add_argument(
        "--variant",
        default="all",
        help="Variant to prepare (e.g. 'K4', 'K4,K5,K6', or 'all'). Default: 'all'",
    )
    parser.add_argument("--mode", choices=("gt",), default="gt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    args = parser.parse_args()

    try:
        results = prepare_phase4_handoff(
            domain=args.domain,
            variant=args.variant,
            mode=args.mode,
            output_root=args.output_root,
            fixture_root=args.fixture_root,
        )
        print(
            f"\nSuccessfully prepared {len(results)} Phase-3 GT handoff(s). "
            f"Ready for Phase-4 execution.",
            flush=True,
        )
        return 0
    except Exception as err:
        print(f"ERROR preparing Phase-3 handoff: {err}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
