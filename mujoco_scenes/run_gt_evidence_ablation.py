"""Evaluate GT semantic, geometric, and joint grounding evidence.

This is an offline oracle-evidence ablation.  It deliberately uses no VLM,
detector, point cloud, search policy, symbolic planner, or robot execution.
It measures the grounding decision boundary only.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from mujoco_scenes.functional_tamp_pipeline.grounding import (
    check_semantic_role_compatibility,
    evaluate_node_for_role,
    ground_graph,
    resolve_evidence_components,
)
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.oracle_evidence import (
    build_oracle_graph,
    intended_outcome,
    kitchen_variants,
    living_room_variants,
    workshop_variants,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedSceneGraph


MODES = ("semantic_only", "geometric_only", "joint")
DOMAINS = ("kitchen", "living_room", "workshop")
COMPONENT_MASKS = {
    "full": ("semantic", "unary", "binary"),
    "no_semantic": ("unary", "binary"),
    "no_unary": ("semantic", "binary"),
    "no_binary": ("semantic", "unary"),
    "semantic_only": ("semantic",),
    "unary_only": ("unary",),
    "binary_only": ("binary",),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _variants(domain: str) -> tuple[str, ...]:
    source = {
        "kitchen": kitchen_variants,
        "living_room": living_room_variants,
        "workshop": workshop_variants,
    }[domain]()
    return tuple(source)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(value)] if isinstance(value, str) else [str(item) for item in value]


def _operation_ground_truth_validity(specification, result, graph: ObservedSceneGraph) -> tuple[int, int]:
    valid = total = 0
    for group in specification.operation_groups:
        bindings = result.operation_bindings.get(group.id, [])
        total += group.required_target_count
        for binding in bindings:
            checks = [
                graph.get_relation(predicate, binding["tool_id"], binding["target_id"])
                for predicate in group.required_relations
            ]
            context = binding.get("context", {})
            if group.context_role and group.context_relations:
                context_id = context.get(group.context_role)
                checks.extend(
                    graph.get_relation(predicate, binding["tool_id"], context_id)
                    for predicate in group.context_relations
                )
            if checks and all(check is not None and check.status == "TRUE" for check in checks):
                valid += 1
    return valid, total


def _assignment_ground_truth_validity(specification, result, graph: ObservedSceneGraph) -> dict[str, int | bool]:
    total_slots = sum(role.minimum_count for role in specification.nodes.values())
    semantic_valid = geometric_valid = joint_valid = selected_slots = 0
    for role_name, role in specification.nodes.items():
        for instance_id in _as_list((result.assignment or {}).get(role_name)):
            node = graph.get_node(instance_id)
            if node is None:
                continue
            selected_slots += 1
            semantic_status, _ = check_semantic_role_compatibility(node, role.semantic_categories)
            geometric_status, _ = evaluate_node_for_role(node, role, evidence_mode="geometric_only")
            if semantic_status == "TRUE":
                semantic_valid += 1
            if geometric_status == "TRUE":
                geometric_valid += 1
            if semantic_status == geometric_status == "TRUE":
                joint_valid += 1
    op_valid, op_total = _operation_ground_truth_validity(specification, result, graph)
    return {
        "role_slots_total": total_slots,
        "role_slots_selected": selected_slots,
        "semantic_valid_role_slots": semantic_valid,
        "geometric_valid_role_slots": geometric_valid,
        "joint_valid_role_slots": joint_valid,
        "operation_bindings_total": op_total,
        "ground_truth_valid_operation_bindings": op_valid,
        "all_selected_bindings_gt_valid": (
            bool(result.complete)
            and selected_slots == total_slots == joint_valid
            and op_valid == op_total
        ),
    }


def evaluate_one(
    domain: str,
    variant: str,
    mode: str,
    *,
    evidence_components: tuple[str, ...] | None = None,
    specification=None,
    graph: ObservedSceneGraph | None = None,
) -> dict[str, Any]:
    specification = specification or GTSpecProvider().provide(domain, "")
    graph = graph or build_oracle_graph(domain, variant, specification)
    start = perf_counter()
    result = ground_graph(
        specification, graph,
        {
            "search_exhausted": True,
            "evidence_mode": mode,
            "evidence_components": evidence_components,
        },
    )
    runtime_ms = (perf_counter() - start) * 1000.0
    expected = intended_outcome(domain, variant)
    predicted = "FEASIBLE" if result.complete else "INFEASIBLE"
    validity = _assignment_ground_truth_validity(specification, result, graph)
    return {
        "domain": domain,
        "variant": variant,
        "evidence_mode": mode,
        "evidence_components": sorted(resolve_evidence_components(mode, evidence_components)),
        "intended_outcome": expected,
        "predicted_outcome": predicted,
        "outcome_correct": predicted == expected,
        "grounding_status": result.status,
        "grounding_complete": result.complete,
        "ground_truth_valid_complete": validity["all_selected_bindings_gt_valid"],
        "runtime_ms": round(runtime_ms, 3),
        **validity,
        "missing_roles": list(result.missing_roles),
        "unsatisfied_relations": list(result.unsatisfied_relations),
        "unresolved_constraints": list(result.unresolved_constraints),
        "assignment": result.assignment,
        "operation_bindings": result.operation_bindings,
        "graph": graph.to_dict(),
        "grounding_result": result.to_dict(),
    }


def _rate(rows: Iterable[dict[str, Any]], predicate) -> float | None:
    values = list(rows)
    if not values:
        return None
    return round(100.0 * sum(bool(predicate(row)) for row in values) / len(values), 2)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for domain in DOMAINS:
        conditions = sorted({row["condition"] for row in rows if row["domain"] == domain})
        for condition in conditions:
            group = [
                row for row in rows
                if row["domain"] == domain and row["condition"] == condition
            ]
            if not group:
                continue
            feasible = [row for row in group if row["intended_outcome"] == "FEASIBLE"]
            infeasible = [row for row in group if row["intended_outcome"] == "INFEASIBLE"]
            completed = [row for row in group if row["grounding_complete"]]
            role_total = sum(int(row["role_slots_selected"]) for row in completed)
            op_total = sum(int(row["operation_bindings_total"]) for row in completed)
            summary.append({
                "domain": domain,
                "condition": condition,
                "enabled_evidence": "+".join(group[0]["evidence_components"]),
                "variants": len(group),
                "outcome_agreement_pct": _rate(group, lambda row: row["outcome_correct"]),
                "feasible_completion_pct": _rate(feasible, lambda row: row["grounding_complete"]),
                "infeasible_rejection_pct": _rate(infeasible, lambda row: not row["grounding_complete"]),
                "false_completion_pct": _rate(infeasible, lambda row: row["grounding_complete"]),
                "gt_valid_selection_pct": _rate(completed, lambda row: row["ground_truth_valid_complete"]),
                "semantic_role_validity_pct": round(100.0 * sum(int(row["semantic_valid_role_slots"]) for row in completed) / role_total, 2) if role_total else None,
                "geometric_role_validity_pct": round(100.0 * sum(int(row["geometric_valid_role_slots"]) for row in completed) / role_total, 2) if role_total else None,
                "operation_binding_validity_pct": round(100.0 * sum(int(row["ground_truth_valid_operation_bindings"]) for row in completed) / op_total, 2) if op_total else None,
                "mean_grounding_ms": round(sum(float(row["runtime_ms"]) for row in group) / len(group), 3),
            })
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", default=",".join(DOMAINS), help="Comma-separated domains")
    parser.add_argument("--variants", default=None, help="Optional comma-separated variant IDs")
    parser.add_argument("--evidence-modes", default=",".join(MODES), help="Comma-separated evidence modes")
    parser.add_argument(
        "--component-masks",
        default=None,
        help=(
            "Optional comma-separated component ablations. Use 'all' for all seven "
            "non-empty masks, or names from " + ", ".join(COMPONENT_MASKS)
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    domains = tuple(item.strip() for item in args.domains.split(",") if item.strip())
    modes = tuple(item.strip() for item in args.evidence_modes.split(",") if item.strip())
    requested_variants = None if args.variants is None else {item.strip() for item in args.variants.split(",") if item.strip()}
    if any(domain not in DOMAINS for domain in domains):
        raise ValueError(f"domains must be selected from {DOMAINS}")
    if any(mode not in MODES for mode in modes):
        raise ValueError(f"evidence modes must be selected from {MODES}")
    if args.component_masks is None:
        conditions = [(mode, None) for mode in modes]
    else:
        requested_masks = tuple(
            item.strip() for item in args.component_masks.split(",") if item.strip()
        )
        if requested_masks == ("all",):
            requested_masks = tuple(COMPONENT_MASKS)
        unknown_masks = set(requested_masks) - set(COMPONENT_MASKS)
        if unknown_masks:
            raise ValueError(
                "component masks must be selected from "
                f"{tuple(COMPONENT_MASKS)}; got {sorted(unknown_masks)}"
            )
        conditions = [
            (name, COMPONENT_MASKS[name]) for name in requested_masks
        ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root or Path("runs") / "gt_evidence_ablation" / stamp
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for domain in domains:
        variants = [variant for variant in _variants(domain) if requested_variants is None or variant in requested_variants]
        if not variants:
            raise ValueError(f"No {domain} variants matched --variants")
        specification = GTSpecProvider().provide(domain, "")
        for variant in variants:
            graph = build_oracle_graph(domain, variant, specification)
            for condition, components in conditions:
                print(f"[gt-evidence] {domain} {variant} {condition}", flush=True)
                row = evaluate_one(
                    # Without an explicit component mask the condition *is* one
                    # of the three published aggregate evidence modes.
                    domain, variant, "joint" if components is not None else condition,
                    evidence_components=components,
                    specification=specification, graph=graph,
                )
                row["condition"] = condition
                rows.append(row)
                _write_json(output / domain / variant / f"{condition}.json", row)

    compact_rows = [{key: value for key, value in row.items() if key not in {"graph", "grounding_result"}} for row in rows]
    summary = summarize(rows)
    _write_json(output / "results.json", compact_rows)
    _write_json(output / "summary.json", {"schema_version": 2, "kind": "PRIVILEGED_GT_EVIDENCE_ABLATION", "rows": summary})
    _write_csv(output / "summary.csv", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
