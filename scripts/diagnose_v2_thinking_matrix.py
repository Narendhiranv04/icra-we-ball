#!/usr/bin/env python3
"""Development-only replay diagnostics for a directory of saved V2 responses."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    convert_v2_to_canonical_document,
    validate_v2_live_contract,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph


def _normalizer(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import normalize_v2_live_document
    except ImportError:
        return deepcopy(document), []
    return normalize_v2_live_document(document)


def _error_code(error: Exception | str | None) -> str | None:
    if error is None:
        return None
    message = str(error)
    return message.split(":", 1)[0] if ":" in message else message


def _validate(document: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        validate_v2_live_contract(document)
        return True, None
    except Exception as exc:  # diagnostics must retain every invalid case
        return False, str(exc)


def _task_instruction(case_dir: Path) -> str:
    diagnostics = sorted((case_dir / "fm_diagnostics").glob("fm_call_*.json"))
    if diagnostics:
        payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
        return str(payload.get("request", {}).get("task_instruction", ""))
    return ""


def diagnose_case(raw_path: Path) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    domain = raw_path.parents[1].name
    variant = raw_path.parent.name
    original_valid, original_error = _validate(raw)
    normalized, normalization_trace = _normalizer(raw)
    normalized_valid, normalized_error = _validate(normalized)
    result: dict[str, Any] = {
        "domain": domain,
        "variant": variant,
        "raw_path": str(raw_path),
        "original_strict_live_valid": original_valid,
        "original_validation_error": original_error,
        "normalization_trace": normalization_trace,
        "normalized_strict_live_valid": normalized_valid,
        "normalized_validation_error": normalized_error,
        "canonicalization_status": "NOT_COMPILED",
        "online_executable_contract_complete": False,
        "unresolved_role_count": 0,
        "unmapped_required_relation_count": 0,
        "disabled_operation_count": 0,
        "task_effect_count": 0,
        "compile_error": None,
    }
    if normalized_valid:
        try:
            canonical = convert_v2_to_canonical_document(normalized)
            graph = compile_candidate_graph(domain, _task_instruction(raw_path.parent), canonical)
            trace = graph.metadata.get("canonicalization_trace", {})
            result.update(
                canonicalization_status=graph.metadata.get("canonicalization_status"),
                online_executable_contract_complete=bool(
                    graph.metadata.get("online_executable_contract_complete")
                ),
                unresolved_role_count=len(trace.get("unresolved_roles", [])),
                unmapped_required_relation_count=len(
                    trace.get("unresolved_required_relations", [])
                ),
                disabled_operation_count=len(trace.get("disabled_groups", [])),
                task_effect_count=len(graph.task_effect_relations),
                graph_node_names=sorted(graph.nodes),
                operation_group_ids=[group.id for group in graph.operation_groups],
                contract_incompleteness_reasons=graph.metadata.get(
                    "contract_incompleteness_reasons", []
                ),
                unresolved_role_diagnostics=trace.get("unresolved_roles", []),
                unmapped_relation_diagnostics=trace.get(
                    "unresolved_required_relations", []
                ),
            )
        except (VLMSpecificationError, Exception) as exc:
            result["compile_error"] = str(exc)
    contract = raw.get("task_contract", {})
    result["raw_role_function_phrases"] = [
        role.get("function", "") for role in contract.get("functional_roles", [])
    ]
    result["raw_relation_phrases"] = [
        relation.get("relation", "") for relation in contract.get("functional_relations", [])
    ]
    result["raw_operation_phrases"] = [
        operation.get("operation", "") for operation in contract.get("operation_pairings", [])
    ]
    result["g_f_ready"] = bool(
        result["canonicalization_status"] == "FULL"
        and result["online_executable_contract_complete"]
        and result["compile_error"] is None
    )
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(group: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "cases": len(group),
            "original_strict_live_valid": sum(r["original_strict_live_valid"] for r in group),
            "normalized_strict_live_valid": sum(r["normalized_strict_live_valid"] for r in group),
            "canonical_full": sum(r["canonicalization_status"] == "FULL" for r in group),
            "online_executable_contract_complete": sum(
                r["online_executable_contract_complete"] for r in group
            ),
            "g_f_ready": sum(r["g_f_ready"] for r in group),
        }

    by_domain = {
        domain: counts([row for row in rows if row["domain"] == domain])
        for domain in sorted({row["domain"] for row in rows})
    }
    inventories: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for field in (
            "raw_role_function_phrases",
            "raw_relation_phrases",
            "raw_operation_phrases",
        ):
            inventories[field].update(row[field])
        inventories["original_validation_errors"].update(
            [_error_code(row["original_validation_error"])]
            if row["original_validation_error"] else []
        )
        inventories["normalized_validation_errors"].update(
            [_error_code(row["normalized_validation_error"])]
            if row["normalized_validation_error"] else []
        )
    return {
        "overall": counts(rows),
        "by_domain": by_domain,
        "recurring_inventories": {
            key: dict(counter.most_common()) for key, counter in inventories.items()
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [diagnose_case(path) for path in sorted(args.matrix_root.glob("*/*/raw_v2.json"))]
    report = summarize(rows)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
