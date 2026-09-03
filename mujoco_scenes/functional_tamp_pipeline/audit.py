"""Evaluation audit for grounding, plan consistency, prompt leakage, and run provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from .models import FunctionalSpecification
from .scene_graph import ObservedSceneGraph
from .system_context_registry import is_valid_planner_argument
from .vlm_spec_provider import VLM_CANONICALIZATION_VERSION
from ..workshop_phase1.fm_adapter import (
    INSPECTION_POLICY_SCHEMA,
    KITCHEN_FUNCTIONAL_GRAPH_SCHEMA,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
)

FORBIDDEN_CHECKER_STRINGS: tuple[str, ...] = (
    "OPEN_CAVITY",
    "ELONGATED_OBJECT",
    "INSERTABLE_IN",
    "REACHES_BOTTOM",
    "CAN_DRIVE_SCREW",
    "CAN_FASTEN",
    "PLANAR_SUPPORT",
    "total_length_m",
)

FORBIDDEN_CANONICAL_REGION_TOKENS: tuple[str, ...] = (
    "D1",
    "D2",
    "C2",
    "B1",
    "C1",
    "LEFT_DRAWER",
    "RIGHT_DRAWER",
    "TOOL_CABINET",
)

FORBIDDEN_ORACLE_STRINGS: tuple[str, ...] = (
    "GTSpecProvider",
    "KitchenGroundTruth",
    "WorkshopGroundTruth",
    "LivingRoomRegionOracle",
    "gt_functional_spec",
)


def get_git_info(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Retrieve git provenance including commit, dirty flag, and dirty source tree diff hash."""
    root = str(repo_root) if repo_root else str(Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "mujoco_scenes", "scripts", "configs"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(status_output)
    except Exception:
        status_output = ""
        dirty = None

    dirty_source_hash: str | None = None
    if dirty:
        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "HEAD", "--", "mujoco_scenes", "scripts", "configs"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            dirty_blob = f"{status_output}\n---DIFF---\n{diff_output}"
            dirty_source_hash = hashlib.sha256(dirty_blob.encode("utf-8")).hexdigest()
        except Exception:
            dirty_source_hash = hashlib.sha256(status_output.encode("utf-8")).hexdigest()

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_dirty_source_hash": dirty_source_hash,
        "is_clean_source_tree": dirty is False,
    }


def compute_prompt_and_schema_hash() -> str:
    schema_blob = json.dumps(
        {
            "system_prompt": SYSTEM_PROMPT,
            "response_schema": RESPONSE_SCHEMA,
            "inspection_policy_schema": INSPECTION_POLICY_SCHEMA,
            "kitchen_schema": KITCHEN_FUNCTIONAL_GRAPH_SCHEMA,
        },
        sort_keys=True,
    )
    return hashlib.sha256(schema_blob.encode("utf-8")).hexdigest()


def compute_provenance_fingerprint(
    *,
    domain: str,
    variant: str,
    model: str | None = None,
    base_url: str | None = None,
    task_instruction: str | None = None,
    search_order: str = "auto",
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Compute an immutable, reproducible provenance fingerprint dictionary."""
    git_info = get_git_info(repo_root)
    model_id = model or os.environ.get("TAMP_FM_MODEL", "qwen35-9b")
    endpoint = base_url or os.environ.get("TAMP_FM_BASE_URL", "http://127.0.0.1:18000/v1")
    schema_hash = compute_prompt_and_schema_hash()
    task_hash = (
        hashlib.sha256((task_instruction or "").strip().encode("utf-8")).hexdigest()
        if task_instruction
        else ""
    )

    fingerprint_core = {
        "schema_version": 2,
        "domain": domain,
        "variant": variant,
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "git_dirty_source_hash": git_info["git_dirty_source_hash"],
        "is_clean_source_tree": git_info["is_clean_source_tree"],
        "model_identifier": model_id,
        "fm_endpoint": endpoint,
        "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
        "prompt_schema_hash": schema_hash,
        "task_instruction_hash": task_hash,
        "search_order_mode": search_order,
    }
    canonical_json = json.dumps(fingerprint_core, sort_keys=True)
    fp_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    fingerprint_core["fingerprint_sha256"] = fp_sha
    return fingerprint_core


def _extract_model_request_data(payload: Any) -> tuple[Any, bool]:
    """Extract model-facing request information while strictly excluding generated model outputs."""
    if isinstance(payload, dict):
        if "sanitized_request" in payload and payload["sanitized_request"] is not None:
            return payload["sanitized_request"], True
        if "request" in payload and payload["request"] is not None:
            return payload["request"], True

        # If it contains both request and response keys (e.g. raw response wrapper),
        # exclude response-only fields.
        response_keys = {
            "content",
            "raw_response",
            "parsed_response",
            "choices",
            "usage",
            "finish_reason",
            "parse_error",
            "json_parse_success",
            "content_length_chars",
            "content_sha256",
        }
        filtered = {k: v for k, v in payload.items() if k not in response_keys}
        if filtered:
            return filtered, True
        return None, False

    if isinstance(payload, list):
        extracted_items = []
        for item in payload:
            extracted, ok = _extract_model_request_data(item)
            if ok and extracted is not None:
                extracted_items.append(extracted)
        if extracted_items:
            return extracted_items, True
        return None, False

    if isinstance(payload, Path):
        try:
            data = json.loads(payload.read_text(encoding="utf-8"))
            return _extract_model_request_data(data)
        except Exception:
            return payload.read_text(encoding="utf-8"), True

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
            return _extract_model_request_data(data)
        except Exception:
            return payload, True

    return str(payload), True


def audit_prompt_leakage(
    payload_or_messages: Any,
    *,
    domain: str | None = None,
) -> dict[str, Any]:
    """Deterministically audit that the outgoing model request contains no leaked internal checkers, canonical regions, or oracles.

    Crucially, model responses/outputs are excluded and never scanned as prompt leakage.
    """
    del domain
    request_data, has_request = _extract_model_request_data(payload_or_messages)
    if not has_request or request_data is None:
        return {
            "audited": False,
            "zero_leakage": None,
            "audit_status": "SKIPPED_NO_REQUEST_PAYLOAD",
            "reason": "No model-facing request payload available for leakage audit (response-only data was excluded)",
            "forbidden_checkers_found": [],
            "forbidden_regions_found": [],
            "forbidden_oracle_symbols_found": [],
            "inspected_size_bytes": 0,
            "payload_sha256": "",
        }

    if isinstance(request_data, (dict, list)):
        serialized = json.dumps(request_data, sort_keys=True)
    else:
        serialized = str(request_data)

    checkers_found = [c for c in FORBIDDEN_CHECKER_STRINGS if c in serialized]
    regions_found = [
        r
        for r in FORBIDDEN_CANONICAL_REGION_TOKENS
        if re.search(rf"\b{re.escape(r)}\b", serialized)
    ]
    oracles_found = [o for o in FORBIDDEN_ORACLE_STRINGS if o in serialized]

    zero_leakage = (
        len(checkers_found) == 0 and len(regions_found) == 0 and len(oracles_found) == 0
    )
    return {
        "audited": True,
        "zero_leakage": zero_leakage,
        "audit_status": "AUDIT_PASSED" if zero_leakage else "AUDIT_FAILED_LEAKAGE_DETECTED",
        "forbidden_oracle_symbols_found": sorted(oracles_found),
        "oracle_symbols_in_prompt": len(oracles_found) > 0,
        "checker_predicates_in_prompt": len(checkers_found) > 0,
        "canonical_regions_in_prompt": len(regions_found) > 0,
        "forbidden_checkers_found": sorted(checkers_found),
        "forbidden_regions_found": sorted(regions_found),
        "inspected_size_bytes": len(serialized.encode("utf-8")),
        "payload_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def audit_plan_grounding(
    specification: FunctionalSpecification,
    graph_o: ObservedSceneGraph,
    ground_result: Any,
    plan: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    home_region: str = "countertop",
    allowed_context_ids: Any = None,
) -> dict[str, Any]:
    """Audit that A* plan adheres to phi* grounding and causal accessibility."""
    violations: list[str] = []
    grounding_complete = bool(getattr(ground_result, "complete", False))
    if not grounding_complete:
        violations.append("Grounding result is not complete")

    assignment = getattr(ground_result, "assignment", {}) or {}
    operation_bindings = getattr(ground_result, "operation_bindings", {}) or {}

    assigned_object_ids: set[str] = set()
    for role_name, val in assignment.items():
        if isinstance(val, str):
            assigned_object_ids.add(val)
            node = graph_o.nodes.get(val)
            if node and "payload_ids" in node.unary_properties:
                assigned_object_ids.update(node.unary_properties["payload_ids"])
        elif isinstance(val, (list, tuple, set)):
            for v in val:
                if isinstance(v, str):
                    assigned_object_ids.add(v)
                    node = graph_o.nodes.get(v)
                    if node and "payload_ids" in node.unary_properties:
                        assigned_object_ids.update(node.unary_properties["payload_ids"])

    # 1. Verify all assigned nodes exist in G_O
    all_assignment_nodes_observed = True
    for obj_id in sorted(assigned_object_ids):
        if obj_id not in graph_o.nodes:
            all_assignment_nodes_observed = False
            violations.append(f"Assigned object '{obj_id}' not found in observed scene graph G_O")

    # 2. Verify all required relations in operation bindings are TRUE
    all_required_relations_true = True
    for group_id, bindings in operation_bindings.items():
        matching_group = next((g for g in specification.operation_groups if g.id == group_id), None)
        if matching_group is None:
            all_required_relations_true = False
            violations.append(
                f"Operation binding specifies interaction group '{group_id}' not declared in functional specification"
            )
            continue
        required_rels = matching_group.required_relations
        for binding in bindings:
            tool_id = binding.get("tool_id")
            target_id = binding.get("target_id")
            for rel in required_rels:
                obs_rel = graph_o.get_relation(rel, tool_id, target_id)
                if obs_rel is None or obs_rel.status != "TRUE":
                    all_required_relations_true = False
                    status_str = obs_rel.status if obs_rel else "MISSING"
                    violations.append(
                        f"Operation binding ({tool_id}, {target_id}) for group '{group_id}' has relation '{rel}' with status '{status_str}' (expected TRUE)"
                    )

    # 3. Plan argument consistency: fail-closed validation against G_O nodes and explicit domain constants
    plan_uses_only_grounded_task_objects = True
    for action in plan:
        op = action.get("operator", "")
        args = action.get("arguments", [])
        for arg in args:
            if not is_valid_planner_argument(
                domain=specification.domain,
                argument=arg,
                graph_o=graph_o,
                assigned_object_ids=assigned_object_ids,
                allowed_context_ids=allowed_context_ids,
            ):
                plan_uses_only_grounded_task_objects = False
                violations.append(f"Action {op}({', '.join(args)}) uses ungrounded object '{arg}'")

    # 4. Preparation accessibility check
    preparation_accessibility_valid = True
    object_locations: dict[str, str] = {}
    for node_id, node in graph_o.nodes.items():
        if node.source_region:
            object_locations[node_id] = node.source_region
        elif node.region:
            object_locations[node_id] = node.region

    held_obj: str | None = None
    for action in plan:
        op = action.get("operator", "")
        args = action.get("arguments", [])
        if op == "PICK":
            obj = args[0]
            held_obj = obj
            object_locations.pop(obj, None)
        elif op == "PLACE":
            obj, dest = args[0], args[1]
            held_obj = None
            object_locations[obj] = dest
        elif op == "POUR":
            _src, tgt = args[0], args[1]
            tgt_loc = object_locations.get(tgt)
            if tgt_loc != home_region:
                preparation_accessibility_valid = False
                violations.append(f"POUR into target '{tgt}' while target is at '{tgt_loc}' (expected '{home_region}')")
        elif op == "STIR":
            _tool, tgt = args[0], args[1]
            tgt_loc = object_locations.get(tgt)
            if tgt_loc != home_region:
                preparation_accessibility_valid = False
                violations.append(f"STIR target '{tgt}' while target is at '{tgt_loc}' (expected '{home_region}')")

    return {
        "grounding_complete": grounding_complete,
        "role_assignments": assignment,
        "operation_bindings": operation_bindings,
        "all_assignment_nodes_observed": all_assignment_nodes_observed,
        "all_required_relations_true": all_required_relations_true,
        "plan_uses_only_grounded_task_objects": plan_uses_only_grounded_task_objects,
        "preparation_accessibility_valid": preparation_accessibility_valid,
        "plan_replay_valid": bool(plan) and len(violations) == 0,
        "violations": violations,
    }
