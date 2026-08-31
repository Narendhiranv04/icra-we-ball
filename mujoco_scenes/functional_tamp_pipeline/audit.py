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


def get_git_info(repo_root: Path | str | None = None) -> tuple[str, bool | None]:
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
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(status)
    except Exception:
        dirty = None
    return commit, dirty


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
    git_commit, git_dirty = get_git_info(repo_root)
    model_id = model or os.environ.get("TAMP_FM_MODEL", "qwen35-9b")
    endpoint = base_url or os.environ.get("TAMP_FM_BASE_URL", "http://127.0.0.1:18000/v1")
    schema_hash = compute_prompt_and_schema_hash()
    task_hash = (
        hashlib.sha256((task_instruction or "").strip().encode("utf-8")).hexdigest()
        if task_instruction
        else ""
    )

    fingerprint_core = {
        "schema_version": 1,
        "domain": domain,
        "variant": variant,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
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


def audit_prompt_leakage(
    payload_or_messages: Any,
    *,
    domain: str | None = None,
) -> dict[str, Any]:
    """Deterministically audit that model prompt/payload contains no leaked internal checkers, canonical regions, or oracles."""
    del domain
    if isinstance(payload_or_messages, (dict, list)):
        serialized = json.dumps(payload_or_messages, sort_keys=True)
    elif isinstance(payload_or_messages, Path):
        serialized = payload_or_messages.read_text(encoding="utf-8")
    elif isinstance(payload_or_messages, str):
        serialized = payload_or_messages
    else:
        serialized = str(payload_or_messages)

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
        "gt_imports_found": len(oracles_found) > 0,
        "oracle_labels_in_prompt": len(regions_found) > 0 or len(checkers_found) > 0,
        "forbidden_checkers_found": sorted(checkers_found),
        "forbidden_regions_found": sorted(regions_found),
        "forbidden_oracles_found": sorted(oracles_found),
        "inspected_size_bytes": len(serialized.encode("utf-8")),
        "payload_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def audit_plan_grounding(
    specification: FunctionalSpecification,
    graph_o: ObservedSceneGraph,
    ground_result: Any,
    plan: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    home_region: str = "countertop",
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
        elif isinstance(val, (list, tuple, set)):
            assigned_object_ids.update(val)

    # 1. Verify all assigned nodes exist in G_O
    all_assignment_nodes_observed = True
    for obj_id in sorted(assigned_object_ids):
        if obj_id not in graph_o.nodes:
            all_assignment_nodes_observed = False
            violations.append(f"Assigned object '{obj_id}' not found in observed scene graph G_O")

    # 2. Verify all required relations in operation bindings are TRUE
    all_required_relations_true = True
    for group_id, bindings in operation_bindings.items():
        # Find group definition if present
        matching_group = next((g for g in specification.operation_groups if g.id == group_id), None)
        required_rels = matching_group.required_relations if matching_group else ("INSERTABLE_IN", "REACHES_BOTTOM")
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

    # 3. Plan argument consistency: ensure task objects used in plan come from assigned objects
    plan_uses_only_grounded_task_objects = True
    # Known region/surface identifiers that are not physical manipulable objects
    known_regions = {
        home_region, "serving_area", "countertop", "shared_table", "personal_table_left",
        "personal_table_right", "staging_tray", "work_surface", "D1", "D2", "C1", "C2", "B1",
        "TOOL_CABINET", "WORKBENCH_DRAWER", "DRILL_PRESS_CABINET",
    }
    for action in plan:
        op = action.get("operator", "")
        args = action.get("arguments", [])
        for arg in args:
            if arg not in known_regions and arg not in assigned_object_ids and not arg.startswith("pos_") and not arg.startswith("slot_"):
                plan_uses_only_grounded_task_objects = False
                violations.append(f"Action {op}({', '.join(args)}) uses ungrounded object '{arg}'")

    # 4. Preparation accessibility check: POUR and STIR targets must be at home_region
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
