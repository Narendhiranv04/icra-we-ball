#!/usr/bin/env python3
"""Package a compact, image-free audit for a three-case V3 development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document,
    normalize_v3_live_document,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_typing import build_role_type_hypotheses


CASES = (("kitchen", "K1"), ("living_room", "L1"), ("workshop", "W3"))


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package(source: Path, output: Path, batch: str) -> None:
    for domain, variant in CASES:
        src = source / domain / variant / "vlm"
        dst = output / domain / variant
        diagnostic = read(src / "fm_diagnostics" / "fm_call_001.json")
        manifest = read(src / "run_manifest.json")
        result = read(src / "result.json")
        record_path = src / "evaluation_record.json"
        record = read(record_path) if record_path.exists() else {}
        raw = json.loads(diagnostic["content"])
        normalized, normalization_trace = normalize_v3_live_document(raw)

        strict_valid = manifest.get("failure_category") != "MALFORMED_VLM_SPECIFICATION"
        task_spec_valid = strict_valid and manifest.get("failure_category") != "TASK_SPECIFICATION_FAILURE"
        graph_compiled = bool(manifest.get("canonicalization_succeeded"))
        images = diagnostic["sanitized_request"]["image_metadata"]
        boundary = {
            "batch": batch,
            "content_sha256": diagnostic["content_sha256"],
            "finish_reason": diagnostic["finish_reason"],
            "image_views": [Path(item["path"]).name for item in images],
            "json_parse_success": diagnostic["json_parse_success"],
            "model": diagnostic["model"],
            "normalization_trace": normalization_trace,
            "num_images": diagnostic["sanitized_request"]["num_images"],
            "strict_v3_valid": strict_valid,
            "task_specification_valid": task_spec_valid,
            "failure_category": manifest.get("failure_category"),
            "failure_reason": manifest.get("failure_reason"),
            "usage": diagnostic["usage"],
        }

        canonical = None
        conversion_error = None
        hypotheses: dict[str, Any] = {}
        try:
            canonical = convert_v3_to_canonical_document(normalized, domain=domain)
            hypotheses = {
                key: value.to_dict()
                for key, value in build_role_type_hypotheses(domain, canonical).items()
            }
        except Exception as exc:  # Preserve the exact deterministic boundary failure.
            conversion_error = f"{type(exc).__name__}: {exc}"

        relations = [] if canonical is None else [
            {
                "id": edge.get("id"),
                "raw_relation": edge.get("relation"),
                "resolved_subject": edge.get("subject_role"),
                "resolved_object": edge.get("object_role"),
                "orientation_options": edge.get("v3_orientation_options", []),
            }
            for edge in canonical.get("functional_relations", [])
        ]
        operations = [] if canonical is None else [
            {
                "id": group.get("id"),
                "raw_operation": group.get("function"),
                "resolved_source": group.get("tool_role"),
                "resolved_target": group.get("target_role"),
                "resolved_anchor": group.get("context_role"),
                "usage_policy": group.get("usage_policy"),
                "slot_assignments": group.get("v3_slot_assignments", []),
            }
            for group in canonical.get("interaction_groups", [])
        ]
        compiled = {
            "compiler_reached": task_spec_valid,
            "graph_compiled": graph_compiled,
            "conversion_error": conversion_error,
            "canonical_roles": [] if canonical is None else [role["id"] for role in canonical["functional_roles"]],
            "relation_orientation_resolutions": relations,
            "operation_participant_slot_resolutions": operations,
            "function_alias_structural_overrides": 0,
            "relation_assisted_role_resolutions": sum(
                hyp["status"] == "RELATION_ASSISTED" for hyp in hypotheses.values()
            ),
            "operation_assisted_role_resolutions": sum(
                hyp["status"] == "OPERATION_ASSISTED" for hyp in hypotheses.values()
            ),
        }
        grounding = {
            "grounding_reached": bool(record.get("candidate_grounding_eligible")),
            "grounding_complete": bool(record.get("grounding_complete")),
            "go_assisted_type_resolutions": 0,
            "grounded_roles": record.get("grounded_roles", []),
            "reason": None if record.get("candidate_grounding_eligible") else manifest.get("failure_reason"),
        }
        final = dict(result)
        final.update({
            "astar_calls": manifest.get("astar_invocations", 0),
            "finish_reason": diagnostic["finish_reason"],
            "fresh_fm_calls": manifest.get("semantic_vlm_requests", 0),
            "num_images": diagnostic["sanitized_request"]["num_images"],
        })
        compact_manifest = {
            key: manifest.get(key) for key in (
                "domain", "variant", "internal_variant", "git_commit", "git_dirty",
                "fm_schema_version", "prompt_schema_hash", "provider_model",
                "semantic_vlm_requests", "transport_retries", "astar_invocations",
                "high_level_replans", "terminal_status", "failure_category",
                "failure_reason", "canonicalization_succeeded", "functional_spec_complete",
                "started_at_utc", "finished_at_utc", "pipeline_runtime_seconds",
            )
        }
        compact_manifest["batch"] = batch
        compact_manifest["inference_config"] = {
            key: manifest["inference_config"].get(key) for key in (
                "model", "temperature", "top_p", "top_k", "presence_penalty", "max_tokens"
            )
        }
        compact_manifest["image_views"] = boundary["image_views"]

        write(dst / "raw_v3.json", raw)
        write(dst / "normalized_v3.json", normalized)
        write(dst / "boundary_result.json", boundary)
        write(dst / "semantic_hypotheses.json", {"conversion_error": conversion_error, "roles": hypotheses})
        write(dst / "compiled_graph_summary.json", compiled)
        write(dst / "grounding_summary.json", grounding)
        write(dst / "final_result.json", final)
        write(dst / "run_manifest.json", compact_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    package(args.source, args.output, args.batch)


if __name__ == "__main__":
    main()
