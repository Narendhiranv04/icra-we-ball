"""Package natural-speed three-scene GT executions into one paper evidence tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from .export_gt_everything import (
    _kitchen_texts,
    _living_texts,
    _probe,
    _read,
    _sha256,
    _workshop_texts,
    _write_json,
    _write_text,
)
from .living_room_variants import load_living_room_variant_contract
from .final_paper_variant_labels import paper_variant_label, resolve_variant_name
from .generate_expected_gt_action_catalogue import format_actions, normalized_action
from .workshop_ground_truth_planner import load_variant_specs


ENVIRONMENTS = ("kitchen", "living_room", "workshop")
EXPECTED_GT_ROOT = Path(__file__).resolve().parents[1] / "EXPECTED_GT_ACTIONS"


def _read_seconds(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip())


def _grouped_locations(
    locations: dict[str, str], all_regions: list[str] | None = None
) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for object_id, region_id in locations.items():
        grouped.setdefault(str(region_id), []).append(str(object_id))
    rows: list[str] = []
    for region_id in all_regions or []:
        grouped.setdefault(str(region_id), [])
    for region_id, object_ids in sorted(grouped.items()):
        rows.append(f"REGION {region_id}")
        rows.extend(f"  - {object_id}" for object_id in sorted(object_ids))
        if not object_ids:
            rows.append("  - EMPTY")
    return rows


def _kitchen_objects_regions(source: Path, variant: str) -> str:
    trace = _read(source / "execution_trace.json").get("actions", [])
    locations = (
        trace[0].get("state_before", {}).get("object_locations", {})
        if trace else {}
    )
    return "\n".join([
        "Environment: kitchen",
        f"Variant: {variant}",
        "Snapshot: initial state before GT execution",
        "",
        "OBJECTS GROUPED BY INITIAL REGION",
        *_grouped_locations(
            locations,
            list(trace[0].get("state_before", {}).get("container_open", {}))
            + (["countertop"] if locations else []),
        ),
    ])


def _living_objects_regions(source: Path, variant: str) -> str:
    resolution_path = source / "execution_entity_resolution.json"
    plan_path = source / "refined_mobile_plan.json"
    if not resolution_path.exists() or not plan_path.exists():
        summary = _read(source / "run_summary.json")
        return "\n".join([
            "Environment: living_room",
            f"Variant: {variant}",
            "Physical object/region execution inventory: NOT INSTANTIATED",
            "",
            "Grounding rejected this variant before Phase-2 entity resolution.",
            f"Reason: {summary.get('reason', 'FUNCTIONAL_WITNESS_NOT_COMPLETE')}",
            "No robot payload manipulation is claimed for this infeasible variant.",
        ])
    resolution = _read(resolution_path)
    plan = _read(plan_path).get("actions", [])
    contract = load_living_room_variant_contract()
    variant_spec = contract["variants"][variant]
    logical_regions = contract["regions"]
    backend_to_initial_region = {
        str(backend_object): str(logical_region)
        for backend_object, logical_region in variant_spec.get("object_locations", {}).items()
    }
    backend_to_logical_region = {
        alias: str(logical_region)
        for logical_region, backend_region in logical_regions.items()
        for alias in (str(backend_region), f"{backend_region}_top")
    }
    destinations = {
        str(action["object"]): str(action["region"])
        for action in plan
        if action.get("operator") == "PLACE"
        and action.get("object") is not None
        and action.get("region") is not None
    }
    rows = [
        "Environment: living_room",
        f"Variant: {variant}",
        "Snapshot: initial state before GT execution",
        "",
        "MOVABLE OBJECTS",
    ]
    for entry in resolution.get("objects", []):
        xyz = ", ".join(f"{float(value):.4f}" for value in entry["backend_centroid_world_m"])
        object_id = str(entry["generic_object_id"])
        backend_body = str(entry["backend_body"])
        initial_region = backend_to_initial_region.get(
            backend_body, "INITIAL_OBJECT_STAGING_ZONE"
        )
        rows.append(
            f"{object_id} | semantic={entry['semantic_role']} | "
            f"backend={backend_body} | initial_region={initial_region} | "
            f"initial_xyz_m=[{xyz}] | "
            f"GT_destination={destinations.get(object_id, 'NONE')}"
        )
    rows.extend(["", "INSPECTED/PLACEMENT REGIONS"])
    for entry in resolution.get("regions", []):
        xyz = ", ".join(f"{float(value):.4f}" for value in entry["backend_centroid_world_m"])
        backend_support = str(entry["backend_support_geom"])
        rows.append(
            f"{entry['generic_region_id']} | logical_region="
            f"{backend_to_logical_region.get(backend_support, 'UNKNOWN')} | "
            f"support={backend_support} | "
            f"centroid_xyz_m=[{xyz}]"
        )
    return "\n".join(rows)


def _workshop_objects_regions(source: Path, variant: str) -> str:
    spec = load_variant_specs()[variant]
    return "\n".join([
        "Environment: workshop",
        f"Variant: {variant}",
        "Snapshot: initial fixed-pair storage configuration",
        "",
        "OBJECTS GROUPED BY INITIAL REGION",
        *_grouped_locations({
            object_id: region_id
            for region_id, object_ids in spec["storage_contents"].items()
            for object_id in object_ids
        }, list(spec["storage_contents"])),
        "",
        "FIXED TASK REGION",
        "MAIN_WORKBENCH_ZONE -> workshop_frame_joint",
    ])


def _summary_path(environment: str, source: Path) -> Path:
    return source / ("run_summary.json" if environment == "living_room" else "summary.json")


def _action_source(environment: str, source: Path) -> Path:
    return source / {
        "kitchen": "gt_plan.json",
        "living_room": "refined_mobile_plan.json",
        "workshop": "action_plan.json",
    }[environment]


def _expected_action_payload(
    environment: str,
    variant: str,
    paper_variant: str,
    recorded: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    catalogue_path = (
        EXPECTED_GT_ROOT / environment / paper_variant / "expected_gt_actions.json"
    )
    if catalogue_path.exists():
        return _read(catalogue_path)
    source = _action_source(environment, recorded)
    if source.exists():
        actions = _read(source).get("actions", [])
    else:
        actions = [{
            "operator": "TERMINATE_INFEASIBLE",
            "arguments": [summary.get("reason", "FUNCTIONAL_WITNESS_NOT_COMPLETE")],
        }]
    normalized = [normalized_action(action) for action in actions]
    return {
        "schema_version": 1,
        "environment": environment,
        "variant": paper_variant,
        "internal_variant": variant,
        "intended_outcome": summary.get("intended_outcome", "UNKNOWN"),
        "description": "Expected actions captured from the executed GT plan.",
        "total_actions": len(normalized),
        "actions": normalized,
    }


def _executed_actions(
    environment: str, source: Path, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    if environment == "living_room":
        physical_path = source / "physical_execution.json"
        if not physical_path.exists():
            return [normalized_action({
                "operator": "TERMINATE_INFEASIBLE",
                "arguments": [summary.get("reason", "FUNCTIONAL_WITNESS_NOT_COMPLETE")],
            })]
        return [
            normalized_action(action)
            for action in _read(physical_path).get("actions", [])
        ]
    trace_path = source / "execution_trace.json"
    if not trace_path.exists():
        return []
    trace = _read(trace_path).get("actions", [])
    if environment == "workshop":
        return [
            normalized_action(row["action"])
            for row in trace if isinstance(row.get("action"), dict)
        ]
    return [normalized_action(row) for row in trace]


def _comparison_signature(action: dict[str, Any]) -> tuple[str, str]:
    return (
        str(action.get("operator", "UNKNOWN")).upper(),
        json.dumps(action.get("arguments", []), sort_keys=True),
    )


def _compare_actions(
    environment: str,
    expected: list[dict[str, Any]],
    executed: list[dict[str, Any]],
) -> dict[str, Any]:
    compared_executed = (
        [action for action in executed if action.get("operator") != "MOVE"]
        if environment == "living_room" else executed
    )
    expected_signatures = [_comparison_signature(action) for action in expected]
    executed_signatures = [_comparison_signature(action) for action in compared_executed]
    first_mismatch = None
    for index in range(max(len(expected_signatures), len(executed_signatures))):
        expected_item = expected_signatures[index] if index < len(expected_signatures) else None
        executed_item = executed_signatures[index] if index < len(executed_signatures) else None
        if expected_item != executed_item:
            first_mismatch = {
                "action_index": index + 1,
                "expected": expected_item,
                "executed": executed_item,
            }
            break
    return {
        "schema_version": 1,
        "comparison_policy": (
            "EXACT_GT_TASK_SEQUENCE_IGNORING_INSERTED_MOVE_ACTIONS"
            if environment == "living_room"
            else "EXACT_ACTION_SEQUENCE"
        ),
        "expected_action_count": len(expected),
        "executed_action_count": len(executed),
        "compared_executed_action_count": len(compared_executed),
        "exact_match": expected_signatures == executed_signatures,
        "first_mismatch": first_mismatch,
    }


def _assignment_source(environment: str, source: Path) -> Path | None:
    name = {
        "kitchen": "gt_assignment.json",
        "workshop": "assignment.json",
    }.get(environment)
    return source / name if name else None


def _builders(environment: str):
    return {
        "kitchen": (_kitchen_texts, _kitchen_objects_regions),
        "living_room": (_living_texts, _living_objects_regions),
        "workshop": (_workshop_texts, _workshop_objects_regions),
    }[environment]


def _paper_text(text: str, internal_variant: str, paper_variant: str) -> str:
    """Expose the concise paper label while retaining traceability."""
    marker = f"Variant: {internal_variant}"
    replacement = (
        f"Variant: {paper_variant}\nInternal variant: {internal_variant}"
        if paper_variant != internal_variant else marker
    )
    return text.replace(marker, replacement, 1)


def _variants(recorded_root: Path, environment: str) -> list[str]:
    environment_root = recorded_root / environment
    variants = []
    for child in sorted(environment_root.iterdir()):
        if child.is_dir() and (child / f"{child.name}_5cam.mp4").is_file():
            variants.append(child.name)
    if not variants:
        raise RuntimeError(f"No recorded variants found under {environment_root}")
    return variants


def _validate_physical_execution_contract(
    environment: str, source: Path, summary: dict[str, Any]
) -> None:
    """Fail closed before packaging object-only or assisted demonstrations."""
    if environment == "kitchen":
        if summary.get("execution_profile") != "STRICT_ROBOT_PHYSICAL_PRIMITIVES":
            raise RuntimeError(
                f"Kitchen {source.name} is not a strict robot execution: "
                f"{summary.get('execution_profile')}"
            )
        if summary.get("assisted_action_count") != 0:
            raise RuntimeError(f"Kitchen {source.name} contains assisted actions")
        if summary.get("direct_payload_pose_write_count") != 0:
            raise RuntimeError(f"Kitchen {source.name} contains direct payload pose writes")
        if summary.get("direct_object_qpos_write_count") != 0:
            raise RuntimeError(f"Kitchen {source.name} contains direct object qpos writes")
        return

    if environment == "living_room":
        if summary.get("status") == "INFEASIBLE_CONFIRMED":
            if summary.get("execution_attempted") is not False:
                raise RuntimeError(
                    f"Living Room {source.name} has an invalid infeasible termination"
                )
            return
        physical_path = source / "physical_execution.json"
        if not physical_path.exists():
            raise RuntimeError(f"Living Room {source.name} lacks physical_execution.json")
        physical = _read(physical_path)
        if physical.get("execution_profile") != "STRICT_PHYSICAL_POSTCONDITION":
            raise RuntimeError(
                f"Living Room {source.name} is not strict physical execution"
            )
        if physical.get("normal_execution_object_qpos_edits") is not False:
            raise RuntimeError(
                f"Living Room {source.name} permits object qpos edits during execution"
            )
        if physical.get("success") is not True:
            raise RuntimeError(f"Living Room {source.name} physical execution failed")
        return

    if environment == "workshop":
        if summary.get("execution_profile") != "CONTACT_GATED_ROBOT_ACTUATED_GT_EXECUTION":
            raise RuntimeError(
                f"Workshop {source.name} is not robot-actuated GT execution"
            )
        if summary.get("direct_payload_pose_write_count") != 0:
            raise RuntimeError(f"Workshop {source.name} contains direct payload pose writes")
        trace = _read(source / "execution_trace.json").get("actions", [])
        def contains_true_flag(value: Any, flags: set[str]) -> bool:
            if isinstance(value, dict):
                return any(value.get(flag) is True for flag in flags) or any(
                    contains_true_flag(child, flags) for child in value.values()
                )
            if isinstance(value, list):
                return any(contains_true_flag(child, flags) for child in value)
            return False

        if any(contains_true_flag(
            row.get("physical_result", {}),
            {"direct_payload_pose_write", "direct_object_qpos_write"},
        ) for row in trace):
            raise RuntimeError(f"Workshop {source.name} trace contains a payload pose write")
        if any(
            not row.get("physical_result", {}).get("robot_actuated_motion", False)
            for row in trace if "physical_result" in row
        ):
            raise RuntimeError(f"Workshop {source.name} contains a non-robot GT action")
        return

    raise ValueError(f"Unknown environment: {environment}")


def package(
    recorded_root: Path,
    unrecorded_root: Path,
    timings_root: Path,
    output_root: Path,
    *,
    environments: tuple[str, ...] = ENVIRONMENTS,
    selected_variants: dict[str, list[str]] | None = None,
    append: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    if append:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "manifest.json"
        records = (
            list(_read(manifest_path).get("records", []))
            if manifest_path.exists() else []
        )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        records = []
    for environment in environments:
        text_builder, objects_builder = _builders(environment)
        variants = (
            selected_variants[environment]
            if selected_variants and environment in selected_variants
            else _variants(recorded_root, environment)
        )
        for variant in variants:
            paper_variant = paper_variant_label(environment, variant)
            recorded = recorded_root / environment / variant
            unrecorded = unrecorded_root / environment / variant
            recorded_summary = _read(_summary_path(environment, recorded))
            unrecorded_summary = _read(_summary_path(environment, unrecorded))
            _validate_physical_execution_contract(
                environment, recorded, recorded_summary
            )
            _validate_physical_execution_contract(
                environment, unrecorded, unrecorded_summary
            )
            expected_payload = _expected_action_payload(
                environment, variant, paper_variant, recorded, recorded_summary
            )
            executed_actions = _executed_actions(
                environment, recorded, recorded_summary
            )
            action_comparison = _compare_actions(
                environment, expected_payload.get("actions", []), executed_actions
            )
            if not action_comparison["exact_match"]:
                raise RuntimeError(
                    f"{environment} {paper_variant} executed GT action sequence "
                    f"does not match the expected catalogue: "
                    f"{action_comparison['first_mismatch']}"
                )
            destination = output_root / environment / paper_variant
            if destination.exists():
                if not replace_existing:
                    raise FileExistsError(
                        f"Final variant already exists: {destination}"
                    )
                shutil.rmtree(destination)
            destination.mkdir(parents=True)

            source_video = recorded / f"{variant}_5cam.mp4"
            video = destination / "robot_execution_5cam.mp4"
            shutil.copy2(source_video, video)
            video_metadata = _probe(video)

            actions, assignments = text_builder(variant, recorded)
            _write_text(
                destination / "gt_actions.txt",
                _paper_text(actions, variant, paper_variant),
            )
            _write_text(
                destination / "function_object_assignments.txt",
                _paper_text(assignments, variant, paper_variant),
            )
            _write_text(
                destination / "objects_and_regions.txt",
                _paper_text(objects_builder(recorded, variant), variant, paper_variant),
            )
            _write_json(destination / "expected_gt_actions.json", expected_payload)
            _write_text(
                destination / "expected_gt_actions.txt",
                format_actions(
                    environment,
                    paper_variant,
                    variant,
                    str(expected_payload.get("intended_outcome", "UNKNOWN")),
                    str(expected_payload.get("description", "")),
                    expected_payload.get("actions", []),
                ),
            )
            executed_payload = {
                "schema_version": 1,
                "environment": environment,
                "variant": paper_variant,
                "internal_variant": variant,
                "total_actions": len(executed_actions),
                "actions": executed_actions,
            }
            _write_json(destination / "executed_gt_actions.json", executed_payload)
            _write_text(
                destination / "executed_gt_actions.txt",
                format_actions(
                    environment,
                    paper_variant,
                    variant,
                    str(expected_payload.get("intended_outcome", "UNKNOWN")),
                    "Actions observed in the recorded physical execution trace.",
                    executed_actions,
                ),
            )
            _write_json(destination / "gt_action_comparison.json", action_comparison)
            _write_text(
                destination / "gt_action_comparison.txt",
                "\n".join([
                    f"Environment: {environment}",
                    f"Variant: {paper_variant}",
                    f"Internal variant: {variant}",
                    f"Comparison policy: {action_comparison['comparison_policy']}",
                    f"Expected actions: {action_comparison['expected_action_count']}",
                    f"Executed actions: {action_comparison['executed_action_count']}",
                    f"Compared executed actions: {action_comparison['compared_executed_action_count']}",
                    f"Exact match: {str(action_comparison['exact_match']).upper()}",
                ]),
            )

            recorded_wall = _read_seconds(
                timings_root / environment / f"{variant}_with_recording_seconds.txt"
            )
            unrecorded_wall = _read_seconds(
                timings_root / environment / f"{variant}_without_recording_seconds.txt"
            )
            camera_manifest = _read(recorded / "camera_manifest.json")
            timing = {
                "variant": paper_variant,
                "internal_variant": variant,
                "environment": environment,
                "with_recording_wall_s": recorded_wall,
                "without_recording_wall_s": unrecorded_wall,
                "recording_overhead_wall_s": recorded_wall - unrecorded_wall,
                "video_duration_s": video_metadata["duration_s"],
                "video_fps": video_metadata["frame_rate"],
                "recorded_summary_reported_wall_s": recorded_summary.get("wall_time_s"),
                "unrecorded_summary_reported_wall_s": unrecorded_summary.get("wall_time_s"),
                "simulation_duration_s": (
                    recorded_summary.get("sim_duration_s")
                    or camera_manifest.get("simulation_duration_s")
                    or camera_manifest.get("duration_sim_s")
                ),
                "speed_policy": "NATURAL_SIMULATION_TIME_NO_POSTPROCESS_SPEEDUP",
            }
            _write_json(destination / "timing.json", timing)
            _write_text(
                destination / "timing.txt",
                "\n".join([
                    f"Environment: {environment}",
                    f"Variant: {paper_variant}",
                    f"Internal variant: {variant}",
                    f"Without recording wall time: {unrecorded_wall:.3f} s",
                    f"With recording wall time: {recorded_wall:.3f} s",
                    f"Recording overhead: {recorded_wall - unrecorded_wall:.3f} s",
                    f"Final video duration: {video_metadata['duration_s']:.3f} s",
                    "Playback policy: natural simulation time; no speed-up or setpts transform",
                ]),
            )
            packaged_summary = dict(recorded_summary)
            packaged_summary["paper_variant"] = paper_variant
            packaged_summary["internal_variant"] = variant
            _write_json(destination / "execution_summary.json", packaged_summary)
            action_source = _action_source(environment, recorded)
            if action_source.exists():
                shutil.copy2(action_source, destination / "gt_actions.json")
            else:
                _write_json(destination / "gt_actions.json", {
                    "actions": [{
                        "operator": "TERMINATE_INFEASIBLE",
                        "arguments": [
                            recorded_summary.get("reason", "FUNCTIONAL_WITNESS_NOT_COMPLETE")
                        ],
                        "reason": "No complete functional assignment exists.",
                    }],
                    "total_actions": 1,
                })
            assignment_source = _assignment_source(environment, recorded)
            if assignment_source is not None:
                shutil.copy2(assignment_source, destination / "function_object_assignments.json")

            records = [
                row for row in records
                if not (
                    row.get("environment") == environment
                    and (
                        row.get("variant") == paper_variant
                        or row.get("source_variant") == variant
                    )
                )
            ]
            records.append({
                "environment": environment,
                "variant": paper_variant,
                "source_variant": variant,
                "video": str(video.relative_to(output_root)),
                "video_sha256": _sha256(video),
                "video_metadata": video_metadata,
                "timing": timing,
                "execution_success": (
                    recorded_summary.get("success")
                    if "success" in recorded_summary
                    else recorded_summary.get("status") in {"SUCCESS", "INFEASIBLE_CONFIRMED"}
                ),
                "gt_action_sequence_exact_match": action_comparison["exact_match"],
            })

    manifest = {
        "schema_version": 1,
        "purpose": "Final-paper natural-speed five-view GT robot executions",
        "total_variants": len(records),
        "environment_counts": {
            environment: sum(row["environment"] == environment for row in records)
            for environment in ENVIRONMENTS
        },
        "all_execution_runs_successful": all(row["execution_success"] for row in records),
        "all_gt_action_sequences_exact_match": all(
            row.get("gt_action_sequence_exact_match", False) for row in records
        ),
        "records": records,
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_text(
        output_root / "README.md",
        """# Final paper GT executions

This is the single packaged evidence tree for Kitchen, Living Room, and
Workshop. Each variant contains a natural-speed merged five-camera MP4, the GT
action sequence, function/object assignments, initial object/region placement,
execution summary, and wall-clock timings with and without recording.

Each directory also contains the predeclared expected GT actions, the actions
observed in the physical execution trace, and an exact sequence-comparison
report. Packaging fails if those GT task sequences differ. Living Room permits
planner-inserted MOVE actions between its frozen PICK/PLACE task actions; those
navigation actions remain visible in the executed trace and are excluded only
from the exact GT task-order comparison.

Paper-facing directories and manifest entries use K1-K12, L1-L10, and W1-W10.
The descriptive implementation identifier is retained as `source_variant` in
the manifest and `internal_variant` in timing metadata for reproducibility.

Videos are captured directly during physics execution at the configured FPS.
No frame dropping for speed-up, FFmpeg `setpts`, or postprocessing time scaling
is applied. Feasible variants execute their complete GT task. Infeasible
Kitchen and Workshop variants execute their available inspection/rejection
sequence. Living Room infeasible variants are rejected before Phase-2 physical
planning, so their evidence explicitly records a no-manipulation termination
instead of fabricating robot actions.
""",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--unrecorded-root", type=Path, required=True)
    parser.add_argument("--timings-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--environment", choices=ENVIRONMENTS)
    parser.add_argument("--variant")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if bool(args.environment) != bool(args.variant):
        parser.error("--environment and --variant must be supplied together")
    environments = (args.environment,) if args.environment else ENVIRONMENTS
    selected = (
        {args.environment: [resolve_variant_name(args.environment, args.variant)]}
        if args.environment and args.variant else None
    )
    manifest = package(
        args.recorded_root,
        args.unrecorded_root,
        args.timings_root,
        args.output_root,
        environments=environments,
        selected_variants=selected,
        append=args.append,
        replace_existing=args.replace_existing,
    )
    print(f"Packaged {manifest['total_variants']} variants into {args.output_root}")
    return 0 if manifest["all_execution_runs_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
