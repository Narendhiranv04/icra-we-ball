"""Build the standardized GT_everything video/action/assignment evidence tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .workshop_ground_truth_planner import load_variant_specs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "GT_everything"
KITCHEN_RUNS = ROOT / "runs" / "kitchen_ground_truth_execution_assisted_suite"
LIVING_RUNS = ROOT / "runs" / "living_room_execution"
WORKSHOP_RUNS = ROOT / "runs" / "workshop_fixed_pair_gt_execution"
LIVING_ASSIGNMENTS = (
    ROOT / "mujoco_scenes" / "benchmark_reports"
    / "living_room_region_feasibility_phase1" / "variants"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate:format=duration,size",
            "-of", "json", str(path),
        ],
        text=True,
    )
    payload = json.loads(raw)
    stream = payload["streams"][0]
    fmt = payload["format"]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_frames"]),
        "frame_rate": stream["r_frame_rate"],
        "duration_s": float(fmt["duration"]),
        "size_bytes": int(fmt["size"]),
    }


def _format_action(index: int, action: dict[str, Any]) -> str:
    operator = str(action.get("operator") or action.get("action") or "UNKNOWN")
    arguments = action.get("arguments")
    if arguments is None:
        arguments = []
        for key in ("object", "region", "carrying", "target_pose"):
            if key in action and action[key] is not None:
                arguments.append(f"{key}={action[key]}")
    if isinstance(arguments, dict):
        arguments = [f"{key}={value}" for key, value in arguments.items()]
    reason = action.get("reason")
    suffix = f"  # {reason}" if reason else ""
    return f"{index:03d}. {operator}({', '.join(map(str, arguments))}){suffix}"


def _kitchen_texts(variant: str, source: Path) -> tuple[str, str]:
    plan = _read(source / "gt_plan.json")["actions"]
    assignment = _read(source / "gt_assignment.json")
    actions = [
        f"Environment: kitchen\nVariant: {variant}\nExecution profile: ASSISTED_DETERMINISTIC_DEMONSTRATION",
        "",
        *[_format_action(index, action) for index, action in enumerate(plan, 1)],
    ]
    rows = [
        "Environment: kitchen",
        f"Variant: {variant}",
        f"Intended outcome: {assignment.get('intended_outcome')}",
        f"Feasible: {assignment.get('is_feasible')}",
        f"Failure reason: {assignment.get('failure_reason') or 'NONE'}",
        "",
        "FUNCTION -> OBJECT ASSIGNMENTS",
    ]
    for role, object_id in sorted((assignment.get("sources") or {}).items()):
        rows.append(f"{role} -> {object_id}")
    for entry in assignment.get("coffee_assignments", []):
        rows.append(f"STIR_COFFEE({entry['target_instance']}) -> {entry['tool_instance']}")
    for entry in assignment.get("soup_assignments", []):
        rows.append(f"SERVING_UTENSIL({entry['target_instance']}) -> {entry['tool_instance']}")
    if not assignment.get("is_feasible"):
        rows.append("COMPLETE_GLOBAL_ASSIGNMENT -> NONE")
    return "\n".join(actions), "\n".join(rows)


def _living_texts(variant: str, source: Path) -> tuple[str, str]:
    summary = _read(source / "run_summary.json")
    plan_path = source / "refined_mobile_plan.json"
    if plan_path.exists():
        plan = _read(plan_path)["actions"]
    else:
        plan = [{
            "operator": "TERMINATE_INFEASIBLE",
            "arguments": [summary.get("reason", "FUNCTIONAL_WITNESS_NOT_COMPLETE")],
            "reason": "No complete Phase-1 functional assignment exists.",
        }]
    actions = [
        f"Environment: living_room\nVariant: {variant}\nExecution profile: PHYSICAL_MOBILE_MANIPULATION",
        "",
        *[_format_action(index, action) for index, action in enumerate(plan, 1)],
    ]
    assignment_path = LIVING_ASSIGNMENTS / variant / "region_assignments.json"
    assignment = _read(assignment_path) if assignment_path.exists() else {"assignments": []}
    rows = [
        "Environment: living_room",
        f"Variant: {variant}",
        f"Intended outcome: {summary.get('intended_outcome')}",
        f"Execution status: {summary.get('status')}",
        "",
        "FUNCTION -> OBJECT/REGION ASSIGNMENTS",
    ]
    for entry in assignment.get("assignments", []):
        rows.append(
            f"{entry['function_id']} -> region={entry['region_id']}; "
            f"objects={', '.join(entry.get('payload_ids', []))}; "
            f"seat={entry.get('seating_target_id', 'NONE')}"
        )
    if not assignment.get("assignments"):
        rows.append("COMPLETE_GLOBAL_ASSIGNMENT -> NONE")
        rows.append(f"REJECTION -> {summary.get('reason', 'FUNCTIONAL_WITNESS_NOT_COMPLETE')}")
    return "\n".join(actions), "\n".join(rows)


def _workshop_texts(variant: str, source: Path) -> tuple[str, str]:
    plan = _read(source / "action_plan.json")["actions"]
    assignment = _read(source / "assignment.json")
    actions = [
        f"Environment: workshop\nVariant: {variant}\nExecution profile: PHYSICS_ASSISTED_GT_EXECUTION",
        "",
        *[_format_action(index, action) for index, action in enumerate(plan, 1)],
    ]
    rows = [
        "Environment: workshop",
        f"Variant: {variant}",
        f"Intended outcome: {assignment.get('intended_outcome')}",
        f"Assignment source: {assignment.get('assignment_source')}",
        "",
        "FUNCTION -> OBJECT/REGION ASSIGNMENTS",
    ]
    if assignment.get("is_feasible"):
        rows.extend([
            f"CAN_DRIVE_SCREW -> {assignment['driver']}",
            f"CAN_FASTEN -> {assignment['fastener']}",
            f"FIXED_INSERTION_TARGET -> {assignment['target_joint']}",
            "DISTRACTOR_ONLY -> workshop_wooden_hammer",
        ])
    else:
        rows.append("COMPLETE_GLOBAL_ASSIGNMENT -> NONE")
        rows.append(f"REJECTION -> {assignment.get('rejection_reason')}")
    return "\n".join(actions), "\n".join(rows)


def _export_environment(
    environment: str,
    source_root: Path,
    variants: list[str],
    text_builder,
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        source = source_root / variant
        source_video = source / f"{variant}_5cam.mp4"
        if not source_video.exists():
            raise FileNotFoundError(f"Missing validated source video: {source_video}")
        destination = OUTPUT / environment / variant
        destination.mkdir(parents=True, exist_ok=True)
        video = destination / "robot_execution.mp4"
        shutil.copy2(source_video, video)
        action_text, assignment_text = text_builder(variant, source)
        _write_text(destination / "robot_action_sequence.txt", action_text)
        _write_text(destination / "function_object_assignments.txt", assignment_text)
        metadata = _probe(video)
        if metadata["frame_count"] <= 0 or metadata["duration_s"] <= 0:
            raise RuntimeError(f"Invalid video generated for {environment}/{variant}")
        rows.append({
            "environment": environment,
            "variant": variant,
            "video": str(video.relative_to(ROOT)),
            "video_sha256": _sha256(video),
            "actions": str((destination / "robot_action_sequence.txt").relative_to(ROOT)),
            "assignments": str((destination / "function_object_assignments.txt").relative_to(ROOT)),
            **metadata,
        })
    return rows


def main() -> int:
    kitchen_suite = _read(KITCHEN_RUNS / "suite_summary.json")
    kitchen_variants = [row["variant_id"] for row in kitchen_suite["results"]]
    living_suite = _read(LIVING_RUNS / "suite_summary.json")
    living_variants = [row["variant"] for row in living_suite["results"]]
    workshop_specs = load_variant_specs()
    workshop_feasible = [
        variant for variant, spec in workshop_specs.items()
        if spec["intended_outcome"] == "FEASIBLE"
    ]
    workshop_infeasible = [
        variant for variant, spec in workshop_specs.items()
        if spec["intended_outcome"] != "FEASIBLE"
    ]

    records = []
    records.extend(_export_environment("kitchen", KITCHEN_RUNS, kitchen_variants, _kitchen_texts))
    records.extend(_export_environment("living_room", LIVING_RUNS, living_variants, _living_texts))
    records.extend(_export_environment(
        "workshop", WORKSHOP_RUNS,
        workshop_feasible + workshop_infeasible, _workshop_texts,
    ))
    manifest = {
        "schema_version": 1,
        "root": "GT_everything",
        "total_environments": 3,
        "total_variants": len(records),
        "environment_counts": {
            name: sum(row["environment"] == name for row in records)
            for name in ("kitchen", "living_room", "workshop")
        },
        "required_files_per_variant": [
            "robot_execution.mp4",
            "robot_action_sequence.txt",
            "function_object_assignments.txt",
        ],
        "records": records,
    }
    _write_json(OUTPUT / "manifest.json", manifest)
    _write_text(
        OUTPUT / "README.md",
        """# GT_everything

Standardized ground-truth robot-execution evidence for all three environments.

Every variant directory contains exactly the requested core evidence:

- `robot_execution.mp4`
- `robot_action_sequence.txt`
- `function_object_assignments.txt`

Counts: 16 Kitchen variants, 10 Living Room variants, and 10 Workshop variants
(36 total). Feasible videos execute the full assigned task. Infeasible videos
show the available inspection/termination behavior and their text files record
that no complete global assignment exists.

Workshop videos use corrected physics-assisted execution: the base follows a
collision-audited front-corridor route; the hand tracks live drawer/door
handles; grasp constraints carry payloads; and the repair aligns the medium
Phillips screw to the actual recessed hole before guided insertion, axial
preload, wrist rotation, helical advance, and measured seated-state validation.
Tiny-object and driver reorientation use visible compliant alignment fixtures
and remain explicitly logged.
Kitchen remains an explicitly assisted GT
demonstration. Living Room uses its existing physical mobile-manipulation
execution with actuator-driven navigation, IK, grasp welds, and measured support
validation. See `manifest.json` for hashes and video metadata.""",
    )
    print(
        f"GT_everything: {len(records)} variants exported "
        f"({manifest['environment_counts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
