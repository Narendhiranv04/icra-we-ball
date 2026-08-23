"""Export redesigned Workshop plans, assignments, and five-view scene snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from .export_gt_everything import _format_action, _write_text
from .workshop_ground_truth_planner import load_variant_specs


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "workshop_fixed_pair_gt_execution"
IMAGES = ROOT / "docs" / "workshop_variant_visualizations"
OUTPUT = ROOT / "GT_everything" / "workshop"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    records = []
    for variant_id, spec in load_variant_specs().items():
        source = RUNS / variant_id
        plan = _read(source / "action_plan.json")["actions"]
        assignment = _read(source / "assignment.json")
        summary = _read(source / "summary.json")
        destination = OUTPUT / variant_id
        destination.mkdir(parents=True, exist_ok=True)

        action_rows = [
            "Environment: workshop",
            f"Variant: {variant_id}",
            "Execution profile: PHYSICS_ASSISTED_GT_EXECUTION",
            f"Validated actions: {summary['actions_completed']}/{summary['total_actions']}",
            "",
            *[_format_action(index, action) for index, action in enumerate(plan, 1)],
        ]
        _write_text(destination / "robot_action_sequence.txt", "\n".join(action_rows))

        assignment_rows = [
            "Environment: workshop",
            f"Variant: {variant_id}",
            f"Intended outcome: {assignment['intended_outcome']}",
            f"Regions inspected: {', '.join(spec['expected_inspection_regions'])}",
            "",
            "FUNCTION -> OBJECT ASSIGNMENTS",
        ]
        if assignment["is_feasible"]:
            assignment_rows.extend([
                f"CAN_DRIVE_SCREW -> {assignment['driver']}",
                f"CAN_FASTEN -> {assignment['fastener']}",
                f"FIXED_INSERTION_TARGET -> {assignment['target_joint']}",
                "DISTRACTOR_ONLY -> workshop_wooden_hammer",
            ])
        else:
            assignment_rows.extend([
                "COMPLETE_DRIVER_SCREW_PAIR -> NONE",
                f"REJECTION -> {assignment['rejection_reason']}",
                "DISTRACTOR_ONLY -> workshop_wooden_hammer",
            ])
        _write_text(
            destination / "function_object_assignments.txt",
            "\n".join(assignment_rows),
        )

        snapshot_source = IMAGES / variant_id / "five_camera_open_storage.png"
        snapshot_destination = destination / "five_camera_scene_snapshot.png"
        shutil.copy2(snapshot_source, snapshot_destination)
        shutil.copy2(source / "summary.json", destination / "execution_summary.json")
        records.append({
            "environment": "workshop",
            "variant": variant_id,
            "outcome": spec["intended_outcome"],
            "actions_validated": summary["actions_completed"],
            "action_count": summary["total_actions"],
            "actions": str((destination / "robot_action_sequence.txt").relative_to(ROOT)),
            "assignments": str((destination / "function_object_assignments.txt").relative_to(ROOT)),
            "snapshot": str(snapshot_destination.relative_to(ROOT)),
            "execution_summary": str((destination / "execution_summary.json").relative_to(ROOT)),
            "files": [
                "robot_action_sequence.txt",
                "function_object_assignments.txt",
                "five_camera_scene_snapshot.png",
                "execution_summary.json",
            ],
        })

    _write_text(
        OUTPUT / "README.md",
        "# Redesigned Workshop GT evidence\n\n"
        "Ten fixed-object position/presence variants. Each variant contains the "
        "validated GT action list, functional assignment, five-camera scene "
        "snapshot, and physical execution summary. Old 14-variant evidence was "
        "retired. New MP4 rendering is intentionally separate from the logical "
        "and physical validation run.\n",
    )
    _write_text(
        OUTPUT / "manifest.json",
        json.dumps({"schema_version": 2, "variants": records}, indent=2, sort_keys=True),
    )
    root_manifest_path = OUTPUT.parent / "manifest.json"
    if root_manifest_path.exists():
        root_manifest = _read(root_manifest_path)
        retained = [
            row for row in root_manifest.get("records", [])
            if row.get("environment") != "workshop"
        ]
        root_manifest["schema_version"] = 2
        root_manifest["records"] = retained + records
        root_manifest["total_variants"] = len(root_manifest["records"])
        root_manifest["environment_counts"] = {
            environment: sum(
                row.get("environment") == environment
                for row in root_manifest["records"]
            )
            for environment in ("kitchen", "living_room", "workshop")
        }
        root_manifest["workshop_evidence_note"] = (
            "Redesigned Workshop evidence contains validated physical traces and "
            "five-view snapshots; retired MP4s were removed and new MP4 rendering "
            "is a separate optional export."
        )
        _write_text(root_manifest_path, json.dumps(root_manifest, indent=2, sort_keys=True))
    print(f"Exported {len(records)} Workshop variants to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
