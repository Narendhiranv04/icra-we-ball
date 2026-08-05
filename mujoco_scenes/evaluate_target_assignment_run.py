"""Offline validation for saved Ablation 3 target-assignment runs.

Runtime inference never imports this module or the evaluation annotations.
All four modes are evaluated from comparison records created from one saved
observation graph; this module neither renders nor reruns perception.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EVALUATION_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "ablation3_multi_target_evaluation.yaml"
)
ASSIGNMENT_MODES = (
    "semantic-only",
    "geometry-only",
    "joint-target-agnostic-count",
    "joint-target-specific",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stages(run_path: Path) -> dict[int, Path]:
    result = {
        int(path.name[:3]): path
        for path in sorted((run_path / "stages").iterdir())
        if path.is_dir() and path.name[:3].isdigit()
    }
    if not result:
        raise ValueError(f"No saved stages under {run_path}")
    return result


def _fingerprint(run_path: Path, relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(set(relative_paths)):
        digest.update(relative.encode("utf-8"))
        path = run_path / relative
        if not path.is_file():
            raise ValueError(f"Evidence path does not exist: {relative}")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _selected_tool_labels(witness: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(assignment.get("semantic", {}).get("canonical_label"))
            for assignment in witness.get("operation_assignments", [])
            if assignment.get("semantic", {}).get("canonical_label")
        }
    )


def evaluate_saved_target_assignment_run(
    run_dir: str | Path,
    *,
    evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
    output_name: str = "offline_assignment_ablation_evaluation.json",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the four saved modes against offline-only expectations."""
    run_path = Path(run_dir).resolve()
    run_config = _load(run_path / "run_config.json")
    summary = _load(run_path / "assignment_ablation_summary.json")
    evaluation_path = Path(evaluation_config).resolve()
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("purpose") != "OFFLINE_EVALUATION_ONLY"
        or not evaluation.get("runtime_import_forbidden")
    ):
        raise ValueError("Evaluation annotations are not offline-only")
    if run_config.get("scene_name") != evaluation.get("scene_name"):
        raise ValueError("Run and evaluation scene names do not match")
    if summary.get("task_id") != evaluation.get("task_id"):
        raise ValueError("Run and evaluation task IDs do not match")

    stage_dirs = _stages(run_path)
    stage_records = {
        int(item["stage"]): item for item in summary["stages"]
    }
    if set(stage_records) != set(stage_dirs):
        raise ValueError("Summary and saved stages differ")
    comparisons = {
        stage: _load(path / "assignment_mode_comparison.json")
        for stage, path in stage_dirs.items()
    }
    if not all(
        item.get("same_observation_evidence") is True
        and item.get("perception_rerun_per_mode") is False
        and set(item.get("modes", {})) == set(ASSIGNMENT_MODES)
        for item in comparisons.values()
    ):
        raise ValueError("Modes are not backed by one shared observation")

    evidence_by_stage = {}
    for stage, comparison in comparisons.items():
        paths = [
            *comparison.get("measurement_cloud_paths", []),
            *comparison.get("semantic_record_paths", []),
        ]
        evidence_by_stage[str(stage)] = {
            "paths": paths,
            "fingerprint_sha256": _fingerprint(run_path, paths),
            "consumed_by_all_modes": True,
        }

    mode_reports = {}
    for mode in ASSIGNMENT_MODES:
        completion_stage = next(
            (
                stage
                for stage in sorted(comparisons)
                if comparisons[stage]["modes"][mode]["status"]
                == "COMPLETE"
            ),
            None,
        )
        expected = evaluation["expected"][mode]
        terminal_stage = (
            completion_stage
            if completion_stage is not None
            else max(comparisons)
        )
        terminal = comparisons[terminal_stage]["modes"][mode]
        actual_status = (
            "COMPLETE" if completion_stage is not None else terminal["status"]
        )
        actual_region = (
            stage_records[completion_stage]["region_id"]
            if completion_stage is not None
            else None
        )
        selected_labels = _selected_tool_labels(terminal)
        required_label = expected.get("required_selected_tool_label")
        checks = {
            "status": actual_status == expected["status"],
            "completion_stage": completion_stage
            == expected.get("completion_stage"),
            "completion_region": actual_region
            == expected.get("completion_region"),
            "required_selected_tool_label": (
                required_label is None or required_label in selected_labels
            ),
        }
        mode_reports[mode] = {
            "mode": mode,
            "production_mode": mode == "joint-target-specific",
            "diagnostic_ablation": mode != "joint-target-specific",
            "intentionally_incorrect": bool(
                expected.get("intentionally_incorrect", False)
            ),
            "failure_kind": expected.get("failure_kind"),
            "actual_status": actual_status,
            "completion_stage": completion_stage,
            "completion_region": actual_region,
            "selected_tool_labels": selected_labels,
            "selected_witness": terminal.get("selected_witness"),
            "operation_assignments": terminal.get(
                "operation_assignments", []
            ),
            "function_group_evaluations": terminal.get(
                "function_group_evaluations", []
            ),
            "checks": checks,
            "matches_expected_result": all(checks.values()),
            "evidence_artifact": str(
                (
                    stage_dirs[terminal_stage]
                    / "assignment_mode_comparison.json"
                ).relative_to(run_path)
            ),
        }

    report = {
        "schema_version": 1,
        "purpose": "OFFLINE_EVALUATION_ONLY",
        "runtime_inference_used_evaluation_annotations": False,
        "scene_name": run_config["scene_name"],
        "task_id": summary["task_id"],
        "evaluation_config": str(evaluation_path),
        "production_mode": "joint-target-specific",
        "modes_share_identical_observation_evidence": True,
        "perception_was_not_rerun": True,
        "evidence_by_stage": evidence_by_stage,
        "modes": mode_reports,
        "all_expected_results_matched": all(
            item["matches_expected_result"]
            for item in mode_reports.values()
        ),
    }
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else run_path / output_name
    )
    _atomic_json(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate saved Ablation 3 assignment modes"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate_saved_target_assignment_run(
        arguments.run_dir,
        evaluation_config=arguments.evaluation_config,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "all_expected_results_matched": report[
                    "all_expected_results_matched"
                ],
                "output": str(
                    arguments.output
                    or arguments.run_dir
                    / "offline_assignment_ablation_evaluation.json"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
