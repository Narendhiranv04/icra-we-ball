"""Offline ground-truth comparison for saved grounding runs.

This module is intentionally outside the runtime observation and resolver
path. It reads only completed run artifacts and the explicitly evaluation-only
annotation file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EVALUATION_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "joint_grounding_evaluation.yaml"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _selected_labels(mode_result: dict[str, Any]) -> dict[str, str | None]:
    return {
        str(record["role"]): record.get("canonical_label")
        for record in mode_result.get("selected_candidate_edges", [])
    }


def _stage_artifact(run_dir: Path, stage: int) -> Path:
    matches = sorted(
        path
        for path in (run_dir / "stages").iterdir()
        if path.is_dir() and path.name.startswith(f"{stage:03d}_")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one saved directory for stage {stage}, found {matches}"
        )
    return matches[0] / "grounding_mode_comparison.json"


def evaluate_saved_run(
    run_dir: str | Path,
    *,
    evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
    output_name: str = "offline_ablation_evaluation.json",
) -> dict[str, Any]:
    """Compare saved ablations with annotations without entering inference."""
    run_path = Path(run_dir).resolve()
    run_config = json.loads((run_path / "run_config.json").read_text())
    ablations = json.loads((run_path / "ablation_summary.json").read_text())
    evaluation = yaml.safe_load(Path(evaluation_config).read_text())
    if (
        evaluation.get("purpose") != "OFFLINE_EVALUATION_ONLY"
        or not evaluation.get("runtime_import_forbidden")
    ):
        raise ValueError("Evaluation annotations are not marked offline-only")
    scene_name = run_config["scene_name"]
    try:
        scene_evaluation = evaluation["scenes"][scene_name]
        expected_modes = scene_evaluation["expected"]
        ground_truth = scene_evaluation["ground_truth"]
    except KeyError as error:
        raise ValueError(
            f"No offline evaluation annotation for scene '{scene_name}'"
        ) from error

    saved_stages = ablations["stages"]
    stage_by_number = {int(record["stage"]): record for record in saved_stages}
    modes_report: dict[str, Any] = {}
    for mode in ("joint", "geometry-only", "semantic-only"):
        expected = expected_modes.get(mode)
        if expected is None:
            continue
        complete_stage_record = next(
            (
                record
                for record in saved_stages
                if record["modes"][mode]["status"] == "COMPLETE"
            ),
            None,
        )
        actual_status = (
            "COMPLETE" if complete_stage_record is not None else "EXHAUSTED"
        )
        completion_stage = (
            int(complete_stage_record["stage"])
            if complete_stage_record is not None
            else None
        )
        terminal_stage = (
            completion_stage
            if completion_stage is not None
            else max(stage_by_number)
        )
        stage_result = json.loads(
            _stage_artifact(run_path, terminal_stage).read_text()
        )["modes"][mode]
        labels = _selected_labels(stage_result)
        expected_status = str(expected.get("status", "COMPLETE"))
        expected_stage = expected.get("completion_stage")
        expected_labels = expected.get("selected_labels", {})
        matches_expected_runtime = (
            actual_status == expected_status
            and (
                expected_stage is None
                or completion_stage == int(expected_stage)
            )
            and (
                not expected_labels
                or labels == expected_labels
            )
        )
        runtime_mismatch_reasons = []
        if actual_status != expected_status:
            runtime_mismatch_reasons.append(
                f"status {actual_status}, expected {expected_status}"
            )
        if (
            expected_stage is not None
            and completion_stage != int(expected_stage)
        ):
            runtime_mismatch_reasons.append(
                f"completion stage {completion_stage}, "
                f"expected {expected_stage}"
            )
        if expected_labels and labels != expected_labels:
            runtime_mismatch_reasons.append(
                f"selected labels {labels}, expected {expected_labels}"
            )
        ground_truth_status = str(
            ground_truth.get("status", "COMPLETE")
        )
        ground_truth_labels = ground_truth.get(
            "selected_labels", {}
        )
        matches_ground_truth = (
            actual_status == ground_truth_status
            and labels == ground_truth_labels
        )
        incorrect_reason = None
        if not matches_ground_truth:
            incorrect_reason = ground_truth.get(
                "incorrect_selection_reasons", {}
            ).get(mode)
            if incorrect_reason is None:
                incorrect_reason = (
                    f"Selected {labels} with status {actual_status}; "
                    f"ground truth requires {ground_truth_labels} with "
                    f"status {ground_truth_status}."
                )
        regions_opened = [
            record["region_id"]
            for record in saved_stages
            if record["region_id"] != "INITIAL"
            and int(record["stage"]) <= terminal_stage
        ]
        modes_report[mode] = {
            "mode": mode,
            "diagnostic_ablation": mode != "joint",
            "actual_status": actual_status,
            "completion_stage": completion_stage,
            "regions_opened": regions_opened,
            "selected_role_assignments": stage_result.get(
                "selected_witness"
            ),
            "selected_labels": labels,
            "semantic_decisions": [
                {
                    "object_id": record["object_id"],
                    "role": record["role"],
                    **record["semantic"],
                }
                for record in stage_result.get(
                    "candidate_evaluations", []
                )
            ],
            "geometric_decisions": [
                {
                    "object_id": record["object_id"],
                    "role": record["role"],
                    "status": record["unary_geometry"]["status"],
                    "checks": record["unary_geometry"]["checks"],
                }
                for record in stage_result.get(
                    "candidate_evaluations", []
                )
            ],
            "relation_decisions": [
                relation
                for assignment in stage_result.get(
                    "assignment_evaluations", []
                )
                for relation in assignment.get("relation_checks", [])
            ],
            "expected": expected,
            "matches_expected_runtime_result": (
                matches_expected_runtime
            ),
            "runtime_mismatch_reason": (
                None
                if matches_expected_runtime
                else "; ".join(runtime_mismatch_reasons)
            ),
            "evaluation_ground_truth": ground_truth,
            "matches_evaluation_ground_truth": matches_ground_truth,
            "failure_reason": incorrect_reason,
            "evidence_artifact": str(
                _stage_artifact(run_path, terminal_stage).relative_to(
                    run_path
                )
            ),
        }

    report = {
        "schema_version": 1,
        "purpose": "OFFLINE_EVALUATION_ONLY",
        "runtime_inference_used_evaluation_annotations": False,
        "scene_name": scene_name,
        "task_id": ablations.get("task_id"),
        "shared_observation_evidence": bool(
            ablations.get("shared_observation_evidence")
        ),
        "evaluation_config": str(Path(evaluation_config).resolve()),
        "modes": modes_report,
        "all_expected_results_matched": all(
            record["matches_expected_runtime_result"]
            for record in modes_report.values()
        ),
    }
    _atomic_json(run_path / output_name, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved joint-grounding ablations offline"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG,
    )
    arguments = parser.parse_args()
    report = evaluate_saved_run(
        arguments.run_dir,
        evaluation_config=arguments.evaluation_config,
    )
    print(
        json.dumps(
            {
                "output": str(
                    arguments.run_dir
                    / "offline_ablation_evaluation.json"
                ),
                "all_expected_results_matched": report[
                    "all_expected_results_matched"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
