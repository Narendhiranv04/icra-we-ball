"""Offline validation for saved function-aware usage-policy runs.

Runtime inference never imports this module or its evaluation annotations.
The evaluator compares the policy modes already saved at every stage; it does
not rerender, rerun YOLO, rebuild point clouds, or modify observations.
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
    / "ablation2_count_reuse_evaluation.yaml"
)
POLICY_MODES = ("always-reusable", "always-distinct", "function-aware")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_directories(run_path: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted((run_path / "stages").iterdir()):
        if path.is_dir() and path.name[:3].isdigit():
            result[int(path.name[:3])] = path
    if not result:
        raise ValueError(f"No saved stages under {run_path}")
    return result


def _fingerprint_paths(run_path: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(set(paths)):
        digest.update(relative.encode("utf-8"))
        path = run_path / relative
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate_saved_usage_policy_run(
    run_dir: str | Path,
    *,
    evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
    output_name: str = "offline_policy_ablation_evaluation.json",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all saved policy modes against offline-only expectations."""
    run_path = Path(run_dir).resolve()
    run_config = _load(run_path / "run_config.json")
    summary = _load(run_path / "policy_ablation_summary.json")
    evaluation_path = Path(evaluation_config).resolve()
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("purpose") != "OFFLINE_EVALUATION_ONLY"
        or not evaluation.get("runtime_import_forbidden")
    ):
        raise ValueError("Evaluation annotations are not marked offline-only")
    if run_config.get("scene_name") != evaluation.get("scene_name"):
        raise ValueError(
            f"Run scene {run_config.get('scene_name')!r} does not match "
            f"evaluation scene {evaluation.get('scene_name')!r}"
        )
    if summary.get("task_id") != evaluation.get("task_id"):
        raise ValueError("Run and evaluation task IDs do not match")

    stage_dirs = _stage_directories(run_path)
    stage_records = {
        int(record["stage"]): record for record in summary["stages"]
    }
    if set(stage_records) != set(stage_dirs):
        raise ValueError("Policy summary and saved stage directories differ")
    comparisons = {
        stage: _load(directory / "policy_mode_comparison.json")
        for stage, directory in stage_dirs.items()
    }
    if not all(
        comparison.get("same_observation_evidence")
        for comparison in comparisons.values()
    ):
        raise ValueError("A policy mode was not marked as same-evidence")

    expected_sequence = list(run_config.get("inspection_sequence", []))
    observed_regions = [
        stage_records[stage]["region_id"]
        for stage in sorted(stage_records)
        if stage_records[stage]["region_id"] != "INITIAL"
    ]
    full_sequence_observed = observed_regions == expected_sequence
    evidence_by_stage: dict[str, Any] = {}
    for stage, comparison in comparisons.items():
        measurement_paths = comparison.get("measurement_cloud_paths", [])
        semantic_paths = comparison.get("semantic_record_paths", [])
        all_paths = [*measurement_paths, *semantic_paths]
        evidence_by_stage[str(stage)] = {
            "measurement_cloud_paths": measurement_paths,
            "semantic_record_paths": semantic_paths,
            "fingerprint_sha256": _fingerprint_paths(run_path, all_paths),
            "consumed_once_by_all_policy_modes": True,
        }

    mode_reports: dict[str, Any] = {}
    expected_modes = evaluation["expected"]
    for mode in POLICY_MODES:
        complete_stage = next(
            (
                stage
                for stage in sorted(stage_records)
                if stage_records[stage]["modes"][mode]["status"]
                == "COMPLETE"
            ),
            None,
        )
        if complete_stage is not None:
            actual_status = "COMPLETE"
            terminal_stage = complete_stage
        elif full_sequence_observed:
            actual_status = "EXHAUSTED"
            terminal_stage = max(stage_records)
        else:
            actual_status = "INCOMPLETE_AT_OBSERVATION_END"
            terminal_stage = max(stage_records)
        terminal = comparisons[terminal_stage]["modes"][mode]
        expected = expected_modes[mode]
        expected_stage = expected.get("completion_stage")
        expected_region = expected.get("completion_region")
        actual_region = stage_records[terminal_stage]["region_id"]
        expected_distinct = expected.get("distinct_physical_tool_count")
        expected_policy_distinct = expected.get(
            "policy_required_distinct_physical_tool_count"
        )
        checks = {
            "status": actual_status == expected["status"],
            "completion_stage": (
                expected_stage is None
                or complete_stage == int(expected_stage)
            ),
            "completion_region": (
                expected_region is None
                or actual_region == expected_region
            ),
            "distinct_physical_tool_count": (
                expected_distinct is None
                or terminal["distinct_physical_tool_count"]
                == int(expected_distinct)
            ),
            "policy_required_distinct_physical_tool_count": (
                expected_policy_distinct is None
                or terminal[
                    "policy_required_distinct_physical_tool_count"
                ]
                == int(expected_policy_distinct)
            ),
        }
        mode_reports[mode] = {
            "mode": mode,
            "production_mode": mode == "function-aware",
            "diagnostic_ablation": mode != "function-aware",
            "actual_status": actual_status,
            "completion_stage": complete_stage,
            "terminal_stage": terminal_stage,
            "completion_region": (
                stage_records[complete_stage]["region_id"]
                if complete_stage is not None
                else None
            ),
            "regions_observed": observed_regions[:terminal_stage],
            "distinct_physical_tool_count": terminal[
                "distinct_physical_tool_count"
            ],
            "policy_required_distinct_physical_tool_count": terminal[
                "policy_required_distinct_physical_tool_count"
            ],
            "satisfied_target_slot_count": terminal[
                "satisfied_target_slot_count"
            ],
            "required_target_slot_count": terminal[
                "required_target_slot_count"
            ],
            "operation_assignments": terminal[
                "operation_assignments"
            ],
            "function_group_evaluations": terminal[
                "function_group_evaluations"
            ],
            "expected": expected,
            "checks": checks,
            "matches_expected_result": all(checks.values()),
            "terminal_evidence_artifact": str(
                (
                    stage_dirs[terminal_stage]
                    / "policy_mode_comparison.json"
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
        "production_mode": "function-aware",
        "policy_modes_share_identical_observation_evidence": True,
        "perception_was_not_rerun": True,
        "full_fixed_sequence_observed": full_sequence_observed,
        "observed_regions": observed_regions,
        "evidence_by_stage": evidence_by_stage,
        "modes": mode_reports,
        "all_expected_results_matched": all(
            record["matches_expected_result"]
            for record in mode_reports.values()
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
        description="Validate saved function-aware usage-policy ablations"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=DEFAULT_EVALUATION_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional writable output path outside a Docker-owned run",
    )
    arguments = parser.parse_args()
    report = evaluate_saved_usage_policy_run(
        arguments.run_dir,
        evaluation_config=arguments.evaluation_config,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "output": str(
                    arguments.output
                    or arguments.run_dir
                    / "offline_policy_ablation_evaluation.json"
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
