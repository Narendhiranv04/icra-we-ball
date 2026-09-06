"""Resumable, immutable Stage-24 ViLaIn-TAMP-Qwen benchmark harness."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.request import urlopen

import yaml

from .artifacts import atomic_write_json, atomic_write_text
from .live_fm import DEFAULT_VLLM_BASE_URL


PROTOCOLS = ("initial_observation_only", "fixed_full_inspection")
REPEATS = 5
CP_LIMIT = 3
SERVED_MODEL_ID = "qwen35-9b"
SERVED_MODEL_ROOT = "Qwen/Qwen3.5-9B"
FAST_DOWNWARD = Path("/home/naren/fast-downward-24.06.1/fast-downward.py")
VAL = Path("/home/naren/VAL/build/vilain-release/bin/Validate")
VAL_COMMIT = "3c7a1f330bdab0ba28a4762bb45c3f06c27fb6d4"
CONFIG_FILES = {
    "kitchen": "kitchen_feasibility_variants.yaml",
    "living_room": "living_room_variants.yaml",
    "workshop": "workshop_variants.yaml",
}
TERMINAL_STATUSES = {
    "SUCCESS",
    "FM_OBJECT_FAILURE",
    "FM_STATE_FAILURE",
    "FM_GOAL_FAILURE",
    "INVALID_PDDL",
    "NO_PLAN",
    "VAL_FAILURE",
    "IDENTITY_FAILURE",
    "REFINEMENT_FAILURE",
    "EXECUTION_FAILURE",
    "PREDICTED_INFEASIBLE",
    "TIMEOUT",
    "INFRASTRUCTURE_FAILURE",
    "BENCHMARK_FAILURE",
}


@dataclass(frozen=True)
class ScheduledRun:
    run_id: str
    domain: str
    variant: str
    observation_protocol: str
    repeat_index: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def authoritative_variants(config_root: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for domain, filename in CONFIG_FILES.items():
        document = yaml.safe_load((config_root / filename).read_text(encoding="utf-8"))
        variants = document.get("variants") if isinstance(document, Mapping) else None
        if not isinstance(variants, Mapping) or not variants:
            raise ValueError(f"{filename} has no authoritative variant registry")
        result[domain] = tuple(str(item) for item in variants)
    return result


def build_schedule(config_root: Path) -> tuple[ScheduledRun, ...]:
    schedule: list[ScheduledRun] = []
    ordinal = 0
    for domain, variants in authoritative_variants(config_root).items():
        for variant in variants:
            for protocol in PROTOCOLS:
                for repeat in range(REPEATS):
                    schedule.append(
                        ScheduledRun(
                            run_id=(
                                f"{domain}__{variant}__{protocol}__r{repeat:02d}"
                            ),
                            domain=domain,
                            variant=variant,
                            observation_protocol=protocol,
                            repeat_index=repeat,
                            seed=240000 + ordinal,
                        )
                    )
                    ordinal += 1
    return tuple(schedule)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def prepare_manifest(
    output_root: Path, *, source_commit: str, config_root: Path
) -> tuple[ScheduledRun, ...]:
    schedule = build_schedule(config_root)
    payload = {
        "schema_version": 1,
        "source_commit": source_commit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_schedule": True,
        "model": {
            "served_model_id": SERVED_MODEL_ID,
            "served_model_root": SERVED_MODEL_ROOT,
            "served_revision": "unverified",
            "base_url": DEFAULT_VLLM_BASE_URL,
            "openai_cloud_used": False,
        },
        "runtime": {
            "cp_limit": CP_LIMIT,
            "fast_downward": str(FAST_DOWNWARD),
            "fast_downward_version": "24.06.1",
            "search_alias": "lama-first",
            "symbolic_timeout_seconds": 200,
            "val": str(VAL),
            "val_commit": VAL_COMMIT,
            "mujoco_version": "3.3.5",
            "repeat_count": REPEATS,
            "protocols": list(PROTOCOLS),
        },
        "runs": [item.to_dict() for item in schedule],
    }
    path = output_root / "schedule_manifest.json"
    rendered = _canonical_json(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable["created_utc"] = payload["created_utc"]
        if comparable != payload:
            raise RuntimeError("existing immutable schedule differs from requested matrix")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        (output_root / "schedule_manifest.sha256").write_text(
            f"{digest}  schedule_manifest.json\n", encoding="utf-8"
        )
    return schedule


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


def _require_frozen_source(repository_root: Path, expected_commit: str) -> None:
    if _git_head(repository_root) != expected_commit:
        raise RuntimeError("working source is not the schedule's frozen commit")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository_root, text=True
    )
    if status.strip():
        raise RuntimeError("official matrix requires a clean working tree")


def _probe_runtime() -> None:
    if not FAST_DOWNWARD.is_file() or not VAL.is_file():
        raise RuntimeError("planner or validator binary is unavailable")
    try:
        with urlopen(f"{DEFAULT_VLLM_BASE_URL}/models", timeout=10) as response:
            payload = json.load(response)
    except Exception as error:
        raise RuntimeError("Qwen tunnel required") from error
    rows = payload.get("data", ()) if isinstance(payload, Mapping) else ()
    if len(rows) != 1 or rows[0].get("id") != SERVED_MODEL_ID:
        raise RuntimeError("unexpected served model identity")


def _command(item: ScheduledRun, artifact_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mujoco_scenes.run_vilain_tamp_baseline",
        "--domain",
        item.domain,
        "--variant",
        item.variant,
        "--observation-mode",
        item.observation_protocol,
        "--model-condition",
        "vilain_tamp_qwen",
        "--live",
        "--execute",
        "--output-directory",
        str(artifact_root),
        "--cp-limit",
        str(CP_LIMIT),
        "--fast-downward",
        str(FAST_DOWNWARD),
        "--val",
        str(VAL),
        "--seed",
        str(item.seed),
    ]


def _semantic_status(result: Mapping[str, Any]) -> str:
    run_status = str(result.get("run_status", ""))
    if run_status == "SUCCESS":
        return "SUCCESS"
    if run_status == "BENCHMARK_FAILURE":
        return "BENCHMARK_FAILURE"
    if run_status == "INFEASIBLE_CORRECT":
        return "PREDICTED_INFEASIBLE"
    if str(result.get("execution_status", "")).startswith("FAILED"):
        return "EXECUTION_FAILURE"
    metrics = result.get("metrics", {})
    distribution = metrics.get("failure_stage_distribution", {}) if isinstance(metrics, Mapping) else {}
    keys = {str(key).upper() for key in distribution}
    if any("IDENTITY" in key or "ENTITY" in key for key in keys):
        return "IDENTITY_FAILURE"
    if any(key in {"IK", "COLLISION", "SKILL_ENVELOPE", "REFINEMENT"} for key in keys):
        return "REFINEMENT_FAILURE"
    planning = str(result.get("planning_status", ""))
    if planning in {"EXHAUSTED", "REPEATED_REVISION", "INVALID_CORRECTION"}:
        return "PREDICTED_INFEASIBLE"
    return "EXECUTION_FAILURE"


def _failure_status(stderr: str) -> str:
    lower = stderr.lower()
    if any(token in lower for token in ("qwen tunnel required", "connection refused", "connection error")):
        return "INFRASTRUCTURE_FAILURE"
    if "object_estimation transport failed" in lower:
        return "FM_OBJECT_FAILURE"
    if "initial_state transport failed" in lower:
        return "FM_STATE_FAILURE"
    if "goal_state transport failed" in lower:
        return "FM_GOAL_FAILURE"
    if "timed out" in lower or "timeout" in lower:
        return "TIMEOUT"
    if "val" in lower and "failed" in lower:
        return "VAL_FAILURE"
    if "pddl" in lower:
        return "INVALID_PDDL"
    return "INFRASTRUCTURE_FAILURE"


CommandRunner = Callable[[Sequence[str], Path, Path, Path], int]


def _subprocess_runner(command: Sequence[str], cwd: Path, stdout: Path, stderr: Path) -> int:
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=out,
            stderr=err,
            check=False,
            timeout=1800,
        ).returncode


def run_one(
    item: ScheduledRun,
    *,
    output_root: Path,
    repository_root: Path,
    source_commit: str,
    command_runner: CommandRunner = _subprocess_runner,
) -> Mapping[str, Any]:
    run_root = output_root / "runs" / item.run_id
    terminal_path = run_root / "terminal_status.json"
    if terminal_path.is_file():
        existing = json.loads(terminal_path.read_text(encoding="utf-8"))
        if existing.get("terminal_status") != "INFRASTRUCTURE_FAILURE":
            return existing
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root = run_root / "artifacts"
    stdout = run_root / "stdout.log"
    stderr = run_root / "stderr.log"
    started = datetime.now(timezone.utc)
    command = _command(item, artifact_root)
    try:
        _probe_runtime()
        return_code = command_runner(command, repository_root, stdout, stderr)
        result_path = artifact_root / "baseline_run_result.json"
        if result_path.is_file():
            baseline = json.loads(result_path.read_text(encoding="utf-8"))
            terminal_status = _semantic_status(baseline)
        else:
            baseline = None
            terminal_status = _failure_status(
                stderr.read_text(encoding="utf-8") if stderr.exists() else ""
            )
    except subprocess.TimeoutExpired as error:
        return_code = None
        baseline = None
        terminal_status = "TIMEOUT"
        atomic_write_text(stderr, f"TimeoutExpired: {error}\n")
    except Exception as error:
        return_code = None
        baseline = None
        terminal_status = "INFRASTRUCTURE_FAILURE"
        atomic_write_text(stderr, f"{type(error).__name__}: {error}\n")
    completed = datetime.now(timezone.utc)
    payload = {
        **item.to_dict(),
        "source_commit": source_commit,
        "terminal_status": terminal_status,
        "return_code": return_code,
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "command": command,
        "artifact_root": str(artifact_root),
        "baseline_result": baseline,
    }
    if terminal_status not in TERMINAL_STATUSES:
        raise AssertionError(f"unknown terminal status {terminal_status}")
    atomic_write_json(terminal_path, payload)
    return payload


def _flat_row(result: Mapping[str, Any]) -> dict[str, Any]:
    baseline = result.get("baseline_result")
    metrics = baseline.get("metrics", {}) if isinstance(baseline, Mapping) else {}
    calls = metrics.get("model_calls_by_type", {}) if isinstance(metrics, Mapping) else {}
    hidden_path = Path(str(result.get("artifact_root", ""))) / "benchmark" / "benchmark_goal_evaluation.json"
    hidden = (
        json.loads(hidden_path.read_text(encoding="utf-8"))
        if hidden_path.is_file()
        else {}
    )
    ground_truth_feasible = hidden.get("ground_truth_feasibility")
    predicted_infeasible = hidden.get("predicted_infeasible")
    return {
        "run_id": result["run_id"],
        "domain": result["domain"],
        "variant": result["variant"],
        "observation_protocol": result["observation_protocol"],
        "repeat_index": result["repeat_index"],
        "seed": result["seed"],
        "source_commit": result["source_commit"],
        "terminal_status": result["terminal_status"],
        "actual_benchmark_success": metrics.get("actual_benchmark_success"),
        "generated_goal_satisfied": metrics.get("generated_goal_satisfied"),
        "correct_infeasibility_recognition": metrics.get("correct_infeasibility_recognition"),
        "ground_truth_feasible": ground_truth_feasible,
        "predicted_infeasible": predicted_infeasible,
        "false_feasible": ground_truth_feasible is False and predicted_infeasible is False,
        "false_infeasible": ground_truth_feasible is True and predicted_infeasible is True,
        "planning_status": metrics.get("planning_status"),
        "fm_calls": metrics.get("model_call_count"),
        "object_calls": calls.get("object_estimation"),
        "init_calls": calls.get("initial_state"),
        "goal_calls": calls.get("goal_state"),
        "cp_calls": calls.get("corrective_planning"),
        "inspections": metrics.get("inspected_region_count"),
        "physical_actions": metrics.get("controller_action_count"),
        "plan_length": metrics.get("symbolic_plan_length"),
        "planning_seconds": metrics.get("symbolic_planning_seconds"),
        "refinement_seconds": metrics.get("geometric_refinement_seconds"),
        "execution_seconds": metrics.get("execution_seconds"),
        "total_seconds": metrics.get("end_to_end_seconds", result.get("elapsed_seconds")),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["terminal_status"] != "INFRASTRUCTURE_FAILURE"]
    return {
        "scheduled_runs": len(rows),
        "completed_runs": len(completed),
        "infrastructure_failures": len(rows) - len(completed),
        "task_successes": sum(row.get("actual_benchmark_success") is True for row in completed),
        "task_success_rate": (
            sum(row.get("actual_benchmark_success") is True for row in completed) / len(completed)
            if completed else None
        ),
        "generated_goal_satisfaction_rate": (
            sum(row.get("generated_goal_satisfied") is True for row in completed)
            / len([row for row in completed if row.get("generated_goal_satisfied") is not None])
            if any(row.get("generated_goal_satisfied") is not None for row in completed)
            else None
        ),
        "infeasibility_accuracy": _mean(completed, "correct_infeasibility_recognition"),
        "false_feasible": sum(row.get("false_feasible") is True for row in completed),
        "false_infeasible": sum(row.get("false_infeasible") is True for row in completed),
        "average_fm_calls": _mean(completed, "fm_calls"),
        "average_cp_calls": _mean(completed, "cp_calls"),
        "average_planning_seconds": _mean(completed, "planning_seconds"),
        "average_refinement_seconds": _mean(completed, "refinement_seconds"),
        "average_execution_seconds": _mean(completed, "execution_seconds"),
        "average_total_seconds": _mean(completed, "total_seconds"),
        "terminal_status_counts": {
            status: sum(row["terminal_status"] == status for row in rows)
            for status in sorted({str(row["terminal_status"]) for row in rows})
        },
    }


def write_aggregates(output_root: Path, schedule: Sequence[ScheduledRun]) -> Mapping[str, Any]:
    results = []
    for item in schedule:
        path = output_root / "runs" / item.run_id / "terminal_status.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            results.append({**item.to_dict(), "source_commit": None, "terminal_status": "INCOMPLETE"})
    rows = [_flat_row(item) for item in results]
    fieldnames = list(rows[0]) if rows else []
    with (output_root / "run_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(
        output_root / "run_results.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    for group_key, filename in (
        ("domain", "aggregate_by_domain.csv"),
        ("observation_protocol", "aggregate_by_protocol.csv"),
        ("variant", "aggregate_by_variant.csv"),
    ):
        groups = sorted({str(row[group_key]) for row in rows})
        aggregates = [{group_key: group, **_aggregate([row for row in rows if row[group_key] == group])} for group in groups]
        keys = [group_key, "scheduled_runs", "completed_runs", "infrastructure_failures", "task_successes", "task_success_rate", "generated_goal_satisfaction_rate", "infeasibility_accuracy", "false_feasible", "false_infeasible", "average_fm_calls", "average_cp_calls", "average_planning_seconds", "average_refinement_seconds", "average_execution_seconds", "average_total_seconds", "terminal_status_counts"]
        with (output_root / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(aggregates)
    overall = _aggregate(rows)
    atomic_write_json(output_root / "aggregate_overall.json", overall)
    manifest = json.loads((output_root / "schedule_manifest.json").read_text(encoding="utf-8"))
    source = manifest["source_commit"]
    accounted = len({row["run_id"] for row in rows}) == len(schedule)
    audit = {
        "all_scheduled_run_ids_accounted_for": accounted,
        "duplicate_run_ids": len({row["run_id"] for row in rows}) != len(rows),
        "completed_source_commits_match": all(
            row["source_commit"] in {source, None} for row in rows
        ),
        "served_model_id": SERVED_MODEL_ID,
        "served_revision": "unverified",
        "openai_cloud_used": False,
        "expected_gt_plan_used": False,
        "proposed_grounding_used": False,
        "hidden_truth_planning_access": False,
        "cp_limit": CP_LIMIT,
        "schedule_sha256": (output_root / "schedule_manifest.sha256").read_text(encoding="utf-8").split()[0],
    }
    atomic_write_json(output_root / "experiment_audit.json", audit)
    return overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen ViLaIn Stage-24 matrix.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--max-runs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[3]
    config_root = repository_root / "mujoco_scenes" / "configs"
    source_commit = _git_head(repository_root)
    schedule = prepare_manifest(args.output_root.resolve(), source_commit=source_commit, config_root=config_root)
    if args.prepare_only:
        print(json.dumps({"scheduled_runs": len(schedule), "source_commit": source_commit}, sort_keys=True))
        return 0
    manifest = json.loads((args.output_root / "schedule_manifest.json").read_text(encoding="utf-8"))
    _require_frozen_source(repository_root, str(manifest["source_commit"]))
    if not args.aggregate_only:
        selected = schedule if args.max_runs is None else schedule[: args.max_runs]
        for index, item in enumerate(selected, 1):
            result = run_one(item, output_root=args.output_root, repository_root=repository_root, source_commit=source_commit)
            print(f"[{index}/{len(selected)}] {item.run_id}: {result['terminal_status']}", flush=True)
    overall = write_aggregates(args.output_root, schedule)
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0 if overall["infrastructure_failures"] == 0 and overall["completed_runs"] == len(schedule) else 2


if __name__ == "__main__":
    raise SystemExit(main())
