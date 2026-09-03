"""Run native or single-call discovery-framework execution trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .artifacts import write_json


GOALS = {
    "kitchen": (
        "Prepare and serve coffee and soup for two people using the available "
        "kitchenware. Stir both coffees and provide each soup bowl with a "
        "suitable utensil."
    ),
    "living_room": (
        "Prepare the living room for two people watching television. Place one "
        "cup and one saucer on each person's fixed individual side table, and "
        "place the TV remote on the fixed shared coffee table."
    ),
}


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("value must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=tuple(GOALS), required=True)
    parser.add_argument("--protocol", choices=("native", "single_call"), required=True)
    parser.add_argument("--variants", type=_csv)
    parser.add_argument("--camera-counts", type=_csv, default=("5",))
    parser.add_argument("--seeds", type=_csv, default=("0",))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-replans", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _command(
    arguments: argparse.Namespace,
    variant: str,
    camera_count: int,
    seed: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        f"mujoco_scenes.run_{arguments.environment}_discovery_replanning",
        "--variant",
        variant,
        "--output-dir",
        str(output),
        "--goal",
        arguments.goal or GOALS[arguments.environment],
        "--base-url",
        arguments.base_url,
        "--model",
        arguments.model,
        "--protocol",
        arguments.protocol,
        "--camera-count",
        str(camera_count),
        "--seed",
        str(seed),
        "--max-replans",
        str(arguments.max_replans),
        "--max-actions",
        str(arguments.max_actions),
        "--max-tokens",
        str(arguments.max_tokens),
        "--timeout-seconds",
        str(arguments.timeout_seconds),
        "--headless",
        "--thinking" if arguments.thinking else "--no-thinking",
    ]


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    default_count = 6
    prefix = "K" if arguments.environment == "kitchen" else "L"
    variants = arguments.variants or tuple(
        f"{prefix}{index}" for index in range(1, default_count + 1)
    )
    try:
        camera_counts = tuple(int(value) for value in arguments.camera_counts)
        seeds = tuple(int(value) for value in arguments.seeds)
    except ValueError as error:
        parser.error(str(error))
    if not camera_counts or not set(camera_counts) <= {1, 3, 5}:
        parser.error("--camera-counts supports only 1,3,5")
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error("--seeds must contain non-negative integers")

    root = arguments.output_root.resolve()
    summary_path = root / "batch_summary.json"
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    if arguments.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in previous.get("runs", ()):
            rows[(str(row["variant"]), int(row["camera_count"]), int(row["seed"]))] = row

    write_json(
        root / "protocol_manifest.json",
        {
            "schema_version": 1,
            "method": "discovery_replanning",
            "protocol": arguments.protocol,
            "environment": arguments.environment,
            "model": arguments.model,
            "goal": arguments.goal or GOALS[arguments.environment],
            "model_call_limit": 1 if arguments.protocol == "single_call" else None,
            "replan_limit": arguments.max_replans,
            "physical_execution": True,
        },
    )
    for variant in variants:
        for camera_count in camera_counts:
            for seed in seeds:
                output = root / variant / f"images_{camera_count}" / f"seed_{seed:03d}"
                result_path = output / "discovery_replanning_result.json"
                key = (variant, camera_count, seed)
                if arguments.resume and result_path.is_file():
                    print(f"[discovery-batch] skip {variant} images={camera_count} seed={seed}", flush=True)
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    rows[key] = _row(arguments, variant, camera_count, seed, output, 0, result)
                    write_json(summary_path, {"schema_version": 1, "runs": list(rows.values())})
                    continue
                if output.exists() and any(output.iterdir()):
                    parser.error(f"incomplete output directory is not empty: {output}")
                print(f"[discovery-batch] {variant} images={camera_count} seed={seed}", flush=True)
                completed = subprocess.run(_command(arguments, variant, camera_count, seed, output), check=False)
                result = (
                    json.loads(result_path.read_text(encoding="utf-8"))
                    if result_path.is_file()
                    else None
                )
                rows[key] = _row(
                    arguments, variant, camera_count, seed, output,
                    completed.returncode, result,
                )
                write_json(summary_path, {"schema_version": 1, "runs": list(rows.values())})
                if result is None and completed.returncode and not arguments.continue_on_error:
                    raise SystemExit(completed.returncode)


def _row(
    arguments: argparse.Namespace,
    variant: str,
    camera_count: int,
    seed: int,
    output: Path,
    return_code: int,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "method": "discovery_replanning",
        "protocol": arguments.protocol,
        "environment": arguments.environment,
        "variant": variant,
        "camera_count": camera_count,
        "seed": seed,
        "return_code": return_code,
        "result_present": result is not None,
        "task_success": result.get("success") if result else None,
        "output_dir": str(output),
    }


if __name__ == "__main__":
    main()
