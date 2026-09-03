"""Run planning-to-GT baseline trials over Kitchen, Living Room, or Workshop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .artifacts import write_json


DEFAULT_KITCHEN_GOAL = (
    "Prepare and serve coffee and soup for two people using the available "
    "kitchenware. Stir both coffees and provide each soup bowl with a suitable "
    "utensil."
)


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("kitchen", "living_room", "workshop"), required=True)
    parser.add_argument(
        "--methods", type=_csv, default=("vlm_tamp", "owl_tamp"),
        help="Comma-separated vlm_tamp,owl_tamp.",
    )
    parser.add_argument("--variants", type=_csv)
    parser.add_argument("--seeds", type=_csv, default=("0",))
    parser.add_argument("--camera-counts", type=_csv, default=("5",))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument(
        "--protocol",
        choices=("native", "single_call"),
        default="native",
        help="Native method protocol or the minimum-call ablation.",
    )
    parser.add_argument(
        "--max-model-calls",
        type=int,
        default=1,
        help="VLM-TAMP calls per episode; one is the initial-plan condition.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs that already contain episode_result.json.",
    )
    return parser


def _command(
    method: str,
    environment: str,
    variant: str,
    seed: int,
    camera_count: int,
    output: Path,
    arguments: argparse.Namespace,
) -> list[str]:
    module = f"{method}_baseline.run_{environment}"
    command = [
        sys.executable,
        "-m",
        module,
        "--variant",
        variant,
        "--output-dir",
        str(output),
        "--seed",
        str(seed),
        "--camera-count",
        str(camera_count),
        "--max-tokens",
        str(arguments.max_tokens),
        "--protocol",
        arguments.protocol,
    ]
    if arguments.base_url:
        command.extend(("--base-url", arguments.base_url))
    if arguments.model:
        command.extend(("--model", arguments.model))
    if method == "retrieval":
        return command
    if method == "vlm_tamp" and environment in {"living_room", "workshop"}:
        command.extend(("--max-model-calls", str(arguments.max_model_calls)))
    if environment == "kitchen":
        goal = arguments.goal or DEFAULT_KITCHEN_GOAL
        command.extend(("--goal", goal))
        if method == "vlm_tamp":
            command.append("--planning-only")
    elif arguments.goal:
        command.extend(("--goal", arguments.goal))
    return command


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.max_model_calls != 1:
        parser.error(
            "Planning-to-GT native/single_call protocols both use one "
            "VLM-TAMP two-stage round; use a separately named replan experiment "
            "for --max-model-calls > 1"
        )
    allowed_methods = {"vlm_tamp", "owl_tamp", "retrieval"}
    if not set(arguments.methods) <= allowed_methods:
        parser.error("--methods supports only vlm_tamp,owl_tamp,retrieval")
    variants = arguments.variants or (
        tuple(f"K{index}" for index in range(1, 13))
        if arguments.environment == "kitchen"
        else tuple(f"L{index}" for index in range(1, 11))
        if arguments.environment == "living_room"
        else tuple(f"W{index}" for index in range(1, 11))
    )
    try:
        seeds = tuple(int(value) for value in arguments.seeds)
    except ValueError as error:
        parser.error(f"--seeds must contain integers: {error}")
    if any(seed < 0 for seed in seeds):
        parser.error("--seeds must be non-negative")
    try:
        camera_counts = tuple(int(value) for value in arguments.camera_counts)
    except ValueError as error:
        parser.error(f"--camera-counts must contain integers: {error}")
    if not camera_counts or not set(camera_counts) <= {1, 3, 5}:
        parser.error("--camera-counts supports only 1,3,5")
    summary_path = arguments.output_root.resolve() / "batch_summary.json"
    write_json(
        arguments.output_root.resolve() / "protocol_manifest.json",
        {
            "schema_version": 1,
            "evaluation": "PLANNING_ONLY_GT_COMPARISON",
            "protocol": arguments.protocol,
            "environment": arguments.environment,
            "methods": list(arguments.methods),
            "model": arguments.model,
            "physical_execution": False,
            "budget_note": (
                "Native method request structure"
                if arguments.protocol == "native"
                else "Minimum viable call ablation: OWL-TAMP sketch only; "
                "VLM-TAMP retains its indivisible two-stage round"
            ),
        },
    )
    rows_by_key: dict[tuple[str, str, str, str, int, int], dict[str, Any]] = {}
    if arguments.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in previous.get("runs", ()):
            key = (
                str(row.get("protocol", "native")), str(row["method"]),
                str(row["environment"]), str(row["variant"]),
                int(row.get("camera_count", 5)), int(row["seed"]),
            )
            rows_by_key[key] = row
    for method in arguments.methods:
        for variant in variants:
            for camera_count in camera_counts:
                for seed in seeds:
                    output = (
                        arguments.output_root.resolve()
                        / method
                        / arguments.environment
                        / variant
                        / f"images_{camera_count}"
                        / f"seed_{seed:03d}"
                    )
                    result_path = output / "episode_result.json"
                    if arguments.resume and result_path.is_file():
                        print(
                            f"[batch] skip completed {method} {arguments.environment} "
                            f"{variant} images={camera_count} seed={seed}",
                            flush=True,
                        )
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        key = (
                            arguments.protocol, method, arguments.environment, variant,
                            camera_count, seed,
                        )
                        rows_by_key[key] = {
                            "method": method,
                            "protocol": arguments.protocol,
                            "environment": arguments.environment,
                            "variant": variant,
                            "camera_count": camera_count,
                            "seed": seed,
                            "return_code": 0,
                            "output_dir": str(output),
                            "gt_comparison": result.get("gt_comparison"),
                        }
                        write_json(
                            summary_path,
                            {"schema_version": 1, "runs": list(rows_by_key.values())},
                        )
                        continue
                    if output.exists() and any(output.iterdir()):
                        parser.error(
                            f"incomplete run directory is not empty: {output}; "
                            "move it aside before resuming"
                        )
                    command = _command(
                        method, arguments.environment, variant, seed,
                        camera_count, output, arguments,
                    )
                    print(
                        f"[batch] {method} {arguments.environment} {variant} "
                        f"images={camera_count} seed={seed}",
                        flush=True,
                    )
                    completed = subprocess.run(command, check=False)
                    result = None
                    if result_path.is_file():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    key = (
                        arguments.protocol, method, arguments.environment, variant,
                        camera_count, seed,
                    )
                    rows_by_key[key] = {
                        "method": method,
                        "protocol": arguments.protocol,
                        "environment": arguments.environment,
                        "variant": variant,
                        "camera_count": camera_count,
                        "seed": seed,
                        "return_code": completed.returncode,
                        "output_dir": str(output),
                        "gt_comparison": (
                            result.get("gt_comparison")
                            if isinstance(result, dict)
                            else None
                        ),
                    }
                    write_json(
                        summary_path,
                        {"schema_version": 1, "runs": list(rows_by_key.values())},
                    )
                    if completed.returncode and not arguments.continue_on_error:
                        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
