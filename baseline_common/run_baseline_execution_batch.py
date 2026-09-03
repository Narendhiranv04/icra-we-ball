"""Run physical baseline episodes under the shared artifact contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .artifacts import write_json


KITCHEN_GOAL = (
    "Prepare and serve coffee and soup for two people using the available "
    "kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil."
)
LIVING_ROOM_GOAL = (
    "Prepare the living room for two people watching television. Place one cup "
    "and one saucer on each person's fixed individual side table, and place the "
    "TV remote on the fixed shared coffee table."
)
GOALS = {"kitchen": KITCHEN_GOAL, "living_room": LIVING_ROOM_GOAL}
VARIANTS = {
    "kitchen": tuple(f"K{index}" for index in range(1, 13)),
    "living_room": tuple(f"L{index}" for index in range(1, 11)),
}
METHODS = ("vlm_tamp", "owl_tamp")
# Retrieval grounds with CLIP rather than a language model and only exists for
# the Living Room, so it is opt-in via --methods rather than a default.
LIVING_ROOM_ONLY_METHODS = ("retrieval",)


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("value must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", choices=tuple(GOALS), default="kitchen"
    )
    parser.add_argument("--methods", type=_csv, default=METHODS)
    parser.add_argument("--variants", type=_csv)
    parser.add_argument("--camera-counts", type=_csv, default=("5",))
    parser.add_argument("--seeds", type=_csv, default=("0",))
    parser.add_argument("--protocol", choices=("native", "single_call", "receding_horizon"), default="native")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--max-model-calls", type=int, default=10)
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="model-native",
        help=(
            "Applied to every model-driven method so the comparison holds "
            "decoding fixed; recorded in protocol_manifest.json."
        ),
    )
    parser.add_argument("--max-replans", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument(
        "--max-sketch-actions",
        type=int,
        default=24,
        help=(
            "OWL-TAMP only: bound on its per-action constraint requests, so a "
            "sketch that degenerates into repetition cannot bill one request "
            "per repeat."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _command(
    method: str,
    variant: str,
    camera_count: int,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    common = [
        sys.executable, "-m",
        "--output-dir", str(output_dir),
        "--goal", args.goal or GOALS[args.environment],
        "--base-url", args.base_url,
        "--model", args.model,
        "--max-tokens", str(args.max_tokens),
        "--seed", str(seed),
        "--camera-count", str(camera_count),
        "--protocol", args.protocol,
    ]
    if args.environment == "kitchen":
        # Only the Kitchen runners own an interactive viewer to suppress; the
        # Living Room physical runtime is constructed headless.
        common.extend(("--headless", "--close-on-complete"))
    module = f"{method}_baseline.run_{args.environment}"
    if method == "vlm_tamp":
        if args.protocol == "receding_horizon":
            raise ValueError("VLM-TAMP has no receding_horizon protocol")
        max_calls = 1 if args.protocol == "single_call" else args.max_model_calls
        # Kitchen selects its physical variant with a dedicated flag; the
        # Living Room runner takes --variant plus --physical-execution.
        variant_flags = (
            ["--physical-variant", variant]
            if args.environment == "kitchen"
            else ["--variant", variant, "--physical-execution"]
        )
        return [
            *common[:2], module,
            *variant_flags,
            *common[2:],
            "--max-model-calls", str(max_calls),
            "--max-total-actions", str(args.max_actions),
            "--decoding", args.decoding,
        ]
    if method == "owl_tamp":
        return [
            *common[:2], module,
            "--variant", variant,
            "--physical-execution",
            *common[2:],
            "--max-replans", str(args.max_replans),
            "--max-total-actions", str(args.max_actions),
            "--max-sketch-actions", str(args.max_sketch_actions),
            "--decoding", args.decoding,
        ]
    if method == "retrieval":
        # Retrieval calls no model, so --base-url/--model/--max-tokens are
        # accepted and ignored; --protocol and the render size still apply.
        return [
            *common[:2], module,
            "--variant", variant,
            "--physical-execution",
            *common[2:],
        ]
    raise ValueError(f"Unsupported method {method!r}")


def _validate(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    methods = tuple(args.methods)
    allowed = set(METHODS)
    if args.environment == "living_room":
        allowed |= set(LIVING_ROOM_ONLY_METHODS)
    if not set(methods) <= allowed:
        raise ValueError(f"--methods supports only {', '.join(sorted(allowed))}")
    if args.protocol == "receding_horizon" and "vlm_tamp" in methods:
        raise ValueError("receding_horizon is currently available only for owl_tamp")
    allowed_variants = VARIANTS[args.environment]
    variants = tuple(args.variants) if args.variants else allowed_variants
    if not variants or any(value not in allowed_variants for value in variants):
        raise ValueError(
            f"--variants must be {args.environment} labels: "
            f"{allowed_variants[0]}-{allowed_variants[-1]}"
        )
    cameras = tuple(int(value) for value in args.camera_counts)
    seeds = tuple(int(value) for value in args.seeds)
    if not cameras or not set(cameras) <= {1, 3, 5}:
        raise ValueError("--camera-counts supports 1,3,5")
    if not seeds or any(value < 0 for value in seeds):
        raise ValueError("--seeds must be non-negative")
    return methods, variants, cameras, seeds


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        methods, variants, cameras, seeds = _validate(args)
    except ValueError as error:
        parser.error(str(error))
    root = args.output_root.resolve()
    summary_path = root / "batch_summary.json"
    rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    if args.resume and summary_path.is_file():
        for row in json.loads(summary_path.read_text(encoding="utf-8")).get("runs", ()):
            rows[(str(row["method"]), str(row["variant"]), int(row["camera_count"]), int(row["seed"]))] = row

    write_json(root / "protocol_manifest.json", {
        "schema_version": 1,
        "environment": args.environment,
        "methods": list(methods),
        "protocol": args.protocol,
        "model": args.model,
        "decoding": args.decoding,
        "max_model_calls": args.max_model_calls,
        "goal": args.goal,
        "physical_execution": True,
        "shared_result": "benchmark_execution_result.json",
    })
    for method in methods:
        for variant in variants:
            for camera_count in cameras:
                for seed in seeds:
                    key = (method, variant, camera_count, seed)
                    output = root / method / variant / f"images_{camera_count}" / f"seed_{seed:03d}"
                    result_path = output / "benchmark_execution_result.json"
                    if args.resume and result_path.is_file():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        rows[key] = _row(args.environment, method, variant, camera_count, seed, output, 0, result)
                        continue
                    if output.exists() and any(output.iterdir()):
                        parser.error(f"incomplete output directory is not empty: {output}")
                    print(f"[baseline-execution] {method} {variant} images={camera_count} seed={seed}", flush=True)
                    completed = subprocess.run(
                        _command(method, variant, camera_count, seed, output, args),
                        check=False,
                    )
                    result = (
                        json.loads(result_path.read_text(encoding="utf-8"))
                        if result_path.is_file() else None
                    )
                    rows[key] = _row(args.environment, method, variant, camera_count, seed, output, completed.returncode, result)
                    write_json(summary_path, {"schema_version": 1, "runs": list(rows.values())})
                    if result is None and completed.returncode and not args.continue_on_error:
                        raise SystemExit(completed.returncode)
    write_json(summary_path, {"schema_version": 1, "runs": list(rows.values())})


def _row(
    environment: str,
    method: str,
    variant: str,
    camera_count: int,
    seed: int,
    output: Path,
    return_code: int,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "method": method,
        "environment": environment,
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
