"""Main CLI entry point for Living-Room Phase 3 Mobile Execution and synchronized 5-camera recording.

Usage examples:
  # List all variants with feasibility and plan status
  python -m mujoco_scenes.run_living_room_execution --list-variants

  # One feasible variant with video recording
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m mujoco_scenes.run_living_room_execution \\
    --variant F0_ALL_OBJECTS_IN_STAGING --record

  # One variant with live visualization and recording
  MUJOCO_GL=glfw python -m mujoco_scenes.run_living_room_execution \\
    --variant F0_ALL_OBJECTS_IN_STAGING --show --record

  # Dry-run sequence generation and stance reachability check for all variants
  python -m mujoco_scenes.run_living_room_execution --variant all --dry-run

  # Full suite execution across all feasible and infeasible variants
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m mujoco_scenes.run_living_room_execution \\
    --variant all --record
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

from .living_room_mobile_execution import run_mobile_execution
from .living_room_recorder import (
    L2_FIVE_CAMERAS,
    LivingRoomRecorder,
    create_camera_manifest,
)
from .living_room_region_function import EXPECTED_VARIANTS, INTEGRATED_PREFIX


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_ROOT = ROOT / "mujoco_scenes/benchmark_reports/living_room_region_feasibility_phase1/variants"
FALLBACK_PHASE1_ROOT = DEFAULT_PHASE1_ROOT
DEFAULT_PHASE2_ROOT = ROOT / "mujoco_scenes/benchmark_reports/living_room_symbolic_phase2/variants"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/living_room_execution"


def resolve_phase1_root(custom_root: str | Path | None = None) -> Path:
    if custom_root:
        path = Path(custom_root)
        if path.is_dir():
            return path
    if DEFAULT_PHASE1_ROOT.is_dir():
        return DEFAULT_PHASE1_ROOT
    if FALLBACK_PHASE1_ROOT.is_dir():
        return FALLBACK_PHASE1_ROOT
    raise FileNotFoundError(f"Could not locate Phase 1 root directory: {DEFAULT_PHASE1_ROOT}")


def resolve_phase2_root(custom_root: str | Path | None = None) -> Path:
    if custom_root:
        path = Path(custom_root)
        if path.is_dir():
            return path
    if DEFAULT_PHASE2_ROOT.is_dir():
        return DEFAULT_PHASE2_ROOT
    raise FileNotFoundError(f"Could not locate Phase 2 root directory: {DEFAULT_PHASE2_ROOT}")


def normalize_variant_name(name: str) -> str:
    """Normalize input variant name to canonical registry key."""
    cleaned = name.strip()
    for key in EXPECTED_VARIANTS:
        if cleaned.upper() == key.upper():
            return key
        if cleaned.upper() == key.split("_")[0].upper():
            return key
    return cleaned


def list_variants_table(phase1_root: Path, phase2_root: Path) -> None:
    """Print formatted summary of all discovered living-room variants."""
    print("\n" + "=" * 105)
    print(f"DISCOVERED LIVING-ROOM INTEGRATED VARIANTS ({len(EXPECTED_VARIANTS)} total)")
    print("=" * 105)
    print(f"{'VARIANT ID':<36} {'OUTCOME':<12} {'PHASE 1':<12} {'PHASE 2 PLAN':<16} {'SCENE NAME'}")
    print("-" * 105)
    for variant_id, expected_status in EXPECTED_VARIANTS.items():
        scene_name = f"{INTEGRATED_PREFIX}{variant_id}"
        p1_dir = phase1_root / variant_id
        p2_dir = phase2_root / variant_id

        p1_status = "MISSING"
        if (p1_dir / "functional_region_witness.json").is_file():
            try:
                data = json.loads((p1_dir / "functional_region_witness.json").read_text())
                p1_status = data.get("status", "UNKNOWN")
            except Exception:
                p1_status = "CORRUPT"

        p2_status = "MISSING"
        if (p2_dir / "plan.json").is_file():
            try:
                data = json.loads((p2_dir / "plan.json").read_text())
                actions = len(data.get("actions", []))
                p2_status = f"PLAN ({actions} steps)"
            except Exception:
                p2_status = "CORRUPT"
        elif (p2_dir / "compilation_result.json").is_file():
            try:
                data = json.loads((p2_dir / "compilation_result.json").read_text())
                p2_status = data.get("status", "REJECTED")
            except Exception:
                p2_status = "REJECTED"

        intended_outcome = "FEASIBLE" if expected_status == "COMPLETE" else "INFEASIBLE"
        print(f"{variant_id:<36} {intended_outcome:<12} {p1_status:<12} {p2_status:<16} {scene_name}")
    print("=" * 105 + "\n")


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse 'WxH' string into (width, height) tuple."""
    try:
        parts = res_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError()
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{res_str}'. Expected format like '640x360' or '426x240'")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def execute_variant(
    variant_id: str,
    phase1_root: Path,
    phase2_root: Path,
    output_root: Path,
    *,
    execute: bool = True,
    record: bool = False,
    show: bool = False,
    fps: int = 20,
    tile_width: int = 640,
    tile_height: int = 360,
    start_task_action: int = 0,
    max_task_actions: int | None = None,
    assisted_suite: bool = False,
) -> dict[str, Any]:
    """Execute a single variant with optional 5-camera recording."""
    variant = normalize_variant_name(variant_id)
    if variant not in EXPECTED_VARIANTS:
        raise ValueError(f"Unknown variant: {variant_id}. Choose from: {list(EXPECTED_VARIANTS.keys())}")

    p1_dir = phase1_root / variant
    p2_dir = phase2_root / variant
    out_dir = output_root / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_outcome = "FEASIBLE" if EXPECTED_VARIANTS[variant] == "COMPLETE" else "INFEASIBLE"
    scene_name = f"{INTEGRATED_PREFIX}{variant}"

    print("\n" + "=" * 90)
    print(f"RUNNING LIVING-ROOM EXECUTION: {variant}")
    print(f"  Scene:           {scene_name}")
    print(f"  Intended:        {expected_outcome}")
    print(f"  Phase 1 Source:  {p1_dir}")
    print(f"  Phase 2 Source:  {p2_dir}")
    print(f"  Output Dir:      {out_dir}")
    print(f"  Mode:            {'PHYSICAL_EXECUTE' if execute else 'PLAN_ONLY_DRY_RUN'}")
    print(f"  Record:          {record} ({fps} fps, tile {tile_width}x{tile_height})")
    print(f"  Show GUI:        {show}")
    print("=" * 90)

    recorder = None
    video_path = out_dir / f"{variant}_5cam.mp4" if record else None

    # For feasible variants or when recording is requested on a scene, setup recorder
    if record or show:
        try:
            from .living_room_region_scene import L2LivingRoomRegionScene
            temp_scene = L2LivingRoomRegionScene(scene_name, "google")
            recorder = LivingRoomRecorder(
                temp_scene,
                output_path=video_path,
                tile_width=tile_width,
                tile_height=tile_height,
                fps=fps,
                show=show,
                record=record,
            )
            recorder.telemetry.variant_id = variant
            recorder.telemetry.scene_name = scene_name
            recorder.telemetry.intended_outcome = expected_outcome
        except Exception as error:
            print(f"WARNING: Could not initialize recorder: {error}")
            recorder = None

    try:
        result = run_mobile_execution(
            phase1_dir=p1_dir,
            phase2_dir=p2_dir,
            output_dir=out_dir,
            variant=variant,
            execute=execute,
            start_task_action=start_task_action,
            max_task_actions=max_task_actions,
            recorder=recorder,
            assisted_suite=assisted_suite,
        )

        # Write camera manifest if recorded
        if record and recorder is not None:
            manifest = create_camera_manifest(
                output_path=video_path,
                mosaic_width=recorder.mosaic_width,
                mosaic_height=recorder.mosaic_height,
                tile_width=recorder.tile_width,
                tile_height=recorder.tile_height,
                fps=recorder.fps,
                total_frames=recorder.total_frames_captured,
                duration_sim_s=recorder.last_capture_sim_time if recorder.last_capture_sim_time >= 0 else 0.0,
            )
            write_json(out_dir / "camera_manifest.json", manifest)

        status = result.get("status", "UNKNOWN")
        print(f"\n>> VARIANT {variant} RESULT: {status} (Wall: {result.get('wall_time_s', 0.0):.2f}s)")
        return result

    except Exception as error:
        print(f"\n>> VARIANT {variant} FAILED WITH EXCEPTION: {error}")
        if recorder is not None:
            try:
                recorder.telemetry.execution_status = "FAILED"
                recorder.hold_final_frame(duration_s=1.0)
                recorder.close()
            except Exception:
                pass
        raise


def run_suite(
    variants: list[str],
    phase1_root: Path,
    phase2_root: Path,
    output_root: Path,
    *,
    execute: bool = True,
    record: bool = False,
    show: bool = False,
    fps: int = 20,
    tile_width: int = 640,
    tile_height: int = 360,
    fail_fast: bool = False,
    assisted_suite: bool = False,
) -> dict[str, Any]:
    """Execute a suite of variants and produce aggregated benchmark summaries."""
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    suite_start = time.monotonic()

    for index, variant in enumerate(variants, 1):
        print(f"\n[{index}/{len(variants)}] Starting suite execution: {variant}")
        try:
            res = execute_variant(
                variant_id=variant,
                phase1_root=phase1_root,
                phase2_root=phase2_root,
                output_root=output_root,
                execute=execute,
                record=record,
                show=show,
                fps=fps,
                tile_width=tile_width,
                tile_height=tile_height,
                assisted_suite=assisted_suite,
            )
            results.append(res)
        except Exception as error:
            failed_res = {
                "variant": variant,
                "status": "FAILED",
                "intended_outcome": "FEASIBLE" if EXPECTED_VARIANTS.get(variant) == "COMPLETE" else "INFEASIBLE",
                "error": str(error),
            }
            results.append(failed_res)
            if fail_fast:
                print(f"\n[FAIL FAST] Aborting suite due to failure in {variant}")
                break

    total_time = time.monotonic() - suite_start

    # Aggregation
    feasible_variants = [v for v in variants if EXPECTED_VARIANTS.get(v) == "COMPLETE"]
    infeasible_variants = [v for v in variants if EXPECTED_VARIANTS.get(v) != "COMPLETE"]

    feasible_passed = sum(1 for r in results if r.get("variant") in feasible_variants and r.get("status") == "SUCCESS")
    infeasible_confirmed = sum(1 for r in results if r.get("variant") in infeasible_variants and r.get("status") == "INFEASIBLE_CONFIRMED")
    total_passed = feasible_passed + infeasible_confirmed
    all_success = total_passed == len(variants)

    summary = {
        "schema_version": 1,
        "suite_name": "living_room_execution_benchmark",
        "total_variants": len(variants),
        "total_passed": total_passed,
        "feasible_total": len(feasible_variants),
        "feasible_passed": feasible_passed,
        "infeasible_total": len(infeasible_variants),
        "infeasible_confirmed": infeasible_confirmed,
        "all_passed": all_success,
        "wall_time_s": total_time,
        "results": results,
    }

    # Write summary JSON and CSV
    write_json(output_root / "suite_summary.json", summary)

    csv_path = output_root / "suite_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "intended_outcome", "status", "mode", "moves", "actions", "wall_time_s", "video"])
        for r in results:
            v = r.get("variant", "")
            video_file = output_root / v / f"{v}_5cam.mp4"
            writer.writerow([
                v,
                r.get("intended_outcome", ""),
                r.get("status", ""),
                r.get("mode", ""),
                r.get("move_count", 0),
                r.get("refined_action_count", 0),
                f"{r.get('wall_time_s', 0.0):.2f}",
                "YES" if video_file.is_file() else "NO",
            ])

    # Print summary table
    print("\n" + "=" * 110)
    print(f"LIVING-ROOM BENCHMARK SUITE SUMMARY ({total_passed}/{len(variants)} passed in {total_time:.1f}s)")
    print("=" * 110)
    print(f"{'VARIANT ID':<36} {'INTENDED':<12} {'RESULT STATUS':<22} {'MOVES':<8} {'ACTIONS':<10} {'TIME (s)':<10} {'VIDEO'}")
    print("-" * 110)
    for r in results:
        v = r.get("variant", "")
        intended = r.get("intended_outcome", "")
        status = r.get("status", "")
        moves = str(r.get("move_count", "-"))
        acts = str(r.get("refined_action_count", "-"))
        t_s = f"{r.get('wall_time_s', 0.0):.2f}"
        vid = "YES" if (output_root / v / f"{v}_5cam.mp4").is_file() else "NO"
        print(f"{v:<36} {intended:<12} {status:<22} {moves:<8} {acts:<10} {t_s:<10} {vid}")
    print("=" * 110 + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrated Living-Room Execution Pipeline & 5-Camera Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List all registered living-room variants and exit.",
    )
    parser.add_argument(
        "--variant",
        default="all",
        help="Variant to run (e.g. F0_ALL_OBJECTS_IN_STAGING) or 'all' (default: all).",
    )
    parser.add_argument(
        "--only-feasible",
        action="store_true",
        help="Run only feasible variants (F0-F6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in planning-only mode without physics stepping.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record synchronized 5-camera mosaic MP4 video.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live OpenCV GUI during execution.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Framerate for video recording (default: 20).",
    )
    parser.add_argument(
        "--tile-resolution",
        type=parse_resolution,
        default="640x360",
        help="Camera tile resolution 'WxH' (default: 640x360).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=None,
        help="Path to Phase 1 root directory with per-variant artifacts.",
    )
    parser.add_argument(
        "--phase2-root",
        type=Path,
        default=None,
        help="Path to Phase 2 root directory with per-variant plans.",
    )
    parser.add_argument(
        "--start-task-action",
        type=int,
        default=0,
        help="Start task action index for partial runs.",
    )
    parser.add_argument(
        "--max-task-actions",
        type=int,
        default=None,
        help="Maximum task actions to execute.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure during multi-variant execution.",
    )
    parser.add_argument(
        "--assisted-suite",
        action="store_true",
        help=(
            "Accept a structurally valid ON relation when only residual "
            "settling speed or final yaw misses the strict postcondition; "
            "support, boundary, non-overlap, penetration, height, release "
            "and floor checks remain mandatory."
        ),
    )

    args = parser.parse_args()

    phase1_root = resolve_phase1_root(args.phase1_root)
    phase2_root = resolve_phase2_root(args.phase2_root)

    if args.list_variants:
        list_variants_table(phase1_root, phase2_root)
        return

    tile_w, tile_h = args.tile_resolution
    execute_mode = not args.dry_run

    if args.variant.lower() == "all":
        all_variants = list(EXPECTED_VARIANTS.keys())
        if args.only_feasible:
            all_variants = [v for v in all_variants if EXPECTED_VARIANTS[v] == "COMPLETE"]
        summary = run_suite(
            variants=all_variants,
            phase1_root=phase1_root,
            phase2_root=phase2_root,
            output_root=args.output_root,
            execute=execute_mode,
            record=args.record,
            show=args.show,
            fps=args.fps,
            tile_width=tile_w,
            tile_height=tile_h,
            fail_fast=args.fail_fast,
            assisted_suite=args.assisted_suite,
        )
        if not summary.get("all_passed"):
            sys.exit(1)
    else:
        norm_v = normalize_variant_name(args.variant)
        result = execute_variant(
            variant_id=norm_v,
            phase1_root=phase1_root,
            phase2_root=phase2_root,
            output_root=args.output_root,
            execute=execute_mode,
            record=args.record,
            show=args.show,
            fps=args.fps,
            tile_width=tile_w,
            tile_height=tile_h,
            start_task_action=args.start_task_action,
            max_task_actions=args.max_task_actions,
            assisted_suite=args.assisted_suite,
        )
        if result.get("status") not in {"SUCCESS", "INFEASIBLE_CONFIRMED"}:
            sys.exit(1)


if __name__ == "__main__":
    main()
