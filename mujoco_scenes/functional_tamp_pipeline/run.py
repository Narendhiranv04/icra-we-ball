"""Canonical EXPLORE -> SATISFY -> PLAN entry point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

from mujoco_scenes.final_paper_variant_labels import resolve_variant_name

from .models import FunctionalRequirementGraph, PipelineResult
from .planning import plan_with_common_astar
from .search import search_until_satisfied
from .search_order import resolve_search_order
from .spec_provider import provider_for_mode


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs" / "functional_tamp_pipeline"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _compute_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_git_provenance() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(status)
    except Exception:
        dirty = None

    return commit, dirty


def _collect_artifacts(run_dir: Path) -> dict[str, str]:
    candidate_map = {
        "functional_specification": "functional_specification.json",
        "functional_graph": "functional_requirement_graph.json",
        "observed_graph": "observed_scene_graph.json",
        "grounding": "graph_grounding_result.json",
        "satisfaction": "satisfaction.json",
        "canonical_grounding_witness": "canonical_grounding_witness.json",
        "action_plan": "action_sequence/action_plan.json",
        "plan": "action_sequence/plan.json",
        "replay_validation": "action_sequence/replay_validation.json",
        "plan_grounding_audit": "plan_grounding_audit.json",
        "result": "result.json",
        "detection_diagnostics": "detection_diagnostics.json",
    }
    artifacts: dict[str, str] = {}
    for key, rel_path in candidate_map.items():
        if (run_dir / rel_path).exists():
            artifacts[key] = rel_path
    return artifacts


def _get_exploration_actuation(domain: str, dry_run: bool) -> str:
    if domain == "kitchen":
        return "direct_sim_articulation"
    if domain == "workshop":
        return "direct_sim_articulation" if dry_run else "robot_physical"
    if domain == "living_room":
        return "not_applicable"
    return "unknown"


def _get_provider_model(mode: str) -> str | None:
    if mode == "vlm":
        return os.getenv("TAMP_FM_MODEL") or os.getenv("FM_MODEL")
    return None


def _load_or_acquire_specification(
    *,
    domain: str,
    mode: str,
    task: str,
    images: list[Path],
    specification_json: Path | str | None,
) -> tuple[FunctionalRequirementGraph, str, str | None]:
    if specification_json is not None:
        spec_path = Path(specification_json)
        if not spec_path.exists():
            raise FileNotFoundError(f"Specification JSON not found: {spec_path}")
        raw_text = spec_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        graph = FunctionalRequirementGraph.from_dict(data)
        graph.validate()
        if graph.domain != domain:
            raise ValueError(
                f"Replayed specification domain {graph.domain!r} does not match requested domain {domain!r}"
            )
        if mode == "gt" and not graph.source.startswith("GT"):
            raise ValueError(
                f"Specification source {graph.source!r} is not compatible with requested mode {mode!r}"
            )
        if mode == "vlm" and not graph.source.startswith("VLM"):
            raise ValueError(
                f"Specification source {graph.source!r} is not compatible with requested mode {mode!r}"
            )
        return graph, "replayed_provider_output", str(spec_path.resolve())

    provider = provider_for_mode(mode)
    specification = provider.provide(domain, task, images)
    return specification, "live_provider", None


def _write_run_manifest(
    *,
    run_dir: Path,
    domain: str,
    variant: str,
    internal_variant: str,
    mode: str,
    specification: FunctionalRequirementGraph | None,
    spec_acquisition: str,
    specification_input: str | None,
    specification_sha256: str | None,
    provider_model: str | None,
    search_order: str,
    search_order_source_effective: str,
    resolved_search_order: tuple[str, ...],
    exploration_actuation: str,
    git_commit: str | None,
    git_dirty: bool | None,
    started_at_utc: str,
    finished_at_utc: str,
    runtime_sec: float,
    terminal_status: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "domain": domain,
        "variant": variant,
        "internal_variant": internal_variant,
        "spec_mode": mode,
        "spec_provider_source": specification.source if specification else None,
        "spec_acquisition": spec_acquisition,
        "specification_sha256": specification_sha256,
        "specification_input": specification_input,
        "provider_model": provider_model,
        "search_order_source_requested": search_order,
        "search_order_source_effective": search_order_source_effective,
        "provider_region_ranking": list(specification.region_ranking) if specification else [],
        "region_order_used": list(resolved_search_order),
        "exploration_actuation": exploration_actuation,
        "execution_state": "planning_only",
        "visualization_requested": False,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "pipeline_runtime_seconds": round(runtime_sec, 4),
        "terminal_status": terminal_status,
        "observer_errors": [],
        "artifacts": _collect_artifacts(run_dir),
    }
    _write_json(run_dir / "run_manifest.json", manifest)


def _capture_workshop_vlm_inputs(scene: Any, output_dir: Path) -> list[Path]:
    import cv2
    from mujoco_scenes.workshop_phase1.capture import MultiViewCameraRig

    output_dir.mkdir(parents=True, exist_ok=True)
    observations = MultiViewCameraRig(scene=scene).capture_stage_observations(
        stage_region="INITIAL", capture_segmentation=False
    )
    paths = []
    for observation in observations:
        path = output_dir / f"initial_{observation.camera_id.lower()}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write VLM input image {path}")
        paths.append(path)
    return paths


def run_pipeline(
    *,
    domain: str,
    variant: str,
    mode: str,
    search_order: str = "provider",
    specification_json: Path | str | None = None,
    output_root: Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
    verbose: bool = False,
    observation_images: list[Path] | None = None,
) -> PipelineResult:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    start_t = time.perf_counter()
    internal_variant = resolve_variant_name(domain, variant)
    run_dir = output_root / domain / variant / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    git_commit, git_dirty = _get_git_provenance()
    exploration_actuation = _get_exploration_actuation(domain, dry_run)
    provider_model = _get_provider_model(mode)
    spec_acquisition = "live_provider" if specification_json is None else "replayed_provider_output"
    specification_input = None if specification_json is None else str(Path(specification_json).resolve())
    specification: FunctionalRequirementGraph | None = None
    specification_sha256: str | None = None
    resolved_search_order: tuple[str, ...] = ()
    search_order_source_effective: str = "not_applicable" if domain == "living_room" else search_order

    try:
        if domain == "kitchen":
            from mujoco_scenes.run_kitchen_vlm_pipeline import _render_initial
            from .domains.kitchen import TASK, run_to_plan, scene_for_variant

            scene = scene_for_variant(internal_variant)
            images = list(observation_images or [])
            if mode == "vlm" and not images and specification_json is None:
                images = _render_initial(scene, run_dir / "vlm_inputs", 640, 480)

            print("[1/5] Functional specification", flush=True)
            specification, spec_acquisition, specification_input = _load_or_acquire_specification(
                domain=domain,
                mode=mode,
                task=TASK,
                images=images,
                specification_json=specification_json,
            )
            _write_json(run_dir / "functional_specification.json", specification.to_dict())
            _write_json(run_dir / "functional_requirement_graph.json", specification.to_dict())
            specification_sha256 = _compute_file_sha256(run_dir / "functional_specification.json")

            resolved_search_order = resolve_search_order(specification, domain, search_order)
            if domain != "living_room" and search_order == "fixed":
                raise RuntimeError("fixed search order is resolved but not executable until Phase 3.1 wiring")

            print("[2/5] Initial perception", flush=True)
            print("[3/5] Functional grounding and ranked region search", flush=True)
            result = run_to_plan(
                variant_label=variant,
                internal_variant=internal_variant,
                mode=mode,
                specification=specification,
                output_dir=run_dir,
                scene=scene,
            )
            if result.assignment:
                print("[4/5] Role assignment", flush=True)
                print(json.dumps(result.assignment, indent=2, sort_keys=True), flush=True)
                print("[5/5] A* planning", flush=True)
                for action in result.plan:
                    print(f"  {action['action_index']:02d}. {action['operator']}({', '.join(action['arguments'])})", flush=True)
            _write_json(run_dir / "result.json", result.to_dict())

            finished_at_utc = datetime.now(timezone.utc).isoformat()
            runtime_sec = time.perf_counter() - start_t
            _write_run_manifest(
                run_dir=run_dir,
                domain=domain,
                variant=variant,
                internal_variant=internal_variant,
                mode=mode,
                specification=specification,
                spec_acquisition=spec_acquisition,
                specification_input=specification_input,
                specification_sha256=specification_sha256,
                provider_model=provider_model,
                search_order=search_order,
                search_order_source_effective=search_order_source_effective,
                resolved_search_order=resolved_search_order,
                exploration_actuation=exploration_actuation,
                git_commit=git_commit,
                git_dirty=git_dirty,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                runtime_sec=runtime_sec,
                terminal_status=result.status,
            )
            return result

        if domain == "living_room":
            from .domains.living_room import TASK, run_to_plan

            images = list(observation_images or [])
            if mode == "vlm" and not images and specification_json is None:
                from PIL import Image
                from mujoco_scenes.living_room_region_scene import (
                    L2_CAMERAS, L2LivingRoomRegionScene,
                )
                from mujoco_scenes.living_room_variants import scene_name

                vlm_scene = L2LivingRoomRegionScene(scene_name(internal_variant), robot="none")
                image_dir = run_dir / "vlm_inputs"
                image_dir.mkdir(parents=True, exist_ok=True)
                for camera in L2_CAMERAS:
                    path = image_dir / f"{camera}.png"
                    Image.fromarray(vlm_scene.render_frame(camera, 1280, 960)).save(path)
                    images.append(path)

            print("[1/5] Functional specification", flush=True)
            specification, spec_acquisition, specification_input = _load_or_acquire_specification(
                domain=domain,
                mode=mode,
                task=TASK,
                images=images,
                specification_json=specification_json,
            )
            _write_json(run_dir / "functional_specification.json", specification.to_dict())
            _write_json(run_dir / "functional_requirement_graph.json", specification.to_dict())
            specification_sha256 = _compute_file_sha256(run_dir / "functional_specification.json")

            resolved_search_order = resolve_search_order(specification, domain, search_order)

            print("[2/5] Initial perception", flush=True)
            print("[3/5] Global functional grounding", flush=True)
            result = run_to_plan(
                variant_label=variant, internal_variant=internal_variant,
                mode=mode, specification=specification, output_dir=run_dir,
            )
            if result.assignment:
                print("[4/5] Role assignment", flush=True)
                print(json.dumps(result.assignment, indent=2, sort_keys=True), flush=True)
                print("[5/5] A* planning", flush=True)
                for action in result.plan:
                    print(f"  {action['action_index']:02d}. {action['operator']}({', '.join(action['arguments'])})", flush=True)
            _write_json(run_dir / "result.json", result.to_dict())

            finished_at_utc = datetime.now(timezone.utc).isoformat()
            runtime_sec = time.perf_counter() - start_t
            _write_run_manifest(
                run_dir=run_dir,
                domain=domain,
                variant=variant,
                internal_variant=internal_variant,
                mode=mode,
                specification=specification,
                spec_acquisition=spec_acquisition,
                specification_input=specification_input,
                specification_sha256=specification_sha256,
                provider_model=provider_model,
                search_order=search_order,
                search_order_source_effective=search_order_source_effective,
                resolved_search_order=resolved_search_order,
                exploration_actuation=exploration_actuation,
                git_commit=git_commit,
                git_dirty=git_dirty,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                runtime_sec=runtime_sec,
                terminal_status=result.status,
            )
            return result

        if domain != "workshop":
            raise NotImplementedError(
                f"The canonical adapter for {domain} has not been integrated yet"
            )

        from mujoco_scenes.workshop_scene import WorkshopScene
        from .domains.workshop import WorkshopDomainAdapter, WorkshopPlanningCompiler

        scene = WorkshopScene(robot="google", variant=internal_variant)
        images = list(observation_images or [])
        if mode == "vlm" and not images and specification_json is None:
            images = _capture_workshop_vlm_inputs(scene, run_dir / "vlm_inputs")

        print("[1/5] Functional specification", flush=True)
        task = WorkshopDomainAdapter.task_instruction
        specification, spec_acquisition, specification_input = _load_or_acquire_specification(
            domain=domain,
            mode=mode,
            task=task,
            images=images,
            specification_json=specification_json,
        )
        _write_json(run_dir / "functional_specification.json", specification.to_dict())
        _write_json(run_dir / "functional_requirement_graph.json", specification.to_dict())
        specification_sha256 = _compute_file_sha256(run_dir / "functional_specification.json")

        resolved_search_order = resolve_search_order(specification, domain, search_order)
        if domain != "living_room" and search_order == "fixed":
            raise RuntimeError("fixed search order is resolved but not executable until Phase 3.1 wiring")

        adapter = WorkshopDomainAdapter(
            internal_variant,
            specification,
            scene=scene,
            physical_open=not dry_run,
            output_dir=str(run_dir / "perception"),
            verbose=verbose,
        )
        print("[2/5] Initial perception", flush=True)
        print("[3/5] Functional grounding and ranked region search", flush=True)
        satisfaction, inspected = search_until_satisfied(adapter, specification)
        _write_json(run_dir / "observed_scene_graph.json", adapter.graph.to_dict())
        _write_json(run_dir / "detection_diagnostics.json", {
            "records": adapter.controller.detection_diagnostics,
        })
        _write_json(run_dir / "graph_grounding_result.json", satisfaction.to_dict())
        _write_json(run_dir / "satisfaction.json", {
            "satisfied": satisfaction.satisfied,
            "status": satisfaction.status,
            "assignment": satisfaction.assignment,
            "missing_requirements": list(satisfaction.missing_requirements),
            "evidence": satisfaction.evidence,
        })
        if not satisfaction.satisfied or satisfaction.assignment is None:
            reason = ", ".join(satisfaction.missing_requirements) or "NO_GLOBAL_ASSIGNMENT"
            print(f"FUNCTIONAL GROUNDING FAILED: {reason}", flush=True)
            result = PipelineResult(
                domain=domain, variant=variant, mode=mode, status=satisfaction.status,
                inspected_regions=inspected, failure_reason=reason,
            )
            _write_json(run_dir / "result.json", result.to_dict())

            finished_at_utc = datetime.now(timezone.utc).isoformat()
            runtime_sec = time.perf_counter() - start_t
            _write_run_manifest(
                run_dir=run_dir,
                domain=domain,
                variant=variant,
                internal_variant=internal_variant,
                mode=mode,
                specification=specification,
                spec_acquisition=spec_acquisition,
                specification_input=specification_input,
                specification_sha256=specification_sha256,
                provider_model=provider_model,
                search_order=search_order,
                search_order_source_effective=search_order_source_effective,
                resolved_search_order=resolved_search_order,
                exploration_actuation=exploration_actuation,
                git_commit=git_commit,
                git_dirty=git_dirty,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                runtime_sec=runtime_sec,
                terminal_status=result.status,
            )
            return result

        print("[4/5] Role assignment", flush=True)
        print("ROLE ASSIGNMENT", flush=True)
        for role in ("driver", "fastener", "work_surface"):
            print(f"{role} -> {satisfaction.assignment[role]}", flush=True)

        print("[5/5] A* planning", flush=True)
        planned = plan_with_common_astar(
            WorkshopPlanningCompiler(), satisfaction.assignment, adapter.planning_context()
        )
        _write_json(run_dir / "action_plan.json", {
            "planner": planned.search.statistics,
            "actions": list(planned.actions),
            "validation": planned.validation,
            "exploratory_open_actions_excluded": True,
        })
        for action in planned.actions:
            print(
                f"  {action['action_index']:02d}. {action['operator']}"
                f"({', '.join(action['arguments'])})",
                flush=True,
            )

        result = PipelineResult(
            domain=domain,
            variant=variant,
            mode=mode,
            status="ACTION_SEQUENCE_READY",
            inspected_regions=inspected,
            assignment=satisfaction.assignment,
            plan=planned.actions,
            search_statistics=planned.search.statistics,
            failure_reason=None,
        )
        _write_json(run_dir / "result.json", result.to_dict())

        finished_at_utc = datetime.now(timezone.utc).isoformat()
        runtime_sec = time.perf_counter() - start_t
        _write_run_manifest(
            run_dir=run_dir,
            domain=domain,
            variant=variant,
            internal_variant=internal_variant,
            mode=mode,
            specification=specification,
            spec_acquisition=spec_acquisition,
            specification_input=specification_input,
            specification_sha256=specification_sha256,
            provider_model=provider_model,
            search_order=search_order,
            search_order_source_effective=search_order_source_effective,
            resolved_search_order=resolved_search_order,
            exploration_actuation=exploration_actuation,
            git_commit=git_commit,
            git_dirty=git_dirty,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            runtime_sec=runtime_sec,
            terminal_status=result.status,
        )
        return result

    except Exception:
        finished_at_utc = datetime.now(timezone.utc).isoformat()
        runtime_sec = time.perf_counter() - start_t
        _write_run_manifest(
            run_dir=run_dir,
            domain=domain,
            variant=variant,
            internal_variant=internal_variant,
            mode=mode,
            specification=specification,
            spec_acquisition=spec_acquisition,
            specification_input=specification_input,
            specification_sha256=specification_sha256,
            provider_model=provider_model,
            search_order=search_order,
            search_order_source_effective=search_order_source_effective,
            resolved_search_order=resolved_search_order,
            exploration_actuation=exploration_actuation,
            git_commit=git_commit,
            git_dirty=git_dirty,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            runtime_sec=runtime_sec,
            terminal_status="PIPELINE_EXCEPTION",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("kitchen", "workshop", "living_room"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--mode", required=True, choices=("gt", "vlm"))
    parser.add_argument(
        "--search-order",
        choices=("provider", "fixed"),
        default="provider",
        help="Search-order source: 'provider' uses specification ranking, 'fixed' uses domain default.",
    )
    parser.add_argument(
        "--specification-json",
        type=Path,
        default=None,
        help="Optional path to a saved functional specification JSON for replay.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--observation-image", type=Path, action="append", default=[])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use direct simulator articulation instead of robot-actuated search OPEN.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        result = run_pipeline(
            domain=args.domain,
            variant=args.variant,
            mode=args.mode,
            search_order=args.search_order,
            specification_json=args.specification_json,
            output_root=args.output_root,
            dry_run=args.dry_run,
            verbose=args.verbose,
            observation_images=args.observation_image,
        )
    except Exception as error:
        print(f"PIPELINE FAILED: {error}", flush=True)
        if args.verbose:
            traceback.print_exc()
        return 1
    print(f"PIPELINE STATUS: {result.status}", flush=True)
    return 0 if result.status in {"ACTION_SEQUENCE_READY", "INFEASIBLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
