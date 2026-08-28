"""Canonical EXPLORE -> SATISFY -> PLAN entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

from mujoco_scenes.final_paper_variant_labels import resolve_variant_name

from .models import PipelineResult
from .planning import plan_with_common_astar
from .search import search_until_satisfied
from .spec_provider import provider_for_mode


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs" / "functional_tamp_pipeline"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    output_root: Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
    verbose: bool = False,
    observation_images: list[Path] | None = None,
) -> PipelineResult:
    internal_variant = resolve_variant_name(domain, variant)
    run_dir = output_root / domain / variant / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    if domain == "kitchen":
        from mujoco_scenes.run_kitchen_vlm_pipeline import _render_initial
        from .domains.kitchen import TASK, run_to_plan, scene_for_variant

        scene = scene_for_variant(internal_variant)
        images = list(observation_images or [])
        if mode == "vlm" and not images:
            images = _render_initial(scene, run_dir / "vlm_inputs", 640, 480)
        print("[1/5] Functional specification", flush=True)
        specification = provider_for_mode(mode).provide(domain, TASK, images)
        _write_json(run_dir / "functional_specification.json", specification.to_dict())
        _write_json(run_dir / "functional_requirement_graph.json", specification.to_dict())
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
        return result

    if domain == "living_room":
        from .domains.living_room import TASK, run_to_plan

        print("[1/5] Functional specification", flush=True)
        images = list(observation_images or [])
        if mode == "vlm" and not images:
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
        specification = provider_for_mode(mode).provide(
            domain, TASK, images
        )
        _write_json(run_dir / "functional_specification.json", specification.to_dict())
        _write_json(run_dir / "functional_requirement_graph.json", specification.to_dict())
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
        return result

    if domain != "workshop":
        raise NotImplementedError(
            f"The canonical adapter for {domain} has not been integrated yet"
        )

    from mujoco_scenes.workshop_scene import WorkshopScene
    from .domains.workshop import WorkshopDomainAdapter, WorkshopPlanningCompiler

    scene = WorkshopScene(robot="google", variant=internal_variant)
    images = list(observation_images or [])
    if mode == "vlm" and not images:
        images = _capture_workshop_vlm_inputs(scene, run_dir / "vlm_inputs")

    print("[1/5] Functional specification", flush=True)
    provider = provider_for_mode(mode)
    task = WorkshopDomainAdapter.task_instruction
    specification = provider.provide(domain, task, images)
    _write_json(run_dir / "functional_specification.json", specification.to_dict())
    _write_json(run_dir / "functional_requirement_graph.json", specification.to_dict())

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
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("kitchen", "workshop", "living_room"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--mode", required=True, choices=("gt", "vlm"))
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
