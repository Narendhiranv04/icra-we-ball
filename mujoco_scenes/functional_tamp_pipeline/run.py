"""Canonical EXPLORE -> SATISFY -> PLAN entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

from mujoco_scenes.final_paper_variant_labels import resolve_variant_name

try:
    from .errors import PipelineError, VLMSpecificationError, ReplaySpecificationError, SearchRegionContractError
    from .models import FunctionalRequirementGraph, PipelineResult, SearchRegionContract, freeze_search_region_contract
    from .planning import plan_with_common_astar
    from .search import search_until_satisfied
    from .search_order import resolve_search_order, validate_search_order_preflight
    from .spec_provider import provider_for_mode
except ImportError:
    from mujoco_scenes.functional_tamp_pipeline.errors import (
        PipelineError, VLMSpecificationError, ReplaySpecificationError, SearchRegionContractError
    )
    from mujoco_scenes.functional_tamp_pipeline.models import (
        FunctionalRequirementGraph, PipelineResult, SearchRegionContract, freeze_search_region_contract
    )
    from mujoco_scenes.functional_tamp_pipeline.planning import plan_with_common_astar
    from mujoco_scenes.functional_tamp_pipeline.search import search_until_satisfied
    from mujoco_scenes.functional_tamp_pipeline.search_order import resolve_search_order, validate_search_order_preflight
    from mujoco_scenes.functional_tamp_pipeline.spec_provider import provider_for_mode


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs" / "functional_tamp_pipeline"


EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class _RunState:
    domain: str
    variant: str
    internal_variant: str
    mode: str
    run_dir: Path
    specification: FunctionalRequirementGraph | None = None
    spec_acquisition: str = "live_provider"
    specification_input: str | None = None
    specification_sha256: str | None = None
    provider_model: str | None = None
    search_order: str = "auto"
    search_order_source_effective: str | None = None
    search_seed_requested: int | None = None
    search_seed_effective: int | None = None
    resolved_search_order: tuple[str, ...] = ()
    search_contract: SearchRegionContract | None = None
    exploration_actuation: str = "unknown"
    visualization_requested: bool = False
    git_commit: str | None = None
    git_dirty: bool | None = None
    started_at_utc: str = ""
    finished_at_utc: str = ""
    runtime_sec: float = 0.0
    terminal_status: str = "PIPELINE_EXCEPTION"
    failure_reason: str | None = None
    failure_category: str | None = None
    observer_errors: list[dict[str, Any]] = field(default_factory=list)


def _make_guarded_observer(
    observer: EventCallback | None,
    state: _RunState,
) -> EventCallback | None:
    if observer is None:
        return None

    def _guarded_callback(event_type: str, payload: dict[str, Any]) -> None:
        enriched_payload = dict(payload)
        if event_type in {"search_region_selected", "search_region_opened"}:
            if "search_order_source_effective" not in enriched_payload and state.search_order_source_effective is not None:
                enriched_payload["search_order_source_effective"] = state.search_order_source_effective
            if "search_seed_effective" not in enriched_payload and state.search_seed_effective is not None:
                enriched_payload["search_seed_effective"] = state.search_seed_effective
        try:
            observer(event_type, enriched_payload)
        except Exception as error:
            print(
                f"OBSERVER ERROR on {event_type}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            state.observer_errors.append({
                "event": event_type,
                "type": type(error).__name__,
                "message": str(error),
            })
    return _guarded_callback


def _emit_event(observer: EventCallback | None, event_type: str, payload: dict[str, Any]) -> None:
    if observer is not None:
        observer(event_type, payload)


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

    # Action plan discovery (Kitchen nested, Workshop root, Living Room plan)
    if (run_dir / "action_sequence" / "action_plan.json").exists():
        artifacts["action_plan"] = "action_sequence/action_plan.json"
        artifacts["final_plan"] = "action_sequence/action_plan.json"
    elif (run_dir / "action_plan.json").exists():
        artifacts["action_plan"] = "action_plan.json"
        artifacts["final_plan"] = "action_plan.json"
    elif (run_dir / "action_sequence" / "plan.json").exists():
        artifacts["final_plan"] = "action_sequence/plan.json"

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


def _acquire_spec_or_fail(
    state: _RunState,
    task: str,
    images: list[Path],
    specification_json: Path | str | None,
) -> PipelineResult | None:
    if state.mode == "vlm" and specification_json is None:
        try:
            state.specification, state.spec_acquisition, state.specification_input = _load_or_acquire_specification(
                domain=state.domain,
                mode=state.mode,
                task=task,
                images=images,
                specification_json=specification_json,
            )
            return None
        except VLMSpecificationError as error:
            state.terminal_status = "VLM_SPEC_FAILED"
            cat = getattr(error, "category", None) or "MALFORMED_VLM_SPECIFICATION"
            state.failure_category = cat
            state.failure_reason = str(error)
            res = PipelineResult(
                domain=state.domain,
                variant=state.variant,
                mode=state.mode,
                status="VLM_SPEC_FAILED",
                failure_reason=str(error),
                failure_category=cat,
            )
            _write_json(state.run_dir / "result.json", res.to_dict())
            return res
    else:
        state.specification, state.spec_acquisition, state.specification_input = _load_or_acquire_specification(
            domain=state.domain,
            mode=state.mode,
            task=task,
            images=images,
            specification_json=specification_json,
        )
        return None


def _write_run_manifest(state: _RunState) -> None:
    manifest = {
        "schema_version": 1,
        "domain": state.domain,
        "variant": state.variant,
        "internal_variant": state.internal_variant,
        "spec_mode": state.mode,
        "spec_provider_source": state.specification.source if state.specification else None,
        "spec_acquisition": state.spec_acquisition,
        "specification_sha256": state.specification_sha256,
        "specification_input": state.specification_input,
        "provider_model": state.provider_model,
        "search_order_source_requested": state.search_order,
        "search_order_source_effective": state.search_order_source_effective,
        "search_seed_requested": state.search_seed_requested,
        "search_seed_effective": state.search_seed_effective,
        "provider_region_ranking": list(state.specification.region_ranking) if state.specification else [],
        "region_order_used": list(state.resolved_search_order),
        "search_policy_version": state.search_contract.policy_version if state.search_contract else "phase3_p3h_v1",
        "search_contract": state.search_contract.to_dict() if state.search_contract else None,
        "exploration_actuation": state.exploration_actuation,
        "execution_state": "planning_only",
        "visualization_requested": state.visualization_requested,
        "git_commit": state.git_commit,
        "git_dirty": state.git_dirty,
        "started_at_utc": state.started_at_utc,
        "finished_at_utc": state.finished_at_utc,
        "pipeline_runtime_seconds": round(state.runtime_sec, 4),
        "terminal_status": state.terminal_status,
        "failure_reason": state.failure_reason,
        "failure_category": state.failure_category,
        "observer_errors": list(state.observer_errors),
        "artifacts": _collect_artifacts(state.run_dir),
    }
    _write_json(state.run_dir / "run_manifest.json", manifest)


def _safe_write_run_manifest(state: _RunState) -> Exception | None:
    try:
        _write_run_manifest(state)
        return None
    except Exception as error:
        print(
            f"RUN MANIFEST WRITE FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return error


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


def _run_pipeline_impl(
    *,
    state: _RunState,
    guarded_observer: EventCallback | None,
    specification_json: Path | str | None,
    dry_run: bool,
    verbose: bool,
    observation_images: list[Path] | None,
) -> PipelineResult:
    if state.domain == "kitchen":
        from mujoco_scenes.run_kitchen_vlm_pipeline import _render_initial
        from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import (
            TASK, run_to_plan, scene_for_variant
        )

        scene = scene_for_variant(state.internal_variant)
        images = list(observation_images or [])
        if state.mode == "vlm" and not images and specification_json is None:
            images = _render_initial(scene, state.run_dir / "vlm_inputs", 1280, 960)

        print("[1/5] Functional specification", flush=True)
        _emit_event(guarded_observer, "stage_changed", {"stage": "specification"})
        fail_res = _acquire_spec_or_fail(
            state=state,
            task=TASK,
            images=images,
            specification_json=specification_json,
        )
        if fail_res is not None:
            return fail_res
        _write_json(state.run_dir / "functional_specification.json", state.specification.to_dict())
        _write_json(state.run_dir / "functional_requirement_graph.json", state.specification.to_dict())
        state.specification_sha256 = _compute_file_sha256(state.run_dir / "functional_specification.json")

        state.search_contract = freeze_search_region_contract(
            state.specification,
            domain=state.domain,
            source=state.search_order,
            mode=state.mode,
            variant=state.variant,
            seed=state.search_seed_requested,
        )
        state.resolved_search_order = state.search_contract.canonical_region_ids
        if state.search_contract.no_search_required:
            state.search_order_source_effective = "not_applicable"
        elif state.search_order in {"oracle", "provider", "random", "fixed"}:
            state.search_order_source_effective = "oracle" if state.search_order == "fixed" else state.search_order
        else:
            state.search_order_source_effective = "oracle" if (state.mode == "gt" and state.variant is not None) else "provider"
        state.search_seed_effective = state.search_contract.search_seed
        _emit_event(guarded_observer, "spec_ready", {
            "graph": state.specification.to_dict(),
            "source": state.specification.source,
            "provider_region_ranking": list(state.specification.region_ranking),
            "search_order_source_effective": state.search_order_source_effective,
            "region_order_used": list(state.resolved_search_order),
            "search_seed_effective": state.search_seed_effective,
        })

        print("[2/5] Initial perception", flush=True)
        _emit_event(guarded_observer, "stage_changed", {"stage": "perception"})
        print("[3/5] Functional grounding and ranked region search", flush=True)
        _emit_event(guarded_observer, "stage_changed", {"stage": "search_grounding"})
        result = run_to_plan(
            variant_label=state.variant,
            internal_variant=state.internal_variant,
            mode=state.mode,
            specification=state.specification,
            output_dir=state.run_dir,
            scene=scene,
            search_order=state.resolved_search_order,
            observer=guarded_observer,
        )
        if result.assignment:
            print("[4/5] Role assignment", flush=True)
            print(json.dumps(result.assignment, indent=2, sort_keys=True), flush=True)
            print("[5/5] A* planning", flush=True)
            _emit_event(guarded_observer, "stage_changed", {"stage": "planning"})
            _emit_event(guarded_observer, "plan_ready", {
                "actions": list(result.plan),
                "search_statistics": result.search_statistics,
            })
            for action in result.plan:
                print(f"  {action['action_index']:02d}. {action['operator']}({', '.join(action['arguments'])})", flush=True)
        _write_json(state.run_dir / "result.json", result.to_dict())
        return result

    if state.domain == "living_room":
        from mujoco_scenes.functional_tamp_pipeline.domains.living_room import (
            TASK, run_to_plan
        )

        images = list(observation_images or [])
        if state.mode == "vlm" and not images and specification_json is None:
            from PIL import Image
            from mujoco_scenes.living_room_region_scene import (
                L2_CAMERAS, L2LivingRoomRegionScene,
            )
            from mujoco_scenes.living_room_variants import scene_name

            vlm_scene = L2LivingRoomRegionScene(scene_name(state.internal_variant), robot="none")
            image_dir = state.run_dir / "vlm_inputs"
            image_dir.mkdir(parents=True, exist_ok=True)
            for camera in L2_CAMERAS:
                path = image_dir / f"{camera}.png"
                Image.fromarray(vlm_scene.render_frame(camera, 1280, 960)).save(path)
                images.append(path)

        print("[1/5] Functional specification", flush=True)
        _emit_event(guarded_observer, "stage_changed", {"stage": "specification"})
        fail_res = _acquire_spec_or_fail(
            state=state,
            task=TASK,
            images=images,
            specification_json=specification_json,
        )
        if fail_res is not None:
            return fail_res
        _write_json(state.run_dir / "functional_specification.json", state.specification.to_dict())
        _write_json(state.run_dir / "functional_requirement_graph.json", state.specification.to_dict())
        state.specification_sha256 = _compute_file_sha256(state.run_dir / "functional_specification.json")

        state.search_contract = freeze_search_region_contract(
            state.specification,
            domain=state.domain,
            source=state.search_order,
            mode=state.mode,
            variant=state.variant,
            seed=state.search_seed_requested,
        )
        state.resolved_search_order = state.search_contract.canonical_region_ids
        if state.search_contract.no_search_required:
            state.search_order_source_effective = "not_applicable"
        elif state.search_order in {"oracle", "provider", "random", "fixed"}:
            state.search_order_source_effective = "oracle" if state.search_order == "fixed" else state.search_order
        else:
            state.search_order_source_effective = "oracle" if (state.mode == "gt" and state.variant is not None) else "provider"
        state.search_seed_effective = state.search_contract.search_seed
        _emit_event(guarded_observer, "spec_ready", {
            "graph": state.specification.to_dict(),
            "source": state.specification.source,
            "provider_region_ranking": list(state.specification.region_ranking),
            "search_order_source_effective": state.search_order_source_effective,
            "region_order_used": list(state.resolved_search_order),
            "search_seed_effective": state.search_seed_effective,
        })

        print("[2/5] Initial perception", flush=True)
        _emit_event(guarded_observer, "stage_changed", {"stage": "perception"})
        print("[3/5] Global functional grounding", flush=True)
        result = run_to_plan(
            variant_label=state.variant,
            internal_variant=state.internal_variant,
            mode=state.mode,
            specification=state.specification,
            output_dir=state.run_dir,
            observer=guarded_observer,
        )
        if result.assignment:
            print("[4/5] Role assignment", flush=True)
            print(json.dumps(result.assignment, indent=2, sort_keys=True), flush=True)
            print("[5/5] A* planning", flush=True)
            _emit_event(guarded_observer, "stage_changed", {"stage": "planning"})
            _emit_event(guarded_observer, "plan_ready", {
                "actions": list(result.plan),
                "search_statistics": result.search_statistics,
            })
            for action in result.plan:
                print(f"  {action['action_index']:02d}. {action['operator']}({', '.join(action['arguments'])})", flush=True)
        _write_json(state.run_dir / "result.json", result.to_dict())
        return result

    if state.domain != "workshop":
        raise NotImplementedError(
            f"The canonical adapter for {state.domain} has not been integrated yet"
        )

    from mujoco_scenes.workshop_scene import WorkshopScene
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import (
        SURFACE, WorkshopDomainAdapter, WorkshopPlanningCompiler
    )

    scene = WorkshopScene(robot="google", variant=state.internal_variant)
    images = list(observation_images or [])
    if state.mode == "vlm" and not images and specification_json is None:
        images = _capture_workshop_vlm_inputs(scene, state.run_dir / "vlm_inputs")

    print("[1/5] Functional specification", flush=True)
    _emit_event(guarded_observer, "stage_changed", {"stage": "specification"})
    task = WorkshopDomainAdapter.task_instruction
    fail_res = _acquire_spec_or_fail(
        state=state,
        task=task,
        images=images,
        specification_json=specification_json,
    )
    if fail_res is not None:
        return fail_res
    _write_json(state.run_dir / "functional_specification.json", state.specification.to_dict())
    _write_json(state.run_dir / "functional_requirement_graph.json", state.specification.to_dict())
    state.specification_sha256 = _compute_file_sha256(state.run_dir / "functional_specification.json")

    state.search_contract = freeze_search_region_contract(
        state.specification,
        domain=state.domain,
        source=state.search_order,
        mode=state.mode,
        variant=state.variant,
        seed=state.search_seed_requested,
    )
    state.resolved_search_order = state.search_contract.canonical_region_ids
    if state.search_contract.no_search_required:
        state.search_order_source_effective = "not_applicable"
    elif state.search_order in {"oracle", "provider", "random", "fixed"}:
        state.search_order_source_effective = "oracle" if state.search_order == "fixed" else state.search_order
    else:
        state.search_order_source_effective = "oracle" if (state.mode == "gt" and state.variant is not None) else "provider"
    state.search_seed_effective = state.search_contract.search_seed
    _emit_event(guarded_observer, "spec_ready", {
        "graph": state.specification.to_dict(),
        "source": state.specification.source,
        "provider_region_ranking": list(state.specification.region_ranking),
        "search_order_source_effective": state.search_order_source_effective,
        "region_order_used": list(state.resolved_search_order),
        "search_seed_effective": state.search_seed_effective,
    })

    adapter = WorkshopDomainAdapter(
        state.internal_variant,
        state.specification,
        scene=scene,
        physical_open=not dry_run,
        output_dir=str(state.run_dir / "perception"),
        verbose=verbose,
        telemetry_enabled=(guarded_observer is not None),
    )
    print("[2/5] Initial perception", flush=True)
    _emit_event(guarded_observer, "stage_changed", {"stage": "perception"})
    print("[3/5] Functional grounding and ranked region search", flush=True)
    _emit_event(guarded_observer, "stage_changed", {"stage": "search_grounding"})
    try:
        satisfaction, inspected = search_until_satisfied(
            adapter,
            state.specification,
            search_contract=state.search_contract,
            search_order=state.resolved_search_order,
            observer=guarded_observer,
        )
    except TypeError:
        satisfaction, inspected = search_until_satisfied(
            adapter,
            state.specification,
            search_order=state.resolved_search_order,
            observer=guarded_observer,
        )
    _write_json(state.run_dir / "observed_scene_graph.json", adapter.graph.to_dict())
    _write_json(state.run_dir / "detection_diagnostics.json", {
        "records": adapter.controller.detection_diagnostics,
    })
    _write_json(state.run_dir / "graph_grounding_result.json", satisfaction.to_dict())
    _write_json(state.run_dir / "satisfaction.json", {
        "satisfied": satisfaction.satisfied,
        "status": satisfaction.status,
        "assignment": satisfaction.assignment,
        "missing_requirements": list(satisfaction.missing_requirements),
        "evidence": satisfaction.evidence,
    })
    if not satisfaction.satisfied or satisfaction.assignment is None:
        reason = ", ".join(satisfaction.missing_requirements) or "NO_GLOBAL_ASSIGNMENT"
        print(f"FUNCTIONAL GROUNDING FAILED: {reason}", flush=True)
        state.terminal_status = satisfaction.status
        state.failure_reason = reason
        state.failure_category = None
        result = PipelineResult(
            domain=state.domain, variant=state.variant, mode=state.mode, status=satisfaction.status,
            inspected_regions=inspected, failure_reason=reason, failure_category=None,
        )
        _write_json(state.run_dir / "result.json", result.to_dict())
        return result

    print("[4/5] Role assignment", flush=True)
    print("ROLE ASSIGNMENT", flush=True)
    for role, assigned_val in sorted(satisfaction.assignment.items()):
        print(f"{role} -> {assigned_val}", flush=True)

    print("[5/5] A* planning", flush=True)
    _emit_event(guarded_observer, "stage_changed", {"stage": "planning"})
    planned = plan_with_common_astar(
        WorkshopPlanningCompiler(), satisfaction.assignment, adapter.planning_context()
    )
    _write_json(state.run_dir / "action_plan.json", {
        "planner": planned.search.statistics,
        "actions": list(planned.actions),
        "validation": planned.validation,
        "exploratory_open_actions_excluded": True,
    })
    from .audit import audit_plan_grounding
    plan_audit = audit_plan_grounding(
        state.specification, adapter.graph, satisfaction, planned.actions, home_region=SURFACE
    )
    _write_json(state.run_dir / "plan_grounding_audit.json", plan_audit)
    _emit_event(guarded_observer, "plan_ready", {
        "actions": list(planned.actions),
        "search_statistics": planned.search.statistics,
    })
    for action in planned.actions:
        print(
            f"  {action['action_index']:02d}. {action['operator']}"
            f"({', '.join(action['arguments'])})",
            flush=True,
        )

    result = PipelineResult(
        domain=state.domain,
        variant=state.variant,
        mode=state.mode,
        status="ACTION_SEQUENCE_READY",
        inspected_regions=inspected,
        assignment=satisfaction.assignment,
        plan=planned.actions,
        search_statistics=planned.search.statistics,
        failure_reason=None,
    )
    _write_json(state.run_dir / "result.json", result.to_dict())
    return result


def run_pipeline(
    *,
    domain: str,
    variant: str,
    mode: str,
    search_order: str = "auto",
    search_seed: int | None = None,
    specification_json: Path | str | None = None,
    visualize: bool = False,
    observer: EventCallback | None = None,
    output_root: Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
    verbose: bool = False,
    observation_images: list[Path] | None = None,
) -> PipelineResult:
    # Early preflight validation before any expensive provider / simulator work
    validate_search_order_preflight(domain, search_order, mode=mode, seed=search_seed)

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

    state = _RunState(
        domain=domain,
        variant=variant,
        internal_variant=internal_variant,
        mode=mode,
        run_dir=run_dir,
        spec_acquisition=spec_acquisition,
        specification_input=specification_input,
        provider_model=provider_model,
        search_order=search_order,
        search_seed_requested=search_seed,
        exploration_actuation=exploration_actuation,
        visualization_requested=visualize,
        git_commit=git_commit,
        git_dirty=git_dirty,
        started_at_utc=started_at_utc,
    )
    guarded_observer = _make_guarded_observer(observer, state)

    _emit_event(guarded_observer, "run_started", {
        "domain": state.domain,
        "variant": state.variant,
        "spec_mode": state.mode,
        "search_order_source_requested": state.search_order,
        "search_seed_requested": state.search_seed_requested,
        "exploration_actuation": state.exploration_actuation,
        "run_dir": str(state.run_dir),
    })

    try:
        result = _run_pipeline_impl(
            state=state,
            guarded_observer=guarded_observer,
            specification_json=specification_json,
            dry_run=dry_run,
            verbose=verbose,
            observation_images=observation_images,
        )
        state.terminal_status = result.status
        _emit_event(guarded_observer, "stage_changed", {"stage": "complete"})
        _emit_event(guarded_observer, "run_finished", {
            "terminal_status": result.status,
            "run_dir": str(state.run_dir),
            "inspected_regions": list(result.inspected_regions),
        })
        return result
    except Exception as error:
        state.terminal_status = "PIPELINE_EXCEPTION"
        _emit_event(guarded_observer, "run_failed", {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "run_dir": str(state.run_dir),
        })
        raise
    finally:
        state.finished_at_utc = datetime.now(timezone.utc).isoformat()
        state.runtime_sec = time.perf_counter() - start_t
        if observer is not None and hasattr(observer, "drain_errors"):
            try:
                display_errors = observer.drain_errors()
                state.observer_errors.extend(display_errors)
            except Exception:
                pass
        _safe_write_run_manifest(state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("kitchen", "workshop", "living_room"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--mode", required=True, choices=("gt", "vlm"))
    parser.add_argument(
        "--search-order",
        choices=("auto", "oracle", "provider", "random", "fixed"),
        default="auto",
        help="Search-order source: 'auto' (default, oracle for GT, provider for VLM), 'oracle', 'provider', 'random', or 'fixed' (deprecated alias for oracle).",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=None,
        help="Random seed for '--search-order random'. Must be a non-negative integer.",
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
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch non-intrusive LiveMosaicViewer to monitor pipeline state in real time.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    visualizer = None
    if args.visualize:
        try:
            from mujoco_scenes.functional_tamp_pipeline.live_visualizer import LivePipelineVisualizer
            visualizer = LivePipelineVisualizer(
                title=f"TAMP Pipeline - {args.domain.upper()} {args.variant} ({args.mode.upper()})"
            )
        except Exception as error:
            print(
                f"VISUALIZER WARNING: Could not initialize visualizer ({type(error).__name__}: {error}). "
                "Proceeding in headless mode.",
                file=sys.stderr,
                flush=True,
            )
            visualizer = None

    result = None
    try:
        result = run_pipeline(
            domain=args.domain,
            variant=args.variant,
            mode=args.mode,
            search_order=args.search_order,
            search_seed=args.search_seed,
            specification_json=args.specification_json,
            visualize=args.visualize,
            observer=visualizer,
            output_root=args.output_root,
            dry_run=args.dry_run,
            verbose=args.verbose,
            observation_images=args.observation_image,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"PIPELINE FAILED: {error}", flush=True)
        if args.verbose:
            traceback.print_exc()
        if visualizer is not None:
            try:
                visualizer.flush_latest_frame(timeout_sec=0.25)
            except Exception:
                pass
        return 1
    finally:
        if visualizer is not None:
            if result is not None and result.status in {"ACTION_SEQUENCE_READY", "INFEASIBLE"}:
                try:
                    visualizer.hold_until_closed()
                except KeyboardInterrupt:
                    pass
            visualizer.close()

    print(f"PIPELINE STATUS: {result.status}", flush=True)
    return 0 if result.status in {"ACTION_SEQUENCE_READY", "INFEASIBLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
