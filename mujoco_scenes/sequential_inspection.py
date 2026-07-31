"""Deterministic closed-to-open sequential observed-state demonstration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from mujoco_scenes.observed_state import ObservedStateRun
from mujoco_scenes.semantic_grounding import (
    SemanticDetector,
    create_semantic_detector,
    load_semantic_config,
)
from mujoco_scenes.task_witness import load_task_requirements


DEFAULT_INSPECTION_ORDER = ("D1", "D2", "C2", "B1", "C1")
REGION_DESTINATIONS = {
    "C1": "cupboard1",
    "C2": "cupboard2",
    "D1": "home",
    "D2": "home",
    "B1": "box",
}
INTERFERING_OPEN_REGIONS = {"C2": "B1", "B1": "C2"}
# The position actuators reach their configured open/closed targets within
# this deterministic window. Running the free-object scene for the historical
# 1000-step default unnecessarily ejects light drawer contents before the
# explicitly configured post-opening settle phase begins.
DIRECT_ACTUATION_STEPS = 200


def resolve_grounding_mode(
    loaded_task: dict[str, Any],
    requested_mode: str,
) -> str:
    """Resolve the production default without changing legacy task meaning."""
    if requested_mode != "auto":
        return requested_mode
    return (
        "joint"
        if loaded_task.get("_task_schema") == "JOINT_ROLE_GROUNDING"
        else "geometry-only"
    )


class SequentialInspectionAdapter:
    """Open containers directly; camera placement is purely virtual."""

    def __init__(self, scene):
        self.scene = scene

    def inspect(self, region_id: str) -> None:
        """Directly open one region; never command a robot or mobile base."""
        if region_id not in REGION_DESTINATIONS:
            available = ", ".join(REGION_DESTINATIONS)
            raise ValueError(
                f"Unknown inspection region '{region_id}'. Available: {available}"
            )
        conflicting = INTERFERING_OPEN_REGIONS.get(region_id)
        if (
            conflicting is not None
            and self.scene.state.container_open_state.get(conflicting, False)
        ):
            # C2's door and B1's lid share physical sweep volume. Closing the
            # previously inspected mechanism preserves deterministic opening.
            self.scene.close_container(
                conflicting, steps=DIRECT_ACTUATION_STEPS
            )
        self.scene.open_container(region_id, steps=DIRECT_ACTUATION_STEPS)
        if hasattr(self.scene, "release_storage_fixture"):
            self.scene.release_storage_fixture(region_id)


def _witness_status(session: ObservedStateRun) -> str:
    witness = session.latest_witness or {}
    return str(witness.get("status", "INCOMPLETE"))


def run_fixed_order_inspection(
    scene,
    session: ObservedStateRun,
    sequence: Iterable[str],
    *,
    adapter: Any,
    observe: Callable[[str, str | None], tuple[Any, Path]],
    stop_on_complete: bool,
) -> ObservedStateRun:
    """Run the common closed-initial, fixed-order observation loop."""
    sequence = tuple(sequence)
    print("\n[OBSERVED STATE] Stage 000: closed initial observation")
    _cloud_run, stage_dir = observe("initial", None)
    print(f"  Witness: {_witness_status(session)}")
    print(f"  Saved: {stage_dir}")

    if stop_on_complete and _witness_status(session) == "COMPLETE":
        session.append_event(
            {
                "event": "INSPECTION_STOPPED_COMPLETE",
                "remaining_regions": list(sequence),
            }
        )
        print("[OBSERVED STATE] Complete witness found before opening a region.")
        return session

    for sequence_index, region_id in enumerate(sequence):
        states = scene.get_region_observation_states()
        if bool(states.get(region_id, {}).get("inspected", False)):
            session.append_event(
                {
                    "event": "REGION_SKIPPED_ALREADY_INSPECTED",
                    "region_id": region_id,
                }
            )
            print(
                f"[OBSERVED STATE] Skip {region_id}: already inspected in "
                "the resumed scene state"
            )
            continue

        print(
            f"[OBSERVED STATE] Stage {session.next_stage:03d}: "
            f"inspect {region_id}"
        )
        adapter.inspect(region_id)
        cloud_run, stage_dir = observe(f"after_{region_id}", region_id)
        print(
            f"  Registry objects: {len(session.registry['objects'])}; "
            f"current fused points: {cloud_run.total_points:,}"
        )
        print(f"  Witness: {_witness_status(session)}")
        print(f"  Saved: {stage_dir}")
        if stop_on_complete and _witness_status(session) == "COMPLETE":
            remaining = list(sequence[sequence_index + 1:])
            session.append_event(
                {
                    "event": "INSPECTION_STOPPED_COMPLETE",
                    "remaining_regions": remaining,
                }
            )
            print(
                "[OBSERVED STATE] Complete witness found; "
                f"remaining unopened regions: {remaining}"
            )
            return session

    if stop_on_complete and _witness_status(session) != "COMPLETE":
        session.append_event(
            {
                "event": "INSPECTION_ORDER_EXHAUSTED",
                "final_witness_status": _witness_status(session),
            }
        )
        print(
            "[OBSERVED STATE] Fixed inspection order exhausted with "
            f"witness status {_witness_status(session)}."
        )
    return session


def run_sequential_inspection(
    scene,
    sequence: Iterable[str] | None = None,
    *,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    width: int = 640,
    height: int = 480,
    voxel_size: float = 0.003,
    task_requirements: str | Path | dict[str, Any] | None = None,
    stop_on_complete: bool = False,
    semantic_detector: SemanticDetector | None = None,
    semantic_backend: str = "none",
    semantic_model: str | None = None,
    semantic_config_path: str | Path | None = None,
    semantic_vocabulary_path: str | Path | None = None,
    semantic_confidence_threshold: float | None = None,
    semantic_min_supporting_views: int | None = None,
    grounding_mode: str = "auto",
    save_semantic_overlays: bool = False,
) -> ObservedStateRun:
    """Observe closed reset, then inspect and persist one region at a time."""
    sequence = tuple(sequence or DEFAULT_INSPECTION_ORDER)
    unknown = [region for region in sequence if region not in REGION_DESTINATIONS]
    if unknown:
        raise ValueError(
            f"Unknown inspection region(s): {', '.join(unknown)}; "
            f"available: {', '.join(REGION_DESTINATIONS)}"
        )
    if scene.state.opened_containers:
        raise RuntimeError(
            "Sequential inspection requires a fresh scene with every region closed"
        )
    loaded_task = load_task_requirements(task_requirements)
    requested_grounding_mode = grounding_mode
    grounding_mode = resolve_grounding_mode(
        loaded_task, grounding_mode
    )
    semantic_config = load_semantic_config(
        semantic_config_path
        if semantic_config_path is not None
        else (
            Path(__file__).resolve().parent
            / "configs"
            / "semantic_grounding.yaml"
        ),
        vocabulary_path=semantic_vocabulary_path,
    )
    if semantic_min_supporting_views is not None:
        semantic_config["fusion"]["minimum_supporting_views"] = int(
            semantic_min_supporting_views
        )
    if semantic_detector is None:
        semantic_detector = create_semantic_detector(
            semantic_config,
            backend=semantic_backend,
            checkpoint=semantic_model,
            confidence_threshold=semantic_confidence_threshold,
        )
    detector_runtime = {
        "name": getattr(
            semantic_detector,
            "name",
            semantic_detector.__class__.__name__,
        ),
        "checkpoint": getattr(semantic_detector, "checkpoint", None),
        "version": getattr(semantic_detector, "version", None),
        "device": getattr(semantic_detector, "device", None),
        "inference_size": getattr(
            semantic_detector, "inference_size", None
        ),
        "confidence_threshold": getattr(
            semantic_detector, "confidence_threshold", None
        ),
    }
    if (
        loaded_task.get("_task_schema") == "JOINT_ROLE_GROUNDING"
        and grounding_mode in {"joint", "semantic-only"}
        and semantic_backend in {"none", "disabled"}
        and semantic_detector.__class__.__name__ == "NullSemanticDetector"
    ):
        raise ValueError(
            "Joint and semantic-only grounding require "
            "--semantic-detector yolo_world or an injected detector"
        )
    session = ObservedStateRun.create_for_scene(
        scene,
        runs_root=runs_root,
        run_id=run_id,
        voxel_size=voxel_size,
        task_requirements=loaded_task,
        semantic_detector=semantic_detector,
        semantic_config=semantic_config,
        grounding_mode=grounding_mode,
        save_semantic_overlays=save_semantic_overlays,
        run_config={
            "mode": "sequential_inspection",
            "inspection_sequence": list(sequence),
            "stop_on_complete": stop_on_complete,
            "resolution": [width, height],
            "uses_robot": False,
            "uses_mobile_base": False,
            "uses_virtual_inspection_rig": True,
            "opening_adapter": "KitchenScene.open_container",
            "opening_actuation_steps": DIRECT_ACTUATION_STEPS,
            "grounding_mode": grounding_mode,
            "requested_grounding_mode": requested_grounding_mode,
            "semantic_backend": semantic_backend,
            # Record the resolved adapter state, not only an optional CLI
            # override. This remains complete when the checkpoint came from
            # semantic_grounding.yaml.
            "semantic_detector": detector_runtime,
            "semantic_model": detector_runtime["checkpoint"],
            "semantic_confidence_threshold": (
                detector_runtime["confidence_threshold"]
                if detector_runtime["confidence_threshold"] is not None
                else semantic_confidence_threshold
            ),
            "semantic_min_supporting_views": (
                semantic_config["fusion"][
                    "minimum_supporting_views"
                ]
            ),
            "save_semantic_overlays": save_semantic_overlays,
        },
    )
    if session.next_stage != 0:
        raise RuntimeError(
            f"Sequential output already contains stages: {session.run_dir}. "
            "Choose a new --run-id so the run begins at 000_initial."
        )

    adapter = SequentialInspectionAdapter(scene)
    def observe(
        stage_label: str,
        region_opened: str | None,
    ) -> tuple[Any, Path]:
        return session.observe_scene(
            scene,
            stage_label=stage_label,
            region_opened=region_opened,
            width=width,
            height=height,
        )

    run_fixed_order_inspection(
        scene,
        session,
        sequence,
        adapter=adapter,
        observe=observe,
        stop_on_complete=stop_on_complete,
    )
    print(f"[OBSERVED STATE] Run complete: {session.run_dir}\n")
    return session
