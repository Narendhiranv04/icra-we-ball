"""Deterministic closed-to-open sequential observed-state demonstration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mujoco

from mujoco_scenes.observed_state import ObservedStateRun


REGION_DESTINATIONS = {
    "C1": "cupboard1",
    "C2": "cupboard2",
    "D1": "home",
    "D2": "home",
    "B1": "box",
}
INTERFERING_OPEN_REGIONS = {"C2": "B1", "B1": "C2"}


class SequentialInspectionAdapter:
    """Reuse the existing base move and controller-driven opening interfaces."""

    def __init__(self, scene):
        self.scene = scene
        self.mover = None
        if scene.has_robot:
            from mujoco_scenes.mobile_motion import MobileMoveExecutor

            self.mover = MobileMoveExecutor(scene.model, scene.data)

    def _move(self, region_id: str) -> None:
        if self.mover is None:
            return
        destination = REGION_DESTINATIONS[region_id]
        self.mover.request_move(destination)
        for _ in range(250_000):
            if not self.mover.busy:
                return
            self.mover.update()
            mujoco.mj_step(self.scene.model, self.scene.data)
        raise RuntimeError(
            f"Existing mobile move action did not finish for {region_id}"
        )

    def inspect(self, region_id: str) -> None:
        """Move to and open exactly one requested region."""
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
            self.scene.close_container(conflicting)
        self._move(region_id)
        self.scene.open_container(region_id)


def run_sequential_inspection(
    scene,
    sequence: Iterable[str],
    *,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    width: int = 640,
    height: int = 480,
    voxel_size: float = 0.003,
) -> ObservedStateRun:
    """Observe closed reset, then inspect and persist one region at a time."""
    sequence = tuple(sequence)
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
    session = ObservedStateRun.create_for_scene(
        scene,
        runs_root=runs_root,
        run_id=run_id,
        voxel_size=voxel_size,
        run_config={
            "mode": "sequential_inspection",
            "inspection_sequence": list(sequence),
            "resolution": [width, height],
            "uses_existing_mobile_move": bool(scene.has_robot),
            "opening_adapter": "KitchenScene.open_container",
        },
    )
    if session.next_stage != 0:
        raise RuntimeError(
            f"Sequential output already contains stages: {session.run_dir}. "
            "Choose a new --run-id so the run begins at 000_initial."
        )
    print(f"\n[OBSERVED STATE] Stage 000: closed initial observation")
    _run, stage_dir = session.observe_scene(
        scene,
        stage_label="initial",
        width=width,
        height=height,
    )
    print(f"  Saved: {stage_dir}")

    adapter = SequentialInspectionAdapter(scene)
    for index, region_id in enumerate(sequence, start=1):
        print(f"[OBSERVED STATE] Stage {index:03d}: inspect {region_id}")
        adapter.inspect(region_id)
        cloud_run, stage_dir = session.observe_scene(
            scene,
            stage_label=f"after_{region_id}",
            region_opened=region_id,
            width=width,
            height=height,
        )
        print(
            f"  Registry objects: {len(session.registry['objects'])}; "
            f"current fused points: {cloud_run.total_points:,}"
        )
        print(f"  Saved: {stage_dir}")
    print(f"[OBSERVED STATE] Run complete: {session.run_dir}\n")
    return session
