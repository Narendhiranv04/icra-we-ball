"""Deterministic local manipulation stance and base-frame pose utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PlanarStance:
    x: float
    y: float
    yaw: float
    anchor_dx: float = 0.0
    anchor_dy: float = 0.0
    anchor_dyaw: float = 0.0


@dataclass(frozen=True)
class StanceEvaluation:
    stance: PlanarStance
    valid: bool
    collision_clearance_m: float | None
    ik_residual_m: float | None
    joint_displacement_rad: float | None
    candidate_index: int
    reason: str | None = None


def yaw_rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def qpos_to_world_stance(
    base_qpos: np.ndarray, *, home_y: float
) -> PlanarStance:
    forward, lateral, yaw = map(float, base_qpos)
    return PlanarStance(-lateral, home_y + forward, yaw)


def world_stance_to_qpos(
    stance: PlanarStance, *, home_y: float
) -> np.ndarray:
    return np.array((stance.y - home_y, -stance.x, stance.yaw), dtype=float)


def base_relative_pose_to_world(
    stance: PlanarStance,
    relative_position: np.ndarray,
    relative_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a robot-base-frame pose at a planar stance into world."""
    rotation = yaw_rotation(stance.yaw)
    position = np.array((stance.x, stance.y, 0.0)) + rotation @ np.asarray(
        relative_position, dtype=float
    )
    return position, rotation @ np.asarray(relative_rotation, dtype=float)


class ManipulationStancePlanner:
    """Rank a deterministic bounded SE(2) search around a named anchor."""

    TRANSLATION_OFFSETS_M = (0.0, -0.05, 0.05, -0.10, 0.10, -0.15, 0.15)
    YAW_OFFSETS_RAD = tuple(
        math.radians(value) for value in (0, -10, 10, -20, 20, -30, 30, -40, 40)
    )
    MAX_TRANSLATION_M = 0.19

    def candidates(self, anchor: PlanarStance) -> tuple[PlanarStance, ...]:
        rows = []
        for dx in self.TRANSLATION_OFFSETS_M:
            for dy in self.TRANSLATION_OFFSETS_M:
                distance = math.hypot(dx, dy)
                if distance > self.MAX_TRANSLATION_M:
                    continue
                for dyaw in self.YAW_OFFSETS_RAD:
                    rows.append((distance, abs(dyaw), dx, dy, dyaw))
        rows.sort()
        return tuple(
            PlanarStance(
                anchor.x + dx,
                anchor.y + dy,
                anchor.yaw + dyaw,
                dx,
                dy,
                dyaw,
            )
            for _, _, dx, dy, dyaw in rows
        )

    def select(
        self,
        anchor: PlanarStance,
        evaluate: Callable[[PlanarStance, int], StanceEvaluation],
    ) -> tuple[StanceEvaluation | None, tuple[StanceEvaluation, ...]]:
        # Candidates are already in the complete deterministic ranking order.
        # Stop at the first valid stance so physical execution is not delayed
        # by hundreds of lower-ranked IK checks.
        evaluations = []
        for index, candidate in enumerate(self.candidates(anchor)):
            row = evaluate(candidate, index)
            evaluations.append(row)
            if row.valid:
                return row, tuple(evaluations)
        return None, tuple(evaluations)
