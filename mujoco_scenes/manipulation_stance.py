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
    feasible_grasp_families: tuple[str, ...] = ()
    feasible_probe_ids: tuple[str, ...] = ()
    predicted_grasp_score: float = 0.0


@dataclass(frozen=True)
class BilateralContactPrediction:
    """Geometric ranking evidence; never authorizes a physical weld."""

    left_pad_target_distance_m: float
    right_pad_target_distance_m: float
    jaw_axis_target_alignment: float
    target_between_jaws: bool
    available_finger_closure_m: float
    required_finger_closure_m: float
    predicted_left_contact_possible: bool
    predicted_right_contact_possible: bool
    predicted_bilateral_contact: bool
    left_first_contact_closure: float | None = None
    right_first_contact_closure: float | None = None
    left_first_contact_distance_m: float | None = None
    right_first_contact_distance_m: float | None = None
    first_contact_closure_delta: float | None = None
    maximum_pre_contact_penetration_m: float = 0.0
    contact_distance_threshold_m: float = 0.0005
    left_target_geom: str | None = None
    right_target_geom: str | None = None


@dataclass(frozen=True)
class GraspStanceEvaluation:
    stance: PlanarStance
    stance_index: int
    grasp_candidate_id: str
    grasp_family: str
    carry_valid: bool
    approach_valid: bool
    grasp_ik_position_error_m: float | None
    grasp_ik_angle_error_rad: float | None
    collision_free: bool
    minimum_collision_clearance_m: float | None
    contact_prediction: BilateralContactPrediction | None
    base_displacement_m: float
    joint_displacement_rad: float | None
    valid: bool
    failure_reason: str | None = None

    @property
    def predicted_grasp_score(self) -> float:
        prediction = self.contact_prediction
        if prediction is None:
            return float("-inf")
        worst_distance = max(
            prediction.left_pad_target_distance_m,
            prediction.right_pad_target_distance_m,
        )
        asymmetry = abs(
            prediction.left_pad_target_distance_m
            - prediction.right_pad_target_distance_m
        )
        return (
            (10.0 if prediction.predicted_bilateral_contact else 0.0)
            + 2.0 * prediction.jaw_axis_target_alignment
            - 20.0 * worst_distance
            - 10.0 * asymmetry
        )


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

    TRANSLATION_OFFSETS_M = (
        0.0, -0.025, 0.025, -0.05, 0.05, -0.075, 0.075,
        -0.10, 0.10, -0.15, 0.15,
    )
    YAW_OFFSETS_RAD = tuple(
        math.radians(value)
        for value in (
            0, -5, 5, -10, 10, -15, 15, -20, 20,
            -25, 25, -30, 30, -35, 35, -40, 40,
            -45, 45, -50, 50,
        )
    )
    MAX_TRANSLATION_M = 0.19
    DEFAULT_SHORTLIST_SIZE = 8

    def candidates(
        self,
        anchor: PlanarStance,
        *,
        preferred_yaw: float | None = None,
    ) -> tuple[PlanarStance, ...]:
        rows = []
        for dx in self.TRANSLATION_OFFSETS_M:
            for dy in self.TRANSLATION_OFFSETS_M:
                distance = math.hypot(dx, dy)
                if distance > self.MAX_TRANSLATION_M:
                    continue
                for dyaw in self.YAW_OFFSETS_RAD:
                    yaw_error = (
                        abs(math.atan2(
                            math.sin(anchor.yaw + dyaw - preferred_yaw),
                            math.cos(anchor.yaw + dyaw - preferred_yaw),
                        ))
                        if preferred_yaw is not None else abs(dyaw)
                    )
                    priority = (
                        (
                            distance + 0.12 * yaw_error,
                            yaw_error,
                            distance,
                        )
                        if preferred_yaw is not None
                        else (distance, abs(dyaw), yaw_error)
                    )
                    rows.append((*priority, dx, dy, dyaw))
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
            for _, _, _, dx, dy, dyaw in rows
        )

    def select(
        self,
        anchor: PlanarStance,
        evaluate: Callable[[PlanarStance, int], StanceEvaluation],
    ) -> tuple[StanceEvaluation | None, tuple[StanceEvaluation, ...]]:
        ranked, evaluations = self.shortlist(
            anchor,
            evaluate,
            maximum=self.DEFAULT_SHORTLIST_SIZE,
        )
        return (ranked[0] if ranked else None), evaluations

    def shortlist(
        self,
        anchor: PlanarStance,
        evaluate: Callable[[PlanarStance, int], StanceEvaluation],
        *,
        maximum: int = DEFAULT_SHORTLIST_SIZE,
        candidate_limit: int | None = None,
        preferred_yaw: float | None = None,
        minimum_evaluations: int | None = None,
    ) -> tuple[tuple[StanceEvaluation, ...], tuple[StanceEvaluation, ...]]:
        """Evaluate a bounded set and rank valid stances by grasp quality."""
        if maximum <= 0:
            raise ValueError("maximum shortlist size must be positive")
        candidates = self.candidates(anchor, preferred_yaw=preferred_yaw)
        if candidate_limit is not None:
            candidates = candidates[:candidate_limit]
        evaluation_rows = []
        valid_count = 0
        for index, candidate in enumerate(candidates):
            row = evaluate(candidate, index)
            evaluation_rows.append(row)
            valid_count += int(row.valid)
            if (
                minimum_evaluations is not None
                and len(evaluation_rows) >= minimum_evaluations
                and valid_count >= maximum
            ):
                break
        evaluations = tuple(evaluation_rows)
        valid = [row for row in evaluations if row.valid]
        valid.sort(
            key=lambda row: (
                -len(row.feasible_grasp_families),
                -row.predicted_grasp_score,
                -(
                    row.collision_clearance_m
                    if row.collision_clearance_m is not None
                    else float("-inf")
                ),
                row.ik_residual_m if row.ik_residual_m is not None else float("inf"),
                math.hypot(row.stance.anchor_dx, row.stance.anchor_dy),
                abs(row.stance.anchor_dyaw),
                (
                    row.joint_displacement_rad
                    if row.joint_displacement_rad is not None
                    else float("inf")
                ),
                row.candidate_index,
            )
        )
        return tuple(valid[:maximum]), evaluations


def rank_grasp_stance_pairs(
    evaluations: tuple[GraspStanceEvaluation, ...],
) -> tuple[GraspStanceEvaluation, ...]:
    """Deterministically rank only complete collision-free pair plans."""
    valid = [row for row in evaluations if row.valid]
    valid.sort(
        key=lambda row: (
            -int(bool(
                row.contact_prediction
                and row.contact_prediction.predicted_bilateral_contact
            )),
            -row.predicted_grasp_score,
            max(
                row.contact_prediction.left_pad_target_distance_m,
                row.contact_prediction.right_pad_target_distance_m,
            ) if row.contact_prediction else float("inf"),
            abs(
                row.contact_prediction.left_pad_target_distance_m
                - row.contact_prediction.right_pad_target_distance_m
            ) if row.contact_prediction else float("inf"),
            (
                row.grasp_ik_position_error_m
                if row.grasp_ik_position_error_m is not None else float("inf")
            ),
            (
                row.grasp_ik_angle_error_rad
                if row.grasp_ik_angle_error_rad is not None else float("inf")
            ),
            row.base_displacement_m,
            (
                row.joint_displacement_rad
                if row.joint_displacement_rad is not None else float("inf")
            ),
            row.stance_index,
            row.grasp_candidate_id,
        )
    )
    return tuple(valid)
