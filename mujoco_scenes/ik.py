"""Inverse-kinematics backends for profile-driven manipulation."""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import mujoco
import numpy as np

from mujoco_scenes.robot_profiles import ManipulationProfile


IK_POSITION_TOLERANCE = 0.0008
IK_ORIENTATION_TOLERANCE = math.radians(0.7)
IK_MAX_ITERATIONS = 1200
IK_MAX_JOINT_STEP = 0.055
MINK_TIMESTEP = 0.1


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to its shortest axis-angle vector."""
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5)
    ):
        raise ValueError("rotation matrix must be a finite orthonormal 3x3 array")
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.ravel())
    if quat[0] < 0:
        quat = -quat
    norm = float(np.linalg.norm(quat[1:]))
    if norm < 1e-10:
        return np.zeros(3)
    return quat[1:] / norm * (2.0 * math.atan2(norm, float(quat[0])))


class _ProfileIKBase:
    def __init__(
        self,
        model: mujoco.MjModel,
        reference: mujoco.MjData,
        profile: ManipulationProfile,
        *,
        orientation_weight: float = 0.30,
        seed_continuity_weight: float = 0.0,
        maximum_seed_delta_rad: float | None = None,
        maximum_iterations: int = IK_MAX_ITERATIONS,
    ):
        if (
            isinstance(orientation_weight, bool)
            or not isinstance(orientation_weight, (int, float))
            or not math.isfinite(float(orientation_weight))
            or orientation_weight < 0
        ):
            raise ValueError("orientation_weight must be finite and non-negative")
        if (
            isinstance(seed_continuity_weight, bool)
            or not isinstance(seed_continuity_weight, (int, float))
            or not math.isfinite(float(seed_continuity_weight))
            or seed_continuity_weight < 0
        ):
            raise ValueError(
                "seed_continuity_weight must be finite and non-negative"
            )
        if maximum_seed_delta_rad is not None and (
            isinstance(maximum_seed_delta_rad, bool)
            or not isinstance(maximum_seed_delta_rad, (int, float))
            or not math.isfinite(float(maximum_seed_delta_rad))
            or maximum_seed_delta_rad <= 0
        ):
            raise ValueError("maximum_seed_delta_rad must be positive")
        if (
            isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, int)
            or maximum_iterations <= 0
        ):
            raise ValueError("maximum_iterations must be a positive integer")
        self.model = model
        self.profile = profile
        self.orientation_weight = float(orientation_weight)
        self.seed_continuity_weight = float(seed_continuity_weight)
        self.maximum_seed_delta_rad = maximum_seed_delta_rad
        self.maximum_iterations = maximum_iterations
        self.reference_qpos = reference.qpos.copy()
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = self.reference_qpos
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, profile.grip_site
        )
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in profile.arm_joints
            ]
        )
        if self.site_id < 0 or np.any(self.joint_ids < 0):
            raise RuntimeError(
                "Manipulation profile does not match the composed model"
            )
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        limits = model.jnt_range[self.joint_ids]
        limited = model.jnt_limited[self.joint_ids].astype(bool)
        self.lower = np.where(limited, limits[:, 0] + 0.015, -math.pi)
        self.upper = np.where(limited, limits[:, 1] - 0.015, math.pi)

    def _initial_qpos(
        self, seed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed = np.asarray(seed, dtype=float)
        if seed.shape != self.lower.shape or not np.all(np.isfinite(seed)):
            raise ValueError("IK seed has the wrong shape or non-finite values")
        seed = np.clip(seed, self.lower, self.upper)
        local_lower = self.lower
        local_upper = self.upper
        if self.maximum_seed_delta_rad is not None:
            local_lower = np.maximum(
                local_lower, seed - self.maximum_seed_delta_rad
            )
            local_upper = np.minimum(
                local_upper, seed + self.maximum_seed_delta_rad
            )
        qpos = self.reference_qpos.copy()
        qpos[self.qpos_addresses] = seed
        return qpos, seed, local_lower, local_upper

    @staticmethod
    def _validate_target(
        target: np.ndarray, target_rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("IK target position must be a finite 3-vector")
        target_rotation = np.asarray(target_rotation, dtype=float)
        rotation_vector(target_rotation)
        return target, target_rotation

    def _result(
        self,
        target: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        return (
            self.data.qpos[self.qpos_addresses].copy(),
            float(
                np.linalg.norm(
                    target - self.data.site_xpos[self.site_id]
                )
            ),
            float(
                np.linalg.norm(
                    rotation_vector(target_rotation @ rotation.T)
                )
            ),
        )


class DampedLeastSquaresIK(_ProfileIKBase):
    """Original dependency-free damped least-squares IK backend."""

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        target, target_rotation = self._validate_target(
            target, target_rotation
        )
        qpos, seed, local_lower, local_upper = self._initial_qpos(seed)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0
        for _ in range(self.maximum_iterations):
            mujoco.mj_forward(self.model, self.data)
            rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            position_error = target - self.data.site_xpos[self.site_id]
            rotation_error = rotation_vector(
                target_rotation @ rotation.T
            )
            error = np.concatenate(
                (position_error, self.orientation_weight * rotation_error)
            )
            if (
                np.linalg.norm(position_error) < IK_POSITION_TOLERANCE
                and (
                    self.orientation_weight == 0.0
                    or np.linalg.norm(rotation_error)
                    < IK_ORIENTATION_TOLERANCE
                )
            ):
                break

            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(
                self.model,
                self.data,
                jac_pos,
                jac_rot,
                self.site_id,
            )
            jacobian = np.vstack(
                (
                    jac_pos[:, self.dof_addresses],
                    self.orientation_weight * jac_rot[:, self.dof_addresses],
                )
            )
            damping = 0.0025
            damped_inverse = np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6),
                np.eye(6),
            )
            pseudoinverse = jacobian.T @ damped_inverse
            delta = pseudoinverse @ error
            current = self.data.qpos[self.qpos_addresses]
            delta += self.seed_continuity_weight * (
                np.eye(len(current)) - pseudoinverse @ jacobian
            ) @ (seed - current)
            self.data.qpos[self.qpos_addresses] = np.clip(
                current
                + np.clip(
                    delta,
                    -IK_MAX_JOINT_STEP,
                    IK_MAX_JOINT_STEP,
                ),
                local_lower,
                local_upper,
            )

        return self._result(target, target_rotation)


class MinkIK(_ProfileIKBase):
    """Mink differential IK with frozen non-arm DOFs and joint limits."""

    def __init__(
        self,
        model: mujoco.MjModel,
        reference: mujoco.MjData,
        profile: ManipulationProfile,
        **solver_options: Any,
    ):
        super().__init__(model, reference, profile, **solver_options)
        try:
            import mink
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Mink IK is unavailable. Install "
                "mujoco_scenes/requirements.txt or select "
                "MUJOCO_IK_BACKEND=legacy."
            ) from error

        self.mink = mink
        arm_dofs = set(int(index) for index in self.dof_addresses)
        self.frozen_dofs = np.array(
            sorted(set(range(model.nv)) - arm_dofs),
            dtype=int,
        )
        self.freeze_constraint = (
            mink.DofFreezingTask(
                model, self.frozen_dofs.tolist()
            )
            if len(self.frozen_dofs)
            else None
        )
        self.configuration_limit = mink.ConfigurationLimit(model)

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        target, target_rotation = self._validate_target(
            target, target_rotation
        )
        qpos, seed, local_lower, local_upper = self._initial_qpos(seed)
        configuration = self.mink.Configuration(self.model, q=qpos)

        frame_task = self.mink.FrameTask(
            frame_name=self.profile.grip_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=self.orientation_weight,
            gain=0.8,
            lm_damping=1e-4,
        )
        frame_task.set_target(
            self.mink.SE3.from_rotation_and_translation(
                self.mink.SO3.from_matrix(target_rotation),
                np.asarray(target, dtype=float),
            )
        )

        posture_cost = np.zeros(self.model.nv)
        posture_cost[self.dof_addresses] = max(
            1e-3, self.seed_continuity_weight
        )
        posture_task = self.mink.PostureTask(
            self.model,
            cost=posture_cost,
            gain=0.05,
        )
        posture_task.set_target(qpos)
        constraints = (
            [self.freeze_constraint]
            if self.freeze_constraint is not None
            else None
        )

        for _ in range(self.maximum_iterations):
            try:
                velocity = self.mink.solve_ik(
                    configuration,
                    [frame_task, posture_task],
                    dt=MINK_TIMESTEP,
                    solver="daqp",
                    damping=1e-5,
                    limits=[self.configuration_limit],
                    constraints=constraints,
                )
            except self.mink.MinkError as error:
                raise RuntimeError(f"Mink IK failed: {error}") from error

            velocity[self.dof_addresses] = np.clip(
                velocity[self.dof_addresses],
                -IK_MAX_JOINT_STEP / MINK_TIMESTEP,
                IK_MAX_JOINT_STEP / MINK_TIMESTEP,
            )
            if len(self.frozen_dofs):
                velocity[self.frozen_dofs] = 0.0
            configuration.integrate_inplace(
                velocity, MINK_TIMESTEP
            )
            configuration.data.qpos[self.qpos_addresses] = np.clip(
                configuration.data.qpos[self.qpos_addresses],
                local_lower,
                local_upper,
            )
            configuration.update()

            rotation = configuration.data.site_xmat[
                self.site_id
            ].reshape(3, 3)
            position_error = float(
                np.linalg.norm(
                    target
                    - configuration.data.site_xpos[self.site_id]
                )
            )
            orientation_error = float(
                np.linalg.norm(
                    rotation_vector(target_rotation @ rotation.T)
                )
            )
            if (
                position_error < IK_POSITION_TOLERANCE
                and (
                    self.orientation_weight == 0.0
                    or orientation_error < IK_ORIENTATION_TOLERANCE
                )
            ):
                break

        self.data.qpos[:] = configuration.data.qpos
        self.data.qvel[:] = 0
        return self._result(target, target_rotation)


class ProfiledIK:
    """Select Mink or the original IK backend without changing call sites."""

    VALID_BACKENDS = {"auto", "mink", "legacy"}

    def __init__(
        self,
        model: mujoco.MjModel,
        reference: mujoco.MjData,
        profile: ManipulationProfile,
        orientation_weight: float = 0.30,
        seed_continuity_weight: float = 0.0,
        maximum_seed_delta_rad: float | None = None,
        maximum_iterations: int = IK_MAX_ITERATIONS,
        *,
        backend: str | None = None,
    ):
        requested_value = backend or os.environ.get(
            "MUJOCO_IK_BACKEND", "legacy"
        )
        if not isinstance(requested_value, str):
            raise TypeError("IK backend must be a string")
        requested = requested_value.strip().lower()
        if requested not in self.VALID_BACKENDS:
            raise ValueError(
                "MUJOCO_IK_BACKEND must be auto, mink, or legacy"
            )
        solver_options = {
            "orientation_weight": orientation_weight,
            "seed_continuity_weight": seed_continuity_weight,
            "maximum_seed_delta_rad": maximum_seed_delta_rad,
            "maximum_iterations": maximum_iterations,
        }

        if requested in {"auto", "mink"}:
            try:
                self._solver: Any = MinkIK(
                    model, reference, profile, **solver_options
                )
                self.backend_name = "mink"
            except ModuleNotFoundError:
                if requested == "mink":
                    raise
                warnings.warn(
                    "Mink is not installed; using legacy damped "
                    "least-squares IK",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._solver = DampedLeastSquaresIK(
                    model, reference, profile, **solver_options
                )
                self.backend_name = "legacy"
        else:
            self._solver = DampedLeastSquaresIK(
                model, reference, profile, **solver_options
            )
            self.backend_name = "legacy"

        self.data = self._solver.data
        # Motion planners sample deterministic alternative IK seeds through
        # this wrapper. Keep the solver's authoritative padded joint bounds
        # available at the public ProfiledIK boundary.
        self.lower = self._solver.lower
        self.upper = self._solver.upper

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        return self._solver.solve(target, seed, target_rotation)


__all__ = [
    "DampedLeastSquaresIK",
    "MinkIK",
    "ProfiledIK",
]
