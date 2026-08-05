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


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
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
    ):
        self.model = model
        self.profile = profile
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

    def _initial_qpos(self, seed: np.ndarray) -> np.ndarray:
        qpos = self.reference_qpos.copy()
        qpos[self.qpos_addresses] = np.clip(
            seed, self.lower, self.upper
        )
        return qpos

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
                    _rotation_vector(target_rotation @ rotation.T)
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
        self.data.qpos[:] = self._initial_qpos(seed)
        self.data.qvel[:] = 0
        for _ in range(IK_MAX_ITERATIONS):
            mujoco.mj_forward(self.model, self.data)
            rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            position_error = target - self.data.site_xpos[self.site_id]
            rotation_error = _rotation_vector(
                target_rotation @ rotation.T
            )
            error = np.concatenate(
                (position_error, 0.30 * rotation_error)
            )
            if (
                np.linalg.norm(position_error) < IK_POSITION_TOLERANCE
                and np.linalg.norm(rotation_error)
                < IK_ORIENTATION_TOLERANCE
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
                    0.30 * jac_rot[:, self.dof_addresses],
                )
            )
            damping = 0.0025
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6),
                error,
            )
            current = self.data.qpos[self.qpos_addresses]
            self.data.qpos[self.qpos_addresses] = np.clip(
                current
                + np.clip(
                    delta,
                    -IK_MAX_JOINT_STEP,
                    IK_MAX_JOINT_STEP,
                ),
                self.lower,
                self.upper,
            )

        return self._result(target, target_rotation)


class MinkIK(_ProfileIKBase):
    """Mink differential IK with frozen non-arm DOFs and joint limits."""

    def __init__(
        self,
        model: mujoco.MjModel,
        reference: mujoco.MjData,
        profile: ManipulationProfile,
    ):
        super().__init__(model, reference, profile)
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
        qpos = self._initial_qpos(seed)
        configuration = self.mink.Configuration(self.model, q=qpos)

        frame_task = self.mink.FrameTask(
            frame_name=self.profile.grip_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.30,
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
        posture_cost[self.dof_addresses] = 1e-3
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

        for _ in range(IK_MAX_ITERATIONS):
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
                self.lower,
                self.upper,
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
                    _rotation_vector(target_rotation @ rotation.T)
                )
            )
            if (
                position_error < IK_POSITION_TOLERANCE
                and orientation_error < IK_ORIENTATION_TOLERANCE
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
        *,
        backend: str | None = None,
    ):
        requested = (
            backend
            or os.environ.get("MUJOCO_IK_BACKEND", "auto")
        ).strip().lower()
        if requested not in self.VALID_BACKENDS:
            raise ValueError(
                "MUJOCO_IK_BACKEND must be auto, mink, or legacy"
            )

        if requested in {"auto", "mink"}:
            try:
                self._solver: Any = MinkIK(
                    model, reference, profile
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
                    model, reference, profile
                )
                self.backend_name = "legacy"
        else:
            self._solver = DampedLeastSquaresIK(
                model, reference, profile
            )
            self.backend_name = "legacy"

        self.data = self._solver.data

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
