import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from mujoco_scenes.ik import ProfiledIK, rotation_vector


TEST_MODEL = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <joint name="base_x" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size=".02" mass="1"/>
      <body name="arm">
        <joint name="x" type="slide" axis="1 0 0" range="-.5 .5"/>
        <joint name="y" type="slide" axis="0 1 0" range="-.5 .5"/>
        <joint name="z" type="slide" axis="0 0 1" range="-.5 .5"/>
        <joint name="rx" type="hinge" axis="1 0 0" range="-180 180"/>
        <joint name="ry" type="hinge" axis="0 1 0" range="-180 180"/>
        <joint name="rz" type="hinge" axis="0 0 1" range="-180 180"/>
        <geom type="sphere" size=".02" mass="1"/>
        <site name="grip"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class ProfiledIKTests(unittest.TestCase):
    def test_rotation_vector_rejects_malformed_rotation(self):
        for matrix in (np.eye(2), np.full((3, 3), np.nan)):
            with self.subTest(shape=matrix.shape):
                with self.assertRaises(ValueError):
                    rotation_vector(matrix)

    def test_solver_rejects_malformed_pose_and_seed(self):
        ik = ProfiledIK(
            self.model, self.data, self.profile, backend="legacy"
        )
        with self.assertRaisesRegex(ValueError, "target position"):
            ik.solve(np.zeros(2), np.zeros(6), np.eye(3))
        with self.assertRaisesRegex(ValueError, "seed"):
            ik.solve(np.zeros(3), np.zeros(5), np.eye(3))
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            ik.solve(np.zeros(3), np.zeros(6), np.ones((3, 3)))

    def test_zero_orientation_weight_does_not_wait_for_rotation(self):
        ik = ProfiledIK(
            self.model,
            self.data,
            self.profile,
            orientation_weight=0.0,
            backend="legacy",
        )
        angle = 1.0
        unreachable_orientation = np.array(
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        with patch("mujoco_scenes.ik.mujoco.mj_jacSite") as jacobian:
            _joints, position_error, _orientation_error = ik.solve(
                np.zeros(3), np.zeros(6), unreachable_orientation
            )
        self.assertLess(position_error, 1e-12)
        jacobian.assert_not_called()

    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(TEST_MODEL)
        self.data = mujoco.MjData(self.model)
        self.profile = SimpleNamespace(
            arm_joints=("x", "y", "z", "rx", "ry", "rz"),
            grip_site="grip",
        )
        angle = 0.2
        self.target_rotation = np.array(
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        self.target = np.array((0.10, -0.08, 0.06))

    def test_mink_and_legacy_backends_reach_the_pose(self):
        for backend in ("mink", "legacy"):
            with self.subTest(backend=backend):
                ik = ProfiledIK(
                    self.model,
                    self.data,
                    self.profile,
                    backend=backend,
                )
                _, position_error, orientation_error = ik.solve(
                    self.target,
                    np.zeros(6),
                    self.target_rotation,
                )
                self.assertLess(position_error, 0.0008)
                self.assertLess(
                    orientation_error, math.radians(0.7)
                )

    def test_mink_freezes_non_arm_degrees_of_freedom(self):
        ik = ProfiledIK(
            self.model,
            self.data,
            self.profile,
            backend="mink",
        )
        ik.solve(
            self.target,
            np.zeros(6),
            self.target_rotation,
        )
        base_joint = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "base_x",
        )
        base_qpos = self.model.jnt_qposadr[base_joint]
        self.assertEqual(float(ik.data.qpos[base_qpos]), 0.0)

    def test_environment_selects_the_backend(self):
        with patch.dict(
            os.environ, {"MUJOCO_IK_BACKEND": "legacy"}
        ):
            ik = ProfiledIK(
                self.model, self.data, self.profile
            )
        self.assertEqual(ik.backend_name, "legacy")

    def test_calibrated_default_uses_legacy_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            ik = ProfiledIK(self.model, self.data, self.profile)
        self.assertEqual(ik.backend_name, "legacy")

    def test_rejects_unknown_backend(self):
        with self.assertRaisesRegex(
            ValueError, "auto, mink, or legacy"
        ):
            ProfiledIK(
                self.model,
                self.data,
                self.profile,
                backend="unknown",
            )

    def test_calibrated_solver_controls_are_supported_by_both_backends(self):
        for backend in ("mink", "legacy"):
            with self.subTest(backend=backend):
                ik = ProfiledIK(
                    self.model,
                    self.data,
                    self.profile,
                    orientation_weight=0.0,
                    seed_continuity_weight=0.01,
                    maximum_seed_delta_rad=0.25,
                    maximum_iterations=20,
                    backend=backend,
                )
                joints, _, _ = ik.solve(
                    np.array((0.02, 0.0, 0.0)),
                    np.zeros(6),
                    np.eye(3),
                )
                self.assertTrue(np.all(np.abs(joints) <= 0.25 + 1e-9))

    def test_generic_manipulation_reexports_the_single_ik_implementation(self):
        from mujoco_scenes.generic_manipulation import (
            ProfiledIK as GenericProfiledIK,
        )

        self.assertIs(GenericProfiledIK, ProfiledIK)


if __name__ == "__main__":
    unittest.main()
