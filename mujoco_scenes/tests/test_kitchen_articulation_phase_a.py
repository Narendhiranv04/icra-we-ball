import unittest

import mujoco

from mujoco_scenes.kitchen_articulation import (
    ArticulationExecutionError,
    ArticulationFailureCode,
    GoogleKitchenArticulationExecutor,
)
from mujoco_scenes.kitchen_articulation import ARTICULATION_SPECS
from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace
from mujoco_scenes.scene_loader import KitchenScene


class KitchenArticulationSpecificationTests(unittest.TestCase):
    def test_all_storage_mechanisms_have_physical_specs(self):
        self.assertEqual(set(ARTICULATION_SPECS), {"D1", "D2", "C1", "C2", "B1"})
        for container, spec in ARTICULATION_SPECS.items():
            with self.subTest(container=container):
                self.assertTrue(spec.handle_site)
                self.assertTrue(spec.handle_geoms)
                self.assertTrue(spec.attachment_name.startswith("google:container_grasp_"))
                self.assertGreater(spec.open_q, spec.closed_q)

    def test_workspace_assignments_preserve_existing_calibration(self):
        self.assertEqual(ARTICULATION_SPECS["D1"].required_workspace, KitchenWorkspace.HOME)
        self.assertEqual(ARTICULATION_SPECS["D2"].required_workspace, KitchenWorkspace.HOME)
        self.assertEqual(ARTICULATION_SPECS["C1"].required_workspace, KitchenWorkspace.LEFT_SIDE)
        self.assertEqual(ARTICULATION_SPECS["C2"].required_workspace, KitchenWorkspace.RIGHT_SIDE)
        self.assertEqual(ARTICULATION_SPECS["B1"].required_workspace, KitchenWorkspace.RIGHT_SIDE)


class KitchenArticulationRuntimeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = KitchenScene(
            "S1_integrated_kitchen_object_function_primary", robot="google"
        )

    def test_handle_attachments_are_inactive_at_reset(self):
        for container in ARTICULATION_SPECS:
            equality = mujoco.mj_name2id(
                self.scene.model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                f"google:container_grasp_{container}",
            )
            self.assertGreaterEqual(equality, 0)
            self.assertEqual(int(self.scene.data.eq_active[equality]), 0)

    def test_low_level_primitive_rejects_wrong_workspace(self):
        executor = GoogleKitchenArticulationExecutor(self.scene)
        with self.assertRaises(ArticulationExecutionError) as raised:
            executor.plan("OPEN", "C1", KitchenWorkspace.HOME)
        self.assertEqual(
            raised.exception.code,
            ArticulationFailureCode.WORKSPACE_PRECONDITION_UNSATISFIED,
        )

    def test_low_level_primitive_rejects_held_payload(self):
        executor = GoogleKitchenArticulationExecutor(
            self.scene, held_object_getter=lambda: "object_0001"
        )
        with self.assertRaises(ArticulationExecutionError) as raised:
            executor.plan("OPEN", "D1", KitchenWorkspace.HOME)
        self.assertEqual(
            raised.exception.code,
            ArticulationFailureCode.HAND_NOT_EMPTY_FOR_ARTICULATION,
        )

    def test_direct_perception_open_api_remains_separate(self):
        scene = KitchenScene("S1_coffee_missing_mug", include_robot=False)
        scene.open_container("D1", steps=1200)
        self.assertIn("D1", scene.state.opened_containers)
        scene.close_container("D1", steps=1200)
        self.assertFalse(scene.state.container_open_state["D1"])


if __name__ == "__main__":
    unittest.main()
