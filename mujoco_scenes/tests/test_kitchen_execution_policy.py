import unittest

from mujoco_scenes.kitchen_execution_policy import (
    KitchenExecutionPolicy,
    KitchenWorkspace,
    required_workspace,
)


class KitchenExecutionPolicyTests(unittest.TestCase):
    def test_container_workspace_map(self):
        self.assertEqual(required_workspace("OPEN", "D1"), KitchenWorkspace.HOME)
        self.assertEqual(required_workspace("CLOSE", "C1"), KitchenWorkspace.LEFT_SIDE)
        self.assertEqual(required_workspace("OPEN", "B1"), KitchenWorkspace.RIGHT_SIDE)

    def test_move_is_inserted_only_when_workspace_differs(self):
        policy = KitchenExecutionPolicy()
        inserted = policy.refine("OPEN", "C1", KitchenWorkspace.HOME)
        self.assertEqual(inserted.refined_actions, (("MOVE", "left_side"), ("OPEN", "C1")))
        self.assertTrue(inserted.auto_move_inserted)
        omitted = policy.refine("CLOSE", "C1", KitchenWorkspace.LEFT_SIDE)
        self.assertEqual(omitted.refined_actions, (("CLOSE", "C1"),))
        self.assertFalse(omitted.auto_move_inserted)

    def test_unknown_container_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown kitchen container"):
            required_workspace("OPEN", "C9")


if __name__ == "__main__":
    unittest.main()
