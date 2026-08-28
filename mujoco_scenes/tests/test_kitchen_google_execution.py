import unittest
from unittest.mock import Mock

from mujoco_scenes.kitchen_google_execution import KitchenGoogleExecutionDispatcher
from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace


class KitchenGoogleExecutionDispatcherTests(unittest.TestCase):
    def _dispatcher_without_model(self):
        dispatcher = object.__new__(KitchenGoogleExecutionDispatcher)
        dispatcher.scene = Mock()
        dispatcher.scene.state.container_open_state = {}
        dispatcher.navigation = Mock()
        dispatcher.navigation.current_physical_location = "home"
        dispatcher.articulation = Mock()
        dispatcher.policy = __import__(
            "mujoco_scenes.kitchen_execution_policy", fromlist=["KitchenExecutionPolicy"]
        ).KitchenExecutionPolicy()
        return dispatcher

    def test_plan_only_refines_without_moving_or_articulating(self):
        dispatcher = self._dispatcher_without_model()
        record = dispatcher.request("OPEN", "C2", execute=False)
        self.assertTrue(record["success"])
        self.assertEqual(record["status"], "PLAN_ONLY")
        self.assertTrue(record["refinement"]["auto_move_inserted"])
        dispatcher.articulation.execute.assert_not_called()

    def test_same_workspace_omits_move(self):
        dispatcher = self._dispatcher_without_model()
        dispatcher.navigation.current_physical_location = "cupboard1"
        result = Mock(success=False, status="EXECUTION_FAILED")
        result.to_dict.return_value = {"success": False}
        dispatcher.articulation.execute.return_value = result
        dispatcher._move = Mock()
        record = dispatcher.request("OPEN", "C1", execute=True)
        dispatcher._move.assert_not_called()
        dispatcher.articulation.execute.assert_called_once_with(
            "OPEN", "C1", KitchenWorkspace.LEFT_SIDE,
            target_q_override=None,
        )
        self.assertFalse(record["success"])


if __name__ == "__main__":
    unittest.main()
