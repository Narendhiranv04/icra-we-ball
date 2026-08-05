import unittest
from types import SimpleNamespace

from mujoco_scenes.foundation_model import Candidate
from mujoco_scenes.living_room_navigation import LivingRoomLayout
from mujoco_scenes.living_room_tamp import (
    STORE_CONTROLLER_TASK,
    LivingRoomObserver,
    LivingRoomStoragePlanner,
    storage_goal_satisfied,
)
from mujoco_scenes.tamp.state import (
    ObjectObservation,
    ObservedState,
    RegionObservation,
    RobotObservation,
)


class _Manipulation:
    def __init__(self):
        self.object_locations = {
            "remote_control": "home",
            "living_room_mug": "home",
            "hardback_book": "home",
            "game_controller": "home",
            "rigid_duster": "duster",
        }
        self.held_object = None
        self.navigation_safe = True
        self.scene = SimpleNamespace(lost_remote_detected=False)


class LivingRoomTampTests(unittest.TestCase):
    def setUp(self):
        self.navigation = SimpleNamespace(current_location="home")
        self.manipulation = _Manipulation()
        self.left = SimpleNamespace(
            is_open=False, navigation_safe=True
        )
        self.right = SimpleNamespace(
            is_open=False, navigation_safe=True
        )
        self.observer = LivingRoomObserver(
            self.navigation,
            self.manipulation,
            self.left,
            self.right,
        )

    def test_closed_drawer_contents_are_not_initially_exposed(self):
        self.manipulation.object_locations[
            "remote_control"
        ] = "media_console_left_drawer"

        state = self.observer()

        region = state.regions["media_console_left_drawer"]
        self.assertFalse(region.inspected)
        self.assertIsNone(region.occupied_by)
        self.assertFalse(state.objects["remote_control"].visible)
        self.assertFalse(
            any(
                relation.subject == "remote_control"
                for relation in state.relations
            )
        )

    def test_opening_drawer_reveals_its_contents(self):
        self.manipulation.object_locations[
            "remote_control"
        ] = "media_console_left_drawer"
        self.left.is_open = True

        state = self.observer()

        region = state.regions["media_console_left_drawer"]
        self.assertTrue(region.inspected)
        self.assertEqual(region.occupied_by, ("remote_control",))
        self.assertTrue(state.objects["remote_control"].visible)

        self.left.is_open = False
        remembered = self.observer()
        self.assertTrue(
            remembered.regions[
                "media_console_left_drawer"
            ].inspected
        )
        self.assertEqual(
            remembered.regions[
                "media_console_left_drawer"
            ].occupied_by,
            ("remote_control",),
        )
        self.assertFalse(
            remembered.objects["remote_control"].visible
        )

    def test_under_sofa_object_requires_positive_visual_observation(self):
        self.manipulation.object_locations["remote_control"] = "under_sofa"
        self.assertFalse(self.observer().objects["remote_control"].visible)

        self.manipulation.scene.lost_remote_detected = True
        observed = self.observer()
        self.assertTrue(observed.objects["remote_control"].visible)
        self.assertEqual(
            observed.objects["remote_control"].location, "under_sofa"
        )

    def test_storage_plan_moves_picks_opens_places_and_closes(self):
        state = self.observer()
        candidate = Candidate(
            "media_console_left_drawer", "drawer"
        )

        actions = LivingRoomStoragePlanner()(
            STORE_CONTROLLER_TASK, candidate, state
        )

        self.assertEqual(
            [action.name for action in actions],
            ["pick", "move", "open", "place", "close"],
        )
        self.assertEqual(
            actions[1].arguments["destination"], "drawer_left"
        )
        self.assertEqual(
            actions[3].arguments["place_site"],
            "left_drawer_place_controller",
        )

    def test_storage_goal_requires_observed_closed_storage(self):
        candidate = Candidate(
            "media_console_right_drawer", "drawer"
        )
        state = ObservedState(
            {
                "game_controller": ObjectObservation(
                    "game_controller",
                    "game_controller",
                    False,
                    "media_console_right_drawer",
                )
            },
            {
                "media_console_right_drawer": RegionObservation(
                    "media_console_right_drawer",
                    "drawer",
                    True,
                    inspected=True,
                    open=False,
                    occupied_by=("game_controller",),
                )
            },
            RobotObservation("drawer_right"),
        )

        self.assertTrue(
            storage_goal_satisfied(
                STORE_CONTROLLER_TASK, candidate, state
            )
        )

    def test_left_and_right_drawer_navigation_stances_are_distinct(self):
        layout = LivingRoomLayout()
        scene = SimpleNamespace(table_pose=(0.0, -0.35, 0.0))

        left = layout.destination_pose(scene, "drawer_left")
        right = layout.destination_pose(scene, "drawer_right")

        self.assertLess(left.x, right.x)
        self.assertEqual(left.y, right.y)


if __name__ == "__main__":
    unittest.main()
