import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mujoco_scenes.living_room_commands import (
    LivingRoomCommand,
    load_living_room_commands,
    parse_living_room_command,
)


class LivingRoomCommandTests(unittest.TestCase):
    def test_parses_grounded_physical_actions(self):
        cases = {
            "move drawer-left": LivingRoomCommand(
                "move", ("drawer_left",)
            ),
            "inspect sofa": LivingRoomCommand("inspect", ("sofa",)),
            "open LEFT": LivingRoomCommand("open", ("left",)),
            "pick game_controller": LivingRoomCommand(
                "pick", ("game_controller",)
            ),
            "place media_shelf": LivingRoomCommand(
                "place", ("media_shelf",)
            ),
            "place": LivingRoomCommand("place"),
            "task store_game_controller": LivingRoomCommand(
                "task", ("store_game_controller",)
            ),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    parse_living_room_command(text), expected
                )

    def test_ground_truth_alias_is_explicit_state_query(self):
        self.assertEqual(
            parse_living_room_command("gt"),
            LivingRoomCommand("state", ("ground_truth",)),
        )

    def test_rejects_unknown_commands_and_wrong_arity(self):
        for text in ("", "dance", "move", "open left now", "help me"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_living_room_command(text)

    def test_loads_commands_and_ignores_comments(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "actions.txt"
            path.write_text(
                "# Test sequence\n"
                "\n"
                "move drawer-left  # normalize the destination\n"
                "open LEFT\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_living_room_commands(path),
                (
                    LivingRoomCommand("move", ("drawer_left",)),
                    LivingRoomCommand("open", ("left",)),
                ),
            )

    def test_loader_reports_the_invalid_line(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "actions.txt"
            path.write_text("move home\ndance\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, rf"{path}:2: Unknown command"
            ):
                load_living_room_commands(path)

    def test_loader_rejects_an_empty_action_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "actions.txt"
            path.write_text("# no actions\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No actions found"):
                load_living_room_commands(path)


if __name__ == "__main__":
    unittest.main()
