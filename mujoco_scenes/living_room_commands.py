"""Parse living-room grounded action files."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path


COMMAND_HELP = (
    "move DESTINATION | inspect sofa | open SIDE | close SIDE | pick OBJECT | "
    "place [STORAGE_TARGET] | task store_game_controller | "
    "state [observed|ground_truth] | help"
)
DEFAULT_ACTION_FILE = (
    Path(__file__).resolve().parent / "configs" / "living_room_actions.txt"
)


@dataclass(frozen=True)
class LivingRoomCommand:
    verb: str
    arguments: tuple[str, ...] = ()


def parse_living_room_command(text: str) -> LivingRoomCommand:
    try:
        words = shlex.split(text, comments=True)
    except ValueError as error:
        raise ValueError(f"Invalid command quoting: {error}") from error
    if not words:
        raise ValueError("Type a command")

    verb = words[0].lower().replace("-", "_")
    arguments = tuple(
        word.lower().replace("-", "_") for word in words[1:]
    )
    if verb == "gt":
        verb, arguments = "state", ("ground_truth",)

    arities = {
        "move": {1},
        "inspect": {1},
        "open": {1},
        "close": {1},
        "pick": {1},
        "place": {0, 1},
        "task": {1},
        "state": {0, 1},
        "help": {0},
    }
    if verb not in arities:
        raise ValueError(f"Unknown command '{verb}'. {COMMAND_HELP}")
    if len(arguments) not in arities[verb]:
        raise ValueError(f"Usage: {COMMAND_HELP}")
    return LivingRoomCommand(verb, arguments)


def load_living_room_commands(
    path: str | Path,
) -> tuple[LivingRoomCommand, ...]:
    """Load one grounded command per non-empty line."""
    action_path = Path(path)
    commands: list[LivingRoomCommand] = []
    try:
        lines = action_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(
            f"Could not read action file '{action_path}': {error}"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            commands.append(parse_living_room_command(line))
        except ValueError as error:
            raise ValueError(
                f"{action_path}:{line_number}: {error}"
            ) from error

    if not commands:
        raise ValueError(f"No actions found in '{action_path}'")
    return tuple(commands)
