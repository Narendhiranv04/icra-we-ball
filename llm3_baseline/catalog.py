"""Load and validate scene action catalogues."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CATALOG_PATH = Path(__file__).with_name("action_catalog.json")
REFERENCE_KINDS = {"object", "region"}


def load_catalog(path: str | Path = CATALOG_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported action catalogue schema")
    scenes = document.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("Action catalogue must define scenes")
    for scene, actions in scenes.items():
        if not isinstance(scene, str) or not scene or not isinstance(actions, dict):
            raise ValueError("Invalid scene action catalogue")
        for name, definition in actions.items():
            _validate_action_definition(name, definition)
    return document


def scene_actions(
    catalog: Mapping[str, Any], scene: str
) -> Mapping[str, Mapping[str, Any]]:
    scenes = catalog.get("scenes", {})
    if scene not in scenes:
        choices = ", ".join(sorted(scenes))
        raise ValueError(f"Unknown scene {scene!r}; choose from {choices}")
    return scenes[scene]


def _validate_action_definition(name: object, definition: object) -> None:
    if not isinstance(name, str) or not name or name != name.upper():
        raise ValueError(f"Invalid action name {name!r}")
    if not isinstance(definition, dict):
        raise ValueError(f"Invalid definition for {name}")
    if set(definition) != {"description", "arguments"}:
        raise ValueError(f"Action {name} has unsupported fields")
    if not isinstance(definition["description"], str):
        raise ValueError(f"Action {name} needs a description")
    arguments = definition["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError(f"Action {name} arguments must be an object")
    for argument, kind in arguments.items():
        if not isinstance(argument, str) or kind not in REFERENCE_KINDS:
            raise ValueError(f"Invalid argument {argument!r} for {name}")
