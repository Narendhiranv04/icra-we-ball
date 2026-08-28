"""Load shared scene action contracts without embedding a planning policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CATALOG_PATH = Path(__file__).with_name("action_catalog.json")


def load_catalog(path: str | Path = CATALOG_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported action catalogue schema")
    scenes = document.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("Action catalogue must define scenes")
    for scene, actions in scenes.items():
        if not isinstance(scene, str) or not scene or not isinstance(actions, dict):
            raise ValueError("Invalid scene action catalogue")
        for name, definition in actions.items():
            if not isinstance(name, str) or name != name.upper():
                raise ValueError(f"Invalid action name {name!r}")
            if not isinstance(definition, dict) or set(definition) != {
                "description",
                "arguments",
            }:
                raise ValueError(f"Invalid definition for {name}")
            description = definition["description"]
            arguments = definition["arguments"]
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Invalid description for {name}")
            if not isinstance(arguments, dict) or any(
                not isinstance(argument, str) or not argument
                for argument in arguments
            ):
                raise ValueError(f"Invalid arguments for {name}")
            if any(
                kind not in {"object", "region", "destination"}
                for kind in arguments.values()
            ):
                raise ValueError(f"Invalid arguments for {name}")
    return document


def scene_actions(
    catalog: Mapping[str, Any], scene: str
) -> Mapping[str, Mapping[str, Any]]:
    scenes = catalog.get("scenes", {})
    if scene not in scenes:
        choices = ", ".join(sorted(scenes))
        raise ValueError(f"Unknown scene {scene!r}; choose from {choices}")
    return scenes[scene]
