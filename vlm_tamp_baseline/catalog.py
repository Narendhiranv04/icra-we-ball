"""Load the formal subgoal vocabulary visible to VLM-TAMP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CATALOG_PATH = Path(__file__).with_name("subgoal_catalog.json")


def load_catalog(path: str | Path = CATALOG_PATH) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported VLM-TAMP subgoal catalogue schema")
    scenes = document.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("Subgoal catalogue must define scenes")
    for scene, predicates in scenes.items():
        if not isinstance(scene, str) or not scene or not isinstance(predicates, dict):
            raise ValueError("Invalid scene subgoal catalogue")
        for predicate, definition in predicates.items():
            if not isinstance(predicate, str) or predicate != predicate.upper():
                raise ValueError(f"Invalid subgoal predicate {predicate!r}")
            if not isinstance(definition, dict) or set(definition) != {
                "description",
                "arguments",
            }:
                raise ValueError(f"Invalid definition for {predicate}")
            if (
                not isinstance(definition["description"], str)
                or not definition["description"].strip()
            ):
                raise ValueError(f"Subgoal {predicate} needs a description")
            arguments = definition["arguments"]
            if (
                not isinstance(arguments, dict)
                or any(not isinstance(name, str) or not name for name in arguments)
                or any(
                    kind not in {"object", "region", "destination"}
                    for kind in arguments.values()
                )
            ):
                raise ValueError(f"Invalid arguments for {predicate}")
    return document


def scene_subgoals(
    catalog: Mapping[str, Any], scene: str
) -> Mapping[str, Mapping[str, Any]]:
    scenes = catalog.get("scenes", {})
    if scene not in scenes:
        choices = ", ".join(sorted(scenes))
        raise ValueError(f"Unknown scene {scene!r}; choose from {choices}")
    return scenes[scene]
