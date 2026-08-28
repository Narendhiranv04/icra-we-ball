"""Discrete skills and LLM3 continuous-parameter bounds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from baseline_common.catalog import CATALOG_PATH, load_catalog, scene_actions

PARAMETER_CATALOG_PATH = Path(__file__).with_name("continuous_parameters.json")


def load_parameter_catalog(
    path: str | Path = PARAMETER_CATALOG_PATH,
) -> Mapping[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported LLM3 parameter catalogue schema")
    scenes = document.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("LLM3 parameter catalogue must define scenes")
    for scene, skills in scenes.items():
        if not isinstance(scene, str) or not scene or not isinstance(skills, dict):
            raise ValueError("Invalid LLM3 parameter scene")
        for skill, parameters in skills.items():
            if not isinstance(skill, str) or skill != skill.upper():
                raise ValueError(f"Invalid LLM3 parameter skill {skill!r}")
            if not isinstance(parameters, dict):
                raise ValueError(f"Parameters for {scene}.{skill} must be an object")
            for name, bounds in parameters.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(bounds, dict)
                    or set(bounds) != {"minimum", "maximum"}
                ):
                    raise ValueError(f"Invalid bounds for {scene}.{skill}.{name}")
                lower, upper = bounds["minimum"], bounds["maximum"]
                if (
                    isinstance(lower, bool)
                    or isinstance(upper, bool)
                    or not isinstance(lower, (int, float))
                    or not isinstance(upper, (int, float))
                    or not math.isfinite(float(lower))
                    or not math.isfinite(float(upper))
                    or float(lower) > float(upper)
                ):
                    raise ValueError(f"Invalid bounds for {scene}.{skill}.{name}")
    return document


def scene_parameters(
    catalog: Mapping[str, Any], scene: str
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    try:
        return catalog["scenes"][scene]
    except KeyError as error:
        raise ValueError(f"No LLM3 continuous parameters for scene {scene!r}") from error

__all__ = [
    "CATALOG_PATH",
    "PARAMETER_CATALOG_PATH",
    "load_catalog",
    "load_parameter_catalog",
    "scene_actions",
    "scene_parameters",
]
