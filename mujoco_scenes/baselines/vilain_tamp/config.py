"""Small validated configuration model for the ViLaIn-TAMP baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml


class Domain(str, Enum):
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    WORKSHOP = "workshop"


class ObservationMode(str, Enum):
    INITIAL_ONLY = "initial_observation_only"
    FIXED_FULL_INSPECTION = "fixed_full_inspection"


class ModelCondition(str, Enum):
    PAPER_FAITHFUL = "paper_faithful"
    MODEL_MATCHED = "model_matched"


@dataclass(frozen=True)
class TimeoutConfig:
    symbolic_seconds: float
    model_seconds: float
    refinement_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("symbolic_seconds", self.symbolic_seconds),
            ("model_seconds", self.model_seconds),
            ("refinement_seconds", self.refinement_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class ExternalToolPaths:
    fast_downward: Path | None
    val: Path | None

    def __post_init__(self) -> None:
        for name, value in (("fast_downward", self.fast_downward), ("val", self.val)):
            if value is not None and not value.is_absolute():
                raise ValueError(f"external_tools.{name} must be an absolute path")


@dataclass(frozen=True)
class BaselineConfig:
    domain: Domain
    observation_mode: ObservationMode
    model_condition: ModelCondition
    max_cp_corrections: int
    timeouts: TimeoutConfig
    output_root: Path
    external_tools: ExternalToolPaths
    object_estimator_model: str
    reasoning_model: str
    symbolic_planner: str
    search_configuration: str
    independent_model_calls: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.max_cp_corrections <= 3:
            raise ValueError("max_cp_corrections must be between 0 and 3")
        if not str(self.output_root):
            raise ValueError("output_root must not be empty")
        for name, value in (
            ("object_estimator_model", self.object_estimator_model),
            ("reasoning_model", self.reasoning_model),
            ("symbolic_planner", self.symbolic_planner),
            ("search_configuration", self.search_configuration),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not self.independent_model_calls:
            raise ValueError("baseline model calls must be independent")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BaselineConfig:
        timeouts = _mapping(data, "timeouts")
        tools = _mapping(data, "external_tools")
        output_root = str(data["output_root"]).strip()
        if not output_root:
            raise ValueError("output_root must not be empty")
        return cls(
            domain=Domain(str(data["domain"])),
            observation_mode=ObservationMode(str(data["observation_mode"])),
            model_condition=ModelCondition(str(data["model_condition"])),
            max_cp_corrections=int(data["max_cp_corrections"]),
            timeouts=TimeoutConfig(
                symbolic_seconds=float(timeouts["symbolic_seconds"]),
                model_seconds=float(timeouts["model_seconds"]),
                refinement_seconds=float(timeouts["refinement_seconds"]),
            ),
            output_root=Path(output_root),
            external_tools=ExternalToolPaths(
                fast_downward=_optional_path(tools.get("fast_downward")),
                val=_optional_path(tools.get("val")),
            ),
            object_estimator_model=str(data["object_estimator_model"]),
            reasoning_model=str(data["reasoning_model"]),
            symbolic_planner=str(data["symbolic_planner"]),
            search_configuration=str(data["search_configuration"]),
            independent_model_calls=bool(data.get("independent_model_calls", True)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaselineConfig:
        source = Path(path)
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"configuration must be a mapping: {source}")
        return cls.from_mapping(loaded)


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))
