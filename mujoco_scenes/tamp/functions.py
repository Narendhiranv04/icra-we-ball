"""Functional predicate definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_FUNCTIONS = (
    Path(__file__).resolve().parents[1] / "configs" / "functions.yaml"
)


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    candidate_kind: str
    description: str


class FunctionRegistry:
    def __init__(self, functions: Mapping[str, FunctionSpec]):
        self._functions = dict(functions)

    @classmethod
    def load(cls, path: Path = DEFAULT_FUNCTIONS) -> FunctionRegistry:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        entries = raw.get("functions")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("function registry must define functions")

        functions = {}
        for name, values in entries.items():
            if not isinstance(values, dict):
                raise ValueError(f"invalid function definition: {name}")
            kind = values.get("candidate_kind")
            if kind not in {"object", "region"}:
                raise ValueError(
                    f"{name}.candidate_kind must be object or region"
                )
            functions[name] = FunctionSpec(
                name=name,
                candidate_kind=kind,
                description=str(values.get("description", "")).strip(),
            )
        return cls(functions)

    def get(self, name: str) -> FunctionSpec:
        try:
            return self._functions[name]
        except KeyError as error:
            raise ValueError(f"unknown function: {name}") from error

    def __contains__(self, name: str) -> bool:
        return name in self._functions
