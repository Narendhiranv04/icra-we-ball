"""Load fixed baseline domains and their natural-language descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


_DOMAIN_ROOT = Path(__file__).resolve().parent
_DOMAIN_KEYS = ("kitchen", "living_room", "workshop")


@dataclass(frozen=True)
class DomainDefinition:
    key: str
    name: str
    text: str
    sha256: str
    type_hierarchy: Mapping[str, str | None]
    predicate_signatures: Mapping[str, tuple[str, ...]]
    action_signatures: Mapping[str, tuple[str, ...]]
    descriptions: Mapping[str, Any]
    domain_path: Path
    knowledge_path: Path


def available_domains() -> tuple[str, ...]:
    return _DOMAIN_KEYS


def load_domain(key: str) -> DomainDefinition:
    normalized = key.strip().lower().replace("-", "_")
    if normalized not in _DOMAIN_KEYS:
        raise ValueError(
            f"unknown baseline domain {key!r}; expected one of {', '.join(_DOMAIN_KEYS)}"
        )
    root = _DOMAIN_ROOT / normalized
    domain_path = root / "domain.pddl"
    knowledge_path = root / "knowledge.yaml"
    text = domain_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(knowledge_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"domain knowledge must be a mapping: {knowledge_path}")

    name = _required_string(loaded, "domain_name")
    types = _type_hierarchy(loaded.get("types"), knowledge_path)
    predicates = _signatures(loaded.get("predicates"), "predicates", knowledge_path)
    actions = _signatures(loaded.get("actions"), "actions", knowledge_path)
    descriptions = loaded.get("descriptions", {})
    if not isinstance(descriptions, Mapping):
        raise ValueError(f"descriptions must be a mapping: {knowledge_path}")

    return DomainDefinition(
        key=normalized,
        name=name,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        type_hierarchy=MappingProxyType(types),
        predicate_signatures=MappingProxyType(predicates),
        action_signatures=MappingProxyType(actions),
        descriptions=MappingProxyType(dict(descriptions)),
        domain_path=domain_path,
        knowledge_path=knowledge_path,
    )


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip().lower()


def _type_hierarchy(value: Any, path: Path) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"types must be a non-empty mapping: {path}")
    result: dict[str, str | None] = {}
    for name, parent in value.items():
        type_name = str(name).strip().lower()
        if not type_name:
            raise ValueError(f"type names must not be empty: {path}")
        result[type_name] = None if parent is None else str(parent).strip().lower()
    return result


def _signatures(value: Any, label: str, path: Path) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping: {path}")
    result: dict[str, tuple[str, ...]] = {}
    for name, record in value.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"{label}.{name} must be a mapping: {path}")
        parameters = record.get("parameters")
        description = record.get("description")
        if not isinstance(parameters, list) or not all(
            isinstance(item, str) and item.strip() for item in parameters
        ):
            raise ValueError(f"{label}.{name}.parameters must be a string list: {path}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{label}.{name}.description must be non-empty: {path}")
        result[str(name).strip().lower()] = tuple(item.strip().lower() for item in parameters)
    return result
