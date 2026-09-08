"""Schema-only adapters for offline V1/V2 FM evaluation.

This module deliberately knows nothing about the production compiler.  It
preserves the distinct V1 interaction-group and V2 operation-pairing shapes
while projecting both into a small, immutable evaluation representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationOperation:
    id: str
    phrase: str
    source_role: str | None
    target_role: str | None
    count: int | None
    reuse_policy: str | None
    anchor_role: str | None = None
    schema: str = ""


@dataclass(frozen=True)
class EvaluationContract:
    schema: str
    roles: tuple[Mapping[str, Any], ...]
    relations: tuple[Mapping[str, Any], ...]
    operations: tuple[EvaluationOperation, ...]


def _mapping_items(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _v1_operation(raw: Mapping[str, Any], index: int) -> EvaluationOperation:
    return EvaluationOperation(
        id=str(raw.get("id", f"v1_operation_{index}")),
        phrase=str(raw.get("function", "")),
        source_role=raw.get("tool_role"),
        target_role=raw.get("target_role"),
        count=raw.get("required_target_count"),
        reuse_policy=raw.get("usage_policy"),
        anchor_role=raw.get("context_role"),
        schema="V1_INTERACTION_GROUP",
    )


def _v2_operation(raw: Mapping[str, Any], index: int) -> EvaluationOperation:
    return EvaluationOperation(
        id=str(raw.get("id", f"v2_operation_{index}")),
        phrase=str(raw.get("operation", "")),
        source_role=raw.get("source_role"),
        target_role=raw.get("target_role"),
        count=raw.get("operation_count"),
        reuse_policy=raw.get("reuse_policy"),
        anchor_role=raw.get("anchor_role"),
        schema="V2_OPERATION_PAIRING",
    )


def extract_evaluation_contract(raw: Any) -> EvaluationContract:
    """Extract a raw contract without compiling, sanitizing, or flattening it."""
    document = raw if isinstance(raw, Mapping) else {}
    if "task_contract" in document:
        contract = document.get("task_contract")
        contract = contract if isinstance(contract, Mapping) else {}
        operations = tuple(
            _v2_operation(item, index)
            for index, item in enumerate(_mapping_items(contract.get("operation_pairings")))
        )
        return EvaluationContract(
            schema="V2",
            roles=_mapping_items(contract.get("functional_roles")),
            relations=_mapping_items(contract.get("functional_relations")),
            operations=operations,
        )

    operations = tuple(
        _v1_operation(item, index)
        for index, item in enumerate(_mapping_items(document.get("interaction_groups")))
    )
    return EvaluationContract(
        schema="V1",
        roles=_mapping_items(document.get("functional_roles")),
        relations=_mapping_items(document.get("functional_relations")),
        operations=operations,
    )
