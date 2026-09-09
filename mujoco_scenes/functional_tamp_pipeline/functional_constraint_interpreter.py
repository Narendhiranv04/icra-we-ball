"""Bounded whole-graph interpretation for V3 FM operations.

The FM contract is semantic evidence rather than an executable call signature.
This module performs only finite, provenance-carrying rewrites supported by the
FM's own roles and relations.  It does not inspect observations, variants, or
benchmark reference data.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Callable, Mapping, Sequence

from .robot_capability_registry import (
    extract_operation_semantic_candidates,
    is_non_physical_operation_phrase,
)


SlotResolver = Callable[[str, Mapping[str, Any], Mapping[str, Mapping[str, Any]], Mapping[str, Any]], list[dict[str, Any]]]


class FunctionalConstraintInterpreter:
    """Resolve operation meaning against the complete explicit FM graph."""

    _MOVE = re.compile(r"\b(move|transfer|relocate|place|position|transport)\b", re.I)
    _CURRENT_LOCATION = re.compile(
        r"\b(initial|current(?:ly)?|starting|staging|storage|source location|holds? the (?:items?|payload))\b",
        re.I,
    )
    _BEVERAGE_MACRO = re.compile(
        r"\b(?:prepare|make|mix|combine)\b.*\b(?:beverage|coffee|drink|mixture)\b|\bfill\b.*\bmix\b",
        re.I,
    )

    def __init__(
        self,
        *,
        domain: str,
        roles_by_id: Mapping[str, Mapping[str, Any]],
        hypotheses: Mapping[str, Any],
        relations: Sequence[Mapping[str, Any]],
        slot_resolver: SlotResolver,
    ) -> None:
        self.domain = domain
        self.roles = roles_by_id
        self.hypotheses = hypotheses
        self.relations = relations
        self.resolve_slots = slot_resolver
        self.trace: list[dict[str, Any]] = []
        self.accounting: list[dict[str, Any]] = []

    def _types(self, raw_role: str) -> set[str]:
        hypothesis = self.hypotheses.get(raw_role)
        return set(getattr(hypothesis, "canonical_role_candidates", ()))

    def _is_current_location(self, raw_role: str) -> bool:
        role = self.roles[raw_role]
        text = " ".join(
            [str(role.get("function", "")), str(role.get("description", "")),
             " ".join(role.get("candidate_categories", ()))]
        ).replace("_", " ")
        return bool(self._CURRENT_LOCATION.search(text)) and role.get("entity_kind") in {"REGION", "FIXED_TARGET", "OBJECT"}

    def _explicit_anchor(
        self, support: str, capability_id: str, *, payload: str | None = None,
    ) -> tuple[str, str, bool] | None:
        candidates = []
        for relation in self.relations:
            participants = list(relation.get("participant_roles", ()))
            mediated = False
            endpoint = support
            if len(participants) != 2:
                continue
            if support not in participants:
                if capability_id != "SUPPORT_DRINKWARE" or payload is None or payload not in participants:
                    continue
                endpoint = payload
                mediated = True
            anchor = participants[1] if participants[0] == endpoint else participants[0]
            phrase = str(relation.get("relation", "")).replace("_", " ")
            anchor_types = self._types(anchor)
            if capability_id == "SUPPORT_DRINKWARE":
                valid = "SEATING_POSITION" in anchor_types and re.search(
                    r"\b(?:near(?:by)?|beside|adjacent|close)\b", phrase, re.I,
                )
            else:
                valid = "SEATING_PAIR" in anchor_types and re.search(r"\b(between|both|accessible)\b", phrase, re.I)
            if valid:
                candidates.append((anchor, str(relation.get("id", "")), mediated, relation))
        unique = {(anchor, relation_id, mediated): relation
                  for anchor, relation_id, mediated, relation in candidates}
        if len(unique) != 1:
            return None
        anchor, relation_id, mediated = next(iter(unique))
        return anchor, relation_id, mediated

    def _join_living_context(self, operation: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        operation_text = str(operation.get("operation", "")).replace("_", " ").replace("-", " ")
        if self.domain != "living_room" or not self._MOVE.search(operation_text):
            return None
        original = list(operation["participant_roles"])
        removals = [None]
        if len(original) == 3:
            removals += [p for p in original if self._is_current_location(p)]
        candidates: list[tuple[dict[str, Any], list[dict[str, Any]], str | None, str, bool, str, str]] = []
        for removed in removals:
            direct = [p for p in original if p != removed] if removed else list(original)
            if len(direct) != 2:
                continue
            base = deepcopy(operation)
            base["participant_roles"] = direct
            for option in self.resolve_slots(self.domain, base, self.roles, self.hypotheses):
                anchor = self._explicit_anchor(
                    option["source_role"], option["capability_id"], payload=option["target_role"],
                )
                if anchor is None:
                    continue
                anchor_role, relation_id, mediated = anchor
                joined = deepcopy(operation)
                joined["participant_roles"] = [*direct, anchor_role]
                resolved = [row for row in self.resolve_slots(self.domain, joined, self.roles, self.hypotheses)
                            if row["capability_id"] == option["capability_id"]]
                if resolved:
                    candidates.append((joined, resolved, removed, relation_id, mediated,
                                       option["target_role"], option["source_role"]))
        unique = {(tuple(item[0]["participant_roles"]), item[1][0]["capability_id"], item[2], item[3]): item for item in candidates}
        if len(unique) != 1:
            return None
        joined, options, removed, relation_id, mediated, payload, support = next(iter(unique.values()))
        if mediated:
            relation = next(item for item in self.relations if str(item.get("id", "")) == relation_id)
            raw_participants = list(relation["participant_roles"])
            anchor_role = next(participant for participant in raw_participants if participant != payload)
            relation["participant_roles"] = [support, anchor_role]
            relation["raw_fm_participant_roles"] = raw_participants
            relation["operation_mediated_payload_role"] = payload
            self.trace.append({
                "code": "OPERATION_MEDIATED_RELATION_TARGET_NORMALIZATION",
                "operation_id": operation["id"],
                "relation_id": relation_id,
                "raw_participant_roles": raw_participants,
                "normalized_participant_roles": [support, anchor_role],
                "provenance": "FM_EXPLICIT_RELATION_AND_OPERATION",
            })
        joined["explicit_participant_roles"] = list(joined["participant_roles"])
        joined["raw_fm_participant_roles"] = original
        joined["graph_join_relation_id"] = relation_id
        if removed:
            joined["current_state_context_roles"] = [removed]
            self.trace.append({
                "code": "CURRENT_STATE_OPERATION_CONTEXT_ELIDED",
                "operation_id": operation["id"], "raw_role": removed,
                "provenance": "FM_ROLE_FUNCTION_AND_GRAPH_STRUCTURE",
            })
        self.trace.append({
            "code": "EXPLICIT_RELATION_OPERATION_CONTEXT_JOIN",
            "operation_id": operation["id"], "relation_id": relation_id,
            "joined_context_role": joined["participant_roles"][-1],
            "capability_id": options[0]["capability_id"],
            "provenance": "FM_EXPLICIT_RELATION",
        })
        return joined, options

    def _lower_beverage_macro(self, operation: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        if self.domain != "kitchen" or not self._BEVERAGE_MACRO.search(str(operation.get("operation", "")).replace("_", " ")):
            return None
        participants = list(operation["participant_roles"])
        sources = [p for p in participants if self._types(p) & {"coffee_source", "water_source"}]
        containers = [p for p in participants if "coffee_container" in self._types(p)]
        if len(sources) != 2 or len(containers) != 1 or len(participants) != 3:
            return None
        lowered = []
        for index, source in enumerate(sorted(sources), 1):
            primitive = {
                "id": f"{operation['id']}__primitive_{index}",
                "operation": "transfer material into receiving container",
                "participant_roles": [source, containers[0]],
                "operation_count": operation["operation_count"],
                "lowered_from_operation_id": operation["id"],
                "explicit_participant_roles": participants,
            }
            if not self.resolve_slots(self.domain, primitive, self.roles, self.hypotheses):
                return None
            lowered.append(primitive)
        self.trace.append({
            "code": "DETERMINISTIC_COMPOSITE_OPERATION_LOWERING",
            "operation_id": operation["id"],
            "primitive_operation_ids": [item["id"] for item in lowered],
            "participant_union": sorted(participants),
            "provenance": "FM_EXPLICIT_OPERATION_AND_ROLE_FUNCTIONS",
        })
        return lowered

    def interpret(self, operations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        interpreted: list[dict[str, Any]] = []
        for raw in operations:
            operation = deepcopy(dict(raw))
            if is_non_physical_operation_phrase(str(operation.get("operation", ""))):
                interpreted.append(operation)
                self.accounting.append({
                    "element_kind": "operation", "raw_id": operation["id"],
                    "disposition": "NON_PHYSICAL_TASK_DIRECTIVE",
                    "canonical_representation": None, "provenance": "FM_EXPLICIT_OPERATION",
                })
                continue
            joined = self._join_living_context(operation)
            if joined is not None:
                normalized, options = joined
                interpreted.append(normalized)
                self.accounting.append({
                    "element_kind": "operation", "raw_id": operation["id"],
                    "disposition": "GRAPH_JOINED_PRIMITIVE_CAPABILITY",
                    "canonical_representation": sorted({row["capability_id"] for row in options}),
                    "provenance": "FM_EXPLICIT_OPERATION_PLUS_RELATION",
                })
                continue
            direct = self.resolve_slots(self.domain, operation, self.roles, self.hypotheses)
            if direct:
                interpreted.append(operation)
                self.accounting.append({
                    "element_kind": "operation", "raw_id": operation["id"],
                    "disposition": "PRIMITIVE_CAPABILITY",
                    "canonical_representation": sorted({row["capability_id"] for row in direct}),
                    "provenance": "FM_EXPLICIT_OPERATION",
                })
                continue
            lowered = self._lower_beverage_macro(operation)
            if lowered is not None:
                interpreted.extend(lowered)
                self.accounting.append({
                    "element_kind": "operation", "raw_id": operation["id"],
                    "disposition": "DETERMINISTIC_COMPOSITE_OPERATION_LOWERING",
                    "canonical_representation": [item["id"] for item in lowered],
                    "provenance": "FM_EXPLICIT_OPERATION_AND_ROLE_FUNCTIONS",
                })
                continue
            interpreted.append(operation)
            self.accounting.append({
                "element_kind": "operation", "raw_id": operation["id"],
                "disposition": "UNRESOLVED_REQUIRED_OPERATION",
                "canonical_representation": None, "provenance": "FM_EXPLICIT_OPERATION",
            })
        return interpreted
