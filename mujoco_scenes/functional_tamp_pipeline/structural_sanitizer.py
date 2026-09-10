"""Stage A: recover structurally meaningful FM graphs without semantic completion.

The input is never mutated. Disabled content remains in the audit, including
ambiguous counts: plurality is not silently converted into a singular role.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping


@dataclass
class SanitizationResult:
    document: dict[str, Any]
    repairs: list[dict[str, Any]]
    semantically_incomplete: bool
    succeeded: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(document=self.document, repairs=self.repairs,
                    semantically_incomplete=self.semantically_incomplete,
                    succeeded=self.succeeded)


def normalize_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")


# Keys of a canonical interaction group whose values are role identifiers.  The
# V3 converter records the capability reading it chose alongside the group, and
# those records name roles too.  Normalizing the group's own endpoints while
# leaving these alone made the two disagree: a contract whose role ids were not
# already lowercase slugs -- "Coffee_Cup" -- had every one of its operations
# rejected as naming a participant the runtime holds in no form, while that
# participant sat in the graph under its normalized name.
_GROUP_ROLE_ID_FIELDS: tuple[str, ...] = (
    "tool_role", "source_role", "target_role", "context_role", "anchor_role",
)
_GROUP_ROLE_ID_LIST_FIELDS: tuple[str, ...] = (
    "v3_participant_roles", "v3_explicit_participant_roles",
    "v3_raw_fm_participant_roles", "v3_witness_roles",
    "v3_current_state_context_roles",
)


def _normalize_group_role_references(item: dict[str, Any]) -> None:
    """Rewrite every role identifier a group carries, at any depth it carries one."""
    for key in _GROUP_ROLE_ID_LIST_FIELDS:
        values = item.get(key)
        if isinstance(values, list):
            item[key] = [normalize_id(value) for value in values if isinstance(value, str)]
    assignments = item.get("v3_slot_assignments")
    if isinstance(assignments, list):
        rewritten = []
        for row in assignments:
            if not isinstance(row, dict):
                continue
            row = dict(row)
            for key in _GROUP_ROLE_ID_FIELDS:
                if isinstance(row.get(key), str):
                    row[key] = normalize_id(row[key])
            rewritten.append(row)
        item["v3_slot_assignments"] = rewritten


def sanitize_functional_graph(raw: Mapping[str, Any], *, domain: str | None = None) -> SanitizationResult:
    if not isinstance(raw, Mapping):
        return SanitizationResult({}, [{"code": "INVALID_DOCUMENT"}], True, False)
    from .fm_schema_v3 import is_v3_document, convert_v3_to_canonical_document
    from .fm_schema_v2 import is_v2_document, convert_v2_to_canonical_document
    if is_v3_document(raw):
        if domain is None:
            return SanitizationResult({}, [{"code": "V3_DOMAIN_REQUIRED"}], True, False)
        doc = convert_v3_to_canonical_document(raw, domain=domain)
    elif is_v2_document(raw):
        doc = convert_v2_to_canonical_document(raw)
    else:
        doc = deepcopy(dict(raw))
    repairs: list[dict[str, Any]] = []
    incomplete = False

    def record(code: str, content: Any, *, semantic: bool = False) -> None:
        nonlocal incomplete
        repairs.append({"code": code, "raw": deepcopy(content)})
        incomplete |= semantic

    roles = doc.get("functional_roles", [])
    if not isinstance(roles, list):
        roles = []
        record("INVALID_ROLE_LIST", doc.get("functional_roles"), semantic=True)
    groups = doc.get("interaction_groups", [])
    groups = groups if isinstance(groups, list) else []
    kept = []
    ids: set[str] = set()
    for raw_role in roles:
        if not isinstance(raw_role, dict):
            record("INVALID_ROLE", raw_role, semantic=True)
            continue
        role = deepcopy(raw_role)
        rid = normalize_id(role.get("id", ""))
        if not rid or rid in ids or not isinstance(role.get("function"), str) or not role["function"].strip():
            record("INVALID_ROLE", role, semantic=True)
            continue
        if role.get("id") != rid:
            record("NORMALIZED_ID", {"old": role.get("id"), "new": rid})
        role["id"] = rid
        if "required_count" not in role:
            text = " ".join(str(role.get(k, "")) for k in ("function", "description"))
            plural = bool(re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|each|both|multiple|several|[2-9]\d*)\b", text, re.I))
            binding_evidence = role.get("binding_cardinality") or role.get("min_count") or role.get("max_count")
            group_evidence = any(isinstance(g, dict) and normalize_id(g.get("target_role", "")) == rid
                                 and g.get("required_target_count", 1) != 1 for g in groups)
            if plural or binding_evidence or group_evidence:
                record("UNRESOLVED_REQUIRED_COUNT", role, semantic=True)
                continue
            role["required_count"] = 1
            record("DEFAULTED_REQUIRED_COUNT", {"role": rid, "value": 1})
        if (type(role["required_count"]) is not int or role["required_count"] < 1
                or role.get("entity_kind") not in {"OBJECT", "REGION", "FIXED_TARGET"}
                or role.get("binding_policy") not in {"DISTINCT", "SHARED", "REUSABLE"}):
            record("INVALID_ROLE", role, semantic=True)
            continue
        # Optional observations do not define role identity or multiplicity.
        candidates = role.get("visible_candidates", [])
        seen: dict[str, Any] = {}
        unique = []
        if not isinstance(candidates, list):
            record("IGNORED_OPTIONAL_METADATA", {"role": rid, "visible_candidates": candidates})
            candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                record("IGNORED_OPTIONAL_METADATA", candidate)
                continue
            cid = normalize_id(candidate.get("id", candidate.get("candidate_id", "")))
            key = cid or repr(sorted(candidate.items()))
            if key in seen:
                code = "DUPLICATE_VISIBLE_CANDIDATE_REMOVED" if candidate == seen[key] else "CONFLICTING_VISIBLE_CANDIDATE_DISABLED"
                record(code, {"role": rid, "candidate": candidate})
                if candidate != seen[key]:
                    unique = [c for c in unique if c != seen[key]]
                continue
            seen[key] = candidate
            unique.append(candidate)
        role["visible_candidates"] = unique
        role.setdefault("description", "")
        for key in ("required_properties", "candidate_categories"):
            if key not in role:
                role[key] = []
            elif not isinstance(role[key], list) or any(not isinstance(v, str) for v in role[key]):
                record("INVALID_ROLE", role, semantic=True)
                break
        else:
            kept.append(role)
            ids.add(rid)
    doc["functional_roles"] = kept
    relations = doc.get("functional_relations", [])
    retained = []
    operation_endpoint_ids = {
        normalize_id(group.get(key, ""))
        for group in groups if isinstance(group, dict)
        for key in ("tool_role", "source_role", "target_role", "context_role", "anchor_role")
        if group.get(key)
    }
    guidance = doc.get("raw_v2_guidance", doc.get("observation_guidance", {}))
    observed_role_ids = {
        normalize_id(role_id)
        for role_id, candidates in guidance.get("visible_candidates_per_role", {}).items()
        if candidates
    } if isinstance(guidance, dict) else set()
    if not isinstance(relations, list):
        record("INVALID_RELATION_LIST", relations, semantic=True)
        relations = []
    for rel in relations:
        if not isinstance(rel, dict):
            record("INVALID_RELATION", rel, semantic=True)
            continue
        item = deepcopy(rel)
        for key in ("subject_role", "object_role"):
            item[key] = normalize_id(item.get(key, ""))
        missing_subject = item["subject_role"] not in ids
        missing_object = item["object_role"] not in ids
        if not missing_subject and missing_object:
            from .relation_interpreter import interpret_task_effect_predicate
            phrase = item.get("relation", item.get("predicate", ""))
            effect_predicate = (
                interpret_task_effect_predicate(phrase)
                if isinstance(phrase, str) else None
            )
            if (
                effect_predicate
                and item["object_role"]
                and item["object_role"] not in operation_endpoint_ids
                and item["object_role"] not in observed_role_ids
            ):
                item["semantic_category"] = "TASK_EFFECT_SEMANTICS"
                item["effect_predicate"] = effect_predicate
                item["object_is_literal"] = True
                retained.append(item)
                record("PRESERVED_TASK_EFFECT_LITERAL", rel)
                continue
        if missing_subject or missing_object:
            record("DANGLING_RELATION_REFERENCE", rel, semantic=True)
            continue
        if not isinstance(item.get("relation", item.get("predicate")), str):
            record("INVALID_RELATION", rel, semantic=True)
            continue
        retained.append(item)
    doc["functional_relations"] = retained
    retained_groups = []
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            record("INVALID_GROUP_REFERENCE", group, semantic=True)
            continue
        item = deepcopy(group)
        keys = ["tool_role", "target_role"] + (["context_role"] if item.get("context_role") else [])
        for key in keys:
            item[key] = normalize_id(item.get(key, ""))
        _normalize_group_role_references(item)
        gid = normalize_id(item.get("id", ""))
        if any(item[k] not in ids for k in keys) or not gid or gid in group_ids:
            record("INVALID_GROUP_REFERENCE", group, semantic=True)
            continue
        item["id"] = gid
        if any(not isinstance(item.get(k, []), list) or any(not isinstance(r, str) for r in item.get(k, []))
               for k in ("required_relations", "context_relations")):
            record("INVALID_GROUP_STRUCTURE", group, semantic=True)
            continue
        # Lists may contain any number of relation phrases; never truncate them.
        retained_groups.append(item)
        group_ids.add(gid)
    doc["interaction_groups"] = retained_groups
    for key in ("current_state_operation_context_roles",):
        values = doc.get(key)
        if isinstance(values, list):
            doc[key] = [normalize_id(value) for value in values if isinstance(value, str)]
    for key in ("inspectable_regions", "inspection_order"):
        if not isinstance(doc.get(key, []), list):
            record("IGNORED_OPTIONAL_METADATA", {key: doc[key]})
            doc[key] = []
        doc.setdefault(key, [])
    return SanitizationResult(doc, repairs, incomplete, bool(kept))
