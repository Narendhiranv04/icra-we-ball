"""Image-conditioned Qwen requirement generation for Kitchen and Living Room.

The model independently produces roles, properties, counts, and visible
candidate objects. This module audits the response against reviewed task
contracts after the call and stops before grounding, allocation, planning, or
execution.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml

from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    UnmappedFunctionalConceptError,
    UnsupportedCheckerCapabilityError,
    VLMSpecificationError,
)
from .workshop_phase1.fm_adapter import FMAdapter


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZATION = (
    Path(__file__).resolve().parent
    / "configs"
    / "kitchen_living_room_vlm_normalization.yaml"
)
SUPPORTED_ENVIRONMENTS = ("kitchen", "living_room")
KITCHEN_SEARCH_REGIONS = {
    "D1": "upper kitchen drawer",
    "D2": "lower kitchen drawer",
    "C2": "upper wall cupboard",
    "B1": "countertop storage box",
    "C1": "lower kitchen cupboard",
}

VLM_CANONICALIZATION_VERSION = "phase3_6a7_v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return document


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def _phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _phrase_score(text: str, alias: str) -> float:
    text_words = set(_phrase(text).split())
    alias_words = set(_phrase(alias).split())
    if not alias_words:
        return 0.0
    return len(text_words & alias_words) / len(alias_words)


def map_living_room_role_function(function_text: str) -> str | None:
    """Deterministic compositional concept matching for Living Room functional roles."""
    norm = _phrase(function_text)
    norm_surface = norm.replace("coffee table", "table").replace("coffee_table", "table")
    words = set(norm_surface.split())

    has_support = any(w in words or w in norm_surface for w in ("support", "surface", "table", "hold", "rest", "platform", "place", "area", "side"))
    has_personal = any(w in words or w in norm_surface for w in ("personal", "individual", "viewer", "person", "seated", "occupant", "seat", "left", "right", "armchair", "each", "single", "one"))
    has_drink = any(w in words or w in norm_surface for w in ("cup", "saucer", "drink", "drinkware", "beverage", "tea", "beverages", "drinkwares", "coffee"))

    has_shared = any(w in words or w in norm_surface for w in ("shared", "central", "both", "common", "middle", "mutual", "two", "viewers", "center"))
    has_remote = any(w in words or w in norm_surface for w in ("remote", "controller", "tv", "television", "control"))

    is_personal_drink = has_support and (has_personal or has_drink) and not (has_shared or has_remote)
    is_shared_remote = has_support and (has_shared or has_remote) and not (has_personal and has_drink)

    if is_personal_drink and not is_shared_remote:
        return "personal_cup_saucer"
    if is_shared_remote and not is_personal_drink:
        return "shared_remote"
    return None


def map_living_room_relation(
    relation_text: str,
    relation_aliases: dict[str, list[str]] | None = None,
    *,
    fail_closed: bool = True,
) -> str | None:
    """Deterministic concept matching for Living Room binary relations."""
    norm = _phrase(relation_text)
    if not norm:
        if fail_closed:
            raise UnmappedFunctionalConceptError("Empty relation text cannot be mapped")
        return None
    matches = set()
    if any(k in norm for k in (
        "near seat", "near the seat", "near armchair", "reach of one seated",
        "near seating", "adjacent to viewer", "within reach of seat",
        "near assigned seat", "adjacent to seat", "adjacent to armchair",
        "within reach of armchair", "reach of seat", "reach of armchair",
    )):
        matches.add("NEAR_SEAT")
    if any(k in norm for k in (
        "accessible from both", "reachable by both", "shared access",
        "both viewers", "between seats", "central access", "accessible to both",
        "accessible from both seating positions", "shared access from both",
    )):
        matches.add("ACCESSIBLE_FROM_BOTH_SEATS")
    if any(k in norm for k in (
        "fit set", "fits set", "fit cup and saucer", "fits cup and saucer",
        "hold cup and saucer", "fit the complete set", "support cup and saucer",
        "hold the complete set", "fit the payload set", "support drinkware",
        "support personal drinkware", "fits personal drinkware",
    )):
        matches.add("FITS_SET_ON")
    if any(k in norm for k in (
        "fit remote", "fits remote", "support remote", "accommodate remote",
        "fit the remote", "fit television remote", "support television remote",
        "fits television remote",
    )):
        matches.add("FITS_ON")

    if relation_aliases:
        for pred in ("NEAR_SEAT", "ACCESSIBLE_FROM_BOTH_SEATS", "FITS_SET_ON", "FITS_ON"):
            for alias in relation_aliases.get(pred, []):
                a_norm = _phrase(alias)
                if a_norm == norm or f" {a_norm} " in f" {norm} " or f" {norm} " in f" {a_norm} ":
                    matches.add(pred)
                    break

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise AmbiguousCanonicalizationError(
            f"Ambiguous living room relation {relation_text!r} matches: {sorted(matches)}"
        )
    if fail_closed:
        raise UnmappedFunctionalConceptError(
            f"VLM living room relation {relation_text!r} cannot be mapped to any reviewed relation"
        )
    return None


def map_living_room_object_payload_role(raw: dict[str, Any]) -> str | None:
    """Deterministic concept matching for Living Room task-explicit payload OBJECT roles."""
    if raw.get("entity_kind") != "OBJECT":
        return None
    fn_desc = f"{raw.get('function', '')} {raw.get('description', '')}"
    cand_cats = " ".join(raw.get("candidate_categories", []))
    norm_all = _phrase(f"{fn_desc} {cand_cats}")
    if not norm_all:
        return None
    has_cup = any(w in norm_all for w in ("cup", "saucer", "drinkware", "drink", "coffee", "tea", "beverage", "cup_saucer_set", "cup saucer", "cups"))
    has_remote = any(w in norm_all for w in ("remote", "controller", "tv_remote", "remote_control", "television_remote", "control device", "tv"))
    if has_cup and not has_remote:
        return "CUP_SAUCER_SET"
    if has_remote and not has_cup:
        return "REMOTE"
    return None


def map_living_room_fixed_target_role(raw: dict[str, Any]) -> str | None:
    """Deterministic concept matching for Living Room contextual FIXED_TARGET roles."""
    if raw.get("entity_kind") != "FIXED_TARGET":
        return None
    fn_desc = f"{raw.get('function', '')} {raw.get('description', '')}"
    norm = _phrase(fn_desc)
    if not norm:
        return None
    if any(k in norm for k in (
        "seating pair", "both seats", "seating area", "pair of seats",
        "armchairs", "armchair positions", "viewer positions", "viewing seats",
    )):
        return "SEATING_PAIR"
    if any(k in norm for k in ("seating position", "seat position", "seating", "seat", "armchair", "chair")):
        return "SEATING_POSITION"
    return None


class EnvironmentVLMRequirementProvider:
    """Generate once, then audit against a frozen environment contract."""

    def __init__(
        self,
        environment: str,
        *,
        fm_adapter: FMAdapter | None = None,
        normalization_path: str | Path = DEFAULT_NORMALIZATION,
    ) -> None:
        if environment not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"environment must be one of {', '.join(SUPPORTED_ENVIRONMENTS)}"
            )
        self.environment = environment
        self.fm_adapter = fm_adapter or FMAdapter()
        normalization = _load_yaml(_resolve(normalization_path))
        if normalization.get("schema_version") != 1:
            raise ValueError("Unsupported Kitchen/Living-Room VLM normalization schema")
        self.unary_property_aliases = normalization.get("unary_property_aliases", {})
        self.binary_relation_aliases = normalization.get("binary_relation_aliases", {})
        self.relation_aliases = normalization.get("relation_aliases", {}) or {
            **self.unary_property_aliases,
            **self.binary_relation_aliases,
        }
        self.environment_config = normalization["environments"][environment]
        self.task_path = _resolve(self.environment_config["task_path"])
        self.vocabulary_path = _resolve(self.environment_config["vocabulary_path"])
        self.manual_task = _load_yaml(self.task_path)
        self.vocabulary = _load_yaml(self.vocabulary_path)
        self.instruction = str(
            self.manual_task[self.environment_config["instruction_field"]]
        )
        self.task_instruction = self.instruction
        self.raw_decomposition: dict[str, Any] | None = None
        self.normalized_task: dict[str, Any] | None = None
        self.normalized_requirements: list[dict[str, Any]] | None = None
        self.normalized_relations: list[dict[str, Any]] = []
        self.vlm_derived_role_vocabulary: tuple[str, ...] = ()
        self.task_explicit_context_vocabulary: tuple[str, ...] = ()
        self.ranked_detector_vocabulary: list[dict[str, Any]] | None = None
        self.normalization_issues: list[str] = []
        self.ready_for_grounding = False
        self.inspection_policy: dict[str, Any] | None = None

    def _vocabulary_aliases(self) -> dict[str, list[str]]:
        raw = self.vocabulary.get("canonical_labels", {})
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"Semantic vocabulary is empty: {self.vocabulary_path}")
        return {
            str(canonical): [str(alias) for alias in aliases]
            for canonical, aliases in raw.items()
        }

    def _role_specs(self) -> list[dict[str, Any]]:
        language_roles = self.environment_config["roles"]
        if self.environment == "kitchen":
            relations_by_subject: dict[str, list[str]] = {}
            for relation in self.manual_task.get("relations", []):
                relations_by_subject.setdefault(relation["subject_role"], []).append(
                    relation["predicate"]
                )
            specifications = []
            for role_id, role in self.manual_task["roles"].items():
                properties = [
                    item["predicate"] for item in role.get("unary_geometry", [])
                ]
                specifications.append(
                    {
                        "role_id": role_id,
                        "canonical_function": role_id.upper(),
                        "entity_kind": "OBJECT",
                        "purpose": language_roles[role_id]["function_hint"],
                        "function_aliases": language_roles[role_id]["function_aliases"],
                        "required_count": int(
                            role.get("count", role.get("binding_cardinality", {}).get(
                                "minimum_distinct_physical_objects", 1
                            ))
                        ),
                        "categories": [
                            preference["canonical_label"]
                            for preference in role["semantic_preferences"]
                        ],
                        "properties": list(dict.fromkeys(properties)),
                    }
                )
            for role_id, role in self.manual_task.get("symbolic_task", {}).get(
                "source_roles", {}
            ).items():
                specifications.append(
                    {
                        "role_id": role_id,
                        "canonical_function": f"PROVIDE_{str(role['provides']).upper()}",
                        "entity_kind": "OBJECT",
                        "purpose": language_roles[role_id]["function_hint"],
                        "function_aliases": language_roles[role_id]["function_aliases"],
                        "required_count": int(role.get("count", 1)),
                        "categories": list(role["accepted_semantic_labels"]),
                        "properties": [],
                    }
                )
            return specifications

        specifications = []
        region_roles = self.manual_task["semantic_requirements"]["region_roles"]
        for role_id, group in self.manual_task["function_groups"].items():
            role_name = group["region_role"]
            specifications.append(
                {
                    "role_id": role_id,
                    "canonical_function": group["function_id"],
                    "entity_kind": "REGION",
                    "purpose": language_roles[role_id]["function_hint"],
                    "function_aliases": language_roles[role_id]["function_aliases"],
                    "required_count": int(group.get("required_target_count", 1)),
                    "categories": list(region_roles[role_name]["accepted_categories"]),
                    "properties": list(group["required_relations"]),
                }
            )
        return specifications

    def _category_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for canonical, aliases in self._vocabulary_aliases().items():
            result[_phrase(canonical)] = canonical
            for alias in aliases:
                result[_phrase(alias)] = canonical
        return result

    def _map_category(self, value: object) -> str | None:
        normalized = _phrase(value)
        if not normalized:
            return None
        aliases = self._category_map()
        if normalized in aliases:
            return aliases[normalized]
        matches = {
            canonical
            for alias, canonical in aliases.items()
            if f" {alias} " in f" {normalized} "
        }
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            raise AmbiguousCanonicalizationError(f"Ambiguous category match for {value!r}: {sorted(matches)}")
        return None

    def _map_properties(self, values: list[str], *, fail_closed: bool = True) -> set[str]:
        mapped: set[str] = set()
        for value in values:
            norm = _phrase(value)
            prop_matches = set()
            for predicate, aliases in self.unary_property_aliases.items():
                for alias in aliases:
                    a_norm = _phrase(alias)
                    if a_norm == norm or f" {a_norm} " in f" {norm} ":
                        prop_matches.add(predicate)
                        break
            if len(prop_matches) == 1:
                mapped.add(next(iter(prop_matches)))
            elif len(prop_matches) > 1:
                raise AmbiguousCanonicalizationError(
                    f"Ambiguous property {value!r} matches multiple unary predicates: {sorted(prop_matches)}"
                )
            elif fail_closed:
                raise UnsupportedCheckerCapabilityError(
                    f"VLM required property {value!r} is not supported by any available checker in {self.environment}"
                )
        return mapped

    def _detector_label(self, canonical: str) -> str:
        aliases = self._vocabulary_aliases()[canonical]
        return aliases[0] if aliases else canonical.replace("_", " ")

    def _candidate_categories(self, raw: dict[str, Any]) -> list[str]:
        categories: list[str] = []
        for candidate in raw.get("visible_candidates", []):
            canonical = self._map_category(candidate.get("label", ""))
            if canonical is not None and canonical not in categories:
                categories.append(canonical)
        for cat in raw.get("candidate_categories", []):
            canonical = self._map_category(cat)
            if canonical is not None and canonical not in categories:
                categories.append(canonical)
        return categories

    def _assign_roles(
        self,
        raw_requirements: list[dict[str, Any]],
        specs: list[dict[str, Any]],
    ) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[str]]:
        issues: list[str] = []
        matched: dict[str, list[dict[str, Any]]] = {}
        for raw in raw_requirements:
            func_text = _phrase(f"{raw.get('function', '')} {raw.get('description', '')}")
            matching_specs = []
            for spec in specs:
                if raw.get("entity_kind") != spec.get("entity_kind"):
                    continue
                for alias in spec.get("function_aliases", []):
                    a_norm = _phrase(alias)
                    if a_norm == func_text or f" {a_norm} " in f" {func_text} ":
                        matching_specs.append(spec)
                        break
            if len(matching_specs) != 1:
                issues.append(
                    f"raw role {raw['id']!r} is unmapped or ambiguous; "
                    f"matches={[s['role_id'] for s in matching_specs]}"
                )
                continue
            spec = matching_specs[0]
            role_id = spec["role_id"]
            matched.setdefault(role_id, []).append(raw)
        for spec in specs:
            if spec["role_id"] not in matched:
                issues.append(f"reviewed role {spec['role_id']!r} was not recovered")
        return [
            (spec, matched[spec["role_id"]])
            for spec in specs
            if spec["role_id"] in matched
        ], issues

    def generate_canonical(
        self,
        instruction: str | None = None,
        *,
        observation_images: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        return self.generate(
            instruction,
            observation_images=observation_images,
            canonical=True,
        )

    def generate(
        self,
        instruction: str | None = None,
        *,
        observation_images: list[str | Path] | None = None,
        require_reviewed_contract: bool = False,
        include_inspection_policy: bool = False,
        canonical: bool = False,
    ) -> dict[str, Any]:
        if self.raw_decomposition is not None:
            if require_reviewed_contract and not self.ready_for_grounding:
                raise ValueError(
                    "VLM output is not ready for grounding: "
                    + "; ".join(self.normalization_issues)
                )
            return self.result()
        task_instruction = instruction or self.instruction
        self.task_instruction = task_instruction
        document = self.fm_adapter.generate_task_requirements(
            task_instruction, observation_images=observation_images or []
        )
        self.raw_decomposition = document
        if include_inspection_policy:
            if self.environment != "kitchen":
                raise ValueError("VLM inspection policy is currently Kitchen-only")
            self.inspection_policy = self.fm_adapter.generate_inspection_priors(
                task_instruction,
                KITCHEN_SEARCH_REGIONS,
                observation_images=observation_images or [],
            )
        if document.get("status") != "SUPPORTED":
            raise VLMSpecificationError(
                f"VLM marked {self.environment} unsupported: "
                f"{document.get('unsupported_reason', 'no reason')}"
            )
        raw_requirements = document.get("functional_roles", [])
        normalized_records = []
        category_rank: dict[str, int] = {}
        issues: list[str] = []

        if not canonical:
            specs = self._role_specs()
            assigned, issues = self._assign_roles(raw_requirements, specs)
            for spec, raw_group in assigned:
                raw = {
                    "id": raw_group[0]["id"],
                    "entity_kind": spec["entity_kind"],
                    "function": " / ".join(item["function"] for item in raw_group),
                    "description": " ".join(item["description"] for item in raw_group),
                    "required_count": sum(item["required_count"] for item in raw_group),
                    "visible_candidates": [
                        candidate
                        for item in raw_group
                        for candidate in item.get("visible_candidates", [])
                    ],
                    "required_properties": list(dict.fromkeys(
                        property_text
                        for item in raw_group
                        for property_text in item.get("required_properties", [])
                    )),
                }
                categories = self._candidate_categories(raw)
                for canonical_cat in categories:
                    if canonical_cat in spec["categories"]:
                        category_rank.setdefault(canonical_cat, len(category_rank) + 1)
                properties = self._map_properties(raw["required_properties"], fail_closed=False)
                missing_properties = set(spec["properties"]) - properties
                if missing_properties:
                    issues.append(
                        f"VLM role {spec['role_id']} omitted required properties: "
                        f"{sorted(missing_properties)}"
                    )
                count_matches = raw["required_count"] == spec["required_count"]
                if not count_matches:
                    issues.append(
                        f"VLM role {spec['role_id']} required_count={raw['required_count']} "
                        f"but reviewed minimum is {spec['required_count']}"
                    )
                normalized_records.append(
                    {
                        "role_id": spec["role_id"],
                        "raw_vlm_role_id": raw["id"],
                        "raw_vlm_role_ids": [item["id"] for item in raw_group],
                        "entity_kind": spec["entity_kind"],
                        "function": spec["canonical_function"],
                        "raw_function": raw["function"],
                        "vlm_required_count": raw["required_count"],
                        "reviewed_required_count": spec["required_count"],
                        "description": raw["description"],
                        "accepted_categories": list(spec["categories"]),
                        "visible_candidates": raw["visible_candidates"],
                        "required_properties": list(spec["properties"]),
                        "semantic_hints": [
                            candidate["label"] for candidate in raw["visible_candidates"] if candidate.get("label")
                        ],
                        "source": "FM",
                        "provenance": "qwen_vlm_normalized_by_reviewed_ontology",
                        "normalization_status": (
                            "COMPLETE"
                            if not missing_properties and count_matches
                            else "REVIEW_REQUIRED"
                        ),
                        "missing_reviewed_properties": sorted(missing_properties),
                    }
                )
            self.normalization_issues = issues
            self.ready_for_grounding = not issues
            if self.ready_for_grounding:
                normalized_task = deepcopy(self.manual_task)
                normalized_task["specification_source"] = (
                    "qwen_vlm_normalized_by_reviewed_ontology"
                )
                normalized_task["generated_from_foundation_model"] = True
                self.normalized_task = normalized_task
            else:
                self.normalized_task = None
        else:
            language_roles = self.environment_config.get("roles", {})
            grouped_requirements: dict[str, list[dict[str, Any]]] = {}
            raw_id_to_canon: dict[str, str] = {}

            for raw in raw_requirements:
                raw_id = raw.get("id")
                raw_kind = raw.get("entity_kind")

                if raw_kind == "FIXED_TARGET":
                    matched_role_id = map_living_room_fixed_target_role(raw)
                    if matched_role_id is None:
                        fn_text = f"{raw.get('function', '')} {raw.get('description', '')}"
                        raise UnmappedFunctionalConceptError(
                            f"VLM role {raw_id!r} with function {fn_text!r} cannot be mapped to any reviewed FIXED_TARGET role"
                        )
                    canonical_func = matched_role_id.upper()
                elif raw_kind == "OBJECT":
                    matched_payload = map_living_room_object_payload_role(raw)
                    if matched_payload is not None:
                        matched_role_id = matched_payload
                        canonical_func = matched_payload
                    else:
                        func_text = str(raw.get("function", ""))
                        comp_role = map_living_room_role_function(f"{func_text} {raw.get('description', '')}")
                        matched_role_id = comp_role
                        if matched_role_id is None:
                            fn_text = f"{raw.get('function', '')} {raw.get('description', '')}"
                            raise UnmappedFunctionalConceptError(
                                f"VLM role {raw_id!r} with function {fn_text!r} cannot be mapped to any reviewed OBJECT role"
                            )
                        canonical_func = (
                            "PERSONAL_CUP_SAUCER_REGION"
                            if matched_role_id == "personal_cup_saucer"
                            else "SHARED_REMOTE_REGION"
                            if matched_role_id == "shared_remote"
                            else matched_role_id.upper()
                        )
                else:
                    func_text = str(raw.get("function", ""))
                    comp_role = map_living_room_role_function(f"{func_text} {raw.get('description', '')}")
                    matched_role_id = comp_role
                    if matched_role_id is None:
                        norm_func = _phrase(f"{func_text} {raw.get('description', '')}")
                        matching_roles = []
                        for r_id, r_cfg in language_roles.items():
                            for alias in r_cfg.get("function_aliases", []):
                                a_norm = _phrase(alias)
                                if a_norm == norm_func or f" {a_norm} " in f" {norm_func} ":
                                    matching_roles.append(r_id)
                                    break
                        if len(matching_roles) == 1:
                            matched_role_id = matching_roles[0]
                        elif len(matching_roles) > 1:
                            raise AmbiguousCanonicalizationError(
                                f"VLM role {raw_id!r} with function {raw.get('function')!r} "
                                f"is ambiguous across canonical roles: {sorted(matching_roles)}"
                            )

                    if matched_role_id is None:
                        raise UnmappedFunctionalConceptError(
                            f"VLM role {raw_id!r} with function {raw.get('function')!r} "
                            "cannot be mapped to any reviewed canonical role"
                        )

                    canonical_func = (
                        "PERSONAL_CUP_SAUCER_REGION"
                        if matched_role_id == "personal_cup_saucer"
                        else "SHARED_REMOTE_REGION"
                        if matched_role_id == "shared_remote"
                        else matched_role_id.upper()
                    )

                raw_id_to_canon[raw_id] = canonical_func
                grouped_requirements.setdefault(canonical_func, []).append({
                    "raw": raw,
                    "canonical_func": canonical_func,
                    "matched_role_id": matched_role_id,
                })

            vlm_role_vocab: list[str] = []
            for canonical_func, group in grouped_requirements.items():
                raw_group = [item["raw"] for item in group]
                matched_role_id = group[0]["matched_role_id"]
                raw_vlm_role_ids = [raw["id"] for raw in raw_group]
                entity_kinds = {raw.get("entity_kind") for raw in raw_group}
                if len(entity_kinds) != 1:
                    raise MalformedVLMSpecificationError(f"Mixed entity kinds in canonical group {canonical_func}: {entity_kinds}")
                entity_kind = entity_kinds.pop()

                binding_policies = {raw.get("binding_policy") for raw in raw_group}
                if len(binding_policies) > 1:
                    raise MalformedVLMSpecificationError(f"Conflicting binding policies in canonical group {canonical_func}: {binding_policies}")
                binding_policy = binding_policies.pop()

                total_count = sum(int(raw.get("required_count", 1)) for raw in raw_group)
                raw_func = " / ".join(dict.fromkeys(raw.get("function", "") for raw in raw_group if raw.get("function")))
                raw_desc = " ".join(dict.fromkeys(raw.get("description", "") for raw in raw_group if raw.get("description")))

                cand_cats = list(dict.fromkeys(
                    cat.strip()
                    for raw in raw_group
                    for cat in raw.get("candidate_categories", [])
                    if str(cat).strip()
                ))
                if entity_kind in ("OBJECT", "REGION") and canonical_func in ("PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION"):
                    if not cand_cats:
                        raise MalformedVLMSpecificationError(
                            f"Discoverable Living Room role {canonical_func!r} must specify non-empty candidate_categories"
                        )
                vlm_role_vocab.extend(cand_cats)
                run_local_cats = list(dict.fromkeys(_phrase(c).replace(" ", "_") for c in cand_cats if c))

                hints = list(dict.fromkeys(
                    candidate["label"]
                    for raw in raw_group
                    for candidate in raw.get("visible_candidates", [])
                    if candidate.get("label")
                ))

                required_properties = sorted(
                    set().union(*(self._map_properties(raw.get("required_properties", []), fail_closed=True) for raw in raw_group))
                )
                normalized_records.append(
                    {
                        "role_id": matched_role_id,
                        "raw_vlm_role_ids": raw_vlm_role_ids,
                        "entity_kind": entity_kind,
                        "binding_policy": binding_policy,
                        "function": canonical_func,
                        "raw_function": raw_func,
                        "vlm_required_count": total_count,
                        "description": raw_desc,
                        "candidate_categories": cand_cats,
                        "accepted_categories": run_local_cats or cand_cats,
                        "required_properties": required_properties,
                        "visible_candidates": [
                            candidate
                            for raw in raw_group
                            for candidate in raw.get("visible_candidates", [])
                        ],
                        "semantic_hints": hints,
                        "source": "FM",
                        "provenance": "qwen_vlm_normalized_by_generic_ontology",
                        "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
                        "normalization_status": "COMPLETE",
                    }
                )

            # Canonicalize relations losslessly
            canonical_relations: list[dict[str, Any]] = []
            for rel_item in self.raw_decomposition.get("functional_relations", []):
                s = rel_item.get("subject_role")
                r = rel_item.get("relation")
                o = rel_item.get("object_role")
                if s not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(
                        f"VLM relation subject role {s!r} not declared in living room roles"
                    )
                if o not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(
                        f"VLM relation object role {o!r} not declared in living room roles"
                    )
                s_canon = raw_id_to_canon[s]
                o_canon = raw_id_to_canon[o]
                mapped_r = map_living_room_relation(r, self.binary_relation_aliases)
                if mapped_r is None:
                    raise UnmappedFunctionalConceptError(
                        f"VLM living room relation {r!r} cannot be mapped to any reviewed relation"
                    )
                canonical_relations.append({
                    "raw_subject_role_id": str(s),
                    "canonical_subject_role_id": s_canon,
                    "raw_relation_text": str(r),
                    "canonical_predicate": mapped_r,
                    "raw_object_role_id": str(o),
                    "canonical_object_role_id": o_canon,
                })

            # Canonicalize interaction groups losslessly into OperationGroup objects
            canonical_operation_groups: list[dict[str, Any]] = []
            for grp in self.raw_decomposition.get("interaction_groups", []):
                gid = grp.get("id")
                t_role = grp.get("tool_role")
                tgt_role = grp.get("target_role")
                if t_role not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(f"Interaction group tool role {t_role!r} not declared")
                if tgt_role not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(f"Interaction group target role {tgt_role!r} not declared")
                t_canon = raw_id_to_canon[t_role]
                tgt_canon = raw_id_to_canon[tgt_role]

                ctx_role = grp.get("context_role")
                ctx_canon = raw_id_to_canon[ctx_role] if (ctx_role and ctx_role in raw_id_to_canon) else None
                if ctx_role and ctx_canon is None:
                    raise MalformedVLMSpecificationError(f"Interaction group context role {ctx_role!r} not declared")

                req_rels: list[str] = []
                for r in grp.get("required_relations", []):
                    mapped_r = map_living_room_relation(r, self.binary_relation_aliases)
                    if mapped_r is None:
                        raise UnmappedFunctionalConceptError(f"Unmapped interaction group relation {r!r}")
                    req_rels.append(mapped_r)

                ctx_rels: list[str] = []
                for r in grp.get("context_relations", []):
                    mapped_r = map_living_room_relation(r, self.binary_relation_aliases)
                    if mapped_r is None:
                        raise UnmappedFunctionalConceptError(f"Unmapped interaction group context relation {r!r}")
                    ctx_rels.append(mapped_r)

                usage_policy = grp.get("usage_policy", "DEDICATED_PER_TARGET")
                canonical_operation_groups.append({
                    "id": str(gid),
                    "function": str(grp.get("function", "")),
                    "tool_role": t_canon,
                    "target_role": tgt_canon,
                    "required_target_count": int(grp.get("required_target_count", 1)),
                    "usage_policy": usage_policy,
                    "required_relations": tuple(req_rels),
                    "context_role": ctx_canon,
                    "context_relations": tuple(ctx_rels),
                    "distinct_within_group": True,
                    "same_tool_must_cover_all_targets": False,
                })

            self.normalized_relations = canonical_relations
            self.normalized_operation_groups = canonical_operation_groups
            self.vlm_derived_role_vocabulary = tuple(dict.fromkeys(vlm_role_vocab))
            self.task_explicit_context_vocabulary = (
                "armchair", "chair", "sofa", "remote control", "tv remote",
                "cup", "saucer", "cup saucer set"
            )
            self.normalization_issues = []
            self.ready_for_grounding = True
            self.normalized_task = None

        self.normalized_requirements = normalized_records

        vocabulary_entries = []
        aliases = self._vocabulary_aliases()
        ordered_categories = sorted(
            aliases,
            key=lambda category: (
                category_rank.get(category, 10_000),
                list(aliases).index(category),
            ),
        )
        for rank, canonical in enumerate(ordered_categories, 1):
            vocabulary_entries.append(
                {
                    "canonical_label": canonical,
                    "aliases": list(aliases[canonical]),
                    "rank": rank,
                    "source": (
                        "VLM_MATCHED_ROLE_CATEGORY"
                        if canonical in category_rank
                        else "REVIEWED_NEGATIVE_OR_CONTEXT_CATEGORY"
                    ),
                }
            )
        self.ranked_detector_vocabulary = vocabulary_entries
        if require_reviewed_contract and not self.ready_for_grounding:
            raise ValueError(
                "VLM output is not ready for grounding: "
                + "; ".join(self.normalization_issues)
            )
        return self.result()

    def result(self) -> dict[str, Any]:
        if self.raw_decomposition is None:
            raise RuntimeError("generate() must be called before result()")
        return {
            "schema_version": 1,
            "environment": self.environment,
            "scope": "VLM_REQUIREMENT_DECOMPOSITION_ONLY",
            "task_instruction": self.task_instruction,
            "initial_observation_images": self.fm_adapter.last_observation_images,
            "raw_vlm_decomposition": self.raw_decomposition,
            "raw_vlm_requirement_response": deepcopy(
                self.fm_adapter.last_raw_requirement_response
            ),
            "raw_vlm_inspection_response": deepcopy(
                self.fm_adapter.last_raw_inspection_response
            ),
            "normalized_requirements": self.normalized_requirements,
            "normalized_relations": self.normalized_relations,
            "normalized_operation_groups": getattr(self, "normalized_operation_groups", []),
            "normalized_task_contract": self.normalized_task,
            "ready_for_grounding": self.ready_for_grounding,
            "reviewed_ontology_audit": {
                "status": "PASS" if self.ready_for_grounding else "REVIEW_REQUIRED",
                "issues": list(self.normalization_issues),
                "note": (
                    "The reviewed ontology was used only after the VLM response; "
                    "it was not included in the model prompt."
                ),
            },
            "ranked_detector_vocabulary": self.ranked_detector_vocabulary,
            "vlm_inspection_policy": deepcopy(self.inspection_policy),
            "fm_calls": self.fm_adapter.metrics.total_calls,
            "observation_search_started": False,
            "semantic_grounding_started": False,
            "allocation_started": False,
            "geometry_verification_started": False,
            "planning_started": False,
            "execution_started": False,
        }
