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
        self.relation_aliases = normalization.get("relation_aliases", {})
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
                properties.extend(relations_by_subject.get(role_id, []))
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
        aliases = self._category_map()
        if normalized in aliases:
            return aliases[normalized]
        matches = {
            canonical
            for alias, canonical in aliases.items()
            if f" {alias} " in f" {normalized} "
            or f" {normalized} " in f" {alias} "
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def _map_properties(self, values: list[str]) -> set[str]:
        mapped: set[str] = set()
        for value in values:
            for predicate, aliases in self.relation_aliases.items():
                if any(_phrase_score(value, alias) >= 0.75 for alias in aliases):
                    mapped.add(predicate)
        return mapped

    def _detector_label(self, canonical: str) -> str:
        aliases = self._vocabulary_aliases()[canonical]
        return aliases[0] if aliases else canonical.replace("_", " ")

    def _candidate_categories(self, raw: dict[str, Any]) -> list[str]:
        categories: list[str] = []
        for candidate in raw["candidate_objects"]:
            canonical = self._map_category(candidate["label"])
            if canonical is not None and canonical not in categories:
                categories.append(canonical)
        return categories

    def _role_match_score(
        self, raw: dict[str, Any], spec: dict[str, Any]
    ) -> float:
        if raw["entity_kind"] != spec["entity_kind"]:
            return 0.0
        function_text = f"{raw['function']} {raw['description']}"
        language_score = max(
            (_phrase_score(function_text, alias)
             for alias in spec["function_aliases"]),
            default=0.0,
        )
        visible_categories = set(self._candidate_categories(raw))
        category_score = (
            len(visible_categories & set(spec["categories"]))
            / max(1, len(visible_categories))
        )
        return language_score + 0.20 * category_score

    def _assign_roles(
        self,
        raw_requirements: list[dict[str, Any]],
        specs: list[dict[str, Any]],
    ) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[str]]:
        issues: list[str] = []
        matched: dict[str, list[dict[str, Any]]] = {}
        for raw in raw_requirements:
            scores = [self._role_match_score(raw, spec) for spec in specs]
            best = max(scores, default=0.0)
            winners = [
                index for index, score in enumerate(scores)
                if score == best and score >= 0.60
            ]
            if len(winners) != 1:
                issues.append(
                    f"raw role {raw['id']!r} is unmapped or ambiguous; "
                    f"reviewed-role scores={dict(zip((s['role_id'] for s in specs), scores))}"
                )
                continue
            spec = specs[winners[0]]
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

    def generate(
        self,
        instruction: str | None = None,
        *,
        observation_images: list[str | Path] | None = None,
        require_reviewed_contract: bool = False,
        include_inspection_policy: bool = False,
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
        specs = self._role_specs()
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
            raise ValueError(
                f"VLM marked {self.environment} unsupported: "
                f"{document.get('unsupported_reason', 'no reason')}"
            )
        raw_requirements = document["functional_requirements"]
        normalized_records = []
        category_rank: dict[str, int] = {}
        assigned, issues = self._assign_roles(raw_requirements, specs)
        for spec, raw_group in assigned:
            raw = {
                "id": raw_group[0]["id"],
                "entity_kind": spec["entity_kind"],
                "function": " / ".join(item["function"] for item in raw_group),
                "description": " ".join(item["description"] for item in raw_group),
                "required_count": sum(item["required_count"] for item in raw_group),
                "candidate_objects": [
                    candidate
                    for item in raw_group
                    for candidate in item["candidate_objects"]
                ],
                "required_properties": list(dict.fromkeys(
                    property_text
                    for item in raw_group
                    for property_text in item["required_properties"]
                )),
            }
            categories = self._candidate_categories(raw)
            for canonical in categories:
                if canonical in spec["categories"]:
                    category_rank.setdefault(canonical, len(category_rank) + 1)
            properties = self._map_properties(raw["required_properties"])
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
                    "visible_candidate_objects": deepcopy(raw["candidate_objects"]),
                    "mapped_visible_candidates": [
                        {
                            **deepcopy(candidate),
                            "canonical_category": self._map_category(candidate["label"]),
                            "accepted_for_role": (
                                self._map_category(candidate["label"])
                                in spec["categories"]
                            ),
                        }
                        for candidate in raw["candidate_objects"]
                    ],
                    "required_properties": list(spec["properties"]),
                    "semantic_hints": [
                        candidate["label"] for candidate in raw["candidate_objects"]
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
        for rank, canonical in enumerate(ordered_categories, start=1):
            vocabulary_entries.append(
                {
                    "canonical_label": canonical,
                    "detector_label": self._detector_label(canonical),
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
