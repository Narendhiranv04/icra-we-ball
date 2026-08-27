"""Reviewed static functional specifications with no solution knowledge."""

from __future__ import annotations

from pathlib import Path
import yaml

from .models import FunctionalRole, FunctionalSpecification
from .spec_provider import FunctionalSpecProvider


class GTSpecProvider(FunctionalSpecProvider):
    def provide(
        self,
        domain: str,
        task_instruction: str,
        observation_images: list[Path] | None = None,
    ) -> FunctionalSpecification:
        if domain == "workshop":
            return self._workshop(task_instruction)
        if domain == "kitchen":
            return self._kitchen(task_instruction)
        if domain == "living_room":
            return self._living_room(task_instruction)
        raise NotImplementedError(f"GT specification adapter is not implemented for {domain}")

    @staticmethod
    def _workshop(task_instruction: str) -> FunctionalSpecification:
        from mujoco_scenes.workshop_phase1.requirements import ManualWorkshopFMContract

        provider = ManualWorkshopFMContract()
        requirements = tuple(provider.get_requirements(task_instruction))
        roles = tuple(
            FunctionalRole(
                name=requirement.function_name,
                semantic_categories=tuple(requirement.accepted_categories),
                unary_properties=tuple(requirement.geometric_constraints),
                required_relations=tuple(requirement.required_relations),
            )
            for requirement in requirements
        )
        vocabulary = (
            "screwdriver", "power drill", "screw", "wooden hammer",
            "Phillips screwdriver", "cordless power drill", "Phillips screw",
        )
        ranking = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
        return FunctionalSpecification(
            domain="workshop",
            task_instruction=task_instruction,
            roles=roles,
            detector_vocabulary=vocabulary,
            candidate_regions=ranking,
            region_ranking=ranking,
            source="GT_FUNCTIONAL_SPEC_ONLY",
            raw_requirements=requirements,
            metadata={
                "detector_label_to_canonical": provider.get_detector_label_to_canonical_map(),
                "alias_to_canonical": provider.get_alias_to_canonical_map(),
            },
        )

    @staticmethod
    def _kitchen(task_instruction: str) -> FunctionalSpecification:
        root = Path(__file__).resolve().parents[1]
        contract_path = root / "configs" / "s1_integrated_kitchen_object_function.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        roles = []
        vocabulary = []
        relations_by_role: dict[str, list[str]] = {}
        for relation in contract.get("relations", []):
            relations_by_role.setdefault(relation["subject_role"], []).append(
                relation["predicate"]
            )
        for name, raw in contract["roles"].items():
            cardinality = raw.get("binding_cardinality", {})
            count = int(raw.get("count", cardinality.get("minimum_distinct_physical_objects", 1)))
            categories = tuple(
                item["canonical_label"] for item in raw.get("semantic_preferences", [])
            )
            for item in raw.get("semantic_preferences", []):
                vocabulary.extend([item["canonical_label"], *item.get("detector_aliases", [])])
            roles.append(FunctionalRole(
                name=name,
                count=count,
                semantic_categories=categories,
                unary_properties=tuple(
                    item["predicate"] for item in raw.get("unary_geometry", [])
                ),
                required_relations=tuple(relations_by_role.get(name, [])),
                distinct=count > 1,
                reusable=cardinality.get("preferred") == "minimize_distinct",
            ))
        for name, raw in contract["symbolic_task"]["source_roles"].items():
            labels = tuple(raw["accepted_semantic_labels"])
            vocabulary.extend(labels)
            roles.append(FunctionalRole(name=name, count=int(raw["count"]), semantic_categories=labels))
        regions = ("D1", "D2", "C2", "B1", "C1")
        return FunctionalSpecification(
            domain="kitchen",
            task_instruction=task_instruction or contract["goal_instruction"],
            roles=tuple(roles),
            detector_vocabulary=tuple(dict.fromkeys(vocabulary)),
            candidate_regions=regions,
            region_ranking=regions,
            source="GT_FUNCTIONAL_SPEC_ONLY",
            raw_requirements=(contract,),
            metadata={
                "semantic_vocabulary_path": str(root / "configs" / "semantic_vocabulary.yaml"),
                "contract_path": str(contract_path),
            },
        )

    @staticmethod
    def _living_room(task_instruction: str) -> FunctionalSpecification:
        root = Path(__file__).resolve().parents[1]
        contract_path = root / "configs" / "l2_integrated_region_function_task.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        roles = []
        vocabulary = []
        for group in contract["function_groups"].values():
            role_name = group["region_role"]
            semantic = contract["semantic_requirements"]["region_roles"][role_name]
            categories = tuple(semantic["accepted_categories"])
            vocabulary.extend(categories)
            roles.append(FunctionalRole(
                name=group["function_id"],
                count=int(group.get("required_target_count", 1)),
                semantic_categories=categories,
                unary_properties=("PLANAR_SUPPORT",),
                required_relations=tuple(group["required_relations"]),
                distinct=group["usage_policy"] == "DEDICATED_REGION_PER_TARGET",
                shared=group["usage_policy"] == "SHARED_REGION_REQUIRED",
            ))
        for labels in contract["semantic_requirements"]["payload_roles"].values():
            vocabulary.extend(labels)
        vocabulary.extend(contract["semantic_requirements"]["seating_categories"])
        return FunctionalSpecification(
            domain="living_room",
            task_instruction=task_instruction or contract["natural_language_goal"],
            roles=tuple(roles), detector_vocabulary=tuple(dict.fromkeys(vocabulary)),
            candidate_regions=(), region_ranking=(), source="GT_FUNCTIONAL_SPEC_ONLY",
            raw_requirements=(contract,),
            metadata={
                "contract_path": str(contract_path),
                "semantic_vocabulary_path": str(
                    root / "configs" / "l2_integrated_region_function_semantic_vocabulary.yaml"
                ),
            },
        )
