"""One-shot VLM functional specification provider."""

from __future__ import annotations

from pathlib import Path

from .models import FunctionalRole, FunctionalSpecification
from .spec_provider import FunctionalSpecProvider


class VLMSpecProvider(FunctionalSpecProvider):
    def provide(
        self,
        domain: str,
        task_instruction: str,
        observation_images: list[Path] | None = None,
    ) -> FunctionalSpecification:
        if domain != "workshop":
            if domain == "kitchen":
                return self._kitchen(task_instruction, observation_images or [])
            if domain == "living_room":
                return self._living_room(task_instruction, observation_images or [])
            raise NotImplementedError(f"VLM specification adapter is not implemented for {domain}")
        from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider

        provider = FMRequirementProvider()
        requirements = tuple(provider.get_requirements(
            task_instruction, observation_images=observation_images or []
        ))
        ranking = tuple(getattr(provider, "region_ranking", ()) or (
            "LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"
        ))
        return FunctionalSpecification(
            domain=domain,
            task_instruction=task_instruction,
            roles=tuple(
                FunctionalRole(
                    name=requirement.function_name,
                    semantic_categories=tuple(requirement.accepted_categories),
                    unary_properties=tuple(requirement.geometric_constraints),
                    required_relations=tuple(requirement.required_relations),
                )
                for requirement in requirements
            ),
            detector_vocabulary=tuple(provider.get_detector_prompts()),
            candidate_regions=ranking,
            region_ranking=ranking,
            source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=requirements,
            metadata={
                "detector_label_to_canonical": provider.get_detector_label_to_canonical_map(),
                "alias_to_canonical": provider.get_alias_to_canonical_map(),
                "raw_decomposition": provider.raw_decomposition,
            },
        )

    @staticmethod
    def _kitchen(task_instruction: str, observation_images: list[Path]) -> FunctionalSpecification:
        from mujoco_scenes.kitchen_vlm_functional_graph import (
            KITCHEN_OBSERVABLE_REGIONS, compile_vlm_functional_graph,
        )
        from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter

        adapter = FMAdapter()
        raw = adapter.generate_kitchen_functional_graph(
            task_instruction,
            KITCHEN_OBSERVABLE_REGIONS,
            observation_images=observation_images,
        )
        contract, vocabularies, trace = compile_vlm_functional_graph(
            raw,
            task_instruction=task_instruction,
            observable_regions=tuple(KITCHEN_OBSERVABLE_REGIONS),
        )
        relations_by_role: dict[str, list[str]] = {}
        for relation in contract.get("relations", []):
            relations_by_role.setdefault(relation["subject_role"], []).append(relation["predicate"])
        roles = []
        for name, role in contract["roles"].items():
            card = role.get("binding_cardinality", {})
            roles.append(FunctionalRole(
                name=name,
                count=int(role.get("count", card.get("minimum_distinct_physical_objects", 1))),
                semantic_categories=tuple(
                    item["canonical_label"] for item in role.get("semantic_preferences", [])
                ),
                unary_properties=tuple(
                    item.get("predicate", item.get("property"))
                    for item in role.get("unary_geometry", [])
                ),
                required_relations=tuple(relations_by_role.get(name, [])),
            ))
        order = tuple(raw["inspection_order"])
        object_vocab = vocabularies["object"]
        prompts = tuple(dict.fromkeys(
            phrase for phrases in object_vocab.values() for phrase in phrases
        ))
        return FunctionalSpecification(
            domain="kitchen", task_instruction=task_instruction,
            roles=tuple(roles), detector_vocabulary=prompts,
            candidate_regions=tuple(KITCHEN_OBSERVABLE_REGIONS),
            region_ranking=order, source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=(contract,),
            metadata={"object_vocabulary": object_vocab, "raw_decomposition": raw, "normalization_trace": trace},
        )

    @staticmethod
    def _living_room(task_instruction: str, observation_images: list[Path]) -> FunctionalSpecification:
        from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider

        provider = EnvironmentVLMRequirementProvider("living_room")
        result = provider.generate(
            task_instruction,
            observation_images=observation_images,
            require_reviewed_contract=True,
        )
        requirements = result["normalized_requirements"]
        roles = tuple(FunctionalRole(
            name=row["function"], count=int(row["reviewed_required_count"]),
            semantic_categories=tuple(row["accepted_categories"]),
            unary_properties=tuple(
                prop for prop in row["required_properties"] if prop == "PLANAR_SUPPORT"
            ),
            required_relations=tuple(row["required_properties"]),
            distinct=row["function"] == "PERSONAL_CUP_SAUCER_REGION",
            shared=row["function"] == "SHARED_REMOTE_REGION",
        ) for row in requirements)
        vocabulary = tuple(
            row["detector_label"] for row in result["ranked_detector_vocabulary"]
        )
        return FunctionalSpecification(
            domain="living_room", task_instruction=task_instruction,
            roles=roles, detector_vocabulary=vocabulary,
            candidate_regions=(), region_ranking=(), source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=(result["normalized_task_contract"],),
            metadata={
                "semantic_vocabulary_path": str(provider.vocabulary_path),
                "raw_decomposition": result["raw_vlm_decomposition"],
                "normalization_audit": result["reviewed_ontology_audit"],
            },
        )
