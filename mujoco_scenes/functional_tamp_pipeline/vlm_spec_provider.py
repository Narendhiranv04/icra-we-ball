"""One-shot VLM functional specification provider."""

from __future__ import annotations

from pathlib import Path

from .models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    NumericConstraint,
    OperationGroup,
)
from .spec_provider import FunctionalSpecProvider


class VLMSpecProvider(FunctionalSpecProvider):
    def provide(
        self,
        domain: str,
        task_instruction: str,
        observation_images: list[Path] | None = None,
    ) -> FunctionalRequirementGraph:
        if domain == "workshop":
            return self._workshop(task_instruction, observation_images or [])
        if domain == "kitchen":
            return self._kitchen(task_instruction, observation_images or [])
        if domain == "living_room":
            return self._living_room(task_instruction, observation_images or [])
        raise NotImplementedError(f"VLM specification adapter is not implemented for {domain}")

    @staticmethod
    def _workshop(task_instruction: str, observation_images: list[Path]) -> FunctionalRequirementGraph:
        from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider

        provider = FMRequirementProvider()
        requirements = tuple(provider.get_requirements(
            task_instruction, observation_images=observation_images
        ))
        ranking = tuple(provider.region_ranking)
        if not ranking:
            raise ValueError("VLM_SPEC_FAILED: VLM failed to provide region ranking for Workshop")

        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        for requirement in requirements:
            func_name = requirement.function_name
            role = FunctionalRole(
                name=func_name,
                entity_kind="OBJECT",
                count=1,
                semantic_categories=tuple(requirement.accepted_categories),
                unary_predicates=(func_name,),
                binding_policy="DISTINCT",
                verification_mode="SEMANTIC_AND_GEOMETRIC",
                description=requirement.description,
                semantic_hints=tuple(requirement.semantic_hints),
            )
            nodes[func_name] = role

            for rel_name in requirement.required_relations:
                target_role = "CAN_FASTEN" if func_name == "CAN_DRIVE_SCREW" else "CAN_DRIVE_SCREW"
                relations.append(FunctionalRelation(
                    subject_role=func_name,
                    predicate=rel_name,
                    object_role=target_role,
                    expected=True,
                ))

        return FunctionalRequirementGraph(
            domain="workshop",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
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
    def _kitchen(task_instruction: str, observation_images: list[Path]) -> FunctionalRequirementGraph:
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

        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        for rel in contract.get("relations", []):
            relations.append(FunctionalRelation(
                subject_role=rel["subject_role"],
                predicate=rel["predicate"],
                object_role=rel["object_role"],
                expected=bool(rel.get("expected", True)),
            ))

        for name, role in contract["roles"].items():
            card = role.get("binding_cardinality", {})
            categories = tuple(
                item["canonical_label"] for item in role.get("semantic_preferences", [])
            )
            unary_preds = []
            numeric_reqs = []
            for item in role.get("unary_geometry", []):
                if "predicate" in item:
                    unary_preds.append(str(item["predicate"]))
                elif "property" in item:
                    numeric_reqs.append(NumericConstraint(
                        property_name=str(item["property"]),
                        operator=str(item.get("operator", ">=")),
                        threshold=float(item["value"]),
                        unit=str(item.get("unit", "m")),
                    ))

            binding = "REUSABLE" if card.get("preferred") == "minimize_distinct" else "DISTINCT"
            nodes[name] = FunctionalRole(
                name=name,
                entity_kind="OBJECT",
                count=int(role.get("count", card.get("minimum_distinct_physical_objects", 1))),
                semantic_categories=categories,
                unary_predicates=tuple(unary_preds),
                numeric_constraints=tuple(numeric_reqs),
                binding_policy=binding,
                verification_mode=str(role.get("vlm_verification_mode", "SEMANTIC_AND_GEOMETRIC")),
            )

        operation_groups: list[OperationGroup] = []
        for gid, grp in contract.get("operation_groups", {}).items():
            policy = grp.get("usage_policy", {})
            mode_str = str(policy.get("mode", "sequential_reuse_allowed")).upper()
            if mode_str == "SEQUENTIAL_REUSE_ALLOWED":
                usage_policy = "SEQUENTIAL_REUSE_ALLOWED"
            else:
                usage_policy = "DEDICATED_PER_TARGET"
            operation_groups.append(OperationGroup(
                id=gid,
                function=str(grp["function"]),
                tool_role=str(grp["tool_role"]),
                target_role=str(grp["target_role"]),
                required_target_count=int(grp["required_target_count"]),
                usage_policy=usage_policy,
                required_relations=tuple(map(str, grp.get("relations", ()))),
            ))

        order = tuple(raw["inspection_order"])
        object_vocab = vocabularies["object"]
        prompts = tuple(dict.fromkeys(
            phrase for phrases in object_vocab.values() for phrase in phrases
        ))
        return FunctionalRequirementGraph(
            domain="kitchen",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            operation_groups=tuple(operation_groups),
            cross_group_reuse_allowed=bool(contract.get("cross_group_reuse", {}).get("allowed", True)),
            detector_vocabulary=prompts,
            candidate_regions=tuple(KITCHEN_OBSERVABLE_REGIONS),
            region_ranking=order,
            source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=(contract,),
            metadata={
                "object_vocabulary": object_vocab,
                "raw_decomposition": raw,
                "normalization_trace": trace,
            },
        )

    @staticmethod
    def _living_room(task_instruction: str, observation_images: list[Path]) -> FunctionalRequirementGraph:
        from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider

        provider = EnvironmentVLMRequirementProvider("living_room")
        result = provider.generate(
            task_instruction,
            observation_images=observation_images,
            require_reviewed_contract=False,
        )
        requirements = result["normalized_requirements"]
        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        for row in requirements:
            func_id = row["function"]
            binding = "SHARED" if func_id == "SHARED_REMOTE_REGION" else "DISTINCT"
            nodes[func_id] = FunctionalRole(
                name=func_id,
                entity_kind="REGION",
                count=int(row["vlm_required_count"]),
                semantic_categories=tuple(row["accepted_categories"]),
                unary_predicates=tuple(
                    prop for prop in row["required_properties"] if prop == "PLANAR_SUPPORT"
                ),
                binding_policy=binding,
                verification_mode="SEMANTIC_AND_GEOMETRIC",
                description=row.get("description", ""),
                semantic_hints=tuple(row.get("semantic_hints", ())),
            )
            for rel in row["required_properties"]:
                target_role = "seating" if "SEAT" in rel else "payload"
                relations.append(FunctionalRelation(
                    subject_role=func_id,
                    predicate=rel,
                    object_role=target_role,
                    expected=True,
                ))

        vocabulary = tuple(
            row["detector_label"] for row in result["ranked_detector_vocabulary"]
        )
        return FunctionalRequirementGraph(
            domain="living_room",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            detector_vocabulary=vocabulary,
            candidate_regions=(),
            region_ranking=(),
            source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=(result["normalized_task_contract"],),
            metadata={
                "semantic_vocabulary_path": str(provider.vocabulary_path),
                "raw_decomposition": result["raw_vlm_decomposition"],
                "normalization_audit": result["reviewed_ontology_audit"],
            },
        )

