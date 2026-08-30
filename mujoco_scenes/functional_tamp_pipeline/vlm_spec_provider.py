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

VLM_CANONICALIZATION_VERSION = "phase3_6a5_v1"


class VLMSpecProvider(FunctionalSpecProvider):
    def provide(
        self,
        domain: str,
        task_instruction: str,
        observation_images: list[Path] | None = None,
    ) -> FunctionalRequirementGraph:
        if domain == "workshop":
            graph = self._workshop(task_instruction, observation_images or [])
        elif domain == "kitchen":
            graph = self._kitchen(task_instruction, observation_images or [])
        elif domain == "living_room":
            graph = self._living_room(task_instruction, observation_images or [])
        else:
            raise NotImplementedError(f"VLM specification adapter is not implemented for {domain}")
        graph.validate()
        return graph

    @staticmethod
    def _workshop(
        task_instruction: str,
        observation_images: list[Path],
        provider: FMRequirementProvider | None = None,
    ) -> FunctionalRequirementGraph:
        from mujoco_scenes.workshop_phase1.requirements import (
            FMRequirementProvider, WORKSHOP_SEARCH_REGIONS,
        )

        if provider is None:
            provider = FMRequirementProvider()
        provider.get_requirements(
            task_instruction, observation_images=observation_images
        )

        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        for role in provider.normalized_roles:
            role_id = role.canonical_role_id
            nodes[role_id] = FunctionalRole(
                name=role_id,
                entity_kind=role.entity_kind,
                count=role.required_count,
                semantic_categories=role.run_local_categories,
                unary_predicates=role.unary_predicates,
                binding_policy=role.binding_policy,
                verification_mode=(
                    "GEOMETRIC_ONLY"
                    if role.entity_kind == "FIXED_TARGET"
                    else ("SEMANTIC_AND_GEOMETRIC" if role.unary_predicates else "SEMANTIC_ONLY")
                ),
                description=role.description,
                semantic_hints=role.semantic_hints,
            )

        for rel in provider.normalized_relations:
            relations.append(FunctionalRelation(
                subject_role=rel.canonical_subject_role_id,
                predicate=rel.canonical_predicate,
                object_role=rel.canonical_object_role_id,
                expected=True,
            ))

        detector_vocab = tuple(dict.fromkeys(provider.vlm_derived_detector_prompts))

        return FunctionalRequirementGraph(
            domain="workshop",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            detector_vocabulary=detector_vocab,
            candidate_regions=tuple(provider.candidate_regions),
            region_ranking=tuple(provider.region_ranking),
            source="VLM_CANONICAL_G_F",
            raw_requirements=tuple(provider._requirements or []),
            metadata={
                "schema_version": 2,
                "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
                "transformation": "LOSSLESS_CANONICAL_G_F_CONSTRUCTION",
                "raw_roles_count": len(provider.normalized_roles),
                "raw_relations_count": len(provider.normalized_relations),
                "vlm_derived_detector_prompts": list(provider.vlm_derived_detector_prompts),
                "evaluation_negative_control_prompts": list(provider.evaluation_negative_control_prompts),
                "detector_label_to_canonical": provider.get_detector_label_to_canonical_map(),
                "alias_to_canonical": provider.get_alias_to_canonical_map(),
                "raw_decomposition": provider.raw_decomposition,
                "transformation_trace": getattr(provider, "transformation_trace", []),
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

            raw_entity_kind = str(role.get("entity_kind", "OBJECT"))
            if raw_entity_kind not in {"OBJECT", "REGION", "FIXED_TARGET"}:
                raise ValueError(f"Unsupported entity_kind {raw_entity_kind!r} for kitchen role {name!r}")

            binding = str(role.get("vlm_binding_policy") or "DISTINCT")
            raw_count = int(role["count"])

            nodes[name] = FunctionalRole(
                name=name,
                entity_kind=raw_entity_kind,
                count=raw_count,
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
            distinct_within = bool(policy.get("distinct_within_group", policy.get("distinct_tools_within_group", usage_policy == "DEDICATED_PER_TARGET")))
            same_tool_covers_all = bool(policy.get("same_tool_must_cover_all_targets", False))
            selection_pref = policy.get("selection_preference")
            operation_groups.append(OperationGroup(
                id=gid,
                function=str(grp["function"]),
                tool_role=str(grp["tool_role"]),
                target_role=str(grp["target_role"]),
                required_target_count=int(grp["required_target_count"]),
                usage_policy=usage_policy,
                required_relations=tuple(map(str, grp.get("relations", ()))),
                distinct_within_group=distinct_within,
                same_tool_must_cover_all_targets=same_tool_covers_all,
                selection_preference=str(selection_pref) if selection_pref is not None else None,
            ))

        resolved_order = tuple(trace.get("inspection_order", ()))
        resolved_regions = tuple(trace.get("candidate_regions", ()))
        object_vocab = vocabularies["object"].get("canonical_labels", vocabularies["object"])
        prompts = tuple(dict.fromkeys(
            phrase for phrases in object_vocab.values() if isinstance(phrases, (list, tuple)) for phrase in phrases
        ))
        return FunctionalRequirementGraph(
            domain="kitchen",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            operation_groups=tuple(operation_groups),
            cross_group_reuse_allowed=bool(contract.get("cross_group_reuse", {}).get("allowed", False)),
            detector_vocabulary=prompts,
            candidate_regions=resolved_regions,
            region_ranking=resolved_order,
            source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=(contract,),
            metadata={
                "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
                "object_vocabulary": object_vocab,
                "raw_decomposition": raw,
                "normalization_trace": trace,
                "symbolic_task": contract.get("symbolic_task", {}),
            },
        )

    @staticmethod
    def _living_room(
        task_instruction: str,
        observation_images: list[Path],
        provider: EnvironmentVLMRequirementProvider | None = None,
    ) -> FunctionalRequirementGraph:
        from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider

        if provider is None:
            provider = EnvironmentVLMRequirementProvider("living_room")
        result = provider.generate_canonical(
            task_instruction,
            observation_images=observation_images,
        )
        requirements = result["normalized_requirements"]
        canonical_relations = result.get("normalized_relations", [])
        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        for row in requirements:
            func_id = row["function"]
            binding = row["binding_policy"]
            count = int(row["vlm_required_count"])
            entity_kind = row["entity_kind"]
            cats = tuple(row["accepted_categories"])
            unary = tuple(
                prop for prop in row.get("required_properties", []) if prop == "PLANAR_SUPPORT"
            )
            nodes[func_id] = FunctionalRole(
                name=func_id,
                entity_kind=entity_kind,
                count=count,
                semantic_categories=cats,
                unary_predicates=unary,
                binding_policy=binding,
                verification_mode="SEMANTIC_AND_GEOMETRIC" if unary else "SEMANTIC_ONLY",
                description=row.get("description", ""),
                semantic_hints=tuple(row.get("semantic_hints", ())),
            )

        for rel_item in canonical_relations:
            relations.append(FunctionalRelation(
                subject_role=rel_item["canonical_subject_role_id"],
                predicate=rel_item["canonical_predicate"],
                object_role=rel_item["canonical_object_role_id"],
                expected=True,
            ))

        vlm_prompts = list(provider.vlm_derived_role_vocabulary)
        context_prompts = list(provider.task_explicit_context_vocabulary)
        vocabulary = tuple(dict.fromkeys(vlm_prompts + context_prompts))

        return FunctionalRequirementGraph(
            domain="living_room",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            operation_groups=(),
            cross_group_reuse_allowed=False,
            detector_vocabulary=vocabulary,
            candidate_regions=(),
            region_ranking=(),
            source="VLM_CANONICAL_G_F",
            raw_requirements=(result.get("normalized_task_contract") or result["raw_vlm_decomposition"],),
            metadata={
                "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
                "semantic_vocabulary_path": str(provider.vocabulary_path),
                "vlm_derived_role_vocabulary": vlm_prompts,
                "task_explicit_context_vocabulary": context_prompts,
                "raw_decomposition": result["raw_vlm_decomposition"],
                "normalization_audit": result["reviewed_ontology_audit"],
            },
        )


