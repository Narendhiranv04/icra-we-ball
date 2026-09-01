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
from . import role_semantic_ontology as semantic_ontology
from .role_semantic_ontology import (
    PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION,
    get_system_role_semantic_categories,
)
from .spec_provider import FunctionalSpecProvider

VLM_CANONICALIZATION_VERSION = "phase3_6a7_2_1_v1"


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
        try:
            graph.validate()
            from .task_interface_validator import validate_runtime_gf
            validate_runtime_gf(graph)
        except Exception as err:
            from .errors import MalformedVLMSpecificationError
            raise MalformedVLMSpecificationError(f"MALFORMED_VLM_SPECIFICATION: {err}") from err
        return graph

    @staticmethod
    def _workshop(
        task_instruction: str,
        observation_images: list[Path],
        provider: FMRequirementProvider | None = None,
        adapter: FMAdapter | None = None,
    ) -> FunctionalRequirementGraph:
        from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
        from mujoco_scenes.workshop_phase1.requirements import (
            FMRequirementProvider,
            WORKSHOP_SEARCH_REGIONS,
            WORKSHOP_VLM_CANONICALIZATION_VERSION,
        )

        if provider is None:
            provider = FMRequirementProvider(fm_adapter=adapter)
        elif adapter is not None and getattr(provider, "fm_adapter", None) is None:
            provider.fm_adapter = adapter

        provider.get_requirements(
            task_instruction, observation_images=observation_images
        )

        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        for role in provider.normalized_roles:
            role_id = role.canonical_role_id
            if role.entity_kind == "OBJECT" and role_id in ("driver", "fastener"):
                if not role.run_local_categories:
                    raise MalformedVLMSpecificationError(
                        f"Workshop functional role {role_id!r} must have non-empty candidate_categories"
                    )
            system_cats = semantic_ontology.get_system_role_semantic_categories("workshop", role_id)
            nodes[role_id] = FunctionalRole(
                name=role_id,
                entity_kind=role.entity_kind,
                count=role.required_count,
                semantic_categories=system_cats,
                unary_predicates=role.unary_predicates,
                binding_policy=role.binding_policy,
                verification_mode=(
                    "GEOMETRIC_ONLY"
                    if role.entity_kind == "FIXED_TARGET"
                    else "SEMANTIC_AND_GEOMETRIC"
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

        raw_resp = getattr(provider, "raw_vlm_response", None) or provider.raw_decomposition
        valid_spec = getattr(provider, "validated_vlm_specification", None) or provider.raw_decomposition
        trace = getattr(provider, "canonicalization_trace", {})

        return FunctionalRequirementGraph(
            domain="workshop",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            operation_groups=tuple(provider.normalized_operation_groups),
            detector_vocabulary=detector_vocab,
            candidate_regions=tuple(provider.candidate_regions),
            region_ranking=tuple(provider.region_ranking),
            source="VLM_CANONICAL_G_F",
            raw_requirements=tuple(provider._requirements or []),
            metadata={
                "schema_version": 2,
                "vlm_canonicalization_version": trace.get("vlm_canonicalization_version", WORKSHOP_VLM_CANONICALIZATION_VERSION),
                "role_semantic_ontology_version": PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION,
                "semantic_acceptance_source": "SYSTEM_ROLE_SEMANTIC_ONTOLOGY",
                "detector_vocabulary_source": "VLM_CANDIDATES_PLUS_RELEVANT_SYSTEM_ALIASES",
                "candidate_categories_used_for_role_identity": False,
                "candidate_categories_used_for_grounding_acceptance": False,
                "candidate_categories_used_for_detector_vocabulary": True,
                "transformation": "LOSSLESS_CANONICAL_G_F_CONSTRUCTION",
                "raw_roles_count": trace.get("raw_roles_count", len(provider.normalized_roles)),
                "raw_relations_count": trace.get("raw_relations_count", len(provider.normalized_relations)),
                "raw_operation_groups_count": trace.get("raw_operation_groups_count", len(provider.normalized_operation_groups)),
                "vlm_derived_detector_prompts": list(provider.vlm_derived_detector_prompts),
                "evaluation_negative_control_prompts": list(provider.evaluation_negative_control_prompts),
                "detector_label_to_canonical": provider.get_detector_label_to_canonical_map(),
                "alias_to_canonical": provider.get_alias_to_canonical_map(),
                "raw_vlm_response": raw_resp,
                "validated_vlm_specification": valid_spec,
                "canonicalization_trace": trace,
                "raw_decomposition": provider.raw_decomposition,
                "transformation_trace": trace.get("transformation_trace", []),
            },
        )

    @staticmethod
    def _kitchen(
        task_instruction: str,
        observation_images: list[Path],
        adapter: FMAdapter | None = None,
    ) -> FunctionalRequirementGraph:
        from mujoco_scenes.kitchen_vlm_functional_graph import (
            KITCHEN_OBSERVABLE_REGIONS, compile_vlm_functional_graph,
        )
        from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter

        if adapter is None:
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
            system_cats = semantic_ontology.get_system_role_semantic_categories("kitchen", name)
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
            min_count = role.get("min_count")
            max_count = role.get("max_count")
            preferred = role.get("preference")

            nodes[name] = FunctionalRole(
                name=name,
                entity_kind=raw_entity_kind,
                count=raw_count,
                min_count=min_count,
                max_count=max_count,
                preference=preferred,
                semantic_categories=system_cats,
                unary_predicates=tuple(unary_preds),
                numeric_constraints=tuple(numeric_reqs),
                binding_policy=binding,
                verification_mode=str(role.get("vlm_verification_mode", "SEMANTIC_AND_GEOMETRIC")),
            )

        for name, raw in contract.get("symbolic_task", {}).get("source_roles", {}).items():
            system_cats = semantic_ontology.get_system_role_semantic_categories("kitchen", name)
            nodes[name] = FunctionalRole(
                name=name,
                entity_kind="OBJECT",
                count=int(raw.get("count", 1)),
                semantic_categories=system_cats,
                binding_policy="DISTINCT",
                verification_mode="SEMANTIC_ONLY",
            )

        operation_groups: list[OperationGroup] = []
        for gid, grp in contract.get("operation_groups", {}).items():
            policy = grp.get("usage_policy", {})
            mode_str = str(policy.get("mode", "sequential_reuse_allowed")).upper()
            if mode_str == "SEQUENTIAL_REUSE_ALLOWED":
                usage_policy = "SEQUENTIAL_REUSE_ALLOWED"
                sel_pref = "minimize_distinct_tools"
            else:
                usage_policy = "DEDICATED_PER_TARGET"
                sel_pref = ""
            distinct_within = bool(policy.get("distinct_within_group", policy.get("distinct_tools_within_group", usage_policy == "DEDICATED_PER_TARGET")))
            same_tool_covers_all = bool(policy.get("same_tool_must_cover_all_targets", False))
            if "selection_preference" in policy:
                sel_pref = str(policy["selection_preference"])
            operation_groups.append(OperationGroup(
                id=gid,
                function=str(grp.get("canonical_function") or grp["function"]),
                tool_role=str(grp["tool_role"]),
                target_role=str(grp["target_role"]),
                required_target_count=int(grp["required_target_count"]),
                usage_policy=usage_policy,
                required_relations=tuple(map(str, grp.get("relations", ()))),
                distinct_within_group=distinct_within,
                same_tool_must_cover_all_targets=same_tool_covers_all,
                selection_preference=sel_pref,
            ))

        resolved_order = tuple(trace.get("inspection_order", ()))
        resolved_regions = tuple(trace.get("candidate_regions", ()))
        object_vocab = vocabularies["object"].get("canonical_labels", vocabularies["object"])
        prompts = tuple(dict.fromkeys(
            phrase for phrases in object_vocab.values() if isinstance(phrases, (list, tuple)) for phrase in phrases
        ))
        raw_resp = getattr(adapter, "last_raw_kitchen_graph_response", None) or getattr(adapter, "last_raw_requirement_response", None) or raw
        valid_spec = getattr(adapter, "last_validated_kitchen_graph_response", None) or raw
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
                "vlm_canonicalization_version": trace.get("vlm_canonicalization_version", VLM_CANONICALIZATION_VERSION),
                "role_semantic_ontology_version": PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION,
                "semantic_acceptance_source": "SYSTEM_ROLE_SEMANTIC_ONTOLOGY",
                "detector_vocabulary_source": "VLM_CANDIDATES_PLUS_RELEVANT_SYSTEM_ALIASES",
                "candidate_categories_used_for_role_identity": False,
                "candidate_categories_used_for_grounding_acceptance": False,
                "candidate_categories_used_for_detector_vocabulary": True,
                "object_vocabulary": object_vocab,
                "raw_vlm_response": raw_resp,
                "validated_vlm_specification": valid_spec,
                "canonicalization_trace": trace,
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
        adapter: FMAdapter | None = None,
    ) -> FunctionalRequirementGraph:
        from mujoco_scenes.environment_vlm_requirements import (
            EnvironmentVLMRequirementProvider,
            LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
        )

        if provider is None:
            provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
        elif adapter is not None and getattr(provider, "fm_adapter", None) is None:
            provider.fm_adapter = adapter

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
            system_cats = semantic_ontology.get_system_role_semantic_categories("living_room", func_id)
            unary = tuple(row.get("required_properties", []))
            nodes[func_id] = FunctionalRole(
                name=func_id,
                entity_kind=entity_kind,
                count=count,
                semantic_categories=system_cats,
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

        operation_groups: list[OperationGroup] = []
        for og_data in result.get("normalized_operation_groups", []):
            operation_groups.append(OperationGroup(
                id=str(og_data["id"]),
                function=str(og_data["function"]),
                tool_role=str(og_data["tool_role"]),
                target_role=str(og_data["target_role"]),
                required_target_count=int(og_data["required_target_count"]),
                usage_policy=str(og_data["usage_policy"]),
                required_relations=tuple(og_data.get("required_relations", ())),
                context_role=str(og_data["context_role"]) if og_data.get("context_role") else None,
                context_relations=tuple(og_data.get("context_relations", ())),
                distinct_within_group=bool(og_data.get("distinct_within_group", True)),
                same_tool_must_cover_all_targets=bool(og_data.get("same_tool_must_cover_all_targets", False)),
            ))

        vlm_prompts = list(provider.vlm_derived_role_vocabulary)
        context_prompts = list(provider.task_explicit_context_vocabulary)
        vocabulary = tuple(dict.fromkeys(vlm_prompts + context_prompts))
        canon_trace = result.get("canonicalization_trace") or {}

        return FunctionalRequirementGraph(
            domain="living_room",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            operation_groups=tuple(operation_groups),
            cross_group_reuse_allowed=False,
            detector_vocabulary=vocabulary,
            candidate_regions=(),
            region_ranking=(),
            source="VLM_CANONICAL_G_F",
            raw_requirements=(result.get("normalized_task_contract") or result["raw_vlm_decomposition"],),
            metadata={
                "vlm_canonicalization_version": canon_trace.get("vlm_canonicalization_version", LIVING_ROOM_VLM_CANONICALIZATION_VERSION),
                "role_semantic_ontology_version": PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION,
                "semantic_acceptance_source": "SYSTEM_ROLE_SEMANTIC_ONTOLOGY",
                "detector_vocabulary_source": "SYSTEM_REVIEWED_ENVIRONMENT_CONTRACT",
                "candidate_categories_used_for_role_identity": False,
                "candidate_categories_used_for_grounding_acceptance": False,
                "candidate_categories_used_for_detector_vocabulary": True,
                "semantic_vocabulary_path": str(provider.vocabulary_path),
                "vlm_derived_role_vocabulary": vlm_prompts,
                "task_explicit_context_vocabulary": context_prompts,
                "raw_vlm_response": result.get("raw_vlm_response"),
                "validated_vlm_specification": result.get("validated_vlm_specification"),
                "canonicalization_trace": canon_trace,
                "raw_decomposition": result["raw_vlm_decomposition"],
                "normalization_audit": result["reviewed_ontology_audit"],
            },
        )


