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

VLM_CANONICALIZATION_VERSION = "phase3_6a3_v1"


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
        requirements = tuple(provider.get_requirements(
            task_instruction, observation_images=observation_images
        ))
        ranking = tuple(provider.generate_inspection_policy(
            task_instruction, observation_images=observation_images
        ))

        candidate_regions = tuple(provider.candidate_regions)
        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        driver_role_name = None
        fastener_role_name = None
        driver_req = None
        fastener_req = None

        for requirement in requirements:
            func_name = requirement.function_name
            role_id = "driver" if func_name == "CAN_DRIVE_SCREW" else ("fastener" if func_name == "CAN_FASTEN" else requirement.requirement_id)
            if func_name == "CAN_DRIVE_SCREW":
                driver_role_name = role_id
                driver_req = requirement
            elif func_name == "CAN_FASTEN":
                fastener_role_name = role_id
                fastener_req = requirement

            role = FunctionalRole(
                name=role_id,
                entity_kind="OBJECT",
                count=1,
                semantic_categories=tuple(requirement.accepted_categories),
                unary_predicates=(func_name,),
                binding_policy="DISTINCT",
                verification_mode="SEMANTIC_AND_GEOMETRIC",
                description=requirement.description,
                semantic_hints=tuple(requirement.semantic_hints),
            )
            nodes[role_id] = role

        driver_id = driver_role_name or "driver"
        fastener_id = fastener_role_name or "fastener"
        target_id = "repair_target"

        needs_target = False
        if driver_req and "REACHES_TARGET" in driver_req.required_relations:
            needs_target = True
        if fastener_req and "COMPATIBLE_WITH_TARGET" in fastener_req.required_relations:
            needs_target = True

        if needs_target:
            nodes[target_id] = FunctionalRole(
                name=target_id,
                entity_kind="FIXED_TARGET",
                count=1,
                semantic_categories=("repair_target", "workshop_frame_joint", "recess"),
                binding_policy="DISTINCT",
                verification_mode="GEOMETRIC_ONLY",
                description="Target repair hole on the workpiece",
            )

        if driver_req:
            if "COMPATIBLE_WITH" in driver_req.required_relations and fastener_id in nodes:
                relations.append(FunctionalRelation(
                    subject_role=driver_id,
                    predicate="COMPATIBLE_WITH",
                    object_role=fastener_id,
                    expected=True,
                ))
            if "REACHES_TARGET" in driver_req.required_relations and target_id in nodes:
                relations.append(FunctionalRelation(
                    subject_role=driver_id,
                    predicate="REACHES_TARGET",
                    object_role=target_id,
                    expected=True,
                ))

        if fastener_req:
            if "COMPATIBLE_WITH_TARGET" in fastener_req.required_relations and target_id in nodes:
                relations.append(FunctionalRelation(
                    subject_role=fastener_id,
                    predicate="COMPATIBLE_WITH_TARGET",
                    object_role=target_id,
                    expected=True,
                ))

        # Derive detector vocabulary strictly from G_F role categories and generic aliases
        role_categories = set()
        for node in nodes.values():
            if node.entity_kind == "OBJECT":
                role_categories.update(node.semantic_categories)

        detector_map = provider.get_detector_label_to_canonical_map()
        prompts = []
        for prompt, canonical in detector_map.items():
            if canonical in role_categories and prompt not in prompts:
                prompts.append(prompt)
        vocabulary = tuple(prompts)

        return FunctionalRequirementGraph(
            domain="workshop",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
            detector_vocabulary=vocabulary,
            candidate_regions=candidate_regions,
            region_ranking=ranking,
            source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=requirements,
            metadata={
                "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
                "detector_label_to_canonical": provider.get_detector_label_to_canonical_map(),
                "alias_to_canonical": provider.get_alias_to_canonical_map(),
                "raw_decomposition": provider.raw_decomposition,
                "transformation_trace": getattr(provider, "transformation_trace", []),
                "evaluation_negative_controls": ["wooden hammer"],
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
        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []
        all_categories: list[str] = []

        for row in requirements:
            func_id = row["function"]
            binding = row.get("binding_policy", "SHARED" if "SHARED" in func_id else "DISTINCT")
            count = int(row["vlm_required_count"])
            entity_kind = row.get("entity_kind", "REGION")
            cats = tuple(row["accepted_categories"])
            all_categories.extend(cats)
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

        raw_to_canonical = {}
        for row in requirements:
            for raw_id in row.get("raw_vlm_role_ids", []):
                raw_to_canonical[raw_id] = row["function"]
            if "raw_vlm_role_id" in row:
                raw_to_canonical[row["raw_vlm_role_id"]] = row["function"]
            raw_to_canonical[row["function"]] = row["function"]
            raw_to_canonical[row["role_id"]] = row["function"]

        raw_relations = result.get("raw_vlm_decomposition", {}).get("functional_relations", [])
        for rel_item in raw_relations:
            s = rel_item.get("subject_role")
            r = rel_item.get("relation")
            o = rel_item.get("object_role")
            if s and r and o:
                canon_s = raw_to_canonical.get(str(s), str(s))
                canon_o = raw_to_canonical.get(str(o), str(o))
                if canon_s in nodes and canon_o in nodes:
                    relations.append(FunctionalRelation(
                        subject_role=canon_s,
                        predicate=str(r),
                        object_role=canon_o,
                        expected=True,
                    ))

        role_categories = set(all_categories)
        role_categories.update(["armchair", "chair", "sofa", "remote_control", "tv_remote", "cup", "saucer", "cup_saucer_set"])
        aliases = provider._vocabulary_aliases()
        prompts: list[str] = []
        for cat in aliases:
            if cat in role_categories:
                prompts.extend(aliases[cat])
        vocabulary = tuple(dict.fromkeys(prompts))

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
            source="VLM_FUNCTIONAL_SPEC",
            raw_requirements=(result.get("normalized_task_contract") or result["raw_vlm_decomposition"],),
            metadata={
                "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
                "semantic_vocabulary_path": str(provider.vocabulary_path),
                "raw_decomposition": result["raw_vlm_decomposition"],
                "normalization_audit": result["reviewed_ontology_audit"],
            },
        )


