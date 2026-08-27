"""Reviewed static functional specifications with no solution knowledge."""

from __future__ import annotations

from pathlib import Path
import yaml

from .models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    NumericConstraint,
    OperationGroup,
)
from .spec_provider import FunctionalSpecProvider


class GTSpecProvider(FunctionalSpecProvider):
    def provide(
        self,
        domain: str,
        task_instruction: str,
        observation_images: list[Path] | None = None,
    ) -> FunctionalRequirementGraph:
        if domain == "workshop":
            graph = self._workshop(task_instruction)
        elif domain == "kitchen":
            graph = self._kitchen(task_instruction)
        elif domain == "living_room":
            graph = self._living_room(task_instruction)
        else:
            raise NotImplementedError(f"GT specification adapter is not implemented for {domain}")
        graph.validate()
        return graph

    @staticmethod
    def _workshop(task_instruction: str) -> FunctionalRequirementGraph:
        from mujoco_scenes.workshop_phase1.requirements import ManualWorkshopFMContract

        provider = ManualWorkshopFMContract()
        requirements = tuple(provider.get_requirements(task_instruction))
        nodes: dict[str, FunctionalRole] = {}
        relations: list[FunctionalRelation] = []

        # Functional roles for driver and fastener
        nodes["driver"] = FunctionalRole(
            name="driver",
            entity_kind="OBJECT",
            count=1,
            semantic_categories=("screwdriver", "power_drill", "Phillips screwdriver", "cordless power drill"),
            unary_predicates=("CAN_DRIVE_SCREW",),
            binding_policy="DISTINCT",
            verification_mode="SEMANTIC_AND_GEOMETRIC",
            description="Tool capable of driving a screw into the workpiece",
        )
        nodes["fastener"] = FunctionalRole(
            name="fastener",
            entity_kind="OBJECT",
            count=1,
            semantic_categories=("screw", "Phillips screw"),
            unary_predicates=("CAN_FASTEN",),
            binding_policy="DISTINCT",
            verification_mode="SEMANTIC_AND_GEOMETRIC",
            description="Fastener capable of threading into the workpiece hole",
        )
        # Fixed repair target on the workpiece
        nodes["repair_target"] = FunctionalRole(
            name="repair_target",
            entity_kind="FIXED_TARGET",
            count=1,
            semantic_categories=("repair_target", "workshop_frame_joint", "recess"),
            binding_policy="DISTINCT",
            verification_mode="GEOMETRIC_ONLY",
            description="Target repair hole on the workpiece",
        )

        # Semantically correct Workshop relations
        relations.append(FunctionalRelation(
            subject_role="driver",
            predicate="COMPATIBLE_WITH",
            object_role="fastener",
            expected=True,
        ))
        relations.append(FunctionalRelation(
            subject_role="driver",
            predicate="REACHES_TARGET",
            object_role="repair_target",
            expected=True,
        ))
        relations.append(FunctionalRelation(
            subject_role="fastener",
            predicate="COMPATIBLE_WITH_TARGET",
            object_role="repair_target",
            expected=True,
        ))

        vocabulary = (
            "screwdriver", "power drill", "screw", "wooden hammer",
            "Phillips screwdriver", "cordless power drill", "Phillips screw",
        )
        ranking = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
        return FunctionalRequirementGraph(
            domain="workshop",
            task_instruction=task_instruction,
            nodes=nodes,
            relations=tuple(relations),
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
    def _kitchen(task_instruction: str) -> FunctionalRequirementGraph:
        root = Path(__file__).resolve().parents[1]
        contract_path = root / "configs" / "s1_integrated_kitchen_object_function.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        nodes: dict[str, FunctionalRole] = {}
        vocabulary: list[str] = []
        relations: list[FunctionalRelation] = []

        for rel in contract.get("relations", []):
            relations.append(FunctionalRelation(
                subject_role=rel["subject_role"],
                predicate=rel["predicate"],
                object_role=rel["object_role"],
                expected=bool(rel.get("expected", True)),
            ))

        for name, raw in contract["roles"].items():
            cardinality = raw.get("binding_cardinality", {})
            min_count = cardinality.get("minimum_distinct_physical_objects")
            max_count = cardinality.get("maximum_distinct_physical_objects")
            preferred = cardinality.get("preferred")
            count = int(raw.get("count", max_count or min_count or 1))
            categories = tuple(
                item["canonical_label"] for item in raw.get("semantic_preferences", [])
            )
            for item in raw.get("semantic_preferences", []):
                vocabulary.extend([item["canonical_label"], *item.get("detector_aliases", [])])

            unary_preds = []
            numeric_reqs = []
            for item in raw.get("unary_geometry", []):
                if "predicate" in item:
                    unary_preds.append(str(item["predicate"]))
                elif "property" in item:
                    numeric_reqs.append(NumericConstraint(
                        property_name=str(item["property"]),
                        operator=str(item.get("operator", ">=")),
                        threshold=float(item["value"]),
                        unit=str(item.get("unit", "m")),
                    ))

            binding = "REUSABLE" if preferred == "minimize_distinct" else "DISTINCT"
            nodes[name] = FunctionalRole(
                name=name,
                entity_kind="OBJECT",
                count=count,
                min_count=min_count,
                max_count=max_count,
                preference=preferred,
                semantic_categories=categories,
                unary_predicates=tuple(unary_preds),
                numeric_constraints=tuple(numeric_reqs),
                binding_policy=binding,
                verification_mode="SEMANTIC_AND_GEOMETRIC" if unary_preds or numeric_reqs else "SEMANTIC_ONLY",
            )

        for name, raw in contract.get("symbolic_task", {}).get("source_roles", {}).items():
            labels = tuple(raw["accepted_semantic_labels"])
            vocabulary.extend(labels)
            nodes[name] = FunctionalRole(
                name=name,
                entity_kind="OBJECT",
                count=int(raw.get("count", 1)),
                semantic_categories=labels,
                binding_policy="DISTINCT",
                verification_mode="SEMANTIC_ONLY",
            )

        operation_groups: list[OperationGroup] = []
        for gid, grp in contract.get("operation_groups", {}).items():
            policy = grp.get("usage_policy", {})
            mode_str = policy.get("mode", "sequential_reuse_allowed").upper()
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
                distinct_within_group=bool(policy.get("distinct_within_group", True)),
                same_tool_must_cover_all_targets=bool(policy.get("same_tool_must_cover_all_targets", False)),
                selection_preference=str(policy.get("selection_preference", "")),
            ))

        regions = ("D1", "D2", "C2", "B1", "C1")
        return FunctionalRequirementGraph(
            domain="kitchen",
            task_instruction=task_instruction or contract.get("goal_instruction", ""),
            nodes=nodes,
            relations=tuple(relations),
            operation_groups=tuple(operation_groups),
            cross_group_reuse_allowed=bool(contract.get("cross_group_reuse", {}).get("allowed", False)),
            detector_vocabulary=tuple(dict.fromkeys(vocabulary)),
            candidate_regions=regions,
            region_ranking=regions,
            source="GT_FUNCTIONAL_SPEC_ONLY",
            raw_requirements=(contract,),
            metadata={
                "semantic_vocabulary_path": str(root / "configs" / "semantic_vocabulary.yaml"),
                "contract_path": str(contract_path),
                "symbolic_task": contract.get("symbolic_task", {}),
            },
        )

    @staticmethod
    def _living_room(task_instruction: str) -> FunctionalRequirementGraph:
        root = Path(__file__).resolve().parents[1]
        contract_path = root / "configs" / "l2_integrated_region_function_task.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        nodes: dict[str, FunctionalRole] = {}
        vocabulary: list[str] = []
        relations: list[FunctionalRelation] = []

        # Region support roles
        for group in contract["function_groups"].values():
            role_name = group["region_role"]
            semantic = contract["semantic_requirements"]["region_roles"][role_name]
            categories = tuple(semantic["accepted_categories"])
            vocabulary.extend(categories)
            func_id = group["function_id"]
            binding = "SHARED" if group["usage_policy"] == "SHARED_REGION_REQUIRED" else "DISTINCT"
            nodes[func_id] = FunctionalRole(
                name=func_id,
                entity_kind="REGION",
                count=int(group.get("required_target_count", 1)),
                semantic_categories=categories,
                unary_predicates=("PLANAR_SUPPORT",),
                binding_policy=binding,
                verification_mode="SEMANTIC_AND_GEOMETRIC",
            )

        # Explicit target nodes
        nodes["CUP_SAUCER_SET"] = FunctionalRole(
            name="CUP_SAUCER_SET",
            entity_kind="OBJECT",
            count=2,
            semantic_categories=("cup_saucer_set", "cup", "saucer"),
            binding_policy="DISTINCT",
            verification_mode="SEMANTIC_ONLY",
        )
        nodes["REMOTE"] = FunctionalRole(
            name="REMOTE",
            entity_kind="OBJECT",
            count=1,
            semantic_categories=("remote_control", "tv_remote"),
            binding_policy="DISTINCT",
            verification_mode="SEMANTIC_ONLY",
        )
        nodes["SEATING_POSITION"] = FunctionalRole(
            name="SEATING_POSITION",
            entity_kind="FIXED_TARGET",
            count=2,
            semantic_categories=("armchair", "chair", "sofa", "seating_position"),
            binding_policy="DISTINCT",
            verification_mode="SEMANTIC_ONLY",
        )
        nodes["SEATING_PAIR"] = FunctionalRole(
            name="SEATING_PAIR",
            entity_kind="FIXED_TARGET",
            count=1,
            semantic_categories=("armchair", "chair", "sofa", "seating_pair"),
            binding_policy="SHARED",
            verification_mode="SEMANTIC_ONLY",
        )

        operation_groups = [
            OperationGroup(
                id="personal_support_group",
                function="SUPPORT_DRINKWARE",
                tool_role="PERSONAL_CUP_SAUCER_REGION",
                target_role="CUP_SAUCER_SET",
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("FITS_SET_ON", "NEAR_SEAT"),
                required_target_count=2,
                distinct_within_group=True,
                same_tool_must_cover_all_targets=False,
            )
        ]

        # Explicit relations for shared region
        relations.append(FunctionalRelation(
            subject_role="SHARED_REMOTE_REGION",
            predicate="FITS_ON",
            object_role="REMOTE",
            expected=True,
        ))
        relations.append(FunctionalRelation(
            subject_role="SHARED_REMOTE_REGION",
            predicate="ACCESSIBLE_FROM_BOTH_SEATS",
            object_role="SEATING_PAIR",
            expected=True,
        ))

        for labels in contract["semantic_requirements"]["payload_roles"].values():
            vocabulary.extend(labels)
        vocabulary.extend(contract["semantic_requirements"]["seating_categories"])
        return FunctionalRequirementGraph(
            domain="living_room",
            task_instruction=task_instruction or contract["natural_language_goal"],
            nodes=nodes,
            relations=tuple(relations),
            operation_groups=tuple(operation_groups),
            cross_group_reuse_allowed=False,
            detector_vocabulary=tuple(dict.fromkeys(vocabulary)),
            candidate_regions=(),
            region_ranking=(),
            source="GT_FUNCTIONAL_SPEC_ONLY",
            raw_requirements=(contract,),
            metadata={
                "contract_path": str(contract_path),
                "semantic_vocabulary_path": str(
                    root / "configs" / "l2_integrated_region_function_semantic_vocabulary.yaml"
                ),
            },
        )


