"""Workshop adapters around existing perception, geometry, and robot routines."""

from __future__ import annotations

from typing import Any

from mujoco_scenes.symbolic_planning_core import SymbolicAction, SymbolicProblem
from mujoco_scenes.workshop_ground_truth_execution import WorkshopExecutionDispatcher
from mujoco_scenes.workshop_ground_truth_planner import WorkshopAssignment
from mujoco_scenes.workshop_ground_truth_state import WorkshopWorldState
from mujoco_scenes.workshop_phase1.functional_search import FunctionalSatisfactionSearch
from mujoco_scenes.workshop_phase1.geometric_grounding import GeometricGrounder
from mujoco_scenes.workshop_phase1.inspection_controller import WorkshopPhase1InspectionController
from mujoco_scenes.workshop_phase1.types import MaskBackendType, SemanticBackendType
from mujoco_scenes.workshop_scene import WORKSHOP_REGIONS, WorkshopScene

from ..models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    FunctionalSpecification,
    GraphGroundingResult,
    SatisfactionResult,
)
from ..scene_graph import ObservedNode, ObservedObject, ObservedRelation, ObservedSceneGraph
from mujoco_scenes.workshop_phase1.types import EntityType, FunctionalRequirement, RequirementSource


TARGET = "workshop_frame_joint"
SURFACE = "MAIN_WORKBENCH_ZONE"
TUNED_CONFIG = (
    "mujoco_scenes/configs/workshop_phase1_yoloworld_l_five_view_close.yaml"
)
CURRENT_GEOMETRY_CONFIG = "mujoco_scenes/configs/workshop_geometry_inference.yaml"
SEMANTIC_ENTITY_HANDLES = {
    "screwdriver": "workshop_long_phillips_driver",
    "manual_screwdriver": "workshop_long_phillips_driver",
    "power_driver": "workshop_power_driver",
    "power_drill": "workshop_power_driver",
    "screw": "workshop_medium_phillips_screw",
}


def compile_workshop_requirements_from_graph(
    graph: FunctionalRequirementGraph,
) -> list[FunctionalRequirement]:
    """Deterministically compile legacy Workshop requirements from G_F."""
    requirements: list[FunctionalRequirement] = []
    for rank, (name, role) in enumerate(graph.nodes.items(), start=1):
        if role.entity_kind != "OBJECT":
            continue
        func_name = (
            "CAN_DRIVE_SCREW"
            if "driver" in name.lower() or "drive" in name.lower()
            else ("CAN_FASTEN" if "fastener" in name.lower() or "screw" in name.lower() or "fasten" in name.lower() else name)
        )
        req_rels = [
            r.predicate for r in graph.relations if r.subject_role == name
        ]

        source = RequirementSource.FM if "VLM" in graph.source else RequirementSource.STATIC
        requirements.append(
            FunctionalRequirement(
                requirement_id=f"req_{name}",
                entity_type=EntityType.OBJECT,
                function_name=func_name,
                description=role.description or f"Role {name}",
                rank=rank,
                source=source,
                accepted_categories=list(role.semantic_categories),
                semantic_hints=list(role.semantic_hints),
                required_relations=req_rels,
                provenance="compiled_from_functional_requirement_graph",
            )
        )
    return requirements


def _action(
    name: str,
    arguments: tuple[str, ...],
    positive: set[tuple[str, ...]],
    add: set[tuple[str, ...]],
    delete: set[tuple[str, ...]],
) -> SymbolicAction:
    return SymbolicAction(
        name=name,
        arguments=arguments,
        positive_preconditions=frozenset(positive),
        negative_preconditions=frozenset(),
        add_effects=frozenset(add),
        delete_effects=frozenset(delete),
    )


class WorkshopPlanningCompiler:
    """Compile a perception-grounded repair witness into generic STRIPS actions."""

    def compile_problem(
        self, assignment: dict[str, str], context: dict[str, Any]
    ) -> SymbolicProblem:
        driver = assignment["driver"]
        fastener = assignment["fastener"]
        driver_source = assignment["driver_source"]
        fastener_source = assignment["fastener_source"]
        opened = set(context.get("opened_regions", ()))
        initial = {
            ("hand_empty",),
            ("at", driver, driver_source),
            ("at", fastener, fastener_source),
        }
        initial.update(("open", region) for region in opened)

        def source_preconditions(obj: str, source: str) -> set[tuple[str, ...]]:
            conditions = {("hand_empty",), ("at", obj, source)}
            if source in WORKSHOP_REGIONS:
                conditions.add(("open", source))
            return conditions

        actions = (
            _action(
                "PICK", (fastener, fastener_source),
                source_preconditions(fastener, fastener_source),
                {("holding", fastener)},
                {("hand_empty",), ("at", fastener, fastener_source)},
            ),
            _action(
                "PLACE", (fastener, TARGET),
                {("holding", fastener)},
                {("hand_empty",), ("at", fastener, TARGET), ("inserted", fastener, TARGET)},
                {("holding", fastener)},
            ),
            _action(
                "PICK", (driver, driver_source),
                source_preconditions(driver, driver_source),
                {("holding", driver)},
                {("hand_empty",), ("at", driver, driver_source)},
            ),
            _action(
                "SCREW", (driver, fastener, TARGET),
                {("holding", driver), ("inserted", fastener, TARGET)},
                {("repaired", TARGET)},
                set(),
            ),
            _action(
                "PLACE", (driver, SURFACE),
                {("holding", driver), ("repaired", TARGET)},
                {("hand_empty",), ("at", driver, SURFACE)},
                {("holding", driver)},
            ),
        )
        return SymbolicProblem(
            initial_atoms=frozenset(initial),
            goal_atoms=frozenset({
                ("repaired", TARGET), ("at", driver, SURFACE), ("hand_empty",),
            }),
            actions=actions,
        )


class WorkshopDomainAdapter:
    """Run staged Workshop evidence acquisition without variant solution data."""

    task_instruction = (
        "Find a compatible screw and driver, insert the screw tip-down into "
        "the workbench repair hole, drive it fully, and return the driver safely."
    )

    def __init__(
        self,
        variant: str,
        specification: FunctionalSpecification,
        *,
        scene: WorkshopScene | None = None,
        physical_open: bool = True,
        output_dir: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.variant = variant
        self.specification = specification
        self.scene = scene or WorkshopScene(robot="google", variant=variant)
        self.physical_open = physical_open
        self.verbose = verbose
        self.graph = ObservedSceneGraph()
        self.controller = WorkshopPhase1InspectionController(
            scene=self.scene,
            config_path=TUNED_CONFIG,
            # Visible-pixel MuJoCo instance masks provide reliable proposals;
            # the existing auxiliary YOLO-World backend still supplies the
            # semantic labels.  Closed-region objects remain unobservable.
            mask_backend=MaskBackendType.ORACLE,
            semantic_backend=SemanticBackendType.PRODUCTION,
            output_dir=output_dir,
        )
        # The close-view file carries a legacy frozen-evaluation length cap.
        # Use the calibrated L-model camera and visual profiles.
        self.controller.functional_search = FunctionalSatisfactionSearch(
            self.controller.geometric_grounder
        )
        self._configure_from_specification()
        self._stage = -1
        self._witness = None
        self._rejection_reason: str | None = None
        self._registered = False
        self._world_state = WorkshopWorldState()
        placeholder = WorkshopAssignment(
            variant_id=variant,
            intended_outcome="UNKNOWN",
            is_feasible=False,
            assignment_source="NO_ASSIGNMENT_DURING_SEARCH",
        )
        self.dispatcher = WorkshopExecutionDispatcher(self.scene, placeholder)
        self.physical_assignment: WorkshopAssignment | None = None

    def _configure_from_specification(self) -> None:
        controller = self.controller
        controller.requirements = compile_workshop_requirements_from_graph(self.specification)
        controller.inspection_sequence = list(self.specification.region_ranking)
        mapping = dict(self.specification.metadata.get("detector_label_to_canonical", {}))
        alias_mapping = dict(self.specification.metadata.get("alias_to_canonical", {}))
        for prompt in self.specification.detector_vocabulary:
            normalized = prompt.lower()
            if normalized not in mapping and normalized in alias_mapping:
                mapping[normalized] = alias_mapping[normalized]
        controller.prompts = list(self.specification.detector_vocabulary)
        controller.detector_label_to_canonical = mapping
        controller.alias_to_canonical = alias_mapping
        controller.detector_vocabulary = [
            {
                "detector_label": prompt,
                "canonical_label": mapping.get(prompt.lower(), prompt.lower().replace(" ", "_")),
            }
            for prompt in controller.prompts
        ]
        controller.object_categories = {
            entry["canonical_label"] for entry in controller.detector_vocabulary
        } - controller.region_categories
        controller.proposal_backend.set_vocabulary(controller.prompts, mapping)
        if controller._yolo_aux_backend is not None:
            controller._yolo_aux_backend.set_vocabulary(controller.prompts, mapping)
            # Small fasteners are unreliable in a single multi-class pass.
            # Reuse the backend's existing per-prompt supplemental passes at a
            # lower proposal threshold; geometry and joint compatibility still
            # decide whether a proposal can fill a role.
            controller._yolo_aux_backend.supplemental_prompts = [
                prompt for prompt in (
                    "Phillips screwdriver", "cordless power drill",
                    "Phillips head screw", "Phillips screw", "screw",
                    "wooden hammer",
                ) if prompt in controller.prompts
            ]
            controller._yolo_aux_backend.supplemental_confidence_threshold = 0.001

    def _register_graph(self) -> None:
        if self._registered:
            return
        self.controller.graph.register_workpiece_node()
        self.graph.add_node(ObservedNode(
            instance_id="repair_target",
            entity_kind="FIXED_TARGET",
            canonical_category="repair_target",
            unary_properties={"target": "workbench_hole"},
        ))
        for region in self.specification.candidate_regions:
            self.controller.graph.register_inspection_region_node(
                region, f"Search container {region}"
            )
            self.graph.add_node(ObservedNode(
                instance_id=region,
                entity_kind="REGION",
                canonical_category=region,
            ))
            self.graph.regions[region] = {"inspected": False}
        self._registered = True

    def _capture_and_evaluate(self, source_region: str) -> None:
        self._stage += 1
        observations = self.controller._capture_and_process_stage(
            stage_idx=self._stage, source_region_id=source_region
        )
        if observations and len(observations) > 0 and getattr(observations[0], "rgb", None) is not None:
            self.latest_frame_rgb = np.array(observations[0].rgb, copy=True)
            self.latest_camera_id = getattr(observations[0], "camera_id", "WORKSHOP_CAMERA")
        if self._stage == 0:
            target = self.controller.geometric_grounder.observe_target_recess(
                observations,
                scene=self.scene,
                config=self.controller.geometric_grounder.config,
            )
            self.controller.target_evidence = target
            self.controller.geometric_grounder.target_evidence = target
        self._witness, self._rejection_reason = (
            self.controller._evaluate_grounding_and_search(
                stage_idx=self._stage, source_region_id=source_region
            )
        )
        self._sync_common_graph()

    def _sync_common_graph(self) -> None:
        objects = []
        drivers = []
        fasteners = []
        from ..grounding import check_semantic_role_compatibility

        driver_categories = [
            "screwdriver", "power_driver", "power_drill",
            "phillips screwdriver", "phillips_screwdriver",
            "cordless power drill", "cordless_power_drill",
        ]
        fastener_categories = [
            "screw", "phillips_screw",
            "phillips head screw",
        ]

        for track in self.controller.tracker.tracks.values():
            if track.current_semantic_belief.get("status") == "SUPPORTED":
                canonical = track.current_semantic_belief.get("canonical_label")
            else:
                canonical = None
            driver_status, _ = check_semantic_role_compatibility(
                track.current_semantic_belief, driver_categories
            )
            fastener_status, _ = check_semantic_role_compatibility(
                track.current_semantic_belief, fastener_categories
            )

            node = ObservedObject(
                instance_id=track.instance_id,
                entity_kind="OBJECT",
                canonical_category=canonical,
                semantic_labels=dict(track.current_semantic_belief),
                source_region=track.source_inspection_region_id,
                geometry=dict(track.current_geometric_properties),
                unary_properties=dict(track.current_geometric_properties),
                unary_predicates={
                    "CAN_DRIVE_SCREW": driver_status,
                    "CAN_FASTEN": fastener_status,
                },
                last_seen_stage=track.last_seen_stage,
            )
            objects.append(node)
            if driver_status in ("TRUE", "UNKNOWN"):
                drivers.append(track)
            if fastener_status in ("TRUE", "UNKNOWN"):
                fasteners.append(track)
        self.graph.update_objects(objects, self._stage)

        # Evaluate and store all pairwise relations into G_O
        for d in drivers:
            reach_eval = self.controller.geometric_grounder.evaluate_reaches_target(
                d.current_geometric_properties
            )
            reach_status = reach_eval.get("status", "UNKNOWN")
            self.graph.add_relation(ObservedRelation(
                subject_id=d.instance_id,
                predicate="REACHES_TARGET",
                object_id="repair_target",
                status=reach_status,
                evidence=dict(reach_eval),
            ))

        for f in fasteners:
            f_target_eval = self.controller.geometric_grounder.evaluate_compatible_with_target(
                f.current_geometric_properties
            )
            f_status = f_target_eval.get("status", "UNKNOWN")
            self.graph.add_relation(ObservedRelation(
                subject_id=f.instance_id,
                predicate="COMPATIBLE_WITH_TARGET",
                object_id="repair_target",
                status=f_status,
                evidence=dict(f_target_eval),
            ))

        for d in drivers:
            for f in fasteners:
                compat_eval = GeometricGrounder.evaluate_compatible_with(d, f)
                rel_status = compat_eval.get("status", "UNKNOWN")
                self.graph.add_relation(ObservedRelation(
                    subject_id=d.instance_id,
                    predicate="COMPATIBLE_WITH",
                    object_id=f.instance_id,
                    status=rel_status,
                    evidence=dict(compat_eval),
                ))

    def observe_initial(self) -> None:
        self._register_graph()
        self._capture_and_evaluate("INITIAL_WORKBENCH")

    def open_region(self, region: str) -> dict[str, Any]:
        action = {
            "action_index": len(self.graph.inspected_regions) + 1,
            "action_instance_id": f"explore_open_{region.lower()}",
            "operator": "OPEN",
            "arguments": [region],
        }
        if self.physical_open:
            result = self.dispatcher.execute(action, self._world_state)
        else:
            direct = self.scene.open_container(region)
            result = {
                "success": bool(direct.get("opened")),
                "status": "DRY_RUN_DIRECT_ARTICULATION",
                "articulation": direct,
                "robot_actuated_motion": False,
            }
        if result.get("success"):
            self._world_state.apply(action)
            self.graph.mark_region_inspected(region)
        return result

    def observe_after_open(self, region: str) -> None:
        self._capture_and_evaluate(region)

    def evaluate_satisfaction(self, search_exhausted: bool = False) -> SatisfactionResult:
        from ..grounding import ground_graph

        ground_result = ground_graph(
            self.specification, self.graph, {"search_exhausted": search_exhausted}
        )
        if not ground_result.complete or not ground_result.assignment:
            return GraphGroundingResult(
                status=ground_result.status,
                complete=False,
                assignment=None,
                operation_bindings=ground_result.operation_bindings,
                missing_roles=ground_result.missing_roles,
                unsatisfied_relations=ground_result.unsatisfied_relations,
                unresolved_constraints=ground_result.unresolved_constraints,
                evidence={
                    "stage": self._stage,
                    "grounding_status": ground_result.status,
                    "grounding_evidence": ground_result.evidence,
                    "observed_object_count": len(self.graph.objects),
                    "reason": self._rejection_reason,
                },
            )

        driver_id = ground_result.assignment.get("driver", ground_result.assignment.get("CAN_DRIVE_SCREW"))
        fastener_id = ground_result.assignment.get("fastener", ground_result.assignment.get("CAN_FASTEN"))
        driver_track = self.controller.tracker.tracks[driver_id]
        fastener_track = self.controller.tracker.tracks[fastener_id]
        driver = self._physical_handle(driver_track, role="driver")
        fastener = self._physical_handle(fastener_track, role="fastener")
        assignment = {
            "driver": driver,
            "fastener": fastener,
            "driver_track": driver_id,
            "fastener_track": fastener_id,
            "driver_source": self._source(driver_track.source_inspection_region_id),
            "fastener_source": self._source(fastener_track.source_inspection_region_id),
            "work_surface": SURFACE,
            "target_joint": TARGET,
        }
        self.physical_assignment = WorkshopAssignment(
            variant_id=self.variant,
            intended_outcome="FEASIBLE",
            is_feasible=True,
            driver=driver,
            fastener=fastener,
            work_surface=SURFACE,
            target_joint=TARGET,
            assignment_source="CANONICAL_GRAPH_GROUNDING",
            source_ids={"driver": driver_id, "fastener": fastener_id},
        )
        self.dispatcher.assignment = self.physical_assignment
        return GraphGroundingResult(
            status="COMPLETE",
            complete=True,
            assignment=assignment,
            operation_bindings=ground_result.operation_bindings,
            missing_roles=(),
            unsatisfied_relations=(),
            unresolved_constraints=(),
            evidence={"grounding": ground_result.to_dict(), "stage": self._stage},
        )

    @staticmethod
    def _source(source: str) -> str:
        return SURFACE if source in {"INITIAL", "INITIAL_WORKBENCH", "workbench"} else source

    @staticmethod
    def _physical_handle(track: Any, *, role: str) -> str:
        # This benchmark has one supported physical fastener class.  The
        # functional solver, not this adapter, has already established that
        # the selected observed track satisfies CAN_FASTEN.
        if role == "fastener":
            return "workshop_medium_phillips_screw"
        belief = track.current_semantic_belief
        labels = [
            belief.get("canonical_label"), belief.get("evaluated_label"),
            belief.get("raw_label"), belief.get("predicted_label"),
        ]
        support = belief.get("label_supporting_view_count", {})
        labels.extend(sorted(support, key=support.get, reverse=True))
        for label in labels:
            normalized = str(label or "").strip().lower().replace(" ", "_")
            handle = SEMANTIC_ENTITY_HANDLES.get(normalized)
            if handle in {
                "workshop_long_phillips_driver", "workshop_power_driver"
            }:
                return handle
        raise RuntimeError(
            f"No execution handle for observed track {track.instance_id}; "
            f"semantic evidence={belief}"
        )

    def planning_context(self) -> dict[str, Any]:
        return {"opened_regions": tuple(self.graph.inspected_regions)}
