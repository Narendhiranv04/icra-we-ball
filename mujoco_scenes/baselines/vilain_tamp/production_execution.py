"""Production-only composition of neutral MuJoCo controller primitives.

The module deliberately receives a completed ViLaIn execution projection.  It
does not inspect benchmark answers while selecting actions or identities.
Privileged variant data is loaded by a separate provider only after the runner
has terminated planning/execution and asks for final scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import Domain
from .contracts import BaselineExecutionPlan, ExecutionProjection
from .evaluation import HiddenBenchmarkContext
from .identity import EntityBinding
from .live_execution import (
    KitchenLiveControllerFacade,
    LiveDomainRuntime,
    LiveExecutionError,
    MuJoCoPhysicalStateObserver,
    WorkshopLiveControllerFacade,
)


class ProductionExecutionError(LiveExecutionError):
    """A selected baseline plan cannot be connected to physical primitives."""


def _entity_binding(payload: Mapping[str, Any]) -> EntityBinding:
    fields = {
        "object_id": str(payload["object_id"]),
        "entity_name": str(payload["entity_name"]),
        "pddl_type": str(payload["pddl_type"]),
        "broad_class": str(payload["broad_class"]),
        "centroid_distance_m": payload.get("centroid_distance_m"),
        "aabb_distance_m": payload.get("aabb_distance_m"),
        "observed_centroid_m": _optional_tuple(payload.get("observed_centroid_m")),
        "entity_centroid_m": _optional_tuple(payload.get("entity_centroid_m")),
        "entity_aabb_min_m": _optional_tuple(payload.get("entity_aabb_min_m")),
        "entity_aabb_max_m": _optional_tuple(payload.get("entity_aabb_max_m")),
        "binding_method": str(payload["binding_method"]),
        "confidence": float(payload["confidence"]),
        "observation_stage_ids": tuple(payload.get("observation_stage_ids", ())),
        "evidence_artifacts": tuple(payload.get("evidence_artifacts", ())),
    }
    return EntityBinding(**fields)


def _optional_tuple(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    result = tuple(float(item) for item in value)
    if len(result) != 3:
        raise ProductionExecutionError("identity geometry is not three-dimensional")
    return result  # type: ignore[return-value]


class ProductionRuntimeProvider:
    """Build an execution runtime from the selected in-memory ViLaIn attempt."""

    def __init__(
        self,
        *,
        scene: Any,
        domain: Domain,
        variant: str,
        corrective: Any,
        fixed_bindings: Mapping[str, EntityBinding],
    ) -> None:
        self.scene = scene
        self.domain = domain
        self.variant = variant
        self.corrective = corrective
        self.fixed_bindings = dict(fixed_bindings)

    def __call__(self, domain: str, variant: str) -> LiveDomainRuntime:
        self._validate_request(domain, variant)
        observer = MuJoCoPhysicalStateObserver(
            self.scene, {}, fixed_bindings=self.fixed_bindings
        )
        return LiveDomainRuntime(
            controller=None,
            physical_state=observer,
            bindings={},
            fixed_bindings=self.fixed_bindings,
        )

    def for_execution(
        self,
        domain: str,
        variant: str,
        *,
        execution_plan: BaselineExecutionPlan,
        projections: Sequence[ExecutionProjection],
    ) -> LiveDomainRuntime:
        self._validate_request(domain, variant)
        bindings = self._selected_bindings(execution_plan)
        observer = MuJoCoPhysicalStateObserver(
            self.scene, bindings, fixed_bindings=self.fixed_bindings
        )
        if self.domain is Domain.KITCHEN:
            primitives = _KitchenPrimitives(
                self.scene, execution_plan, tuple(projections), bindings
            )
            controller = KitchenLiveControllerFacade(
                primitives,
                observer,
                bindings={**self.fixed_bindings, **bindings},
            )
        elif self.domain is Domain.WORKSHOP:
            primitives = _WorkshopPrimitives(
                self.scene, execution_plan, tuple(projections)
            )
            controller = WorkshopLiveControllerFacade(primitives, observer)
        else:
            controller = _build_living_controller(
                self.scene, tuple(projections), bindings, self.fixed_bindings, observer
            )
        return LiveDomainRuntime(
            controller=controller,
            physical_state=observer,
            bindings=bindings,
            fixed_bindings=self.fixed_bindings,
        )

    def _validate_request(self, domain: str, variant: str) -> None:
        if domain != self.domain.value or variant != self.variant:
            raise ProductionExecutionError("execution runtime identity mismatch")

    def _selected_bindings(
        self, execution_plan: BaselineExecutionPlan
    ) -> dict[str, EntityBinding]:
        result = getattr(self.corrective, "last_result", None)
        attempts = getattr(result, "tamp_attempts", ())
        matches = [
            item
            for item in attempts
            if item.success and item.attempt_index == execution_plan.selected_attempt_index
        ]
        if len(matches) != 1:
            raise ProductionExecutionError(
                "selected execution attempt has no unique in-memory identity result"
            )
        path = Path(matches[0].artifacts.get("identity_resolution", ""))
        if not path.is_file():
            raise ProductionExecutionError("selected identity artifact is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        movable = payload.get("movable", {})
        rows = movable.get("bindings", ()) if isinstance(movable, Mapping) else ()
        bindings = tuple(_entity_binding(row) for row in rows)
        by_id = {item.object_id: item for item in bindings}
        if len(by_id) != len(bindings):
            raise ProductionExecutionError("selected identity artifact has duplicate IDs")
        return by_id


class _KitchenPrimitives:
    """Thin plan-derived facade over the existing generic Kitchen executor."""

    def __init__(
        self,
        scene: Any,
        execution_plan: BaselineExecutionPlan,
        projections: tuple[ExecutionProjection, ...],
        bindings: Mapping[str, EntityBinding],
    ) -> None:
        from mujoco_scenes.kitchen_ground_truth_execution import (
            KitchenGroundTruthExecutionDispatcher,
        )
        from mujoco_scenes.kitchen_ground_truth_planner import GroundTruthAssignment

        inventory, resolution = _kitchen_inventory(projections, bindings)
        assignment = _kitchen_plan_assignment(
            scene, execution_plan, projections, bindings
        )
        if not isinstance(assignment, GroundTruthAssignment):
            raise AssertionError("plan-derived Kitchen assignment has wrong type")
        self.dispatcher = KitchenGroundTruthExecutionDispatcher(
            scene,
            assignment,
            inventory=inventory,
            resolution=resolution,
            assisted_suite=False,
            allow_assisted_pick_recovery=False,
        )

    def open(self, storage: str) -> Mapping[str, Any]:
        return self.dispatcher.open_container(storage.upper())

    def pick(self, object_id: str) -> Mapping[str, Any]:
        return self.dispatcher.pick(object_id)

    def place(self, object_id: str, destination: str) -> Mapping[str, Any]:
        return self.dispatcher.place(object_id, destination.lower())

    def pour(self, source_id: str, target_id: str) -> Mapping[str, Any]:
        return self.dispatcher.pour(source_id, target_id)

    def stir(self, tool_id: str, target_id: str) -> Mapping[str, Any]:
        return self.dispatcher.stir(tool_id, target_id)


def _kitchen_inventory(
    projections: Sequence[ExecutionProjection],
    bindings: Mapping[str, EntityBinding],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_by_id: dict[str, str] = {}
    for projection in projections:
        if projection.pddl_operator == "pick-from":
            source_by_id.setdefault(
                projection.pddl_arguments[0], projection.pddl_arguments[1]
            )
    rows = []
    accepted = []
    for object_id, binding in sorted(bindings.items()):
        source = source_by_id.get(object_id, "countertop")
        source_kind, container = _kitchen_source(source)
        centroid = binding.entity_centroid_m or binding.observed_centroid_m
        if centroid is None:
            raise ProductionExecutionError(f"{object_id} has no physical centroid")
        dimensions = {}
        if binding.entity_aabb_min_m and binding.entity_aabb_max_m:
            extents = np.asarray(binding.entity_aabb_max_m) - np.asarray(
                binding.entity_aabb_min_m
            )
            dimensions = {
                "length": float(extents[0]),
                "width": float(extents[1]),
                "height": float(extents[2]),
            }
        family = _kitchen_grasp_family(binding)
        functions = {
            "utensil": ("coffee_stirrer", "soup_utensil"),
            "vessel": ("coffee_vessel", "soup_bowl"),
            "source": (),
        }.get(binding.pddl_type, ())
        context = {
            "object_id": object_id,
            "observed_source_region": source,
            "observed_source_stage": 0,
            "required_workspace": "home",
            "source_container": container,
            "source_kind": source_kind,
            "container_must_be_open": container is not None,
        }
        rows.append(
            {
                "generic_object_id": object_id,
                "semantic_label": binding.pddl_type,
                "selected_functions": list(functions),
                "observed_centroid_world_m": list(centroid),
                "observed_dimensions_m": dimensions,
                "geometric_properties": (
                    {
                        "opening_width_m": {
                            "value": 0.70 * dimensions["width"],
                            "source": "PHYSICAL_COLLISION_AABB",
                        },
                        "opening_length_m": {
                            "value": 0.70 * dimensions["length"],
                            "source": "PHYSICAL_COLLISION_AABB",
                        },
                        "cavity_depth_m": {
                            "value": dimensions["height"],
                            "source": "PHYSICAL_COLLISION_AABB",
                        },
                    }
                    if binding.pddl_type == "vessel" and dimensions
                    else {}
                ),
                "source_context": context,
            }
        )
        accepted.append(
            {
                "generic_object_id": object_id,
                "physical_backend_body": binding.entity_name,
                "grasp_family": family,
                "observed_source_context": context,
                "selected_functions": list(functions),
            }
        )
    return (
        {"execution_mode": "VILAIN_TAMP_BASELINE", "objects": rows},
        {"one_to_one": True, "accepted": accepted, "rejected": []},
    )


def _kitchen_source(source: str) -> tuple[str, str | None]:
    normalized = source.lower()
    if normalized in {"d1", "d2"}:
        return "DRAWER", normalized.upper()
    if normalized in {"c1", "c2"}:
        return "CUPBOARD", normalized.upper()
    if normalized == "b1":
        return "BOX", "B1"
    return "TABLE", None


def _kitchen_grasp_family(binding: EntityBinding) -> str:
    if binding.pddl_type == "utensil":
        return "UTENSIL"
    if binding.pddl_type == "vessel":
        return "BOWL" if "bowl" in binding.entity_name.lower() else "VESSEL"
    if binding.pddl_type == "source":
        return "KETTLE" if "kettle" in binding.entity_name.lower() else "JAR_SOURCE"
    return "GENERIC"


def _kitchen_plan_assignment(
    scene: Any,
    execution_plan: BaselineExecutionPlan,
    projections: Sequence[ExecutionProjection],
    bindings: Mapping[str, EntityBinding],
) -> Any:
    from mujoco_scenes.kitchen_ground_truth_planner import GroundTruthAssignment

    pours = [item for item in projections if item.pddl_operator == "pour"]
    stirs = [item for item in projections if item.pddl_operator == "stir"]
    inserts = [item for item in projections if item.pddl_operator == "place-in"]
    sources: dict[str, str] = {}
    for item in pours:
        content = item.pddl_arguments[2]
        sources[f"{content}_source"] = item.pddl_arguments[0]
    coffee_targets = [
        {"instance_name": item.pddl_arguments[1]} for item in pours
    ]
    coffee_tools = {item.pddl_arguments[1]: item.pddl_arguments[0] for item in stirs}
    soup_assignments = [
        {
            "tool_instance": item.pddl_arguments[0],
            "target_instance": item.pddl_arguments[1],
            "tool_kind": "utensil",
            "target_kind": "vessel",
        }
        for item in inserts
    ]
    return GroundTruthAssignment(
        variant_id=str(getattr(scene, "scene_name", "vilain")),
        scene_name=str(getattr(scene, "scene_name", "vilain")),
        intended_outcome="UNKNOWN",
        is_feasible=True,
        failure_reason=None,
        coffee_targets=coffee_targets,
        soup_targets=[{"instance_name": row["target_instance"]} for row in soup_assignments],
        sources=sources,
        coffee_assignments=[],
        soup_assignments=soup_assignments,
        coffee_tools_by_target=coffee_tools,
        coffee_targets_by_tool={},
        soup_utensils_by_target={row["target_instance"]: row["tool_instance"] for row in soup_assignments},
        soup_targets_by_utensil={row["tool_instance"]: row["target_instance"] for row in soup_assignments},
        unique_coffee_tools=sorted(set(coffee_tools.values())),
        unique_soup_utensils=sorted({row["tool_instance"] for row in soup_assignments}),
    )


class _WorkshopPrimitives:
    """Plan-derived facade over the neutral Workshop execution dispatcher."""

    def __init__(
        self,
        scene: Any,
        execution_plan: BaselineExecutionPlan,
        projections: tuple[ExecutionProjection, ...],
    ) -> None:
        del execution_plan
        from mujoco_scenes.workshop_ground_truth_execution import (
            WorkshopExecutionDispatcher,
        )
        from mujoco_scenes.workshop_ground_truth_planner import WorkshopAssignment
        from mujoco_scenes.workshop_ground_truth_state import WorkshopWorldState

        drive = next(item for item in projections if item.pddl_operator == "drive")
        placement = next(
            item
            for item in projections
            if item.pddl_operator == "place-on"
            and item.pddl_arguments[0] == drive.pddl_arguments[0]
        )
        self.assignment = WorkshopAssignment(
            variant_id=str(getattr(scene, "variant_name", "vilain")),
            intended_outcome="UNKNOWN",
            is_feasible=True,
            driver=drive.controller_arguments[0],
            fastener=drive.controller_arguments[1],
            target_joint=drive.controller_arguments[2],
            work_surface=placement.controller_arguments[1],
            assignment_source="VILAIN_EXECUTION_PROJECTION",
        )
        self.state = WorkshopWorldState()
        for item in projections:
            if item.pddl_operator == "pick-from":
                self.state.object_locations[item.controller_arguments[0]] = (
                    item.controller_arguments[1]
                )
        self.dispatcher = WorkshopExecutionDispatcher(scene, self.assignment)

    def _run(self, operator: str, arguments: Sequence[str]) -> Mapping[str, Any]:
        action = {"operator": operator, "arguments": list(arguments)}
        result = self.dispatcher.execute(action, self.state)
        if result.get("success") is True:
            self.state.apply(action)
        return result

    def open(self, storage: str) -> Mapping[str, Any]:
        return self._run("OPEN", (storage,))

    def pick(self, object_id: str, source: str) -> Mapping[str, Any]:
        return self._run("PICK", (object_id, source))

    def insert(self, object_id: str, target: str) -> Mapping[str, Any]:
        return self._run("PLACE", (object_id, target))

    def place(self, object_id: str, target: str) -> Mapping[str, Any]:
        return self._run("PLACE", (object_id, target))

    def drive(self, driver: str, fastener: str, target: str) -> Mapping[str, Any]:
        return self._run("SCREW", (driver, fastener, target))


def _build_living_controller(
    scene: Any,
    projections: tuple[ExecutionProjection, ...],
    bindings: Mapping[str, EntityBinding],
    fixed_bindings: Mapping[str, EntityBinding],
    observer: Any,
) -> Any:
    from .living_controller_runtime import LivingRoomPhysicalController
    from .live_execution import LivingRoomLiveControllerFacade

    primitives = LivingRoomPhysicalController(
        scene=scene,
        projections=projections,
        bindings=bindings,
        fixed_bindings=fixed_bindings,
    )
    return LivingRoomLiveControllerFacade(primitives, observer)


class BenchmarkRegistryHiddenContextProvider:
    """Read privileged variant truth only when final evaluation requests it."""

    def __init__(self, config_root: str | Path) -> None:
        self.config_root = Path(config_root).resolve()

    def load(self, domain: str, variant: str) -> HiddenBenchmarkContext:
        from mujoco_scenes.final_paper_variant_labels import resolve_variant_name

        domain_key = domain.strip().lower().replace("-", "_")
        internal = resolve_variant_name(domain_key, variant)
        filename = {
            "kitchen": "kitchen_feasibility_variants.yaml",
            "living_room": "living_room_variants.yaml",
            "workshop": "workshop_variants.yaml",
        }[domain_key]
        document = yaml.safe_load(
            (self.config_root / filename).read_text(encoding="utf-8")
        )
        spec = document["variants"][internal]
        feasible = spec["intended_outcome"] == "FEASIBLE"
        requirements = _hidden_requirements(domain_key, document, spec)
        return HiddenBenchmarkContext(
            domain=domain_key,
            variant=variant,
            ground_truth_feasibility=feasible,
            requirements=requirements,
            evidence_artifacts=(),
        )


def _hidden_requirements(
    domain: str, document: Mapping[str, Any], spec: Mapping[str, Any]
) -> Mapping[str, Any]:
    if domain == "living_room":
        objects = document["objects"]
        regions = document["regions"]
        return {
            "left_payloads": [objects["left_cup"], objects["left_saucer"]],
            "right_payloads": [objects["right_cup"], objects["right_saucer"]],
            "remote": objects["remote"],
            "left_support": regions["PERSONAL_TABLE_LEFT"],
            "right_support": regions["PERSONAL_TABLE_RIGHT"],
            "shared_support": regions["SHARED_TABLE"],
        }
    if domain == "workshop":
        vocabulary = document["fixed_object_vocabulary"]
        return {
            "compatible_drivers": [vocabulary["manual_driver"], vocabulary["power_driver"]],
            "compatible_fasteners": [vocabulary["screw"]],
            "target": "workshop_frame_joint",
            "workbench": "MAIN_WORKBENCH_ZONE",
            "inspection_order": list(document["search_order"]),
            "storage_contents": dict(spec.get("storage_contents", {})),
            "minimum_insertion_depth_m": 0.008,
            "maximum_insertion_depth_m": 0.018,
            "radial_tolerance_m": 0.004,
            "orientation_tolerance_rad": 0.05,
        }
    required_sources = document["required_sources"]
    role_kinds = document["role_object_kinds"]
    return {
        "coffee_vessels": ["ab3_narrow_deep_cup", "ab3_medium_deep_mug"],
        "soup_vessels": ["ab3_shallow_bowl", "ab3_deep_bowl"],
        "water_sources": list(required_sources["water_source"]),
        "coffee_sources": list(required_sources["coffee_source"]),
        "suitable_stirrers": list(role_kinds["coffee_stirrer"]),
        "suitable_soup_utensils": list(role_kinds["soup_eating_utensil"]),
        "serving_support": "serving_area",
        "water_content": "water",
        "coffee_content": "coffee",
    }
