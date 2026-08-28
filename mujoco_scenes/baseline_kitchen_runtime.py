"""Live, observation-bounded kitchen runtime shared by comparison baselines.

This module owns simulator plumbing, not a planning policy.  LLM3 and
VLM-TAMP each keep their own planner, executive, prompt, and command-line
entry point while using the same cameras and calibrated physical skills.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from baseline_common.artifacts import write_json
from baseline_common.execution import observation_from_state
from baseline_common.models import Entity, Observation, Region

from .kitchen_execution_bundle import (
    DEFAULT_TASK,
    KitchenExecutionBundle,
    build_kitchen_execution_bundle,
)
from .kitchen_articulation import ARTICULATION_SPECS
from .geometry_checker import look_at_camera_rotation
from .kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from .kitchen_phase_c_execution import KitchenPhaseCExecutionDispatcher
from .kitchen_tamp_execution import KitchenExecutionObserver
from .tamp.physical_dispatcher import MuJoCoSkillDispatcher


DEFAULT_RIG_CONFIG = Path(__file__).parent / "configs" / "inspection_rigs.yaml"
PUBLIC_REGIONS = (
    Region("countertop", "countertop", "open", True),
    Region("serving_area", "serving area", "open", True),
)
STATIC_REGION_GEOMS = {
    "countertop": ("counter_surface",),
    "serving_area": ("serving_surface",),
}
KITCHEN_CAMERA_SUBSETS = {
    1: ("inspection_top",),
    3: ("inspection_left", "inspection_right", "inspection_top"),
    5: (
        "inspection_left",
        "inspection_right",
        "inspection_top",
        "inspection_front",
        "inspection_close",
    ),
}


def _unique_annotation_aliases(labels: Mapping[str, str]) -> dict[str, str]:
    totals: dict[str, int] = {}
    for label in labels.values():
        normalized = label.strip().lower().replace(" ", "_")
        totals[normalized] = totals.get(normalized, 0) + 1
    seen: dict[str, int] = {}
    result = {}
    for object_id, label in sorted(labels.items()):
        normalized = label.strip().lower().replace(" ", "_")
        seen[normalized] = seen.get(normalized, 0) + 1
        result[object_id] = (
            normalized
            if totals[normalized] == 1
            else f"{normalized}_{seen[normalized]}"
        )
    return result


def _public_region_reference(value: Any) -> Any:
    if isinstance(value, str) and value.strip().upper() in {
        "INITIAL",
        "TABLE",
        "TABLETOP",
    }:
        return "countertop"
    return value


def _effect(name: str, *arguments: str) -> str:
    return f"{name}({','.join(arguments)})"


def _goal_contract(bundle: KitchenExecutionBundle) -> "KitchenGoalContract":
    """Compile private benchmark goal facts to observable physical effects."""
    facts = bundle.symbolic_result.get("validation", {}).get("goal_facts")
    if facts is None:
        facts = bundle.symbolic_result.get("compiled", {}).get("goal", {}).get("facts")
    if facts is None:
        facts = bundle.symbolic_result.get("goal", {}).get("facts", ())
    goal_facts = [tuple(map(str, row)) for row in facts or ()]
    actions = [
        (
            str(row.get("action", "")).upper(),
            tuple(map(str, row.get("arguments", ()))),
        )
        for row in bundle.plan
    ]
    labels = {
        str(row["generic_object_id"]): str(row.get("semantic_label") or "unknown_object")
        for row in bundle.inventory.get("objects", ())
    }
    required: list[str] = []
    stir_targets: list[str] = []
    receptacle_requirements: list[tuple[str, str]] = []
    for fact in goal_facts:
        if len(fact) == 3 and fact[0] == "at":
            if fact[2] in labels:
                requirement = (fact[2], labels.get(fact[1], "unknown_object"))
                if requirement not in receptacle_requirements:
                    receptacle_requirements.append(requirement)
            else:
                candidate = _effect("placed", fact[1], fact[2])
                if candidate not in required:
                    required.append(candidate)
        elif len(fact) == 2 and fact[0] == "stirred":
            if fact[1] not in stir_targets:
                stir_targets.append(fact[1])
        elif len(fact) == 3 and fact[0] == "contains":
            matches = [
                _effect("poured", args[0], args[1])
                for action, args in actions
                if action == "POUR" and len(args) >= 2 and args[1] == fact[1]
            ]
            for match in matches:
                if match not in required:
                    required.append(match)
    return KitchenGoalContract(
        tuple(required),
        tuple(stir_targets),
        tuple(receptacle_requirements),
        labels,
    )


@dataclass(frozen=True)
class KitchenGoalContract:
    required_effects: tuple[str, ...]
    stir_targets: tuple[str, ...]
    receptacle_requirements: tuple[tuple[str, str], ...]
    object_labels: dict[str, str]

    def satisfied_by(self, observed_effects: Iterable[str]) -> bool:
        effects = set(observed_effects)
        if not set(self.required_effects) <= effects:
            return False
        parsed = [KitchenEffectLedger._parse(effect) for effect in effects]
        if any(
            not any(
                row is not None
                and row[0] == "stirred"
                and len(row[1]) == 2
                and row[1][1] == target
                for row in parsed
            )
            for target in self.stir_targets
        ):
            return False
        for target, accepted_label in self.receptacle_requirements:
            if not any(
                row is not None
                and row[0] == "placed"
                and len(row[1]) == 2
                and row[1][1] == target
                and self.object_labels.get(row[1][0]) == accepted_label
                for row in parsed
            ):
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "required_effects": list(self.required_effects),
            "stir_targets": list(self.stir_targets),
            "receptacle_requirements": [
                {"target_id": target, "accepted_label": label}
                for target, label in self.receptacle_requirements
            ],
            "visibility": "PRIVATE_EVALUATOR_ONLY",
        }


class KitchenEffectLedger:
    """Record only postconditions returned by successful physical skills."""

    def __init__(self, contract: KitchenGoalContract):
        self.contract = contract
        self.effects: set[str] = set()
        self.revision = 0

    def accept(self, effects: tuple[str, ...]) -> None:
        before = set(self.effects)
        for effect in map(str, effects):
            parsed = self._parse(effect)
            if parsed is not None and parsed[0] == "holding" and parsed[1]:
                self.effects = {
                    row for row in self.effects if not row.startswith("holding(")
                }
            elif parsed is not None and parsed[0] == "placed" and parsed[1]:
                object_id = parsed[1][0]
                self.effects = {
                    row
                    for row in self.effects
                    if not row.startswith(f"placed({object_id},")
                    and row != f"holding({object_id})"
                }
            self.effects.add(effect)
        if self.effects != before:
            self.revision += 1

    @property
    def goal_satisfied(self) -> bool:
        return self.contract.satisfied_by(self.effects)

    @staticmethod
    def _parse(effect: str) -> tuple[str, tuple[str, ...]] | None:
        if "(" not in effect or not effect.endswith(")"):
            return None
        name, raw = effect[:-1].split("(", 1)
        return name, tuple(part for part in raw.split(",") if part)

    def augment(self, observation: Observation) -> Observation:
        facts = {item.entity_id: dict(item.facts) for item in observation.entities}
        for effect in sorted(self.effects):
            parsed = self._parse(effect)
            if parsed is None:
                continue
            name, arguments = parsed
            if name == "placed" and len(arguments) == 2:
                facts.get(arguments[0], {})["region_id"] = arguments[1]
            elif name == "poured" and len(arguments) == 2:
                target = facts.get(arguments[1])
                if target is not None:
                    values = set(target.get("poured_from", ()))
                    values.add(arguments[0])
                    target["poured_from"] = sorted(values)
            elif name == "stirred" and len(arguments) == 2:
                target = facts.get(arguments[1])
                if target is not None:
                    values = set(target.get("stirred_with", ()))
                    values.add(arguments[0])
                    target["stirred_with"] = sorted(values)
        entities = tuple(
            Entity(item.entity_id, item.kind, item.label, facts[item.entity_id])
            for item in observation.entities
        )
        return Observation(
            observation.scene,
            observation.revision,
            entities,
            observation.regions,
            observation.robot,
            self.goal_satisfied,
        )


class BaselineKitchenRuntime:
    """Own one fresh Google-robot scene for a complete baseline episode."""

    def __init__(
        self,
        bundle: KitchenExecutionBundle,
        output_dir: str | Path,
        *,
        goal_contract: KitchenGoalContract | None = None,
        rig_config: str | Path = DEFAULT_RIG_CONFIG,
        image_width: int = 640,
        image_height: int = 480,
        camera_count: int = 5,
        viewer_camera: str = "free",
        show_viewer: bool = True,
    ):
        if (
            isinstance(image_width, bool)
            or not isinstance(image_width, int)
            or isinstance(image_height, bool)
            or not isinstance(image_height, int)
            or image_width <= 0
            or image_height <= 0
        ):
            raise ValueError("Baseline image dimensions must be positive integers")
        self.bundle = bundle
        self.scene = bundle.scene
        self.output_dir = Path(output_dir).resolve()
        self.rig_config = yaml.safe_load(
            Path(rig_config).read_text(encoding="utf-8")
        )
        if not isinstance(self.rig_config, dict):
            raise ValueError("Kitchen inspection rig must be a mapping")
        if not isinstance(self.rig_config.get("camera_slots"), dict):
            raise ValueError("Kitchen inspection rig requires camera_slots")
        self.camera_slots = dict(self.rig_config["camera_slots"])
        if len(self.camera_slots) != 5:
            raise ValueError("The kitchen baseline protocol requires five camera slots")
        if camera_count not in KITCHEN_CAMERA_SUBSETS:
            raise ValueError("Kitchen camera_count must be one of 1, 3, or 5")
        self.camera_count = camera_count
        self.selected_camera_slots = KITCHEN_CAMERA_SUBSETS[camera_count]
        if any(name not in self.camera_slots for name in self.selected_camera_slots):
            raise ValueError("Kitchen camera subset references an unknown camera slot")
        if any(
            not isinstance(logical_name, str)
            or not logical_name.strip()
            or not isinstance(model_name, str)
            or not model_name.strip()
            for logical_name, model_name in self.camera_slots.items()
        ):
            raise ValueError("Kitchen camera slots require non-empty string names")
        self.camera_ids: dict[str, int] = {}
        for logical_name, model_name in self.camera_slots.items():
            camera_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_CAMERA, model_name
            )
            if camera_id < 0:
                raise ValueError(f"Missing inspection camera {model_name}")
            self.camera_ids[logical_name] = camera_id
        self.active_inspection_region = "INITIAL"
        self.image_width = image_width
        self.image_height = image_height
        self.viewer_camera = viewer_camera
        self.show_viewer = show_viewer
        self.contract = goal_contract or _goal_contract(bundle)
        if not (
            self.contract.required_effects
            or self.contract.stir_targets
            or self.contract.receptacle_requirements
        ):
            raise ValueError("The private kitchen goal contract has no effects")
        self.ledger = KitchenEffectLedger(self.contract)
        self.viewer = None
        self.renderer = None
        self.status = "Initializing baseline episode"
        self._last_physical_status: str | None = None
        self.capture_index = 0
        self._cached_fingerprint: tuple[Any, ...] | None = None
        self._cached_images: tuple[dict[str, str], ...] = ()
        self._latest_visible_object_ids: frozenset[str] = frozenset()

        self.phase_b = KitchenPhaseBExecutionDispatcher(
            self.scene,
            bundle.inventory,
            bundle.resolution,
            step_callback=self._physical_step,
        )
        self.phase_c = KitchenPhaseCExecutionDispatcher(
            self.phase_b,
            bundle.registry,
            [],
        )
        self.observer = KitchenExecutionObserver(self.phase_b)
        self.dispatcher = MuJoCoSkillDispatcher(
            self.phase_c,
            inspect=self.inspect,
        )
        self.object_geom_ids = {
            str(row["generic_object_id"]): self._subtree_geom_ids(
                str(row["physical_backend_body"])
            )
            for row in bundle.resolution.get("accepted", ())
        }
        self.object_annotation_aliases = _unique_annotation_aliases(
            {
                object_id: self.contract.object_labels.get(
                    object_id, "unknown_object"
                )
                for object_id in self.object_geom_ids
            }
        )
        self.region_geom_ids = {
            region_id: self._subtree_geom_ids(spec.moving_body)
            for region_id, spec in ARTICULATION_SPECS.items()
        }
        self.region_geom_ids.update(
            {
                region_id: self._named_geom_ids(geom_names)
                for region_id, geom_names in STATIC_REGION_GEOMS.items()
            }
        )
        missing_region_annotations = {
            region.region_id for region in PUBLIC_REGIONS
        } - self.region_geom_ids.keys()
        if missing_region_annotations:
            missing = ", ".join(sorted(missing_region_annotations))
            raise ValueError(f"Public regions lack annotation geometry: {missing}")
        write_json(
            self.output_dir / "_private_evaluation" / "goal_contract.json",
            self.contract.as_dict(),
        )
        write_json(
            self.output_dir / "shared_observation_contract.json",
            {
                "schema_version": 1,
                "camera_count": self.camera_count,
                "camera_ids": list(self.selected_camera_slots),
                "camera_ablation": "NESTED_TOP_LEFT_RIGHT_THEN_ALL",
                "rgb_annotation": "UNIQUE_SEMANTIC_ALIASES_ONLY",
                "textualized_state": (
                    "VISIBLE_IDS_OBSERVABLE_RELATIONS_REGION_STATE_ROBOT_STATE"
                ),
                "object_semantic_labels_exposed": True,
                "alias_to_planning_id_map_exposed": True,
                "functional_roles_exposed": False,
                "semantic_detector_outputs_exposed": False,
                "instance_correspondence_source": "MUJOCO_INSTANCE_SEGMENTATION",
                "instance_correspondence_is_oracle": True,
                "shared_by": ["llm3", "vlm_tamp", "owl_tamp"],
            },
        )

    def _subtree_geom_ids(self, root_body_name: str) -> frozenset[int]:
        root = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, root_body_name
        )
        if root < 0:
            raise ValueError(f"Missing annotation body {root_body_name}")
        result = set()
        for geom_id, body_id in enumerate(self.scene.model.geom_bodyid):
            current = int(body_id)
            while current > 0:
                if current == root:
                    result.add(geom_id)
                    break
                current = int(self.scene.model.body_parentid[current])
        if not result:
            raise ValueError(f"Annotation body {root_body_name} has no visible geoms")
        return frozenset(result)

    def _named_geom_ids(self, geom_names: Iterable[str]) -> frozenset[int]:
        result = set()
        for geom_name in geom_names:
            geom_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id < 0:
                raise ValueError(f"Missing annotation geom {geom_name}")
            result.add(geom_id)
        return frozenset(result)

    @classmethod
    def from_phase1(
        cls,
        phase1_run_dir: str | Path,
        output_dir: str | Path,
        *,
        task_requirements: str | Path = DEFAULT_TASK,
        **kwargs: Any,
    ) -> "BaselineKitchenRuntime":
        output = Path(output_dir).resolve()
        bundle = build_kitchen_execution_bundle(
            phase1_run_dir,
            output_dir=output / "_private_evaluation",
            task_requirements=task_requirements,
            include_all_observed_objects=True,
        )
        return cls(bundle, output, **kwargs)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> "BaselineKitchenRuntime":
        """Build an anonymous-ID benchmark observation without Phase-1 semantics.

        This factory is planning-only. MuJoCo names and all hidden instances
        remain adapter-private; the VLM sees only IDs visible in the five RGB
        views and the semantic-neutral state.
        """
        from copy import deepcopy

        import yaml

        from .final_paper_variant_labels import resolve_variant_name
        from .kitchen_execution_bundle import KitchenExecutionBundle
        from .kitchen_ground_truth_execution import (
            build_oracle_inventory_and_resolution,
        )
        from .scene_loader import KITCHEN_FEASIBILITY_VARIANTS, KitchenScene

        internal = resolve_variant_name("kitchen", variant)
        document = yaml.safe_load(
            KITCHEN_FEASIBILITY_VARIANTS.read_text(encoding="utf-8")
        )
        variants = document.get("variants", {}) if isinstance(document, dict) else {}
        if internal not in variants:
            raise ValueError(f"Unknown Kitchen variant {variant!r}")
        scene = KitchenScene(
            str(variants[internal]["scene_name"]), include_robot=True, robot="google"
        )
        raw_inventory, raw_resolution = build_oracle_inventory_and_resolution(scene)
        backend_names = sorted(
            str(row["generic_object_id"])
            for row in raw_inventory.get("objects", ())
        )
        generic_for_backend = {
            backend: f"object_{index:04d}"
            for index, backend in enumerate(backend_names, start=1)
        }
        private_label_for_backend = {
            str(row["physical_backend_body"]): str(
                row.get("semantic_label", "unknown_object")
            )
            for row in raw_resolution.get("accepted", ())
        }
        inventory_rows = []
        for source in raw_inventory.get("objects", ()):
            row = deepcopy(source)
            backend = str(row["generic_object_id"])
            generic = generic_for_backend[backend]
            row["generic_object_id"] = generic
            row["semantic_label"] = private_label_for_backend[backend]
            row["selected_functions"] = []
            row["source_context"]["object_id"] = generic
            inventory_rows.append(row)
        accepted = []
        for source in raw_resolution.get("accepted", ()):
            row = deepcopy(source)
            backend = str(row["physical_backend_body"])
            generic = generic_for_backend[backend]
            row["generic_object_id"] = generic
            row["source_context"]["object_id"] = generic
            accepted.append(row)
        inventory = {
            "execution_mode": "PLANNING_ONLY_ANONYMOUS_INSTANCE_IDS",
            "planner_received_backend_names": False,
            "scene_name": scene.scene_name,
            "objects": inventory_rows,
        }
        resolution = {
            "one_to_one": True,
            "all_resolved": True,
            "accepted": accepted,
            "rejected": [],
        }
        registry = {
            "objects": {
                row["generic_object_id"]: {
                    "generic_object_id": row["generic_object_id"]
                }
                for row in inventory_rows
            }
        }
        output = Path(output_dir).resolve()
        bundle = KitchenExecutionBundle(
            output,
            scene,
            inventory,
            resolution,
            registry,
            {},
            [],
            {},
        )
        labels = {
            str(row["generic_object_id"]): str(row.get("semantic_label", "unknown"))
            for row in accepted
        }
        contract = KitchenGoalContract(("__planning_goal_unset__",), (), (), labels)
        runtime = cls(bundle, output, goal_contract=contract, **kwargs)
        write_json(
            output / "_private_evaluation" / "variant_adapter.json",
            {
                "variant": variant,
                "internal_variant": internal,
                "generic_for_backend": generic_for_backend,
                "model_visible": False,
            },
        )
        return runtime

    def open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.show_viewer:
            return
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.scene.model, self.scene.data)
        if self.viewer_camera == "free":
            mujoco.mjv_defaultFreeCamera(self.scene.model, self.viewer.cam)
        else:
            camera_id = mujoco.mj_name2id(
                self.scene.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                self.viewer_camera,
            )
            if camera_id < 0:
                raise ValueError(f"Unknown viewer camera: {self.viewer_camera}")
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = camera_id
        self.sync("Ready for model planning")

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def wait_for_viewer(self) -> None:
        while self.viewer is not None and self.viewer.is_running():
            self.sync(self.status)
            time.sleep(1.0 / 60.0)

    def sync(self, status: str) -> None:
        self.status = status
        if self.viewer is None or not self.viewer.is_running():
            return
        if hasattr(self.viewer, "set_texts"):
            self.viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Baseline kitchen execution",
                    status,
                )
            )
        self.viewer.sync()

    def _physical_step(self) -> None:
        controller = self.phase_b.manipulation.executor
        physical_status = str(controller.status)
        if physical_status != self._last_physical_status:
            self._last_physical_status = physical_status
            print(f"[physical] {physical_status}", flush=True)
        top_level = self.status.split(" | controller: ", 1)[0]
        self.sync(f"{top_level} | controller: {physical_status}")

    def inspect(self, region_id: str) -> dict[str, Any]:
        self.sync(f"Inspecting {region_id}")
        if region_id not in self.scene.get_region_observation_states():
            return {
                "success": False,
                "failure_code": "UNKNOWN_REGION",
                "message": f"Unknown inspectable kitchen region {region_id}",
            }
        if region_id in self.phase_b.physically_open_containers():
            self.scene.record_container_opened(region_id)
            self.active_inspection_region = region_id
            return {"success": True, "status": "ALREADY_INSPECTED"}
        result = self.phase_b.phase_a.request("OPEN", region_id, execute=True)
        if result.get("success"):
            self.active_inspection_region = region_id
        return result

    def accept_effects(self, effects: tuple[str, ...]) -> None:
        self.ledger.accept(effects)
        if effects:
            self.sync("Observed: " + ", ".join(effects))

    def goal_verifier(self, observation: Observation) -> bool:
        return observation.goal_satisfied and self.ledger.goal_satisfied

    def probe_base_motion(self, source: str, target: str) -> dict[str, Any] | None:
        """Plan an RRT* base path on copied MuJoCo state for a TAMP stream.

        This is a read-only certificate: the live robot, controller targets,
        ledger, and viewer are never advanced by the planning query.
        """
        from .kitchen_execution_policy import (
            KitchenWorkspace,
            WORKSPACE_DESTINATIONS,
        )
        from .mobile_motion import MobileMoveExecutor, physical_location

        try:
            source_destination = WORKSPACE_DESTINATIONS[KitchenWorkspace(source)]
            target_destination = WORKSPACE_DESTINATIONS[KitchenWorkspace(target)]
        except (KeyError, ValueError):
            return None
        copied = mujoco.MjData(self.scene.model)
        for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active"):
            destination = getattr(copied, name, None)
            current = getattr(self.scene.data, name, None)
            if destination is not None and current is not None:
                destination[...] = current
        copied.time = self.scene.data.time
        planner = MobileMoveExecutor(self.scene.model, copied, "google")
        source_physical = physical_location(source_destination)
        pose = planner.physical_poses[source_physical]
        joints = list(planner.joint_addresses)
        actuators = list(planner.actuator_ids)
        base_command = np.array(
            (pose.y - planner.home_pose.y, -pose.x, pose.yaw), dtype=float
        )
        copied.qpos[joints] = base_command
        copied.qvel[list(planner.joint_velocity_addresses)] = 0.0
        copied.ctrl[actuators] = base_command
        planner.current_physical_location = source_physical
        planner.current_symbolic_location = source
        mujoco.mj_forward(self.scene.model, copied)
        try:
            planner.request_move(target_destination)
        except (RuntimeError, ValueError):
            return None
        return {
            "source_workspace": source,
            "target_workspace": target,
            "planner": "RRTStarPlanner",
            "waypoint_count": len(planner.targets),
            "waypoints": [list(map(float, row)) for row in planner.targets],
            "live_state_mutated": False,
        }

    def observe(self) -> tuple[Observation, tuple[dict[str, str], ...]]:
        bounded = self.observe_state()
        return bounded, self._images()

    def observe_state(self) -> Observation:
        """Observe symbolic state without paying for five RGB renders."""
        state = self.observer()
        raw = observation_from_state("kitchen", state)
        sanitized_entities = tuple(
            Entity(
                item.entity_id,
                item.kind,
                self.object_annotation_aliases.get(item.entity_id, item.entity_id),
                {
                    key: (
                        _public_region_reference(value)
                        if key in {"source_region", "region_id"}
                        else value
                    )
                    for key, value in item.facts.items()
                    if key in {"dimensions_m", "source_region", "location", "region_id"}
                },
            )
            for item in raw.entities
        )
        existing_regions = {item.region_id: item for item in raw.regions}
        for region in PUBLIC_REGIONS:
            existing_regions.setdefault(region.region_id, region)
        bounded = Observation(
            raw.scene,
            raw.revision,
            sanitized_entities,
            tuple(existing_regions.values()),
            raw.robot,
            False,
        )
        bounded = self.ledger.augment(bounded)
        self._latest_visible_object_ids = bounded.object_ids
        write_json(
            self.output_dir / "latest_observation.json",
            bounded.as_semantic_neutral_prompt_dict(),
        )
        write_json(
            self.output_dir / "_private_evaluation" / "latest_observation.json",
            bounded.as_prompt_dict(),
        )
        return bounded

    @staticmethod
    def _segmentation_box(
        segmentation: np.ndarray, geom_ids: frozenset[int]
    ) -> tuple[int, int, int, int, int] | None:
        mask = BaselineKitchenRuntime._segmentation_mask(segmentation, geom_ids)
        rows, columns = np.nonzero(mask)
        if len(rows) < 6:
            return None
        return (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
            int(len(rows)),
        )

    @staticmethod
    def _segmentation_mask(
        segmentation: np.ndarray, geom_ids: frozenset[int]
    ) -> np.ndarray:
        if not geom_ids:
            return np.zeros(segmentation.shape[:2], dtype=bool)
        return (
            (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            & np.isin(segmentation[:, :, 0], tuple(geom_ids))
        )

    @staticmethod
    def _draw_mask_outline(
        draw: ImageDraw.ImageDraw,
        mask: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        padded = np.pad(mask, 1, constant_values=False)
        interior = mask.copy()
        interior &= padded[:-2, 1:-1]
        interior &= padded[2:, 1:-1]
        interior &= padded[1:-1, :-2]
        interior &= padded[1:-1, 2:]
        boundary = mask & ~interior
        thick = boundary.copy()
        boundary_padded = np.pad(boundary, 1, constant_values=False)
        thick |= boundary_padded[:-2, 1:-1]
        thick |= boundary_padded[2:, 1:-1]
        thick |= boundary_padded[1:-1, :-2]
        thick |= boundary_padded[1:-1, 2:]
        thick &= mask
        bitmap = Image.fromarray(np.where(thick, 255, 0).astype(np.uint8))
        draw.bitmap((0, 0), bitmap, fill=color)

    @staticmethod
    def _intersection_area(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        return max(0, right - left + 1) * max(0, bottom - top + 1)

    @staticmethod
    def _draw_id_annotation(
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int, int],
        identifier: str,
        color: tuple[int, int, int],
        font: ImageFont.ImageFont,
        *,
        image_size: tuple[int, int],
        occupied_labels: list[tuple[int, int, int, int]],
        forbidden_boxes: tuple[tuple[int, int, int, int], ...],
        draw_box: bool = True,
        avoid_target: bool = True,
    ) -> None:
        left, top, right, bottom, _pixels = box
        if draw_box:
            draw.rectangle((left, top, right, bottom), outline=color, width=2)
        text_box = draw.textbbox((left, top), identifier, font=font)
        width = text_box[2] - text_box[0] + 4
        height = text_box[3] - text_box[1] + 4
        image_width, image_height = image_size
        inside_candidates = [
            ((left + right - width) // 2, top + 4),
            ((left + right - width) // 2, bottom - height - 4),
            ((left + right - width) // 2, (top + bottom - height) // 2),
        ]
        outside_candidates = [
            (left, top - height - 2),
            (right - width + 1, top - height - 2),
            (left, bottom + 2),
            (right - width + 1, bottom + 2),
            (right + 2, top),
            (left - width - 2, top),
        ]
        candidates = (
            inside_candidates + outside_candidates
            if not avoid_target
            else outside_candidates + inside_candidates
        )
        max_x = max(0, image_width - width - 1)
        max_y = max(0, image_height - height - 1)
        for y in range(0, max_y + 1, height + 2):
            candidates.extend(((0, y), (max_x, y)))
        for x in range(0, max_x + 1, width + 2):
            candidates.extend(((x, 0), (x, max_y)))

        def clamp(candidate: tuple[int, int]) -> tuple[int, int, int, int]:
            x = min(max(0, candidate[0]), max(0, image_width - width - 1))
            y = min(max(0, candidate[1]), max(0, image_height - height - 1))
            return (x, y, x + width, y + height)

        target = (left, top, right, bottom)
        ranked = []
        for order, candidate in enumerate(candidates):
            label = clamp(candidate)
            target_overlap = (
                BaselineKitchenRuntime._intersection_area(label, target)
                if avoid_target
                else 0
            )
            object_overlap = sum(
                BaselineKitchenRuntime._intersection_area(label, forbidden)
                for forbidden in forbidden_boxes
            )
            label_overlap = sum(
                BaselineKitchenRuntime._intersection_area(label, occupied)
                for occupied in occupied_labels
            )
            ranked.append(
                (
                    target_overlap * 10_000
                    + object_overlap * 5_000
                    + label_overlap * 100
                    + order,
                    label,
                )
            )
        _score, label = min(ranked, key=lambda row: row[0])
        occupied_labels.append(label)
        label_left, label_top, label_right, label_bottom = label
        draw.rectangle(label, fill=(0, 0, 0))
        draw.text((label_left + 2, label_top + 2), identifier, fill=color, font=font)

    def _annotate_frame(
        self, frame: np.ndarray, segmentation: np.ndarray
    ) -> tuple[Image.Image, dict[str, Any]]:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default(size=11)
        except TypeError:
            font = ImageFont.load_default()
        manifest: dict[str, Any] = {"objects": [], "regions": []}
        region_rows = []
        for region_id, geom_ids in sorted(self.region_geom_ids.items()):
            box = self._segmentation_box(segmentation, geom_ids)
            if box is None:
                continue
            region_rows.append((region_id, geom_ids, box))
            manifest["regions"].append(
                {"id": region_id, "bbox_xyxy": list(box[:4]), "pixel_count": box[4]}
            )
        object_rows = []
        annotation_aliases = getattr(
            self,
            "object_annotation_aliases",
            _unique_annotation_aliases(self.contract.object_labels),
        )
        for object_id in sorted(self._latest_visible_object_ids):
            box = self._segmentation_box(
                segmentation, self.object_geom_ids.get(object_id, frozenset())
            )
            if box is None:
                continue
            object_rows.append((object_id, box))
            manifest["objects"].append(
                {
                    "id": object_id,
                    "semantic_label": annotation_aliases.get(
                        object_id, "unknown_object"
                    ),
                    "bbox_xyxy": list(box[:4]),
                    "pixel_count": box[4],
                }
            )
        object_boxes = tuple(tuple(row[1][:4]) for row in object_rows)
        occupied_labels: list[tuple[int, int, int, int]] = []
        for region_id, geom_ids, box in region_rows:
            is_static_surface = region_id in STATIC_REGION_GEOMS
            if is_static_surface:
                self._draw_mask_outline(
                    draw,
                    self._segmentation_mask(segmentation, geom_ids),
                    (255, 190, 40),
                )
            self._draw_id_annotation(
                draw,
                box,
                region_id,
                (255, 190, 40),
                font,
                image_size=image.size,
                occupied_labels=occupied_labels,
                forbidden_boxes=object_boxes,
                draw_box=not is_static_surface,
                avoid_target=not is_static_surface,
            )
        for object_id, box in object_rows:
            semantic_name = annotation_aliases.get(
                object_id, "unknown_object"
            )
            self._draw_id_annotation(
                draw,
                box,
                semantic_name.replace("_", " "),
                (40, 235, 255),
                font,
                image_size=image.size,
                occupied_labels=occupied_labels,
                forbidden_boxes=object_boxes,
            )
        return image, manifest

    def _fingerprint(self) -> tuple[Any, ...]:
        regions = self.scene.get_region_observation_states()
        held = self.phase_b.manipulation.executor.held_object
        return (
            self.ledger.revision,
            held,
            tuple(
                (name, bool(row["open"]), bool(row["inspected"]))
                for name, row in sorted(regions.items())
            ),
            round(float(self.scene.data.time), 6),
            self.active_inspection_region,
        )

    def _configure_inspection_rig(self) -> None:
        region = self.rig_config["regions"][self.active_inspection_region]
        target_base = np.asarray(region["target_world_m"], dtype=float)
        rig_position = np.asarray(region["rig_position_world_m"], dtype=float)
        up_world = np.asarray(region["up_world"], dtype=float)
        for logical_name in self.camera_slots:
            camera_id = self.camera_ids[logical_name]
            camera = region["cameras"][logical_name]
            position = rig_position + np.asarray(camera["position_offset_m"], dtype=float)
            target = target_base + np.asarray(
                camera.get("look_at_offset_m", (0.0, 0.0, 0.0)), dtype=float
            )
            rotation = look_at_camera_rotation(position, target, up_world)
            quaternion = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
            self.scene.model.cam_pos[camera_id] = position
            self.scene.model.cam_quat[camera_id] = quaternion
            self.scene.model.cam_mat0[camera_id] = rotation.reshape(-1)
            self.scene.model.cam_mode[camera_id] = int(
                mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            )
            self.scene.model.cam_targetbodyid[camera_id] = -1
            self.scene.model.cam_fovy[camera_id] = float(
                camera.get("fovy_degrees", 60.0)
            )

    def _images(self) -> tuple[dict[str, str], ...]:
        fingerprint = self._fingerprint()
        if fingerprint == self._cached_fingerprint:
            return self._cached_images
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.scene.model,
                height=self.image_height,
                width=self.image_width,
            )
        self.capture_index += 1
        frame_dir = self.output_dir / "observations" / f"{self.capture_index:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        self._configure_inspection_rig()
        mujoco.mj_forward(self.scene.model, self.scene.data)
        images = []
        annotation_manifest: dict[str, Any] = {
            "schema_version": 1,
            "semantics_exposed": True,
            "object_annotation": "UNIQUE_SEMANTIC_ALIAS_ONLY",
            "region_annotation": "PERSISTENT_REGION_ID_ONLY",
            "cameras": {},
        }
        for logical_name in self.selected_camera_slots:
            camera = self.camera_slots[logical_name]
            camera_id = self.camera_ids[logical_name]
            self.renderer.update_scene(self.scene.data, camera=camera_id)
            frame = self.renderer.render()
            raw_image = Image.fromarray(frame)
            raw_path = frame_dir / f"raw_{camera}.png"
            raw_image.save(raw_path, format="PNG")
            self.renderer.enable_segmentation_rendering()
            segmentation = self.renderer.render()
            self.renderer.disable_segmentation_rendering()
            image, camera_manifest = self._annotate_frame(frame, segmentation)
            annotation_manifest["cameras"][logical_name] = camera_manifest
            path = frame_dir / f"{camera}.png"
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            path.write_bytes(image_bytes)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            images.append(
                {
                    "camera": logical_name,
                    "data_url": f"data:image/png;base64,{encoded}",
                }
            )
        write_json(frame_dir / "annotations.json", annotation_manifest)
        if not images:
            raise RuntimeError("No configured kitchen camera could be rendered")
        self._cached_fingerprint = fingerprint
        self._cached_images = tuple(images)
        self.sync(f"Captured observation {self.capture_index}")
        return self._cached_images


__all__ = [
    "BaselineKitchenRuntime",
    "DEFAULT_RIG_CONFIG",
    "KitchenEffectLedger",
    "KitchenGoalContract",
    "KITCHEN_CAMERA_SUBSETS",
]
