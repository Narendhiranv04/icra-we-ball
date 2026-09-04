"""Live controller facades, physical state extraction, and runner bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .artifacts import atomic_write_json
from .contracts import BaselineExecutionPlan, ExecutionProjection, SerializableContract
from .evaluation import TerminalStateSnapshot
from .execution.base import project_plan
from .execution.kitchen import (
    KitchenExecutionAdapter,
    build_kitchen_inventory,
)
from .execution.living_room import LivingRoomExecutionAdapter
from .execution.workshop import (
    WorkshopExecutionAdapter,
    build_workshop_controller_contract,
)
from .identity import EntityBinding
from .runner import ExecutionStageResult


class LiveExecutionError(ValueError):
    """Raised when live runtime wiring violates the execution boundary."""


class PhysicalStateObserver(Protocol):
    def capture_object(self, entity: str) -> Mapping[str, Any]: ...

    def verify_open(self, entity: str) -> Mapping[str, Any]: ...

    def verify_pick(
        self, entity: str, before: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def verify_place(
        self, entity: str, target: str, *, contained: bool
    ) -> Mapping[str, Any]: ...

    def verify_skill(
        self,
        operator: str,
        arguments: Sequence[str],
        controller_result: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def record_action(
        self,
        operator: str,
        arguments: Sequence[str],
        controller_result: Mapping[str, Any],
    ) -> None: ...

    def snapshot(self, domain: str, predicted_infeasible: bool) -> TerminalStateSnapshot: ...


class KitchenLiveControllerFacade:
    """Map kitchen controller requests to generic physical primitives."""

    _METHODS = {"OPEN": "open", "PICK": "pick", "PLACE": "place", "POUR": "pour", "STIR": "stir"}

    def __init__(
        self,
        primitives: Any,
        physical_state: PhysicalStateObserver,
        *,
        bindings: Mapping[str, EntityBinding] | None = None,
    ) -> None:
        self.primitives = primitives
        self.physical_state = physical_state
        supplied_bindings = dict(bindings or {})
        if any(key != value.object_id for key, value in supplied_bindings.items()):
            raise LiveExecutionError("kitchen binding keys must match object IDs")
        if len({value.entity_name for value in supplied_bindings.values()}) != len(
            supplied_bindings
        ):
            raise LiveExecutionError("kitchen controller bindings must be one-to-one")
        self.primitive_ids = {
            value.entity_name: value.object_id
            for value in supplied_bindings.values()
        }
        self._before: dict[str, Mapping[str, Any]] = {}

    def execute_action(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        operator, arguments = _request(request, self._METHODS)
        if operator not in self._METHODS:
            return {"success": False, "status": "UNSUPPORTED_CONTROLLER_ACTION"}
        if operator == "PICK":
            self._before[str(request["action_instance_id"])] = self.physical_state.capture_object(arguments[0])
        primitive_arguments = tuple(self.primitive_ids.get(item, item) for item in arguments)
        result = _invoke_primitive(
            self.primitives, self._METHODS[operator], primitive_arguments, operator
        )
        return result

    def verify_postcondition(
        self,
        request: Mapping[str, Any],
        controller_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        operator, arguments = _request(request, self._METHODS)
        if operator not in self._METHODS:
            return {"success": False, "reason": "unsupported controller action"}
        pddl_operator = str(request.get("pddl_operator", "")).lower().replace("_", "-")
        if operator == "OPEN":
            result = self.physical_state.verify_open(arguments[0])
        elif operator == "PICK":
            before = self._before.get(str(request["action_instance_id"]), {})
            result = self.physical_state.verify_pick(arguments[0], before)
        elif operator == "PLACE":
            result = self.physical_state.verify_place(
                arguments[0], arguments[1], contained=pddl_operator == "place-in"
            )
        else:
            result = self.physical_state.verify_skill(
                operator, arguments, controller_result
            )
        if result.get("success") is True:
            self.physical_state.record_action(operator, arguments, controller_result)
        return result


class LivingRoomLiveControllerFacade:
    """Satisfy the living-room motion protocols using physical primitives."""

    def __init__(
        self,
        primitives: Any,
        physical_state: PhysicalStateObserver,
        *,
        navigation_target: Callable[[str], Any] | None = None,
        placement_target: Callable[[str, str, str, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.primitives = primitives
        self.physical_state = physical_state
        self.navigation_target = navigation_target
        self.placement_target = placement_target
        self._before: dict[str, Mapping[str, Any]] = {}

    def move_to(self, target_entity: str, *, carrying_entity: str | None) -> Mapping[str, Any]:
        direct = getattr(self.primitives, "move_to", None)
        if callable(direct):
            return _mapping_result(direct(target_entity, carrying_entity=carrying_entity))
        if self.navigation_target is None:
            raise LiveExecutionError("living-room navigation target resolver is missing")
        goal = self.navigation_target(target_entity)
        path = self.primitives.plan(goal)
        return _mapping_result(self.primitives.execute(path))

    def pick(self, payload_entity: str) -> Mapping[str, Any]:
        self._before[payload_entity] = self.physical_state.capture_object(payload_entity)
        result = _invoke_primitive(self.primitives, "pick", (payload_entity,), "PICK")
        return result

    def verify_held(self, payload_entity: str) -> Mapping[str, Any]:
        return self.physical_state.verify_pick(payload_entity, self._before.get(payload_entity, {}))

    def destination_for(
        self,
        *,
        payload_id: str,
        payload_entity: str,
        support_id: str,
        support_entity: str,
    ) -> Mapping[str, Any]:
        if self.placement_target is not None:
            return dict(self.placement_target(payload_id, payload_entity, support_id, support_entity))
        resolver = getattr(self.primitives, "destination_for", None)
        if not callable(resolver):
            raise LiveExecutionError("living-room placement target resolver is missing")
        return _mapping_result(
            resolver(
                payload_id=payload_id,
                payload_entity=payload_entity,
                support_id=support_id,
                support_entity=support_entity,
            )
        )

    def place(
        self,
        payload_entity: str,
        support_entity: str,
        destination: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        result = _mapping_result(self.primitives.place(payload_entity, support_entity, destination))
        return result

    def verify_physical_on(
        self,
        payload_entity: str,
        support_entity: str,
        destination: Mapping[str, Any],
        place_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del destination
        result = dict(
            self.physical_state.verify_place(
                payload_entity, support_entity, contained=False
            )
        )
        verified = result.get("success") is True
        if verified:
            self.physical_state.record_action(
                "PLACE", (payload_entity, support_entity), place_result
            )
        return {**result, "relation": "ON", "verified": verified}


class WorkshopLiveControllerFacade:
    """Map workshop requests to open/pick/place/insert/drive primitives."""

    def __init__(self, primitives: Any, physical_state: PhysicalStateObserver) -> None:
        self.primitives = primitives
        self.physical_state = physical_state
        self._before: dict[str, Mapping[str, Any]] = {}

    def execute_action(self, request: Mapping[str, Any], controller_contract: Any) -> Mapping[str, Any]:
        del controller_contract
        operator = str(request.get("operator", "")).upper()
        pddl_operator = str(request.get("pddl_operator", "")).lower().replace("_", "-")
        arguments = _string_arguments(request)
        method = {
            "OPEN": "open",
            "PICK": "pick",
            "PLACE": "insert" if pddl_operator == "insert" else "place",
            "SCREW": "drive",
        }.get(operator)
        if method is None:
            return {"success": False, "status": "UNSUPPORTED_CONTROLLER_ACTION"}
        if operator == "PICK":
            self._before[str(request["action_instance_id"])] = self.physical_state.capture_object(arguments[0])
        result = _invoke_primitive(self.primitives, method, arguments, operator)
        return result

    def verify_postcondition(
        self,
        request: Mapping[str, Any],
        controller_result: Mapping[str, Any],
        controller_contract: Any,
    ) -> Mapping[str, Any]:
        del controller_contract
        operator = str(request.get("operator", "")).upper()
        pddl_operator = str(request.get("pddl_operator", "")).lower().replace("_", "-")
        arguments = _string_arguments(request)
        if operator == "OPEN":
            result = self.physical_state.verify_open(arguments[0])
        elif operator == "PICK":
            result = self.physical_state.verify_pick(
                arguments[0], self._before.get(str(request["action_instance_id"]), {})
            )
        elif operator == "PLACE":
            result = self.physical_state.verify_place(
                arguments[0], arguments[1], contained=pddl_operator == "insert"
            )
        else:
            result = self.physical_state.verify_skill(
                "DRIVE", arguments, controller_result
            )
        if result.get("success") is True:
            recorded_operator = "INSERT" if pddl_operator == "insert" else operator
            self.physical_state.record_action(
                recorded_operator, arguments, controller_result
            )
        return result


@dataclass(frozen=True)
class LiveDomainRuntime:
    """Execution-only dependencies resolved for one domain variant."""

    controller: Any
    physical_state: PhysicalStateObserver
    bindings: Mapping[str, EntityBinding]
    fixed_bindings: Mapping[str, EntityBinding] = field(default_factory=dict)


class LiveExecutionStage:
    """Execute projections through domain adapters and persist terminal evidence."""

    def __init__(self, runtime_provider: Callable[[str, str], LiveDomainRuntime]) -> None:
        self.runtime_provider = runtime_provider

    def execute(
        self,
        *,
        domain: str,
        variant: str,
        execution_plan: BaselineExecutionPlan,
        projections: Sequence[ExecutionProjection],
        output_root: Path,
    ) -> ExecutionStageResult:
        domain_key = _domain_key(domain)
        runtime = self.runtime_provider(domain_key, variant)
        if not isinstance(runtime, LiveDomainRuntime):
            raise LiveExecutionError("runtime provider returned an invalid runtime")
        projections = tuple(projections)
        _validate_execution_input(domain_key, execution_plan, projections, runtime)
        output_root = Path(output_root)
        entity_path = atomic_write_json(
            output_root / "entity_resolution.json",
            {
                "bindings": [binding.to_dict() for binding in (*runtime.bindings.values(), *runtime.fixed_bindings.values())],
                "source": "VILAIN_ENTITY_BINDINGS",
            },
        )

        if domain_key == "kitchen":
            inventory = build_kitchen_inventory(
                projections, runtime.bindings, fixed_bindings=runtime.fixed_bindings
            )
            result = KitchenExecutionAdapter(controller=runtime.controller, inventory=inventory).execute(projections)
            effects = tuple(result.effect_ledger)
        elif domain_key == "living_room":
            controller = runtime.controller
            result = LivingRoomExecutionAdapter(mobile=controller, picker=controller, placer=controller).execute(projections)
            effects = ()
        elif domain_key == "workshop":
            contract = build_workshop_controller_contract(projections)
            result = WorkshopExecutionAdapter(dispatcher=runtime.controller, controller_contract=contract).execute(projections)
            effects = tuple(result.effect_ledger)
        else:
            raise LiveExecutionError(f"unsupported live execution domain: {domain!r}")

        terminal = runtime.physical_state.snapshot(domain_key, False)
        trace_path = atomic_write_json(
            output_root / "execution_trace.json",
            {"domain": domain_key, "variant": variant, "actions": [item.to_dict() for item in result.actions]},
        )
        ledger_path = atomic_write_json(
            output_root / "effect_ledger.json",
            {"effects": [_serializable(item) for item in effects]},
        )
        terminal_path = atomic_write_json(output_root / "terminal_state.json", terminal.to_dict())
        status = "SUCCESS" if result.success else "FAILED"
        payload = {
            "status": status,
            "success": result.success,
            "terminal_failure_code": (
                result.terminal_failure_code.value
                if getattr(result.terminal_failure_code, "value", None) is not None
                else result.terminal_failure_code
            ),
            "terminal_failure_message": result.terminal_failure_message,
            "actions_attempted": len(result.actions),
            "actions_planned": len(projections),
        }
        result_path = atomic_write_json(output_root / "execution_result.json", payload)
        return ExecutionStageResult(
            status=status,
            success=result.success,
            terminal_state=terminal,
            effect_ledger=effects,
            artifact_paths={
                "execution_entity_resolution": str(entity_path),
                "execution_trace": str(trace_path),
                "effect_ledger": str(ledger_path),
                "terminal_state": str(terminal_path),
                "execution_result": str(result_path),
            },
            metrics={
                "actions_planned": len(projections),
                "actions_attempted": len(result.actions),
                "actions_succeeded": sum(item.success for item in result.actions),
                "verified_effects": len(effects),
            },
        )

    def terminal_without_execution(
        self, *, domain: str, variant: str, predicted_infeasible: bool
    ) -> TerminalStateSnapshot:
        runtime = self.runtime_provider(_domain_key(domain), variant)
        if not isinstance(runtime, LiveDomainRuntime):
            raise LiveExecutionError("runtime provider returned an invalid runtime")
        return runtime.physical_state.snapshot(_domain_key(domain), predicted_infeasible)


class MuJoCoPhysicalStateObserver:
    """Extract neutral contact, pose, containment, and articulation evidence."""

    def __init__(
        self,
        scene: Any,
        bindings: Mapping[str, EntityBinding],
        *,
        fixed_bindings: Mapping[str, EntityBinding] | None = None,
        robot_body_prefixes: Sequence[str] = ("google",),
        penetration_tolerance_m: float = 0.005,
        linear_stability_m_s: float = 0.03,
        angular_stability_rad_s: float = 0.10,
        mujoco_module: Any | None = None,
    ) -> None:
        if not hasattr(scene, "model") or not hasattr(scene, "data"):
            raise LiveExecutionError("physical state source must expose MuJoCo model and data")
        if mujoco_module is None:
            import mujoco as mujoco_module
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.mujoco = mujoco_module
        self.bindings = dict(bindings)
        self.fixed_bindings = dict(fixed_bindings or {})
        self.robot_body_prefixes = tuple(robot_body_prefixes)
        self.penetration_tolerance_m = penetration_tolerance_m
        self.linear_stability_m_s = linear_stability_m_s
        self.angular_stability_rad_s = angular_stability_rad_s
        self.action_measurements: dict[str, Any] = {}

    def capture_object(self, entity: str) -> Mapping[str, Any]:
        return self._entity_state(entity)

    def verify_open(self, entity: str) -> Mapping[str, Any]:
        record = self._articulation(entity)
        return {"success": record.get("open") is True, **record}

    def verify_pick(self, entity: str, before: Mapping[str, Any]) -> Mapping[str, Any]:
        after = self._entity_state(entity)
        before_position = np.asarray(before.get("world_position_m", ()), dtype=float)
        after_position = np.asarray(after.get("world_position_m", ()), dtype=float)
        displacement = (
            float(np.linalg.norm(after_position - before_position))
            if before_position.shape == (3,) and after_position.shape == (3,)
            else None
        )
        source_cleared = not after.get("support_contact", False) or (
            displacement is not None and displacement >= 0.01
        )
        success = bool(after.get("held") is True and source_cleared and not after.get("floor_contact", False))
        return {"success": success, "held": after.get("held"), "source_cleared": source_cleared, "displacement_m": displacement, "physical_state": after}

    def verify_place(self, entity: str, target: str, *, contained: bool) -> Mapping[str, Any]:
        state = self._entity_state(entity)
        target_state = self._entity_state(target)
        if contained:
            relation_ok = self._contained(entity, target)
        else:
            relation_ok = state.get("support_entity") == target and state.get("support_contact") is True
        inside = self._inside_footprint(entity, target)
        success = bool(
            state.get("held") is False
            and state.get("stable") is True
            and relation_ok
            and inside
            and state.get("floor_contact") is False
            and state.get("invalid_penetration") is False
        )
        return {
            "success": success,
            "released": state.get("held") is False,
            "stable": state.get("stable"),
            "relation_verified": relation_ok,
            "inside_target_footprint": inside,
            "floor_contact": state.get("floor_contact"),
            "invalid_penetration": state.get("invalid_penetration"),
            "object_state": state,
            "target_state": target_state,
        }

    def verify_skill(
        self,
        operator: str,
        arguments: Sequence[str],
        controller_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        operator = operator.upper()
        flag = {
            "POUR": "pour_motion_verified",
            "STIR": "stir_motion_verified",
            "DRIVE": "drive_motion_verified",
        }.get(operator)
        if flag is None:
            return {"success": False, "reason": "unsupported physical skill"}
        nested = controller_result.get("physical_postcondition", {})
        nested_verified = nested.get("verified") if isinstance(nested, Mapping) else None
        motion_verified = controller_result.get(flag) is True or nested_verified is True
        payload_held = True
        if operator in {"POUR", "STIR"} and arguments:
            payload_held = self._entity_state(arguments[0]).get("held") is True
        drive = controller_result.get("drive_metrics", {})
        drive_verified = True
        if operator == "DRIVE":
            drive_verified = bool(
                controller_result.get("joint_repaired") is True
                or controller_result.get("joint_repaired_state") is True
                or (isinstance(drive, Mapping) and drive.get("verified") is True)
            )
            motion_verified = drive_verified
        success = bool(motion_verified and payload_held and drive_verified)
        return {"success": success, "motion_verified": motion_verified, "payload_held": payload_held, "drive_verified": drive_verified}

    def record_action(
        self,
        operator: str,
        arguments: Sequence[str],
        controller_result: Mapping[str, Any],
    ) -> None:
        operator = operator.upper()
        if operator == "SCREW":
            operator = "DRIVE"
        if operator in {"POUR", "STIR", "DRIVE"}:
            self.action_measurements[operator.lower()] = {
                "arguments": list(arguments),
                "controller_result": dict(controller_result),
            }
        if operator in {"INSERT", "PLACE", "DRIVE"}:
            for key in ("insertion_metrics", "drive_metrics"):
                value = controller_result.get(key)
                if isinstance(value, Mapping):
                    self.action_measurements[key] = dict(value)
            if controller_result.get("joint_repaired") is True:
                self.action_measurements["joint_repaired"] = True
            if controller_result.get("joint_repaired_state") is True:
                self.action_measurements["joint_repaired"] = True
        if operator == "INSERT" and "insertion_metrics" not in self.action_measurements:
            keys = {
                "fastener",
                "target",
                "depth_m",
                "insertion_depth_m",
                "radial_error_m",
                "orientation_error_rad",
                "vertical_axis_error_rad",
                "head_above_tip",
                "head_above_tip_m",
            }
            metrics = {key: controller_result[key] for key in keys if key in controller_result}
            if len(arguments) >= 2:
                metrics.setdefault("fastener", arguments[0])
                metrics.setdefault("target", arguments[1])
            if metrics:
                metrics.setdefault("depth_m", metrics.get("insertion_depth_m"))
                metrics.setdefault("orientation_error_rad", metrics.get("vertical_axis_error_rad"))
                metrics.setdefault("head_above_tip", float(metrics.get("head_above_tip_m", 0.0)) > 0.0)
                self.action_measurements["insertion_metrics"] = metrics

    def snapshot(self, domain: str, predicted_infeasible: bool) -> TerminalStateSnapshot:
        self.mujoco.mj_forward(self.model, self.data)
        all_bindings = {**self.fixed_bindings, **self.bindings}
        entity_to_id = {binding.entity_name: object_id for object_id, binding in all_bindings.items()}
        objects: dict[str, Mapping[str, Any]] = {}
        held = []
        contained: dict[str, list[str]] = {}
        articulation: dict[str, Mapping[str, Any]] = {}
        for object_id, binding in all_bindings.items():
            raw = self._entity_state(binding.entity_name)
            support_entity = raw.get("support_entity")
            support_id = entity_to_id.get(str(support_entity)) if support_entity else None
            record = {
                **raw,
                "support": support_id,
                "released": raw.get("held") is False,
                "inside_support_footprint": (
                    self._inside_footprint(binding.entity_name, str(support_entity))
                    if support_entity else False
                ),
            }
            record.pop("support_entity", None)
            objects[object_id] = record
            if binding.entity_name != object_id:
                objects[binding.entity_name] = {
                    **record,
                    "support": support_entity,
                    "symbolic_alias": object_id,
                }
            if raw.get("held") is True:
                held.append(object_id)
            articulation[object_id] = self._articulation(binding.entity_name)
        for container_id, container in all_bindings.items():
            members = [
                object_id for object_id, binding in self.bindings.items()
                if object_id != container_id and self._contained(binding.entity_name, container.entity_name)
            ]
            if members:
                contained[container_id] = sorted(members)
                physical_members = sorted(
                    all_bindings[item].entity_name for item in members
                )
                contained[container.entity_name] = physical_members
                for member_id in members:
                    physical_member = all_bindings[member_id].entity_name
                    member_state = objects[member_id]
                    contained_stably = bool(
                        member_state.get("released") is True
                        and member_state.get("stable") is True
                    )
                    member_state["contained_stably"] = contained_stably
                    if physical_member in objects:
                        objects[physical_member]["contained_stably"] = contained_stably
            if container.entity_name != container_id:
                articulation[container.entity_name] = articulation[container_id]

        measurements = dict(self.action_measurements)
        relations: dict[str, Any] = {"contained_in": contained, "articulation": articulation}
        insertion = measurements.get("insertion_metrics")
        if isinstance(insertion, Mapping):
            relations["insertion"] = _entity_measurements_with_aliases(
                insertion, entity_to_id
            )
        drive = measurements.get("drive")
        if isinstance(drive, Mapping):
            args = drive.get("arguments", ())
            if isinstance(args, (list, tuple)) and len(args) == 3:
                measurements.update(
                    used_driver=str(args[0]),
                    used_fastener=str(args[1]),
                    used_target=str(args[2]),
                    used_driver_symbol=entity_to_id.get(str(args[0])),
                    used_fastener_symbol=entity_to_id.get(str(args[1])),
                    used_target_symbol=entity_to_id.get(str(args[2])),
                )
        return TerminalStateSnapshot(
            domain=_domain_key(domain),
            predicted_infeasible=predicted_infeasible,
            objects=objects,
            relations=relations,
            held_objects=tuple(sorted(held)),
            measurements=measurements,
        )

    def _entity_state(self, entity: str) -> Mapping[str, Any]:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, entity)
        if body_id < 0:
            return {"present": False, "entity_name": entity}
        self.mujoco.mj_forward(self.model, self.data)
        minimum, maximum = self._aabb(body_id)
        contacts = self._contacts(body_id)
        support = self._support_entity(body_id, contacts)
        velocity = np.zeros(6, dtype=float)
        self.mujoco.mj_objectVelocity(
            self.model, self.data, self.mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
        )
        floor_contact = any(item["other_is_floor"] for item in contacts)
        invalid = any(
            item["distance_m"] < -self.penetration_tolerance_m
            and not item["other_is_robot"]
            and self._bound_ancestor(str(item["other_body"])) != support
            for item in contacts
        )
        return {
            "present": True,
            "entity_name": entity,
            "world_position_m": [float(value) for value in self.data.xpos[body_id]],
            "world_quaternion": [float(value) for value in self.data.xquat[body_id]],
            "aabb_min_m": minimum.tolist(),
            "aabb_max_m": maximum.tolist(),
            "held": self._held(body_id),
            "stable": bool(
                np.linalg.norm(velocity[3:]) <= self.linear_stability_m_s
                and np.linalg.norm(velocity[:3]) <= self.angular_stability_rad_s
            ),
            "support_entity": support,
            "support_contact": support is not None,
            "floor_contact": floor_contact,
            "invalid_penetration": invalid,
            "contacts": contacts,
        }

    def _articulation(self, entity: str) -> Mapping[str, Any]:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, entity)
        if body_id < 0:
            return {"present": False, "open": False}
        joints = [
            index
            for index in range(self.model.njnt)
            if self._is_descendant(int(self.model.jnt_bodyid[index]), body_id)
            and int(self.model.jnt_type[index])
            != int(self.mujoco.mjtJoint.mjJNT_FREE)
        ]
        values = []
        fractions = []
        for joint_id in joints:
            address = int(self.model.jnt_qposadr[joint_id])
            value = float(self.data.qpos[address])
            lower, upper = map(float, self.model.jnt_range[joint_id])
            fraction = abs(value) / max(abs(lower), abs(upper), 1e-12)
            values.append(value)
            fractions.append(fraction)
        open_fraction = max(fractions, default=0.0)
        return {"present": True, "joint_values": values, "open_fraction": open_fraction, "open": bool(joints and open_fraction >= 0.5)}

    def _held(self, body_id: int) -> bool:
        for equality_id in range(self.model.neq):
            if not bool(self.data.eq_active[equality_id]):
                continue
            first = int(self.model.eq_obj1id[equality_id])
            second = int(self.model.eq_obj2id[equality_id])
            if body_id not in {first, second}:
                continue
            other = second if first == body_id else first
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if any(name.startswith(prefix) for prefix in self.robot_body_prefixes):
                return True
        return False

    def _body_geoms(self, body_id: int) -> tuple[int, ...]:
        return tuple(
            geom_id
            for geom_id in range(self.model.ngeom)
            if self._is_descendant(int(self.model.geom_bodyid[geom_id]), body_id)
        )

    def _aabb(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        minima, maxima = [], []
        for geom_id in self._body_geoms(body_id):
            size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            extent = np.abs(rotation) @ size
            centre = np.asarray(self.data.geom_xpos[geom_id], dtype=float)
            minima.append(centre - extent)
            maxima.append(centre + extent)
        if not minima:
            centre = np.asarray(self.data.xpos[body_id], dtype=float)
            return centre.copy(), centre.copy()
        return np.min(minima, axis=0), np.max(maxima, axis=0)

    def _contacts(self, body_id: int) -> list[dict[str, Any]]:
        geoms = set(self._body_geoms(body_id))
        rows = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            if not geoms.intersection(pair):
                continue
            other_geom = pair[1] if pair[0] in geoms else pair[0]
            other_body_id = int(self.model.geom_bodyid[other_geom])
            if self._is_descendant(other_body_id, body_id):
                continue
            other_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, other_body_id) or "world"
            geom_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            rows.append({
                "other_body": other_name,
                "other_geom": geom_name,
                "distance_m": float(contact.dist),
                "other_is_floor": other_body_id == 0 or "floor" in geom_name.lower(),
                "other_is_robot": any(other_name.startswith(prefix) for prefix in self.robot_body_prefixes),
            })
        return rows

    def _support_entity(self, body_id: int, contacts: Sequence[Mapping[str, Any]]) -> str | None:
        object_z = float(self.data.xpos[body_id, 2])
        candidates = []
        for item in contacts:
            name = str(item["other_body"])
            other_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if other_id <= 0 or item["other_is_robot"] or float(self.data.xpos[other_id, 2]) > object_z:
                continue
            candidates.append((float(self.data.xpos[other_id, 2]), name))
        support = max(candidates, default=(0.0, None))[1]
        return self._bound_ancestor(support) if support is not None else None

    def _bound_ancestor(self, entity: str) -> str:
        body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, entity
        )
        bound_names = {
            binding.entity_name
            for binding in (*self.bindings.values(), *self.fixed_bindings.values())
        }
        while body_id > 0:
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            if name in bound_names:
                return name
            body_id = int(self.model.body_parentid[body_id])
        return entity

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return ancestor_id == 0 and body_id == 0

    def _inside_footprint(self, entity: str, target: str) -> bool:
        entity_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, entity)
        target_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, target)
        if entity_id < 0 or target_id < 0:
            return False
        lower, upper = self._aabb(target_id)
        position = np.asarray(self.data.xpos[entity_id], dtype=float)
        return bool(np.all(position[:2] >= lower[:2] - 0.015) and np.all(position[:2] <= upper[:2] + 0.015))

    def _contained(self, entity: str, target: str) -> bool:
        entity_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, entity)
        target_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, target)
        if entity_id < 0 or target_id < 0:
            return False
        object_min, object_max = self._aabb(entity_id)
        target_min, target_max = self._aabb(target_id)
        centre = (object_min + object_max) * 0.5
        return bool(np.all(centre >= target_min - 0.01) and np.all(centre <= target_max + 0.01))


def _validate_execution_input(
    domain: str,
    plan: BaselineExecutionPlan,
    projections: Sequence[ExecutionProjection],
    runtime: LiveDomainRuntime,
) -> None:
    if not isinstance(plan, BaselineExecutionPlan):
        raise LiveExecutionError("live execution requires a BaselineExecutionPlan")
    if _domain_key(plan.domain) != domain:
        raise LiveExecutionError("execution plan domain mismatch")
    if tuple(plan.normalized_actions) != tuple(plan.symbolic_plan.actions):
        raise LiveExecutionError("execution plan normalized actions differ from symbolic plan")
    expected = project_plan(
        domain,
        plan.normalized_actions,
        runtime.bindings,
        fixed_bindings=runtime.fixed_bindings,
    )
    if len(expected) != len(projections):
        raise LiveExecutionError("execution projection count mismatch")
    fields = ("action_instance_id", "pddl_operator", "pddl_arguments", "controller_operator", "controller_arguments", "resolved_entities")
    for index, (wanted, actual) in enumerate(zip(expected, projections)):
        if not isinstance(actual, ExecutionProjection) or any(
            getattr(wanted, field_name) != getattr(actual, field_name)
            for field_name in fields
        ):
            raise LiveExecutionError(f"execution projection {index} is not derived from runtime bindings")


def _request(request: Mapping[str, Any], methods: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    operator = str(request.get("operator", "")).upper()
    if operator not in methods:
        return operator, _string_arguments(request)
    return operator, _string_arguments(request)


def _string_arguments(request: Mapping[str, Any]) -> tuple[str, ...]:
    arguments = request.get("arguments")
    if not isinstance(arguments, (list, tuple)) or any(not isinstance(item, str) or not item for item in arguments):
        raise LiveExecutionError("controller request arguments must be non-empty strings")
    return tuple(arguments)


def _invoke_primitive(primitives: Any, method_name: str, arguments: Sequence[str], operator: str) -> Mapping[str, Any]:
    method = getattr(primitives, method_name, None)
    if callable(method):
        return _mapping_result(method(*arguments))
    generic = getattr(primitives, "execute_phase2_action", None)
    if callable(generic):
        return _mapping_result(generic({"action": operator, "arguments": list(arguments)}))
    return {"success": False, "status": "UNSUPPORTED_CONTROLLER_ACTION", "operator": operator}


def _mapping_result(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveExecutionError("physical controller result must be a mapping")
    return dict(value)


def _domain_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"kitchen", "living_room", "workshop"}:
        raise LiveExecutionError(f"unsupported domain: {value!r}")
    return normalized


def _serializable(value: Mapping[str, Any] | SerializableContract) -> Mapping[str, Any]:
    return value.to_dict() if isinstance(value, SerializableContract) else dict(value)


def _entity_measurements_with_aliases(
    value: Mapping[str, Any], entity_to_id: Mapping[str, str]
) -> Mapping[str, Any]:
    result = dict(value)
    for key in ("fastener", "target", "driver"):
        if key in result:
            result[f"{key}_symbol"] = entity_to_id.get(str(result[key]))
    result.setdefault("verified", True)
    return result
