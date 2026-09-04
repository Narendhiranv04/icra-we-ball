"""Direct living-room mobile manipulation for projected ViLaIn actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..artifacts import atomic_write_json
from ..contracts import ExecutionProjection, SerializableContract


class LivingRoomExecutionFailureCode(str, Enum):
    UNRESOLVED_ENTITY = "UNRESOLVED_ENTITY"
    UNSUPPORTED_CONTROLLER_ACTION = "UNSUPPORTED_CONTROLLER_ACTION"
    MOBILE_FAILURE = "MOBILE_FAILURE"
    PICK_FAILURE = "PICK_FAILURE"
    PLACE_FAILURE = "PLACE_FAILURE"
    POSTCONDITION_FAILURE = "POSTCONDITION_FAILURE"
    EXECUTOR_EXCEPTION = "EXECUTOR_EXCEPTION"


class LivingRoomExecutionContractError(ValueError):
    """Raised when an execution input is not a valid living-room projection."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: LivingRoomExecutionFailureCode | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class LivingRoomActionExecution(SerializableContract):
    action_index: int
    action_instance_id: str
    pddl_operator: str
    pddl_arguments: tuple[str, ...]
    controller_operator: str
    controller_arguments: tuple[str, ...]
    success: bool
    mobile_result: Mapping[str, Any] = field(default_factory=dict)
    motion_result: Mapping[str, Any] = field(default_factory=dict)
    destination: Mapping[str, Any] = field(default_factory=dict)
    postcondition_result: Mapping[str, Any] = field(default_factory=dict)
    failure_code: LivingRoomExecutionFailureCode | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class LivingRoomExecutionResult(SerializableContract):
    success: bool
    actions: tuple[LivingRoomActionExecution, ...]
    terminal_failure_code: LivingRoomExecutionFailureCode | None
    terminal_failure_message: str | None


class MobileMotionExecutor(Protocol):
    """Lower-level base motion facade used by the additive adapter."""

    def move_to(
        self,
        target_entity: str,
        *,
        carrying_entity: str | None,
    ) -> Mapping[str, Any]: ...


class PickMotionExecutor(Protocol):
    """Lower-level physical pick facade."""

    def pick(self, payload_entity: str) -> Mapping[str, Any]: ...

    def verify_held(self, payload_entity: str) -> Mapping[str, Any]: ...


class PlaceMotionExecutor(Protocol):
    """Lower-level destination, place, and physical-ON facade."""

    def destination_for(
        self,
        *,
        payload_id: str,
        payload_entity: str,
        support_id: str,
        support_entity: str,
    ) -> Mapping[str, Any]: ...

    def place(
        self,
        payload_entity: str,
        support_entity: str,
        destination: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def verify_physical_on(
        self,
        payload_entity: str,
        support_entity: str,
        destination: Mapping[str, Any],
        place_result: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class LivingRoomExecutionAdapter:
    """Execute PICK/PLACE projections with direct motion executors."""

    def __init__(
        self,
        *,
        mobile: MobileMotionExecutor,
        picker: PickMotionExecutor,
        placer: PlaceMotionExecutor,
    ) -> None:
        required = (
            (mobile, "move_to"),
            (picker, "pick"),
            (picker, "verify_held"),
            (placer, "destination_for"),
            (placer, "place"),
            (placer, "verify_physical_on"),
        )
        missing = [
            name
            for component, name in required
            if not callable(getattr(component, name, None))
        ]
        if missing:
            raise LivingRoomExecutionContractError(
                "motion executor is missing methods: " + ", ".join(missing)
            )
        self.mobile = mobile
        self.picker = picker
        self.placer = placer

    def execute(
        self,
        projections: Sequence[ExecutionProjection],
        *,
        output_root: str | Path | None = None,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> LivingRoomExecutionResult:
        if external_method_artifacts is not None:
            raise LivingRoomExecutionContractError(
                "external method artifacts are not valid living-room execution input"
            )
        results: list[LivingRoomActionExecution] = []
        held_entity: str | None = None

        for action_index, projection in enumerate(projections):
            try:
                operator = _validate_projection(projection)
            except LivingRoomExecutionContractError as error:
                if error.failure_code is None:
                    raise
                action = _failed_action(
                    action_index,
                    projection,
                    error.failure_code,
                    str(error),
                )
                results.append(action)
                return self._finish(results, action, output_root)

            if operator == "pick-from":
                action = self._execute_pick(action_index, projection, held_entity)
                if action.success:
                    held_entity = projection.controller_arguments[0]
            else:
                action = self._execute_place(action_index, projection, held_entity)
                if action.success:
                    held_entity = None
            results.append(action)
            if not action.success:
                return self._finish(results, action, output_root)

        return self._finish(results, None, output_root)

    def _execute_pick(
        self,
        action_index: int,
        projection: ExecutionProjection,
        held_entity: str | None,
    ) -> LivingRoomActionExecution:
        payload = projection.controller_arguments[0]
        if held_entity is not None:
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.PICK_FAILURE,
                f"cannot PICK while holding {held_entity!r}",
            )
        try:
            mobile_result = _mapping_result(
                self.mobile.move_to(payload, carrying_entity=None),
                "mobile",
            )
        except Exception as error:
            return _exception_action(action_index, projection, "mobile", error)
        if not _succeeded(mobile_result):
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.MOBILE_FAILURE,
                _failure_message(mobile_result, "mobile motion failed"),
                mobile_result=mobile_result,
            )
        try:
            pick_result = _mapping_result(self.picker.pick(payload), "pick")
        except Exception as error:
            return _exception_action(
                action_index,
                projection,
                "pick",
                error,
                mobile_result=mobile_result,
            )
        if not _succeeded(pick_result):
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.PICK_FAILURE,
                _failure_message(pick_result, "pick motion failed"),
                mobile_result=mobile_result,
                motion_result=pick_result,
            )
        try:
            held_result = _mapping_result(
                self.picker.verify_held(payload), "held-state"
            )
        except Exception as error:
            return _exception_action(
                action_index,
                projection,
                "held-state verification",
                error,
                mobile_result=mobile_result,
                motion_result=pick_result,
            )
        if not _held_state_succeeded(held_result):
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.POSTCONDITION_FAILURE,
                _failure_message(held_result, "held-state postcondition failed"),
                mobile_result=mobile_result,
                motion_result=pick_result,
                postcondition_result=held_result,
            )
        return _successful_action(
            action_index,
            projection,
            mobile_result=mobile_result,
            motion_result=pick_result,
            postcondition_result=held_result,
        )

    def _execute_place(
        self,
        action_index: int,
        projection: ExecutionProjection,
        held_entity: str | None,
    ) -> LivingRoomActionExecution:
        payload, support = projection.controller_arguments
        payload_id, support_id = projection.pddl_arguments
        if held_entity != payload:
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.PLACE_FAILURE,
                f"PLACE requires held payload {payload!r}; observed {held_entity!r}",
            )
        try:
            destination = _mapping_result(
                self.placer.destination_for(
                    payload_id=payload_id,
                    payload_entity=payload,
                    support_id=support_id,
                    support_entity=support,
                ),
                "placement destination",
            )
        except Exception as error:
            return _exception_action(action_index, projection, "destination", error)
        if not destination:
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.PLACE_FAILURE,
                "placement destination is empty",
            )
        try:
            mobile_result = _mapping_result(
                self.mobile.move_to(support, carrying_entity=payload),
                "mobile",
            )
        except Exception as error:
            return _exception_action(
                action_index,
                projection,
                "mobile",
                error,
                destination=destination,
            )
        if not _succeeded(mobile_result):
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.MOBILE_FAILURE,
                _failure_message(mobile_result, "mobile carry motion failed"),
                mobile_result=mobile_result,
                destination=destination,
            )
        try:
            place_result = _mapping_result(
                self.placer.place(payload, support, destination),
                "place",
            )
        except Exception as error:
            return _exception_action(
                action_index,
                projection,
                "place",
                error,
                mobile_result=mobile_result,
                destination=destination,
            )
        if not _succeeded(place_result):
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.PLACE_FAILURE,
                _failure_message(place_result, "place motion failed"),
                mobile_result=mobile_result,
                motion_result=place_result,
                destination=destination,
            )
        try:
            on_result = _mapping_result(
                self.placer.verify_physical_on(
                    payload,
                    support,
                    destination,
                    place_result,
                ),
                "physical ON verification",
            )
        except Exception as error:
            return _exception_action(
                action_index,
                projection,
                "physical ON verification",
                error,
                mobile_result=mobile_result,
                motion_result=place_result,
                destination=destination,
            )
        if not _physical_on_succeeded(on_result):
            return _failed_action(
                action_index,
                projection,
                LivingRoomExecutionFailureCode.POSTCONDITION_FAILURE,
                _failure_message(on_result, "physical ON postcondition failed"),
                mobile_result=mobile_result,
                motion_result=place_result,
                destination=destination,
                postcondition_result=on_result,
            )
        return _successful_action(
            action_index,
            projection,
            mobile_result=mobile_result,
            motion_result=place_result,
            destination=destination,
            postcondition_result=on_result,
        )

    @staticmethod
    def _finish(
        actions: Sequence[LivingRoomActionExecution],
        failed_action: LivingRoomActionExecution | None,
        output_root: str | Path | None,
    ) -> LivingRoomExecutionResult:
        result = LivingRoomExecutionResult(
            success=failed_action is None,
            actions=tuple(actions),
            terminal_failure_code=(
                failed_action.failure_code if failed_action is not None else None
            ),
            terminal_failure_message=(
                failed_action.failure_message if failed_action is not None else None
            ),
        )
        if output_root is not None:
            atomic_write_json(
                Path(output_root) / "living_room_execution.json",
                result.to_dict(),
            )
        return result


def _validate_projection(projection: ExecutionProjection) -> str:
    if not isinstance(projection, ExecutionProjection):
        raise LivingRoomExecutionContractError(
            "living-room execution requires ExecutionProjection contracts"
        )
    operator = projection.pddl_operator.strip().lower().replace("_", "-")
    if operator == "pick-from":
        expected_operator, pddl_arity, controller_arity = "PICK", 2, 1
    elif operator == "place-on":
        expected_operator, pddl_arity, controller_arity = "PLACE", 2, 2
    else:
        raise LivingRoomExecutionContractError(
            f"UNSUPPORTED_CONTROLLER_ACTION: {operator!r}",
            failure_code=LivingRoomExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    if (
        projection.controller_operator != expected_operator
        or len(projection.pddl_arguments) != pddl_arity
        or len(projection.controller_arguments) != controller_arity
    ):
        raise LivingRoomExecutionContractError(
            f"invalid {operator!r} controller projection",
            failure_code=LivingRoomExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    if (
        projection.controller_arguments != projection.resolved_entities
        or any(not value.strip() for value in projection.controller_arguments)
    ):
        raise LivingRoomExecutionContractError(
            "UNRESOLVED_ENTITY: projected controller identity mismatch",
            failure_code=LivingRoomExecutionFailureCode.UNRESOLVED_ENTITY,
        )
    return operator


def _mapping_result(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} executor result is not a mapping")
    return dict(value)


def _succeeded(result: Mapping[str, Any]) -> bool:
    return bool(result.get("success", False))


def _held_state_succeeded(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("success", False)
        or str(result.get("validation_status", "")).upper() == "TRUE"
    )


def _physical_on_succeeded(result: Mapping[str, Any]) -> bool:
    return bool(result.get("relation") == "ON" and result.get("verified") is True)


def _failure_message(result: Mapping[str, Any], default: str) -> str:
    return str(
        result.get("message")
        or result.get("reason")
        or result.get("status")
        or default
    )


def _successful_action(
    action_index: int,
    projection: ExecutionProjection,
    *,
    mobile_result: Mapping[str, Any],
    motion_result: Mapping[str, Any],
    destination: Mapping[str, Any] | None = None,
    postcondition_result: Mapping[str, Any],
) -> LivingRoomActionExecution:
    return LivingRoomActionExecution(
        action_index=action_index,
        action_instance_id=projection.action_instance_id,
        pddl_operator=projection.pddl_operator,
        pddl_arguments=projection.pddl_arguments,
        controller_operator=projection.controller_operator,
        controller_arguments=projection.controller_arguments,
        success=True,
        mobile_result=dict(mobile_result),
        motion_result=dict(motion_result),
        destination=dict(destination or {}),
        postcondition_result=dict(postcondition_result),
    )


def _failed_action(
    action_index: int,
    projection: ExecutionProjection,
    failure_code: LivingRoomExecutionFailureCode,
    message: str,
    *,
    mobile_result: Mapping[str, Any] | None = None,
    motion_result: Mapping[str, Any] | None = None,
    destination: Mapping[str, Any] | None = None,
    postcondition_result: Mapping[str, Any] | None = None,
) -> LivingRoomActionExecution:
    return LivingRoomActionExecution(
        action_index=action_index,
        action_instance_id=projection.action_instance_id,
        pddl_operator=projection.pddl_operator,
        pddl_arguments=projection.pddl_arguments,
        controller_operator=projection.controller_operator,
        controller_arguments=projection.controller_arguments,
        success=False,
        mobile_result=dict(mobile_result or {}),
        motion_result=dict(motion_result or {}),
        destination=dict(destination or {}),
        postcondition_result=dict(postcondition_result or {}),
        failure_code=failure_code,
        failure_message=message,
    )


def _exception_action(
    action_index: int,
    projection: ExecutionProjection,
    component: str,
    error: Exception,
    *,
    mobile_result: Mapping[str, Any] | None = None,
    motion_result: Mapping[str, Any] | None = None,
    destination: Mapping[str, Any] | None = None,
) -> LivingRoomActionExecution:
    return _failed_action(
        action_index,
        projection,
        LivingRoomExecutionFailureCode.EXECUTOR_EXCEPTION,
        f"{component} raised {type(error).__name__}: {error}",
        mobile_result=mobile_result,
        motion_result=motion_result,
        destination=destination,
    )
