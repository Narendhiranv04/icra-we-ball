"""Direct workshop execution for independently projected ViLaIn actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..artifacts import atomic_write_json
from ..contracts import ExecutionProjection, SerializableContract


class WorkshopExecutionFailureCode(str, Enum):
    UNRESOLVED_ENTITY = "UNRESOLVED_ENTITY"
    UNSUPPORTED_CONTROLLER_ACTION = "UNSUPPORTED_CONTROLLER_ACTION"
    PRECONDITION_FAILURE = "PRECONDITION_FAILURE"
    CONTROLLER_EXCEPTION = "CONTROLLER_EXCEPTION"
    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"
    POSTCONDITION_FAILURE = "POSTCONDITION_FAILURE"


class WorkshopExecutionContractError(ValueError):
    """Raised when workshop execution input violates the baseline boundary."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: WorkshopExecutionFailureCode | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class WorkshopControllerContract(SerializableContract):
    """Action-derived fields required by the existing physical controller."""

    driver_id: str
    fastener_id: str
    target_id: str
    work_surface_id: str
    driver_entity: str
    fastener_entity: str
    target_entity: str
    work_surface_entity: str
    drive_action_instance_id: str

    @property
    def driver(self) -> str:
        return self.driver_entity

    @property
    def fastener(self) -> str:
        return self.fastener_entity

    @property
    def target_joint(self) -> str:
        return self.target_entity

    @property
    def work_surface(self) -> str:
        return self.work_surface_entity


@dataclass(frozen=True)
class WorkshopDriveEffect(SerializableContract):
    action_index: int
    action_instance_id: str
    effect: str
    symbolic_arguments: tuple[str, str, str]
    resolved_entities: tuple[str, str, str]
    controller_status: str | None


@dataclass(frozen=True)
class WorkshopActionExecution(SerializableContract):
    action_index: int
    action_instance_id: str
    pddl_operator: str
    pddl_arguments: tuple[str, ...]
    controller_operator: str
    controller_arguments: tuple[str, ...]
    success: bool
    controller_result: Mapping[str, Any] = field(default_factory=dict)
    postcondition_result: Mapping[str, Any] = field(default_factory=dict)
    failure_code: WorkshopExecutionFailureCode | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class WorkshopExecutionResult(SerializableContract):
    success: bool
    actions: tuple[WorkshopActionExecution, ...]
    effect_ledger: tuple[WorkshopDriveEffect, ...]
    terminal_failure_code: WorkshopExecutionFailureCode | None
    terminal_failure_message: str | None


class WorkshopControllerDispatcher(Protocol):
    """Minimal physical dispatcher facade, supplied with action-derived data."""

    def execute_action(
        self,
        request: Mapping[str, Any],
        controller_contract: WorkshopControllerContract,
    ) -> Mapping[str, Any]: ...

    def verify_postcondition(
        self,
        request: Mapping[str, Any],
        controller_result: Mapping[str, Any],
        controller_contract: WorkshopControllerContract,
    ) -> Mapping[str, Any]: ...


_RULES: dict[str, tuple[str, int]] = {
    "open-storage": ("OPEN", 1),
    "pick-from": ("PICK", 2),
    "insert": ("PLACE", 2),
    "drive": ("SCREW", 3),
    "place-on": ("PLACE", 2),
}


def build_workshop_controller_contract(
    projections: Sequence[ExecutionProjection],
    *,
    external_method_artifacts: Mapping[str, Any] | None = None,
) -> WorkshopControllerContract:
    """Derive the controller tuple solely from DRIVE/INSERT/PLACE arguments."""
    if external_method_artifacts is not None:
        raise WorkshopExecutionContractError(
            "external method artifacts are not valid workshop controller input"
        )
    for projection in projections:
        _validate_projection(projection)
    drives = [
        projection
        for projection in projections
        if _operator(projection) == "drive"
    ]
    if len(drives) != 1:
        raise WorkshopExecutionContractError(
            "workshop plan must contain exactly one DRIVE action"
        )
    drive = drives[0]
    driver_id, fastener_id, target_id = drive.pddl_arguments
    driver_entity, fastener_entity, target_entity = drive.controller_arguments

    insertions = [
        projection
        for projection in projections
        if _operator(projection) == "insert"
        and projection.pddl_arguments == (fastener_id, target_id)
        and projection.controller_arguments == (fastener_entity, target_entity)
    ]
    if len(insertions) != 1:
        raise WorkshopExecutionContractError(
            "DRIVE must have one matching INSERT(fastener,target)"
        )
    driver_places = [
        projection
        for projection in projections
        if _operator(projection) == "place-on"
        and projection.pddl_arguments[0] == driver_id
        and projection.controller_arguments[0] == driver_entity
    ]
    if len(driver_places) != 1:
        raise WorkshopExecutionContractError(
            "DRIVE driver must have one matching PLACE-ON destination"
        )
    work_surface_id = driver_places[0].pddl_arguments[1]
    work_surface_entity = driver_places[0].controller_arguments[1]
    return WorkshopControllerContract(
        driver_id=driver_id,
        fastener_id=fastener_id,
        target_id=target_id,
        work_surface_id=work_surface_id,
        driver_entity=driver_entity,
        fastener_entity=fastener_entity,
        target_entity=target_entity,
        work_surface_entity=work_surface_entity,
        drive_action_instance_id=drive.action_instance_id,
    )


class WorkshopExecutionAdapter:
    """Execute a projected workshop sequence and stop at its first failure."""

    def __init__(
        self,
        *,
        dispatcher: WorkshopControllerDispatcher,
        controller_contract: WorkshopControllerContract,
    ) -> None:
        if not callable(getattr(dispatcher, "execute_action", None)):
            raise WorkshopExecutionContractError("dispatcher has no execute_action method")
        if not callable(getattr(dispatcher, "verify_postcondition", None)):
            raise WorkshopExecutionContractError(
                "dispatcher has no verify_postcondition method"
            )
        if not isinstance(controller_contract, WorkshopControllerContract):
            raise WorkshopExecutionContractError(
                "workshop execution requires a WorkshopControllerContract"
            )
        self.dispatcher = dispatcher
        self.controller_contract = controller_contract

    def execute(
        self,
        projections: Sequence[ExecutionProjection],
        *,
        output_root: str | Path | None = None,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> WorkshopExecutionResult:
        if external_method_artifacts is not None:
            raise WorkshopExecutionContractError(
                "external method artifacts are not valid workshop execution input"
            )
        results: list[WorkshopActionExecution] = []
        effects: list[WorkshopDriveEffect] = []
        held: str | None = None
        inserted: tuple[str, str] | None = None

        for action_index, projection in enumerate(projections):
            try:
                operator = _validate_projection(projection)
                _validate_against_controller_contract(
                    projection, self.controller_contract
                )
            except WorkshopExecutionContractError as error:
                if error.failure_code is None:
                    raise
                action = _failed_action(
                    action_index, projection, error.failure_code, str(error)
                )
                results.append(action)
                return self._finish(results, effects, action, output_root)

            precondition_error = _precondition_error(
                operator,
                projection,
                held=held,
                inserted=inserted,
                controller_contract=self.controller_contract,
            )
            if precondition_error is not None:
                action = _failed_action(
                    action_index,
                    projection,
                    WorkshopExecutionFailureCode.PRECONDITION_FAILURE,
                    precondition_error,
                )
                results.append(action)
                return self._finish(results, effects, action, output_root)

            request = {
                "action_index": action_index,
                "action_instance_id": projection.action_instance_id,
                "operator": projection.controller_operator,
                "arguments": list(projection.controller_arguments),
                "pddl_operator": operator,
            }
            try:
                controller_value = self.dispatcher.execute_action(
                    request, self.controller_contract
                )
                controller_result = _mapping_result(controller_value, "controller")
            except Exception as error:
                action = _failed_action(
                    action_index,
                    projection,
                    WorkshopExecutionFailureCode.CONTROLLER_EXCEPTION,
                    f"{type(error).__name__}: {error}",
                )
                results.append(action)
                return self._finish(results, effects, action, output_root)
            if not bool(controller_result.get("success", False)):
                action = _failed_action(
                    action_index,
                    projection,
                    WorkshopExecutionFailureCode.CONTROLLER_FAILURE,
                    _failure_message(controller_result, "controller reported failure"),
                    controller_result=controller_result,
                )
                results.append(action)
                return self._finish(results, effects, action, output_root)

            try:
                postcondition_value = self.dispatcher.verify_postcondition(
                    request, controller_result, self.controller_contract
                )
                postcondition_result = _mapping_result(
                    postcondition_value, "postcondition"
                )
            except Exception as error:
                postcondition_result = {
                    "success": False,
                    "message": f"{type(error).__name__}: {error}",
                }
            if not bool(postcondition_result.get("success", False)):
                action = _failed_action(
                    action_index,
                    projection,
                    WorkshopExecutionFailureCode.POSTCONDITION_FAILURE,
                    _failure_message(
                        postcondition_result, "physical postcondition failed"
                    ),
                    controller_result=controller_result,
                    postcondition_result=postcondition_result,
                )
                results.append(action)
                return self._finish(results, effects, action, output_root)

            action = _successful_action(
                action_index,
                projection,
                controller_result,
                postcondition_result,
            )
            results.append(action)
            if operator == "pick-from":
                held = projection.controller_arguments[0]
            elif operator in {"insert", "place-on"}:
                held = None
                if operator == "insert":
                    inserted = (
                        projection.controller_arguments[0],
                        projection.controller_arguments[1],
                    )
            elif operator == "drive":
                effects.append(
                    _drive_effect(action_index, projection, controller_result)
                )

        return self._finish(results, effects, None, output_root)

    @staticmethod
    def _finish(
        actions: Sequence[WorkshopActionExecution],
        effects: Sequence[WorkshopDriveEffect],
        failed_action: WorkshopActionExecution | None,
        output_root: str | Path | None,
    ) -> WorkshopExecutionResult:
        result = WorkshopExecutionResult(
            success=failed_action is None,
            actions=tuple(actions),
            effect_ledger=tuple(effects),
            terminal_failure_code=(
                failed_action.failure_code if failed_action is not None else None
            ),
            terminal_failure_message=(
                failed_action.failure_message if failed_action is not None else None
            ),
        )
        if output_root is not None:
            destination = Path(output_root)
            atomic_write_json(
                destination / "workshop_execution_effect_ledger.json",
                {"effects": [effect.to_dict() for effect in effects]},
            )
            atomic_write_json(
                destination / "workshop_execution.json", result.to_dict()
            )
        return result


def _operator(projection: ExecutionProjection) -> str:
    return projection.pddl_operator.strip().lower().replace("_", "-")


def _validate_projection(projection: ExecutionProjection) -> str:
    if not isinstance(projection, ExecutionProjection):
        raise WorkshopExecutionContractError(
            "workshop execution requires ExecutionProjection contracts"
        )
    operator = _operator(projection)
    rule = _RULES.get(operator)
    if rule is None:
        raise WorkshopExecutionContractError(
            f"UNSUPPORTED_CONTROLLER_ACTION: {operator!r}",
            failure_code=WorkshopExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    expected_controller, arity = rule
    if (
        projection.controller_operator != expected_controller
        or len(projection.pddl_arguments) != arity
        or len(projection.controller_arguments) != arity
    ):
        raise WorkshopExecutionContractError(
            f"invalid {operator!r} controller projection",
            failure_code=WorkshopExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    if (
        projection.controller_arguments != projection.resolved_entities
        or any(not argument.strip() for argument in projection.controller_arguments)
    ):
        raise WorkshopExecutionContractError(
            "UNRESOLVED_ENTITY: projected controller identity mismatch",
            failure_code=WorkshopExecutionFailureCode.UNRESOLVED_ENTITY,
        )
    return operator


def _validate_against_controller_contract(
    projection: ExecutionProjection,
    contract: WorkshopControllerContract,
) -> None:
    operator = _operator(projection)
    if operator == "drive" and (
        projection.pddl_arguments
        != (contract.driver_id, contract.fastener_id, contract.target_id)
        or projection.controller_arguments
        != (contract.driver, contract.fastener, contract.target_joint)
        or projection.action_instance_id != contract.drive_action_instance_id
    ):
        raise WorkshopExecutionContractError(
            "DRIVE tuple differs from the action-derived controller contract",
            failure_code=WorkshopExecutionFailureCode.UNRESOLVED_ENTITY,
        )
    if operator == "insert" and (
        projection.pddl_arguments != (contract.fastener_id, contract.target_id)
        or projection.controller_arguments != (contract.fastener, contract.target_joint)
    ):
        raise WorkshopExecutionContractError(
            "INSERT tuple differs from the plan-specified fastener and target",
            failure_code=WorkshopExecutionFailureCode.UNRESOLVED_ENTITY,
        )
    if operator == "place-on" and projection.pddl_arguments[0] == contract.driver_id:
        if (
            projection.pddl_arguments[1] != contract.work_surface_id
            or projection.controller_arguments
            != (contract.driver, contract.work_surface)
        ):
            raise WorkshopExecutionContractError(
                "driver PLACE-ON differs from its plan-specified destination",
                failure_code=WorkshopExecutionFailureCode.UNRESOLVED_ENTITY,
            )


def _precondition_error(
    operator: str,
    projection: ExecutionProjection,
    *,
    held: str | None,
    inserted: tuple[str, str] | None,
    controller_contract: WorkshopControllerContract,
) -> str | None:
    arguments = projection.controller_arguments
    if operator == "open-storage":
        return None if held is None else "OPEN requires an empty gripper"
    if operator == "pick-from":
        return None if held is None else f"PICK attempted while holding {held!r}"
    if operator in {"insert", "place-on"}:
        return (
            None
            if held == arguments[0]
            else f"PLACE requires held object {arguments[0]!r}; observed {held!r}"
        )
    if held != controller_contract.driver:
        return "DRIVE requires the exact plan-specified driver to be held"
    if inserted != (controller_contract.fastener, controller_contract.target_joint):
        return "DRIVE requires the exact plan-specified fastener to be inserted"
    return None


def _mapping_result(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} result is not a mapping")
    return dict(value)


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
    controller_result: Mapping[str, Any],
    postcondition_result: Mapping[str, Any],
) -> WorkshopActionExecution:
    return WorkshopActionExecution(
        action_index=action_index,
        action_instance_id=projection.action_instance_id,
        pddl_operator=projection.pddl_operator,
        pddl_arguments=projection.pddl_arguments,
        controller_operator=projection.controller_operator,
        controller_arguments=projection.controller_arguments,
        success=True,
        controller_result=dict(controller_result),
        postcondition_result=dict(postcondition_result),
    )


def _failed_action(
    action_index: int,
    projection: ExecutionProjection,
    failure_code: WorkshopExecutionFailureCode,
    message: str,
    *,
    controller_result: Mapping[str, Any] | None = None,
    postcondition_result: Mapping[str, Any] | None = None,
) -> WorkshopActionExecution:
    return WorkshopActionExecution(
        action_index=action_index,
        action_instance_id=projection.action_instance_id,
        pddl_operator=projection.pddl_operator,
        pddl_arguments=projection.pddl_arguments,
        controller_operator=projection.controller_operator,
        controller_arguments=projection.controller_arguments,
        success=False,
        controller_result=dict(controller_result or {}),
        postcondition_result=dict(postcondition_result or {}),
        failure_code=failure_code,
        failure_message=message,
    )


def _drive_effect(
    action_index: int,
    projection: ExecutionProjection,
    controller_result: Mapping[str, Any],
) -> WorkshopDriveEffect:
    symbolic = projection.pddl_arguments
    resolved = projection.controller_arguments
    if len(symbolic) != 3 or len(resolved) != 3:
        raise WorkshopExecutionContractError("DRIVE effect requires three arguments")
    status = controller_result.get("status")
    return WorkshopDriveEffect(
        action_index=action_index,
        action_instance_id=projection.action_instance_id,
        effect="DRIVE_COMPLETED",
        symbolic_arguments=(symbolic[0], symbolic[1], symbolic[2]),
        resolved_entities=(resolved[0], resolved[1], resolved[2]),
        controller_status=str(status) if status is not None else None,
    )
