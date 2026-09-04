"""Direct kitchen execution for independently projected ViLaIn actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..artifacts import atomic_write_json
from ..contracts import ExecutionProjection, SerializableContract
from ..identity import EntityBinding


class KitchenExecutionContractError(ValueError):
    """Raised when projected execution input violates the baseline contract."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: KitchenExecutionFailureCode | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class KitchenExecutionFailureCode(str, Enum):
    UNRESOLVED_ENTITY = "UNRESOLVED_ENTITY"
    UNSUPPORTED_CONTROLLER_ACTION = "UNSUPPORTED_CONTROLLER_ACTION"
    CONTROLLER_EXCEPTION = "CONTROLLER_EXCEPTION"
    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"
    POSTCONDITION_FAILURE = "POSTCONDITION_FAILURE"


@dataclass(frozen=True)
class KitchenInventoryEntry(SerializableContract):
    object_id: str
    entity_name: str
    pddl_type: str
    broad_class: str
    fixed: bool
    source_location_id: str | None
    source_location_entity: str | None
    binding_method: str
    binding_confidence: float
    evidence_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class KitchenExecutionInventory(SerializableContract):
    entries: tuple[KitchenInventoryEntry, ...]

    def by_object_id(self) -> dict[str, KitchenInventoryEntry]:
        return {entry.object_id: entry for entry in self.entries}

    def controller_payload(self) -> dict[str, Any]:
        """Expose neutral IDs and controller bodies without semantic assignments."""
        return {
            "execution_mode": "VILAIN_TAMP_BASELINE",
            "objects": [
                {
                    "generic_object_id": entry.object_id,
                    "physical_backend_body": entry.entity_name,
                    "pddl_type": entry.pddl_type,
                    "broad_class": entry.broad_class,
                    "fixed": entry.fixed,
                    "source_location_id": entry.source_location_id,
                    "source_location_entity": entry.source_location_entity,
                    "resolution_method": entry.binding_method,
                    "binding_confidence": entry.binding_confidence,
                    "evidence_artifacts": list(entry.evidence_artifacts),
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True)
class KitchenEffectLedgerEntry(SerializableContract):
    action_index: int
    action_instance_id: str
    effect: str
    symbolic_arguments: tuple[str, ...]
    resolved_entities: tuple[str, ...]
    controller_status: str | None


@dataclass(frozen=True)
class KitchenActionExecution(SerializableContract):
    action_index: int
    action_instance_id: str
    pddl_operator: str
    pddl_arguments: tuple[str, ...]
    controller_operator: str
    controller_arguments: tuple[str, ...]
    success: bool
    controller_success: bool
    postcondition_success: bool
    controller_result: Mapping[str, Any]
    postcondition_result: Mapping[str, Any]
    failure_code: KitchenExecutionFailureCode | None
    failure_message: str | None


@dataclass(frozen=True)
class KitchenExecutionResult(SerializableContract):
    success: bool
    actions: tuple[KitchenActionExecution, ...]
    effect_ledger: tuple[KitchenEffectLedgerEntry, ...]
    terminal_failure_code: KitchenExecutionFailureCode | None
    terminal_failure_message: str | None


class KitchenControllerContract(Protocol):
    """Minimal interface implemented by a kitchen controller facade."""

    def execute_action(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def verify_postcondition(
        self,
        request: Mapping[str, Any],
        controller_result: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


_PROJECTION_RULES: dict[str, tuple[str, int, tuple[int, ...]]] = {
    "open-storage": ("OPEN", 1, (0,)),
    "pick-from": ("PICK", 2, (0,)),
    "place-on": ("PLACE", 2, (0, 1)),
    "pour": ("POUR", 3, (0, 1)),
    "stir": ("STIR", 2, (0, 1)),
    "place-in": ("PLACE", 2, (0, 1)),
}

_IDENTITY_ARGUMENTS: dict[str, tuple[int, ...]] = {
    "open-storage": (0,),
    "pick-from": (0, 1),
    "place-on": (0, 1),
    "pour": (0, 1),
    "stir": (0, 1),
    "place-in": (0, 1),
}


def build_kitchen_inventory(
    projections: Sequence[ExecutionProjection],
    bindings: Mapping[str, EntityBinding],
    *,
    fixed_bindings: Mapping[str, EntityBinding] | None = None,
    external_method_artifacts: Mapping[str, Any] | None = None,
) -> KitchenExecutionInventory:
    """Build only the inventory identities referenced by the projected plan."""
    if external_method_artifacts is not None:
        raise KitchenExecutionContractError(
            "external method artifacts are not valid kitchen inventory input"
        )
    _validate_binding_map(bindings, "movable")
    fixed = dict(fixed_bindings or {})
    _validate_binding_map(fixed, "fixed")
    overlap = set(bindings).intersection(fixed)
    if overlap:
        raise KitchenExecutionContractError(
            f"duplicate movable/fixed identity IDs: {sorted(overlap)!r}"
        )
    available = {**fixed, **dict(bindings)}
    referenced: list[str] = []
    source_locations: dict[str, tuple[str, str]] = {}

    for projection in projections:
        operator, _, _ = _validate_projection(projection)
        for argument_index in _IDENTITY_ARGUMENTS[operator]:
            symbolic_id = projection.pddl_arguments[argument_index]
            if symbolic_id not in available:
                raise KitchenExecutionContractError(
                    f"UNRESOLVED_ENTITY: {symbolic_id!r}"
                )
            if symbolic_id not in referenced:
                referenced.append(symbolic_id)
        if operator == "pick-from":
            object_id, location_id = projection.pddl_arguments
            location = available[location_id]
            previous = source_locations.get(object_id)
            current = (location_id, location.entity_name)
            if previous is not None and previous != current:
                raise KitchenExecutionContractError(
                    f"conflicting PICK source locations for {object_id!r}"
                )
            source_locations[object_id] = current

    entries = []
    for symbolic_id in referenced:
        binding = available[symbolic_id]
        source = source_locations.get(symbolic_id)
        entries.append(
            KitchenInventoryEntry(
                object_id=symbolic_id,
                entity_name=binding.entity_name,
                pddl_type=binding.pddl_type,
                broad_class=binding.broad_class,
                fixed=symbolic_id in fixed,
                source_location_id=source[0] if source else None,
                source_location_entity=source[1] if source else None,
                binding_method=binding.binding_method,
                binding_confidence=binding.confidence,
                evidence_artifacts=binding.evidence_artifacts,
            )
        )
    inventory = KitchenExecutionInventory(tuple(entries))
    _validate_projection_entities(projections, inventory)
    return inventory


class KitchenExecutionAdapter:
    """Execute projected actions directly and stop at the first failed action."""

    def __init__(
        self,
        *,
        controller: KitchenControllerContract,
        inventory: KitchenExecutionInventory,
    ) -> None:
        if not callable(getattr(controller, "execute_action", None)):
            raise KitchenExecutionContractError("controller has no execute_action method")
        if not callable(getattr(controller, "verify_postcondition", None)):
            raise KitchenExecutionContractError(
                "controller has no verify_postcondition method"
            )
        if not isinstance(inventory, KitchenExecutionInventory):
            raise KitchenExecutionContractError(
                "kitchen execution requires a KitchenExecutionInventory"
            )
        self.controller = controller
        self.inventory = inventory

    def execute(
        self,
        projections: Sequence[ExecutionProjection],
        *,
        output_root: str | Path | None = None,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> KitchenExecutionResult:
        if external_method_artifacts is not None:
            raise KitchenExecutionContractError(
                "external method artifacts are not valid kitchen execution input"
            )
        action_results: list[KitchenActionExecution] = []
        effects: list[KitchenEffectLedgerEntry] = []

        for action_index, projection in enumerate(projections):
            try:
                _validate_projection(projection)
                _validate_projection_entities((projection,), self.inventory)
            except KitchenExecutionContractError as error:
                if error.failure_code is None:
                    raise
                action = _failed_action(
                    action_index,
                    projection,
                    error.failure_code,
                    str(error),
                )
                action_results.append(action)
                return self._finish(action_results, effects, action, output_root)
            request = {
                "action_index": action_index,
                "action_instance_id": projection.action_instance_id,
                "operator": projection.controller_operator,
                "arguments": list(projection.controller_arguments),
                "pddl_operator": projection.pddl_operator,
            }
            try:
                controller_value = self.controller.execute_action(request)
            except Exception as error:
                action = _failed_action(
                    action_index,
                    projection,
                    KitchenExecutionFailureCode.CONTROLLER_EXCEPTION,
                    f"{type(error).__name__}: {error}",
                )
                action_results.append(action)
                return self._finish(action_results, effects, action, output_root)
            if not isinstance(controller_value, Mapping):
                action = _failed_action(
                    action_index,
                    projection,
                    KitchenExecutionFailureCode.CONTROLLER_FAILURE,
                    "controller result is not a mapping",
                )
                action_results.append(action)
                return self._finish(action_results, effects, action, output_root)
            controller_result = dict(controller_value)
            if not bool(controller_result.get("success", False)):
                message = str(
                    controller_result.get("message")
                    or controller_result.get("status")
                    or "controller reported failure"
                )
                action = _failed_action(
                    action_index,
                    projection,
                    KitchenExecutionFailureCode.CONTROLLER_FAILURE,
                    message,
                    controller_result=controller_result,
                )
                action_results.append(action)
                return self._finish(action_results, effects, action, output_root)

            try:
                postcondition_value = self.controller.verify_postcondition(
                    request, controller_result
                )
            except Exception as error:
                postcondition_value = {
                    "success": False,
                    "message": f"{type(error).__name__}: {error}",
                }
            if not isinstance(postcondition_value, Mapping):
                postcondition_result = {
                    "success": False,
                    "message": "postcondition result is not a mapping",
                }
            else:
                postcondition_result = dict(postcondition_value)
            if not bool(postcondition_result.get("success", False)):
                message = str(
                    postcondition_result.get("message")
                    or postcondition_result.get("reason")
                    or "physical postcondition was not verified"
                )
                action = _failed_action(
                    action_index,
                    projection,
                    KitchenExecutionFailureCode.POSTCONDITION_FAILURE,
                    message,
                    controller_result=controller_result,
                    postcondition_result=postcondition_result,
                )
                action_results.append(action)
                return self._finish(action_results, effects, action, output_root)

            action = KitchenActionExecution(
                action_index=action_index,
                action_instance_id=projection.action_instance_id,
                pddl_operator=projection.pddl_operator,
                pddl_arguments=projection.pddl_arguments,
                controller_operator=projection.controller_operator,
                controller_arguments=projection.controller_arguments,
                success=True,
                controller_success=True,
                postcondition_success=True,
                controller_result=controller_result,
                postcondition_result=postcondition_result,
                failure_code=None,
                failure_message=None,
            )
            action_results.append(action)
            effect = _verified_effect(action_index, projection, controller_result)
            if effect is not None:
                effects.append(effect)

        return self._finish(action_results, effects, None, output_root)

    def _finish(
        self,
        actions: Sequence[KitchenActionExecution],
        effects: Sequence[KitchenEffectLedgerEntry],
        failed_action: KitchenActionExecution | None,
        output_root: str | Path | None,
    ) -> KitchenExecutionResult:
        result = KitchenExecutionResult(
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
                destination / "kitchen_inventory.json",
                self.inventory.controller_payload(),
            )
            atomic_write_json(
                destination / "execution_effect_ledger.json",
                {"effects": [entry.to_dict() for entry in effects]},
            )
            atomic_write_json(destination / "kitchen_execution.json", result.to_dict())
        return result


def _validate_binding_map(
    bindings: Mapping[str, EntityBinding], label: str
) -> None:
    for key, binding in bindings.items():
        if not isinstance(binding, EntityBinding):
            raise KitchenExecutionContractError(
                f"{label} identities must use EntityBinding contracts"
            )
        if key != binding.object_id:
            raise KitchenExecutionContractError(
                f"{label} identity keys must match their object IDs"
            )


def _validate_projection(
    projection: ExecutionProjection,
) -> tuple[str, str, tuple[int, ...]]:
    if not isinstance(projection, ExecutionProjection):
        raise KitchenExecutionContractError(
            "kitchen execution requires ExecutionProjection contracts"
        )
    operator = projection.pddl_operator.strip().lower().replace("_", "-")
    rule = _PROJECTION_RULES.get(operator)
    if rule is None:
        raise KitchenExecutionContractError(
            f"UNSUPPORTED_CONTROLLER_ACTION: {operator!r}",
            failure_code=KitchenExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    expected_controller, arity, controller_indices = rule
    if len(projection.pddl_arguments) != arity:
        raise KitchenExecutionContractError(
            f"{operator!r} expects {arity} PDDL arguments",
            failure_code=KitchenExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    if projection.controller_operator != expected_controller:
        raise KitchenExecutionContractError(
            f"projection for {operator!r} must call {expected_controller}",
            failure_code=KitchenExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    expected_arguments = tuple(
        projection.resolved_entities[index]
        for index in range(len(projection.resolved_entities))
    )
    if projection.controller_arguments != expected_arguments:
        raise KitchenExecutionContractError(
            "controller arguments differ from resolved projected entities",
            failure_code=KitchenExecutionFailureCode.UNRESOLVED_ENTITY,
        )
    if len(projection.controller_arguments) != len(controller_indices):
        raise KitchenExecutionContractError(
            f"projection for {operator!r} has the wrong controller arity",
            failure_code=KitchenExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION,
        )
    return operator, expected_controller, controller_indices


def _validate_projection_entities(
    projections: Sequence[ExecutionProjection],
    inventory: KitchenExecutionInventory,
) -> None:
    by_id = inventory.by_object_id()
    for projection in projections:
        operator, _, controller_indices = _validate_projection(projection)
        for resolved_index, pddl_index in enumerate(controller_indices):
            symbolic_id = projection.pddl_arguments[pddl_index]
            entry = by_id.get(symbolic_id)
            if entry is None:
                raise KitchenExecutionContractError(
                    f"UNRESOLVED_ENTITY: {symbolic_id!r}",
                    failure_code=KitchenExecutionFailureCode.UNRESOLVED_ENTITY,
                )
            if entry.entity_name != projection.controller_arguments[resolved_index]:
                raise KitchenExecutionContractError(
                    f"identity mismatch for projected entity {symbolic_id!r}",
                    failure_code=KitchenExecutionFailureCode.UNRESOLVED_ENTITY,
                )
        if operator == "pick-from":
            object_id, location_id = projection.pddl_arguments
            object_entry = by_id.get(object_id)
            location_entry = by_id.get(location_id)
            if object_entry is None or location_entry is None:
                raise KitchenExecutionContractError(
                    f"UNRESOLVED_ENTITY: PICK source for {object_id!r}",
                    failure_code=KitchenExecutionFailureCode.UNRESOLVED_ENTITY,
                )
            if (
                object_entry.source_location_id != location_id
                or object_entry.source_location_entity != location_entry.entity_name
            ):
                raise KitchenExecutionContractError(
                    f"inventory source mismatch for {object_id!r}",
                    failure_code=KitchenExecutionFailureCode.UNRESOLVED_ENTITY,
                )


def _failed_action(
    action_index: int,
    projection: ExecutionProjection,
    failure_code: KitchenExecutionFailureCode,
    message: str,
    *,
    controller_result: Mapping[str, Any] | None = None,
    postcondition_result: Mapping[str, Any] | None = None,
) -> KitchenActionExecution:
    return KitchenActionExecution(
        action_index=action_index,
        action_instance_id=projection.action_instance_id,
        pddl_operator=projection.pddl_operator,
        pddl_arguments=projection.pddl_arguments,
        controller_operator=projection.controller_operator,
        controller_arguments=projection.controller_arguments,
        success=False,
        controller_success=bool(
            controller_result and controller_result.get("success", False)
        ),
        postcondition_success=bool(
            postcondition_result and postcondition_result.get("success", False)
        ),
        controller_result=dict(controller_result or {}),
        postcondition_result=dict(postcondition_result or {}),
        failure_code=failure_code,
        failure_message=message,
    )


def _verified_effect(
    action_index: int,
    projection: ExecutionProjection,
    controller_result: Mapping[str, Any],
) -> KitchenEffectLedgerEntry | None:
    operator = projection.pddl_operator.strip().lower().replace("_", "-")
    if operator == "pour":
        effect = "POUR_COMPLETED"
        symbolic_arguments = projection.pddl_arguments
    elif operator == "stir":
        effect = "STIR_COMPLETED"
        symbolic_arguments = projection.pddl_arguments
    else:
        return None
    status_value = controller_result.get("status")
    return KitchenEffectLedgerEntry(
        action_index=action_index,
        action_instance_id=projection.action_instance_id,
        effect=effect,
        symbolic_arguments=symbolic_arguments,
        resolved_entities=projection.resolved_entities,
        controller_status=str(status_value) if status_value is not None else None,
    )
