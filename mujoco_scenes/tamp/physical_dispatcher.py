"""Adapters from symbolic skills to the shared MuJoCo execution layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Protocol

from .skills import FailureCode, SkillAction, SkillResult, SkillStartError


class PhysicalDispatcher(Protocol):
    def execute_phase2_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute one canonical physical action."""


InspectionHandler = Callable[[str], Mapping[str, Any]]


_FAILURE_CODES = {
    "PRECONDITION": FailureCode.PRECONDITION_FAILED,
    "NOT_HELD": FailureCode.PRECONDITION_FAILED,
    "HELD_STATE_INVALID": FailureCode.PRECONDITION_FAILED,
    "GRIPPER_OCCUPIED": FailureCode.PRECONDITION_FAILED,
    "EMPTY_GRIPPER": FailureCode.PRECONDITION_FAILED,
    "COLLISION": FailureCode.COLLISION,
    "IK": FailureCode.IK_FAILED,
    "PATH": FailureCode.PATH_BLOCKED,
    "RRT": FailureCode.PATH_BLOCKED,
    "GRASP": FailureCode.GRASP_FAILED,
    "HELD": FailureCode.GRASP_FAILED,
    "DROP": FailureCode.GRASP_FAILED,
    "PLACE": FailureCode.PLACEMENT_FAILED,
    "DESTINATION": FailureCode.PLACEMENT_FAILED,
    "OCCUPIED": FailureCode.TARGET_OCCUPIED,
    "VISIBLE": FailureCode.OBJECT_NOT_VISIBLE,
    "RESOLUTION": FailureCode.OBJECT_NOT_VISIBLE,
    "UNSUPPORTED": FailureCode.PRECONDITION_FAILED,
}


def _json_safe(value: Any) -> Any:
    """Normalize physical telemetry before it crosses the baseline boundary."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return str(value)


def _failure_code(status: str, message: str = "") -> FailureCode:
    combined = f"{status} {message}".upper()
    occupied_hand_phrases = (
        "GRIPPER IS NOT AVAILABLE",
        "GRIPPER IS ALREADY",
        "ALREADY HOLDING",
        "IDLE EMPTY GRIPPER",
        "EMPTY HAND",
    )
    if any(phrase in combined for phrase in occupied_hand_phrases):
        return FailureCode.PRECONDITION_FAILED
    return next(
        (mapped for token, mapped in _FAILURE_CODES.items() if token in combined),
        FailureCode.INTERNAL_ERROR,
    )


def canonical_action(action: SkillAction) -> dict[str, Any]:
    """Translate public skill arguments to the Phase-B/C action contract."""
    name = action.name.upper()
    values = action.arguments

    def required(*aliases: str) -> str:
        for alias in aliases:
            value = values.get(alias)
            if isinstance(value, str) and value:
                return value
        raise SkillStartError(
            FailureCode.PRECONDITION_FAILED,
            f"{name} requires one of: {', '.join(aliases)}",
        )

    if name == "PICK":
        arguments = [required("object_id", "object")]
    elif name == "PLACE":
        arguments = [
            required("object_id", "object"),
            required("region_id", "region", "destination"),
        ]
    elif name == "POUR":
        arguments = [
            required("source_id", "source"),
            required("target_id", "target"),
        ]
        content = values.get("content")
        if isinstance(content, str) and content:
            arguments.append(content)
    elif name == "STIR":
        arguments = [
            required("tool_id", "tool"),
            required("target_id", "target"),
        ]
    elif name == "INSPECT":
        arguments = [required("region_id", "region")]
    else:
        raise SkillStartError(
            FailureCode.PRECONDITION_FAILED,
            f"Unsupported physical skill {name!r}",
        )
    return {"action": name, "arguments": arguments}


def result_to_skill(
    result: Mapping[str, Any],
    *,
    verified_effects: tuple[str, ...] = (),
) -> SkillResult:
    success = result.get("success")
    if not isinstance(success, bool):
        return SkillResult.failed(
            FailureCode.INTERNAL_ERROR,
            "Physical result field 'success' must be boolean",
            recoverable=False,
            details=_json_safe(result),
        )
    if success:
        raw_effects = result.get("effects", ())
        if isinstance(raw_effects, str) or not isinstance(
            raw_effects, (list, tuple, set, frozenset)
        ) or any(not isinstance(effect, str) for effect in raw_effects):
            return SkillResult.failed(
                FailureCode.INTERNAL_ERROR,
                "Physical result field 'effects' must be a sequence of strings",
                recoverable=False,
                details=_json_safe(result),
            )
        effects = tuple(effect for effect in raw_effects if effect)
        return SkillResult.succeeded(*(effects or verified_effects))
    status = str(
        result.get("failure_code")
        or result.get("status")
        or "EXECUTION_FAILED"
    ).upper()
    message = str(result.get("message") or status)
    code = _failure_code(status, message)
    return SkillResult.failed(
        code,
        message,
        recoverable=code is not FailureCode.INTERNAL_ERROR,
        details=_json_safe(result),
    )


def _effects_for(request: Mapping[str, Any]) -> tuple[str, ...]:
    action = str(request["action"])
    arguments = list(request["arguments"])
    if action == "PICK":
        return (f"holding({arguments[0]})",)
    if action == "PLACE":
        return (f"placed({arguments[0]},{arguments[1]})",)
    if action == "POUR":
        return (f"poured({arguments[0]},{arguments[1]})",)
    if action == "STIR":
        return (f"stirred({arguments[0]},{arguments[1]})",)
    if action == "INSPECT":
        return (f"inspected({arguments[0]})",)
    return ()


class MuJoCoSkillDispatcher:
    """Expose the synchronous Phase-C dispatcher through the TAMP interface."""

    def __init__(
        self,
        physical: PhysicalDispatcher,
        *,
        inspect: InspectionHandler | None = None,
    ):
        self.physical = physical
        self.inspect = inspect
        self._result: SkillResult | None = None
        self._active = False

    def prepare(self, actions: tuple[SkillAction, ...]) -> SkillResult:
        """Freeze an observed-state plan in a plan-aware physical backend."""
        authorize = getattr(self.physical, "authorize_plan", None)
        if not callable(authorize):
            return SkillResult.succeeded()
        try:
            authorize([canonical_action(action) for action in actions])
        except SkillStartError as error:
            return SkillResult.failed(
                error.code,
                str(error),
                recoverable=error.recoverable,
            )
        except Exception as error:
            code = _failure_code(type(error).__name__, str(error))
            return SkillResult.failed(
                code,
                str(error),
                recoverable=code is not FailureCode.INTERNAL_ERROR,
                details={
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
            )
        return SkillResult.succeeded()

    def start(self, action: SkillAction) -> None:
        if self._active:
            raise SkillStartError(
                FailureCode.PRECONDITION_FAILED,
                "Another physical skill is already active",
            )
        request = canonical_action(action)
        self._active = True
        try:
            if request["action"] == "INSPECT":
                if self.inspect is None:
                    raw: Mapping[str, Any] = {
                        "success": False,
                        "failure_code": "UNSUPPORTED_INSPECTION",
                    }
                else:
                    raw = self.inspect(request["arguments"][0])
            else:
                raw = self.physical.execute_phase2_action(request)
            self._result = result_to_skill(
                raw,
                verified_effects=_effects_for(request),
            )
        except SkillStartError:
            self._active = False
            raise
        except Exception as error:
            code = _failure_code(type(error).__name__, str(error))
            self._result = SkillResult.failed(
                code,
                str(error),
                recoverable=code is not FailureCode.INTERNAL_ERROR,
                details={
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
            )

    def update(self) -> SkillResult | None:
        if not self._active:
            return None
        result = self._result
        self._result = None
        self._active = False
        return result
