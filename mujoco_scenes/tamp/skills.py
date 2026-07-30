"""Common symbolic interface for asynchronous simulator skills."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class FailureCode(StrEnum):
    PRECONDITION_FAILED = "precondition_failed"
    PATH_BLOCKED = "path_blocked"
    IK_FAILED = "ik_failed"
    COLLISION = "collision"
    GRASP_FAILED = "grasp_failed"
    PLACEMENT_FAILED = "placement_failed"
    OBJECT_NOT_VISIBLE = "object_not_visible"
    TARGET_OCCUPIED = "target_occupied"
    FUNCTION_UNSATISFIED = "function_unsatisfied"
    EFFECT_NOT_OBSERVED = "effect_not_observed"
    NO_CANDIDATE = "no_candidate"
    INFERENCE_FAILED = "inference_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class SkillAction:
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    success: bool
    effects: tuple[str, ...] = ()
    failure_code: FailureCode | None = None
    message: str = ""
    recoverable: bool = True

    @classmethod
    def succeeded(cls, *effects: str) -> SkillResult:
        return cls(True, tuple(effects))

    @classmethod
    def failed(
        cls,
        code: FailureCode,
        message: str,
        *,
        recoverable: bool = True,
    ) -> SkillResult:
        return cls(
            False,
            failure_code=code,
            message=message,
            recoverable=recoverable,
        )


class SkillStartError(RuntimeError):
    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


class SkillDispatcher(Protocol):
    def start(self, action: SkillAction) -> None:
        """Start one action or raise when its preconditions are false."""

    def update(self) -> SkillResult | None:
        """Advance the active action and return its terminal result."""
