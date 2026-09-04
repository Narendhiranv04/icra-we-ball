"""Cloned-scene sequence preflight for the independent ViLaIn baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .artifacts import atomic_write_json
from .contracts import (
    ExecutionProjection,
    RefinementFailure,
    RefinementStage,
    SerializableContract,
    SymbolicAction,
)


ADAPTATION_LABEL = (
    "MuJoCo cloned-scene sequence preflight, an adaptation of ViLaIn-TAMP’s "
    "MoveIt Task Constructor refinement."
)


class RefinementContractError(ValueError):
    """Raised when preflight inputs violate the baseline refinement contract."""


@dataclass(frozen=True)
class RefinementStageContext:
    attempt_index: int
    action: SymbolicAction
    projection: ExecutionProjection
    stage: RefinementStage
    stage_purpose: str
    interaction_mode: str
    predicted_state: Mapping[str, Any]


@dataclass(frozen=True)
class RefinementStageOutcome(SerializableContract):
    stage: RefinementStage
    success: bool
    candidate_artifacts: tuple[str, ...] = ()
    trajectory_artifacts: tuple[str, ...] = ()
    resolved_entities: tuple[str, ...] = ()
    chosen_grasp: Mapping[str, Any] | None = None
    chosen_target_pose: Mapping[str, Any] | None = None
    collision_free: bool | None = None
    reachable: bool | None = None
    numeric_evidence: Mapping[str, float] = field(default_factory=dict)
    predicted_terminal_state: Mapping[str, Any] | None = None
    reason_code: str | None = None
    summary: str | None = None
    robot_or_arm: str | None = None
    involved_entities: tuple[str, ...] = ()
    collision_pair: tuple[str, str] | None = None
    backend_trace_artifact: str | None = None
    recoverable_by_problem_revision: bool = True

    def __post_init__(self) -> None:
        if self.success and (self.reason_code is not None or self.summary is not None):
            raise ValueError("successful stage outcomes must not contain failure details")
        if not self.success and (not self.reason_code or not self.summary):
            raise ValueError("failed stage outcomes require a reason code and summary")
        if self.stage is RefinementStage.STATE_TRANSITION:
            if self.success and self.predicted_terminal_state is None:
                raise ValueError("state-transition success requires a predicted terminal state")
        elif self.predicted_terminal_state is not None:
            raise ValueError("only state-transition outcomes may predict terminal state")


@dataclass(frozen=True)
class ActionRefinementCertificate(SerializableContract):
    action_index: int
    action_instance_id: str
    operator: str
    arguments: tuple[str, ...]
    resolved_entities: tuple[str, ...]
    stages: tuple[RefinementStageOutcome, ...]
    predicted_terminal_state: Mapping[str, Any]
    elapsed_seconds: float


@dataclass(frozen=True)
class SequenceRefinementCertificate(SerializableContract):
    attempt_index: int
    adaptation_label: str
    actions: tuple[ActionRefinementCertificate, ...]
    final_predicted_state: Mapping[str, Any]
    elapsed_seconds: float


@dataclass(frozen=True)
class SequenceRefinementResult(SerializableContract):
    success: bool
    certificate: SequenceRefinementCertificate | None
    failure: RefinementFailure | None

    def __post_init__(self) -> None:
        if self.success != (self.certificate is not None):
            raise ValueError("success must contain exactly one refinement certificate")
        if self.success == (self.failure is not None):
            raise ValueError("failure must contain exactly one structured error")


class RefinementStageBackend(Protocol):
    """One pluggable geometric stage operating only on a planning copy."""

    def refine_stage(
        self,
        planning_scene: Any,
        context: RefinementStageContext,
    ) -> RefinementStageOutcome: ...


class PlanningSceneStateAdapter(Protocol):
    """Read and advance predicted state on the planning scene only."""

    def snapshot(self, planning_scene: Any) -> Mapping[str, Any]: ...

    def apply_predicted_transition(
        self,
        planning_scene: Any,
        action: SymbolicAction,
        predicted_terminal_state: Mapping[str, Any],
    ) -> None: ...


_STANDARD_SEQUENCE = (
    RefinementStage.ENTITY_RESOLUTION,
    RefinementStage.GRASP_GENERATION,
    RefinementStage.IK,
    RefinementStage.TRAJECTORY,
    RefinementStage.COLLISION,
    RefinementStage.SKILL_ENVELOPE,
    RefinementStage.STATE_TRANSITION,
)

_SUPPORTED_OPERATORS = frozenset(
    {
        "open-storage",
        "pick-from",
        "place-on",
        "insert",
        "pour",
        "stir",
        "drive",
        "place-in",
    }
)

_PURPOSES: dict[str, dict[RefinementStage, str]] = {
    "open-storage": {
        RefinementStage.GRASP_GENERATION: "handle_pose_candidates",
        RefinementStage.SKILL_ENVELOPE: "opening_start_end_envelope",
    },
    "pick-from": {
        RefinementStage.GRASP_GENERATION: "object_grasp_candidates",
        RefinementStage.TRAJECTORY: "pregrasp_approach_attach_retreat",
        RefinementStage.SKILL_ENVELOPE: "pick_start_end_envelope",
    },
    "place-on": {
        RefinementStage.GRASP_GENERATION: "target_pose_candidates",
        RefinementStage.TRAJECTORY: "transfer_release_retreat",
        RefinementStage.SKILL_ENVELOPE: "place_start_end_envelope",
    },
    "insert": {
        RefinementStage.GRASP_GENERATION: "insertion_pose_candidates",
        RefinementStage.TRAJECTORY: "transfer_insertion_release",
        RefinementStage.SKILL_ENVELOPE: "insertion_start_end_envelope",
    },
    "pour": {
        RefinementStage.GRASP_GENERATION: "source_grasp_and_target_relative_pose",
        RefinementStage.TRAJECTORY: "approach_pour_return_paths",
        RefinementStage.SKILL_ENVELOPE: "pour_object_centric_envelope",
    },
    "stir": {
        RefinementStage.GRASP_GENERATION: "tool_grasp_and_target_relative_pose",
        RefinementStage.TRAJECTORY: "approach_insert_return_paths",
        RefinementStage.SKILL_ENVELOPE: "stir_object_centric_envelope",
    },
    "drive": {
        RefinementStage.GRASP_GENERATION: "driver_grasp_and_axis_alignment",
        RefinementStage.TRAJECTORY: "approach_align_return_paths",
        RefinementStage.SKILL_ENVELOPE: "drive_object_centric_envelope",
    },
    "place-in": {
        RefinementStage.GRASP_GENERATION: "vessel_relative_target_pose",
        RefinementStage.TRAJECTORY: "transfer_release_retreat",
        RefinementStage.SKILL_ENVELOPE: "place_in_start_end_envelope",
    },
}

_DEFAULT_PURPOSES = {
    RefinementStage.ENTITY_RESOLUTION: "resolved_execution_entities",
    RefinementStage.GRASP_GENERATION: "interaction_candidates",
    RefinementStage.IK: "reachability_and_inverse_kinematics",
    RefinementStage.TRAJECTORY: "collision_checked_approach_transfer_retreat",
    RefinementStage.COLLISION: "configuration_and_path_collision_certificate",
    RefinementStage.SKILL_ENVELOPE: "object_centric_start_end_envelope",
    RefinementStage.STATE_TRANSITION: "predicted_terminal_transition",
}

_BLACK_BOX_OPERATORS = frozenset({"pour", "stir", "drive"})


class MuJoCoSequenceRefiner:
    """Coordinate full-plan preflight on one fresh planning-scene clone."""

    def __init__(
        self,
        *,
        stage_backends: Mapping[RefinementStage, RefinementStageBackend],
        state_adapter: PlanningSceneStateAdapter,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        missing = set(_STANDARD_SEQUENCE).difference(stage_backends)
        if missing:
            raise RefinementContractError(
                "missing refinement stage backends: "
                + ", ".join(stage.value for stage in sorted(missing, key=lambda item: item.value))
            )
        self.stage_backends = dict(stage_backends)
        self.state_adapter = state_adapter
        self.clock = clock

    def refine(
        self,
        *,
        attempt_index: int,
        actions: Sequence[SymbolicAction],
        projections: Sequence[ExecutionProjection],
        planning_scene_factory: Callable[[], Any],
        output_root: str | Path | None = None,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> SequenceRefinementResult:
        if external_method_artifacts is not None:
            raise RefinementContractError(
                "external method artifacts are not valid refinement input"
            )
        if attempt_index < 0:
            raise RefinementContractError("attempt_index must be non-negative")
        if len(actions) != len(projections):
            raise RefinementContractError("actions and projections must have equal length")
        for action, projection in zip(actions, projections):
            if not isinstance(action, SymbolicAction) or not isinstance(
                projection, ExecutionProjection
            ):
                raise RefinementContractError(
                    "refinement requires symbolic-action and execution-projection contracts"
                )
            if action.action_instance_id != projection.action_instance_id:
                raise RefinementContractError("action/projection identity mismatch")
            if (
                action.operator.lower().replace("_", "-")
                != projection.pddl_operator.lower().replace("_", "-")
                or action.arguments != projection.pddl_arguments
            ):
                raise RefinementContractError("action/projection content mismatch")

        started = self.clock()
        planning_scene = planning_scene_factory()
        predicted_state = dict(self.state_adapter.snapshot(planning_scene))
        action_certificates: list[ActionRefinementCertificate] = []
        for action, projection in zip(actions, projections):
            operator = action.operator.lower().replace("_", "-")
            if operator not in _SUPPORTED_OPERATORS:
                failure = _unsupported_failure(attempt_index, action, projection)
                return self._failed(failure, output_root)
            action_started = self.clock()
            stage_outcomes: list[RefinementStageOutcome] = []
            for stage in _STANDARD_SEQUENCE:
                context = RefinementStageContext(
                    attempt_index=attempt_index,
                    action=action,
                    projection=projection,
                    stage=stage,
                    stage_purpose=_stage_purpose(operator, stage),
                    interaction_mode=(
                        "BLACK_BOX_SKILL_ENVELOPE"
                        if stage is RefinementStage.SKILL_ENVELOPE
                        and operator in _BLACK_BOX_OPERATORS
                        else "GEOMETRIC_PREFLIGHT"
                    ),
                    predicted_state=dict(predicted_state),
                )
                try:
                    outcome = self.stage_backends[stage].refine_stage(
                        planning_scene, context
                    )
                except Exception as error:
                    outcome = RefinementStageOutcome(
                        stage=stage,
                        success=False,
                        resolved_entities=projection.resolved_entities,
                        reason_code="REFINEMENT_BACKEND_EXCEPTION",
                        summary=f"{type(error).__name__}: {error}",
                        involved_entities=projection.resolved_entities,
                        recoverable_by_problem_revision=False,
                    )
                if not isinstance(outcome, RefinementStageOutcome):
                    raise RefinementContractError(
                        f"backend returned an invalid outcome for {stage.value}"
                    )
                if outcome.stage is not stage:
                    raise RefinementContractError(
                        f"backend returned {outcome.stage.value} for {stage.value}"
                    )
                if (
                    outcome.success
                    and stage is RefinementStage.ENTITY_RESOLUTION
                    and outcome.resolved_entities != projection.resolved_entities
                ):
                    raise RefinementContractError(
                        "entity-resolution outcome differs from execution projection"
                    )
                stage_outcomes.append(outcome)
                if not outcome.success:
                    failure = _failure_from_outcome(
                        attempt_index, action, projection, outcome
                    )
                    return self._failed(failure, output_root)
                if stage is RefinementStage.STATE_TRANSITION:
                    terminal = dict(outcome.predicted_terminal_state or {})
                    self.state_adapter.apply_predicted_transition(
                        planning_scene, action, terminal
                    )
                    predicted_state = dict(self.state_adapter.snapshot(planning_scene))
            action_certificates.append(
                ActionRefinementCertificate(
                    action_index=action.action_index,
                    action_instance_id=action.action_instance_id,
                    operator=operator,
                    arguments=action.arguments,
                    resolved_entities=projection.resolved_entities,
                    stages=tuple(stage_outcomes),
                    predicted_terminal_state=dict(predicted_state),
                    elapsed_seconds=self.clock() - action_started,
                )
            )

        certificate = SequenceRefinementCertificate(
            attempt_index=attempt_index,
            adaptation_label=ADAPTATION_LABEL,
            actions=tuple(action_certificates),
            final_predicted_state=dict(predicted_state),
            elapsed_seconds=self.clock() - started,
        )
        result = SequenceRefinementResult(True, certificate, None)
        if output_root is not None:
            atomic_write_json(Path(output_root) / "refinement.json", result.to_dict())
        return result

    @staticmethod
    def _failed(
        failure: RefinementFailure,
        output_root: str | Path | None,
    ) -> SequenceRefinementResult:
        result = SequenceRefinementResult(False, None, failure)
        if output_root is not None:
            atomic_write_json(Path(output_root) / "refinement.json", result.to_dict())
        return result


def _stage_purpose(operator: str, stage: RefinementStage) -> str:
    return _PURPOSES.get(operator, {}).get(stage, _DEFAULT_PURPOSES[stage])


def _failure_from_outcome(
    attempt_index: int,
    action: SymbolicAction,
    projection: ExecutionProjection,
    outcome: RefinementStageOutcome,
) -> RefinementFailure:
    return RefinementFailure(
        attempt_index=attempt_index,
        action_index=action.action_index,
        action_instance_id=action.action_instance_id,
        operator=action.operator.lower().replace("_", "-"),
        arguments=action.arguments,
        stage=outcome.stage,
        reason_code=outcome.reason_code or "REFINEMENT_STAGE_FAILED",
        summary=outcome.summary or "refinement stage failed",
        robot_or_arm=outcome.robot_or_arm,
        involved_entities=(
            outcome.involved_entities or projection.resolved_entities
        ),
        collision_pair=outcome.collision_pair,
        numeric_evidence=dict(outcome.numeric_evidence),
        backend_trace_artifact=outcome.backend_trace_artifact,
        recoverable_by_problem_revision=outcome.recoverable_by_problem_revision,
    )


def _unsupported_failure(
    attempt_index: int,
    action: SymbolicAction,
    projection: ExecutionProjection,
) -> RefinementFailure:
    return RefinementFailure(
        attempt_index=attempt_index,
        action_index=action.action_index,
        action_instance_id=action.action_instance_id,
        operator=action.operator,
        arguments=action.arguments,
        stage=RefinementStage.ENTITY_RESOLUTION,
        reason_code="UNSUPPORTED_REFINEMENT_ACTION",
        summary=f"no geometric refinement stages exist for {action.operator!r}",
        robot_or_arm=None,
        involved_entities=projection.resolved_entities,
        collision_pair=None,
        numeric_evidence={},
        backend_trace_artifact=None,
        recoverable_by_problem_revision=False,
    )
