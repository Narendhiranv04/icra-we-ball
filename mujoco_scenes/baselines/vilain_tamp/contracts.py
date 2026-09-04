"""Serialization-friendly data contracts owned by the ViLaIn-TAMP baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class StringEnum(str, Enum):
    """String-valued enum whose values serialize without custom encoders."""


class ObjectEstimateStatus(StringEnum):
    OBSERVED = "OBSERVED"
    AMBIGUOUS = "AMBIGUOUS"
    LOST = "LOST"


class ProblemSource(StringEnum):
    INITIAL = "INITIAL"
    CP = "CP"


class ValidationStage(StringEnum):
    INTERNAL = "INTERNAL"
    TRANSLATOR = "TRANSLATOR"
    PLAN_VAL = "PLAN_VAL"


class RefinementStage(StringEnum):
    ENTITY_RESOLUTION = "ENTITY_RESOLUTION"
    GRASP_GENERATION = "GRASP_GENERATION"
    IK = "IK"
    TRAJECTORY = "TRAJECTORY"
    COLLISION = "COLLISION"
    SKILL_ENVELOPE = "SKILL_ENVELOPE"
    STATE_TRANSITION = "STATE_TRANSITION"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class SerializableContract:
    """Shared deterministic conversion for frozen baseline records."""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ViLaInObservation(SerializableContract):
    domain: str
    observation_mode: str
    stage_id: str
    camera_frames: tuple[Mapping[str, Any], ...]
    opened_region_id: str | None
    capture_timestamp: str
    inspection_ordinal: int | None
    content_hash: str


@dataclass(frozen=True)
class ObjectEstimate(SerializableContract):
    object_id: str
    label: str
    pddl_type: str
    description: str
    detections: tuple[Mapping[str, Any], ...]
    estimated_centroid_m: tuple[float, float, float] | None
    centroid_covariance: tuple[tuple[float, float, float], ...] | None
    observation_stage_ids: tuple[str, ...]
    status: ObjectEstimateStatus


@dataclass(frozen=True)
class GeneratedPDDLProblem(SerializableContract):
    attempt_index: int
    source: ProblemSource
    domain_name: str
    domain_sha256: str
    problem_text: str
    declared_objects: tuple[str, ...]
    initial_atoms: tuple[str, ...]
    goal_atoms: tuple[str, ...]
    raw_response_artifact: str
    problem_sha256: str


@dataclass(frozen=True)
class PDDLValidationResult(SerializableContract):
    valid: bool
    stage: ValidationStage
    diagnostics: tuple[str, ...]
    exit_code: int | None
    stdout_artifact: str | None
    stderr_artifact: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class SymbolicAction(SerializableContract):
    action_index: int
    action_instance_id: str
    operator: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class SymbolicPlan(SerializableContract):
    attempt_index: int
    planner_name: str
    planner_version: str
    search_configuration: str
    actions: tuple[SymbolicAction, ...]
    plan_cost: float | None
    planner_time_seconds: float
    raw_plan_artifacts: tuple[str, ...]
    plan_sha256: str


@dataclass(frozen=True)
class RefinementFailure(SerializableContract):
    attempt_index: int
    action_index: int | None
    action_instance_id: str | None
    operator: str | None
    arguments: tuple[str, ...]
    stage: RefinementStage
    reason_code: str
    summary: str
    robot_or_arm: str | None
    involved_entities: tuple[str, ...]
    collision_pair: tuple[str, str] | None
    numeric_evidence: Mapping[str, float]
    backend_trace_artifact: str | None
    recoverable_by_problem_revision: bool


@dataclass(frozen=True)
class CorrectivePlanningAttempt(SerializableContract):
    correction_index: int
    initial_problem_sha256: str
    prior_problem_sha256: str
    failure: RefinementFailure
    history_problem_hashes: tuple[str, ...]
    history_error_hashes: tuple[str, ...]
    model: str
    request_artifact: str
    raw_response_artifact: str
    revised_problem_sha256: str | None
    status: str
    latency_and_usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineExecutionPlan(SerializableContract):
    selected_attempt_index: int
    domain: str
    symbolic_plan: SymbolicPlan
    refinement_certificate: Mapping[str, Any]
    normalized_actions: tuple[SymbolicAction, ...]


@dataclass(frozen=True)
class ExecutionProjection(SerializableContract):
    action_instance_id: str
    pddl_operator: str
    pddl_arguments: tuple[str, ...]
    controller_operator: str
    controller_arguments: tuple[str, ...]
    resolved_entities: tuple[str, ...]
    binding_method: str
    binding_confidence: float
    binding_evidence_artifacts: tuple[str, ...]
    skill_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineRunResult(SerializableContract):
    run_status: str
    selected_attempt_index: int | None
    planning_status: str
    refinement_status: str
    execution_status: str
    generated_goal_status: str
    benchmark_status: str
    artifact_paths: Mapping[str, str]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class BenchmarkGoalEvaluation(SerializableContract):
    domain: str
    variant: str
    ground_truth_feasibility: bool
    requirement_checks: tuple[Mapping[str, Any], ...]
    actual_task_success: bool
    predicted_infeasible: bool
    correct_infeasibility_recognition: bool
    benchmark_outcome_correct: bool
    evidence_artifacts: tuple[str, ...]
