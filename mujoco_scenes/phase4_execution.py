"""Deterministic Phase-4 execution of immutable Phase-3 plan artifacts.

This module is intentionally downstream of functional grounding and planning.
It consumes the persisted phi* and final symbolic action sequence without
selecting roles, changing bindings, or adding task-level actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Protocol


FORBIDDEN_STRICT_TELEMETRY_FLAGS = frozenset({
    "assisted_execution",
    "assisted_postcondition_accepted",
    "assisted_validation",
    "direct_payload_pose_write",
    "direct_object_qpos_write",
    "direct_fastener_qpos_write",
    "direct_fastener_qpos_write_used",
    "direct_task_success_write",
    "direct_task_success_write_used",
    "direct_task_state_fallback_used",
    "initial_payload_qpos_reset_used",
})


class ExecutionFailure(str, Enum):
    NONE = "NONE"
    INVALID_HANDOFF = "INVALID_HANDOFF"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    ENTITY_MAPPING_FAILURE = "ENTITY_MAPPING_FAILURE"
    PRECONDITION_STATE_FAILURE = "PRECONDITION_STATE_FAILURE"
    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"
    POSTCONDITION_VERIFICATION_FAILURE = "POSTCONDITION_VERIFICATION_FAILURE"


class UpstreamPhase3Blocked(ValueError):
    """Raised when Phase 3 explicitly ended without an executable plan."""


class Phase4EntityMappingError(ValueError):
    """Raised before control when a frozen planner ID cannot be resolved."""


@dataclass(frozen=True)
class ResolvedEntity:
    planner_id: str
    entity_kind: str
    simulator_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionExecutionResult:
    action_index: int
    action_instance_id: str
    operator: str
    arguments: list[str]
    success: bool
    failure: str
    resolved_arguments: list[dict[str, Any]]
    primitive: str | None
    pre_check: dict[str, Any]
    controller_result: dict[str, Any] | None
    post_check: dict[str, Any]
    wall_duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase3Handoff:
    run_dir: Path
    domain: str
    variant: str
    internal_variant: str
    source: str
    specification_sha256: str | None
    assignment: dict[str, Any]
    operation_bindings: dict[str, list[dict[str, Any]]]
    inspected_regions: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    artifacts: dict[str, Path]
    artifact_sha256: dict[str, str]


class DomainExecutionAdapter(Protocol):
    entity_resolution: dict[str, Any]

    def execute_inspection_open(self, region: str) -> dict[str, Any]: ...

    def execute_action(self, action: dict[str, Any]) -> ActionExecutionResult: ...

    def final_verification(self) -> dict[str, Any]: ...


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase-3 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_strict_telemetry(
    inspection_results: list[dict[str, Any]],
    action_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive strictness from successful primitive telemetry, fail closed."""
    violations: list[dict[str, str]] = []

    def scan(value: Any, path: str, phase: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if key in FORBIDDEN_STRICT_TELEMETRY_FLAGS and child is True:
                    violations.append({
                        "phase": phase,
                        "path": child_path,
                        "flag": key,
                    })
                scan(child, child_path, phase)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, f"{path}[{index}]", phase)

    for index, row in enumerate(inspection_results):
        if row.get("success"):
            scan(row, f"inspection_results[{index}]", "INSPECTION_OPEN")
    for index, row in enumerate(action_results):
        if row.get("success"):
            scan(row, f"action_results[{index}]", "TASK_ACTION")
    return {
        "verified": not violations,
        "forbidden_flags": sorted(FORBIDDEN_STRICT_TELEMETRY_FLAGS),
        "violations": violations,
    }


def _validate_actions(actions: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(actions, list) or not actions:
        raise ValueError("Final action sequence must contain a non-empty actions list")
    validated = []
    for expected_index, raw in enumerate(actions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Action {expected_index} is not an object")
        index = raw.get("action_index", raw.get("step", expected_index - 1) + 1)
        operator = raw.get("operator")
        arguments = raw.get("arguments")
        if isinstance(arguments, dict):
            if str(operator).upper() == "PICK":
                arguments = [arguments.get("object")]
            elif str(operator).upper() == "PLACE":
                arguments = [arguments.get("object"), arguments.get("region")]
            else:
                arguments = list(arguments.values())
        instance_id = raw.get(
            "action_instance_id",
            f"fact_{expected_index:03d}_{str(operator).lower()}",
        )
        if index != expected_index:
            raise ValueError(
                f"Action index mismatch: expected {expected_index}, received {index!r}"
            )
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError(f"Action {expected_index} has no operator")
        if not isinstance(arguments, list) or not all(
            isinstance(argument, str) and argument for argument in arguments
        ):
            raise ValueError(f"Action {expected_index} has invalid arguments")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"Action {expected_index} has no action_instance_id")
        validated.append(
            {
                "action_index": index,
                "action_instance_id": instance_id,
                "operator": operator.upper(),
                "arguments": list(arguments),
            }
        )
    return tuple(validated)


def load_phase3_handoff(run_dir: Path) -> Phase3Handoff:
    """Load and fail-closed validate the immutable Phase-3 execution handoff."""
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "run_manifest.json")
    result = _read_json(run_dir / "result.json")
    grounding = _read_json(run_dir / "graph_grounding_result.json")
    if manifest.get("terminal_status") != "ACTION_SEQUENCE_READY":
        raise UpstreamPhase3Blocked(
            "CURRENT_UPSTREAM_PHASE3_BLOCKED: Phase-3 terminal_status="
            f"{manifest.get('terminal_status')!r}, result_status="
            f"{result.get('status')!r}"
        )
    final_rel = (manifest.get("artifacts") or {}).get("final_plan")
    if not isinstance(final_rel, str):
        raise ValueError("Phase-3 manifest does not identify a final_plan artifact")
    plan_path = (run_dir / final_rel).resolve()
    if run_dir not in plan_path.parents:
        raise ValueError("Phase-3 final_plan path escapes the run directory")
    plan = _read_json(plan_path)

    if manifest.get("execution_state") != "planning_only":
        raise ValueError("Phase-3 run is not an immutable planning_only handoff")
    if result.get("status") != "ACTION_SEQUENCE_READY":
        raise ValueError("Phase-3 result does not contain a ready action sequence")
    if not grounding.get("complete") or not isinstance(
        grounding.get("assignment"), dict
    ):
        raise ValueError("Phase-3 phi* is absent or incomplete")

    actions = _validate_actions(plan.get("actions"))
    result_actions = _validate_actions(result.get("plan"))
    if actions != result_actions:
        raise ValueError("Persisted final plan differs from result.json plan")
    validation = plan.get("validation")
    replay_rel = (manifest.get("artifacts") or {}).get("replay_validation")
    if validation is None and isinstance(replay_rel, str):
        validation = _read_json(run_dir / replay_rel)
    if isinstance(validation, dict) and validation.get("status") != "VALID":
        raise ValueError("Phase-3 independent replay did not validate the final plan")

    artifacts = {
        "manifest": run_dir / "run_manifest.json",
        "grounding": run_dir / "graph_grounding_result.json",
        "plan": plan_path,
    }
    observed_graph_rel = (manifest.get("artifacts") or {}).get("observed_graph")
    if isinstance(observed_graph_rel, str):
        observed_graph_path = (run_dir / observed_graph_rel).resolve()
        if run_dir not in observed_graph_path.parents:
            raise ValueError("Phase-3 observed graph path escapes run directory")
        if observed_graph_path.is_file():
            artifacts["observed_graph"] = observed_graph_path
    specification_rel = (manifest.get("artifacts") or {}).get(
        "functional_specification"
    )
    if isinstance(specification_rel, str):
        specification_path = (run_dir / specification_rel).resolve()
        if run_dir not in specification_path.parents:
            raise ValueError("Phase-3 functional specification path escapes run directory")
        if specification_path.is_file():
            artifacts["functional_specification"] = specification_path
    return Phase3Handoff(
        run_dir=run_dir,
        domain=str(manifest["domain"]),
        variant=str(manifest["variant"]),
        internal_variant=str(manifest["internal_variant"]),
        source=str(manifest.get("spec_provider_source") or "UNKNOWN"),
        specification_sha256=manifest.get("specification_sha256"),
        assignment=dict(grounding["assignment"]),
        operation_bindings={
            str(key): list(value)
            for key, value in grounding.get("operation_bindings", {}).items()
        },
        inspected_regions=tuple(map(str, result.get("inspected_regions", ()))),
        actions=actions,
        artifacts=artifacts,
        artifact_sha256={key: _sha256(path) for key, path in artifacts.items()},
    )


class Phase4Executor:
    """Execute an authoritative action sequence and stop on first failure."""

    def __init__(self, handoff: Phase3Handoff, adapter: DomainExecutionAdapter):
        self.handoff = handoff
        self.adapter = adapter

    def run(self, *, max_actions: int | None = None) -> dict[str, Any]:
        selected = self.handoff.actions
        if max_actions is not None:
            if max_actions < 1:
                raise ValueError("max_actions must be positive")
            selected = selected[:max_actions]
        started = time.perf_counter()
        inspection_records = []
        for region in self.handoff.inspected_regions:
            try:
                record = self.adapter.execute_inspection_open(region)
            except Exception as error:
                record = {
                    "region": region,
                    "success": False,
                    "failure": ExecutionFailure.CONTROLLER_FAILURE.value,
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            inspection_records.append(record)
            if not record.get("success"):
                break
        inspections_succeeded = (
            len(inspection_records) == len(self.handoff.inspected_regions)
            and all(row.get("success") for row in inspection_records)
        )
        records = []
        if inspections_succeeded:
            for action in selected:
                record = self.adapter.execute_action(action)
                records.append(record.to_dict())
                if not record.success:
                    break
        complete_sequence = len(selected) == len(self.handoff.actions)
        all_succeeded = len(records) == len(selected) and all(
            row["success"] for row in records
        )
        strict_audit = audit_strict_telemetry(inspection_records, records)
        strict_verified = bool(strict_audit["verified"])
        final = (
            self.adapter.final_verification()
            if complete_sequence and all_succeeded and strict_verified
            else {"performed": False, "reason": "PARTIAL_OR_FAILED_SEQUENCE"}
        )
        partial_smoke = not complete_sequence
        partial_smoke_success = bool(
            partial_smoke
            and inspections_succeeded
            and all_succeeded
            and strict_verified
        )
        success = bool(
            complete_sequence
            and inspections_succeeded
            and all_succeeded
            and strict_verified
            and (not complete_sequence or final.get("success"))
        )
        task_failure = next(
            (row["failure"] for row in records if not row["success"]),
            ExecutionFailure.NONE.value,
        )
        if not inspections_succeeded:
            failure_stage = "INSPECTION_OPEN"
            failure = "INSPECTION_EXECUTION_FAILURE"
        elif task_failure != ExecutionFailure.NONE.value:
            failure_stage = (
                "ENTITY_RESOLUTION"
                if task_failure == ExecutionFailure.ENTITY_MAPPING_FAILURE.value
                else "TASK_ACTION"
            )
            failure = task_failure
        elif not strict_verified:
            violation_phases = {
                row["phase"] for row in strict_audit["violations"]
            }
            failure_stage = (
                "INSPECTION_OPEN"
                if violation_phases == {"INSPECTION_OPEN"}
                else "TASK_ACTION"
            )
            failure = "STRICT_EXECUTION_TELEMETRY_VIOLATION"
        elif complete_sequence and not final.get("success"):
            failure_stage = "FINAL_VERIFICATION"
            failure = "FINAL_VERIFICATION_FAILURE"
        else:
            failure_stage = None
            failure = ExecutionFailure.NONE.value
        return {
            "schema_version": 2,
            "phase": "PHASE_4_EXECUTION",
            "domain": self.handoff.domain,
            "variant": self.handoff.variant,
            "internal_variant": self.handoff.internal_variant,
            "functional_specification_source": self.handoff.source,
            "specification_sha256": self.handoff.specification_sha256,
            "phase3_artifacts": {
                key: str(value) for key, value in self.handoff.artifacts.items()
            },
            "phase3_artifact_sha256": dict(self.handoff.artifact_sha256),
            "final_action_sequence": list(self.handoff.actions),
            "entity_resolution": self.adapter.entity_resolution,
            "inspection_execution": {
                "regions": list(self.handoff.inspected_regions),
                "actions_requested": len(self.handoff.inspected_regions),
                "actions_completed": sum(
                    bool(row.get("success")) for row in inspection_records
                ),
                "results": inspection_records,
                "success": inspections_succeeded,
            },
            "task_plan_execution": {
                "actions": list(selected),
                "results": records,
            },
            "actions_requested": len(selected),
            "actions_completed": sum(row["success"] for row in records),
            "full_sequence_requested": complete_sequence,
            "partial_smoke": partial_smoke,
            "partial_smoke_success": partial_smoke_success,
            "action_results": records,
            "final_verification": final,
            "failure": failure,
            "failure_stage": failure_stage,
            "strict_execution": True,
            "strict_telemetry_verification": strict_audit,
            "direct_task_state_fallback_used": not strict_verified,
            "wall_duration_s": time.perf_counter() - started,
            "success": success,
        }
