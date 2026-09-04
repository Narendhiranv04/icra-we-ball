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
    "direct_velocity_write",
    "direct_velocity_reset",
    "alignment_fixture_used",
    "staging_fixture_used",
    "installed_target_fixture_used",
    "post_release_dynamics_modified",
    "target_alignment_constraint_used",
    "installed_fastener_constraint_used",
    "payload_pose_snap_used",
    "staging_constraint_used",
})

DIRECT_TASK_STATE_FLAGS = frozenset({
    "direct_task_success_write", "direct_task_success_write_used",
    "direct_task_state_fallback_used",
})
DIRECT_PAYLOAD_STATE_FLAGS = frozenset({
    "direct_payload_pose_write", "direct_object_qpos_write",
    "direct_fastener_qpos_write", "direct_fastener_qpos_write_used",
    "direct_velocity_write", "direct_velocity_reset",
    "initial_payload_qpos_reset_used",
})
ASSISTED_FIXTURE_FLAGS = frozenset({
    "alignment_fixture_used", "staging_fixture_used",
    "installed_target_fixture_used",
    "target_alignment_constraint_used",
    "installed_fastener_constraint_used",
    "staging_constraint_used",
})


class ExecutionFailure(str, Enum):
    NONE = "NONE"
    INVALID_HANDOFF = "INVALID_HANDOFF"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    ENTITY_MAPPING_FAILURE = "ENTITY_MAPPING_FAILURE"
    PRECONDITION_STATE_FAILURE = "PRECONDITION_STATE_FAILURE"
    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"
    POSTCONDITION_VERIFICATION_FAILURE = "POSTCONDITION_VERIFICATION_FAILURE"


PLANNER_FAILURE_CODES = frozenset({
    "REGION_CLOSED", "OBJECT_NOT_VISIBLE", "ACCESS_BLOCKED",
    "TARGET_OCCUPIED", "WRONG_TOOL", "INCOMPATIBLE_TARGET",
    "MOTION_INFEASIBLE", "LOGICAL_PRECONDITION_FAILED", "EXECUTION_ERROR",
})


def classify_planner_failure(
    message: str | None,
    *,
    infrastructure_failure: str,
    operator: str | None = None,
) -> str | None:
    """Map domain diagnostics to a stable planner-facing failure code."""
    if infrastructure_failure == ExecutionFailure.NONE.value:
        return None
    normalized = str(message or "").upper()
    if "CLOSED" in normalized or "NOT OPEN" in normalized:
        return "REGION_CLOSED"
    if any(token in normalized for token in (
        "OBJECT_NOT_VISIBLE", "NOT VISIBLE", "UNRESOLVED",
        "MISSING_OBJECT", "MISSING OBJECT",
    )):
        return "OBJECT_NOT_VISIBLE"
    if "TARGET_OCCUPIED" in normalized or "TARGET OCCUPIED" in normalized:
        return "TARGET_OCCUPIED"
    if (
        "WRONG_TOOL" in normalized
        or ("WRONG" in normalized and "TOOL" in normalized)
    ):
        return "WRONG_TOOL"
    if "INCOMPATIBLE" in normalized:
        return "INCOMPATIBLE_TARGET"
    if any(token in normalized for token in (
        "IK_UNREACHABLE", "MOTION_INFEASIBLE", "NO COLLISION-FREE IK",
        "UNSAFE", "COLLISION", "REACHABILITY",
    )):
        return "MOTION_INFEASIBLE"
    if any(token in normalized for token in (
        "GRASP", "CONTACT", "ACCESS_BLOCKED", "APPROACH_NOT_REACHED",
    )):
        return "ACCESS_BLOCKED"
    if infrastructure_failure == ExecutionFailure.PRECONDITION_STATE_FAILURE.value:
        return "LOGICAL_PRECONDITION_FAILED"
    if infrastructure_failure == ExecutionFailure.ENTITY_MAPPING_FAILURE.value:
        return "OBJECT_NOT_VISIBLE" if operator == "PICK" else "INCOMPATIBLE_TARGET"
    return "EXECUTION_ERROR"


def normalize_planner_failure_code(
    code: str | None,
    message: str | None,
    *,
    infrastructure_failure: str,
    operator: str | None = None,
) -> str | None:
    """Preserve only public codes; derive a stable code for all other input."""
    if code in PLANNER_FAILURE_CODES:
        return code
    diagnostic = message if message else code
    return classify_planner_failure(
        diagnostic,
        infrastructure_failure=infrastructure_failure,
        operator=operator,
    )


class UpstreamPhase3Blocked(ValueError):
    """Raised when Phase 3 explicitly ended without an executable plan."""


class Phase4EntityMappingError(ValueError):
    """Raised before control when a frozen planner ID cannot be resolved."""


class Phase4ViewerClosed(RuntimeError):
    """Raised when a user closes the optional live execution viewer."""


def guard_phase4_live_viewer(callback: Any, *args: Any, **kwargs: Any) -> None:
    """Translate an intentionally closed ffplay window into a safe abort."""
    from .live_mosaic_viewer import LiveMosaicViewerClosed

    try:
        callback(*args, **kwargs)
    except LiveMosaicViewerClosed as error:
        raise Phase4ViewerClosed(str(error)) from error


def emit_phase4_progress(
    phase: str,
    index: int,
    total: int,
    operator: str,
    arguments: list[str],
    *,
    success: bool | None = None,
    controller_status: str | None = None,
    failure_code: str | None = None,
) -> None:
    """Print stable 1-based progress without changing structured results."""
    rendered = f"{operator}({', '.join(arguments)})"
    prefix = f"[{phase} {index:02d}/{total:02d}]"
    if success is None:
        print(f"{prefix} {rendered}", flush=True)
        return
    outcome = "SUCCESS" if success else "FAILED"
    print(f"{prefix} {outcome} {rendered}", flush=True)
    if not success:
        if controller_status:
            print(f"  controller_status={controller_status}", flush=True)
        if failure_code:
            print(f"  failure_code={failure_code}", flush=True)


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
    failure_code: str | None = None

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
    replay_validation_source: str


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
    """Audit every attempted primitive, including failed attempts.

    Contact-gated, zero-snap grasp equalities are intentionally allowed after
    physical bilateral contact. Target-alignment, staging, and installed-state
    fixtures are task assistance and are forbidden.
    """
    violations: list[dict[str, str]] = []

    def scan(value: Any, path: str, phase: str) -> None:
        if isinstance(value, dict):
            if (
                value.get("physical_handle_grasp_constraint_used") is True
                and value.get("handle_grasp_constraint_contact_gated") is not True
            ):
                violations.append({
                    "phase": phase,
                    "path": path,
                    "flag": "ungated_handle_grasp_constraint_used",
                })
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
        scan(row, f"inspection_results[{index}]", "INSPECTION_OPEN")
    for index, row in enumerate(action_results):
        scan(row, f"action_results[{index}]", "TASK_ACTION")
    flags = {row["flag"] for row in violations}
    return {
        "verified": not violations,
        "strict_execution_violation_detected": bool(violations),
        "direct_task_state_write_used": bool(flags & DIRECT_TASK_STATE_FLAGS),
        "direct_task_state_fallback_used": (
            "direct_task_state_fallback_used" in flags
        ),
        "direct_payload_state_write_used": bool(
            flags & DIRECT_PAYLOAD_STATE_FLAGS
        ),
        "assisted_task_fixture_used": bool(flags & ASSISTED_FIXTURE_FLAGS),
        "post_release_dynamics_modified": (
            "post_release_dynamics_modified" in flags
        ),
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


def _validate_replay_evidence(
    validation: Any,
    actions: tuple[dict[str, Any], ...],
) -> None:
    """Fail closed unless evidence is the independent replay of this plan."""
    if not isinstance(validation, dict):
        raise ValueError("Phase-3 independent replay evidence is missing")
    if (
        validation.get("status") != "VALID"
        or validation.get("goal_status") != "GOAL_SATISFIED"
        or validation.get("missing_goals") != []
        or validation.get("validator") != "independent_symbolic_replay_v1"
        or validation.get("uses_planner_transition") is not False
    ):
        raise ValueError("Phase-3 independent replay did not validate the final plan")
    steps = validation.get("steps")
    if not isinstance(steps, list) or len(steps) != len(actions):
        raise ValueError("Phase-3 replay steps do not match the final plan")
    for expected_step, (step, action) in enumerate(zip(steps, actions)):
        if not isinstance(step, dict) or (
            step.get("step") != expected_step
            or step.get("status") != "VALID"
            or step.get("failure") is not None
            or str(step.get("operator", "")).upper() != action["operator"]
            or step.get("arguments") != action["arguments"]
        ):
            raise ValueError(
                "Phase-3 replay step differs from final action "
                f"{action['action_index']}"
            )


def load_phase3_handoff(run_dir: Path) -> Phase3Handoff:
    """Load and fail-closed validate the immutable Phase-3 execution handoff."""
    run_dir = run_dir.resolve()
    manifest_file = run_dir / "run_manifest.json"
    if not manifest_file.is_file():
        raise UpstreamPhase3Blocked(
            f"Required Phase-3 handoff is missing at {run_dir}.\n\n"
            "Generate it deterministically before Phase-4 execution with:\n"
            "  python -m mujoco_scenes.prepare_phase4_handoff --domain <domain> --variant <variant> --mode <mode>"
        )
    manifest = _read_json(manifest_file)
    if manifest.get("terminal_status") != "ACTION_SEQUENCE_READY":
        raise UpstreamPhase3Blocked(
            "CURRENT_UPSTREAM_PHASE3_BLOCKED: Phase-3 terminal_status="
            f"{manifest.get('terminal_status')!r}"
        )
    artifacts_manifest = manifest.get("artifacts") or {}
    result_rel = artifacts_manifest.get("result")
    if not isinstance(result_rel, str):
        raise ValueError("Phase-3 manifest does not identify a result artifact")
    result_path = (run_dir / result_rel).resolve()
    if run_dir not in result_path.parents:
        raise ValueError("Phase-3 result path escapes the run directory")
    result = _read_json(result_path)
    grounding = _read_json(run_dir / "graph_grounding_result.json")
    final_rel = artifacts_manifest.get("final_plan")
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
    replay_rel = artifacts_manifest.get("replay_validation")
    if isinstance(replay_rel, str):
        replay_path = (run_dir / replay_rel).resolve()
        if run_dir not in replay_path.parents:
            raise ValueError("Phase-3 replay-validation path escapes run directory")
        validation = _read_json(replay_path)
        replay_validation_source = "EXPLICIT_REPLAY_ARTIFACT"
    else:
        replay_path = plan_path
        validation = plan.get("validation")
        replay_validation_source = "EMBEDDED_FINAL_PLAN_VALIDATION"
    _validate_replay_evidence(validation, actions)
    audit_rel = (manifest.get("artifacts") or {}).get("plan_grounding_audit")
    if not isinstance(audit_rel, str):
        raise ValueError("Phase-3 manifest has no plan_grounding_audit artifact")
    audit_path = (run_dir / audit_rel).resolve()
    if run_dir not in audit_path.parents:
        raise ValueError("Phase-3 plan-grounding-audit path escapes run directory")
    audit = _read_json(audit_path)
    if (
        audit.get("violations") != []
        or audit.get("plan_replay_valid") is not True
        or audit.get("grounding_complete") is not True
    ):
        raise ValueError("Phase-3 plan-grounding audit is not valid")

    artifacts = {
        "manifest": run_dir / "run_manifest.json",
        "result": result_path,
        "grounding": run_dir / "graph_grounding_result.json",
        "plan": plan_path,
        "replay_validation": replay_path,
        "plan_grounding_audit": audit_path,
    }
    observed_graph_rel = (manifest.get("artifacts") or {}).get("observed_graph")
    if not isinstance(observed_graph_rel, str):
        raise ValueError("Phase-3 manifest has no final observed_graph artifact")
    observed_graph_path = (run_dir / observed_graph_rel).resolve()
    if run_dir not in observed_graph_path.parents:
        raise ValueError("Phase-3 observed graph path escapes run directory")
    _read_json(observed_graph_path)
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
        replay_validation_source=replay_validation_source,
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
        inspection_total = len(self.handoff.inspected_regions)
        for inspection_index, region in enumerate(
            self.handoff.inspected_regions, start=1
        ):
            emit_phase4_progress(
                "INSPECTION", inspection_index, inspection_total,
                "OPEN", [region],
            )
            try:
                record = self.adapter.execute_inspection_open(region)
            except Exception as error:
                record = {
                    "region": region,
                    "success": False,
                    "failure": ExecutionFailure.CONTROLLER_FAILURE.value,
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                    "failure_code": classify_planner_failure(
                        str(error),
                        infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
                        operator="OPEN",
                    ),
                }
            inspection_records.append(record)
            emit_phase4_progress(
                "INSPECTION", inspection_index, inspection_total,
                "OPEN", [region],
                success=bool(record.get("success")),
                controller_status=(
                    (record.get("controller_result") or {}).get("status")
                ),
                failure_code=record.get("failure_code"),
            )
            if not record.get("success"):
                break
        inspections_succeeded = (
            len(inspection_records) == len(self.handoff.inspected_regions)
            and all(row.get("success") for row in inspection_records)
        )
        records = []
        if inspections_succeeded:
            for action in selected:
                emit_phase4_progress(
                    "TASK", int(action["action_index"]),
                    len(self.handoff.actions), str(action["operator"]),
                    list(action["arguments"]),
                )
                try:
                    record = self.adapter.execute_action(action)
                except Exception as error:
                    operator = str(action.get("operator", "UNKNOWN"))
                    message = f"{type(error).__name__}: {error}"
                    record = ActionExecutionResult(
                        action_index=int(action.get("action_index", 0)),
                        action_instance_id=str(
                            action.get("action_instance_id", "unknown_action")
                        ),
                        operator=operator,
                        arguments=list(action.get("arguments", [])),
                        success=False,
                        failure=ExecutionFailure.CONTROLLER_FAILURE.value,
                        resolved_arguments=[],
                        primitive=None,
                        pre_check={"success": False, "performed": False},
                        controller_result={
                            "success": False,
                            "status": "UNEXPECTED_ADAPTER_EXCEPTION",
                            "failure_type": type(error).__name__,
                            "failure_reason": str(error),
                            "message": str(error),
                        },
                        post_check={"success": False, "performed": False},
                        wall_duration_s=0.0,
                        failure_code=normalize_planner_failure_code(
                            None,
                            message,
                            infrastructure_failure=(
                                ExecutionFailure.CONTROLLER_FAILURE.value
                            ),
                            operator=operator,
                        ),
                    )
                records.append(record.to_dict())
                emit_phase4_progress(
                    "TASK", record.action_index, len(self.handoff.actions),
                    record.operator, record.arguments,
                    success=record.success,
                    controller_status=(
                        (record.controller_result or {}).get("status")
                    ),
                    failure_code=record.failure_code,
                )
                if not record.success:
                    break
        complete_sequence = len(selected) == len(self.handoff.actions)
        all_succeeded = len(records) == len(selected) and all(
            row["success"] for row in records
        )
        strict_audit = audit_strict_telemetry(inspection_records, records)
        final = (
            self.adapter.final_verification()
            if complete_sequence and all_succeeded
            else {"performed": False, "reason": "PARTIAL_OR_FAILED_SEQUENCE"}
        )
        partial_smoke = not complete_sequence
        partial_smoke_success = bool(
            partial_smoke
            and inspections_succeeded
            and all_succeeded
        )
        success = bool(
            complete_sequence
            and inspections_succeeded
            and all_succeeded
            and (not complete_sequence or final.get("success"))
        )
        task_failure = next(
            (row["failure"] for row in records if not row["success"]),
            ExecutionFailure.NONE.value,
        )
        task_failure_code = next(
            (row.get("failure_code") for row in records if not row["success"]),
            None,
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
            "phase3_replay_validation_source": (
                self.handoff.replay_validation_source
            ),
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
            "failure_code": (
                next(
                    (row.get("failure_code") for row in inspection_records
                     if not row.get("success")),
                    None,
                )
                if not inspections_succeeded else task_failure_code
            ),
            "failure_stage": failure_stage,
            "execution_mode": "P4_BENCH",
            "strict_execution": False,
            "strict_telemetry_verification": strict_audit,
            "strict_execution_violation_detected": strict_audit[
                "strict_execution_violation_detected"
            ],
            "direct_task_state_write_used": strict_audit[
                "direct_task_state_write_used"
            ],
            "direct_payload_state_write_used": strict_audit[
                "direct_payload_state_write_used"
            ],
            "assisted_task_fixture_used": strict_audit[
                "assisted_task_fixture_used"
            ],
            "post_release_dynamics_modified": strict_audit[
                "post_release_dynamics_modified"
            ],
            "direct_task_state_fallback_used": strict_audit[
                "direct_task_state_fallback_used"
            ],
            "wall_duration_s": time.perf_counter() - started,
            "success": success,
        }


if __name__ == "__main__":
    from .run_phase4_execution import main
    raise SystemExit(main())
