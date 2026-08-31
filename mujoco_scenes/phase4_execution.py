"""Deterministic Phase-4 execution of immutable Phase-3 plan artifacts.

This module is intentionally downstream of functional grounding and planning.
It consumes the persisted phi* and final symbolic action sequence without
selecting roles, changing bindings, or adding task-level actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Protocol


class ExecutionFailure(str, Enum):
    NONE = "NONE"
    INVALID_HANDOFF = "INVALID_HANDOFF"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    ENTITY_MAPPING_FAILURE = "ENTITY_MAPPING_FAILURE"
    PRECONDITION_STATE_FAILURE = "PRECONDITION_STATE_FAILURE"
    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"
    POSTCONDITION_VERIFICATION_FAILURE = "POSTCONDITION_VERIFICATION_FAILURE"


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


class DomainExecutionAdapter(Protocol):
    entity_resolution: dict[str, Any]

    def execute_action(self, action: dict[str, Any]) -> ActionExecutionResult: ...

    def final_verification(self) -> dict[str, Any]: ...


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase-3 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


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
    final_rel = (manifest.get("artifacts") or {}).get("final_plan")
    if not isinstance(final_rel, str):
        raise ValueError("Phase-3 manifest does not identify a final_plan artifact")
    plan_path = (run_dir / final_rel).resolve()
    if run_dir not in plan_path.parents:
        raise ValueError("Phase-3 final_plan path escapes the run directory")
    plan = _read_json(plan_path)

    if manifest.get("execution_state") != "planning_only":
        raise ValueError("Phase-3 run is not an immutable planning_only handoff")
    if manifest.get("terminal_status") != "ACTION_SEQUENCE_READY":
        raise ValueError(
            "Phase-3 handoff is not executable: "
            f"terminal_status={manifest.get('terminal_status')!r}"
        )
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
        records = []
        for action in selected:
            record = self.adapter.execute_action(action)
            records.append(record.to_dict())
            if not record.success:
                break
        complete_sequence = len(selected) == len(self.handoff.actions)
        all_succeeded = len(records) == len(selected) and all(
            row["success"] for row in records
        )
        final = (
            self.adapter.final_verification()
            if complete_sequence and all_succeeded
            else {"performed": False, "reason": "PARTIAL_OR_FAILED_SEQUENCE"}
        )
        success = bool(all_succeeded and (not complete_sequence or final.get("success")))
        return {
            "schema_version": 1,
            "phase": "PHASE_4_EXECUTION",
            "domain": self.handoff.domain,
            "variant": self.handoff.variant,
            "internal_variant": self.handoff.internal_variant,
            "functional_specification_source": self.handoff.source,
            "specification_sha256": self.handoff.specification_sha256,
            "phase3_artifacts": {
                key: str(value) for key, value in self.handoff.artifacts.items()
            },
            "final_action_sequence": list(self.handoff.actions),
            "entity_resolution": self.adapter.entity_resolution,
            "actions_requested": len(selected),
            "actions_completed": sum(row["success"] for row in records),
            "full_sequence_requested": complete_sequence,
            "action_results": records,
            "final_verification": final,
            "failure": next(
                (row["failure"] for row in records if not row["success"]),
                ExecutionFailure.NONE.value,
            ),
            "wall_duration_s": time.perf_counter() - started,
            "success": success,
        }
