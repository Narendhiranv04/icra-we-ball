import json
from pathlib import Path

import pytest

from mujoco_scenes.phase4_execution import (
    ActionExecutionResult,
    ExecutionFailure,
    Phase4Executor,
    load_phase3_handoff,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _handoff(tmp_path: Path, *, inspected_regions=()):
    action = {
        "action_index": 1,
        "action_instance_id": "fact_001_pick",
        "operator": "PICK",
        "arguments": ["object_0001"],
    }
    _write(
        tmp_path / "run_manifest.json",
        {
            "domain": "kitchen",
            "variant": "K1",
            "internal_variant": "F0_ALL_VISIBLE",
            "spec_provider_source": "GT_FUNCTIONAL_SPEC_ONLY",
            "specification_sha256": "abc",
            "execution_state": "planning_only",
            "terminal_status": "ACTION_SEQUENCE_READY",
            "artifacts": {"final_plan": "action_sequence/action_plan.json"},
        },
    )
    _write(
        tmp_path / "result.json",
        {
            "status": "ACTION_SEQUENCE_READY",
            "plan": [action],
            "inspected_regions": list(inspected_regions),
        },
    )
    _write(
        tmp_path / "graph_grounding_result.json",
        {
            "complete": True,
            "assignment": {"tool": "object_0001"},
            "operation_bindings": {},
        },
    )
    _write(
        tmp_path / "action_sequence/action_plan.json",
        {"actions": [action], "validation": {"status": "VALID"}},
    )
    return load_phase3_handoff(tmp_path)


def test_load_phase3_handoff_preserves_exact_plan_contract(tmp_path):
    handoff = _handoff(tmp_path)
    assert handoff.source == "GT_FUNCTIONAL_SPEC_ONLY"
    assert handoff.assignment == {"tool": "object_0001"}
    assert handoff.actions[0]["arguments"] == ["object_0001"]
    assert set(handoff.artifact_sha256) == {"manifest", "grounding", "plan"}


def test_load_phase3_handoff_rejects_result_plan_mismatch(tmp_path):
    _handoff(tmp_path)
    _write(
        tmp_path / "result.json",
        {"status": "ACTION_SEQUENCE_READY", "plan": []},
    )
    with pytest.raises(ValueError, match="non-empty|differs"):
        load_phase3_handoff(tmp_path)


class _Adapter:
    entity_resolution = {"all_resolved": True}

    def __init__(self, succeed=True, inspection_succeed=True):
        self.succeed = succeed
        self.inspection_succeed = inspection_succeed
        self.calls = []

    def execute_inspection_open(self, region):
        return {
            "region": region,
            "success": self.inspection_succeed,
            "failure": (
                ExecutionFailure.NONE.value
                if self.inspection_succeed
                else ExecutionFailure.CONTROLLER_FAILURE.value
            ),
            "direct_container_state_write_used": False,
        }

    def execute_action(self, action):
        self.calls.append(action)
        return ActionExecutionResult(
            action_index=action["action_index"],
            action_instance_id=action["action_instance_id"],
            operator=action["operator"],
            arguments=action["arguments"],
            success=self.succeed,
            failure=(
                ExecutionFailure.NONE.value
                if self.succeed
                else ExecutionFailure.CONTROLLER_FAILURE.value
            ),
            resolved_arguments=[],
            primitive="test",
            pre_check={"success": True},
            controller_result={"success": self.succeed},
            post_check={"success": self.succeed},
            wall_duration_s=0.0,
        )

    def final_verification(self):
        return {"performed": True, "success": True}


def test_executor_records_partial_smoke_without_claiming_full_verification(tmp_path):
    handoff = _handoff(tmp_path)
    result = Phase4Executor(handoff, _Adapter()).run(max_actions=1)
    assert result["success"]
    assert result["full_sequence_requested"]
    assert result["final_verification"]["performed"]


def test_executor_stops_and_classifies_controller_failure(tmp_path):
    handoff = _handoff(tmp_path)
    result = Phase4Executor(handoff, _Adapter(succeed=False)).run()
    assert not result["success"]
    assert result["actions_completed"] == 0
    assert result["failure"] == ExecutionFailure.CONTROLLER_FAILURE.value


def test_executor_replays_inspection_before_immutable_plan(tmp_path):
    handoff = _handoff(tmp_path, inspected_regions=("D1", "C2"))
    adapter = _Adapter()
    result = Phase4Executor(handoff, adapter).run()
    assert result["inspection_execution"]["regions"] == ["D1", "C2"]
    assert result["inspection_execution"]["actions_completed"] == 2
    assert adapter.calls == list(handoff.actions)
    assert result["strict_execution"] is True
    assert result["direct_task_state_fallback_used"] is False


def test_executor_stops_before_plan_when_inspection_fails(tmp_path):
    handoff = _handoff(tmp_path, inspected_regions=("D1",))
    adapter = _Adapter(inspection_succeed=False)
    result = Phase4Executor(handoff, adapter).run()
    assert not result["success"]
    assert result["failure_stage"] == "INSPECTION_OPEN"
    assert result["failure"] == "INSPECTION_EXECUTION_FAILURE"
    assert adapter.calls == []
