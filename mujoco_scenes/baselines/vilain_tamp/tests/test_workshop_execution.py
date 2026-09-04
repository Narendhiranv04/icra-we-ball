from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    ExecutionProjection,
    SymbolicAction,
)
from mujoco_scenes.baselines.vilain_tamp.execution.base import project_plan
from mujoco_scenes.baselines.vilain_tamp.execution.workshop import (
    WorkshopControllerContract,
    WorkshopExecutionAdapter,
    WorkshopExecutionContractError,
    WorkshopExecutionFailureCode,
    build_workshop_controller_contract,
)
from mujoco_scenes.baselines.vilain_tamp.identity import (
    EntityBinding,
    fixed_entity_binding,
)


def _binding(object_id: str, entity_name: str, pddl_type: str) -> EntityBinding:
    return EntityBinding(
        object_id=object_id,
        entity_name=entity_name,
        pddl_type=pddl_type,
        broad_class=pddl_type,
        centroid_distance_m=0.01,
        aabb_distance_m=0.0,
        observed_centroid_m=(0.0, 0.0, 0.0),
        entity_centroid_m=(0.01, 0.0, 0.0),
        entity_aabb_min_m=(-0.02, -0.02, -0.02),
        entity_aabb_max_m=(0.02, 0.02, 0.02),
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
        confidence=0.95,
        observation_stage_ids=("000_initial",),
        evidence_artifacts=(f"identity/{object_id}.json",),
    )


def _plan() -> tuple[ExecutionProjection, ...]:
    movable = {
        "driver_1": _binding("driver_1", "long_driver_body", "driver"),
        "fastener_1": _binding("fastener_1", "medium_screw_body", "fastener"),
    }
    fixed = {
        "driver_cabinet": fixed_entity_binding(
            "driver_cabinet", "tool_cabinet", broad_class="storage"
        ),
        "parts_drawer": fixed_entity_binding(
            "parts_drawer", "left_drawer", broad_class="storage"
        ),
        "frame_joint": fixed_entity_binding(
            "frame_joint", "workshop_frame_joint", broad_class="target"
        ),
        "main_bench": fixed_entity_binding(
            "main_bench", "main_workbench", broad_class="surface"
        ),
    }
    rows = (
        ("open-storage", ("parts_drawer",)),
        ("open-storage", ("driver_cabinet",)),
        ("pick-from", ("fastener_1", "parts_drawer")),
        ("insert", ("fastener_1", "frame_joint")),
        ("pick-from", ("driver_1", "driver_cabinet")),
        ("drive", ("driver_1", "fastener_1", "frame_joint")),
        ("place-on", ("driver_1", "main_bench")),
    )
    actions = tuple(
        SymbolicAction(
            action_index=index,
            action_instance_id=f"vilain_00_{index + 1:03d}_{operator.replace('-', '_')}",
            operator=operator,
            arguments=arguments,
        )
        for index, (operator, arguments) in enumerate(rows)
    )
    return project_plan("workshop", actions, movable, fixed_bindings=fixed)


class FakeWorkshopDispatcher:
    def __init__(
        self,
        *,
        controller_failure_at: int | None = None,
        postcondition_failure_at: int | None = None,
        exception_at: int | None = None,
    ) -> None:
        self.controller_failure_at = controller_failure_at
        self.postcondition_failure_at = postcondition_failure_at
        self.exception_at = exception_at
        self.calls: list[dict[str, Any]] = []
        self.postcondition_calls: list[dict[str, Any]] = []
        self.contracts: list[WorkshopControllerContract] = []

    def execute_action(
        self,
        request: Mapping[str, Any],
        controller_contract: WorkshopControllerContract,
    ):
        copied = dict(request)
        copied["arguments"] = list(request["arguments"])
        self.calls.append(copied)
        self.contracts.append(controller_contract)
        index = int(request["action_index"])
        if index == self.exception_at:
            raise RuntimeError("synthetic workshop controller crash")
        if index == self.controller_failure_at:
            return {"success": False, "status": "SYNTHETIC_CONTROLLER_FAILURE"}
        return {
            "success": True,
            "status": f"{request['operator']}_PHYSICAL_MOTION_VERIFIED",
        }

    def verify_postcondition(
        self,
        request: Mapping[str, Any],
        controller_result: Mapping[str, Any],
        controller_contract: WorkshopControllerContract,
    ):
        del controller_result
        assert controller_contract is self.contracts[-1]
        copied = dict(request)
        copied["arguments"] = list(request["arguments"])
        self.postcondition_calls.append(copied)
        index = int(request["action_index"])
        if index == self.postcondition_failure_at:
            return {"success": False, "reason": "synthetic physical mismatch"}
        operator = str(request["operator"])
        if operator == "OPEN":
            return {"success": True, "articulation_verified": True}
        if operator == "PICK":
            return {"success": True, "held_verified": True}
        if operator == "SCREW":
            return {"success": True, "joint_repaired_state": True}
        if request["pddl_operator"] == "insert":
            return {"success": True, "insertion_verified": True}
        return {"success": True, "surface_place_verified": True}


def _adapter(dispatcher=None):
    projections = _plan()
    contract = build_workshop_controller_contract(projections)
    return (
        WorkshopExecutionAdapter(
            dispatcher=dispatcher or FakeWorkshopDispatcher(),
            controller_contract=contract,
        ),
        projections,
        contract,
    )


def test_mock_plan_drives_exact_plan_specified_tuple_and_emits_effect(
    tmp_path: Path,
) -> None:
    dispatcher = FakeWorkshopDispatcher()
    adapter, projections, contract = _adapter(dispatcher)

    result = adapter.execute(projections, output_root=tmp_path)

    assert result.success
    assert contract.driver_id == "driver_1"
    assert contract.fastener_id == "fastener_1"
    assert contract.target_id == "frame_joint"
    assert contract.work_surface_id == "main_bench"
    assert (contract.driver, contract.fastener, contract.target_joint) == (
        "long_driver_body",
        "medium_screw_body",
        "workshop_frame_joint",
    )
    drive_request = dispatcher.calls[5]
    assert drive_request["operator"] == "SCREW"
    assert drive_request["arguments"] == [
        "long_driver_body",
        "medium_screw_body",
        "workshop_frame_joint",
    ]
    assert dispatcher.calls[3]["arguments"] == [
        "medium_screw_body",
        "workshop_frame_joint",
    ]
    assert dispatcher.calls[6]["arguments"] == [
        "long_driver_body",
        "main_workbench",
    ]
    assert all(item is contract for item in dispatcher.contracts)
    assert len(result.effect_ledger) == 1
    effect = result.effect_ledger[0]
    assert effect.effect == "DRIVE_COMPLETED"
    assert effect.symbolic_arguments == (
        "driver_1",
        "fastener_1",
        "frame_joint",
    )
    assert effect.resolved_entities == (
        "long_driver_body",
        "medium_screw_body",
        "workshop_frame_joint",
    )
    persisted = json.loads(
        (tmp_path / "workshop_execution.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (tmp_path / "workshop_execution_effect_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["success"] is True
    assert [row["effect"] for row in ledger["effects"]] == ["DRIVE_COMPLETED"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "workshop_execution.json",
        "workshop_execution_effect_ledger.json",
    ]


def test_controller_contract_contains_only_action_derived_fields() -> None:
    contract = build_workshop_controller_contract(_plan())
    payload = contract.to_dict()

    assert payload == {
        "driver_id": "driver_1",
        "fastener_id": "fastener_1",
        "target_id": "frame_joint",
        "work_surface_id": "main_bench",
        "driver_entity": "long_driver_body",
        "fastener_entity": "medium_screw_body",
        "target_entity": "workshop_frame_joint",
        "work_surface_entity": "main_workbench",
        "drive_action_instance_id": "vilain_00_006_drive",
    }
    assert not any("assignment" in key for key in payload)
    assert not any("role" in key for key in payload)


@pytest.mark.parametrize(
    ("controller_failure", "postcondition_failure", "expected_code"),
    [
        (5, None, WorkshopExecutionFailureCode.CONTROLLER_FAILURE),
        (None, 5, WorkshopExecutionFailureCode.POSTCONDITION_FAILURE),
    ],
)
def test_failed_drive_never_emits_drive_effect(
    tmp_path: Path,
    controller_failure: int | None,
    postcondition_failure: int | None,
    expected_code: WorkshopExecutionFailureCode,
) -> None:
    dispatcher = FakeWorkshopDispatcher(
        controller_failure_at=controller_failure,
        postcondition_failure_at=postcondition_failure,
    )
    adapter, projections, _ = _adapter(dispatcher)

    result = adapter.execute(projections, output_root=tmp_path)

    assert not result.success
    assert result.terminal_failure_code is expected_code
    assert len(result.actions) == 6
    assert not result.effect_ledger
    assert len(dispatcher.calls) == 6
    assert json.loads(
        (tmp_path / "workshop_execution_effect_ledger.json").read_text(
            encoding="utf-8"
        )
    ) == {"effects": []}


def test_failure_after_verified_drive_preserves_drive_effect() -> None:
    dispatcher = FakeWorkshopDispatcher(controller_failure_at=6)
    adapter, projections, _ = _adapter(dispatcher)

    result = adapter.execute(projections)

    assert not result.success
    assert result.terminal_failure_code is WorkshopExecutionFailureCode.CONTROLLER_FAILURE
    assert [effect.effect for effect in result.effect_ledger] == ["DRIVE_COMPLETED"]
    assert len(result.actions) == 7


def test_controller_exception_is_recorded_and_stops_first_failure() -> None:
    dispatcher = FakeWorkshopDispatcher(exception_at=2)
    adapter, projections, _ = _adapter(dispatcher)

    result = adapter.execute(projections)

    assert not result.success
    assert result.terminal_failure_code is WorkshopExecutionFailureCode.CONTROLLER_EXCEPTION
    assert "synthetic workshop controller crash" in (
        result.terminal_failure_message or ""
    )
    assert len(result.actions) == 3
    assert len(dispatcher.calls) == 3
    assert not result.effect_ledger


def test_contract_rejects_missing_or_mismatched_drive_dependencies() -> None:
    projections = _plan()
    with pytest.raises(WorkshopExecutionContractError, match="exactly one DRIVE"):
        build_workshop_controller_contract(
            tuple(item for item in projections if item.pddl_operator != "drive")
        )

    mismatched_insert = ExecutionProjection(
        **{
            **projections[3].to_dict(),
            "pddl_arguments": ("other_fastener", "frame_joint"),
            "controller_arguments": ("other_body", "workshop_frame_joint"),
            "resolved_entities": ("other_body", "workshop_frame_joint"),
        }
    )
    with pytest.raises(WorkshopExecutionContractError, match="matching INSERT"):
        build_workshop_controller_contract(
            (*projections[:3], mismatched_insert, *projections[4:])
        )


def test_place_and_drive_preconditions_use_exact_action_tuple() -> None:
    adapter, projections, _ = _adapter()

    place_first = adapter.execute((projections[-1],))
    drive_without_insert = adapter.execute(
        (projections[4], projections[5])
    )

    assert (
        place_first.terminal_failure_code
        is WorkshopExecutionFailureCode.PRECONDITION_FAILURE
    )
    assert (
        drive_without_insert.terminal_failure_code
        is WorkshopExecutionFailureCode.PRECONDITION_FAILURE
    )
    assert "fastener" in (drive_without_insert.terminal_failure_message or "")


def test_unsupported_and_unresolved_actions_fail_without_dispatch() -> None:
    dispatcher = FakeWorkshopDispatcher()
    adapter, projections, _ = _adapter(dispatcher)
    unsupported = ExecutionProjection(
        action_instance_id="vilain_00_999_weld",
        pddl_operator="weld",
        pddl_arguments=("fastener_1", "frame_joint"),
        controller_operator="WELD",
        controller_arguments=("medium_screw_body", "workshop_frame_joint"),
        resolved_entities=("medium_screw_body", "workshop_frame_joint"),
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB+PUBLIC_FIXED_ENTITY",
        binding_confidence=0.95,
        binding_evidence_artifacts=(),
        skill_parameters={},
    )
    unresolved = ExecutionProjection(
        **{
            **projections[0].to_dict(),
            "controller_arguments": ("",),
            "resolved_entities": ("",),
        }
    )

    unsupported_result = adapter.execute((unsupported,))
    unresolved_result = adapter.execute((unresolved,))

    assert (
        unsupported_result.terminal_failure_code
        is WorkshopExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION
    )
    assert (
        unresolved_result.terminal_failure_code
        is WorkshopExecutionFailureCode.UNRESOLVED_ENTITY
    )
    assert not dispatcher.calls


def test_external_artifacts_and_incomplete_dispatcher_are_rejected() -> None:
    projections = _plan()
    with pytest.raises(WorkshopExecutionContractError, match="external method"):
        build_workshop_controller_contract(
            projections, external_method_artifacts={"opaque": "forbidden"}
        )

    adapter, _, _ = _adapter()
    with pytest.raises(WorkshopExecutionContractError, match="external method"):
        adapter.execute(
            projections, external_method_artifacts={"opaque": "forbidden"}
        )

    class IncompleteDispatcher:
        def execute_action(self, request, controller_contract):
            del request, controller_contract
            return {"success": True}

    with pytest.raises(WorkshopExecutionContractError, match="verify_postcondition"):
        WorkshopExecutionAdapter(
            dispatcher=IncompleteDispatcher(),  # type: ignore[arg-type]
            controller_contract=build_workshop_controller_contract(projections),
        )
