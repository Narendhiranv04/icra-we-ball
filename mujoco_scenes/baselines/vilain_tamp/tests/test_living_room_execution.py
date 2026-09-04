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
from mujoco_scenes.baselines.vilain_tamp.execution.living_room import (
    LivingRoomExecutionAdapter,
    LivingRoomExecutionContractError,
    LivingRoomExecutionFailureCode,
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
        "cup_1": _binding("cup_1", "left_cup_body", "payload"),
        "remote_1": _binding("remote_1", "remote_body", "payload"),
    }
    fixed = {
        "staging": fixed_entity_binding(
            "staging", "staging_surface", broad_class="location"
        ),
        "left_table": fixed_entity_binding(
            "left_table", "left_table_top", broad_class="support"
        ),
        "coffee_table": fixed_entity_binding(
            "coffee_table", "coffee_table_top", broad_class="support"
        ),
    }
    actions = (
        SymbolicAction(0, "vilain_00_001_pick", "pick-from", ("cup_1", "staging")),
        SymbolicAction(1, "vilain_00_002_place", "place-on", ("cup_1", "left_table")),
        SymbolicAction(2, "vilain_00_003_pick", "pick-from", ("remote_1", "staging")),
        SymbolicAction(3, "vilain_00_004_place", "place-on", ("remote_1", "coffee_table")),
    )
    return project_plan(
        "living_room", actions, movable, fixed_bindings=fixed
    )


class FakeMobileExecutor:
    def __init__(self, *, fail_at: int | None = None, raise_at: int | None = None):
        self.fail_at = fail_at
        self.raise_at = raise_at
        self.calls: list[tuple[str, str | None]] = []

    def move_to(self, target_entity: str, *, carrying_entity: str | None):
        index = len(self.calls)
        self.calls.append((target_entity, carrying_entity))
        if index == self.raise_at:
            raise RuntimeError("synthetic mobile crash")
        if index == self.fail_at:
            return {"success": False, "status": "NO_COLLISION_FREE_PATH"}
        return {"success": True, "path": ["start", target_entity]}


class FakePickExecutor:
    def __init__(
        self,
        *,
        fail_at: int | None = None,
        held_failure_at: int | None = None,
        raise_at: int | None = None,
    ) -> None:
        self.fail_at = fail_at
        self.held_failure_at = held_failure_at
        self.raise_at = raise_at
        self.pick_calls: list[str] = []
        self.verify_calls: list[str] = []

    def pick(self, payload_entity: str):
        index = len(self.pick_calls)
        self.pick_calls.append(payload_entity)
        if index == self.raise_at:
            raise RuntimeError("synthetic pick crash")
        if index == self.fail_at:
            return {"success": False, "status": "GRASP_FAILED"}
        return {"success": True, "status": "PICK_COMPLETED"}

    def verify_held(self, payload_entity: str):
        index = len(self.verify_calls)
        self.verify_calls.append(payload_entity)
        if index == self.held_failure_at:
            return {"validation_status": "FALSE", "reason": "GRASP_WELD_INACTIVE"}
        return {"validation_status": "TRUE"}


class FakePlaceExecutor:
    def __init__(
        self,
        *,
        place_failure_at: int | None = None,
        on_failure_at: int | None = None,
        empty_destination_at: int | None = None,
        raise_component: str | None = None,
    ) -> None:
        self.place_failure_at = place_failure_at
        self.on_failure_at = on_failure_at
        self.empty_destination_at = empty_destination_at
        self.raise_component = raise_component
        self.destination_calls: list[dict[str, str]] = []
        self.place_calls: list[tuple[str, str, Mapping[str, Any]]] = []
        self.verify_calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def destination_for(
        self,
        *,
        payload_id: str,
        payload_entity: str,
        support_id: str,
        support_entity: str,
    ):
        index = len(self.destination_calls)
        self.destination_calls.append(
            {
                "payload_id": payload_id,
                "payload_entity": payload_entity,
                "support_id": support_id,
                "support_entity": support_entity,
            }
        )
        if self.raise_component == "destination":
            raise RuntimeError("synthetic destination failure")
        if index == self.empty_destination_at:
            return {}
        return {
            "payload_id": payload_id,
            "support_id": support_id,
            "support_entity": support_entity,
            "position_world_m": [float(index), 0.1, 0.6],
        }

    def place(
        self,
        payload_entity: str,
        support_entity: str,
        destination: Mapping[str, Any],
    ):
        index = len(self.place_calls)
        self.place_calls.append((payload_entity, support_entity, dict(destination)))
        if self.raise_component == "place":
            raise RuntimeError("synthetic place crash")
        if index == self.place_failure_at:
            return {"success": False, "status": "RELEASE_FAILED"}
        return {"success": True, "status": "RELEASED"}

    def verify_physical_on(
        self,
        payload_entity: str,
        support_entity: str,
        destination: Mapping[str, Any],
        place_result: Mapping[str, Any],
    ):
        del place_result
        index = len(self.verify_calls)
        self.verify_calls.append((payload_entity, support_entity, dict(destination)))
        if self.raise_component == "verify":
            raise RuntimeError("synthetic ON verifier crash")
        if index == self.on_failure_at:
            return {
                "relation": "ON",
                "verified": False,
                "reason": "PAYLOAD_ON_FLOOR",
            }
        return {
            "relation": "ON",
            "verified": True,
            "released": True,
            "stable": True,
            "inside_support_footprint": True,
            "floor_contact": False,
            "invalid_penetration": False,
        }


def _adapter(*, mobile=None, picker=None, placer=None):
    return LivingRoomExecutionAdapter(
        mobile=mobile or FakeMobileExecutor(),
        picker=picker or FakePickExecutor(),
        placer=placer or FakePlaceExecutor(),
    )


def test_synthetic_plan_calls_only_projected_payload_and_support_ids(
    tmp_path: Path,
) -> None:
    mobile = FakeMobileExecutor()
    picker = FakePickExecutor()
    placer = FakePlaceExecutor()

    result = _adapter(mobile=mobile, picker=picker, placer=placer).execute(
        _plan(), output_root=tmp_path
    )

    assert result.success
    assert mobile.calls == [
        ("left_cup_body", None),
        ("left_table_top", "left_cup_body"),
        ("remote_body", None),
        ("coffee_table_top", "remote_body"),
    ]
    assert picker.pick_calls == ["left_cup_body", "remote_body"]
    assert picker.verify_calls == ["left_cup_body", "remote_body"]
    assert [call[:2] for call in placer.place_calls] == [
        ("left_cup_body", "left_table_top"),
        ("remote_body", "coffee_table_top"),
    ]
    assert placer.destination_calls == [
        {
            "payload_id": "cup_1",
            "payload_entity": "left_cup_body",
            "support_id": "left_table",
            "support_entity": "left_table_top",
        },
        {
            "payload_id": "remote_1",
            "payload_entity": "remote_body",
            "support_id": "coffee_table",
            "support_entity": "coffee_table_top",
        },
    ]
    assert all(
        action.postcondition_result.get("validation_status") == "TRUE"
        or action.postcondition_result.get("verified") is True
        for action in result.actions
    )
    persisted = json.loads(
        (tmp_path / "living_room_execution.json").read_text(encoding="utf-8")
    )
    assert persisted["success"] is True
    assert [path.name for path in tmp_path.iterdir()] == [
        "living_room_execution.json"
    ]


def test_place_requires_independently_verified_physical_on() -> None:
    placer = FakePlaceExecutor(on_failure_at=0)

    result = _adapter(placer=placer).execute(_plan())

    assert not result.success
    assert result.terminal_failure_code is LivingRoomExecutionFailureCode.POSTCONDITION_FAILURE
    assert len(result.actions) == 2
    assert result.actions[-1].motion_result["success"] is True
    assert result.actions[-1].postcondition_result["relation"] == "ON"
    assert result.actions[-1].postcondition_result["verified"] is False
    assert len(placer.place_calls) == 1
    assert len(placer.verify_calls) == 1


@pytest.mark.parametrize(
    ("mobile", "picker", "placer", "expected_code", "completed"),
    [
        (
            FakeMobileExecutor(fail_at=0),
            FakePickExecutor(),
            FakePlaceExecutor(),
            LivingRoomExecutionFailureCode.MOBILE_FAILURE,
            1,
        ),
        (
            FakeMobileExecutor(),
            FakePickExecutor(fail_at=0),
            FakePlaceExecutor(),
            LivingRoomExecutionFailureCode.PICK_FAILURE,
            1,
        ),
        (
            FakeMobileExecutor(),
            FakePickExecutor(held_failure_at=0),
            FakePlaceExecutor(),
            LivingRoomExecutionFailureCode.POSTCONDITION_FAILURE,
            1,
        ),
        (
            FakeMobileExecutor(),
            FakePickExecutor(),
            FakePlaceExecutor(place_failure_at=0),
            LivingRoomExecutionFailureCode.PLACE_FAILURE,
            2,
        ),
    ],
)
def test_motion_failure_stops_at_first_failed_action(
    mobile,
    picker,
    placer,
    expected_code: LivingRoomExecutionFailureCode,
    completed: int,
) -> None:
    result = _adapter(mobile=mobile, picker=picker, placer=placer).execute(_plan())

    assert not result.success
    assert result.terminal_failure_code is expected_code
    assert len(result.actions) == completed
    assert not result.actions[-1].success


@pytest.mark.parametrize("component", ["destination", "place", "verify"])
def test_place_component_exception_is_recorded_and_stops(component: str) -> None:
    placer = FakePlaceExecutor(raise_component=component)

    result = _adapter(placer=placer).execute(_plan())

    assert not result.success
    assert result.terminal_failure_code is LivingRoomExecutionFailureCode.EXECUTOR_EXCEPTION
    expected_label = "physical ON verification" if component == "verify" else component
    assert expected_label in (result.terminal_failure_message or "")
    assert len(result.actions) == 2


@pytest.mark.parametrize(
    ("mobile", "picker", "expected_component"),
    [
        (FakeMobileExecutor(raise_at=0), FakePickExecutor(), "mobile"),
        (FakeMobileExecutor(), FakePickExecutor(raise_at=0), "pick"),
    ],
)
def test_pick_path_executor_exception_is_recorded(
    mobile, picker, expected_component: str
) -> None:
    result = _adapter(mobile=mobile, picker=picker).execute(_plan())

    assert not result.success
    assert result.terminal_failure_code is LivingRoomExecutionFailureCode.EXECUTOR_EXCEPTION
    assert expected_component in (result.terminal_failure_message or "")
    assert len(result.actions) == 1


def test_empty_concrete_destination_fails_before_mobile_place_motion() -> None:
    mobile = FakeMobileExecutor()
    placer = FakePlaceExecutor(empty_destination_at=0)

    result = _adapter(mobile=mobile, placer=placer).execute(_plan())

    assert not result.success
    assert result.terminal_failure_code is LivingRoomExecutionFailureCode.PLACE_FAILURE
    assert "destination is empty" in (result.terminal_failure_message or "")
    assert mobile.calls == [("left_cup_body", None)]
    assert not placer.place_calls


def test_place_without_matching_held_payload_fails_without_motion() -> None:
    mobile = FakeMobileExecutor()
    picker = FakePickExecutor()
    placer = FakePlaceExecutor()

    result = _adapter(mobile=mobile, picker=picker, placer=placer).execute(
        (_plan()[1],)
    )

    assert not result.success
    assert result.terminal_failure_code is LivingRoomExecutionFailureCode.PLACE_FAILURE
    assert not mobile.calls
    assert not picker.pick_calls
    assert not placer.destination_calls


def test_second_pick_while_holding_fails_without_dispatching_second_pick() -> None:
    projections = _plan()
    picker = FakePickExecutor()

    result = _adapter(picker=picker).execute((projections[0], projections[2]))

    assert not result.success
    assert result.terminal_failure_code is LivingRoomExecutionFailureCode.PICK_FAILURE
    assert picker.pick_calls == ["left_cup_body"]


def test_unsupported_and_unresolved_projections_are_recorded_not_dispatched() -> None:
    unsupported = ExecutionProjection(
        action_instance_id="vilain_00_999_move",
        pddl_operator="move",
        pddl_arguments=("staging", "left_table"),
        controller_operator="MOVE",
        controller_arguments=("left_table_top",),
        resolved_entities=("left_table_top",),
        binding_method="PUBLIC_FIXED_ENTITY",
        binding_confidence=1.0,
        binding_evidence_artifacts=(),
        skill_parameters={},
    )
    unresolved = ExecutionProjection(
        action_instance_id="vilain_00_998_place",
        pddl_operator="place-on",
        pddl_arguments=("cup_1", "left_table"),
        controller_operator="PLACE",
        controller_arguments=("left_cup_body", ""),
        resolved_entities=("left_cup_body", ""),
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
        binding_confidence=0.0,
        binding_evidence_artifacts=(),
        skill_parameters={},
    )

    unsupported_result = _adapter().execute((unsupported,))
    unresolved_result = _adapter().execute((unresolved,))

    assert (
        unsupported_result.terminal_failure_code
        is LivingRoomExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION
    )
    assert (
        unresolved_result.terminal_failure_code
        is LivingRoomExecutionFailureCode.UNRESOLVED_ENTITY
    )


def test_external_artifacts_and_incomplete_motion_executor_are_rejected() -> None:
    with pytest.raises(LivingRoomExecutionContractError, match="external method"):
        _adapter().execute(
            _plan(), external_method_artifacts={"opaque": "forbidden"}
        )

    class IncompleteMobile:
        pass

    with pytest.raises(LivingRoomExecutionContractError, match="move_to"):
        LivingRoomExecutionAdapter(
            mobile=IncompleteMobile(),  # type: ignore[arg-type]
            picker=FakePickExecutor(),
            placer=FakePlaceExecutor(),
        )
