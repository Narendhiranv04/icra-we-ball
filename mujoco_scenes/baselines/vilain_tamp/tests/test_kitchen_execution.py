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
from mujoco_scenes.baselines.vilain_tamp.execution.kitchen import (
    KitchenExecutionAdapter,
    KitchenExecutionContractError,
    KitchenExecutionFailureCode,
    KitchenExecutionInventory,
    build_kitchen_inventory,
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


def _actions() -> tuple[SymbolicAction, ...]:
    rows = (
        ("pick-from", ("source_1", "counter")),
        ("pour", ("source_1", "mug_1", "coffee")),
        ("place-on", ("source_1", "counter")),
        ("pick-from", ("spoon_1", "counter")),
        ("stir", ("spoon_1", "mug_1")),
        ("place-on", ("spoon_1", "counter")),
    )
    return tuple(
        SymbolicAction(
            action_index=index,
            action_instance_id=f"vilain_00_{index + 1:03d}_{operator.replace('-', '_')}",
            operator=operator,
            arguments=arguments,
        )
        for index, (operator, arguments) in enumerate(rows)
    )


def _identities():
    movable = {
        "source_1": _binding("source_1", "coffee_jar_body", "source"),
        "mug_1": _binding("mug_1", "white_mug_body", "vessel"),
        "spoon_1": _binding("spoon_1", "steel_spoon_body", "utensil"),
        "unused_1": _binding("unused_1", "unused_body", "vessel"),
    }
    fixed = {
        "counter": fixed_entity_binding(
            "counter",
            "countertop_body",
            broad_class="surface",
            evidence_artifacts=("scene/fixed_entities.json",),
        )
    }
    return movable, fixed


def _projected_plan():
    movable, fixed = _identities()
    projections = project_plan(
        "kitchen", _actions(), movable, fixed_bindings=fixed
    )
    inventory = build_kitchen_inventory(
        projections, movable, fixed_bindings=fixed
    )
    return projections, inventory


class FakeKitchenController:
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
        self.requests: list[dict[str, Any]] = []
        self.postcondition_requests: list[dict[str, Any]] = []

    def execute_action(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        copied = dict(request)
        copied["arguments"] = list(request["arguments"])
        self.requests.append(copied)
        index = int(request["action_index"])
        if index == self.exception_at:
            raise RuntimeError("synthetic controller crash")
        if index == self.controller_failure_at:
            return {"success": False, "status": "SYNTHETIC_CONTROLLER_FAILURE"}
        return {
            "success": True,
            "status": f"{request['operator']}_MOTION_VERIFIED",
        }

    def verify_postcondition(
        self,
        request: Mapping[str, Any],
        controller_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del controller_result
        copied = dict(request)
        copied["arguments"] = list(request["arguments"])
        self.postcondition_requests.append(copied)
        if int(request["action_index"]) == self.postcondition_failure_at:
            return {"success": False, "reason": "synthetic pose mismatch"}
        return {"success": True, "evidence": "mock physical state"}


def test_direct_pour_stir_place_sequence_and_verified_effect_ledger(
    tmp_path: Path,
) -> None:
    projections, inventory = _projected_plan()
    controller = FakeKitchenController()

    result = KitchenExecutionAdapter(
        controller=controller, inventory=inventory
    ).execute(projections, output_root=tmp_path)

    assert result.success
    assert [
        (request["operator"], request["arguments"])
        for request in controller.requests
    ] == [
        ("PICK", ["coffee_jar_body"]),
        ("POUR", ["coffee_jar_body", "white_mug_body"]),
        ("PLACE", ["coffee_jar_body", "countertop_body"]),
        ("PICK", ["steel_spoon_body"]),
        ("STIR", ["steel_spoon_body", "white_mug_body"]),
        ("PLACE", ["steel_spoon_body", "countertop_body"]),
    ]
    assert len(controller.postcondition_requests) == len(projections)
    assert [effect.effect for effect in result.effect_ledger] == [
        "POUR_COMPLETED",
        "STIR_COMPLETED",
    ]
    assert result.effect_ledger[0].symbolic_arguments == (
        "source_1",
        "mug_1",
        "coffee",
    )
    assert result.effect_ledger[1].symbolic_arguments == ("spoon_1", "mug_1")
    persisted = json.loads(
        (tmp_path / "kitchen_execution.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (tmp_path / "execution_effect_ledger.json").read_text(encoding="utf-8")
    )
    assert persisted["success"] is True
    assert [row["effect"] for row in ledger["effects"]] == [
        "POUR_COMPLETED",
        "STIR_COMPLETED",
    ]


def test_inventory_is_plan_bounded_and_contains_direct_identity_evidence() -> None:
    _, inventory = _projected_plan()
    by_id = inventory.by_object_id()

    assert set(by_id) == {"source_1", "mug_1", "spoon_1", "counter"}
    assert "unused_1" not in by_id
    assert by_id["source_1"].entity_name == "coffee_jar_body"
    assert by_id["source_1"].source_location_id == "counter"
    assert by_id["source_1"].source_location_entity == "countertop_body"
    assert by_id["counter"].fixed
    payload = inventory.controller_payload()
    assert payload["execution_mode"] == "VILAIN_TAMP_BASELINE"
    forbidden_keys = {
        "assignment",
        "operation_bindings",
        "selected_functions",
        "coffee_targets",
        "coffee_tools_by_target",
    }
    assert not forbidden_keys.intersection(
        key for row in payload["objects"] for key in row
    )


@pytest.mark.parametrize(
    ("controller_failure", "postcondition_failure", "expected_code"),
    [
        (1, None, KitchenExecutionFailureCode.CONTROLLER_FAILURE),
        (None, 1, KitchenExecutionFailureCode.POSTCONDITION_FAILURE),
    ],
)
def test_failed_pour_stops_sequence_and_does_not_emit_effect(
    tmp_path: Path,
    controller_failure: int | None,
    postcondition_failure: int | None,
    expected_code: KitchenExecutionFailureCode,
) -> None:
    projections, inventory = _projected_plan()
    controller = FakeKitchenController(
        controller_failure_at=controller_failure,
        postcondition_failure_at=postcondition_failure,
    )

    result = KitchenExecutionAdapter(
        controller=controller, inventory=inventory
    ).execute(projections, output_root=tmp_path)

    assert not result.success
    assert result.terminal_failure_code is expected_code
    assert len(result.actions) == 2
    assert len(controller.requests) == 2
    assert not result.effect_ledger
    assert json.loads(
        (tmp_path / "execution_effect_ledger.json").read_text(encoding="utf-8")
    ) == {"effects": []}


def test_failed_stir_preserves_prior_verified_pour_only() -> None:
    projections, inventory = _projected_plan()
    controller = FakeKitchenController(postcondition_failure_at=4)

    result = KitchenExecutionAdapter(
        controller=controller, inventory=inventory
    ).execute(projections)

    assert not result.success
    assert result.terminal_failure_code is KitchenExecutionFailureCode.POSTCONDITION_FAILURE
    assert [effect.effect for effect in result.effect_ledger] == ["POUR_COMPLETED"]
    assert len(controller.requests) == 5


def test_controller_exception_is_recorded_and_terminates() -> None:
    projections, inventory = _projected_plan()
    controller = FakeKitchenController(exception_at=0)

    result = KitchenExecutionAdapter(
        controller=controller, inventory=inventory
    ).execute(projections)

    assert not result.success
    assert result.terminal_failure_code is KitchenExecutionFailureCode.CONTROLLER_EXCEPTION
    assert "synthetic controller crash" in (result.terminal_failure_message or "")
    assert len(result.actions) == 1
    assert not result.effect_ledger


def test_open_is_dispatched_from_projected_fixed_identity() -> None:
    movable, fixed = _identities()
    fixed["drawer"] = fixed_entity_binding(
        "drawer", "drawer_D1_body", broad_class="storage"
    )
    action = SymbolicAction(0, "vilain_00_001_open", "open-storage", ("drawer",))
    projections = project_plan(
        "kitchen", (action,), movable, fixed_bindings=fixed
    )
    inventory = build_kitchen_inventory(
        projections, movable, fixed_bindings=fixed
    )
    controller = FakeKitchenController()

    result = KitchenExecutionAdapter(
        controller=controller, inventory=inventory
    ).execute(projections)

    assert result.success
    assert controller.requests[0]["operator"] == "OPEN"
    assert controller.requests[0]["arguments"] == ["drawer_D1_body"]
    assert not result.effect_ledger


def test_unresolved_pick_source_is_rejected_without_guessing() -> None:
    movable, _ = _identities()
    action = SymbolicAction(
        0, "vilain_00_001_pick", "pick-from", ("source_1", "counter")
    )
    projections = project_plan("kitchen", (action,), movable)

    with pytest.raises(KitchenExecutionContractError, match="UNRESOLVED_ENTITY"):
        build_kitchen_inventory(projections, movable)


def test_unsupported_or_tampered_projection_is_rejected_before_dispatch() -> None:
    projections, inventory = _projected_plan()
    controller = FakeKitchenController()
    unsupported = ExecutionProjection(
        action_instance_id="vilain_00_999_teleport",
        pddl_operator="teleport",
        pddl_arguments=("source_1",),
        controller_operator="TELEPORT",
        controller_arguments=("coffee_jar_body",),
        resolved_entities=("coffee_jar_body",),
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
        binding_confidence=0.95,
        binding_evidence_artifacts=("identity/source_1.json",),
        skill_parameters={},
    )

    unsupported_result = KitchenExecutionAdapter(
        controller=controller, inventory=inventory
    ).execute((unsupported,))
    assert not unsupported_result.success
    assert (
        unsupported_result.terminal_failure_code
        is KitchenExecutionFailureCode.UNSUPPORTED_CONTROLLER_ACTION
    )
    assert not controller.requests

    tampered = ExecutionProjection(
        **{
            **projections[0].to_dict(),
            "controller_arguments": ("different_body",),
            "resolved_entities": ("different_body",),
        }
    )
    tampered_result = KitchenExecutionAdapter(
        controller=controller, inventory=inventory
    ).execute((tampered,))
    assert not tampered_result.success
    assert (
        tampered_result.terminal_failure_code
        is KitchenExecutionFailureCode.UNRESOLVED_ENTITY
    )
    assert not controller.requests


def test_missing_execution_inventory_entity_fails_without_dispatch() -> None:
    projections, _ = _projected_plan()
    controller = FakeKitchenController()

    result = KitchenExecutionAdapter(
        controller=controller,
        inventory=KitchenExecutionInventory(()),
    ).execute((projections[0],))

    assert not result.success
    assert result.terminal_failure_code is KitchenExecutionFailureCode.UNRESOLVED_ENTITY
    assert len(result.actions) == 1
    assert not controller.requests


def test_external_method_artifacts_and_incomplete_controller_are_rejected() -> None:
    projections, inventory = _projected_plan()
    movable, fixed = _identities()
    with pytest.raises(KitchenExecutionContractError, match="external method"):
        build_kitchen_inventory(
            projections,
            movable,
            fixed_bindings=fixed,
            external_method_artifacts={"opaque": "forbidden"},
        )
    with pytest.raises(KitchenExecutionContractError, match="external method"):
        KitchenExecutionAdapter(
            controller=FakeKitchenController(), inventory=inventory
        ).execute(projections, external_method_artifacts={"opaque": "forbidden"})

    class IncompleteController:
        def execute_action(self, request):
            del request
            return {"success": True}

    with pytest.raises(KitchenExecutionContractError, match="verify_postcondition"):
        KitchenExecutionAdapter(
            controller=IncompleteController(),  # type: ignore[arg-type]
            inventory=inventory,
        )
