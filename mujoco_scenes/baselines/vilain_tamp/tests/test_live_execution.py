from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    BaselineExecutionPlan,
    ExecutionProjection,
    SymbolicAction,
    SymbolicPlan,
)
from mujoco_scenes.baselines.vilain_tamp.evaluation import TerminalStateSnapshot
from mujoco_scenes.baselines.vilain_tamp.execution.base import project_plan
from mujoco_scenes.baselines.vilain_tamp.execution.kitchen import (
    KitchenExecutionAdapter,
    build_kitchen_inventory,
)
from mujoco_scenes.baselines.vilain_tamp.execution.workshop import (
    WorkshopExecutionAdapter,
    build_workshop_controller_contract,
)
from mujoco_scenes.baselines.vilain_tamp.identity import EntityBinding
from mujoco_scenes.baselines.vilain_tamp.live_execution import (
    KitchenLiveControllerFacade,
    LiveDomainRuntime,
    LiveExecutionError,
    LiveExecutionStage,
    LivingRoomLiveControllerFacade,
    MuJoCoPhysicalStateObserver,
    WorkshopLiveControllerFacade,
)


@dataclass
class FakePhysicalState:
    postcondition_success: bool = True

    def __post_init__(self) -> None:
        self.recorded: list[tuple[str, tuple[str, ...]]] = []

    def capture_object(self, entity: str) -> Mapping[str, Any]:
        return {"entity": entity, "world_position_m": [0.0, 0.0, 0.0]}

    def verify_open(self, entity: str) -> Mapping[str, Any]:
        return {"success": self.postcondition_success, "entity": entity, "open": self.postcondition_success}

    def verify_pick(self, entity: str, before: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"success": self.postcondition_success, "entity": entity, "held": self.postcondition_success, "before": dict(before)}

    def verify_place(self, entity: str, target: str, *, contained: bool) -> Mapping[str, Any]:
        return {"success": self.postcondition_success, "entity": entity, "target": target, "contained": contained}

    def verify_skill(self, operator: str, arguments: Sequence[str], controller_result: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"success": self.postcondition_success, "operator": operator, "physical": True}

    def record_action(self, operator: str, arguments: Sequence[str], controller_result: Mapping[str, Any]) -> None:
        self.recorded.append((operator, tuple(arguments)))

    def snapshot(self, domain: str, predicted_infeasible: bool) -> TerminalStateSnapshot:
        return TerminalStateSnapshot(
            domain=domain,
            predicted_infeasible=predicted_infeasible,
            objects={"payload": {"present": True, "held": False}},
            relations={"contained_in": {}, "articulation": {}},
            held_objects=(),
            measurements={"source": "physical"},
        )


class RecordingPrimitives:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*arguments: Any, **keywords: Any) -> Mapping[str, Any]:
            self.calls.append((name, (*arguments, *(f"{key}={value}" for key, value in keywords.items()))))
            return {
                "success": self.success,
                "status": "OK" if self.success else "FAILED",
                "pour_motion_verified": self.success,
                "stir_motion_verified": self.success,
                "drive_motion_verified": self.success,
                "joint_repaired": self.success,
            }

        return call


def binding(object_id: str, entity: str, pddl_type: str = "movable") -> EntityBinding:
    return EntityBinding(
        object_id=object_id,
        entity_name=entity,
        pddl_type=pddl_type,
        broad_class=pddl_type,
        centroid_distance_m=0.0,
        aabb_distance_m=0.0,
        observed_centroid_m=(0.0, 0.0, 0.0),
        entity_centroid_m=(0.0, 0.0, 0.0),
        entity_aabb_min_m=(-0.1, -0.1, -0.1),
        entity_aabb_max_m=(0.1, 0.1, 0.1),
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
        confidence=1.0,
        observation_stage_ids=("initial",),
        evidence_artifacts=("observations/initial.json",),
    )


@pytest.mark.parametrize(
    ("operator", "arguments", "method"),
    [
        ("OPEN", ("cabinet",), "open"),
        ("PICK", ("cup",), "pick"),
        ("PLACE", ("cup", "table"), "place"),
        ("POUR", ("jar", "cup"), "pour"),
        ("STIR", ("spoon", "cup"), "stir"),
    ],
)
def test_kitchen_facade_maps_every_controller_operator(operator, arguments, method) -> None:
    primitives = RecordingPrimitives()
    facade = KitchenLiveControllerFacade(primitives, FakePhysicalState())
    result = facade.execute_action(
        {"action_instance_id": "a0", "operator": operator, "arguments": arguments}
    )
    assert result["success"] is True
    assert primitives.calls == [(method, arguments)]


def test_kitchen_facade_fails_closed_for_unknown_operator() -> None:
    primitives = RecordingPrimitives()
    result = KitchenLiveControllerFacade(primitives, FakePhysicalState()).execute_action(
        {"action_instance_id": "a0", "operator": "FLY", "arguments": ("cup",)}
    )
    assert result["success"] is False
    assert primitives.calls == []


def test_kitchen_legacy_primitive_ids_are_derived_only_from_entity_bindings() -> None:
    primitives = RecordingPrimitives()
    facade = KitchenLiveControllerFacade(
        primitives,
        FakePhysicalState(),
        bindings={"cup_1": binding("cup_1", "cup_body")},
    )
    result = facade.execute_action(
        {"action_instance_id": "a0", "operator": "PICK", "arguments": ("cup_body",)}
    )
    assert result["success"] is True
    assert primitives.calls == [("pick", ("cup_1",))]


def test_living_room_facade_maps_motion_pick_and_place() -> None:
    primitives = RecordingPrimitives()
    facade = LivingRoomLiveControllerFacade(primitives, FakePhysicalState())
    assert facade.move_to("table", carrying_entity=None)["success"] is True
    assert facade.pick("cup")["success"] is True
    destination = facade.destination_for(
        payload_id="cup_1", payload_entity="cup", support_id="table_1", support_entity="table"
    )
    assert destination["success"] is True
    assert facade.place("cup", "table", destination)["success"] is True
    assert [name for name, _ in primitives.calls] == ["move_to", "pick", "destination_for", "place"]


@pytest.mark.parametrize(
    ("pddl_operator", "controller_operator", "method"),
    [
        ("open-storage", "OPEN", "open"),
        ("pick-from", "PICK", "pick"),
        ("place-on", "PLACE", "place"),
        ("insert", "PLACE", "insert"),
        ("drive", "SCREW", "drive"),
    ],
)
def test_workshop_facade_maps_every_controller_operator(pddl_operator, controller_operator, method) -> None:
    primitives = RecordingPrimitives()
    facade = WorkshopLiveControllerFacade(primitives, FakePhysicalState())
    argument_count = {"open-storage": 1, "pick-from": 2, "place-on": 2, "insert": 2, "drive": 3}[pddl_operator]
    arguments = tuple(f"entity_{index}" for index in range(argument_count))
    result = facade.execute_action(
        {"action_instance_id": "a0", "operator": controller_operator, "pddl_operator": pddl_operator, "arguments": arguments},
        object(),
    )
    assert result["success"] is True
    assert primitives.calls == [(method, arguments)]


@pytest.mark.parametrize("operator", ["POUR", "STIR"])
def test_kitchen_effect_is_gated_by_physical_postcondition(operator: str) -> None:
    pddl_operator = operator.lower()
    symbolic_arguments = ("source", "target", "content") if operator == "POUR" else ("source", "target")
    controller_arguments = ("source_body", "target_body")
    projection = ExecutionProjection(
        action_instance_id="a0",
        pddl_operator=pddl_operator,
        pddl_arguments=symbolic_arguments,
        controller_operator=operator,
        controller_arguments=controller_arguments,
        resolved_entities=controller_arguments,
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
        binding_confidence=1.0,
        binding_evidence_artifacts=(),
    )
    bindings = {
        "source": binding("source", "source_body"),
        "target": binding("target", "target_body"),
    }
    inventory = build_kitchen_inventory((projection,), bindings)
    failed = KitchenExecutionAdapter(
        controller=KitchenLiveControllerFacade(RecordingPrimitives(), FakePhysicalState(False)),
        inventory=inventory,
    ).execute((projection,))
    assert failed.success is False
    assert failed.effect_ledger == ()

    succeeded = KitchenExecutionAdapter(
        controller=KitchenLiveControllerFacade(RecordingPrimitives(), FakePhysicalState(True)),
        inventory=inventory,
    ).execute((projection,))
    assert succeeded.success is True
    assert [effect.effect for effect in succeeded.effect_ledger] == [f"{operator}_COMPLETED"]


def test_controller_success_does_not_bypass_pick_postcondition() -> None:
    projection = ExecutionProjection(
        action_instance_id="a0",
        pddl_operator="pick-from",
        pddl_arguments=("payload", "source"),
        controller_operator="PICK",
        controller_arguments=("payload_body",),
        resolved_entities=("payload_body",),
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
        binding_confidence=1.0,
        binding_evidence_artifacts=(),
    )
    inventory = build_kitchen_inventory(
        (projection,),
        {"payload": binding("payload", "payload_body")},
        fixed_bindings={"source": binding("source", "source_body", "storage")},
    )
    result = KitchenExecutionAdapter(
        controller=KitchenLiveControllerFacade(
            RecordingPrimitives(success=True), FakePhysicalState(False)
        ),
        inventory=inventory,
    ).execute((projection,))
    assert result.success is False
    assert result.actions[0].controller_success is True
    assert result.actions[0].postcondition_success is False


def test_drive_effect_is_gated_by_physical_postcondition() -> None:
    actions = (
        SymbolicAction(0, "a0", "pick-from", ("fastener", "source")),
        SymbolicAction(1, "a1", "insert", ("fastener", "target")),
        SymbolicAction(2, "a2", "pick-from", ("driver", "source")),
        SymbolicAction(3, "a3", "drive", ("driver", "fastener", "target")),
        SymbolicAction(4, "a4", "place-on", ("driver", "workbench")),
    )
    movable = {
        "fastener": binding("fastener", "fastener_body", "fastener"),
        "driver": binding("driver", "driver_body", "driver"),
    }
    fixed = {
        "source": binding("source", "source_body", "storage"),
        "target": binding("target", "target_body", "target"),
        "workbench": binding("workbench", "workbench_body", "surface"),
    }
    projections = project_plan("workshop", actions, movable, fixed_bindings=fixed)
    contract = build_workshop_controller_contract(projections)

    succeeded = WorkshopExecutionAdapter(
        dispatcher=WorkshopLiveControllerFacade(RecordingPrimitives(), FakePhysicalState()),
        controller_contract=contract,
    ).execute(projections)
    assert succeeded.success is True
    assert [effect.effect for effect in succeeded.effect_ledger] == ["DRIVE_COMPLETED"]

    class DriveFailure(FakePhysicalState):
        def verify_skill(self, operator, arguments, controller_result):
            return {"success": operator != "DRIVE", "operator": operator}

    failed = WorkshopExecutionAdapter(
        dispatcher=WorkshopLiveControllerFacade(RecordingPrimitives(), DriveFailure()),
        controller_contract=contract,
    ).execute(projections)
    assert failed.success is False
    assert failed.effect_ledger == ()


def test_live_stage_rejects_projection_not_derived_from_entity_bindings(tmp_path: Path) -> None:
    action = SymbolicAction(0, "a0", "pick-from", ("payload", "source"))
    plan = symbolic_execution_plan("living_room", (action,))
    movable = {"payload": binding("payload", "payload_body")}
    fixed = {"source": binding("source", "source_body", "location")}
    expected = project_plan("living_room", (action,), movable, fixed_bindings=fixed)[0]
    forged = ExecutionProjection(
        **{**expected.to_dict(), "controller_arguments": ("wrong_body",), "resolved_entities": ("wrong_body",)}
    )
    runtime = LiveDomainRuntime(
        controller=LivingRoomLiveControllerFacade(RecordingPrimitives(), FakePhysicalState()),
        physical_state=FakePhysicalState(),
        bindings=movable,
        fixed_bindings=fixed,
    )
    with pytest.raises(LiveExecutionError, match="not derived"):
        LiveExecutionStage(lambda domain, variant: runtime).execute(
            domain="living_room", variant="v0", execution_plan=plan, projections=(forged,), output_root=tmp_path
        )


def test_live_stage_runs_living_room_and_persists_terminal_artifacts(tmp_path: Path) -> None:
    actions = (
        SymbolicAction(0, "a0", "pick-from", ("payload", "source")),
        SymbolicAction(1, "a1", "place-on", ("payload", "support")),
    )
    plan = symbolic_execution_plan("living_room", actions)
    movable = {"payload": binding("payload", "payload_body")}
    fixed = {
        "source": binding("source", "source_body", "location"),
        "support": binding("support", "support_body", "support"),
    }
    projections = project_plan("living_room", actions, movable, fixed_bindings=fixed)
    physical = FakePhysicalState()
    runtime = LiveDomainRuntime(
        controller=LivingRoomLiveControllerFacade(RecordingPrimitives(), physical),
        physical_state=physical,
        bindings=movable,
        fixed_bindings=fixed,
    )
    result = LiveExecutionStage(lambda domain, variant: runtime).execute(
        domain="living_room", variant="v0", execution_plan=plan, projections=projections, output_root=tmp_path
    )
    assert result.success is True
    assert set(result.artifact_paths) == {
        "execution_entity_resolution", "execution_trace", "effect_ledger", "terminal_state", "execution_result"
    }
    assert all(Path(path).is_file() for path in result.artifact_paths.values())


def test_mujoco_terminal_snapshot_is_derived_from_scene_state() -> None:
    body_names = ("world", "support", "payload", "drawer", "google_gripper")
    geom_names = ("support_geom", "payload_geom", "drawer_geom", "gripper_geom", "floor")

    class FakeMuJoCo:
        class mjtObj:
            mjOBJ_BODY = 1
            mjOBJ_GEOM = 2

        class mjtJoint:
            mjJNT_FREE = 0

        @staticmethod
        def mj_name2id(model, object_type, name):
            names = body_names if object_type == 1 else geom_names
            return names.index(name) if name in names else -1

        @staticmethod
        def mj_id2name(model, object_type, index):
            names = body_names if object_type == 1 else geom_names
            return names[index]

        @staticmethod
        def mj_forward(model, data):
            return None

        @staticmethod
        def mj_objectVelocity(model, data, object_type, body_id, velocity, local):
            velocity[:] = 0.0

    model = SimpleNamespace(
        nbody=5,
        ngeom=5,
        njnt=1,
        neq=1,
        body_parentid=np.array([0, 0, 0, 0, 0]),
        geom_bodyid=np.array([1, 2, 3, 4, 0]),
        geom_size=np.array(
            [[0.3, 0.3, 0.1], [0.04, 0.04, 0.01], [0.2, 0.2, 0.2], [0.03, 0.03, 0.03], [2.0, 2.0, 0.1]]
        ),
        jnt_bodyid=np.array([3]),
        jnt_type=np.array([2]),
        jnt_qposadr=np.array([0]),
        jnt_range=np.array([[0.0, 0.4]]),
        eq_obj1id=np.array([4]),
        eq_obj2id=np.array([2]),
    )
    data = SimpleNamespace(
        qpos=np.array([0.35]),
        xpos=np.array([[0, 0, 0], [0, 0, 0.2], [0, 0, 0.31], [0.8, 0, 0.2], [0, 0, 1.0]], dtype=float),
        xquat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1)),
        geom_xpos=np.array([[0, 0, 0.2], [0, 0, 0.31], [0.8, 0, 0.2], [0, 0, 1.0], [0, 0, 0]], dtype=float),
        geom_xmat=np.tile(np.eye(3).reshape(1, 9), (5, 1)),
        contact=[SimpleNamespace(geom1=1, geom2=0, dist=0.0)],
        ncon=1,
        eq_active=np.array([0]),
    )
    observer = MuJoCoPhysicalStateObserver(
        SimpleNamespace(model=model, data=data),
        {"payload": binding("payload", "payload")},
        fixed_bindings={
            "support": binding("support", "support", "support"),
            "drawer": binding("drawer", "drawer", "storage"),
        },
        mujoco_module=FakeMuJoCo,
    )
    snapshot = observer.snapshot("living_room", False)
    assert snapshot.objects["payload"]["world_position_m"] == pytest.approx(data.xpos[2])
    assert snapshot.objects["payload"]["support"] == "support"
    assert snapshot.objects["payload"]["support_contact"] is True
    assert snapshot.relations["articulation"]["drawer"]["open"] is True
    assert "benchmark_success" not in snapshot.measurements
    data.eq_active[0] = 1
    assert observer.snapshot("living_room", False).held_objects == ("payload",)


def symbolic_execution_plan(domain: str, actions: tuple[SymbolicAction, ...]) -> BaselineExecutionPlan:
    symbolic = SymbolicPlan(
        attempt_index=0,
        planner_name="test",
        planner_version="1",
        search_configuration="test",
        actions=actions,
        plan_cost=1.0,
        planner_time_seconds=0.0,
        raw_plan_artifacts=(),
        plan_sha256="a" * 64,
    )
    return BaselineExecutionPlan(
        selected_attempt_index=0,
        domain=domain,
        symbolic_plan=symbolic,
        refinement_certificate={"status": "SUCCESS"},
        normalized_actions=actions,
    )
