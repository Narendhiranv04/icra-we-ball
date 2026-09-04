from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    ExecutionProjection,
    RefinementStage,
    SymbolicAction,
)
from mujoco_scenes.baselines.vilain_tamp.live_refinement import (
    MuJoCoGeometryKernel,
    MuJoCoPlanningSceneFactory,
    RefinementThresholds,
    create_live_refinement_runtime,
)


class FakeObjectType:
    mjOBJ_BODY = 1
    mjOBJ_SITE = 2
    mjOBJ_JOINT = 3


class FakeJointType:
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_SLIDE = 2
    mjJNT_HINGE = 3


class FakeModel:
    def __init__(self) -> None:
        self.body_names = (
            "world",
            "source_body",
            "target_body",
            "robot:gripper",
            "storage_body",
            "obstacle_body",
            "storage_door",
        )
        self.body_ids = {name: index for index, name in enumerate(self.body_names)}
        self.body_parentid = np.asarray((0, 0, 0, 0, 0, 0, 4))
        self.site_names = ("source_body_grasp", "storage_body_handle")
        self.site_bodyid = np.asarray((1, 6))
        self.joint_names = ("arm_0", "arm_1", "source_free", "storage_slide")
        self.joint_ids = {name: index for index, name in enumerate(self.joint_names)}
        self.ngeom = 5
        self.nsite = 2
        self.geom_bodyid = np.asarray((1, 2, 3, 6, 5))
        self.geom_aabb = np.asarray(
            (
                (0, 0, 0, 0.03, 0.03, 0.10),
                (0, 0, 0, 0.12, 0.12, 0.10),
                (0, 0, 0, 0.02, 0.02, 0.02),
                (0, 0, 0, 0.10, 0.10, 0.10),
                (0, 0, 0, 0.04, 0.04, 0.04),
            ),
            dtype=float,
        )
        self.geom_contype = np.ones(self.ngeom, dtype=int)
        self.geom_conaffinity = np.ones(self.ngeom, dtype=int)
        self.jnt_qposadr = np.asarray((0, 1, 2, 9))
        self.jnt_dofadr = np.asarray((0, 1, 2, 8))
        self.jnt_type = np.asarray((3, 3, 0, 2))
        self.jnt_range = np.asarray(((-2, 2), (-2, 2), (0, 0), (0, 0.3)), dtype=float)
        self.body_jntnum = np.asarray((0, 1, 0, 0, 0, 0, 1))
        self.body_jntadr = np.asarray((-1, 2, -1, -1, -1, -1, 3))
        self.fixed_positions = np.asarray(
            (
                (0, 0, 0),
                (0, 0, 0),
                (0.5, 0, 0.10),
                (0, 0, 0),
                (0.8, 0, 0.10),
                (2.0, 2.0, 2.0),
                (0.8, 0, 0.10),
            ),
            dtype=float,
        )
        self.fixed_rotations = np.repeat(np.eye(3)[None, :, :], 7, axis=0)


class FakeData:
    def __init__(self, model: FakeModel) -> None:
        self.qpos = np.zeros(10)
        self.qpos[2:5] = (0.0, 0.0, 0.10)
        self.qpos[5] = 1.0
        self.qvel = np.zeros(9)
        self.act = np.zeros(0)
        self.ctrl = np.zeros(0)
        self.mocap_pos = np.zeros((0, 3))
        self.mocap_quat = np.zeros((0, 4))
        self.eq_active = np.zeros(0, dtype=np.uint8)
        self.time = 0.0
        self.xpos = model.fixed_positions.copy()
        self.xmat = model.fixed_rotations.reshape(7, 9).copy()
        self.geom_xpos = np.zeros((model.ngeom, 3))
        self.geom_xmat = np.repeat(np.eye(3).reshape(1, 9), model.ngeom, axis=0)
        self.site_xpos = np.zeros((model.nsite, 3))


class FakeModelLoader:
    saved: FakeModel | None = None

    @classmethod
    def from_binary_path(cls, path: str) -> FakeModel:
        assert Path(path).is_file()
        assert cls.saved is not None
        return deepcopy(cls.saved)


class FakeMuJoCo:
    mjtObj = FakeObjectType
    mjtJoint = FakeJointType
    MjModel = FakeModelLoader

    def __init__(self) -> None:
        self.forward_calls = 0

    @staticmethod
    def MjData(model: FakeModel) -> FakeData:
        return FakeData(model)

    @staticmethod
    def mj_saveModel(model: FakeModel, path: str, buffer: object) -> None:
        assert buffer is None
        FakeModelLoader.saved = deepcopy(model)
        Path(path).write_bytes(b"fake-mjb")

    def mj_forward(self, model: FakeModel, data: FakeData) -> None:
        self.forward_calls += 1
        source_rotation = _quaternion_matrix(data.qpos[5:9])
        data.xpos[:] = model.fixed_positions
        data.xpos[1] = data.qpos[2:5]
        data.xpos[3] = (data.qpos[0], 0.0, data.qpos[1])
        data.xpos[6, 0] += data.qpos[9]
        data.xmat[:] = model.fixed_rotations.reshape(7, 9)
        data.xmat[1] = source_rotation.reshape(-1)
        for geom_id, body_id in enumerate(model.geom_bodyid):
            data.geom_xpos[geom_id] = data.xpos[body_id]
            data.geom_xmat[geom_id] = data.xmat[body_id]
        data.site_xpos[0] = data.xpos[1] + source_rotation @ np.asarray((0, 0, 0.1))
        data.site_xpos[1] = data.xpos[6] + np.asarray((0, 0, 0.1))

    @staticmethod
    def mj_name2id(model: FakeModel, object_type: int, name: str) -> int:
        if object_type == FakeObjectType.mjOBJ_BODY:
            return model.body_ids.get(name, -1)
        if object_type == FakeObjectType.mjOBJ_JOINT:
            return model.joint_ids.get(name, -1)
        if object_type == FakeObjectType.mjOBJ_SITE:
            return model.site_names.index(name) if name in model.site_names else -1
        return -1

    @staticmethod
    def mj_id2name(model: FakeModel, object_type: int, index: int) -> str | None:
        if object_type == FakeObjectType.mjOBJ_BODY:
            return model.body_names[index]
        if object_type == FakeObjectType.mjOBJ_SITE:
            return model.site_names[index]
        if object_type == FakeObjectType.mjOBJ_JOINT:
            return model.joint_names[index]
        return None

    @staticmethod
    def mj_geomDistance(model, data, first, second, distance_limit, from_to) -> float:
        del distance_limit, from_to
        half_a = model.geom_aabb[first, 3:]
        half_b = model.geom_aabb[second, 3:]
        gaps = np.abs(data.geom_xpos[first] - data.geom_xpos[second]) - half_a - half_b
        return float(np.max(gaps))


class FakeIK:
    def __init__(self, model, data, profile) -> None:
        del model, data, profile

    def solve(self, target, seed, rotation):
        del seed, rotation
        return np.asarray((target[0], target[2])), 0.001, 0.01


class FakeCollisionChecker:
    fail = False
    calls: list[tuple[np.ndarray, np.ndarray, frozenset[int]]] = []

    def __init__(self, model, data, profile) -> None:
        del model, data, profile

    def segment_valid(self, start, goal, allowed_environment_bodies, resolution):
        del resolution
        self.calls.append((start.copy(), goal.copy(), allowed_environment_bodies))
        return not self.fail, "robot collision" if self.fail else None


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        (
            (1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w),
            (2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w),
            (2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y),
        )
    )


def _profile() -> object:
    return SimpleNamespace(
        arm_joints=("arm_0", "arm_1"),
        top_down_rotation=np.eye(3),
        gripper_body="robot:gripper",
    )


def _scene() -> tuple[object, FakeMuJoCo]:
    model = FakeModel()
    data = FakeData(model)
    data.qpos[:2] = (0.0, 0.20)
    data.qvel[:2] = (0.3, 0.4)
    data.time = 7.5
    mujoco = FakeMuJoCo()
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data), mujoco


def _action(index: int, operator: str, entities: tuple[str, ...]):
    arity = {"open-storage": 1, "pour": 3, "drive": 3}.get(operator, 2)
    action = SymbolicAction(
        index,
        f"vilain_00_{index:03d}_{operator.replace('-', '_')}",
        operator,
        tuple(f"arg_{item}" for item in range(arity)),
    )
    projection = ExecutionProjection(
        action.action_instance_id,
        action.operator,
        action.arguments,
        operator.upper().replace("-", "_"),
        entities,
        entities,
        "ONE_TO_ONE_CLASS_CENTROID_AABB",
        0.9,
        (),
        {},
    )
    return action, projection


def _kernel(thresholds: RefinementThresholds | None = None):
    return MuJoCoGeometryKernel(
        thresholds=thresholds or RefinementThresholds(joint_interpolation_step_rad=0.1),
        ik_factory=FakeIK,
        collision_checker_factory=FakeCollisionChecker,
    )


def _planning(live_scene: object, mujoco: FakeMuJoCo, initial_state=None):
    return MuJoCoPlanningSceneFactory(
        live_scene,
        initial_state=initial_state,
        mujoco_module=mujoco,
        profile=_profile(),
    )()


def _run(scene, mujoco, pairs, tmp_path, *, initial_state=None, kernel=None):
    runtime = create_live_refinement_runtime(
        scene,
        initial_state=initial_state,
        mujoco_module=mujoco,
        profile=_profile(),
        kernel=kernel or _kernel(),
    )
    planning = runtime.planning_scene_factory()
    result = runtime.refiner.refine(
        attempt_index=0,
        actions=tuple(pair[0] for pair in pairs),
        projections=tuple(pair[1] for pair in pairs),
        planning_scene_factory=lambda: planning,
        output_root=tmp_path,
    )
    return result, planning


def test_factory_clones_model_and_data_and_both_are_independent() -> None:
    live, mujoco = _scene()
    planning = _planning(live, mujoco, {"marker": "planning-only"})
    planning.model.geom_aabb[0, 3] = 9.0
    planning.data.qpos[0] = 9.0
    planning.data.qvel[:] = 0.0
    assert planning.model is not live.model
    assert planning.data is not live.data
    assert live.model.geom_aabb[0, 3] == 0.03
    assert tuple(live.data.qpos[:2]) == (0.0, 0.2)
    assert tuple(live.data.qvel[:2]) == (0.3, 0.4)
    assert planning.predicted_state == {"marker": "planning-only"}
    assert len(planning.source_qpos_sha256) == 64


def test_factory_clones_real_mujoco_model_and_data_when_available() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="robot:gripper" pos="0 0 0.2">
              <joint name="arm_0" type="slide" axis="1 0 0"/>
              <joint name="arm_1" type="slide" axis="0 0 1"/>
              <geom type="sphere" size="0.02"/>
            </body>
            <body name="source_body" pos="0 0 0.1">
              <freejoint name="source_free"/>
              <geom type="box" size="0.03 0.03 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[0] = 0.17
    mujoco.mj_forward(model, data)
    live = SimpleNamespace(model=model, data=data)
    planning = MuJoCoPlanningSceneFactory(
        live, mujoco_module=mujoco, profile=_profile()
    )()

    planning.model.geom_pos[0, 0] = 3.0
    planning.data.qpos[0] = -0.4
    assert planning.model is not model
    assert planning.data is not data
    assert model.geom_pos[0, 0] == 0.0
    assert data.qpos[0] == pytest.approx(0.17)


def test_pick_attach_retreat_then_place_uses_propagated_payload_state(
    tmp_path: Path,
) -> None:
    live, mujoco = _scene()
    pick = _action(0, "pick-from", ("source_body",))
    place = _action(1, "place-on", ("source_body", "target_body"))
    result, planning = _run(live, mujoco, (pick, place), tmp_path)
    assert result.success
    assert (
        result.certificate.actions[0].predicted_terminal_state["held_entity"]
        == "source_body"
    )
    assert result.certificate.actions[1].predicted_terminal_state["held_entity"] is None
    assert planning.attachment is None
    assert planning.data.xpos[1] == pytest.approx((0.5, 0.0, 0.33))
    assert live.data.xpos[1] == pytest.approx((0.0, 0.0, 0.10))
    second_entities = json.loads(
        (
            tmp_path
            / "refinement_stages"
            / place[0].action_instance_id
            / "entity_resolution.json"
        ).read_text()
    )
    assert second_entities["entities"][0]["centroid_m"][2] > 0.15


def test_open_updates_articulation_and_next_action_sees_new_geometry(
    tmp_path: Path,
) -> None:
    live, mujoco = _scene()
    first = _action(0, "open-storage", ("storage_body",))
    second = _action(1, "open-storage", ("storage_body",))
    result, planning = _run(live, mujoco, (first, second), tmp_path)
    assert result.success
    assert planning.data.qpos[9] == pytest.approx(0.3)
    assert live.data.qpos[9] == 0.0
    trace = json.loads(
        (
            tmp_path
            / "refinement_stages"
            / second[0].action_instance_id
            / "entity_resolution.json"
        ).read_text()
    )
    assert trace["entities"][0]["centroid_m"][0] == pytest.approx(1.1)


def test_payload_swept_collision_rejects_path_robot_checker_accepts(
    tmp_path: Path,
) -> None:
    live, mujoco = _scene()
    live.model.fixed_positions[5] = (0.25, 0.0, 0.27)
    mujoco.mj_forward(live.model, live.data)
    pour = _action(0, "pour", ("source_body", "target_body"))
    FakeCollisionChecker.fail = False
    result, _ = _run(
        live,
        mujoco,
        (pour,),
        tmp_path,
        initial_state={"held_entity": "source_body"},
    )
    assert not result.success
    assert result.failure.stage is RefinementStage.COLLISION
    assert result.failure.reason_code == "PATH_COLLISION"
    assert result.failure.collision_pair == ("source_body", "obstacle_body")


@pytest.mark.parametrize(
    ("operator", "passing_thresholds", "failing_thresholds"),
    (
        (
            "pour",
            RefinementThresholds(),
            RefinementThresholds(pour_min_tilt_rad=np.deg2rad(70)),
        ),
        (
            "stir",
            RefinementThresholds(),
            RefinementThresholds(stir_radius_fraction=0.1),
        ),
        ("drive", RefinementThresholds(), RefinementThresholds(drive_max_depth_m=0.01)),
    ),
)
def test_black_box_skill_has_real_geometric_pass_and_fail_envelopes(
    operator: str,
    passing_thresholds: RefinementThresholds,
    failing_thresholds: RefinementThresholds,
    tmp_path: Path,
) -> None:
    live, mujoco = _scene()
    # The fake target is a solid AABB; disable its collision proxy here so the
    # test isolates the object-centric envelope instead of modeling a cavity.
    live.model.geom_contype[1] = 0
    live.model.geom_conaffinity[1] = 0
    if operator == "drive":
        live.model.geom_aabb[1, 3:] = (0.02, 0.02, 0.08)
    pair = _action(0, operator, ("source_body", "target_body"))
    held = {"held_entity": "source_body"}
    passed, _ = _run(
        live,
        mujoco,
        (pair,),
        tmp_path / "pass",
        initial_state=held,
        kernel=_kernel(passing_thresholds),
    )
    failed, _ = _run(
        live,
        mujoco,
        (pair,),
        tmp_path / "fail",
        initial_state=held,
        kernel=_kernel(failing_thresholds),
    )
    assert passed.success
    envelope = json.loads(
        (
            tmp_path
            / "pass"
            / "refinement_stages"
            / pair[0].action_instance_id
            / "skill_envelope.json"
        ).read_text()
    )["envelope"]
    assert envelope["controller_invoked"] is False
    assert envelope["constraints"]
    assert all(envelope["constraints"].values())
    trajectory = json.loads(
        (
            tmp_path
            / "pass"
            / "refinement_stages"
            / pair[0].action_instance_id
            / "trajectory.json"
        ).read_text()
    )["joint_waypoints"]
    chosen = json.loads(
        (
            tmp_path
            / "pass"
            / "refinement_stages"
            / pair[0].action_instance_id
            / "ik.json"
        ).read_text()
    )["chosen"]
    assert trajectory[-1] == pytest.approx(chosen["approach_qpos"])
    assert chosen["target_qpos"] in trajectory
    assert not failed.success
    assert failed.failure.stage is RefinementStage.SKILL_ENVELOPE
    assert failed.failure.reason_code == "SKILL_ENVELOPE_FAILED"


def test_robot_collision_stops_before_envelope_or_transition(tmp_path: Path) -> None:
    live, mujoco = _scene()
    pair = _action(0, "pour", ("source_body", "target_body"))
    FakeCollisionChecker.fail = True
    try:
        result, _ = _run(live, mujoco, (pair,), tmp_path)
    finally:
        FakeCollisionChecker.fail = False
    assert not result.success
    assert result.failure.stage is RefinementStage.COLLISION
    stage_root = tmp_path / "refinement_stages" / pair[0].action_instance_id
    assert not (stage_root / "skill_envelope.json").exists()
    assert not (stage_root / "state_transition.json").exists()


def test_missing_projected_body_is_structured_entity_failure(tmp_path: Path) -> None:
    live, mujoco = _scene()
    pair = _action(0, "pour", ("missing_body", "target_body"))
    result, _ = _run(live, mujoco, (pair,), tmp_path)
    assert not result.success
    assert result.failure.stage is RefinementStage.ENTITY_RESOLUTION
    assert result.failure.reason_code == "MISSING_SCENE_ENTITY"


def test_pick_prefers_named_grasp_site(tmp_path: Path) -> None:
    live, mujoco = _scene()
    pair = _action(0, "pick-from", ("source_body",))
    result, _ = _run(live, mujoco, (pair,), tmp_path)
    assert result.success
    trace = json.loads(
        (
            tmp_path
            / "refinement_stages"
            / pair[0].action_instance_id
            / "grasp_generation.json"
        ).read_text()
    )
    assert all(item["source"] == "NAMED_GRASP_SITE" for item in trace["candidates"])
