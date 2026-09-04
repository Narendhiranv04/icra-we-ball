from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mujoco_scenes.baselines.vilain_tamp.config import Domain, ObservationMode
from mujoco_scenes.baselines.vilain_tamp.live_observations import (
    CANONICAL_CAMERAS,
    LiveObservationError,
    MuJoCoRGBDCaptureBackend,
    SceneRegionOpeningBackend,
    create_live_observation_runtime,
)
from mujoco_scenes.baselines.vilain_tamp.observations import (
    FIXED_INSPECTION_ORDERS,
    prompt_observation_payload,
)


class FakeModel:
    def __init__(self, camera_count: int) -> None:
        self.cam_fovy = np.full(camera_count, 60.0)


class FakeData:
    def __init__(self, camera_count: int) -> None:
        self.cam_xpos = np.zeros((camera_count, 3))
        self.cam_xpos[:, 2] = 1.0
        self.cam_xmat = np.tile(np.eye(3).reshape(1, 9), (camera_count, 1))


class FakeScene:
    def __init__(self, cameras: tuple[str, ...], *, with_groups: bool = False) -> None:
        self.camera_ids = {name: index for index, name in enumerate(cameras)}
        self.model = FakeModel(len(cameras))
        self.data = FakeData(len(cameras))
        self.opened: list[str] = []
        if with_groups:
            self.perception_render_geom_groups = (0, 2)

    def open_container(self, region_id: str):
        self.opened.append(region_id)
        return {"hidden_body_names": ["must-not-propagate"]}


class FakeOption:
    def __init__(self) -> None:
        self.geomgroup = np.ones(6, dtype=np.uint8)


class FakeRenderer:
    instances: list["FakeRenderer"] = []

    def __init__(self, model, *, height: int, width: int) -> None:
        del model
        self.height = height
        self.width = width
        self.depth = False
        self.closed = False
        self.updates = []
        self.__class__.instances.append(self)

    def disable_depth_rendering(self) -> None:
        self.depth = False

    def enable_depth_rendering(self) -> None:
        self.depth = True

    def disable_segmentation_rendering(self) -> None:
        pass

    def update_scene(self, data, **kwargs) -> None:
        del data
        self.updates.append(kwargs)

    def render(self):
        if self.depth:
            return np.full((self.height, self.width), 2.5, dtype=np.float32)
        return np.full((self.height, self.width, 3), 127, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeMujoco:
    class mjtObj:
        mjOBJ_CAMERA = 7

    Renderer = FakeRenderer
    MjvOption = FakeOption

    @staticmethod
    def mj_name2id(model, object_type, name: str) -> int:
        del object_type
        return getattr(model, "camera_ids", {}).get(name, -1)

    @staticmethod
    def mj_forward(model, data) -> None:
        del model, data


def fake_scene(domain: Domain, *, with_groups: bool = False) -> FakeScene:
    scene = FakeScene(CANONICAL_CAMERAS[domain], with_groups=with_groups)
    scene.model.camera_ids = scene.camera_ids
    return scene


@pytest.mark.parametrize("domain", tuple(Domain))
def test_live_capture_records_metric_depth_and_camera_calibration(domain: Domain) -> None:
    scene = fake_scene(domain, with_groups=domain is Domain.WORKSHOP)
    backend = MuJoCoRGBDCaptureBackend(
        domain=domain,
        scene=scene,
        width=8,
        height=6,
        mujoco_module=FakeMujoco,
    )
    capture = backend.capture(CANONICAL_CAMERAS[domain][0], "000_initial")

    assert capture.rgb_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert capture.depth_m.shape == (6, 8)
    assert capture.depth_m.dtype == np.float32
    assert np.all(capture.depth_m == 2.5)
    assert capture.intrinsics[0][0] == pytest.approx(3.0 * np.sqrt(3.0))
    assert capture.intrinsics[0][2] == 4.0
    assert capture.intrinsics[1][2] == 3.0
    assert np.allclose(
        capture.extrinsics,
        ((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 1), (0, 0, 0, 1)),
    )
    renderer = FakeRenderer.instances[-1]
    assert renderer.closed
    if domain is Domain.WORKSHOP:
        option = renderer.updates[0]["scene_option"]
        assert option.geomgroup.tolist() == [1, 0, 1, 0, 0, 0]


@pytest.mark.parametrize("domain", tuple(Domain))
def test_live_initial_only_uses_canonical_multiview_without_opening(
    domain: Domain, tmp_path: Path
) -> None:
    scene = fake_scene(domain)
    runtime = create_live_observation_runtime(
        domain=domain,
        variant="fixture-variant",
        observation_mode=ObservationMode.INITIAL_ONLY,
        output_root=tmp_path,
        width=8,
        height=6,
        scene=scene,
        mujoco_module=FakeMujoco,
    )
    result = runtime.protocol.acquire()

    assert runtime.scene is scene
    assert scene.opened == []
    assert len(result.observations) == 1
    assert tuple(
        frame.camera_id for frame in result.observations[0].camera_frames
    ) == CANONICAL_CAMERAS[domain]


@pytest.mark.parametrize("domain", (Domain.KITCHEN, Domain.WORKSHOP))
def test_live_full_inspection_opens_every_region_in_fixed_order(
    domain: Domain, tmp_path: Path
) -> None:
    scene = fake_scene(domain)
    runtime = create_live_observation_runtime(
        domain=domain,
        variant="fixture-variant",
        observation_mode=ObservationMode.FIXED_FULL_INSPECTION,
        output_root=tmp_path,
        width=4,
        height=3,
        scene=scene,
        mujoco_module=FakeMujoco,
    )
    result = runtime.protocol.acquire()

    assert tuple(scene.opened) == FIXED_INSPECTION_ORDERS[domain]
    assert len(result.observations) == 1 + len(FIXED_INSPECTION_ORDERS[domain])
    persisted = json.loads(result.inspection_trace_path.read_text(encoding="utf-8"))
    assert [row["region_id"] for row in persisted["openings"]] == list(
        FIXED_INSPECTION_ORDERS[domain]
    )


def test_model_payload_excludes_variant_and_backend_identifiers(tmp_path: Path) -> None:
    scene = fake_scene(Domain.KITCHEN)
    runtime = create_live_observation_runtime(
        domain=Domain.KITCHEN,
        variant="I4_MISSING_KETTLE",
        observation_mode=ObservationMode.FIXED_FULL_INSPECTION,
        output_root=tmp_path,
        width=4,
        height=3,
        scene=scene,
        mujoco_module=FakeMujoco,
    )
    result = runtime.protocol.acquire()
    payload = json.dumps(prompt_observation_payload(result.observations), sort_keys=True)

    assert "I4_MISSING_KETTLE" not in payload
    assert "must-not-propagate" not in payload
    assert "body" not in payload.lower()


def test_opening_and_capture_reject_noncanonical_inputs() -> None:
    scene = fake_scene(Domain.KITCHEN)
    opener = SceneRegionOpeningBackend(domain=Domain.KITCHEN, scene=scene)
    with pytest.raises(LiveObservationError, match="fixed inspection order"):
        opener.open_region("SECRET_REGION")
    backend = MuJoCoRGBDCaptureBackend(
        domain=Domain.KITCHEN,
        scene=scene,
        mujoco_module=FakeMujoco,
    )
    with pytest.raises(LiveObservationError, match="not canonical"):
        backend.capture("hidden_oracle_camera", "000_initial")
