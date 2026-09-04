from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mujoco_scenes.baselines.vilain_tamp.config import Domain, ObservationMode
from mujoco_scenes.baselines.vilain_tamp.observations import (
    CameraFrameCapture,
    FIXED_INSPECTION_ORDERS,
    ObservationProtocol,
    prompt_observation_payload,
)


INTRINSICS = (
    (500.0, 0.0, 320.0),
    (0.0, 500.0, 240.0),
    (0.0, 0.0, 1.0),
)
EXTRINSICS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FakeCaptureBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def capture(self, camera_id: str, stage_id: str) -> CameraFrameCapture:
        self.calls.append((camera_id, stage_id))
        value = float(len(self.calls))
        return CameraFrameCapture(
            camera_id=camera_id,
            view_description=f"Public view {camera_id}",
            rgb_png=b"\x89PNG\r\n\x1a\nsynthetic",
            depth_m=np.full((2, 3), value, dtype=np.float64),
            intrinsics=INTRINSICS,
            extrinsics=EXTRINSICS,
        )


class FakeOpeningBackend:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def open_region(self, region_id: str) -> dict[str, Any]:
        self.opened.append(region_id)
        return {
            "complete": True,
            "hidden_body_names": ["must_not_be_persisted"],
        }


def fixed_clock() -> str:
    return "2026-09-04T00:00:00+00:00"


def make_protocol(
    tmp_path: Path,
    *,
    domain: Domain,
    mode: ObservationMode,
    capture: FakeCaptureBackend | None = None,
    opener: FakeOpeningBackend | None = None,
) -> tuple[ObservationProtocol, FakeCaptureBackend, FakeOpeningBackend | None]:
    capture = capture or FakeCaptureBackend()
    protocol = ObservationProtocol(
        domain=domain,
        observation_mode=mode,
        camera_ids=("front", "side"),
        output_root=tmp_path,
        capture_backend=capture,
        opening_backend=opener,
        clock=fixed_clock,
    )
    return protocol, capture, opener


def test_initial_only_captures_once_and_opens_nothing(tmp_path: Path) -> None:
    opener = FakeOpeningBackend()
    protocol, capture, _ = make_protocol(
        tmp_path,
        domain=Domain.KITCHEN,
        mode=ObservationMode.INITIAL_ONLY,
        opener=opener,
    )
    result = protocol.acquire()
    assert [item.stage_id for item in result.observations] == ["000_initial"]
    assert opener.opened == []
    assert capture.calls == [("front", "000_initial"), ("side", "000_initial")]
    assert result.inspection_trace == ()


@pytest.mark.parametrize(
    "domain",
    [Domain.KITCHEN, Domain.WORKSHOP],
)
def test_full_inspection_uses_entire_fixed_order_without_early_stop(
    tmp_path: Path, domain: Domain
) -> None:
    opener = FakeOpeningBackend()
    protocol, _, _ = make_protocol(
        tmp_path,
        domain=domain,
        mode=ObservationMode.FIXED_FULL_INSPECTION,
        opener=opener,
    )
    result = protocol.acquire()
    expected = FIXED_INSPECTION_ORDERS[domain]
    assert opener.opened == list(expected)
    assert [item.opened_region_id for item in result.observations] == [None, *expected]
    assert [row["inspection_ordinal"] for row in result.inspection_trace] == list(
        range(1, len(expected) + 1)
    )


def test_living_room_full_condition_is_canonical_initial_multiview(
    tmp_path: Path,
) -> None:
    protocol, capture, _ = make_protocol(
        tmp_path,
        domain=Domain.LIVING_ROOM,
        mode=ObservationMode.FIXED_FULL_INSPECTION,
    )
    result = protocol.acquire()
    assert len(result.observations) == 1
    assert len(capture.calls) == 2
    assert result.inspection_trace == ()


def test_rgb_depth_calibration_and_hashes_are_persisted(tmp_path: Path) -> None:
    protocol, _, _ = make_protocol(
        tmp_path,
        domain=Domain.KITCHEN,
        mode=ObservationMode.INITIAL_ONLY,
    )
    result = protocol.acquire()
    frame = result.observations[0].camera_frames[0]
    rgb_path = tmp_path / frame.rgb_path
    depth_path = tmp_path / frame.depth_path
    calibration_path = tmp_path / frame.calibration_path
    assert rgb_path.read_bytes().startswith(b"\x89PNG")
    depth = np.load(depth_path, allow_pickle=False)
    assert depth.dtype == np.float32
    assert depth.shape == (2, 3)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration["depth_unit"] == "meter"
    assert calibration["intrinsics"] == [list(row) for row in INTRINSICS]
    assert calibration["extrinsics"] == [list(row) for row in EXTRINSICS]
    assert len(frame.rgb_sha256) == 64
    assert len(frame.depth_sha256) == 64
    assert len(frame.calibration_sha256) == 64
    assert result.manifest_path.is_file()
    assert result.inspection_trace_path.is_file()


def test_prompt_payload_exposes_only_public_rgb_metadata(tmp_path: Path) -> None:
    opener = FakeOpeningBackend()
    protocol, _, _ = make_protocol(
        tmp_path,
        domain=Domain.WORKSHOP,
        mode=ObservationMode.FIXED_FULL_INSPECTION,
        opener=opener,
    )
    result = protocol.acquire()
    payload = prompt_observation_payload(result.observations)
    rendered = json.dumps(payload, sort_keys=True)
    assert "must_not_be_persisted" not in rendered
    assert "variant" not in rendered.lower()
    assert "body_name" not in rendered.lower()
    assert "depth_path" not in rendered
    assert "calibration_path" not in rendered
    assert payload["stages"][0]["images"][0]["rgb_path"].endswith("rgb.png")


def test_full_inspection_requires_opener_for_storage_domains(tmp_path: Path) -> None:
    protocol, _, _ = make_protocol(
        tmp_path,
        domain=Domain.KITCHEN,
        mode=ObservationMode.FIXED_FULL_INSPECTION,
    )
    with pytest.raises(ValueError, match="region-opening backend"):
        protocol.acquire()
