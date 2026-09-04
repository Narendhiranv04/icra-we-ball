"""Fair initial-only and task-independent full-inspection acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .artifacts import atomic_write_json, sha256_file
from .config import Domain, ObservationMode
from .contracts import CameraFrameArtifacts, ViLaInObservation


FIXED_INSPECTION_ORDERS: Mapping[Domain, tuple[str, ...]] = {
    Domain.KITCHEN: ("D1", "D2", "C2", "B1", "C1"),
    Domain.WORKSHOP: ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    Domain.LIVING_ROOM: (),
}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CameraFrameCapture:
    """One backend-produced RGB-D frame with metric camera calibration."""

    camera_id: str
    view_description: str
    rgb_png: bytes
    depth_m: np.ndarray
    intrinsics: tuple[tuple[float, ...], ...]
    extrinsics: tuple[tuple[float, ...], ...]


class CaptureBackend(Protocol):
    def capture(self, camera_id: str, stage_id: str) -> CameraFrameCapture: ...


class RegionOpeningBackend(Protocol):
    def open_region(self, region_id: str) -> object: ...


@dataclass(frozen=True)
class ObservationAcquisitionResult:
    observations: tuple[ViLaInObservation, ...]
    inspection_trace: tuple[Mapping[str, object], ...]
    manifest_path: Path
    inspection_trace_path: Path


class ObservationProtocol:
    """Acquire an immutable observation sequence without task-aware stopping."""

    def __init__(
        self,
        *,
        domain: Domain | str,
        observation_mode: ObservationMode | str,
        camera_ids: Sequence[str],
        output_root: str | Path,
        capture_backend: CaptureBackend,
        opening_backend: RegionOpeningBackend | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.domain = domain if isinstance(domain, Domain) else Domain(domain)
        self.observation_mode = (
            observation_mode
            if isinstance(observation_mode, ObservationMode)
            else ObservationMode(observation_mode)
        )
        self.camera_ids = tuple(camera_ids)
        if not self.camera_ids:
            raise ValueError("camera_ids must contain at least one camera")
        for camera_id in self.camera_ids:
            _validate_component(camera_id, "camera ID")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be unique")
        self.output_root = Path(output_root)
        self.capture_backend = capture_backend
        self.opening_backend = opening_backend
        self.clock = clock or _utc_now

    def acquire(self) -> ObservationAcquisitionResult:
        observations = [self._capture_stage("000_initial", None, None)]
        trace: list[Mapping[str, object]] = []

        if self.observation_mode is ObservationMode.FIXED_FULL_INSPECTION:
            order = FIXED_INSPECTION_ORDERS[self.domain]
            if order and self.opening_backend is None:
                raise ValueError("fixed full inspection requires a region-opening backend")
            for ordinal, region_id in enumerate(order, start=1):
                started = time.perf_counter()
                assert self.opening_backend is not None
                self.opening_backend.open_region(region_id)
                duration = time.perf_counter() - started
                trace.append(
                    {
                        "inspection_ordinal": ordinal,
                        "region_id": region_id,
                        "status": "OPENED",
                        "duration_seconds": duration,
                    }
                )
                stage_id = f"{ordinal:03d}_{region_id.lower()}"
                observations.append(self._capture_stage(stage_id, region_id, ordinal))

        trace_path = atomic_write_json(
            self.output_root / "inspection_trace.json",
            {"domain": self.domain.value, "openings": trace},
        )
        manifest_path = atomic_write_json(
            self.output_root / "observation_manifest.json",
            {
                "domain": self.domain.value,
                "observation_mode": self.observation_mode.value,
                "fixed_inspection_order": list(
                    FIXED_INSPECTION_ORDERS[self.domain]
                    if self.observation_mode is ObservationMode.FIXED_FULL_INSPECTION
                    else ()
                ),
                "observations": [observation.to_dict() for observation in observations],
                "inspection_trace_path": trace_path.relative_to(self.output_root).as_posix(),
            },
        )
        return ObservationAcquisitionResult(
            observations=tuple(observations),
            inspection_trace=tuple(trace),
            manifest_path=manifest_path,
            inspection_trace_path=trace_path,
        )

    def _capture_stage(
        self,
        stage_id: str,
        opened_region_id: str | None,
        inspection_ordinal: int | None,
    ) -> ViLaInObservation:
        stage_root = self.output_root / "stages" / stage_id / "cameras"
        frames = tuple(
            self._persist_frame(
                self.capture_backend.capture(camera_id, stage_id),
                expected_camera_id=camera_id,
                stage_root=stage_root,
            )
            for camera_id in self.camera_ids
        )
        content_hash = _observation_hash(stage_id, opened_region_id, frames)
        return ViLaInObservation(
            domain=self.domain.value,
            observation_mode=self.observation_mode.value,
            stage_id=stage_id,
            camera_frames=frames,
            opened_region_id=opened_region_id,
            capture_timestamp=self.clock(),
            inspection_ordinal=inspection_ordinal,
            content_hash=content_hash,
        )

    def _persist_frame(
        self,
        capture: CameraFrameCapture,
        *,
        expected_camera_id: str,
        stage_root: Path,
    ) -> CameraFrameArtifacts:
        if capture.camera_id != expected_camera_id:
            raise ValueError(
                f"capture returned camera {capture.camera_id!r}; expected {expected_camera_id!r}"
            )
        _validate_capture(capture)
        camera_root = stage_root / capture.camera_id
        rgb_path = _atomic_write_bytes(camera_root / "rgb.png", capture.rgb_png)

        depth_buffer = io.BytesIO()
        np.save(depth_buffer, np.asarray(capture.depth_m, dtype=np.float32), allow_pickle=False)
        depth_path = _atomic_write_bytes(camera_root / "depth.npy", depth_buffer.getvalue())

        calibration_path = atomic_write_json(
            camera_root / "camera.json",
            {
                "camera_id": capture.camera_id,
                "view_description": capture.view_description,
                "depth_unit": "meter",
                "intrinsics": [list(row) for row in capture.intrinsics],
                "extrinsics": [list(row) for row in capture.extrinsics],
                "image_shape": list(np.asarray(capture.depth_m).shape),
                "rgb_sha256": sha256_file(rgb_path),
                "depth_sha256": sha256_file(depth_path),
            },
        )
        return CameraFrameArtifacts(
            camera_id=capture.camera_id,
            view_description=capture.view_description,
            rgb_path=rgb_path.relative_to(self.output_root).as_posix(),
            depth_path=depth_path.relative_to(self.output_root).as_posix(),
            calibration_path=calibration_path.relative_to(self.output_root).as_posix(),
            rgb_sha256=sha256_file(rgb_path),
            depth_sha256=sha256_file(depth_path),
            calibration_sha256=sha256_file(calibration_path),
        )


def prompt_observation_payload(
    observations: Sequence[ViLaInObservation],
) -> dict[str, object]:
    """Return only image ordering and public view descriptions for prompting."""

    if not observations:
        raise ValueError("at least one observation is required")
    domain = observations[0].domain
    mode = observations[0].observation_mode
    if any(item.domain != domain or item.observation_mode != mode for item in observations):
        raise ValueError("all observations must share a domain and observation mode")
    return {
        "domain": domain,
        "observation_mode": mode,
        "stages": [
            {
                "stage_id": observation.stage_id,
                "inspection_ordinal": observation.inspection_ordinal,
                "opened_region_id": observation.opened_region_id,
                "images": [
                    {
                        "camera_id": frame.camera_id,
                        "view_description": frame.view_description,
                        "rgb_path": frame.rgb_path,
                        "rgb_sha256": frame.rgb_sha256,
                    }
                    for frame in observation.camera_frames
                ],
            }
            for observation in observations
        ],
    }


def _validate_capture(capture: CameraFrameCapture) -> None:
    _validate_component(capture.camera_id, "camera ID")
    if not capture.view_description.strip():
        raise ValueError("view description must not be empty")
    if not capture.rgb_png:
        raise ValueError("RGB PNG payload must not be empty")
    depth = np.asarray(capture.depth_m)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("metric depth must be a non-empty two-dimensional array")
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError("metric depth must be numeric")
    if np.any(np.isfinite(depth) & (depth < 0)):
        raise ValueError("metric depth must not contain negative finite values")
    if _matrix_shape(capture.intrinsics) != (3, 3):
        raise ValueError("intrinsics must be a 3x3 matrix")
    if _matrix_shape(capture.extrinsics) != (4, 4):
        raise ValueError("extrinsics must be a 4x4 matrix")


def _matrix_shape(matrix: tuple[tuple[float, ...], ...]) -> tuple[int, int]:
    if not matrix:
        return (0, 0)
    widths = {len(row) for row in matrix}
    return (len(matrix), widths.pop()) if len(widths) == 1 else (len(matrix), -1)


def _observation_hash(
    stage_id: str,
    opened_region_id: str | None,
    frames: Sequence[CameraFrameArtifacts],
) -> str:
    material = {
        "stage_id": stage_id,
        "opened_region_id": opened_region_id,
        "frames": [
            {
                "camera_id": frame.camera_id,
                "rgb_sha256": frame.rgb_sha256,
                "depth_sha256": frame.depth_sha256,
                "calibration_sha256": frame.calibration_sha256,
            }
            for frame in frames
        ],
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _validate_component(value: str, label: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} is not a safe path component: {value!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
