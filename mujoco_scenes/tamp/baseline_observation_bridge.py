"""Bridge the existing MuJoCo planning observation into TAMP state snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from baseline_common.models import Observation

from .discovery_replanning import CameraObservation, PlanningSnapshot
from .skills import FailureCode, SkillAction, SkillResult
from .state import ObjectObservation, ObservedState, RegionObservation, Relation, RobotObservation


def observation_to_observed_state(
    observation: Observation,
    *,
    camera_visible_object_ids: Iterable[str] | None = None,
) -> ObservedState:
    """Convert a bounded observation without adding unobserved objects.

    When camera evidence is provided, an object must appear in at least one
    selected RGB/segmentation view. A held object remains observable through
    the robot state even if it is outside every camera frame.
    """
    visible_ids = (
        None
        if camera_visible_object_ids is None
        else frozenset(str(item) for item in camera_visible_object_ids)
    )
    robot = dict(observation.robot)
    held_object = _string_or_none(robot.get("holding"))
    objects = {
        entity.entity_id: ObjectObservation(
            entity.entity_id,
            entity.label,
            True,
            _location(entity.facts),
            dict(entity.facts),
        )
        for entity in observation.entities
        if visible_ids is None
        or entity.entity_id in visible_ids
        or entity.entity_id == held_object
    }
    regions = {
        region.region_id: RegionObservation(
            region.region_id,
            region.label,
            True,
            inspected=region.inspected,
            open={"open": True, "closed": False}.get(region.state),
        )
        for region in observation.regions
    }
    relations = tuple(
        Relation(object_id, "in_region", location)
        for object_id, item in objects.items()
        if (location := item.location) and location in regions
    )
    return ObservedState(
        objects,
        regions,
        RobotObservation(
            str(robot.get("workspace") or robot.get("location") or "home"),
            held_object,
        ),
        relations,
        revision=observation.revision,
    )


class BaselineRuntimeSnapshotObserver:
    """Capture a bounded state and fresh encoded RGB views from a runtime."""

    def __init__(self, runtime: Any):
        self.runtime = runtime
        self._known_object_ids: set[str] = set()

    def __call__(self) -> PlanningSnapshot:
        observation, images = self.runtime.observe()
        manifest = getattr(self.runtime, "latest_annotation_manifest", None)
        camera_ids = _camera_visible_object_ids(manifest)
        if _has_camera_evidence(manifest):
            self._known_object_ids.update(camera_ids)
            visible_ids: frozenset[str] | None = frozenset(self._known_object_ids)
        else:
            visible_ids = None
        camera_images = tuple(
            CameraObservation(str(item["camera"]), str(item["data_url"]))
            for item in images
            if isinstance(item, dict)
            and isinstance(item.get("camera"), str)
            and isinstance(item.get("data_url"), str)
        )
        return PlanningSnapshot(
            observation_to_observed_state(
                observation,
                camera_visible_object_ids=visible_ids,
            ),
            camera_images,
        )


def _has_camera_evidence(manifest: object) -> bool:
    return (
        isinstance(manifest, Mapping)
        and isinstance(manifest.get("cameras"), Mapping)
        and bool(manifest["cameras"])
    )


def _camera_visible_object_ids(manifest: object) -> frozenset[str]:
    """Read only object IDs with non-empty rendered screen-space evidence."""
    if not isinstance(manifest, Mapping):
        return frozenset()
    cameras = manifest.get("cameras", {})
    if not isinstance(cameras, Mapping):
        return frozenset()
    visible: set[str] = set()
    for camera in cameras.values():
        if not isinstance(camera, Mapping):
            continue
        rows = camera.get("objects", ())
        if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, Mapping)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            object_id = row.get("id")
            pixels = row.get("pixel_count")
            if isinstance(object_id, str) and object_id and isinstance(pixels, int) and pixels > 0:
                visible.add(object_id)
    return frozenset(visible)


def observed_skill_precheck(
    action: SkillAction, state: ObservedState
) -> SkillResult | None:
    """Reject a stale skill using only the latest bounded observation."""
    name = action.name.upper()
    values = action.arguments
    visible = {
        object_id for object_id, observation in state.objects.items() if observation.visible
    }
    held = state.robot.held_object

    def required(key: str) -> str | None:
        value = values.get(key)
        return value if isinstance(value, str) and value else None

    if name == "PICK":
        object_id = required("object_id")
        if object_id not in visible:
            return SkillResult.failed(
                FailureCode.OBJECT_NOT_VISIBLE,
                f"{object_id or 'object'} is not visible in the current camera observation",
            )
        if held is not None:
            return SkillResult.failed(
                FailureCode.PRECONDITION_FAILED,
                f"Cannot pick while holding {held}",
            )
    elif name == "PLACE":
        object_id = required("object_id")
        destination = required("region_id")
        if held != object_id:
            return SkillResult.failed(
                FailureCode.PRECONDITION_FAILED,
                f"PLACE requires holding {object_id!r}, but holding {held!r}",
            )
        if destination not in visible and destination not in state.regions:
            return SkillResult.failed(
                FailureCode.OBJECT_NOT_VISIBLE,
                f"PLACE destination {destination!r} is not currently known",
            )
    elif name in {"POUR", "STIR"}:
        tool_key = "source_id" if name == "POUR" else "tool_id"
        target_id = required("target_id")
        tool_id = required(tool_key)
        if tool_id != held:
            return SkillResult.failed(
                FailureCode.PRECONDITION_FAILED,
                f"{name} requires holding {tool_id!r}, but holding {held!r}",
            )
        if target_id not in visible:
            return SkillResult.failed(
                FailureCode.OBJECT_NOT_VISIBLE,
                f"{name} target {target_id!r} is not visible in the current camera observation",
            )
    elif name == "INSPECT":
        region_id = required("region_id")
        if held is not None:
            return SkillResult.failed(
                FailureCode.PRECONDITION_FAILED,
                f"Cannot inspect {region_id!r} while holding {held}",
            )
        if region_id not in state.regions:
            return SkillResult.failed(
                FailureCode.OBJECT_NOT_VISIBLE,
                f"INSPECT region {region_id!r} is not currently known",
            )
    return None


def _location(facts: object) -> str | None:
    if not isinstance(facts, dict):
        return None
    return _string_or_none(facts.get("region_id") or facts.get("source_region") or facts.get("location"))


def _string_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
