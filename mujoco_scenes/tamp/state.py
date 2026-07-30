"""Observation-bounded symbolic state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    category: str
    visible: bool
    location: str | None = None
    facts: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.object_id,
            "category": self.category,
            "visible": self.visible,
            "location": self.location,
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class RegionObservation:
    region_id: str
    category: str
    visible: bool
    inspected: bool = False
    open: bool | None = None
    occupied_by: tuple[str, ...] | None = None
    facts: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.region_id,
            "category": self.category,
            "visible": self.visible,
            "inspected": self.inspected,
            "open": self.open,
            "occupied_by": (
                list(self.occupied_by)
                if self.occupied_by is not None
                else None
            ),
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class RobotObservation:
    location: str
    held_object: str | None = None
    motion_ready: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "location": self.location,
            "held_object": self.held_object,
            "motion_ready": self.motion_ready,
        }


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    status: str = "true"

    def as_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "status": self.status,
        }


@dataclass(frozen=True)
class ObservedState:
    objects: Mapping[str, ObjectObservation]
    regions: Mapping[str, RegionObservation]
    robot: RobotObservation
    relations: tuple[Relation, ...] = ()
    revision: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "robot": self.robot.as_dict(),
            "objects": {
                key: value.as_dict() for key, value in self.objects.items()
            },
            "regions": {
                key: value.as_dict() for key, value in self.regions.items()
            },
            "relations": [relation.as_dict() for relation in self.relations],
        }

    def visible_object(self, object_id: str) -> ObjectObservation | None:
        observation = self.objects.get(object_id)
        return observation if observation and observation.visible else None
