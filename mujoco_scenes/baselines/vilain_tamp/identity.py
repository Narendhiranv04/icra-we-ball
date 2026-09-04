"""Baseline-only association of visual object estimates to scene entities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .contracts import ObjectEstimate, ObjectEstimateStatus, SerializableContract


class IdentityResolutionError(RuntimeError):
    """Raised when association is unresolved or geometrically ambiguous."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        object_ids: Sequence[str] = (),
        candidate_entities: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.object_ids = tuple(object_ids)
        self.candidate_entities = tuple(candidate_entities)


class IdentityInputError(ValueError):
    """Raised when association receives anything except baseline evidence."""


@dataclass(frozen=True)
class EntityCandidate(SerializableContract):
    entity_name: str
    broad_class: str
    compatible_pddl_types: tuple[str, ...]
    centroid_m: tuple[float, float, float]
    aabb_min_m: tuple[float, float, float]
    aabb_max_m: tuple[float, float, float]
    visible_stage_ids: tuple[str, ...]
    movable: bool
    evidence_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entity_name.strip() or not self.broad_class.strip():
            raise ValueError("entity name and broad class must not be empty")
        if not self.compatible_pddl_types:
            raise ValueError("candidate must expose at least one compatible PDDL type")
        vectors = (self.centroid_m, self.aabb_min_m, self.aabb_max_m)
        values = tuple(value for vector in vectors for value in vector)
        if any(len(vector) != 3 for vector in vectors) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("candidate geometry must contain finite 3-D values")
        if any(lower > upper for lower, upper in zip(self.aabb_min_m, self.aabb_max_m)):
            raise ValueError("candidate AABB minimum must not exceed its maximum")


@dataclass(frozen=True)
class EntityBinding(SerializableContract):
    object_id: str
    entity_name: str
    pddl_type: str
    broad_class: str
    centroid_distance_m: float | None
    aabb_distance_m: float | None
    observed_centroid_m: tuple[float, float, float] | None
    entity_centroid_m: tuple[float, float, float] | None
    entity_aabb_min_m: tuple[float, float, float] | None
    entity_aabb_max_m: tuple[float, float, float] | None
    binding_method: str
    confidence: float
    observation_stage_ids: tuple[str, ...]
    evidence_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class IdentityResolutionResult(SerializableContract):
    bindings: tuple[EntityBinding, ...]
    maximum_distance_m: float
    ambiguity_margin_m: float

    def by_object_id(self) -> dict[str, EntityBinding]:
        return {binding.object_id: binding for binding in self.bindings}


class BaselineIdentityResolver:
    """Perform deterministic one-to-one class and centroid association."""

    def __init__(
        self,
        *,
        maximum_distance_m: float,
        ambiguity_margin_m: float,
    ) -> None:
        if maximum_distance_m <= 0:
            raise ValueError("maximum_distance_m must be greater than zero")
        if ambiguity_margin_m < 0 or ambiguity_margin_m >= maximum_distance_m:
            raise ValueError(
                "ambiguity_margin_m must be non-negative and below the distance limit"
            )
        self.maximum_distance_m = maximum_distance_m
        self.ambiguity_margin_m = ambiguity_margin_m

    def resolve(
        self,
        object_estimates: Sequence[ObjectEstimate],
        candidates: Sequence[EntityCandidate],
        *,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> IdentityResolutionResult:
        if external_method_artifacts is not None:
            raise IdentityInputError("external method artifacts are not valid identity input")
        if any(
            isinstance(item, Mapping) and "functional_role" in item
            for item in object_estimates
        ):
            raise IdentityInputError("functional-role fields are forbidden identity input")
        if not all(isinstance(item, ObjectEstimate) for item in object_estimates):
            raise IdentityInputError("identity input must use ObjectEstimate contracts only")
        if not all(isinstance(item, EntityCandidate) for item in candidates):
            raise IdentityInputError("scene input must use EntityCandidate contracts only")
        object_ids = [item.object_id for item in object_estimates]
        entity_names = [item.entity_name for item in candidates]
        if len(object_ids) != len(set(object_ids)):
            raise IdentityInputError("object estimates contain duplicate IDs")
        if len(entity_names) != len(set(entity_names)):
            raise IdentityInputError("scene candidates contain duplicate entity names")
        if not object_estimates:
            return IdentityResolutionResult(
                bindings=(),
                maximum_distance_m=self.maximum_distance_m,
                ambiguity_margin_m=self.ambiguity_margin_m,
            )

        edges: dict[str, tuple[tuple[float, EntityCandidate], ...]] = {}
        estimates_by_id = {item.object_id: item for item in object_estimates}
        for estimate in sorted(object_estimates, key=lambda item: item.object_id):
            if (
                estimate.status is not ObjectEstimateStatus.OBSERVED
                or estimate.estimated_centroid_m is None
            ):
                raise IdentityResolutionError(
                    "UNRESOLVED_ENTITY",
                    f"{estimate.object_id!r} has no usable observed centroid",
                    object_ids=(estimate.object_id,),
                )
            if len(estimate.estimated_centroid_m) != 3 or not all(
                math.isfinite(value) for value in estimate.estimated_centroid_m
            ):
                raise IdentityInputError(
                    f"{estimate.object_id!r} has invalid observed centroid geometry"
                )
            compatible = []
            for candidate in candidates:
                if estimate.pddl_type.lower() not in {
                    value.lower() for value in candidate.compatible_pddl_types
                }:
                    continue
                if not set(estimate.observation_stage_ids).intersection(
                    candidate.visible_stage_ids
                ):
                    continue
                distance = _distance(estimate.estimated_centroid_m, candidate.centroid_m)
                aabb_distance = _point_aabb_distance(
                    estimate.estimated_centroid_m,
                    candidate.aabb_min_m,
                    candidate.aabb_max_m,
                )
                if (
                    distance <= self.maximum_distance_m
                    and aabb_distance <= self.maximum_distance_m
                ):
                    compatible.append((distance, candidate))
            compatible.sort(key=lambda item: (item[0], item[1].entity_name))
            if not compatible:
                raise IdentityResolutionError(
                    "UNRESOLVED_ENTITY",
                    f"no class-compatible visible entity is close enough to {estimate.object_id!r}",
                    object_ids=(estimate.object_id,),
                )
            if (
                len(compatible) > 1
                and compatible[1][0] - compatible[0][0] <= self.ambiguity_margin_m
            ):
                raise IdentityResolutionError(
                    "AMBIGUOUS_ENTITY",
                    f"multiple entities are geometrically indistinguishable for {estimate.object_id!r}",
                    object_ids=(estimate.object_id,),
                    candidate_entities=(
                        compatible[0][1].entity_name,
                        compatible[1][1].entity_name,
                    ),
                )
            edges[estimate.object_id] = tuple(compatible)

        assignments = _complete_assignments(edges, limit=2)
        if not assignments:
            raise IdentityResolutionError(
                "UNRESOLVED_ONE_TO_ONE",
                "no complete one-to-one entity association exists",
                object_ids=tuple(sorted(object_ids)),
            )
        best_cost, best = assignments[0]
        if (
            len(assignments) > 1
            and assignments[1][0] - best_cost <= self.ambiguity_margin_m
        ):
            differing = tuple(
                object_id
                for object_id in sorted(best)
                if best[object_id].entity_name
                != assignments[1][1][object_id].entity_name
            )
            raise IdentityResolutionError(
                "AMBIGUOUS_ONE_TO_ONE",
                "multiple one-to-one associations are within the ambiguity margin",
                object_ids=differing,
                candidate_entities=tuple(
                    sorted(
                        {
                            best[item].entity_name for item in differing
                        }
                        | {
                            assignments[1][1][item].entity_name for item in differing
                        }
                    )
                ),
            )

        bindings = []
        for object_id in sorted(best):
            estimate = estimates_by_id[object_id]
            candidate = best[object_id]
            distance = _distance(estimate.estimated_centroid_m, candidate.centroid_m)
            aabb_distance = _point_aabb_distance(
                estimate.estimated_centroid_m,
                candidate.aabb_min_m,
                candidate.aabb_max_m,
            )
            bindings.append(
                EntityBinding(
                    object_id=object_id,
                    entity_name=candidate.entity_name,
                    pddl_type=estimate.pddl_type,
                    broad_class=candidate.broad_class,
                    centroid_distance_m=distance,
                    aabb_distance_m=aabb_distance,
                    observed_centroid_m=estimate.estimated_centroid_m,
                    entity_centroid_m=candidate.centroid_m,
                    entity_aabb_min_m=candidate.aabb_min_m,
                    entity_aabb_max_m=candidate.aabb_max_m,
                    binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
                    confidence=max(0.0, 1.0 - distance / self.maximum_distance_m),
                    observation_stage_ids=estimate.observation_stage_ids,
                    evidence_artifacts=tuple(sorted(set(candidate.evidence_artifacts))),
                )
            )
        return IdentityResolutionResult(
            bindings=tuple(bindings),
            maximum_distance_m=self.maximum_distance_m,
            ambiguity_margin_m=self.ambiguity_margin_m,
        )


def fixed_entity_binding(
    symbolic_id: str,
    entity_name: str,
    *,
    broad_class: str,
    evidence_artifacts: Sequence[str] = (),
) -> EntityBinding:
    """Represent a public fixed-location identity for execution projection."""
    if not symbolic_id.strip() or not entity_name.strip() or not broad_class.strip():
        raise ValueError("fixed entity identity fields must not be empty")
    return EntityBinding(
        object_id=symbolic_id,
        entity_name=entity_name,
        pddl_type="fixed-location",
        broad_class=broad_class,
        centroid_distance_m=None,
        aabb_distance_m=None,
        observed_centroid_m=None,
        entity_centroid_m=None,
        entity_aabb_min_m=None,
        entity_aabb_max_m=None,
        binding_method="PUBLIC_FIXED_ENTITY",
        confidence=1.0,
        observation_stage_ids=(),
        evidence_artifacts=tuple(evidence_artifacts),
    )


def _complete_assignments(
    edges: Mapping[str, Sequence[tuple[float, EntityCandidate]]],
    *,
    limit: int,
) -> list[tuple[float, dict[str, EntityCandidate]]]:
    ordered_ids = sorted(edges, key=lambda object_id: (len(edges[object_id]), object_id))
    results: list[tuple[float, dict[str, EntityCandidate]]] = []

    def visit(
        ordinal: int,
        used: set[str],
        cost: float,
        current: dict[str, EntityCandidate],
    ) -> None:
        if len(results) >= limit and cost > results[-1][0]:
            return
        if ordinal == len(ordered_ids):
            results.append((cost, dict(current)))
            results.sort(
                key=lambda item: (
                    item[0],
                    tuple(item[1][key].entity_name for key in sorted(item[1])),
                )
            )
            del results[limit:]
            return
        object_id = ordered_ids[ordinal]
        for distance, candidate in edges[object_id]:
            if candidate.entity_name in used:
                continue
            used.add(candidate.entity_name)
            current[object_id] = candidate
            visit(ordinal + 1, used, cost + distance, current)
            del current[object_id]
            used.remove(candidate.entity_name)

    visit(0, set(), 0.0, {})
    return results


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _point_aabb_distance(
    point: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    squared = 0.0
    for value, minimum, maximum in zip(point, lower, upper):
        if value < minimum:
            squared += (minimum - value) ** 2
        elif value > maximum:
            squared += (value - maximum) ** 2
    return math.sqrt(squared)
