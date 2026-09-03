"""Frozen Canonical Predicate Signature Registry for Phase 3 Functional Grounding.

Defines the single authoritative immutable contract for:
1. Canonical predicate names and domain scoping.
2. Arity and subject/object entity kinds.
3. Canonical role families and directional constraints.
4. Evidence / checker ownership.
5. Active runtime G_F vs legacy observation-diagnostic vs unsupported emittable status.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from .errors import MalformedVLMSpecificationError


class PredicateStatus(str, Enum):
    """Lifecycle status of a domain-scoped predicate."""
    ACTIVE_CANONICAL = "ACTIVE_CANONICAL"
    LEGACY_DIAGNOSTIC = "LEGACY_DIAGNOSTIC"
    CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED = "CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED"


@dataclass(frozen=True)
class PredicateSignature:
    """Immutable signature definition for a domain-scoped functional predicate."""

    domain: str
    name: str
    arity: int
    subject_kinds: tuple[str, ...]
    object_kinds: tuple[str, ...] = ()
    allowed_subject_roles: tuple[str, ...] = ()
    allowed_object_roles: tuple[str, ...] = ()
    evidence_owner: str = ""
    status: PredicateStatus = PredicateStatus.ACTIVE_CANONICAL
    active_in_functional_graph: bool = True
    is_legacy_capability_marker: bool = False
    description: str = ""


_PREDICATE_REGISTRY_RAW: dict[tuple[str, str], PredicateSignature] = {
    # -------------------------------------------------------------------------
    # Kitchen Domain Predicates
    # -------------------------------------------------------------------------
    ("kitchen", "OPEN_CAVITY"): PredicateSignature(
        domain="kitchen",
        name="OPEN_CAVITY",
        arity=1,
        subject_kinds=("OBJECT",),
        allowed_subject_roles=("coffee_container", "soup_container"),
        evidence_owner="GeometricChecker (mesh/pointcloud cavity)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Container has an open upper cavity capable of receiving liquid or items.",
    ),
    ("kitchen", "ELONGATED_OBJECT"): PredicateSignature(
        domain="kitchen",
        name="ELONGATED_OBJECT",
        arity=1,
        subject_kinds=("OBJECT",),
        allowed_subject_roles=("coffee_stirrer", "soup_eating_utensil"),
        evidence_owner="GeometricChecker (aspect ratio)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Object has an elongated aspect ratio suitable for stirring or eating.",
    ),
    ("kitchen", "INSERTABLE_IN"): PredicateSignature(
        domain="kitchen",
        name="INSERTABLE_IN",
        arity=2,
        subject_kinds=("OBJECT",),
        object_kinds=("OBJECT",),
        allowed_subject_roles=("coffee_stirrer", "soup_eating_utensil"),
        allowed_object_roles=("coffee_container", "soup_container"),
        evidence_owner="GeometricChecker (bounding cross-section / convex hull insertion)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Tool object cross-section fits inside the opening of the target container.",
    ),
    ("kitchen", "REACHES_BOTTOM"): PredicateSignature(
        domain="kitchen",
        name="REACHES_BOTTOM",
        arity=2,
        subject_kinds=("OBJECT",),
        object_kinds=("OBJECT",),
        allowed_subject_roles=("coffee_stirrer", "soup_eating_utensil"),
        allowed_object_roles=("coffee_container", "soup_container"),
        evidence_owner="GeometricChecker (length vs container depth clearance)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Tool object length reaches the bottom of the target container with manipulation clearance.",
    ),

    # -------------------------------------------------------------------------
    # Living Room Domain Predicates
    # -------------------------------------------------------------------------
    ("living_room", "PLANAR_SUPPORT"): PredicateSignature(
        domain="living_room",
        name="PLANAR_SUPPORT",
        arity=1,
        subject_kinds=("REGION",),
        allowed_subject_roles=("PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION"),
        evidence_owner="PhysicalSpatialVerifier (surface normal and flatness)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Region provides a horizontal planar surface suitable for supporting items.",
    ),
    ("living_room", "FITS_SET_ON"): PredicateSignature(
        domain="living_room",
        name="FITS_SET_ON",
        arity=2,
        subject_kinds=("REGION",),
        object_kinds=("OBJECT",),
        allowed_subject_roles=("PERSONAL_CUP_SAUCER_REGION",),
        allowed_object_roles=("CUP_SAUCER_SET",),
        evidence_owner="PhysicalSpatialVerifier (surface area and bounds clearance)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Region surface area and boundary support the composite cup and saucer drinkware set.",
    ),
    ("living_room", "FITS_ON"): PredicateSignature(
        domain="living_room",
        name="FITS_ON",
        arity=2,
        subject_kinds=("REGION",),
        object_kinds=("OBJECT",),
        allowed_subject_roles=("SHARED_REMOTE_REGION",),
        allowed_object_roles=("REMOTE",),
        evidence_owner="PhysicalSpatialVerifier (surface area and bounds clearance)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Region surface area and boundary support the handheld remote control device.",
    ),
    ("living_room", "NEAR_SEAT"): PredicateSignature(
        domain="living_room",
        name="NEAR_SEAT",
        arity=2,
        subject_kinds=("REGION",),
        object_kinds=("FIXED_TARGET",),
        allowed_subject_roles=("PERSONAL_CUP_SAUCER_REGION",),
        allowed_object_roles=("SEATING_POSITION",),
        evidence_owner="PhysicalSpatialVerifier (proximity distance to seating anchor)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Region is adjacent / reachable from a specific viewer seating position.",
    ),
    ("living_room", "ACCESSIBLE_FROM_BOTH_SEATS"): PredicateSignature(
        domain="living_room",
        name="ACCESSIBLE_FROM_BOTH_SEATS",
        arity=2,
        subject_kinds=("REGION",),
        object_kinds=("FIXED_TARGET",),
        allowed_subject_roles=("SHARED_REMOTE_REGION",),
        allowed_object_roles=("SEATING_PAIR",),
        evidence_owner="PhysicalSpatialVerifier (dual reach distance to pair of seating anchors)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Region is accessible to seated viewers from both seating positions in the pair.",
    ),

    # -------------------------------------------------------------------------
    # Workshop Domain Predicates
    # -------------------------------------------------------------------------
    ("workshop", "COMPATIBLE_WITH"): PredicateSignature(
        domain="workshop",
        name="COMPATIBLE_WITH",
        arity=2,
        subject_kinds=("OBJECT",),
        object_kinds=("OBJECT",),
        allowed_subject_roles=("driver",),
        allowed_object_roles=("fastener",),
        evidence_owner="WorkshopGeometricGrounder (bit-to-head geometric match)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Driver tool head geometry is compatible with the fastener head recess.",
    ),
    ("workshop", "REACHES_TARGET"): PredicateSignature(
        domain="workshop",
        name="REACHES_TARGET",
        arity=2,
        subject_kinds=("OBJECT",),
        object_kinds=("FIXED_TARGET",),
        allowed_subject_roles=("driver",),
        allowed_object_roles=("repair_target",),
        evidence_owner="WorkshopGeometricGrounder (driver length vs workpiece recess reach)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Driver tool reach is sufficient to access the target repair recess on the workpiece.",
    ),
    ("workshop", "COMPATIBLE_WITH_TARGET"): PredicateSignature(
        domain="workshop",
        name="COMPATIBLE_WITH_TARGET",
        arity=2,
        subject_kinds=("OBJECT",),
        object_kinds=("FIXED_TARGET",),
        allowed_subject_roles=("fastener",),
        allowed_object_roles=("repair_target",),
        evidence_owner="WorkshopGeometricGrounder (fastener thread/diameter match with workpiece hole)",
        status=PredicateStatus.ACTIVE_CANONICAL,
        active_in_functional_graph=True,
        description="Fastener size and threading are compatible with the target repair hole on the workpiece.",
    ),
    ("workshop", "CAN_DRIVE_SCREW"): PredicateSignature(
        domain="workshop",
        name="CAN_DRIVE_SCREW",
        arity=1,
        subject_kinds=("OBJECT",),
        allowed_subject_roles=("driver",),
        evidence_owner="SemanticBelief (detector category matching)",
        status=PredicateStatus.LEGACY_DIAGNOSTIC,
        active_in_functional_graph=False,
        is_legacy_capability_marker=True,
        description="Legacy observation-diagnostic capability marker for driver tools; redundant with semantic role category matching.",
    ),
    ("workshop", "CAN_FASTEN"): PredicateSignature(
        domain="workshop",
        name="CAN_FASTEN",
        arity=1,
        subject_kinds=("OBJECT",),
        allowed_subject_roles=("fastener",),
        evidence_owner="SemanticBelief (detector category matching)",
        status=PredicateStatus.LEGACY_DIAGNOSTIC,
        active_in_functional_graph=False,
        is_legacy_capability_marker=True,
        description="Legacy observation-diagnostic capability marker for fasteners; redundant with semantic role category matching.",
    ),
    # Unsupported canonicalizer-emittable predicates (carried to P3-G for disposition)
    ("workshop", "LOCATED_ON"): PredicateSignature(
        domain="workshop",
        name="LOCATED_ON",
        arity=2,
        subject_kinds=("FIXED_TARGET", "OBJECT"),
        object_kinds=("REGION",),
        evidence_owner="WorkshopSceneContext (support surface placement)",
        status=PredicateStatus.CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED,
        active_in_functional_graph=False,
        description="Emitted by workshop canonicalizer for workbench support; unsupported in canonical G_F (carried to P3-G).",
    ),
    ("workshop", "PLANAR_SUPPORT"): PredicateSignature(
        domain="workshop",
        name="PLANAR_SUPPORT",
        arity=1,
        subject_kinds=("REGION", "OBJECT"),
        evidence_owner="WorkshopSceneContext (support surface flatness)",
        status=PredicateStatus.CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED,
        active_in_functional_graph=False,
        description="Emitted by workshop canonicalizer for support surfaces; unsupported in canonical Workshop G_F (carried to P3-G).",
    ),
    ("workshop", "OPEN_CAVITY"): PredicateSignature(
        domain="workshop",
        name="OPEN_CAVITY",
        arity=1,
        subject_kinds=("OBJECT",),
        evidence_owner="WorkshopSceneContext",
        status=PredicateStatus.CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED,
        active_in_functional_graph=False,
        description="Emitted by workshop canonicalizer for hollow containers; unsupported in canonical Workshop G_F (carried to P3-G).",
    ),
    ("workshop", "ELONGATED_OBJECT"): PredicateSignature(
        domain="workshop",
        name="ELONGATED_OBJECT",
        arity=1,
        subject_kinds=("OBJECT",),
        evidence_owner="WorkshopSceneContext",
        status=PredicateStatus.CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED,
        active_in_functional_graph=False,
        description="Emitted by workshop canonicalizer for slender tools; unsupported in canonical Workshop G_F (carried to P3-G).",
    ),
}

# Authoritative read-only registry mapping
PREDICATE_REGISTRY: MappingProxyType[tuple[str, str], PredicateSignature] = MappingProxyType(
    _PREDICATE_REGISTRY_RAW
)


def get_predicate_signature(domain: str, predicate: str) -> PredicateSignature | None:
    """Retrieve predicate signature by domain and name."""
    return PREDICATE_REGISTRY.get((domain.strip().lower(), predicate.strip()))


def get_active_predicates(domain: str) -> tuple[PredicateSignature, ...]:
    """Return all active functional predicates for a given domain."""
    d_norm = domain.strip().lower()
    return tuple(
        sig for (d, _), sig in PREDICATE_REGISTRY.items()
        if d == d_norm and sig.active_in_functional_graph
    )


def validate_predicate_signature(
    domain: str,
    predicate: str,
    *,
    subject_kind: str,
    object_kind: str | None = None,
    subject_role: str | None = None,
    object_role: str | None = None,
    allow_legacy: bool = False,
) -> None:
    """Validate that a predicate usage complies with the domain's frozen signature."""
    d_norm = domain.strip().lower()
    sig = get_predicate_signature(d_norm, predicate)
    if sig is None:
        raise MalformedVLMSpecificationError(
            f"Unknown predicate {predicate!r} for domain {domain!r}"
        )

    if sig.status == PredicateStatus.CANONICALIZER_EMITTABLE_BUT_UNSUPPORTED:
        raise MalformedVLMSpecificationError(
            f"Predicate {predicate!r} in domain {domain!r} is canonicalizer-emittable but unsupported in canonical G_F"
        )

    if not sig.active_in_functional_graph and not allow_legacy:
        raise MalformedVLMSpecificationError(
            f"Predicate {predicate!r} in domain {domain!r} is a legacy capability marker and not active in canonical G_F"
        )

    # Arity check
    if sig.arity == 1:
        if object_kind is not None or object_role is not None:
            raise MalformedVLMSpecificationError(
                f"Unary predicate {predicate!r} in domain {domain!r} cannot take an object argument"
            )
    elif sig.arity == 2:
        if object_kind is None:
            raise MalformedVLMSpecificationError(
                f"Binary predicate {predicate!r} in domain {domain!r} requires an object argument"
            )

    # Entity kind checks
    if subject_kind not in sig.subject_kinds:
        raise MalformedVLMSpecificationError(
            f"Predicate {predicate!r} in domain {domain!r} expects subject entity_kind in {sig.subject_kinds}, got {subject_kind!r}"
        )
    if sig.arity == 2 and object_kind is not None and object_kind not in sig.object_kinds:
        raise MalformedVLMSpecificationError(
            f"Predicate {predicate!r} in domain {domain!r} expects object entity_kind in {sig.object_kinds}, got {object_kind!r}"
        )

    # Role family checks
    if sig.allowed_subject_roles and subject_role is not None and subject_role not in sig.allowed_subject_roles:
        raise MalformedVLMSpecificationError(
            f"Predicate {predicate!r} in domain {domain!r} expects subject role in {sig.allowed_subject_roles}, got {subject_role!r}"
        )
    if sig.allowed_object_roles and object_role is not None and object_role not in sig.allowed_object_roles:
        raise MalformedVLMSpecificationError(
            f"Predicate {predicate!r} in domain {domain!r} expects object role in {sig.allowed_object_roles}, got {object_role!r}"
        )
