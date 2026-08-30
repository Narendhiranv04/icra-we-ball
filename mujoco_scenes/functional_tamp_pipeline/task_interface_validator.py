"""Canonical G_F task-interface completeness validation.

Ensures that a canonical FunctionalRequirementGraph specifies all grounding-relevant
roles, relations, and operation groups required for the explicit task domain before
downstream perception or symbolic compilation.
"""

from __future__ import annotations

from typing import Any

from .errors import MalformedVLMSpecificationError
from .models import FunctionalRequirementGraph


LIVING_ROLE_EXPECTED_ENTITY_KINDS = {
    "PERSONAL_CUP_SAUCER_REGION": "REGION",
    "SHARED_REMOTE_REGION": "REGION",
    "CUP_SAUCER_SET": "OBJECT",
    "REMOTE": "OBJECT",
    "SEATING_POSITION": "FIXED_TARGET",
    "SEATING_PAIR": "FIXED_TARGET",
}

KITCHEN_REQUIRED_ROLES = {
    "coffee_container": "OBJECT",
    "soup_container": "OBJECT",
    "coffee_stirrer": "OBJECT",
    "soup_eating_utensil": "OBJECT",
    "coffee_source": "OBJECT",
    "water_source": "OBJECT",
}

WORKSHOP_REQUIRED_ROLES = {
    "driver": "OBJECT",
    "fastener": "OBJECT",
    "repair_target": "FIXED_TARGET",
}


def validate_kitchen_gf_completeness(graph: FunctionalRequirementGraph) -> None:
    """Validate that Kitchen G_F contains all grounding-relevant roles and operation groups."""
    for role_name, expected_kind in KITCHEN_REQUIRED_ROLES.items():
        if role_name not in graph.nodes:
            raise MalformedVLMSpecificationError(
                f"Kitchen G_F missing required task role {role_name!r}"
            )
        node = graph.nodes[role_name]
        if node.entity_kind != expected_kind:
            raise MalformedVLMSpecificationError(
                f"Kitchen role {role_name!r} must have entity_kind {expected_kind!r}, got {node.entity_kind!r}"
            )

    has_coffee_stir = False
    has_soup_utensil = False
    for grp in graph.operation_groups:
        if not grp.required_relations:
            raise MalformedVLMSpecificationError(
                f"Kitchen operation group {grp.id!r} has empty required_relations"
            )
        if grp.tool_role == "coffee_stirrer" and grp.target_role == "coffee_container":
            has_coffee_stir = True
        if grp.tool_role == "soup_eating_utensil" and grp.target_role == "soup_container":
            has_soup_utensil = True

    if not has_coffee_stir:
        raise MalformedVLMSpecificationError(
            "Kitchen G_F missing required coffee stirring operation group (coffee_stirrer -> coffee_container)"
        )
    if not has_soup_utensil:
        raise MalformedVLMSpecificationError(
            "Kitchen G_F missing required soup utensil operation group (soup_eating_utensil -> soup_container)"
        )


def validate_living_room_gf_completeness(graph: FunctionalRequirementGraph) -> None:
    """Validate that Living Room G_F contains all task anchors, support regions, relations, and groups."""
    for role_name, expected_kind in LIVING_ROLE_EXPECTED_ENTITY_KINDS.items():
        if role_name not in graph.nodes:
            raise MalformedVLMSpecificationError(
                f"Living Room G_F missing required task role {role_name!r}"
            )
        node = graph.nodes[role_name]
        if node.entity_kind != expected_kind:
            raise MalformedVLMSpecificationError(
                f"Living Room role {role_name!r} must have entity_kind {expected_kind!r}, got {node.entity_kind!r}"
            )

    # Validate required shared relations
    fits_on_found = False
    accessible_found = False
    for r in graph.relations:
        if (
            r.subject_role == "SHARED_REMOTE_REGION"
            and r.predicate == "FITS_ON"
            and r.object_role == "REMOTE"
        ):
            fits_on_found = True
        if (
            r.subject_role == "SHARED_REMOTE_REGION"
            and r.predicate == "ACCESSIBLE_FROM_BOTH_SEATS"
            and r.object_role == "SEATING_PAIR"
        ):
            accessible_found = True

    if not fits_on_found:
        raise MalformedVLMSpecificationError(
            "Living Room G_F missing required relation: SHARED_REMOTE_REGION -- FITS_ON --> REMOTE"
        )
    if not accessible_found:
        raise MalformedVLMSpecificationError(
            "Living Room G_F missing required relation: SHARED_REMOTE_REGION -- ACCESSIBLE_FROM_BOTH_SEATS --> SEATING_PAIR"
        )

    # Validate required personal operation group
    personal_grp_found = False
    for grp in graph.operation_groups:
        if not grp.required_relations:
            raise MalformedVLMSpecificationError(
                f"Living Room operation group {grp.id!r} has empty required_relations"
            )
        if (
            grp.tool_role == "PERSONAL_CUP_SAUCER_REGION"
            and grp.target_role == "CUP_SAUCER_SET"
            and grp.context_role == "SEATING_POSITION"
            and grp.usage_policy == "DEDICATED_PER_TARGET"
            and "FITS_SET_ON" in grp.required_relations
            and "NEAR_SEAT" in grp.context_relations
        ):
            personal_grp_found = True

    if not personal_grp_found:
        raise MalformedVLMSpecificationError(
            "Living Room G_F missing required personal support OperationGroup "
            "(tool=PERSONAL_CUP_SAUCER_REGION, target=CUP_SAUCER_SET, context=SEATING_POSITION, "
            "policy=DEDICATED_PER_TARGET, required_relations=[FITS_SET_ON], context_relations=[NEAR_SEAT])"
        )


def validate_workshop_gf_completeness(graph: FunctionalRequirementGraph) -> None:
    """Validate that Workshop G_F contains driver, fastener, repair_target and the 3 core relations."""
    for role_name, expected_kind in WORKSHOP_REQUIRED_ROLES.items():
        if role_name not in graph.nodes:
            raise MalformedVLMSpecificationError(
                f"Workshop G_F missing required task role {role_name!r}"
            )
        node = graph.nodes[role_name]
        if node.entity_kind != expected_kind:
            raise MalformedVLMSpecificationError(
                f"Workshop role {role_name!r} must have entity_kind {expected_kind!r}, got {node.entity_kind!r}"
            )

    has_compat = False
    has_reaches = False
    has_target_compat = False
    for r in graph.relations:
        if r.subject_role == "driver" and r.predicate == "COMPATIBLE_WITH" and r.object_role == "fastener":
            has_compat = True
        if r.subject_role == "driver" and r.predicate == "REACHES_TARGET" and r.object_role == "repair_target":
            has_reaches = True
        if r.subject_role == "fastener" and r.predicate == "COMPATIBLE_WITH_TARGET" and r.object_role == "repair_target":
            has_target_compat = True

    if not has_compat:
        raise MalformedVLMSpecificationError(
            "Workshop G_F missing required relation: driver -- COMPATIBLE_WITH --> fastener"
        )
    if not has_reaches:
        raise MalformedVLMSpecificationError(
            "Workshop G_F missing required relation: driver -- REACHES_TARGET --> repair_target"
        )
    if not has_target_compat:
        raise MalformedVLMSpecificationError(
            "Workshop G_F missing required relation: fastener -- COMPATIBLE_WITH_TARGET --> repair_target"
        )


def validate_canonical_task_interface(graph: FunctionalRequirementGraph) -> None:
    """Validate task-interface completeness for canonical FunctionalRequirementGraph."""
    domain = graph.domain.lower()
    if domain == "kitchen":
        validate_kitchen_gf_completeness(graph)
    elif domain == "living_room":
        validate_living_room_gf_completeness(graph)
    elif domain == "workshop":
        validate_workshop_gf_completeness(graph)
    else:
        raise MalformedVLMSpecificationError(f"Unknown domain for completeness validation: {graph.domain!r}")
