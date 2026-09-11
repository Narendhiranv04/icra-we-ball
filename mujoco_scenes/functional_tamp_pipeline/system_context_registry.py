"""Formal System Context and Planning Constants Registry for Phase 3.

Distinguishes four formal architectural categories across domains:
1. SELECTABLE_FUNCTIONAL_ASSET: Represented by G_F functional roles, filled by ground_graph selection into phi*.
2. SYSTEM_FIXED_FUNCTIONAL_ANCHOR: Participates in task relations as FIXED_TARGET; identity and geometry supplied by scene context.
3. PLANNER_CONTEXT_CONSTANT: Non-phi* constants used deterministically by symbolic compilers and planners.
4. SEARCH_REGION: Environment inspection and search containers.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable


_SELECTABLE_ROLES_RAW: dict[str, frozenset[str]] = {
    "kitchen": frozenset({
        "coffee_container",
        "soup_container",
        "coffee_stirrer",
        "soup_eating_utensil",
        "water_source",
        "coffee_source",
    }),
    "living_room": frozenset({
        "PERSONAL_CUP_SAUCER_REGION",
        "SHARED_REMOTE_REGION",
        "CUP_SAUCER_SET",
        "REMOTE",
    }),
    "workshop": frozenset({
        "driver",
        "fastener",
    }),
}

_SYSTEM_FIXED_ANCHORS_RAW: dict[str, frozenset[str]] = {
    "kitchen": frozenset(),
    "living_room": frozenset({"SEATING_POSITION", "SEATING_PAIR"}),
    "workshop": frozenset({"repair_target"}),
}

_PLANNER_CONTEXT_CONSTANTS_RAW: dict[str, frozenset[str]] = {
    "kitchen": frozenset({"countertop", "serving_area", "dining_table"}),
    "living_room": frozenset({"staging_tray"}),
    "workshop": frozenset({"MAIN_WORKBENCH_ZONE", "workshop_frame_joint"}),
}

# The entity kind each runtime role is, in the runtime's own representation:
# OBJECT for something the robot can pick up and carry, REGION for an area or
# surface that receives things, FIXED_TARGET for a calibrated reference point.
#
# This says what the runtime is, not what any task requires.  It exists because
# the wire contract asks the model for the same distinction about every
# participant it declares, and without it the two statements could not be read
# against each other.  A participant the model declared movable was canonicalized
# onto a region of the room whenever the categories it listed happened to read
# like a surface -- "refreshment setting", declared OBJECT, came back as the side
# table it belongs on -- and the payload then disappeared from the task.
_ROLE_ENTITY_KINDS_RAW: dict[str, dict[str, str]] = {
    "kitchen": {
        "coffee_container": "OBJECT",
        "soup_container": "OBJECT",
        "coffee_stirrer": "OBJECT",
        "soup_eating_utensil": "OBJECT",
        "water_source": "OBJECT",
        "coffee_source": "OBJECT",
        "countertop": "REGION",
        "serving_area": "REGION",
        "dining_table": "REGION",
    },
    "living_room": {
        "PERSONAL_CUP_SAUCER_REGION": "REGION",
        "SHARED_REMOTE_REGION": "REGION",
        "CUP_SAUCER_SET": "OBJECT",
        "REMOTE": "OBJECT",
        "SEATING_POSITION": "FIXED_TARGET",
        "SEATING_PAIR": "FIXED_TARGET",
        "staging_tray": "REGION",
    },
    "workshop": {
        "driver": "OBJECT",
        "fastener": "OBJECT",
        "repair_target": "FIXED_TARGET",
        "MAIN_WORKBENCH_ZONE": "REGION",
        "workshop_frame_joint": "FIXED_TARGET",
    },
}

# The one contradiction the wire prompt states sharply enough to read against
# the runtime: something the robot carries is not an area of the room.  Those two
# answers exclude each other, and each rules the other's roles out.
#
# FIXED_TARGET is deliberately in neither class, so it excludes nothing and is
# excluded by nothing.  The prompt calls it "a fixed reference or interaction
# point", and the model applies that to the same things it calls objects and
# areas: the assembly a screw is driven through is a part one could pick up, and
# the runtime fixes it as a site on the bench, so both answers are honest.
# Treating it as stationary made a workpiece declared OBJECT unable to be the
# repair target, which is the only role in that domain it could be.
CARRIED_ENTITY_KINDS: frozenset[str] = frozenset({"OBJECT"})
AREA_ENTITY_KINDS: frozenset[str] = frozenset({"REGION"})


_SEARCH_REGIONS_RAW: dict[str, frozenset[str]] = {
    "kitchen": frozenset({"D1", "D2", "C1", "C2", "B1"}),
    "living_room": frozenset(),
    "workshop": frozenset({"LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"}),
}

# Authoritative read-only registry mappings
SELECTABLE_ROLES: MappingProxyType[str, frozenset[str]] = MappingProxyType(_SELECTABLE_ROLES_RAW)
SYSTEM_FIXED_ANCHORS: MappingProxyType[str, frozenset[str]] = MappingProxyType(_SYSTEM_FIXED_ANCHORS_RAW)
PLANNER_CONTEXT_CONSTANTS: MappingProxyType[str, frozenset[str]] = MappingProxyType(_PLANNER_CONTEXT_CONSTANTS_RAW)
SEARCH_REGIONS: MappingProxyType[str, frozenset[str]] = MappingProxyType(_SEARCH_REGIONS_RAW)


def get_domain_selectable_roles(domain: str) -> frozenset[str]:
    """Return immutable set of registered selectable functional role names for domain."""
    return SELECTABLE_ROLES.get(domain.strip().lower(), frozenset())


def get_domain_system_fixed_anchors(domain: str) -> frozenset[str]:
    """Return immutable set of registered system-owned fixed functional anchors for domain."""
    return SYSTEM_FIXED_ANCHORS.get(domain.strip().lower(), frozenset())


def get_domain_planner_context_constants(domain: str) -> frozenset[str]:
    """Return immutable set of registered symbolic planner constants for domain."""
    return PLANNER_CONTEXT_CONSTANTS.get(domain.strip().lower(), frozenset())


def get_runtime_role_entity_kind(domain: str, role: str) -> str | None:
    """The entity kind the runtime represents ``role`` as, or None if unregistered."""
    return _ROLE_ENTITY_KINDS_RAW.get(domain.strip().lower(), {}).get(str(role))


def runtime_roles_compatible_with_entity_kind(domain: str, entity_kind: str) -> frozenset[str]:
    """Runtime roles whose own entity kind does not contradict ``entity_kind``.

    The only contradiction is carried against area; see the classes above for
    why FIXED_TARGET is in neither.  A role whose kind the runtime never
    declared is compatible with everything, so an unregistered role is never
    excluded by this.
    """
    declared = _ROLE_ENTITY_KINDS_RAW.get(domain.strip().lower(), {})
    kind = str(entity_kind or "").strip().upper()
    if kind in CARRIED_ENTITY_KINDS:
        opposite = AREA_ENTITY_KINDS
    elif kind in AREA_ENTITY_KINDS:
        opposite = CARRIED_ENTITY_KINDS
    else:
        return frozenset(declared)
    return frozenset(name for name, own in declared.items() if own not in opposite)


def get_domain_search_regions(domain: str) -> frozenset[str]:
    """Return immutable set of registered search regions for domain."""
    return SEARCH_REGIONS.get(domain.strip().lower(), frozenset())


def is_valid_planner_argument(
    domain: str,
    argument: str,
    graph_o: Any,
    assigned_object_ids: set[str],
    allowed_context_ids: Iterable[str] | None = None,
) -> bool:
    """Validate whether an action argument is grounded in perception or registered domain context."""
    # 1. Selectable physical object assigned in phi* or component payload
    if argument in assigned_object_ids:
        return True

    # 2. Actual observed scene graph node of context entity_kind (REGION or FIXED_TARGET)
    if hasattr(graph_o, "nodes") and argument in graph_o.nodes:
        node = graph_o.nodes[argument]
        if getattr(node, "entity_kind", "") in {"REGION", "FIXED_TARGET"}:
            return True

    # 3. Registered domain planner context constant
    domain_constants = get_domain_planner_context_constants(domain)
    if argument in domain_constants:
        return True

    # 4. Registered domain search region
    domain_search_regions = get_domain_search_regions(domain)
    if argument in domain_search_regions:
        return True

    # 5. Explicitly allowed context IDs (restricted: for standard domains, must be a known domain constant,
    # registered search region, or actual observed REGION/FIXED_TARGET node; NEVER an unassigned OBJECT)
    if allowed_context_ids is not None:
        allowed_set = set(allowed_context_ids)
        if argument in allowed_set:
            d_norm = domain.strip().lower()
            if d_norm in {"kitchen", "living_room", "workshop"}:
                if argument in domain_constants or argument in domain_search_regions:
                    return True
                if hasattr(graph_o, "nodes") and argument in graph_o.nodes:
                    node = graph_o.nodes[argument]
                    if getattr(node, "entity_kind", "") in {"REGION", "FIXED_TARGET"}:
                        return True
            else:
                # Custom/test domain
                return True

    return False
