"""Formal System Context and Planning Constants Registry for Phase 3.

Distinguishes four formal architectural categories across domains:
1. SELECTABLE_FUNCTIONAL_ASSET: Represented by G_F functional roles, filled by ground_graph selection into phi*.
2. SYSTEM_FIXED_FUNCTIONAL_ANCHOR: Participates in task relations as FIXED_TARGET; identity and geometry supplied by scene context.
3. PLANNER_CONTEXT_CONSTANT: Non-phi* constants used deterministically by symbolic compilers and planners.
4. SEARCH_REGION: Environment inspection and search containers.
"""

from __future__ import annotations

from typing import Any, Iterable

PLANNER_CONTEXT_CONSTANTS: dict[str, frozenset[str]] = {
    "kitchen": frozenset({"countertop", "serving_area"}),
    "living_room": frozenset({"staging_tray"}),
    "workshop": frozenset({"MAIN_WORKBENCH_ZONE", "workshop_frame_joint", "work_surface"}),
}

SYSTEM_FIXED_ANCHORS: dict[str, frozenset[str]] = {
    "kitchen": frozenset(),
    "living_room": frozenset({"SEATING_POSITION", "SEATING_PAIR"}),
    "workshop": frozenset({"repair_target"}),
}

SEARCH_REGIONS: dict[str, frozenset[str]] = {
    "kitchen": frozenset({"D1", "D2", "C1", "C2", "B1"}),
    "living_room": frozenset(),
    "workshop": frozenset({"LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"}),
}


def get_domain_planner_context_constants(domain: str) -> frozenset[str]:
    """Return immutable set of registered symbolic planner constants for domain."""
    return PLANNER_CONTEXT_CONSTANTS.get(domain.strip().lower(), frozenset())


def get_domain_system_fixed_anchors(domain: str) -> frozenset[str]:
    """Return immutable set of registered system-owned fixed functional anchors for domain."""
    return SYSTEM_FIXED_ANCHORS.get(domain.strip().lower(), frozenset())


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

    # 4. Explicitly allowed context IDs (restricted: for standard domains, must be a known domain constant or node)
    if allowed_context_ids is not None:
        allowed_set = set(allowed_context_ids)
        if argument in allowed_set:
            d_norm = domain.strip().lower()
            if d_norm in {"kitchen", "living_room", "workshop"}:
                if argument in domain_constants or (hasattr(graph_o, "nodes") and argument in graph_o.nodes):
                    return True
            else:
                # Custom/test domain
                return True

    return False
