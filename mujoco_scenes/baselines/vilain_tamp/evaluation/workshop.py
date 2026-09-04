"""Hidden terminal requirements for the workshop benchmark."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import (
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    check,
    effect_exists,
    physical_on,
    required_sequence,
    required_string,
)


def evaluate_workshop_requirements(
    terminal_state: TerminalStateSnapshot,
    effect_ledger: Sequence[Mapping[str, Any]],
    hidden_context: HiddenBenchmarkContext,
) -> tuple[Mapping[str, Any], ...]:
    """Check compatibility, insertion geometry, repair, and driver placement."""
    requirements = hidden_context.requirements
    compatible_drivers = required_sequence(requirements, "compatible_drivers")
    compatible_fasteners = required_sequence(requirements, "compatible_fasteners")
    target = required_string(requirements, "target")
    workbench = required_string(requirements, "workbench")
    expected_driver = _first_compatible_driver(requirements, compatible_drivers)
    used_driver = _measurement_string(terminal_state, "used_driver")
    used_fastener = _measurement_string(terminal_state, "used_fastener")
    used_target = _measurement_string(terminal_state, "used_target")
    insertion = terminal_state.relations.get("insertion", {})
    if not isinstance(insertion, Mapping):
        insertion = {}

    min_depth = float(requirements.get("minimum_insertion_depth_m", 0.012))
    max_depth = float(requirements.get("maximum_insertion_depth_m", 0.018))
    radial_tolerance = float(requirements.get("radial_tolerance_m", 0.004))
    orientation_tolerance = float(
        requirements.get("orientation_tolerance_rad", 0.03)
    )
    insertion_geometry = bool(
        insertion.get("fastener") == used_fastener
        and insertion.get("target") == target
        and float(insertion.get("radial_error_m", float("inf")))
        <= radial_tolerance
        and min_depth
        <= float(insertion.get("depth_m", float("-inf")))
        <= max_depth
        and float(insertion.get("orientation_error_rad", float("inf")))
        <= orientation_tolerance
        and insertion.get("head_above_tip") is True
    )
    compatible_tuple = bool(
        used_driver in compatible_drivers
        and used_fastener in compatible_fasteners
        and used_target == target
    )
    repaired = terminal_state.measurements.get("joint_repaired") is True
    certified_drive = bool(
        used_driver
        and used_fastener
        and used_target
        and effect_exists(
            effect_ledger,
            "DRIVE_COMPLETED",
            (used_driver, used_fastener, used_target),
        )
    )
    driver_safe = bool(
        used_driver
        and physical_on(terminal_state.objects.get(used_driver, {}), workbench)
    )
    first_encountered = expected_driver is not None and used_driver == expected_driver

    return (
        check("compatible_driver_fastener_target_tuple", compatible_tuple),
        check("fastener_insertion_geometry_valid", insertion_geometry),
        check("joint_physically_repaired", repaired and certified_drive),
        check("first_compatible_driver_used", first_encountered),
        check("driver_left_safely_on_main_workbench", driver_safe),
        check("no_object_held", not terminal_state.held_objects),
    )


def _measurement_string(snapshot: TerminalStateSnapshot, key: str) -> str | None:
    value = snapshot.measurements.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _first_compatible_driver(
    requirements: Mapping[str, Any], compatible_drivers: Sequence[str]
) -> str | None:
    """Resolve first compatible strictly from the fixed region traversal."""
    order = requirements.get("inspection_order")
    contents = requirements.get("storage_contents")
    if isinstance(order, (list, tuple)) and isinstance(contents, Mapping):
        compatible = set(compatible_drivers)
        for region in order:
            region_contents = contents.get(str(region), ())
            if not isinstance(region_contents, (list, tuple)):
                continue
            for object_id in region_contents:
                if str(object_id) in compatible:
                    return str(object_id)
        return None
    configured = requirements.get("first_compatible_driver")
    if isinstance(configured, str) and configured.strip():
        return configured
    return None
