"""Pure, measurement-source-independent Phase-1 relation predicates."""

from __future__ import annotations

from typing import Any


def evaluate_insertable_in(tool_cross_section_m: float | None,
                           target_opening_width_m: float | None,
                           clearance_margin_m: float) -> dict[str, Any]:
    if tool_cross_section_m is None or target_opening_width_m is None:
        return {"status": "UNKNOWN", "pass_margin_m": None,
                "reason": "REQUIRED_MEASUREMENT_MISSING"}
    margin = float(target_opening_width_m) - float(tool_cross_section_m) - float(clearance_margin_m)
    return {"status": "TRUE" if margin > 0 else "FALSE",
            "pass_margin_m": margin,
            "evaluated_inequality": "tool_cross_section_m + clearance_margin_m < target_opening_width_m",
            "tool_cross_section_m": float(tool_cross_section_m),
            "target_opening_width_m": float(target_opening_width_m),
            "clearance_margin_m": float(clearance_margin_m)}


def evaluate_reaches_bottom(tool_usable_length_m: float | None,
                            target_cavity_depth_m: float | None,
                            grip_allowance_m: float) -> dict[str, Any]:
    if tool_usable_length_m is None or target_cavity_depth_m is None:
        return {"status": "UNKNOWN", "pass_margin_m": None,
                "reason": "REQUIRED_MEASUREMENT_MISSING"}
    margin = float(tool_usable_length_m) - float(grip_allowance_m) - float(target_cavity_depth_m)
    return {"status": "TRUE" if margin >= 0 else "FALSE",
            "pass_margin_m": margin,
            "evaluated_inequality": "tool_usable_length_m - grip_allowance_m >= target_cavity_depth_m",
            "tool_usable_length_m": float(tool_usable_length_m),
            "target_cavity_depth_m": float(target_cavity_depth_m),
            "grip_allowance_m": float(grip_allowance_m)}
