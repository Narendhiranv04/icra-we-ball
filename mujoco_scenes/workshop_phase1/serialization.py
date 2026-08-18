"""Serialization utilities and strict anti-leak sanitizers for Workshop Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORBIDDEN_SIMULATOR_SUBSTRINGS = (
    "workshop_long_",
    "workshop_medium_",
    "workshop_short_",
    "workshop_flathead_",
    "workshop_stubby_",
    "workshop_hex_",
    "workshop_power_",
    "workshop_pliers",
    "workshop_combination_",
    "MAIN_WORKBENCH_ZONE",
    "PRIVILEGED_WORKSHOP_ORACLE_SPECS",
    "expected_solution",
)


def sanitize_production_data(data: Any) -> Any:
    """Recursively convert numpy arrays, dataclasses, and primitives to standard JSON-serializable types."""
    if data is None:
        return None
    if isinstance(data, (bool, int, float, str)):
        return data
    if hasattr(data, "to_dict"):
        return sanitize_production_data(data.to_dict())
    if hasattr(data, "__dict__"):
        return sanitize_production_data(vars(data))
    if isinstance(data, dict):
        return {str(k): sanitize_production_data(v) for k, v in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [sanitize_production_data(v) for v in data]
    if hasattr(data, "tolist"):  # numpy array / scalar
        return sanitize_production_data(data.tolist())
    return str(data)


def assert_no_backend_names(data: Any, context_name: str = "production_payload") -> None:
    """Validate that serialized production payload contains zero privileged simulator strings."""
    json_str = json.dumps(sanitize_production_data(data))
    for forbidden in FORBIDDEN_SIMULATOR_SUBSTRINGS:
        if forbidden in json_str:
            raise ValueError(
                f"Privilege leakage detected in {context_name}: found forbidden simulator string '{forbidden}'."
            )


def write_production_json(
    payload: Any,
    output_path: Path,
    verify_no_leaks: bool = True,
) -> None:
    """Write production JSON artifact with leak verification."""
    clean_data = sanitize_production_data(payload)
    if verify_no_leaks:
        assert_no_backend_names(clean_data, context_name=output_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)
