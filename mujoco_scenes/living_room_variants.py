"""Source of truth for the redesigned fixed-table Living Room variants."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


CONFIG = Path(__file__).resolve().parent / "configs" / "living_room_variants.yaml"
PREFIX = "L2_integrated_living_room_region_function_"


def load_living_room_variant_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    variants = payload.get("variants", {})
    if len(variants) != 10:
        raise ValueError("Living Room contract must contain exactly 10 variants")
    for variant_id, spec in variants.items():
        expected_prefix = "F" if spec["intended_outcome"] == "FEASIBLE" else "I"
        if not variant_id.startswith(expected_prefix):
            raise ValueError(f"Outcome/ID mismatch for {variant_id}")
    return deepcopy(payload)


def load_living_room_variants() -> dict[str, dict[str, Any]]:
    return load_living_room_variant_contract()["variants"]


def scene_name(variant_id: str) -> str:
    if variant_id not in load_living_room_variants():
        raise ValueError(f"Unknown Living Room variant: {variant_id}")
    return PREFIX + variant_id
