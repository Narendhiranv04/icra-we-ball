"""Stable short labels for the final-paper scene variants."""

from __future__ import annotations


VARIANT_LABELS: dict[str, tuple[str, ...]] = {
    "kitchen": (
        "F0_ALL_VISIBLE",
        "F1_HIDDEN_COFFEE_VESSEL",
        "F2_HIDDEN_SOUP_BOWL",
        "F3_HIDDEN_VESSELS_MIXED",
        "F4_TOOLS_IN_DRAWERS",
        "F5_FULL_DISTRIBUTED_SEARCH",
        "I0_MISSING_COFFEE_VESSEL",
        "I1_MISSING_SOUP_BOWL",
        "I2_MISSING_COFFEE_SPOON",
        "I3_MISSING_SOUP_UTENSIL",
        "I4_MISSING_KETTLE",
        "I5_MISSING_COFFEE_JAR",
    ),
    "living_room": (
        "F0_ALL_OBJECTS_IN_STAGING",
        "F1_LEFT_SAUCER_PREPLACED",
        "F2_LEFT_SAUCER_ON_SHARED",
        "F3_LEFT_CUP_ON_SHARED",
        "F4_SAUCER_PREPLACED_CUP_ON_SHARED",
        "F5_LEFT_PAIR_ON_SHARED",
        "I0_NO_SHARED_TABLE",
        "I1_NO_LEFT_PERSONAL_TABLE",
        "I2_NO_PERSONAL_TABLES",
        "I3_NO_TABLES",
    ),
    "workshop": (
        "F0_MANUAL_FIRST_ONE_REGION",
        "F1_POWER_FIRST_ONE_REGION",
        "F2_MANUAL_FIRST_TWO_REGIONS",
        "F3_POWER_FIRST_TWO_REGIONS",
        "F4_MANUAL_FIRST_THREE_REGIONS",
        "F5_POWER_FIRST_THREE_REGIONS",
        "F6_MANUAL_ONLY",
        "F7_POWER_ONLY",
        "I0_NO_DRIVER",
        "I1_NO_SCREW",
    ),
}

PREFIXES = {"kitchen": "K", "living_room": "L", "workshop": "W"}

HELDOUT_VARIANT_LABELS: dict[str, tuple[str, ...]] = {
    "kitchen": (
        "HK1_DISPERSED_SEARCH",
        "HK2_SINGLE_REGION_SEARCH",
        "HK3_MULTI_STORAGE_SEARCH",
        "HK4_MISSING_COFFEE_STIRRER",
        "HK5_MISSING_SOUP_BOWL",
    ),
    "living_room": (
        "HL1_RIGHT_SAUCER_PREPLACED",
        "HL2_RIGHT_PAIR_ON_SHARED",
        "HL3_SAUCER_PREPLACED_CUP_ON_SHARED",
        "HL4_NO_RIGHT_PERSONAL_TABLE",
        "HL5_NO_SHARED_TABLE",
    ),
    "workshop": (
        "HW1_SCREW_IN_RIGHT_DRAWER",
        "HW2_POWER_AND_SCREW_RIGHT_DRAWER",
        "HW3_MANUAL_AND_SCREW_CABINET",
        "HW4_NO_SCREW_IN_STORAGE",
        "HW5_NO_DRIVER_IN_STORAGE",
    ),
}

HELDOUT_PREFIXES = {"kitchen": "HK", "living_room": "HL", "workshop": "HW"}


def paper_variant_label(environment: str, internal_variant: str) -> str:
    """Return K1/L1/W1 or HK1/HL1/HW1 style label, preserving unknown test/local variants."""
    variants = VARIANT_LABELS.get(environment, ())
    try:
        return f"{PREFIXES[environment]}{variants.index(internal_variant) + 1}"
    except (KeyError, ValueError):
        pass
    h_variants = HELDOUT_VARIANT_LABELS.get(environment, ())
    try:
        return f"{HELDOUT_PREFIXES[environment]}{h_variants.index(internal_variant) + 1}"
    except (KeyError, ValueError):
        return internal_variant


def resolve_variant_name(environment: str, name: str) -> str:
    """Resolve a short label case-insensitively, or return an internal ID."""
    variants = VARIANT_LABELS.get(environment)
    if variants is None:
        raise ValueError(f"Unknown environment: {environment}")
    normalized = name.strip().upper()
    h_prefix = HELDOUT_PREFIXES.get(environment, "")
    if h_prefix and normalized.startswith(h_prefix) and normalized[len(h_prefix):].isdigit():
        h_variants = HELDOUT_VARIANT_LABELS.get(environment, ())
        h_index = int(normalized[len(h_prefix):]) - 1
        if 0 <= h_index < len(h_variants):
            return h_variants[h_index]
    prefix = PREFIXES[environment]
    if normalized.startswith(prefix) and normalized[len(prefix):].isdigit():
        index = int(normalized[len(prefix):]) - 1
        if 0 <= index < len(variants):
            return variants[index]
    return name


def variant_mapping(environment: str) -> dict[str, str]:
    """Return short-label to internal-ID mapping in execution order."""
    return {
        paper_variant_label(environment, variant): variant
        for variant in VARIANT_LABELS[environment]
    }
