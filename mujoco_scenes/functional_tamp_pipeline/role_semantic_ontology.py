"""System-owned functional role semantic ontology.

Defines the single authoritative source of semantic category acceptance
for functional roles across all domains (Kitchen, Workshop, Living Room).
Consumed by both GT and VLM specification providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION = "phase3_p3i_1_semantic_ontology_v1"

# Shared System Role Semantic Categories
# These are reviewed, system-owned acceptance sets consumed by both GT and VLM providers.
SYSTEM_ROLE_SEMANTIC_CATEGORIES: dict[str, dict[str, tuple[str, ...]]] = {
    "workshop": {
        "driver": (
            "screwdriver",
            "power_driver",
            "power_drill",
            "Phillips screwdriver",
            "cordless power drill",
        ),
        "fastener": (
            "screw",
            "Phillips screw",
            "Phillips head screw",
        ),
        "repair_target": (
            "repair_target",
            "workshop_frame_joint",
            "recess",
        ),
    },
    "kitchen": {
        "coffee_container": ("cup", "mug"),
        "soup_container": ("bowl",),
        "coffee_stirrer": ("spoon",),
        "soup_eating_utensil": ("spoon",),
        "coffee_source": ("coffee_source",),
        "water_source": ("kettle",),
    },
    "living_room": {
        "PERSONAL_CUP_SAUCER_REGION": ("side_table", "end_table"),
        "SHARED_REMOTE_REGION": ("coffee_table", "central_table", "side_table"),
        "CUP_SAUCER_SET": ("cup_saucer_set", "cup", "saucer"),
        "REMOTE": ("remote_control", "tv_remote"),
        "SEATING_POSITION": ("armchair", "chair", "sofa", "seating_position"),
        "SEATING_PAIR": ("armchair", "chair", "sofa", "seating_pair"),
    },
}


def get_system_role_semantic_categories(
    domain: str,
    canonical_role_id: str,
) -> tuple[str, ...]:
    """Retrieve the system-owned canonical semantic categories accepted for a functional role."""
    domain_roles = SYSTEM_ROLE_SEMANTIC_CATEGORIES.get(domain)
    if domain_roles is None:
        raise KeyError(f"Unknown domain {domain!r} in system role semantic ontology")
    categories = domain_roles.get(canonical_role_id)
    if categories is None:
        raise KeyError(f"Unknown role {canonical_role_id!r} for domain {domain!r} in system role semantic ontology")
    return categories


def get_all_system_role_semantic_categories(domain: str) -> dict[str, tuple[str, ...]]:
    """Retrieve all system-owned role semantic categories for a given domain."""
    domain_roles = SYSTEM_ROLE_SEMANTIC_CATEGORIES.get(domain)
    if domain_roles is None:
        raise KeyError(f"Unknown domain {domain!r} in system role semantic ontology")
    return dict(domain_roles)


def build_task_detector_vocabulary(
    system_role_categories: set[str] | list[str] | tuple[str, ...],
    raw_vlm_candidate_categories: list[str] | tuple[str, ...],
    base_semantic_ontology: dict[str, Any],
) -> dict[str, list[str]]:
    """Build a task-scoped detector vocabulary for YOLO-World.

    Includes only:
      1. System canonical categories required by active task roles.
      2. Reviewed aliases for those relevant canonical categories from the base ontology.
      3. Raw FM candidate categories mapped to relevant canonical categories or
         retained as unmapped task-specific detector prompts.
    Excludes:
      Unrelated global concepts (e.g. remote_control, book, coaster, game_controller, duster).
    """
    base_canon_labels = dict(base_semantic_ontology.get("canonical_labels", {}))

    # Build reverse alias lookup
    alias_to_canon: dict[str, str] = {}
    for canon_k, aliases in base_canon_labels.items():
        for alias in aliases:
            alias_to_canon[alias.strip().lower()] = canon_k

    relevant_canon: set[str] = set()
    for cat in system_role_categories:
        norm = cat.strip().lower()
        if norm in base_canon_labels:
            relevant_canon.add(norm)
        elif norm in alias_to_canon:
            relevant_canon.add(alias_to_canon[norm])

    # Process raw VLM candidate categories
    unmapped_raw_prompts: list[str] = []
    for cat in raw_vlm_candidate_categories:
        norm = cat.strip().lower()
        norm_space = norm.replace("_", " ")
        if norm in base_canon_labels:
            relevant_canon.add(norm)
        elif norm_space in base_canon_labels:
            relevant_canon.add(norm_space)
        elif norm in alias_to_canon:
            relevant_canon.add(alias_to_canon[norm])
        elif norm_space in alias_to_canon:
            relevant_canon.add(alias_to_canon[norm_space])
        else:
            words = norm_space.split()
            matched = False
            for w in reversed(words):
                if w in base_canon_labels:
                    relevant_canon.add(w)
                    matched = True
                    break
                elif w in alias_to_canon:
                    relevant_canon.add(alias_to_canon[w])
                    matched = True
                    break
            if not matched and norm and norm not in unmapped_raw_prompts:
                unmapped_raw_prompts.append(norm)

    # Construct task-scoped vocabulary
    task_vocab: dict[str, list[str]] = {}
    for canon in sorted(relevant_canon):
        if canon in base_canon_labels:
            task_vocab[canon] = list(base_canon_labels[canon])

    existing_aliases = {
        alias.strip().lower()
        for aliases in task_vocab.values()
        for alias in aliases
    }
    for raw_p in unmapped_raw_prompts:
        raw_space = raw_p.replace("_", " ")
        if (
            raw_p not in task_vocab
            and raw_space not in task_vocab
            and raw_p not in existing_aliases
            and raw_space not in existing_aliases
        ):
            task_vocab[raw_p] = [raw_space]
            existing_aliases.add(raw_space)

    return task_vocab
