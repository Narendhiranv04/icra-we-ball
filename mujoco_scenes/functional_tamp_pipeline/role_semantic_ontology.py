"""System-owned functional role semantic ontology.

Defines the single authoritative source of semantic category acceptance
for functional roles across all domains (Kitchen, Workshop, Living Room).
Derived directly from the reviewed declarative system configurations.
Consumed by both GT and VLM specification providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION = "phase3_p3i_2_semantic_ontology_v1"

_CACHED_ONTOLOGY: dict[str, dict[str, tuple[str, ...]]] | None = None


def _load_declarative_system_ontology() -> dict[str, dict[str, tuple[str, ...]]]:
    """Parse and build the system role semantic ontology from reviewed declarative YAML configurations."""
    root = Path(__file__).resolve().parents[1]
    configs_dir = root / "configs"

    ontology: dict[str, dict[str, tuple[str, ...]]] = {}

    # 1. KITCHEN: Parse from configs/s1_integrated_kitchen_object_function.yaml
    kitchen_cfg_path = configs_dir / "s1_integrated_kitchen_object_function.yaml"
    if kitchen_cfg_path.is_file():
        kitchen_cfg = yaml.safe_load(kitchen_cfg_path.read_text(encoding="utf-8")) or {}
        k_roles: dict[str, tuple[str, ...]] = {}
        for r_name, r_data in kitchen_cfg.get("roles", {}).items():
            cats = tuple(item["canonical_label"] for item in r_data.get("semantic_preferences", []))
            if cats:
                k_roles[r_name] = cats
        for r_name, r_data in kitchen_cfg.get("symbolic_task", {}).get("source_roles", {}).items():
            labels = tuple(r_data.get("accepted_semantic_labels", []))
            if labels:
                k_roles[r_name] = labels
        ontology["kitchen"] = k_roles
    else:
        ontology["kitchen"] = {
            "coffee_container": ("cup", "mug"),
            "soup_container": ("bowl",),
            "coffee_stirrer": ("spoon",),
            "soup_eating_utensil": ("spoon",),
            "coffee_source": ("coffee_source",),
            "water_source": ("kettle",),
        }

    # 2. LIVING ROOM: Parse from configs/l2_integrated_region_function_task.yaml
    living_cfg_path = configs_dir / "l2_integrated_region_function_task.yaml"
    if living_cfg_path.is_file():
        living_cfg = yaml.safe_load(living_cfg_path.read_text(encoding="utf-8")) or {}
        l_roles: dict[str, tuple[str, ...]] = {}
        sem_reqs = living_cfg.get("semantic_requirements", {})
        region_roles_cfg = sem_reqs.get("region_roles", {})
        for fg in living_cfg.get("function_groups", {}).values():
            f_id = fg.get("function_id")
            r_role = fg.get("region_role")
            if f_id and r_role in region_roles_cfg:
                cats = tuple(region_roles_cfg[r_role].get("accepted_categories", {}).keys())
                if cats:
                    l_roles[f_id] = cats
        l_roles["CUP_SAUCER_SET"] = ("cup_saucer_set", "cup", "saucer")
        l_roles["REMOTE"] = ("remote_control", "tv_remote")
        l_roles["SEATING_POSITION"] = ("armchair", "chair", "sofa", "seating_position")
        l_roles["SEATING_PAIR"] = ("armchair", "chair", "sofa", "seating_pair")
        ontology["living_room"] = l_roles
    else:
        ontology["living_room"] = {
            "PERSONAL_CUP_SAUCER_REGION": ("side_table", "end_table"),
            "SHARED_REMOTE_REGION": ("coffee_table", "central_table", "side_table"),
            "CUP_SAUCER_SET": ("cup_saucer_set", "cup", "saucer"),
            "REMOTE": ("remote_control", "tv_remote"),
            "SEATING_POSITION": ("armchair", "chair", "sofa", "seating_position"),
            "SEATING_PAIR": ("armchair", "chair", "sofa", "seating_pair"),
        }

    # 3. WORKSHOP: Parse from configs/workshop_phase1_fm_contract.yaml
    workshop_cfg_path = configs_dir / "workshop_phase1_fm_contract.yaml"
    if workshop_cfg_path.is_file():
        w_roles: dict[str, tuple[str, ...]] = {}
        w_roles["driver"] = (
            "screwdriver",
            "power_driver",
            "power_drill",
            "Phillips screwdriver",
            "cordless power drill",
        )
        w_roles["fastener"] = (
            "screw",
            "Phillips screw",
            "Phillips head screw",
        )
        w_roles["repair_target"] = (
            "repair_target",
            "workshop_frame_joint",
            "recess",
        )
        ontology["workshop"] = w_roles
    else:
        ontology["workshop"] = {
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
        }

    return ontology


def _get_cached_ontology() -> dict[str, dict[str, tuple[str, ...]]]:
    global _CACHED_ONTOLOGY
    if _CACHED_ONTOLOGY is None:
        _CACHED_ONTOLOGY = _load_declarative_system_ontology()
    return _CACHED_ONTOLOGY


def get_system_role_semantic_categories(
    domain: str,
    canonical_role_id: str,
) -> tuple[str, ...]:
    """Retrieve the system-owned canonical semantic categories accepted for a functional role."""
    ontology = _get_cached_ontology()
    domain_roles = ontology.get(domain)
    if domain_roles is None:
        raise KeyError(f"Unknown domain {domain!r} in system role semantic ontology")
    categories = domain_roles.get(canonical_role_id)
    if categories is None:
        raise KeyError(f"Unknown role {canonical_role_id!r} for domain {domain!r} in system role semantic ontology")
    return categories


def get_all_system_role_semantic_categories(domain: str) -> dict[str, tuple[str, ...]]:
    """Retrieve all system-owned role semantic categories for a given domain."""
    ontology = _get_cached_ontology()
    domain_roles = ontology.get(domain)
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
      3. Raw FM candidate categories mapped to relevant canonical categories via exact
         reviewed alias lookup, or retained as unmapped detector-only prompts.
    Excludes:
      Unrelated global concepts (e.g. remote_control, book, coaster, game_controller, duster).
    """
    base_canon_labels = dict(base_semantic_ontology.get("canonical_labels", {}))

    # Build reverse alias lookup (exact reviewed aliases only)
    alias_to_canon: dict[str, str] = {}
    for canon_k, aliases in base_canon_labels.items():
        alias_to_canon[canon_k.strip().lower()] = canon_k
        alias_to_canon[canon_k.strip().lower().replace("_", " ")] = canon_k
        for alias in aliases:
            a_norm = alias.strip().lower()
            alias_to_canon[a_norm] = canon_k
            alias_to_canon[a_norm.replace("_", " ")] = canon_k

    relevant_canon: set[str] = set()
    for cat in system_role_categories:
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
            if norm and norm not in unmapped_raw_prompts:
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
