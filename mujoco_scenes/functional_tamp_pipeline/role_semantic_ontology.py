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

from .errors import SemanticOntologyConfigurationError

PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION = "phase3_p3i_4_semantic_ontology_v1"

_CACHED_ONTOLOGY: dict[str, dict[str, tuple[str, ...]]] | None = None


def clear_cached_ontology() -> None:
    """Clear cached system ontology for test isolation."""
    global _CACHED_ONTOLOGY
    _CACHED_ONTOLOGY = None


reset_cached_ontology = clear_cached_ontology


def _load_declarative_system_ontology(
    configs_dir: Path | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Parse and build the system role semantic ontology from reviewed declarative YAML configurations.

    Fails closed immediately with SemanticOntologyConfigurationError if any required
    configuration file or canonical role entry is missing, malformed, or empty.
    """
    if configs_dir is None:
        root = Path(__file__).resolve().parents[1]
        configs_dir = root / "configs"

    ontology: dict[str, dict[str, tuple[str, ...]]] = {}

    # 1. KITCHEN: Parse from configs/s1_integrated_kitchen_object_function.yaml
    kitchen_cfg_path = configs_dir / "s1_integrated_kitchen_object_function.yaml"
    if not kitchen_cfg_path.is_file():
        raise SemanticOntologyConfigurationError(
            f"Missing reviewed semantic ontology for kitchen: {kitchen_cfg_path}"
        )
    try:
        kitchen_cfg = yaml.safe_load(kitchen_cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SemanticOntologyConfigurationError(
            f"Malformed kitchen semantic ontology YAML in {kitchen_cfg_path}: {e}"
        ) from e
    if not isinstance(kitchen_cfg, dict):
        raise SemanticOntologyConfigurationError(
            f"Malformed kitchen semantic ontology config in {kitchen_cfg_path}"
        )
    try:
        k_roles: dict[str, tuple[str, ...]] = {}
        for r_name, r_data in kitchen_cfg.get("roles", {}).items():
            if not isinstance(r_data, dict):
                continue
            cats = tuple(
                item["canonical_label"]
                for item in r_data.get("semantic_preferences", [])
                if isinstance(item, dict) and "canonical_label" in item
            )
            if cats:
                k_roles[r_name] = cats
        for r_name, r_data in (
            kitchen_cfg.get("symbolic_task", {}).get("source_roles", {}).items()
        ):
            if not isinstance(r_data, dict):
                continue
            labels = tuple(r_data.get("accepted_semantic_labels", []))
            if labels:
                k_roles[r_name] = labels
    except Exception as e:
        raise SemanticOntologyConfigurationError(
            f"Malformed kitchen roles section in {kitchen_cfg_path}: {e}"
        ) from e

    required_k_roles = {
        "coffee_container",
        "soup_container",
        "coffee_stirrer",
        "soup_eating_utensil",
        "coffee_source",
        "water_source",
    }
    missing_k = required_k_roles - set(k_roles)
    if missing_k:
        raise SemanticOntologyConfigurationError(
            f"Missing semantic acceptance entries for kitchen roles {sorted(missing_k)} in {kitchen_cfg_path}"
        )
    ontology["kitchen"] = k_roles

    # 2. LIVING ROOM: Parse from configs/l2_integrated_region_function_task.yaml
    living_cfg_path = configs_dir / "l2_integrated_region_function_task.yaml"
    if not living_cfg_path.is_file():
        raise SemanticOntologyConfigurationError(
            f"Missing reviewed semantic ontology for living_room: {living_cfg_path}"
        )
    try:
        living_cfg = yaml.safe_load(living_cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SemanticOntologyConfigurationError(
            f"Malformed living_room semantic ontology YAML in {living_cfg_path}: {e}"
        ) from e
    if not isinstance(living_cfg, dict):
        raise SemanticOntologyConfigurationError(
            f"Malformed living_room semantic ontology config in {living_cfg_path}"
        )
    l_roles: dict[str, tuple[str, ...]] = {}
    sem_reqs = living_cfg.get("semantic_requirements")
    if not isinstance(sem_reqs, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing or malformed semantic_requirements section in {living_cfg_path}"
        )

    # Explicit functional_roles is authoritative
    explicit_l_roles = sem_reqs.get("functional_roles")
    if not isinstance(explicit_l_roles, dict) or not explicit_l_roles:
        raise SemanticOntologyConfigurationError(
            f"Missing or empty semantic_requirements.functional_roles in {living_cfg_path}"
        )
    try:
        for r_name, r_cats in explicit_l_roles.items():
            if isinstance(r_cats, (list, tuple)) and r_cats:
                l_roles[r_name] = tuple(str(c) for c in r_cats)
    except Exception as e:
        raise SemanticOntologyConfigurationError(
            f"Malformed living_room functional_roles in {living_cfg_path}: {e}"
        ) from e

    required_l_roles = {
        "PERSONAL_CUP_SAUCER_REGION",
        "SHARED_REMOTE_REGION",
        "CUP_SAUCER_SET",
        "REMOTE",
        "SEATING_POSITION",
        "SEATING_PAIR",
    }
    missing_l = required_l_roles - set(l_roles)
    if missing_l:
        raise SemanticOntologyConfigurationError(
            f"Missing semantic acceptance entries for living_room roles {sorted(missing_l)} in {living_cfg_path}"
        )
    ontology["living_room"] = l_roles

    # 3. WORKSHOP: Parse from configs/workshop_phase1_fm_contract.yaml
    workshop_cfg_path = configs_dir / "workshop_phase1_fm_contract.yaml"
    if not workshop_cfg_path.is_file():
        raise SemanticOntologyConfigurationError(
            f"Missing reviewed semantic ontology for workshop: {workshop_cfg_path}"
        )
    try:
        workshop_cfg = yaml.safe_load(workshop_cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SemanticOntologyConfigurationError(
            f"Malformed workshop semantic ontology YAML in {workshop_cfg_path}: {e}"
        ) from e
    if not isinstance(workshop_cfg, dict):
        raise SemanticOntologyConfigurationError(
            f"Malformed workshop semantic ontology config in {workshop_cfg_path}"
        )
    w_roles: dict[str, tuple[str, ...]] = {}

    # Explicit functional_roles is authoritative
    explicit_w_roles = workshop_cfg.get("functional_roles") or workshop_cfg.get(
        "system_role_acceptance"
    )
    if not isinstance(explicit_w_roles, dict) or not explicit_w_roles:
        raise SemanticOntologyConfigurationError(
            f"Missing or empty functional_roles in {workshop_cfg_path}"
        )
    try:
        for r_name, r_cats in explicit_w_roles.items():
            if isinstance(r_cats, (list, tuple)) and r_cats:
                w_roles[r_name] = tuple(str(c) for c in r_cats)
    except Exception as e:
        raise SemanticOntologyConfigurationError(
            f"Malformed workshop functional_roles in {workshop_cfg_path}: {e}"
        ) from e

    required_w_roles = {"driver", "fastener", "repair_target"}
    missing_w = required_w_roles - set(w_roles)
    if missing_w:
        raise SemanticOntologyConfigurationError(
            f"Missing semantic acceptance entries for workshop roles {sorted(missing_w)} in {workshop_cfg_path}"
        )
    ontology["workshop"] = w_roles

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
        raise KeyError(
            f"Unknown role {canonical_role_id!r} for domain {domain!r} in system role semantic ontology"
        )
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
