"""System-owned functional role semantic ontology.

Defines the single authoritative source of semantic category acceptance
for functional roles across all domains (Kitchen, Workshop, Living Room).
Loaded strictly and exclusively from the isolated runtime ontology configuration:
configs/runtime_functional_semantic_ontology.yaml

Zero GT/reference task knowledge is loaded at runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import yaml

from .errors import SemanticOntologyConfigurationError

PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION = "phase3_p3i_4_semantic_ontology_v1"
RUNTIME_FUNCTIONAL_SEMANTIC_ONTOLOGY_VERSION = "runtime_functional_semantic_ontology_v1"
RUNTIME_ONTOLOGY_FILENAME = "runtime_functional_semantic_ontology.yaml"

_CACHED_ONTOLOGY: dict[str, dict[str, tuple[str, ...]]] | None = None
_CACHED_DETECTOR_ALIASES: dict[str, list[str]] | None = None
_CACHED_ONTOLOGY_HASH: str | None = None


def clear_cached_ontology() -> None:
    """Clear cached system ontology for test isolation."""
    global _CACHED_ONTOLOGY, _CACHED_DETECTOR_ALIASES, _CACHED_ONTOLOGY_HASH
    _CACHED_ONTOLOGY = None
    _CACHED_DETECTOR_ALIASES = None
    _CACHED_ONTOLOGY_HASH = None


reset_cached_ontology = clear_cached_ontology


def get_runtime_semantic_ontology_path(configs_dir: Path | None = None) -> Path:
    """Return the absolute path to the runtime functional semantic ontology."""
    if configs_dir is None:
        root = Path(__file__).resolve().parents[1]
        configs_dir = root / "configs"
    return configs_dir / RUNTIME_ONTOLOGY_FILENAME


def get_runtime_semantic_ontology_hash(configs_dir: Path | None = None) -> str:
    """Compute and return the SHA-256 hash of the runtime semantic ontology."""
    global _CACHED_ONTOLOGY_HASH
    if _CACHED_ONTOLOGY_HASH is not None and configs_dir is None:
        return _CACHED_ONTOLOGY_HASH
    cfg_path = get_runtime_semantic_ontology_path(configs_dir)
    if not cfg_path.is_file():
        raise SemanticOntologyConfigurationError(
            f"Missing runtime functional semantic ontology configuration: {cfg_path}"
        )
    h = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    if configs_dir is None:
        _CACHED_ONTOLOGY_HASH = h
    return h


def _load_declarative_system_ontology(
    configs_dir: Path | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Parse and build the system role semantic ontology from the runtime-only YAML configuration.

    Fails closed immediately with SemanticOntologyConfigurationError if the configuration file
    or any canonical role entry is missing, malformed, or empty.
    """
    global _CACHED_DETECTOR_ALIASES, _CACHED_ONTOLOGY_HASH
    cfg_path = get_runtime_semantic_ontology_path(configs_dir)
    if not cfg_path.is_file():
        raise SemanticOntologyConfigurationError(
            f"Missing runtime functional semantic ontology: {cfg_path}"
        )

    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(raw_text)
    except Exception as e:
        raise SemanticOntologyConfigurationError(
            f"Malformed runtime functional semantic ontology YAML in {cfg_path}: {e}"
        ) from e

    if not isinstance(cfg, dict):
        raise SemanticOntologyConfigurationError(
            f"Malformed runtime functional semantic ontology config in {cfg_path}: expected mapping"
        )

    ontology: dict[str, dict[str, tuple[str, ...]]] = {}

    # Extract domains structure: support either `domains: {kitchen: ...}` or `roles: {kitchen: ...}`
    domains_data = cfg.get("domains")
    if not isinstance(domains_data, dict):
        domains_data = cfg.get("roles")
    if not isinstance(domains_data, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing or invalid 'domains' or 'roles' section in {cfg_path}"
        )

    # 1. KITCHEN
    k_data = domains_data.get("kitchen")
    if not isinstance(k_data, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing 'kitchen' domain in {cfg_path}"
        )
    k_roles_data = k_data.get("roles", k_data)
    if not isinstance(k_roles_data, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing 'kitchen.roles' mapping in {cfg_path}"
        )
    k_roles: dict[str, tuple[str, ...]] = {}
    for r_name, r_info in k_roles_data.items():
        if isinstance(r_info, dict):
            cats = r_info.get("accepted_categories") or r_info.get("semantic_preferences")
            if isinstance(cats, (list, tuple)) and cats:
                # Handle list of dicts or list of strings
                if isinstance(cats[0], dict) and "canonical_label" in cats[0]:
                    k_roles[r_name] = tuple(str(item["canonical_label"]) for item in cats if "canonical_label" in item)
                else:
                    k_roles[r_name] = tuple(str(c) for c in cats)
        elif isinstance(r_info, (list, tuple)) and r_info:
            k_roles[r_name] = tuple(str(c) for c in r_info)

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
            f"Missing semantic acceptance entries for kitchen roles {sorted(missing_k)} in {cfg_path}"
        )
    ontology["kitchen"] = k_roles

    # 2. LIVING ROOM
    l_data = domains_data.get("living_room")
    if not isinstance(l_data, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing 'living_room' domain in {cfg_path}"
        )
    l_roles_data = l_data.get("roles", l_data)
    if not isinstance(l_roles_data, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing 'living_room.roles' mapping in {cfg_path}"
        )
    l_roles: dict[str, tuple[str, ...]] = {}
    for r_name, r_info in l_roles_data.items():
        if isinstance(r_info, dict):
            cats = r_info.get("accepted_categories") or r_info.get("accepted_semantic_labels")
            if isinstance(cats, (list, tuple)) and cats:
                l_roles[r_name] = tuple(str(c) for c in cats)
        elif isinstance(r_info, (list, tuple)) and r_info:
            l_roles[r_name] = tuple(str(c) for c in r_info)

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
            f"Missing semantic acceptance entries for living_room roles {sorted(missing_l)} in {cfg_path}"
        )
    ontology["living_room"] = l_roles

    # 3. WORKSHOP
    w_data = domains_data.get("workshop")
    if not isinstance(w_data, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing 'workshop' domain in {cfg_path}"
        )
    w_roles_data = w_data.get("roles", w_data)
    if not isinstance(w_roles_data, dict):
        raise SemanticOntologyConfigurationError(
            f"Missing 'workshop.roles' mapping in {cfg_path}"
        )
    w_roles: dict[str, tuple[str, ...]] = {}
    for r_name, r_info in w_roles_data.items():
        if isinstance(r_info, dict):
            cats = r_info.get("accepted_categories") or r_info.get("accepted_semantic_labels")
            if isinstance(cats, (list, tuple)) and cats:
                w_roles[r_name] = tuple(str(c) for c in cats)
        elif isinstance(r_info, (list, tuple)) and r_info:
            w_roles[r_name] = tuple(str(c) for c in r_info)

    required_w_roles = {"driver", "fastener", "repair_target"}
    missing_w = required_w_roles - set(w_roles)
    if missing_w:
        raise SemanticOntologyConfigurationError(
            f"Missing semantic acceptance entries for workshop roles {sorted(missing_w)} in {cfg_path}"
        )
    ontology["workshop"] = w_roles

    # Detector aliases
    raw_aliases = cfg.get("detector_aliases") or cfg.get("canonical_labels") or {}
    aliases: dict[str, list[str]] = {}
    if isinstance(raw_aliases, dict):
        for k, v in raw_aliases.items():
            if isinstance(v, (list, tuple)):
                aliases[str(k)] = [str(x) for x in v]
            elif isinstance(v, dict) and "aliases" in v:
                aliases[str(k)] = [str(x) for x in v["aliases"]]
    _CACHED_DETECTOR_ALIASES = aliases
    _CACHED_ONTOLOGY_HASH = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    return ontology


def _get_cached_ontology() -> dict[str, dict[str, tuple[str, ...]]]:
    global _CACHED_ONTOLOGY
    if _CACHED_ONTOLOGY is None:
        _CACHED_ONTOLOGY = _load_declarative_system_ontology()
    return _CACHED_ONTOLOGY


def get_runtime_detector_aliases() -> dict[str, list[str]]:
    """Retrieve runtime detector aliases for YOLO-World perception."""
    global _CACHED_DETECTOR_ALIASES
    if _CACHED_DETECTOR_ALIASES is None:
        _load_declarative_system_ontology()
    return dict(_CACHED_DETECTOR_ALIASES or {})


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
    base_semantic_ontology: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Build a task-scoped detector vocabulary for YOLO-World.

    Includes only:
      1. System canonical categories required by active task roles.
      2. Reviewed aliases for those relevant canonical categories from the base ontology or runtime ontology.
      3. Raw FM candidate categories mapped to relevant canonical categories via exact
         reviewed alias lookup, or retained as unmapped detector-only prompts.
    Excludes:
      Unrelated global concepts (e.g. remote_control, book, coaster, game_controller, duster).
    """
    if base_semantic_ontology is None:
        base_canon_labels = get_runtime_detector_aliases()
    elif isinstance(base_semantic_ontology, dict):
        if "canonical_labels" in base_semantic_ontology:
            base_canon_labels = dict(base_semantic_ontology.get("canonical_labels", {}))
        elif "detector_aliases" in base_semantic_ontology:
            base_canon_labels = dict(base_semantic_ontology.get("detector_aliases", {}))
        else:
            base_canon_labels = dict(base_semantic_ontology)
    else:
        base_canon_labels = get_runtime_detector_aliases()

    # Build reverse alias lookup (exact reviewed aliases only)
    alias_to_canon: dict[str, str] = {}
    for canon_k, aliases in base_canon_labels.items():
        alias_to_canon[canon_k.strip().lower()] = canon_k
        alias_to_canon[canon_k.strip().lower().replace("_", " ")] = canon_k
        for alias in aliases:
            a_norm = alias.strip().lower()
            alias_to_canon[a_norm] = canon_k
            alias_to_canon[a_norm.replace("_", " ")] = canon_k

    system_canon: set[str] = set()
    for cat in system_role_categories:
        norm = cat.strip().lower()
        norm_space = norm.replace("_", " ")
        if norm in base_canon_labels:
            system_canon.add(norm)
        elif norm_space in base_canon_labels:
            system_canon.add(norm_space)
        elif norm in alias_to_canon:
            system_canon.add(alias_to_canon[norm])
        elif norm_space in alias_to_canon:
            system_canon.add(alias_to_canon[norm_space])

    relevant_canon: set[str] = set(system_canon)

    # Process raw VLM candidate categories
    unmapped_raw_prompts: list[str] = []
    for cat in raw_vlm_candidate_categories:
        norm = cat.strip().lower()
        norm_space = norm.replace("_", " ")
        target_canon = None
        if norm in base_canon_labels:
            target_canon = norm
        elif norm_space in base_canon_labels:
            target_canon = norm_space
        elif norm in alias_to_canon:
            target_canon = alias_to_canon[norm]
        elif norm_space in alias_to_canon:
            target_canon = alias_to_canon[norm_space]
        else:
            words = norm_space.split()
            if len(words) > 1:
                head = words[-1]
                if head in base_canon_labels:
                    target_canon = head
                elif head in alias_to_canon:
                    target_canon = alias_to_canon[head]

        if target_canon and target_canon in system_canon:
            alias_to_canon[norm] = target_canon
            alias_to_canon[norm_space] = target_canon
        elif not target_canon:
            if norm and not any(norm.endswith(h) for h in ("container", "utensil", "item", "object")) and norm not in unmapped_raw_prompts:
                unmapped_raw_prompts.append(norm)

    # Construct task-scoped vocabulary
    task_vocab: dict[str, list[str]] = {}
    for canon in sorted(relevant_canon):
        if canon in base_canon_labels:
            task_vocab[canon] = [
                lbl for lbl in base_canon_labels[canon]
                if not any(lbl.strip().lower().replace("_", " ").endswith(h) for h in ("container", "utensil", "item", "object", "source"))
            ]
            for k, target in alias_to_canon.items():
                if target == canon and k not in task_vocab[canon] and not any(k.strip().lower().replace("_", " ").endswith(h) for h in ("container", "utensil", "item", "object", "source")):
                    task_vocab[canon].append(k)

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
