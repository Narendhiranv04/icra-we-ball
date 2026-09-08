"""Safe Relation Semantic Interpreter for Phase 3.

Maps free-form natural language relation phrases to canonical active binary predicates
based on deterministic semantic candidate extraction and endpoint-signature intersection.
Adheres strictly to the Phase 3 contract:
1. Endpoint filtering alone NEVER chooses a predicate (no len(valid)==1 blind choice).
2. Semantic candidates proposed by deterministic linguistic cue analysis.
3. Multi-predicate compilation supported when text contains multiple distinct concepts.
4. Direction normalization permitted only when supported by semantic evidence.
5. Unmapped required relations fail closed as UNMAPPABLE_REQUIRED_RELATION.
6. Emits detailed relation interpretation trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Sequence

from .predicate_registry import (
    PredicateSignature,
    get_active_predicates,
    get_endpoint_valid_binary_predicates,
    get_predicate_signature,
)


@dataclass(frozen=True)
class InterpretedPredicate:
    subject_role: str
    predicate_name: str
    object_role: str


@dataclass
class RelationInterpretationResult:
    status: str
    interpreted_predicates: tuple[InterpretedPredicate, ...]
    direction_normalized: bool = False
    reason: str = ""
    category: str = "PHYSICAL_VERIFIER"  # "PHYSICAL_VERIFIER" or "TASK_CAUSAL_SEMANTICS"
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return bool(self.interpreted_predicates)


# Deterministic linguistic cues for proposing semantic predicate candidates
_SEMANTIC_PREDICATE_CUES: dict[tuple[str, str], tuple[str, ...]] = {
    # Kitchen
    ("kitchen", "INSERTABLE_IN"): (
        "insertable in", "insertable into", "insert into", "inserted into", "insert in",
        "fits in", "fits inside", "fit in", "fit inside", "enters opening", "enters cavity",
        "inside container", "into container", "place inside", "placed inside", "contained in",
        "inserted", "place in", "placed in", "fits opening", "inside",
    ),
    ("kitchen", "REACHES_BOTTOM"): (
        "reaches bottom", "reach bottom", "touches bottom", "touch bottom", "deep enough",
        "sufficient depth", "reaches the base", "bottom clearance", "reaches full depth",
        "length reaches bottom", "stir to bottom", "reach depth", "depth clearance",
    ),

    # Living Room
    ("living_room", "FITS_SET_ON"): (
        "fits set on", "fits on", "placed on", "place on", "supported on", "support on",
        "rests on", "rest on", "tabletop for set", "surface for cup and saucer",
        "drinkware on", "refreshment on", "cup and saucer on", "staging surface for set",
        "support surface", "table for refreshments", "staging surface",
        "tabletop support", "tabletop support for drinkware set", "support for drinkware set",
        "support for drinkware", "drinkware set", "supports drinkware set", "support drinkware set",
    ),
    ("living_room", "FITS_ON"): (
        "fits on", "placed on", "place on", "supported on", "support on", "rests on",
        "rest on", "tabletop for remote", "surface for remote", "remote on",
        "entertainment control on", "control device on", "support remote",
    ),
    ("living_room", "NEAR_SEAT"): (
        "near seat", "near seating", "beside seat", "adjacent to seat", "close to seat",
        "accessible to seat", "convenient to seat", "at seating position", "viewer seating position",
        "near viewer", "beside viewer", "adjacent to viewer",
    ),
    ("living_room", "ACCESSIBLE_FROM_BOTH_SEATS"): (
        "accessible from both seats", "accessible from both", "reachable from both seats",
        "shared between seats", "between both seats", "serves both seats", "central to both seats",
        "accessible to pair of seats", "paired seating", "accessible from pair",
        "accessible to viewers", "accessible between seats",
    ),

    # Workshop
    ("workshop", "COMPATIBLE_WITH"): (
        "compatible with", "compatible with fastener", "compatible with screw",
        "engages fastener", "engages screw", "engages head", "fits screw", "fits fastener",
        "fits head", "driver bit matches", "matches recess", "transmits torque",
        "transmit torque", "drives screw", "fastens with", "operates on", "used together",
        "mechanically engages", "engages with", "driver engages", "fits recess",
        "compatible tool", "matches fastener", "tightens", "rotates fastener",
    ),
    ("workshop", "REACHES_TARGET"): (
        "reaches target", "reaches repair target", "reach target", "reaches hole",
        "reach hole", "reaches workpiece", "accesses target", "accesses workpiece",
        "sufficient reach to target", "length reaches target", "accesses repair recess",
        "reaches joint", "reaches location",
    ),
    ("workshop", "COMPATIBLE_WITH_TARGET"): (
        "compatible with", "compatible with target", "compatible with repair target", "fits into hole",
        "fits target hole", "screws into target", "threaded into target", "thread matches",
        "diameter matches", "inserted into workpiece", "fastens into target",
        "inserted into target", "secured at target", "secures into hole",
        "fastened at target", "installed in target", "anchored in hole",
    ),
}

# Linguistic cues that explicitly indicate inverse direction
_INVERSE_DIRECTION_CUES: dict[tuple[str, str], tuple[str, ...]] = {
    ("kitchen", "INSERTABLE_IN"): (
        "contains", "holds", "receives", "receptacle for", "container for",
        "encloses", "houses",
    ),
    ("workshop", "COMPATIBLE_WITH"): (
        "driven by", "engaged by", "fastened by", "tightened by", "turned by",
        "operated by",
    ),
    ("workshop", "COMPATIBLE_WITH_TARGET"): (
        "receives fastener", "accepts fastener", "threaded for",
    ),
}


def _normalize_text(text: str) -> str:
    """Normalize text by lowering, removing punctuation, and collapsing whitespace."""
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(s.split())


def _get_domain_alias_tables(domain: str) -> dict[str, Sequence[str]]:
    """Load reviewed domain alias tables as secondary backend linguistic evidence."""
    tables: dict[str, Sequence[str]] = {}
    if domain == "kitchen":
        try:
            from mujoco_scenes.kitchen_vlm_functional_graph import BINARY_RELATION_ALIASES
            tables.update(BINARY_RELATION_ALIASES)
        except ImportError:
            pass
    elif domain == "living_room":
        try:
            from mujoco_scenes.environment_vlm_requirements import LIVING_BINARY_RELATION_ALIASES
            tables.update(LIVING_BINARY_RELATION_ALIASES)
        except ImportError:
            pass
    return tables


def _extract_semantic_candidates(
    domain: str,
    norm_phrase: str,
) -> set[str]:
    """Deterministically propose candidate predicate names matching keywords in phrase."""
    d_norm = domain.strip().lower()
    candidates: set[str] = set()

    for (d, pred_name), cues in _SEMANTIC_PREDICATE_CUES.items():
        if d != d_norm:
            continue
        for cue in cues:
            cue_norm = _normalize_text(cue)
            # Match exact or as word substring
            if cue_norm == norm_phrase or re.search(r"\b" + re.escape(cue_norm) + r"\b", norm_phrase):
                candidates.add(pred_name)
                break

    # Secondary: check reviewed domain alias tables
    for pred_name, aliases in _get_domain_alias_tables(d_norm).items():
        if pred_name in candidates:
            continue
        for alias in aliases:
            a_norm = _normalize_text(alias)
            if a_norm == norm_phrase or re.search(r"\b" + re.escape(a_norm) + r"\b", norm_phrase):
                candidates.add(pred_name)
                break

    return candidates


def _extract_inverse_semantic_candidates(
    domain: str,
    norm_phrase: str,
) -> set[str]:
    """Deterministically propose candidate predicate names matching inverse cues."""
    d_norm = domain.strip().lower()
    candidates: set[str] = set()

    for (d, pred_name), cues in _INVERSE_DIRECTION_CUES.items():
        if d != d_norm:
            continue
        for cue in cues:
            cue_norm = _normalize_text(cue)
            if cue_norm == norm_phrase or re.search(r"\b" + re.escape(cue_norm) + r"\b", norm_phrase):
                candidates.add(pred_name)
                break
    return candidates


# Deterministic linguistic cues for task/causal semantic relations
_TASK_CAUSAL_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "PROVIDES_MATERIAL_TO": (
        "provides material to", "provides material into", "supplies material to", "supplies to",
        "transfers material to", "transfers content to", "transfers contents to", "pours into", "poured into",
        "dispenses into", "fills", "provides contents to", "provides ingredients to", "source for",
        "provides coffee to", "provides water to", "provides soup to", "poured in", "pours in",
        "supplies", "feeds into", "supplies material into",
    ),
    "ACTS_ON": (
        "acts on", "operates on", "manipulates", "manipulates interior of", "stirs contents of",
        "stirs contents", "works on", "applies to", "interacts with", "stirs", "mixes",
        "mixes contents of", "stirs interior of", "manipulates fastener", "manipulates component",
        "manipulate or install", "drives", "fastens",
    ),
    "PAIRED_WITH": (
        "paired with", "is paired with", "accompanied by", "alongside", "served with",
        "served alongside", "associated with", "complements", "set with", "provided alongside",
        "arranged with", "provided for", "arranged for", "placed with",
    ),
    "INSTALLED_AT": (
        "installed at", "installed in", "secured at", "secured in", "fastened at", "fastened in",
        "anchored at", "anchored in", "attached to", "mounted at", "placed at target",
        "installed on", "secured on",
    ),
    "CONNECTED_TO": (
        "connects to", "connected to", "joins with", "joined with", "fastened together",
    ),
}

_TASK_CAUSAL_INVERSE_CUES: dict[str, tuple[str, ...]] = {
    "PROVIDES_MATERIAL_TO": (
        "receives material from", "receives contents from", "receives coffee from",
        "receives water from", "receives ingredients from", "receives from", "filled by",
        "supplied by", "receives liquid from",
    ),
    "ACTS_ON": (
        "manipulated by", "operated by", "acted on by", "stirred by", "mixed by",
    ),
    "INSTALLED_AT": (
        "receives component", "holds component", "receives fastener", "anchors component",
    ),
}


_TASK_EFFECT_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "CONTAINS": (
        "contains", "contain", "holds contents", "holds material",
        "filled with", "has contents", "has material",
    ),
}

_TRANSFER_OPERATION_CUES: tuple[str, ...] = (
    "transfer", "pour", "fill", "dispense", "load", "add",
)


def interpret_task_effect_predicate(raw_phrase: str) -> str | None:
    """Return a state/effect predicate supported directly by relation text."""
    norm_phrase = _normalize_text(raw_phrase)
    for predicate, cues in _TASK_EFFECT_RELATION_CUES.items():
        if any(
            (cue_norm := _normalize_text(cue)) == norm_phrase
            or re.search(r"\b" + re.escape(cue_norm) + r"\b", norm_phrase)
            for cue in cues
        ):
            return predicate
    return None


def has_compatible_explicit_effect_operation(
    predicate: str,
    raw_subject: str,
    raw_object: str,
    groups: Sequence[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Match an expressed state edge to an explicitly expressed operation.

    This recognizes direction only; it never creates an operation from a
    relation.  For CONTAINS, the material/source must flow into the carrier.
    """
    if predicate != "CONTAINS":
        return False, None
    for group in groups:
        source = group.get("tool_role") or group.get("source_role")
        target = group.get("target_role")
        phrase = _normalize_text(group.get("function") or group.get("operation") or "")
        if (
            source == raw_object
            and target == raw_subject
            and any(re.search(r"\b" + re.escape(cue) + r"\b", phrase) for cue in _TRANSFER_OPERATION_CUES)
        ):
            return True, str(group.get("id", "")) or None
    return False, None


def _extract_task_causal_candidates(norm_phrase: str) -> tuple[str, bool] | None:
    """Deterministically check if phrase matches a task/causal semantic relation."""
    # Check forward cues
    for pred, cues in _TASK_CAUSAL_RELATION_CUES.items():
        for cue in cues:
            cue_norm = _normalize_text(cue)
            if cue_norm == norm_phrase or re.search(r"\b" + re.escape(cue_norm) + r"\b", norm_phrase):
                return (pred, False)

    # Check inverse cues
    for pred, cues in _TASK_CAUSAL_INVERSE_CUES.items():
        for cue in cues:
            cue_norm = _normalize_text(cue)
            if cue_norm == norm_phrase or re.search(r"\b" + re.escape(cue_norm) + r"\b", norm_phrase):
                return (pred, True)

    return None


def interpret_relation(
    domain: str,
    raw_phrase: str,
    subject_role: str,
    object_role: str,
    subject_kind: str,
    object_kind: str,
    *,
    required: bool = True,
) -> RelationInterpretationResult:
    """Interpret a natural language relation between declared canonical roles into active predicates.

    Fails closed if semantic evidence is absent, even if an endpoint pair has only one valid predicate.
    """
    d_norm = domain.strip().lower()
    norm_phrase = _normalize_text(raw_phrase)
    evidence: dict[str, Any] = {
        "domain": d_norm,
        "raw_phrase": raw_phrase,
        "norm_phrase": norm_phrase,
        "subject_role": subject_role,
        "object_role": object_role,
        "subject_kind": subject_kind,
        "object_kind": object_kind,
        "required": required,
    }

    if not norm_phrase:
        status = "UNINTERPRETABLE_REQUIRED_RELATION" if required else "SOFT_OPTIONAL_RELATION"
        return RelationInterpretationResult(
            status=status,
            category="UNKNOWN",
            interpreted_predicates=(),
            reason="Empty relation phrase",
            evidence=evidence,
        )

    # 1. Endpoint-valid predicates in forward direction
    forward_valid_sigs = get_endpoint_valid_binary_predicates(
        d_norm,
        subject_role=subject_role,
        object_role=object_role,
        subject_kind=subject_kind,
        object_kind=object_kind,
    )
    forward_valid_names = {sig.name for sig in forward_valid_sigs}
    evidence["forward_valid_predicates"] = sorted(forward_valid_names)

    # 2. Extract semantic candidates from phrase (including exact predicate names)
    semantic_candidates = _extract_semantic_candidates(d_norm, norm_phrase)
    for sig in forward_valid_sigs:
        pred_norm = _normalize_text(sig.name)
        if pred_norm in norm_phrase or norm_phrase == pred_norm:
            semantic_candidates.add(sig.name)
    evidence["semantic_candidates"] = sorted(semantic_candidates)

    # 3. Intersect with forward valid predicates
    matched_forward = semantic_candidates.intersection(forward_valid_names)
    evidence["matched_forward"] = sorted(matched_forward)

    if matched_forward:
        if len(matched_forward) == 1:
            pred_name = next(iter(matched_forward))
            exact = _normalize_text(pred_name) in norm_phrase or norm_phrase == _normalize_text(pred_name)
            return RelationInterpretationResult(
                status="EXACT_CANONICAL_MATCH" if exact else "LEXICAL_SEMANTIC_MATCH",
                category="PHYSICAL_VERIFIER",
                interpreted_predicates=(
                    InterpretedPredicate(subject_role, pred_name, object_role),
                ),
                direction_normalized=False,
                reason=f"Unique semantic match {pred_name} for forward endpoints",
                evidence=evidence,
            )
        else:
            # Multi-predicate check: text explicitly supports multiple distinct active predicates
            ordered_preds: list[InterpretedPredicate] = [
                InterpretedPredicate(subject_role, pred_name, object_role)
                for pred_name in sorted(matched_forward)
            ]
            return RelationInterpretationResult(
                status="MULTI_PREDICATE_MATCH",
                category="PHYSICAL_VERIFIER",
                interpreted_predicates=tuple(ordered_preds),
                direction_normalized=False,
                reason=f"Multi-predicate match {sorted(matched_forward)} explicitly evidenced in phrase",
                evidence=evidence,
            )

    # 5. Check direction normalization (reverse endpoints)
    reverse_valid_sigs = get_endpoint_valid_binary_predicates(
        d_norm,
        subject_role=object_role,
        object_role=subject_role,
        subject_kind=object_kind,
        object_kind=subject_kind,
    )
    reverse_valid_names = {sig.name for sig in reverse_valid_sigs}
    evidence["reverse_valid_predicates"] = sorted(reverse_valid_names)

    # Check inverse cues first, then standard cues with reverse valid
    inverse_candidates = _extract_inverse_semantic_candidates(d_norm, norm_phrase)
    evidence["inverse_candidates"] = sorted(inverse_candidates)

    matched_reverse_by_inverse = inverse_candidates.intersection(reverse_valid_names)
    matched_reverse_by_standard = semantic_candidates.intersection(reverse_valid_names)

    matched_reverse = matched_reverse_by_inverse or matched_reverse_by_standard
    evidence["matched_reverse"] = sorted(matched_reverse)

    if matched_reverse:
        if len(matched_reverse) == 1:
            pred_name = next(iter(matched_reverse))
            # Direction normalized: subject is object_role, object is subject_role
            return RelationInterpretationResult(
                status="DIRECTION_NORMALIZED",
                category="PHYSICAL_VERIFIER",
                interpreted_predicates=(
                    InterpretedPredicate(object_role, pred_name, subject_role),
                ),
                direction_normalized=True,
                reason=f"Semantic evidence supports direction normalization to {pred_name}({object_role}, {subject_role})",
                evidence=evidence,
            )
        else:
            return RelationInterpretationResult(
                status="AMBIGUOUS_RELATION",
                category="PHYSICAL_VERIFIER",
                interpreted_predicates=(),
                reason=f"Ambiguous reverse relation matches multiple predicates: {sorted(matched_reverse)}",
                evidence=evidence,
            )

    # 6. Check task/causal semantic relations if no physical verifier matched
    causal_match = _extract_task_causal_candidates(norm_phrase)
    if causal_match is not None:
        pred_name, is_inverse = causal_match
        evidence["task_causal_predicate"] = pred_name
        evidence["is_inverse"] = is_inverse
        if is_inverse:
            return RelationInterpretationResult(
                status="TASK_CAUSAL_SEMANTIC_MATCH",
                category="TASK_CAUSAL_SEMANTICS",
                interpreted_predicates=(
                    InterpretedPredicate(object_role, pred_name, subject_role),
                ),
                direction_normalized=True,
                reason=f"Task/causal semantic match {pred_name} (direction normalized) for ({object_role}, {subject_role})",
                evidence=evidence,
            )
        else:
            return RelationInterpretationResult(
                status="TASK_CAUSAL_SEMANTIC_MATCH",
                category="TASK_CAUSAL_SEMANTICS",
                interpreted_predicates=(
                    InterpretedPredicate(subject_role, pred_name, object_role),
                ),
                direction_normalized=False,
                reason=f"Task/causal semantic match {pred_name} for ({subject_role}, {object_role})",
                evidence=evidence,
            )

    # 7. If semantic candidates exist for physical predicates but none match valid endpoints
    if semantic_candidates:
        evidence["reason"] = "Semantic candidates not compatible with declared endpoint roles/kinds"
        status = "UNINTERPRETABLE_REQUIRED_RELATION" if required else "SOFT_OPTIONAL_RELATION"
        return RelationInterpretationResult(
            status=status,
            category="UNKNOWN",
            interpreted_predicates=(),
            reason=f"Semantic candidates {sorted(semantic_candidates)} incompatible with endpoints ({subject_role}, {object_role})",
            evidence=evidence,
        )

    # 8. Zero semantic evidence found: MUST FAIL CLOSED!
    # Even if len(forward_valid_names) == 1, we NEVER guess without semantic evidence.
    evidence["reason"] = "No semantic evidence in relation phrase matching active domain predicates or task causal semantics"
    status = "UNINTERPRETABLE_REQUIRED_RELATION" if required else "SOFT_OPTIONAL_RELATION"
    return RelationInterpretationResult(
        status=status,
        category="UNKNOWN",
        interpreted_predicates=(),
        reason=f"Relation text {raw_phrase!r} provides no semantic evidence for valid predicates {sorted(forward_valid_names)} or task causal semantics",
        evidence=evidence,
    )
