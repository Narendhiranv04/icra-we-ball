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


@dataclass(frozen=True)
class RelationSemanticCandidate:
    """Endpoint-independent meaning nominated by relation text."""

    predicate_name: str
    direction: str = "FORWARD"
    category: str = "PHYSICAL_VERIFIER"


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
        "supports refreshment setting", "support refreshment setting",
    ),
    ("living_room", "FITS_ON"): (
        "fits on", "placed on", "place on", "supported on", "support on", "rests on",
        "rest on", "tabletop for remote", "surface for remote", "remote on",
        "entertainment control on", "control device on", "support remote",
        "supports entertainment control", "support entertainment control",
    ),
    ("living_room", "NEAR_SEAT"): (
        "near seat", "near seating", "beside seat", "adjacent to seat", "close to seat",
        "accessible to seat", "convenient to seat", "at seating position", "viewer seating position",
        "near viewer", "beside viewer", "adjacent to viewer",
        "near", "nearby", "located near", "located adjacent", "located adjacent to", "is adjacent to",
        "physically adjacent to", "positioned adjacent to", "placed near", "placed adjacent to",
    ),
    ("living_room", "ACCESSIBLE_FROM_BOTH_SEATS"): (
        "accessible from both seats", "accessible from both", "reachable from both seats",
        "shared between seats", "between both seats", "serves both seats", "central to both seats",
        "accessible to pair of seats", "paired seating", "accessible from pair",
        "accessible to viewers", "accessible between seats",
        "accessible from", "accessible to", "within reach of", "reachable from",
        "accessible by all viewers", "is accessible by",
    ),

    # Workshop
    ("workshop", "COMPATIBLE_WITH"): (
        "compatible with", "compatible with fastener", "compatible with screw",
        "engages fastener", "engages screw", "engages head", "fits screw", "fits fastener",
        "fits head", "driver bit matches", "matches recess", "transmits torque",
        "transmit torque", "drives screw", "fastens with", "operates on", "used together",
        "mechanically engages", "engages with", "driver engages", "fits recess",
        "compatible tool", "matches fastener", "tightens", "rotates fastener",
        "physically engages with fastening interface", "interact mechanically with fastener",
        "mechanically compatible with", "engages fastening interface",
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
        "compatible with fastening target", "fits fastening interface",
    ),
}

# ---------------------------------------------------------------------------
# Morphology-tolerant semantic cues
# ---------------------------------------------------------------------------
#
# The contiguous phrase lists above match only wordings that have already been
# seen.  "stirs" was listed and "stir" was not, so an imperative carried no
# meaning at all; "suit_for", "must physically suit" and "is suitable for" all
# say compatibility and none of them was listed.  Rather than keep appending
# sentences, these patterns state the *stem inventory* of each meaning, so any
# inflection of a word that expresses it reads the same way.
#
# They nominate candidates only.  Which nomination survives is still decided by
# the predicate's endpoint signature, and a phrase that nominates nothing still
# fails closed -- so a generous stem list cannot invent a relation, it can only
# let an endpoint pair that admits exactly one predicate recognise the wording
# the model actually used.
_FIT_OR_COMPATIBILITY = r"\b(?:fit|suit|match|mate|compatib|conform|correspond|accommodat)\w*"
# Verb forms only.  A nominalisation is not a statement about the two roles:
# "must maintain 45 degree tilt during operation" names the activity and says
# nothing the runtime can verify, and reading "operation" as "operates on"
# turned an uninterpretable relation into a physical requirement.
_TOOL_USE = (
    r"\b(?:us(?:e|es|ed|ing|able)|appl(?:y|ies|ied|ying)|operat(?:e|es|ed|ing)|"
    r"manipulat(?:e|es|ed|ing)|handl(?:e|es|ed|ing)|wield(?:s|ed|ing)?|"
    r"employ(?:s|ed|ing)?|deploy(?:s|ed|ing)?)\b"
    r"|\bact(?:s|ed|ing)?\s+(?:up)?on\b"
    r"|\bwork(?:s|ed|ing)?\s+(?:on|with)\b"
)
# Fastening words only count where they are used as verbs.  The bare stem also
# forms the attributive gerund the model uses for *kinds* -- "the fastening
# component", "the fastening tool" -- and reading those as statements about the
# action made "the tool acts on the fastening component" a mechanical-fit claim
# instead of the causal statement it is.
_FASTENING_STEMS = (
    r"fasten|secur|attach|join|screw|bolt|instal|affix|mount|anchor|connect|"
    r"thread|rivet|clip|glu|weld|clamp|fix"
)
_FASTENING_ACTION = (
    rf"\b(?:{_FASTENING_STEMS})(?:s|es|ed|ing)?\b\s*"
    rf"(?:to|onto|into|in|at|with|together|against)\b"
    rf"|\b(?:{_FASTENING_STEMS})(?:s|es|ed)\b"
    rf"|\b(?:is|are|be|been|being|must|should|needs?|has|have|to)\s+"
    rf"(?:\w+\s+){{0,3}}?(?:{_FASTENING_STEMS})\w*"
)
_SUPPORT_PLACEMENT = (
    r"\b(?:plac|position|rest|support|situat|locat|set|put|lay|stand|arrang|deposit|leav|left)\w*"
    r"[^.;:]{0,30}?\b(?:on|onto|upon|atop|over)\b"
    r"|\b(?:on|onto|upon|atop)\s+(?:the\s+|a\s+|an\s+)?(?:\w+\s+){0,2}?"
    r"(?:surface|table|tabletop|top|platform|stand|shelf|shelves|counter|countertop|"
    r"tray|bench|workbench|desk|region|area|zone)\b"
    r"|\b(?:support|hold|bear|carr|underneath|beneath)\w*"
)
_PROXIMITY = (
    r"\b(?:near|nearby|beside|adjacen\w*|proximate|proximity|alongside|"
    r"next to|close to|by the side of|at hand)\b"
)
_ACCESSIBILITY = r"\b(?:access|reachab|reach)\w*|\bwithin reach\b"
# "Between" is left out deliberately: it states where a thing sits relative to
# two seats, which the domain already has its own predicate for, and reading it
# as accessibility replaced a positional claim with a different one.
_SHARED_BY_TWO = (
    r"\b(?:shared|share|shares|sharing|common|mutual|joint|either|both|"
    r"each of the two|all (?:the )?(?:people|viewers|occupants|users))\b"
)

_SEMANTIC_PREDICATE_PATTERNS: dict[tuple[str, str], tuple[str, ...]] = {
    ("kitchen", "INSERTABLE_IN"): (
        r"\b(?:insert|submerg|immers|dip|plung|introduc|lower)\w*",
        _FIT_OR_COMPATIBILITY,
        r"\b(?:go|goes|going|plac|put|set)\w*[^.;:]{0,30}?\b(?:in|into|inside|within)\b",
        r"\b(?:stir|mix|agitat|whisk|blend|swirl|beat)\w*",
    ),
    ("kitchen", "REACHES_BOTTOM"): (
        r"\b(?:reach|touch|extend|contact)\w*[^.;:]{0,25}?\b(?:bottom|base|floor|full depth)\b",
        r"\b(?:deep|long)\s+enough\b",
        r"\b(?:sufficient|adequate|enough)\s+(?:depth|length|reach)\b",
        r"\b(?:full|entire|whole)\s+(?:depth|length)\b",
        r"\b(?:stir|mix|agitat|whisk|blend)\w*[^.;:]{0,30}?\b(?:bottom|base|throughout|thoroughly)\b",
    ),
    ("living_room", "FITS_SET_ON"): (_SUPPORT_PLACEMENT, _FIT_OR_COMPATIBILITY),
    ("living_room", "FITS_ON"): (_SUPPORT_PLACEMENT, _FIT_OR_COMPATIBILITY),
    ("living_room", "NEAR_SEAT"): (
        _PROXIMITY,
        r"\b(?:for|serv|assign|allocat|dedicat|belong|own|personal|individual|"
        r"respective|each)\w*[^.;:]{0,30}?"
        r"\b(?:seat|seating|chair|sofa|couch|person|people|occupant|viewer)s?\b",
    ),
    ("living_room", "ACCESSIBLE_FROM_BOTH_SEATS"): (_ACCESSIBILITY, _SHARED_BY_TWO),
    # In the workshop the three physical verifiers are separated by their
    # endpoints alone -- a tool with a fastener, a tool with the site, a
    # fastener with the site -- so one fastening vocabulary serves all three and
    # the signature decides which relation a sentence stated.
    # Tool-use wording -- "the tool acts on the component", "the tool is used to
    # manipulate it" -- is causal semantics about what the task does, and the
    # domain keeps that apart from the mechanical fit the runtime verifies.  Only
    # fastening and fit vocabulary nominates a physical verifier here.
    ("workshop", "COMPATIBLE_WITH"): (
        _FASTENING_ACTION, _FIT_OR_COMPATIBILITY,
        r"\b(?:driv|tighten|turn|rotat|torqu|engag)\w*",
    ),
    # Tool-use wording *is* admissible for reach, and only for reach: the domain
    # has no causal predicate between a tool and the site, so there is no
    # separation to preserve, and needing to get at the place is the robot's own
    # precondition for acting there.  "The tool acts upon the fastening
    # location" is a statement about reach and nothing else.
    ("workshop", "REACHES_TARGET"): (
        _FASTENING_ACTION, _FIT_OR_COMPATIBILITY, _ACCESSIBILITY, _TOOL_USE,
        r"\b(?:driv|tighten|turn|rotat|torqu|engag)\w*",
    ),
    ("workshop", "COMPATIBLE_WITH_TARGET"): (
        _FASTENING_ACTION, _FIT_OR_COMPATIBILITY,
        r"\b(?:into|onto|in|at)\s+(?:the\s+|a\s+)?(?:\w+\s+){0,2}?"
        r"(?:hole|recess|joint|slot|site|location|target|spot|point|workpiece|surface|position)\b",
    ),
}

# Support and placement wordings say which of two things is the surface only
# through the roles involved, so the same stems are offered in both directions
# and the endpoint signature settles it.
_INVERSE_DIRECTION_PATTERNS: dict[tuple[str, str], tuple[str, ...]] = {
    ("kitchen", "INSERTABLE_IN"): (
        r"\b(?:contain|hold|enclos|hous|receiv|accept|keep)\w*",
        r"\b(?:is|are|be|been|being)\s+(?:\w+\s+){0,2}?fill\w*",
    ),
    ("living_room", "FITS_SET_ON"): (_SUPPORT_PLACEMENT,),
    ("living_room", "FITS_ON"): (_SUPPORT_PLACEMENT,),
    ("workshop", "COMPATIBLE_WITH"): (
        r"\b(?:driven|engaged|fastened|tightened|turned|operated|applied|used|installed)\s+by\b",
    ),
    ("workshop", "COMPATIBLE_WITH_TARGET"): (
        r"\b(?:receiv|accept|host|take)\w*[^.;:]{0,25}?"
        r"\b(?:fastener|screw|bolt|component|part|fixing)s?\b",
        r"\bthreaded for\b",
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
    ("living_room", "FITS_SET_ON"): (
        "supported by", "supported on", "placed on", "rests on", "located upon",
    ),
    ("living_room", "FITS_ON"): (
        "supported by", "supported on", "placed on", "rests on", "located upon",
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

    for (d, pred_name), patterns in _SEMANTIC_PREDICATE_PATTERNS.items():
        if d != d_norm or pred_name in candidates:
            continue
        if any(re.search(pattern, norm_phrase) for pattern in patterns):
            candidates.add(pred_name)

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
    for (d, pred_name), patterns in _INVERSE_DIRECTION_PATTERNS.items():
        if d != d_norm or pred_name in candidates:
            continue
        if any(re.search(pattern, norm_phrase) for pattern in patterns):
            candidates.add(pred_name)
    return candidates


def extract_relation_semantic_candidates(
    domain: str,
    raw_phrase: str,
) -> tuple[RelationSemanticCandidate, ...]:
    """Extract relation meanings from text without consulting endpoint roles.

    Endpoint signatures are deliberately applied later by the compiler's joint
    role resolver.  Text must nominate every candidate returned here.
    """
    norm_phrase = _normalize_text(raw_phrase)
    if not norm_phrase:
        return ()
    candidates = [
        RelationSemanticCandidate(name, "FORWARD")
        for name in sorted(_extract_semantic_candidates(domain, norm_phrase))
    ]
    candidates.extend(
        RelationSemanticCandidate(name, "REVERSE")
        for name in sorted(_extract_inverse_semantic_candidates(domain, norm_phrase))
        if (name, "REVERSE") not in {(c.predicate_name, c.direction) for c in candidates}
    )
    for name, reverse in all_task_causal_candidates(norm_phrase):
        candidates.append(RelationSemanticCandidate(
            name, "REVERSE" if reverse else "FORWARD", "TASK_CAUSAL_SEMANTICS"
        ))
    effect = interpret_task_effect_predicate(raw_phrase)
    if effect:
        candidates.append(RelationSemanticCandidate(effect, "FORWARD", "TASK_EFFECT_SEMANTICS"))
    # A verb separated from its particle nominates the same meaning as the
    # adjacent form; it is added rather than substituted, so a sentence that
    # matches both ways yields one candidate.
    for predicate, category in _extract_verb_particle_candidates(norm_phrase):
        candidates.append(RelationSemanticCandidate(predicate, "FORWARD", category))
    return tuple(dict.fromkeys(candidates))


# Deterministic linguistic cues for task/causal semantic relations
_TASK_CAUSAL_RELATION_CUES: dict[str, tuple[str, ...]] = {
    "PROVIDES_MATERIAL_TO": (
        "provides material to", "provides material into", "supplies material to", "supplies to",
        "transfers material to", "transfers content to", "transfers contents to", "pours into", "poured into",
        "dispenses into", "fills", "provides contents to", "provides ingredients to", "source for",
        "provides coffee to", "provides water to", "provides soup to", "poured in", "pours in",
        "supplies", "feeds into", "supplies material into",
        "provides fluid to", "provides fillable fluid to", "provides content to",
        "provides content for", "provides coffee for blend", "provides water for blend",
        "supplies fluid to", "supplies solid to",
        # Combining and mixing are how the model usually words a transfer into a
        # vessel.  The symmetric ones appear in the inverse list too, because
        # "combined with" asserts no direction and the endpoint signature is
        # what decides which way round the runtime can read it.
        "combine into", "combines into", "combined into", "combine in", "combined in",
        "added to", "adds to", "add to", "added into", "adds into",
        "mixed into", "mixes into", "mixed with", "mixes with", "combined with",
        "combine with", "combines with", "mixes", "mixed",
    ),
    "ACTS_ON": (
        "acts on", "operates on", "manipulates", "manipulates interior of", "stirs contents of",
        "stirs contents", "works on", "applies to", "interacts with", "stirs", "mixes",
        "mixes contents of", "stirs interior of", "manipulates fastener", "manipulates component",
        "manipulate or install", "drives", "fastens",
        "used to stir", "used for stirring", "performs mixing action in",
        "acts upon during fastening", "physically acts upon", "applies force to",
        "performs securement action on", "actuates connection between", "capable of securing",
    ),
    "PAIRED_WITH": (
        "paired with", "is paired with", "accompanied by", "alongside", "served with",
        "served alongside", "complements", "set with", "provided alongside",
        "arranged with", "provided for", "arranged for", "placed with",
        "must accompany", "accompanied by", "provided with", "accommodates eating utensil",
        "associated for serving", "is served with", "placed adjacent to serving of",
        "associated with serving", "associated with servicing", "associated with soup", "assigned to serve",
        # Ordinary ways to say the same thing, each one a frozen trial's wording.
        "accompanies", "accompany", "to accompany", "goes with", "comes with",
    ),
    "INSTALLED_AT": (
        "installed at", "installed in", "secured at", "secured in", "fastened at", "fastened in",
        "anchored at", "anchored in", "attached to", "mounted at", "placed at target",
        "installed on", "secured on", "attaches to", "mechanically attaches to",
        "secures to", "physically attached to during fastening",
    ),
    "CONNECTED_TO": (
        "connects to", "connected to", "joins with", "joined with", "fastened together",
    ),
    "SITUATED_BETWEEN": (
        "situated between", "located between", "positioned between", "between seats",
    ),
}

# Causal meanings by stem inventory, for the same reason as the physical
# verifiers above: "stirs" was listed but "stir" was not, and "combine",
# "uses_for_action" and "equipped_with" say things the exact lists never
# happened to contain.  Each family stays lexically distinct from the others,
# because two causal predicates can share an endpoint pair and then only the
# wording tells them apart.
_TASK_CAUSAL_RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "PROVIDES_MATERIAL_TO": (
        r"\b(?:pour|transfer|dispens|decant|empt|tip|ladl|scoop)\w*",
        r"\b(?:combin|blend|incorporat)\w*",
        r"\b(?:add|adds|added|adding)\b",
        r"\b(?:suppl|provid|feed|deliver|contribut|sourc)\w*",
        r"\b(?:fill|fills|filled|filling)\b",
        # A bare mention of contents is not a transfer; a directional particle
        # after it is what makes "combine ingredients into the cup" one.
        r"\b(?:ingredient|material|content|constituent)s?\b[^.;:]{0,20}?\b(?:into|onto|to|in)\b",
    ),
    "ACTS_ON": (
        r"\b(?:stir|agitat|whisk|swirl|beat|churn|fold)(?:s|es|ed|ing)?\b",
        r"\b(?:mix|mixes|mixed|mixing)\b",
        _TOOL_USE,
        r"\b(?:driv(?:e|es|en|ing)|tighten(?:s|ed|ing)?|turn(?:s|ed|ing)?|"
        r"rotat(?:e|es|ed|ing)|torqu(?:e|es|ed|ing))\b",
    ),
    "PAIRED_WITH": (
        # "Associated with" on its own is too vague to be a pairing claim; the
        # domain requires an explicit pairing operation to corroborate it, so it
        # is not nominated here.
        r"\b(?:pair|accompan|complement|equip)\w*",
        r"\b(?:has|have|having|includ|compris|com(?:e|es))\w*"
        r"[^.;:]{0,20}?\b(?:part|utensil|spoon|fork|knife|cutlery|implement|accessor)\w*",
        r"\b(?:serv|provid|arrang|plac|set|offer)\w*[^.;:]{0,25}?"
        r"\b(?:with|alongside|together with|beside|next to|adjacent to)\b",
        r"\b(?:is|are|be|goes|go)\s+(?:served\s+)?(?:together\s+)?with\b",
        r"\b(?:alongside|together with)\b",
    ),
    "INSTALLED_AT": (_FASTENING_ACTION,),
    "CONNECTED_TO": (
        r"\b(?:connect|join|coupl|link|unit|bond|weld)(?:s|es|ed|ing)?\b\s*"
        r"(?:to|with|at|into|together)\b",
    ),
    "SITUATED_BETWEEN": (
        r"\bbetween\b",
    ),
}

_TASK_CAUSAL_INVERSE_PATTERNS: dict[str, tuple[str, ...]] = {
    "PROVIDES_MATERIAL_TO": (
        r"\b(?:receiv|obtain|tak)\w*[^.;:]{0,25}?\bfrom\b",
        r"\b(?:fill|suppl|provid|prepar|mad|mak|brew|produc|form|creat|constitut)\w*"
        r"\s+(?:up\s+)?(?:by|from|with|of|out of)\b",
        r"\b(?:combined|mixed|blended)\s+from\b",
    ),
    "ACTS_ON": (
        r"\b(?:act|operat|manipulat|work|handl|us|appl|stir|mix|agitat|driv|turn)\w*"
        r"\s+(?:up)?(?:on\s+)?by\b",
    ),
}


_TASK_CAUSAL_INVERSE_CUES: dict[str, tuple[str, ...]] = {
    "PROVIDES_MATERIAL_TO": (
        "receives material from", "receives contents from", "receives coffee from",
        "receives water from", "receives ingredients from", "receives from", "filled by",
        "supplied by", "receives liquid from",
        "receives transfer from", "receiving from", "transferred from",
        "requires contents from", "receives content from",
        "receives ingredient from", "receives food from",
        # The carrier described by what it was made from.
        "prepared from", "prepared with", "prepared by combination",
        "prepared by combination of", "made from", "made with", "made of",
        "combined from", "mixed from", "brewed from", "brewed with",
        "mixed with", "mixes with", "combined with", "combine with", "combines with",
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
    "PLACED_ON": (
        "on", "placed on", "placed upon", "positioned on", "supported on", "supported by",
        "rests on", "rests upon", "located upon", "located on", "must be placed on",
    ),
}

# The same stem-inventory treatment for end states.  "contains" was listed and
# "contained_in" was not, so the commonest way a model states that a transfer
# has happened carried no meaning at all.
_TASK_EFFECT_RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "CONTAINS": (
        r"\b(?:contain|enclos|hous)\w*",
        r"\bfill\w*",
        # "Hold" alone is not containment: a surface holds the items standing on
        # it as readily as a vessel holds what is in it, and reading "can hold
        # the drinkware set" as containment added an end state the model never
        # stated.  It counts when what is held is contents rather than an object.
        r"\bhold\w*\s+(?:\w+\s+){0,2}?(?:contents?|material|ingredient|liquid|substance)s?\b",
        r"\bhas\s+(?:the\s+)?(?:contents|material|ingredients?)\b",
        r"\b(?:is|are|be|been|being)\s+(?:\w+\s+){0,2}?(?:in|inside|within)\b",
    ),
    "PLACED_ON": (
        r"\b(?:plac|position|rest|support|situat|locat|set|put|lay|stand|leav|left|deposit)\w*"
        r"[^.;:]{0,25}?\b(?:on|onto|upon|atop|over)\b",
    ),
}

# A statement the model hedged is a guess about how the scene already is, not a
# requirement the task imposes: "the fastener is potentially contained in the
# cupboard" says where to look, and treating it as an end state the robot must
# bring about invented a requirement the instruction never made.
_HEDGED_CURRENT_STATE = re.compile(
    r"\b(?:potential|possib|probab|likel|perhaps|maybe|presumab|suspect|apparent|"
    r"seem|appear|may|might|could)\w*\b",
    re.I,
)


def relation_states_a_hedged_possibility(raw_phrase: str) -> bool:
    """Whether the wording marks itself as a guess about the current scene."""
    return bool(_HEDGED_CURRENT_STATE.search(_normalize_text(raw_phrase)))


# Where things are now, as opposed to where the task must put them.  "The
# fastener and tool are found inside the storage container" is the model saying
# where to look; read as a requirement it asked the robot to put its tools back
# into a cupboard, which no instruction ever did.
_STATES_PRESENT_WHEREABOUTS = re.compile(
    r"\b(?:is|are|was|were|be|being|been)\s+(?:\w+\s+){0,2}?"
    r"(?:found|discover|locat|situat|kept|stor|held|sitting|resting|hidden|conceal)\w*\s+"
    r"(?:in|inside|within|on|at|under)\b"
    r"|\b(?:found|discovered)\s+(?:in|inside|within|at|on)\b"
    r"|\b(?:in|inside|within)\s+(?:the\s+|a\s+|an\s+)?(?:\w+\s+){0,2}?"
    r"(?:storage|cabinet|cupboard|drawer|box|container|bin|chest|closet)s?\b",
    re.I,
)


def relation_states_where_things_currently_are(raw_phrase: str) -> bool:
    """Whether the wording describes the present scene rather than a requirement."""
    normalized = _normalize_text(raw_phrase)
    return bool(
        _STATES_PRESENT_WHEREABOUTS.search(normalized)
        or _HEDGED_CURRENT_STATE.search(normalized)
    )


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
    for predicate, patterns in _TASK_EFFECT_RELATION_PATTERNS.items():
        if any(re.search(pattern, norm_phrase) for pattern in patterns):
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
    if predicate not in {"CONTAINS", "PLACED_ON"}:
        return False, None
    for group in groups:
        source = group.get("tool_role") or group.get("source_role")
        target = group.get("target_role")
        phrase = _normalize_text(group.get("function") or group.get("operation") or "")
        contains_match = (
            predicate == "CONTAINS"
            and source == raw_object
            and target == raw_subject
            and any(re.search(r"\b" + re.escape(cue) + r"\b", phrase) for cue in _TRANSFER_OPERATION_CUES)
        )
        placement_match = (
            predicate == "PLACED_ON"
            and source == raw_subject
            and target == raw_object
            and bool(re.search(r"\b(place|transfer|relocate|position|distribute|move|support)\w*\b", phrase))
        )
        if contains_match or placement_match:
            return True, str(group.get("id", "")) or None
    return False, None


def has_compatible_explicit_pairing_operation(
    raw_phrase: str,
    raw_subject: str,
    raw_object: str,
    groups: Sequence[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Corroborate otherwise-generic association text with an explicit FM operation."""
    if _normalize_text(raw_phrase) != "associated with":
        return False, None
    for group in groups:
        source = group.get("tool_role") or group.get("source_role")
        target = group.get("target_role")
        operation = _normalize_text(group.get("function") or group.get("operation") or "")
        if (
            source == raw_subject
            and target == raw_object
            and re.search(r"\b(pair|provide|place|serve|accompany|arrange|associate)\w*\b", operation)
        ):
            return True, str(group.get("id", "")) or None
    return False, None


# ---------------------------------------------------------------------------
# Verb-particle relations, where the verb and its preposition are not adjacent
# ---------------------------------------------------------------------------
#
# English routinely separates a phrasal verb from its particle: "combined to
# form a drink IN a cup", "placed neatly ON the table", "joined firmly AT the
# marked location".  A contiguous cue list cannot match any of those, so a
# sentence carrying perfectly ordinary transfer or placement semantics was read
# as carrying none, and the relation became an unrepresentable requirement.
#
# These are linguistic families, not phrases: a set of verb stems and the set
# of directional particles that family takes, matched across a bounded gap
# inside one clause.  A sentence boundary always stops the match, so the verb
# and the particle have to belong to the same statement.
_VERB_PARTICLE_FAMILIES: tuple[tuple[str, str, str, str], ...] = (
    # (predicate, category, verb stems, directional particles)
    ("PROVIDES_MATERIAL_TO", "TASK_CAUSAL_SEMANTICS",
     r"combin|mix|blend|pour|transfer|dispens|fill|decant|top up",
     r"into|onto|in|to"),
    ("PLACED_ON", "TASK_EFFECT_SEMANTICS",
     r"plac|put|set|position|rest|arrang|lay|stand|seat|deposit",
     r"on|onto|upon|atop|over"),
    ("INSTALLED_AT", "TASK_CAUSAL_SEMANTICS",
     r"instal|fasten|secur|screw|bolt|affix|attach|mount|driv",
     r"at|into|in|onto|on|through"),
    ("CONNECTED_TO", "TASK_CAUSAL_SEMANTICS",
     r"connect|join|coupl|coupled|link|coupling",
     r"to|with|at|into|together"),
)

# The gap may not cross a clause boundary, and it may not swallow a second verb
# phrase; forty characters is about one noun phrase with a modifier.
_PARTICLE_GAP = r"[^.;:]{0,40}?"


def _extract_verb_particle_candidates(norm_phrase: str) -> list[tuple[str, str]]:
    """Predicates whose verb and directional particle both appear in one clause."""
    found: list[tuple[str, str]] = []
    for predicate, category, verbs, particles in _VERB_PARTICLE_FAMILIES:
        pattern = (rf"\b(?:{verbs})\w*\b{_PARTICLE_GAP}\b(?:{particles})\b")
        if re.search(pattern, norm_phrase):
            found.append((predicate, category))
    return found

def _extract_task_causal_pattern_candidates(norm_phrase: str) -> list[tuple[str, bool]]:
    """Every causal meaning the stem inventories nominate, inverse readings first."""
    found: list[tuple[str, bool]] = []
    for pred, patterns in _TASK_CAUSAL_INVERSE_PATTERNS.items():
        if any(re.search(pattern, norm_phrase) for pattern in patterns):
            found.append((pred, True))
    inverse_named = {pred for pred, _ in found}
    for pred, patterns in _TASK_CAUSAL_RELATION_PATTERNS.items():
        if pred in inverse_named:
            continue
        if any(re.search(pattern, norm_phrase) for pattern in patterns):
            found.append((pred, False))
    return found


def all_task_causal_candidates(norm_phrase: str) -> list[tuple[str, bool]]:
    """The exact-cue reading, if any, together with every stem-nominated one."""
    found: list[tuple[str, bool]] = []
    exact = _extract_task_causal_candidates(norm_phrase)
    if exact is not None:
        found.append(exact)
    for candidate in _extract_task_causal_pattern_candidates(norm_phrase):
        if candidate[0] not in {pred for pred, _ in found}:
            found.append(candidate)
    return found


def causal_endpoints_admit(
    domain: str, predicate: str, subject_role: str, object_role: str, inverse: bool
) -> bool:
    """Whether the runtime lets this causal predicate relate these two roles.

    Used only to choose between meanings the wording already nominated; a
    predicate no wording nominated is never reachable from here.
    """
    from .semantic_typing import causal_predicate_endpoint_pairs

    pairs = causal_predicate_endpoint_pairs(domain, predicate)
    if not pairs:
        return False
    ordered = (object_role, subject_role) if inverse else (subject_role, object_role)
    return ordered in pairs


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
    semantic_candidates = {
        candidate.predicate_name
        for candidate in extract_relation_semantic_candidates(d_norm, raw_phrase)
        if candidate.category == "PHYSICAL_VERIFIER" and candidate.direction == "FORWARD"
    }
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

    # 6. Check task/causal semantic relations if no physical verifier matched.
    # A wording can nominate several causal meanings at once -- "provided with"
    # is both a transfer and an accompaniment -- and two causal predicates can
    # share an endpoint pair.  Where the wording is plural, the runtime's own
    # causal signature is allowed to choose between meanings the text already
    # nominated, and anything still plural fails closed rather than being
    # settled by the order the names happen to sort in.
    causal_candidates = all_task_causal_candidates(norm_phrase)
    causal_match: tuple[str, bool] | None = None
    if len(causal_candidates) == 1:
        causal_match = causal_candidates[0]
    elif causal_candidates:
        admitted = [
            candidate for candidate in causal_candidates
            if causal_endpoints_admit(d_norm, candidate[0], subject_role, object_role, candidate[1])
        ]
        evidence["task_causal_candidates"] = [
            {"predicate": pred, "inverse": inv} for pred, inv in causal_candidates
        ]
        evidence["task_causal_admitted"] = [
            {"predicate": pred, "inverse": inv} for pred, inv in admitted
        ]
        if len(admitted) == 1:
            causal_match = admitted[0]
        else:
            status = "AMBIGUOUS_RELATION" if required else "SOFT_OPTIONAL_RELATION"
            return RelationInterpretationResult(
                status=status,
                category="UNKNOWN",
                interpreted_predicates=(),
                reason=(
                    f"Relation text {raw_phrase!r} nominates several task causal meanings "
                    f"{sorted({pred for pred, _ in causal_candidates})} and the endpoints "
                    f"({subject_role}, {object_role}) do not single one out"
                ),
                evidence=evidence,
            )
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
