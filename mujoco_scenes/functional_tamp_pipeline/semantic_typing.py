"""Pure FM semantic hypothesis construction, independent of compilation and G_O."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable

from .models import RoleTypeHypothesis
from .predicate_registry import get_predicate_signature
from .relation_interpreter import extract_relation_semantic_candidates
from .robot_capability_registry import extract_operation_semantic_candidates
from .system_context_registry import (
    AREA_ENTITY_KINDS,
    CARRIED_ENTITY_KINDS,
    get_domain_planner_context_constants,
    get_domain_selectable_roles,
    get_domain_system_fixed_anchors,
    get_runtime_role_entity_kind,
)


@dataclass(frozen=True)
class FunctionSemanticEvidence:
    preferred_roles: tuple[str, ...] = ()
    excluded_roles: tuple[str, ...] = ()
    explicit_families: tuple[str, ...] = ()
    evidence_strength: str = "UNKNOWN"
    provenance: str = "FM_FUNCTION_TEXT"
    evidence_source: str = "FUNCTION_TEXT"
    family_precedence_rules: tuple[str, ...] = ()
    runtime_context_only: bool = False


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", str(value)).lower()).strip()


def role_text_scopes(role: dict[str, Any]) -> dict[str, str]:
    """Split one FM role's wording into the scopes that mean different things.

    PURPOSE is the job the model says the participant does; IDENTITY is what
    kind of thing it says could serve; NAME is the identifier it chose.  The
    three disagree often enough to matter: a role whose purpose is "holds
    refreshments" is a support when its identity is a table and a payload when
    its identity is a cup.  Reading only whichever field happened to contain a
    keyword -- the previous behaviour -- discarded exactly the evidence that
    settles such cases, so a seat declared with chair and sofa categories
    yielded no seating evidence at all.
    """
    return {
        "PURPOSE_TEXT": _normalize(
            " ".join((str(role.get("function", "")), str(role.get("description", ""))))
        ),
        "IDENTITY_TEXT": _normalize(" ".join((
            " ".join(role.get("candidate_categories", ()) or ()),
            " ".join(role.get("required_properties", ()) or ()),
        ))),
        "NAME_TEXT": _normalize(str(role.get("id", ""))),
    }


def _text(role: dict[str, Any]) -> str:
    """Whole-role wording: every scope, in a stable order."""
    scopes = role_text_scopes(role)
    return _normalize(" ".join(
        scopes[key] for key in ("PURPOSE_TEXT", "NAME_TEXT", "IDENTITY_TEXT") if scopes[key]
    ))


# ---------------------------------------------------------------------------
# Role semantic families
# ---------------------------------------------------------------------------
# Each family is nominated by ordinary-language cues, scored per text scope so
# that precedence rules below can ask *where* the evidence came from.  Nothing
# here names a benchmark variant, an expected role list, or a scene object.

# A fastener noun heading a receiving feature -- "screw hole", "bolt recess" --
# names the place the fastener goes, so it must not read as the fastener itself.
_NOT_A_RECEIVING_FEATURE = (
    r"(?!\s+(?:holes?|recess(?:es)?|seats?|sites?|slots?|threads?|openings?|bores?|locations?))"
)

# What kind of thing an implement is.  The action words below appear just as
# readily in the passive -- "liquid to be mixed into the beverage", "holds the
# coffee mixture" -- where they describe what happens *to* the role, so an
# action alone never makes something an implement.
_IMPLEMENT_KIND = re.compile(
    r"\b(tools?|implements?|instruments?|utensils?|cutlery|"
    r"spoons?|forks?|knives|knife|stirrers?|whisks?|sticks?|spatulas?|ladles?|"
    r"drivers?|screwdrivers?|wrench(?:es)?|drills?|pliers?|spanners?)\b",
    re.I,
)

# What an implement is *for*: the model states the action, not the thing acted on.
_IMPLEMENT_ACTION = re.compile(
    r"\b(stir\w*|mix\w*|blend\w*|agitat\w*|whisk\w*|swirl\w*|"
    r"eat\w*|scoop\w*|"
    r"driv\w*|tighten\w*|fasten\w*|screw\w*|apply\w*|"
    r"used (?:to|for) (?:stir|mix|eat|drive|turn|tighten|apply|fasten)\w*)\b",
    re.I,
)


_FAMILY_CUES: dict[str, str] = {
    # "Ingredient" only says source when the role is the ingredient or supplies
    # one.  A container that "receives coffee and water ingredients" mentions the
    # material it takes in, and reading that as supply made the cup the source.
    "SOURCE": (
        r"\b(sources?|supply|supplies|supplied|supplying|provider|provides material|dispens\w+|"
        r"ingredient (?:source|supply|for|of)|(?:coffee|water|soup|food|material) ingredients?|"
        r"(?:raw|dry|solid|liquid|powdered) ingredients?|"
        r"raw material|stock of|reservoir|material for (?:the )?(?:coffee|drink|beverage|soup)|"
        r"holding (?:dry|drinking|raw)|contains? (?:dry )?(?:coffee|water|material|ingredient|liquid)|"
        r"liquid base|base for (?:the )?(?:coffee|drink|beverage)|"
        # The name of a material is a name for the thing it comes out of.  A
        # role offered as "coffee beans" or "broth" is the model describing a
        # supply, and reading it as no kind of participant at all left it with
        # every canonical role as a candidate -- which is neither a reading nor
        # an honest refusal.  Whether this domain *has* a source role for that
        # material is decided further down, and separately.
        r"broths?|stews?|curr(?:y|ies)|"
        r"instant coffee|ground coffee|coffee (?:beans?|grounds?|powder|granules)|"
        r"(?:tap|bottled|filtered|drinking|hot|boiling) water|water supply|liquid solvent|"
        r"(?:soup|coffee|food|drink|water) (?:substance|portion|contents?|material))\b"
    ),
    "DESTINATION": (
        r"\b(receiv(?:e|es|ing|er)|destination|containers?|vessels?|receptacles?|"
        r"cups?|mugs?|glass(?:es)?|bowls?|dish(?:es)?|tumblers?|"
        r"prepared (?:coffee|soup|drink|beverage)|served (?:coffee|soup)|"
        r"holds? (?:coffee|soup|liquid|food)|holds? (?:coffee|soup) for|"
        r"into which|drinkable size|serving (?:cup|bowl|dish))\b"
    ),
    "INSTRUMENT": (
        r"\b(tools?|implements?|instruments?|utensils?|drivers?|screwdrivers?|wrench(?:es)?|"
        r"drills?|pliers?|spanners?|applicators?|spoons?|forks?|knives|knife|stirrers?|whisks?|"
        r"stir\w*|mix\w*|agitat\w*|blend\w*|drive\w*|tighten\w*|turn(?:s|ing)? (?:the )?(?:screw|bolt|fastener)|"
        r"used (?:to|for) (?:stir|mix|eat|drive|turn|tighten|apply|fasten)\w*)\b"
    ),
    "SUPPORT": (
        r"\b(tables?|tabletops?|surfaces?|platforms?|supports?|supporting|stands?|shelves|shelf|"
        r"counters?|countertops?|desks?|workbench(?:es)?|benchtops?|trays?|"
        r"placement area|central area|staging area|work ?space|"
        r"flat ?top|planar|place for (?:refreshments?|items?|the (?:cup|tool|remote))|"
        r"holds? items?|area (?:for|where) (?:placing|items|the tool)|rest\w* (?:place|area))\b"
    ),
    "SEATING": (
        r"\b(chairs?|armchairs?|sofas?|couch(?:es)?|stools?|settees?|recliners?|seats?|seating|"
        r"sits?|sitting|sitter|seated|occupants?|"
        r"where (?:a |the )?(?:person|people|viewer|occupant)s? (?:sit|sits|will sit)|"
        r"(?:person|people|viewer|occupant)s? (?:sit|sits)|for sitting|viewer position)\b"
    ),
    "COMPONENT": (
        r"(?:\b(?:screws?|bolts?|fasteners?|nuts?|washers?|clips?|rivets?|dowels?|pins?)\b"
        + _NOT_A_RECEIVING_FEATURE + r")|"
        r"\b(joining (?:element|component|part|piece)|connecting (?:element|component|part)|"
        r"connectors?|hardware items?|installed component|"
        r"component to be (?:installed|secured|attached|fastened)|"
        # Something the model says is to be fastened *onto something else* is
        # the thing being fastened.  The preposition is what carries it: "the
        # place where the fastening occurs" contains the same participle and
        # names the site instead.
        r"(?:to be|must be|needs? to be|need to be|is|are|being)\s+(?:\w+\s+){0,3}?"
        r"(?:fasten|attach|join|secur|connect|affix|mount|install)\w*\s+"
        r"(?:to|at|onto|into|against)\b)"
    ),
    "FIXED_TARGET": (
        r"\b(marked (?:spot|point|position|location|site|target|joint)|"
        r"fastening (?:site|location|point|spot|interface)|repair target|fixtures?|"
        r"workpieces?|assembly receiv(?:ing|er)|"
        r"(?:screw|bolt|fastener|pilot|mounting|joint|threaded)\s+(?:holes?|recess(?:es)?|seats?|slots?|bores?)|"
        r"connection points?|attachment points?|"
        r"need\w* fastening|requir\w* fastening|destination for fastening|"
        # The plainest possible statement of a receiving site, which previously
        # had to be spelled out with a noun in front of it to be recognized.
        r"receiv\w*\s+(?:the\s+|a\s+)?(?:fastener|screw|bolt|component|part|fixing)s?|"
        r"where (?:the )?fastening (?:must )?(?:occur|occurs|happens|takes place)|"
        r"(?:object|part|assembly|location|spot|place)[^.]{0,40}(?:to be|receiv\w+|requiring|where the|where)[^.]{0,40}fasten\w*|"
        r"object (?:to be|receiving) secured|fixed (?:workpiece|point|target|receiving (?:location|site|target)))\b"
    ),
    "PAYLOAD": (
        r"\b(payloads?|refreshment set(?:ting)?s?|refreshment stations?|cup and saucer|drinkware|"
        r"cups?|mugs?|saucers?|plates?|dinnerware|tableware|crockery|drinking vessels?|"
        r"remote controls?|remotes?|media controls?|entertainment control(?:ler)?s?|controllers?|"
        r"tv remote|television remote|"
        r"device (?:used to|for controlling|to control)|controlling (?:television|tv|display)|"
        r"control(?:s|ler)? (?:the )?(?:television|tv|display)|consumables|operates tv)\b"
    ),
    "DISPLAY_CONTEXT": r"\b(televisions?|tv|tv screen|displays?|monitors?|screens?)\b",
    "STORAGE_CONTEXT": (
        r"\b(cabinets?|cupboards?|drawers?|storage bins?|"
        r"storage (?:unit|structure|space|area|region|spot)s?|"
        r"enclosed space|openable|can be opened|closed structure)\b"
    ),
}

# "X to be fastened TO Y" names X, the thing being fastened; "the place where
# the fastening occurs" names Y, the site receiving it.  The participle alone
# cannot tell them apart -- both sentences contain "fastened" -- so what
# separates them is whether a preposition follows it, pointing at something
# else the fastening happens against.
# Wording that states the defining function of a role the runtime distinguishes
# from the kind of thing it also is.  A stated function of this strength is not
# a passing mention of another participant, so a category naming the broader
# kind does not overrule it.
_STATED_FUNCTION_OVER_KIND: dict[str, str] = {
    "FIXED_TARGET": (
        r"\b(?:requir\w*|need\w*|receiv\w*|await\w*|to be)\s+(?:the\s+)?"
        r"(?:fasten\w*|screw\w*|bolt\w*|secur\w*|attach\w*|join\w*)"
        r"|\bwhere (?:the )?fastening\b|\bfastening (?:site|location|point|spot)\b"
    ),
    "SEATING": r"\bwhere (?:a |the )?(?:person|people|viewer|occupant)s? (?:sit|sits|will sit)\b",
}


# The patient construction specifically: the role is the thing *being* fastened
# onto something else.  "Receive fastening at the joint hole" is the opposite
# valency -- the role receives the fastening -- and must not match, which is why
# the participle has to sit inside a passive frame rather than merely appear
# somewhere before a preposition.
# The opposite valency, stated actively: the role receives the fastening.  This
# has to be its own pattern rather than the broader receiving-assembly cue,
# because that cue counts "to be fastened" as receiving -- which is exactly the
# conflation the discriminator below exists to undo.
_STATES_IT_RECEIVES_FASTENING = re.compile(
    r"\b(?:receiv|accept|take|host)\w*\s+(?:the\s+|a\s+)?"
    r"(?:fasten\w*|screw\w*|bolt\w*|component|part|fixing)",
    re.I,
)


_FASTENED_ONTO_SOMETHING_ELSE = re.compile(
    r"\b(?:to be|must be|needs? to be|need to be|is|are|being)\s+"
    r"(?:\w+\s+){0,3}?"
    r"(?:fasten|attach|join|secur|connect|affix|mount|install)\w*\s+"
    r"(?:to|at|onto|into|against)\b",
    re.I,
)


_FAMILY_FOR_DOMAIN_ROLE: dict[str, dict[str, str]] = {
    "kitchen": {
        "coffee_source": "SOURCE", "water_source": "SOURCE",
        "coffee_container": "DESTINATION", "soup_container": "DESTINATION",
        "coffee_stirrer": "INSTRUMENT", "soup_eating_utensil": "INSTRUMENT",
        "countertop": "SUPPORT", "serving_area": "SUPPORT", "dining_table": "SUPPORT",
    },
    "living_room": {
        "PERSONAL_CUP_SAUCER_REGION": "SUPPORT",
        "SHARED_REMOTE_REGION": "SUPPORT",
        "CUP_SAUCER_SET": "PAYLOAD", "REMOTE": "PAYLOAD",
        "SEATING_POSITION": "SEATING", "SEATING_PAIR": "SEATING",
        "staging_tray": "SUPPORT",
    },
    "workshop": {
        "driver": "INSTRUMENT", "fastener": "COMPONENT",
        "repair_target": "FIXED_TARGET", "MAIN_WORKBENCH_ZONE": "SUPPORT",
        "workshop_frame_joint": "FIXED_TARGET",
    },
}

_FASTENER_KIND = re.compile(
    r"\b(screws?|bolts?|nuts?|washers?|clips?|rivets?|dowels?)\b" + _NOT_A_RECEIVING_FEATURE
)
_FASTENER_RECEIVING_FEATURE = re.compile(
    r"\b(?:screw|bolt|fastener|pilot|mounting|joint|threaded)\s+(?:holes?|recess(?:es)?|seats?|slots?|bores?)\b"
)
_RECEIVES_FASTENING = re.compile(
    r"\b(assembly receiv\w+|receiv\w+ (?:the )?(?:fastener|screw|bolt|component)|"
    r"to be (?:fastened|secured|attached)|fixtures?|workpieces?)\b"
)
_STORAGE_STRUCTURE = re.compile(
    r"\b(cabinets?|cupboards?|drawers?|bins?|boxes|box|canisters?|containers? unit|"
    r"lockers?|compartments?|chests?|storage (?:unit|structure|space|area|region|container)s?|"
    # Ordinary ways to say the same thing.  A cupboard described as "a closed
    # structure that may hold the component or tool" is a place to search, and
    # missing that wording read the cupboard as the tool it might hold.
    r"closed (?:structure|space|area|region|volume)s?)\b"
)
_STORES_THINGS = re.compile(
    # Morphology, not a word list: "may hold" is the same claim as "holds".
    r"\b(contain\w*|hold\w*|stor\w*|enclos\w*|keep\w*|hous\w*|hidden|conceal\w*)\b[^.]{0,40}"
    r"\b(fasteners?|screws?|bolts?|components?|parts?|tools?|items?|utensils?|supplies|equipment)\b"
    r"|\benclosed space\b|\bcan be opened\b|\bopenable\b|\bto search\b"
    # A place is described as somewhere to search either by what it holds or by
    # the searching itself, and the second wording puts the noun first, which an
    # ordered verb-then-noun pattern can never match.
    r"|\bsearchable\b|\bmight be hidden\b|\bmay be hidden\b|\brequire\w* searching\b"
    r"|\bworth searching\b|\bto be searched\b|\bcould be searched\b"
)
_EXPLICIT_IMPLEMENT_NOUN = re.compile(
    r"\b(tools?|implements?|instruments?|utensils?|drivers?|screwdrivers?|wrench(?:es)?|drills?|"
    r"pliers?|spanners?|applicators?|spoons?|forks?|knives|knife|stirrers?|whisks?|sticks?)\b"
)
_VESSEL_IDENTITY = re.compile(
    r"\b(containers?|vessels?|receptacles?|cups?|mugs?|glass(?:es)?|bowls?|dish(?:es)?|"
    r"pots?|pans?|kettles?|jugs?|tumblers?|jars?|cans?|bottles?)\b"
)
_SEATING_IDENTITY = re.compile(
    r"\b(chairs?|armchairs?|sofas?|couch(?:es)?|stools?|settees?|recliners?|seats?|seating|for sitting)\b"
)
_SUPPORT_IDENTITY = re.compile(
    r"\b(tables?|tabletops?|surfaces?|platforms?|stands?|shelves|shelf|counters?|countertops?|"
    r"desks?|workbench(?:es)?|trays?|flat ?top|planar)\b"
)
_PAYLOAD_IDENTITY = re.compile(
    r"\b(cups?|mugs?|saucers?|plates?|bowls?|dish(?:es)?|glass(?:es)?|drinkware|"
    r"remotes?|remote controls?|controllers?|electronic devices?|handsets?)\b"
)
_EXPLICIT_SOURCE = re.compile(
    r"\b(sources?|supply|supplies|supplied|provider|provides material|dispens\w+|"
    r"ingredient (?:source|supply|for|of)|raw material|liquid base)\b"
)
_EXPLICIT_RECEIVING = re.compile(
    r"\b(receiv(?:e|es|ed|ing|er)|is filled|are filled|filled with|into which|destination for|"
    r"holds? the (?:prepared|served)|takes? in)\b"
)
_EXPLICIT_TOOL = re.compile(
    r"\b(tools?|implements?|instruments?|drivers?|screwdrivers?|wrench(?:es)?|drills?|pliers?|applicators?)\b"
)
# A surface the task says something is put back, left or rested on is stating a
# support function outright, not merely carrying a surface noun in passing.
_STATES_IT_RECEIVES_A_RETURNED_ITEM = re.compile(
    r"\b(?:tool|equipment|item|implement|instrument|driver|screwdriver|part|object)s?\b"
    r"[^.;:]{0,40}?\b(?:is|are|to be|must be|should be|gets?|being)\s+"
    r"(?:\w+\s+){0,2}?(?:return|replac|rest|left|stow|put back|set down|plac|store)\w*"
    r"|\b(?:return|replac|rest|leav\w*|stow|stash|put back|set down|stor)\w*\s+"
    r"(?:the\s+|a\s+|any\s+)?(?:\w+\s+){0,2}?"
    r"(?:tool|equipment|item|implement|instrument|driver|screwdriver)s?\b",
    re.I,
)
_SUPPORT_NEAR_SEAT = re.compile(
    r"\b(tables?|surfaces?|supports?|platforms?)\b[^.]{0,40}\b(near|nearby|beside|adjacent|next to)\b"
    r"|\b(near|nearby|beside|adjacent|next to)\b[^.]{0,40}\b(seat|seats|seating|chair|chairs|sofa|couch)\b"
)
# A seat mentioned through a spatial or access preposition is being referred to,
# not claimed: "shared support accessible to both seats" describes the support.
_EXPLICIT_SUPPORT_PURPOSE = re.compile(
    r"\b(tables?|tabletops?|surfaces?|platforms?|stands?|shelves|shelf|counters?|countertops?|"
    r"desks?|workbench(?:es)?|trays?)\b"
    r"|\b(?:holds?|hold|supports?|support|place for|for placing|to place|rest\w*)\b"
    r"[^.]{0,30}\b(items?|refreshments?|things?|objects?|drinks?|cups?|payloads?|equipment|tools?)\b"
)
_REFERENTIAL_SEATING = re.compile(
    r"\b(accessible|reachable|access|near|nearby|beside|adjacent|next to|between|"
    r"relative to|close to|convenient to|within reach of|shared by|common to)\b"
    r"[^.]{0,25}\b(seats?|seating|chairs?|armchairs?|sofas?|couch(?:es)?)\b"
)
_PERSONAL_SPATIAL = re.compile(
    r"\b(personal|individual|own|per person|per[- ]?person|for (?:a|one|each|every) person|each person|"
    r"side table|end table|nearby|near ?by|beside|adjacent|next to|"
    r"near (?:the |a )?(?:seat|chair|sofa|couch|person|viewer|occupant))\b"
)
_SHARED_ACCESS = re.compile(
    r"\b(shared|share|common|communal|both|either|between|jointly|mutual|central|centre|center|"
    r"accessible (?:to|from|by) (?:both|all|each|every)|reachable (?:by|from) (?:both|all|each)|"
    r"deliberately common)\b"
)


# Families the runtime recognises but deliberately realises with no functional
# role: a television is scene furniture the task never handles, and a drawer or
# cabinet is a place to search rather than a participant to bind.  Recognising
# them keeps such a role out of the functional graph without pretending the
# runtime failed to understand it.
_CONTEXT_ONLY_FAMILIES: dict[str, frozenset[str]] = {
    "kitchen": frozenset({"STORAGE_CONTEXT"}),
    "living_room": frozenset({"DISPLAY_CONTEXT", "STORAGE_CONTEXT"}),
    "workshop": frozenset({"STORAGE_CONTEXT"}),
}

# Families that say what an object *is*, as opposed to the causal position it
# occupies.  When the model's own candidate categories name one of these, they
# are the better witness than a job description that merely mentions another
# participant -- "the area where the tool must be left" is a surface, not a tool.
_KIND_FAMILIES = frozenset({
    "INSTRUMENT", "SUPPORT", "DESTINATION", "SEATING", "PAYLOAD",
    "COMPONENT", "FIXED_TARGET",
})


def context_only_families(domain: str) -> frozenset[str]:
    return _CONTEXT_ONLY_FAMILIES.get(domain, frozenset())


def _declared_families(domain: str) -> set[str]:
    """Families the domain either realizes with a role or recognizes as context.

    Detecting a family the domain neither realizes nor recognizes would exclude
    every candidate and turn an ordinary role into an unrepresented semantic, so
    detection is restricted to the families that mean something here.
    """
    return {
        family for family in _FAMILY_FOR_DOMAIN_ROLE.get(domain, {}).values()
        if family != "OTHER"
    } | set(context_only_families(domain))


ACCEPTANCE_VOCABULARY = "RUNTIME_ACCEPTANCE_CATEGORY"


def acceptance_vocabulary_match(domain: str, role: dict[str, Any]) -> tuple[str, ...]:
    """Runtime roles whose own acceptance vocabulary the model's categories name.

    Grounding decides whether an observed label suits a role by comparing it
    with that role's declared acceptance categories.  When the model's
    candidate_categories name one of those very categories, the two layers are
    talking about the same thing, and compile-time typing ought to reach the
    role grounding would accept.  Leaving them unconnected meant a coffee
    source offered as a "jar" -- a category coffee_source itself declares --
    was read as a cupboard to search, and the coffee it supplies was lost.

    Only a unique match counts.  "spoon" is declared by both the coffee stirrer
    and the soup eating utensil, so it says which family the role belongs to and
    not which role it is; the rest of the wording has to settle that.
    """
    from .role_semantic_ontology import (
        get_all_system_role_semantic_categories,
        _normalize_label_text,
    )
    offered = {
        _normalize_label_text(category)
        for category in (role.get("candidate_categories") or ())
        if category
    }
    if not offered:
        return ()
    declared = get_all_system_role_semantic_categories(domain)
    return tuple(sorted(
        name for name, categories in declared.items()
        if offered & {_normalize_label_text(category) for category in categories}
    ))


def detect_role_families(domain: str, role: dict[str, Any]) -> tuple[set[str], dict[str, set[str]]]:
    """Nominate semantic families from the whole role, keeping scope provenance."""
    scopes = role_text_scopes(role)
    declared = _declared_families(domain)
    families: set[str] = set()
    provenance: dict[str, set[str]] = {}
    for family, pattern in _FAMILY_CUES.items():
        if family not in declared:
            continue
        matched = {name for name, text in scopes.items() if text and re.search(pattern, text, re.I)}
        if matched:
            families.add(family)
            provenance[family] = matched
    # Only where the role's own account of its job said nothing.  What the model
    # says a thing is *for* outranks a category it listed: "contain coffee
    # powder", offered among categories including "cup", is a supply and not a
    # cup.  The vocabulary decides when the purpose is silent, which is the gap
    # it was added for -- "Liquid for coffee", offered as a kettle.
    if not any("PURPOSE_TEXT" in scope_names for scope_names in provenance.values()):
        accepted = acceptance_vocabulary_match(domain, role)
        if len(accepted) == 1:
            family = canonical_role_family(domain, accepted[0])
            if family in declared:
                families.add(family)
                provenance.setdefault(family, set()).add(ACCEPTANCE_VOCABULARY)
    return families, provenance


# A role whose canonical type was settled by the shape of the contract -- the
# relations and operations it participates in, the entity kind the model
# declared for it, the fact that another participant is already pinned to the
# alternative -- rather than by matching its function wording alone.  A
# conclusion drawn from the whole graph outranks the isolated function-alias
# mapper, which cannot see that a role is an operation participant and therefore
# sometimes calls a manipulable target a fixed piece of planner context.
STRUCTURALLY_EVIDENCED_ROLE_TYPE_STATUSES = frozenset({
    "STRUCTURAL_OVERRIDE_OF_WEAK_FUNCTION_ALIAS",
    "GLOBAL_GRAPH_CONSISTENCY_OVERRIDE",
    "RELATION_ASSISTED",
    "OPERATION_ASSISTED",
    "JOINT_SEMANTIC_RESOLUTION",
})


def canonical_role_family(domain: str, role: str) -> str:
    return _FAMILY_FOR_DOMAIN_ROLE.get(domain, {}).get(role, "OTHER")


def family_entity_kinds(domain: str, family: str) -> frozenset[str]:
    """The entity kinds the runtime's own roles of this family are.

    Derived from the family table and the runtime entity kinds rather than
    listed again, so a family cannot drift out of step with the roles in it.
    """
    return frozenset(
        kind
        for role, fam in _FAMILY_FOR_DOMAIN_ROLE.get(domain, {}).items()
        if fam == family
        and (kind := get_runtime_role_entity_kind(domain, role)) is not None
    )


def families_the_declared_kind_rules_out(
    domain: str, declared_kind: str, families: Iterable[str]
) -> set[str]:
    """Co-nominated families the model's own entity kind contradicts.

    The contract asks, for every participant, whether the thing is carried or is
    an area, and it answers separately from the function text.  A family the
    runtime realizes only with roles of the other class contradicts that answer.

    Consulted for one purpose only: deciding whether a *category list* naming
    only families of the other class should outrank the job description.  The
    declaration is not trusted to strike a family the job description or the
    role's own name supports, because the model mislabels often enough that
    doing so does real damage -- a workshop contract declared its workbench
    surface an OBJECT, and striking SUPPORT left an incidental implement cue
    standing, so the bench became the screwdriver and collided with the real
    tool.  The surviving rule can only set aside a claim made by the categories
    alone.

    Returned only when striking those leaves a family that positively *is* of
    the declared class, which is what keeps it from deciding questions the
    declaration cannot settle:

      * a side table the model called an OBJECT nominates SUPPORT and SEATING.
        SUPPORT is an area, so it is contradicted -- but SEATING is a fixed
        reference, not something carried, so nothing the declaration names
        survives and the empty set is returned.  Reading "not an area, therefore
        the seat" turned a table into a chair.
      * the assembly a screw is driven into is declared an OBJECT and is a fixed
        reference to the runtime.  Nothing about it is an area, so again nothing
        is struck, and it stays the repair target rather than becoming the screw.

    What it does settle is the case it was added for: "holds refreshments for a
    person", declared an OBJECT, nominating PAYLOAD and SUPPORT.  SUPPORT is an
    area and PAYLOAD is carried, so the declaration picks the payload -- and the
    refreshments stop being read as the table they go on.
    """
    kind = str(declared_kind or "").strip().upper()
    if kind in CARRIED_ENTITY_KINDS:
        opposite = AREA_ENTITY_KINDS
    elif kind in AREA_ENTITY_KINDS:
        opposite = CARRIED_ENTITY_KINDS
    else:
        return set()
    families = set(families)
    ruled_out = {
        family for family in families
        if (kinds := family_entity_kinds(domain, family)) and kinds <= opposite
    }
    if not ruled_out:
        return set()
    survives_as_declared = any(
        kind in family_entity_kinds(domain, family)
        for family in families - ruled_out
    )
    return ruled_out if survives_as_declared else set()


def _in_any_scope(pattern, scopes: dict[str, str]) -> bool:
    """Search each text scope separately.

    Never join the scopes before matching: concatenation invents adjacency that
    the model never wrote.  A seat whose description mentions a table beside it
    read as "beside ... seat" once the role id was appended, and typed as the
    table.
    """
    expression = pattern if hasattr(pattern, "search") else re.compile(pattern, re.I)
    return any(text and expression.search(text) for text in scopes.values())


def _resolve_family_precedence(
    domain: str,
    families: set[str],
    scopes: dict[str, str],
    provenance: dict[str, set[str]],
    declared_entity_kind: str = "",
) -> tuple[set[str], list[str]]:
    """Reduce co-nominated families using what each scope actually claimed.

    Several families can legitimately co-occur -- a source stored in a vessel,
    a side table described by its distance to a seat.  Each rule below says
    which claim wins and why, and every rule is decided from the model's own
    wording rather than from any expectation about the scene.
    """
    authoritative = set(families)
    applied: list[str] = []
    identity, purpose = scopes["IDENTITY_TEXT"], scopes["PURPOSE_TEXT"]

    def drop(family: str, rule: str) -> None:
        if family in authoritative:
            authoritative.discard(family)
            applied.append(rule)

    # The model naming a role's acceptance vocabulary is the two semantic layers
    # agreeing about what the thing is, and that outranks a family picked up
    # from a word in passing.
    by_vocabulary = {
        family for family, scope_names in provenance.items()
        if ACCEPTANCE_VOCABULARY in scope_names
    }
    if len(by_vocabulary) == 1:
        for family in sorted(authoritative - by_vocabulary):
            drop(family, "RUNTIME_ACCEPTANCE_VOCABULARY_OVER_INCIDENTAL_CUE")
        authoritative |= by_vocabulary



    # A structure the model describes as holding or enclosing the things the
    # task needs is somewhere to search, not one of those things.  But being
    # closable is not being a cupboard: a jar of coffee is openable and is
    # still the coffee supply, and reading it as somewhere to search lost the
    # coffee from the task.  So the storage sense has to come from what the
    # model says the thing is *for*, or from what it calls it -- not from a
    # property listed in passing.
    named_storage = {
        key: value for key, value in scopes.items() if key in {"PURPOSE_TEXT", "NAME_TEXT"}
    }
    if (
        _in_any_scope(_STORAGE_STRUCTURE, scopes)
        and _in_any_scope(_STORES_THINGS, scopes)
        and (
            _in_any_scope(_STORAGE_STRUCTURE, named_storage)
            or _in_any_scope(_STORES_THINGS, named_storage)
        )
    ):
        for family in ("COMPONENT", "INSTRUMENT", "SOURCE", "DESTINATION", "PAYLOAD"):
            drop(family, "STORAGE_STRUCTURE_IS_NOT_ITS_CONTENTS")
        authoritative.add("STORAGE_CONTEXT")
    # A role whose stated purpose is an implement's action is the implement, not
    # the material it works on and not the vessel it goes into.  "Stir the
    # coffee mixture", offered as a spoon that "fits in a vessel", was read as
    # a receptacle because a passing property mentioned one.
    if (
        "INSTRUMENT" in authoritative
        and _IMPLEMENT_ACTION.search(purpose)
        and _in_any_scope(_IMPLEMENT_KIND, scopes)
    ):
        for family in ("SOURCE", "DESTINATION"):
            drop(family, "STATED_IMPLEMENT_ACTION_OVER_WHAT_IT_ACTS_ON")
    # And the converse.  The action words read just as well in the passive --
    # "liquid to be mixed with coffee", "holds the coffee mixture" -- where they
    # say what happens *to* the role.  Something the model never calls an
    # implement is not one just because mixing was mentioned near it, so long as
    # the model said what it is some other way.
    elif (
        "INSTRUMENT" in authoritative
        and len(authoritative) > 1
        and not _in_any_scope(_IMPLEMENT_KIND, scopes)
    ):
        drop("INSTRUMENT", "AN_ACTION_WORD_ALONE_IS_NOT_AN_IMPLEMENT")
    # A material named as a material is that material, whatever the job
    # description says about where it ends up.  "Hot food item served in the
    # bowl", offered as "soup broth stew", was read as the bowl.
    if (
        "SOURCE" in authoritative
        and {"IDENTITY_TEXT", "NAME_TEXT"} & provenance.get("SOURCE", set())
        and provenance.get("DESTINATION", set()) == {"PURPOSE_TEXT"}
    ):
        drop("DESTINATION", "MATERIAL_IDENTITY_OVER_WHERE_IT_IS_PUT")
    # A place the model describes as storage is where things are kept, not where
    # the task is asked to put them.
    if "STORAGE_CONTEXT" in authoritative and re.search(
        _FAMILY_CUES["STORAGE_CONTEXT"], purpose, re.I
    ):
        drop("SUPPORT", "STATED_STORAGE_PURPOSE_IS_NOT_A_TASK_SUPPORT")
    # What the model says the thing *is* outranks a job description that merely
    # mentions another participant, so long as the categories name a kind at all.
    identity_kinds = {
        family for family in _KIND_FAMILIES
        if family in _declared_families(domain)
        and identity and re.search(_FAMILY_CUES[family], identity, re.I)
    }
    # ... except where the job description states the defining function of a
    # role the runtime keeps apart from that kind.  "A specific area on the
    # workbench requiring fastening" is a workbench surface *and* the fastening
    # site, and only the second is a role the runtime can fasten into; letting
    # the kind win read the site as the bench it sits on.
    stated_function_families = {
        family for family in _STATED_FUNCTION_OVER_KIND
        if family in _declared_families(domain)
        and purpose and re.search(_STATED_FUNCTION_OVER_KIND[family], purpose, re.I)
    }
    # ... and except where every kind the categories claim is of the other class
    # from the one the model declared this participant to be.  The categories
    # are then describing where the thing belongs rather than what it is, which
    # is the ordinary way to name a thing: "holds refreshments for a person",
    # declared an OBJECT, offered as "table, surface, tray".  Letting the
    # categories win there read the refreshments as the side table they go on.
    #
    # Only when the declaration picks out one of the families already nominated,
    # so a category list is set aside for a positive reading and never merely
    # because it disagrees.
    if (
        declared_entity_kind
        and identity_kinds
        and families_the_declared_kind_rules_out(
            domain, declared_entity_kind, identity_kinds | authoritative
        ) >= identity_kinds
    ):
        identity_kinds = set()
        applied.append("DECLARED_ENTITY_KIND_OVER_A_CATEGORY_LIST_OF_THE_OTHER_CLASS")
    if identity_kinds:
        for family in sorted(_KIND_FAMILIES - identity_kinds - stated_function_families):
            drop(family, "CANDIDATE_CATEGORY_KIND_OVER_JOB_DESCRIPTION")
    # Support against seating, in order of how directly each claim was made.
    if {"SUPPORT", "SEATING"} <= authoritative:
        seating_purpose = bool(
            re.search(_FAMILY_CUES["SEATING"], purpose, re.I)
            and not _REFERENTIAL_SEATING.search(purpose)
        )
        support_purpose = bool(_EXPLICIT_SUPPORT_PURPOSE.search(purpose))
        if seating_purpose:
            # The model said this is where someone sits.
            drop("SUPPORT", "STATED_SEATING_PURPOSE_OVER_INCIDENTAL_SURFACE")
        elif support_purpose:
            # The model said this is a surface for things.  A stool among its
            # candidate categories is one way to realize that, not a claim that
            # the role is itself a seat.
            drop("SEATING", "STATED_SUPPORT_PURPOSE_OVER_AMBIGUOUS_SEATING_CATEGORY")
        elif _in_any_scope(_SUPPORT_NEAR_SEAT, scopes):
            drop("SEATING", "SUPPORT_DESCRIBED_RELATIVE_TO_SEATING")
        elif _SEATING_IDENTITY.search(identity):
            # Seating furniture, and nothing said about holding things.
            drop("SUPPORT", "SEATING_IDENTITY_OVER_INCIDENTAL_SURFACE")
        elif _in_any_scope(_REFERENTIAL_SEATING, scopes):
            drop("SEATING", "SEATING_REFERRED_TO_RATHER_THAN_CLAIMED")
    # An explicit implement is not the component it drives nor the thing it works on.
    if "INSTRUMENT" in authoritative and _in_any_scope(_EXPLICIT_TOOL, scopes):
        drop("COMPONENT", "EXPLICIT_IMPLEMENT_OVER_COMPONENT")
        drop("FIXED_TARGET", "EXPLICIT_IMPLEMENT_OVER_FIXED_TARGET")
    # Instrument evidence that is only an activity word -- "for mixing" -- does
    # not outrank a vessel the model actually named: a pot used for mixing is
    # still the vessel, not the implement.
    if (
        "INSTRUMENT" in authoritative
        and not _in_any_scope(_EXPLICIT_IMPLEMENT_NOUN, scopes)
        and (_in_any_scope(_VESSEL_IDENTITY, scopes) or _in_any_scope(_SUPPORT_IDENTITY, scopes))
        and authoritative & {"DESTINATION", "SUPPORT"}
    ):
        drop("INSTRUMENT", "NAMED_VESSEL_OR_SURFACE_OVER_ACTIVITY_WORD")
    if (
        "COMPONENT" in authoritative
        and (_in_any_scope(_FASTENER_KIND, scopes) or _in_any_scope(
            r"\b(joining (?:element|component|part|piece)|connecting (?:element|component|part)|"
            r"connectors?|installed component|component to be (?:installed|secured|attached|fastened))\b",
            scopes))
        and not _in_any_scope(r"\b(assembly receiv\w+|workpieces?|fixtures?)\b", scopes)
    ):
        drop("FIXED_TARGET", "EXPLICIT_JOINING_COMPONENT_OVER_FIXED_TARGET")
    # A named fastener kind is the fastener, however often the text mentions the
    # workpiece it goes into.  Without this, "screw compatible with the workpiece
    # hole" typed as the workpiece.
    fastener_identity = _in_any_scope(_FASTENER_KIND, scopes)
    if "FIXED_TARGET" in authoritative and _in_any_scope(_FASTENER_RECEIVING_FEATURE, scopes):
        drop("COMPONENT", "FASTENER_RECEIVING_FEATURE_OVER_FASTENER_KIND")
    elif "COMPONENT" in authoritative and fastener_identity:
        drop("FIXED_TARGET", "NAMED_FASTENER_KIND_OVER_RECEIVING_TARGET")
    elif "FIXED_TARGET" in authoritative and _in_any_scope(_RECEIVES_FASTENING, scopes):
        drop("COMPONENT", "RECEIVING_ASSEMBLY_OVER_COMPONENT")
    # The place where the work must happen is an interaction point, not the
    # bench it happens to sit on.  But a role that states both functions at once
    # -- "surface where the fastening occurs and the tool is returned" -- has
    # expressed the support function explicitly, and erasing it here left the
    # stated tool return with no destination and no way to recover one.  Both
    # readings survive as candidates so the operations the model wrote can
    # decide which one this role fills.
    if "FIXED_TARGET" in authoritative:
        if "SUPPORT" in authoritative and _in_any_scope(
            _STATES_IT_RECEIVES_A_RETURNED_ITEM, scopes
        ):
            applied.append("STATED_RETURN_SURFACE_KEEPS_SUPPORT_READING")
        else:
            drop("SUPPORT", "FASTENING_SITE_OVER_SUPPORT_SURFACE")
        drop("DESTINATION", "FASTENING_SITE_OVER_CONTAINER_SHAPE")
    # A payload named as a device or as drinkware is not the screen it controls.
    if "PAYLOAD" in authoritative:
        drop("DISPLAY_CONTEXT", "PAYLOAD_OVER_DISPLAY_CONTEXT")
    # Payload against support: what the model says it *is* decides.
    if {"PAYLOAD", "SUPPORT"} <= authoritative:
        payload_identity = bool(_PAYLOAD_IDENTITY.search(identity))
        support_identity = bool(_SUPPORT_IDENTITY.search(identity))
        if payload_identity and not support_identity:
            drop("SUPPORT", "PAYLOAD_IDENTITY_OVER_SUPPORT_WORDING")
        elif support_identity and not payload_identity:
            drop("PAYLOAD", "SUPPORT_IDENTITY_OVER_PAYLOAD_WORDING")
    # Causal direction: an explicit statement of what the role receives settles
    # it against a passing mention of the material involved.
    if (
        {"SOURCE", "DESTINATION"} <= authoritative
        and _in_any_scope(_EXPLICIT_RECEIVING, scopes)
        and not _in_any_scope(_EXPLICIT_SOURCE, scopes)
    ):
        drop("SOURCE", "EXPLICIT_RECEIVING_OVER_MATERIAL_MENTION")
    # Supplying material is a causal position; the vessel it lives in is not.
    if "SOURCE" in authoritative and _in_any_scope(_EXPLICIT_SOURCE, scopes):
        drop("DESTINATION", "EXPLICIT_SOURCE_OVER_CONTAINER_SHAPE")
    if "SOURCE" in authoritative:
        drop("SUPPORT", "CAUSAL_SOURCE_OVER_SUPPORT")
    if "DESTINATION" in authoritative:
        drop("SUPPORT", "CAUSAL_DESTINATION_OVER_SUPPORT")
    # A functional claim outranks an incidental mention of scene furniture or of
    # the cupboard something lives in.
    context = context_only_families(domain)
    if authoritative & context and authoritative - context:
        for family in sorted(authoritative & context):
            drop(family, "FUNCTIONAL_FAMILY_OVER_RECOGNISED_CONTEXT")
    # Last, because this is a discriminator rather than a nomination and it has
    # to survive the rules above.  Something the model says is to be fastened
    # *to* something else is the thing being fastened, not the site receiving it
    # and not the surface it names as its destination.  With nothing to separate
    # them, "the part that needs to be fastened to the site" was read as the
    # site, and then collided with the site the model had also declared.
    if (
        _FASTENED_ONTO_SOMETHING_ELSE.search(purpose)
        and not _STATES_IT_RECEIVES_FASTENING.search(purpose)
        and ("FIXED_TARGET" in families or "SUPPORT" in families)
        and "COMPONENT" in _declared_families(domain)
    ):
        for family in ("FIXED_TARGET", "SUPPORT"):
            drop(family, "FASTENED_ONTO_SOMETHING_ELSE_IS_THE_COMPONENT")
        if not authoritative:
            authoritative.add("COMPONENT")
            applied.append("FASTENED_ONTO_SOMETHING_ELSE_IS_THE_COMPONENT")
    return authoritative, list(dict.fromkeys(applied))


def _preferred_roles(
    domain: str,
    role: dict[str, Any],
    families: set[str],
    scopes: dict[str, str],
) -> set[str]:
    """Pick the single role each family points at, when the wording says which."""
    identity, purpose = scopes["IDENTITY_TEXT"], scopes["PURPOSE_TEXT"]
    preferred: set[str] = set()

    if domain == "kitchen":
        if "SOURCE" in families:
            water = bool(_in_any_scope(r"\bwater\b", scopes))
            coffee = bool(_in_any_scope(r"\bcoffee\b", scopes))
            if _in_any_scope(r"\b(?:provide|supply|source|holding|base)\s+(?:of\s+|the\s+)?water\b|\bwater\s+source\b", scopes):
                preferred.add("water_source")
            elif _in_any_scope(r"\b(?:provide|supply|source|holding)\s+(?:the\s+)?coffee\b|\bcoffee\s+(?:ingredient\s+)?source\b", scopes):
                preferred.add("coffee_source")
            elif water and not coffee:
                preferred.add("water_source")
            elif coffee and not water:
                preferred.add("coffee_source")
            elif water and coffee:
                # Both materials named.  The identity words say which one this
                # participant supplies; a "liquid base for coffee" whose
                # categories are tap water is the water, not the grounds.
                if re.search(r"\bwater\b", identity) and not re.search(r"\bcoffee\b", identity):
                    preferred.add("water_source")
                elif re.search(r"\bcoffee\b", identity) and not re.search(r"\bwater\b", identity):
                    preferred.add("coffee_source")
                elif re.search(r"\b(liquid|fluid|water)\b", purpose) and not re.search(
                        r"\b(dry|solid|powder|granule|grounds|beans)\b", purpose):
                    preferred.add("water_source")
                elif _in_any_scope(r"\b(dry|solid|powder|granule|grounds|beans)\b", scopes):
                    preferred.add("coffee_source")
        if "DESTINATION" in families:
            if _in_any_scope(r"\b(coffee|beverage|drink)\b", scopes): preferred.add("coffee_container")
            if _in_any_scope(r"\bsoup\b", scopes): preferred.add("soup_container")
            if not preferred & {"coffee_container", "soup_container"}:
                if _PAYLOAD_IDENTITY.search(identity) and re.search(r"\b(cups?|mugs?|glass(?:es)?|tumblers?)\b", identity):
                    preferred.add("coffee_container")
                elif re.search(r"\b(bowls?|dish(?:es)?)\b", identity):
                    preferred.add("soup_container")
        if "INSTRUMENT" in families:
            if _in_any_scope(r"\b(stir|mix|agitat|blend)\w*\b", scopes): preferred.add("coffee_stirrer")
            if _in_any_scope(r"\b(eat|consum|soup)\w*\b", scopes): preferred.add("soup_eating_utensil")
        if "SUPPORT" in families:
            if _in_any_scope(r"\b(serv(?:e|es|ing|ice)|delivery|handoff)\b", scopes):
                preferred.add("serving_area")
            elif _in_any_scope(r"\b(dining|meal)\b", scopes):
                preferred.add("dining_table")
            else:
                preferred.add("countertop")
    elif domain == "living_room":
        if "SUPPORT" in families:
            personal = bool(_in_any_scope(_PERSONAL_SPATIAL, scopes))
            shared = bool(_in_any_scope(_SHARED_ACCESS, scopes))
            if shared and not personal:
                preferred.add("SHARED_REMOTE_REGION")
            elif personal and not shared:
                preferred.add("PERSONAL_CUP_SAUCER_REGION")
            elif not personal and not shared and _in_any_scope(
                    r"\b(coffee table|centre table|center table)\b", scopes):
                # A coffee table is the room's common surface unless the model
                # said this one belongs to a person.
                preferred.add("SHARED_REMOTE_REGION")
        if "SEATING" in families:
            # A seating reference the model wrote in the plural, or called shared
            # or paired, is the pair of seats; anything else is one of them.
            preferred.add(
                "SEATING_PAIR"
                if _in_any_scope(
                    r"\b(pairs?|both|two seats|all seats|together|seats|"
                    r"shared seating|seating pair|seating context)\b", scopes)
                else "SEATING_POSITION"
            )
        if "PAYLOAD" in families:
            preferred.add(
                "REMOTE"
                if _in_any_scope(r"\b(remotes?|control\w*|entertainment|television|tv|handsets?)\b", scopes)
                else "CUP_SAUCER_SET"
            )
    elif domain == "workshop":
        if "INSTRUMENT" in families: preferred.add("driver")
        if "COMPONENT" in families: preferred.add("fastener")
        if "FIXED_TARGET" in families: preferred.add("repair_target")
        if "SUPPORT" in families: preferred.add("MAIN_WORKBENCH_ZONE")
    return preferred


def function_semantic_evidence(
    domain: str,
    role: dict[str, Any],
    runtime_roles: set[str],
    weak_preference: str | None = None,
) -> FunctionSemanticEvidence:
    """Extract positive preferences and explicit family exclusions without scores."""
    scopes = role_text_scopes(role)
    families, provenance = detect_role_families(domain, role)
    authoritative, precedence_rules = _resolve_family_precedence(
        domain, families, scopes, provenance,
        declared_entity_kind=str(role.get("entity_kind", "")),
    )
    preferred = _preferred_roles(domain, role, authoritative, scopes)
    accepted = acceptance_vocabulary_match(domain, role)
    if len(accepted) == 1 and accepted[0] in runtime_roles and ACCEPTANCE_VOCABULARY in (
        provenance.get(canonical_role_family(domain, accepted[0]), set())
    ) and canonical_role_family(domain, accepted[0]) in authoritative:
        # The vocabulary is what nominated this family, and grounding would
        # accept an object of this category for this role, so this is the role
        # and not merely its family.
        preferred = {accepted[0]}

    if not preferred and weak_preference in runtime_roles:
        preferred.add(weak_preference)

    excluded = {
        candidate for candidate in runtime_roles
        if authoritative and canonical_role_family(domain, candidate) not in authoritative
    }
    runtime_context_only = bool(authoritative) and authoritative <= context_only_families(domain)
    if domain == "kitchen" and "SOURCE" in authoritative and _in_any_scope(
        r"\bsoup\b", scopes
    ) and not _in_any_scope(r"\b(coffee|water)\b", scopes):
        # The current runtime has coffee and water sources, but no soup-material
        # source role. Preserve that as representability failure; never relabel it.
        excluded |= {candidate for candidate in runtime_roles
                     if canonical_role_family(domain, candidate) == "SOURCE"}
    elif domain == "kitchen" and "SOURCE" in authoritative:
        if preferred == {"water_source"}:
            excluded.add("coffee_source")
        elif preferred == {"coffee_source"}:
            excluded.add("water_source")
    contributing = sorted({
        scope for family in authoritative for scope in provenance.get(family, ())
    })
    # When the categories the model offered contradict the job description it
    # wrote, and the categories win, say so: the resolution rests on the rest of
    # the role rather than on the sentence that named its function.
    overridden_purpose_family = any(
        family not in authoritative and "PURPOSE_TEXT" in provenance.get(family, ())
        for family in families
    )
    source = "+".join(contributing) if contributing else "NO_FAMILY_EVIDENCE"
    if overridden_purpose_family and "CANDIDATE_CATEGORY_KIND_OVER_JOB_DESCRIPTION" in precedence_rules:
        source = "GLOBAL_GRAPH_CONSISTENCY_OVERRIDE"
    return FunctionSemanticEvidence(
        preferred_roles=tuple(sorted(preferred & runtime_roles)),
        excluded_roles=tuple(sorted(excluded)),
        explicit_families=tuple(sorted(authoritative)),
        evidence_strength="EXPLICIT_FAMILY" if authoritative else ("WEAK_ALIAS" if preferred else "UNKNOWN"),
        evidence_source=source,
        family_precedence_rules=tuple(precedence_rules),
        runtime_context_only=runtime_context_only,
    )


def causal_predicate_endpoint_pairs(domain: str, predicate: str) -> set[tuple[str, str]]:
    """Runtime role pairs a task-causal predicate can relate, for meaning selection."""
    return _causal_pairs(domain, predicate)


def _causal_pairs(domain: str, predicate: str) -> set[tuple[str, str]]:
    if domain == "kitchen":
        if predicate == "PROVIDES_MATERIAL_TO":
            return {(s, t) for s in ("coffee_source", "water_source") for t in ("coffee_container", "soup_container")}
        if predicate == "ACTS_ON":
            return {("coffee_stirrer", "coffee_container"), ("soup_eating_utensil", "soup_container")}
        if predicate == "PAIRED_WITH": return {("soup_eating_utensil", "soup_container")}
    if domain == "workshop":
        if predicate == "ACTS_ON": return {("driver", "fastener")}
        if predicate in {"INSTALLED_AT", "CONNECTED_TO"}: return {("fastener", "repair_target")}
    if domain == "living_room" and predicate == "SITUATED_BETWEEN":
        return {(support, seating) for support in ("PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION")
                for seating in ("SEATING_POSITION", "SEATING_PAIR")}
    return set()


# Which kinds of participant a stated end state relates.  A task effect is not
# a precondition the robot checks, but it is still a claim about what its two
# endpoints *are*: something contains a material, and something is placed on a
# support.  Declared at the level of role families rather than role names, so a
# new role of a known family is covered without another table.
_TASK_EFFECT_FAMILY_PAIRS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "CONTAINS": (frozenset({"DESTINATION"}), frozenset({"SOURCE", "PAYLOAD", "COMPONENT"})),
    "PLACED_ON": (frozenset({"PAYLOAD", "DESTINATION", "INSTRUMENT", "COMPONENT"}),
                  frozenset({"SUPPORT", "FIXED_TARGET"})),
}


def _task_effect_pairs(domain: str, predicate: str) -> set[tuple[str, str]]:
    families = _TASK_EFFECT_FAMILY_PAIRS.get(predicate)
    if families is None:
        return set()
    subject_families, object_families = families
    declared = _FAMILY_FOR_DOMAIN_ROLE.get(domain, {})
    subjects = [role for role, family in declared.items() if family in subject_families]
    objects = [role for role, family in declared.items() if family in object_families]
    return {(s, o) for s in subjects for o in objects if s != o}


def relation_canonical_role_pairs(
    domain: str, predicate: str, category: str
) -> set[tuple[str, str]]:
    """Return registered endpoint-type pairs for a relation meaning."""
    if category == "PHYSICAL_VERIFIER":
        signature = get_predicate_signature(domain, predicate)
        if not signature or signature.arity != 2 or not signature.active_in_functional_graph:
            return set()
        return {(s, o) for s in signature.allowed_subject_roles for o in signature.allowed_object_roles}
    if category == "TASK_CAUSAL_SEMANTICS":
        return _causal_pairs(domain, predicate)
    if category == "TASK_EFFECT_SEMANTICS":
        return _task_effect_pairs(domain, predicate)
    return set()


# Provenance markers the runtime writes on roles it authored itself.
RUNTIME_AUTHORED_ROLE_PROVENANCE = frozenset({
    "OPERATION_INDUCED_ABSTRACT_ROLE_REQUIREMENT",
    "EXPLICIT_CONTEXT_SET_CANONICALIZATION",
})


def build_role_type_hypotheses(
    domain: str,
    document: dict[str, Any],
    *,
    weak_mapper: Callable[[str, dict[str, Any], dict[str, Any]], tuple[str | None, str]] | None = None,
) -> dict[str, RoleTypeHypothesis]:
    """Build finite role domains from function evidence and hard graph structure."""
    runtime_roles = (
        set(get_domain_selectable_roles(domain))
        | set(get_domain_system_fixed_anchors(domain))
        | set(get_domain_planner_context_constants(domain))
    )
    roles = {str(role["id"]): role for role in document.get("functional_roles", ())}
    domains: dict[str, set[str]] = {}
    evidences: dict[str, list[dict[str, Any]]] = {rid: [] for rid in roles}
    functions: dict[str, FunctionSemanticEvidence] = {}
    weak_mapped: dict[str, str | None] = {}
    constrained: dict[str, set[str]] = {rid: set() for rid in roles}
    structural_conflicts: set[str] = set()
    declared_kind_narrowed: set[str] = set()

    # Roles the runtime itself introduced -- an operation-induced abstract slot,
    # an enumerated context set -- already name the canonical role they stand
    # for.  Re-deriving that from the placeholder wording would let a slot the
    # runtime created for the pair of seats come back as one of them.
    runtime_authored = {
        rid: str(role["canonical_role"])
        for rid, role in roles.items()
        if role.get("provenance") in RUNTIME_AUTHORED_ROLE_PROVENANCE
        and str(role.get("canonical_role", "")) in runtime_roles
    }
    for rid, role in roles.items():
        mapped, rule = (None, "NONE")
        if rid in runtime_authored:
            domains[rid] = {runtime_authored[rid]}
            functions[rid] = FunctionSemanticEvidence(
                preferred_roles=(runtime_authored[rid],),
                explicit_families=(canonical_role_family(domain, runtime_authored[rid]),),
                evidence_strength="RUNTIME_AUTHORED_ROLE",
                evidence_source=str(role.get("provenance")),
            )
            weak_mapped[rid] = runtime_authored[rid]
            evidences[rid].append({
                "source": str(role.get("provenance")),
                "canonical_role": runtime_authored[rid],
                "induced_by_operation": role.get("induced_by_operation"),
                "induced_capability": role.get("induced_capability"),
                "induced_slot": role.get("induced_slot"),
            })
            continue
        if weak_mapper is not None:
            try: mapped, rule = weak_mapper(domain, role, document)
            except Exception as exc: rule = f"MAPPER_ERROR:{exc}"
        evidence = function_semantic_evidence(domain, role, runtime_roles, mapped)
        weak_mapped[rid] = mapped if mapped in runtime_roles else None
        functions[rid] = evidence
        domains[rid] = runtime_roles - set(evidence.excluded_roles)
        evidences[rid].append({
            "source": evidence.evidence_source, "preferred_roles": list(evidence.preferred_roles),
            "excluded_roles": list(evidence.excluded_roles),
            "explicit_families": list(evidence.explicit_families),
            "strength": evidence.evidence_strength, "legacy_mapper_rule": rule,
            "family_precedence_rules": list(evidence.family_precedence_rules),
        })
        # Family precedence may have used the declared entity kind to choose
        # between co-nominated families; see the rule in _resolve_family_precedence.
        # Record that here so the status below reflects a reading the contract's
        # own shape settled rather than a function-alias guess.
        if "DECLARED_ENTITY_KIND_OVER_A_CATEGORY_LIST_OF_THE_OTHER_CLASS" in (
            evidence.family_precedence_rules
        ):
            declared_kind_narrowed.add(rid)
            evidences[rid].append({
                "source": "FM_DECLARED_ENTITY_KIND",
                "status": "FAMILY_OF_THE_DECLARED_ENTITY_KIND_SELECTED",
                "declared_entity_kind": str(role.get("entity_kind", "")),
                "explicit_families": list(evidence.explicit_families),
            })

    binary_constraints: list[tuple[str, str, set[tuple[str, str]], dict[str, Any]]] = []
    for relation in document.get("functional_relations", ()):
        rs, ro = relation.get("subject_role"), relation.get("object_role")
        if rs not in roles or ro not in roles: continue
        phrase = str(relation.get("relation", relation.get("predicate", "")))
        pairs: set[tuple[str, str]] = set()
        meanings = extract_relation_semantic_candidates(domain, phrase)
        # One phrase can carry several readings at once: "the control is
        # supported by the surface" states an end state *and* says that the
        # surface is a support.  Dropping the whole relation because one reading
        # was a task effect threw the endpoint evidence away with it, and the
        # only place that evidence existed was this sentence.
        #
        # Several readings are a disjunction, so the constraint is the union of
        # what each reading admits.  A union can only under-constrain, never
        # narrow onto the wrong reading -- which is what taking one reading
        # alone would risk, since "the cup contains the coffee" read only as an
        # insertion would type the coffee as the thing inserted.
        readings: list[str] = []
        for meaning in meanings:
            current = relation_canonical_role_pairs(domain, meaning.predicate_name, meaning.category)
            if not current:
                continue
            if meaning.category == "PHYSICAL_VERIFIER" and meaning.predicate_name in {
                "NEAR_SEAT", "ACCESSIBLE_FROM_BOTH_SEATS", "COMPATIBLE_WITH"
            }:
                current |= {(o, s) for s, o in current}
            if meaning.direction == "REVERSE": current = {(o, s) for s, o in current}
            if relation.get("unordered_participants"):
                current |= {(o, s) for s, o in current}
            pairs |= current
            readings.append(f"{meaning.category}:{meaning.predicate_name}:{meaning.direction}")
        if pairs:
            binary_constraints.append((rs, ro, pairs, {
                "source": "RELATION_TEXT", "raw_phrase": phrase,
                "readings_unioned": readings,
            }))

    for operation in document.get("interaction_groups", ()) or document.get("operations", ()):
        rs = operation.get("tool_role") or operation.get("source_role")
        ro, ra = operation.get("target_role"), operation.get("context_role") or operation.get("anchor_role")
        if rs not in roles or ro not in roles: continue
        phrase = str(operation.get("function") or operation.get("operation") or "")
        capabilities = extract_operation_semantic_candidates(domain, phrase)
        v3_assignments = operation.get("v3_slot_assignments", [])
        if v3_assignments:
            raw_layouts = {
                (row.get("source_role"), row.get("target_role"), row.get("anchor_role"))
                for row in v3_assignments
            }
            pairs = set() if len(raw_layouts) > 1 else {
                (row["source_type"], row["target_type"])
                for row in v3_assignments
            }
        else:
            pairs = {(s, t) for cap in capabilities for s in cap.allowed_source_roles for t in cap.allowed_target_roles}
            if domain == "living_room": pairs |= {(t, s) for s, t in pairs}
        pairs = {(s, t) for s, t in pairs if s in runtime_roles and t in runtime_roles}
        if pairs and rs != ro:
            binary_constraints.append((rs, ro, pairs, {
                "source": "OPERATION_TEXT", "raw_phrase": phrase,
                "capabilities": [cap.capability_id for cap in capabilities],
            }))
        if ra in roles and capabilities and not (
            v3_assignments and len({row.get("anchor_role") for row in v3_assignments}) > 1
        ):
            allowed_anchors = {a for cap in capabilities for a in cap.allowed_anchor_roles} & runtime_roles
            before = set(domains[ra])
            constrained[ra].add("OPERATION_TEXT")
            evidences[ra].append({"source": "OPERATION_TEXT", "raw_phrase": phrase,
                                  "allowed_anchor_roles": sorted(allowed_anchors)})
            narrowed = before & allowed_anchors
            if before and not narrowed:
                structural_conflicts.add(ra)
            else:
                domains[ra] = narrowed

    changed = True
    while changed:
        changed = False
        for rs, ro, pairs, detail in binary_constraints:
            viable = {(s, o) for s, o in pairs if s in domains[rs] and o in domains[ro]}
            before_s, before_o = set(domains[rs]), set(domains[ro])
            if not viable:
                structural_conflicts.update((rs, ro))
                continue
            domains[rs] &= {s for s, _ in viable}
            domains[ro] &= {o for _, o in viable}
            constrained[rs].add(detail["source"]); constrained[ro].add(detail["source"])
            if detail not in evidences[rs]: evidences[rs].append(detail)
            if detail not in evidences[ro]: evidences[ro].append(detail)
            changed |= before_s != domains[rs] or before_o != domains[ro]

    # The contract declares its participants separately, so two of them are two
    # things.  A runtime role that some other participant is already pinned to,
    # and pinned to alone, is therefore not what an undecided participant is --
    # reading it that way would collapse two declared participants into one.
    #
    # Only undecided participants are narrowed, and only by roles another
    # participant has no alternative to.  Two participants the contract really
    # does mean as the same role stay as they are, because neither is undecided.
    # Without this, a participant whose own words name no category -- "holds
    # refreshments for a person" -- stayed ambiguous between the payloads of the
    # domain, the operation over it had two equally good readings, and slot
    # completion refused it on the tie.
    # Read against what each participant actually resolves to, which is the
    # arc-consistent domain after the single-preferred-role reduction below --
    # not the raw domain.  A participant the function text names outright still
    # carries the whole domain here, and comparing raw domains found nothing
    # pinned anywhere.
    def _resolved(rid: str) -> set[str]:
        values = set(domains[rid])
        preferred_here = set(functions[rid].preferred_roles) & values
        if len(preferred_here) == 1 and (
            functions[rid].evidence_strength == "EXPLICIT_FAMILY" or not constrained[rid]
        ):
            return preferred_here
        return values

    pinned_elsewhere = {
        next(iter(resolved))
        for other in roles
        if len(resolved := _resolved(other)) == 1
    }
    for rid in roles:
        # Same restriction as the entity-kind reading above, for the same
        # reason: a participant the contract characterizes in no way is
        # ambiguous over the whole domain because nothing is known about it, and
        # striking the roles other participants claimed would leave a smaller
        # set that looks like knowledge.  Only a participant whose family the
        # contract settled is narrowed here.
        if not functions[rid].explicit_families:
            continue
        resolved_here = _resolved(rid)
        if len(resolved_here) <= 1:
            continue
        taken = resolved_here & pinned_elsewhere
        narrowed = resolved_here - pinned_elsewhere
        if not taken or not narrowed:
            continue
        domains[rid] = narrowed
        constrained[rid].add("DECLARED_PARTICIPANT_DISTINCTNESS")
        evidences[rid].append({
            "source": "DECLARED_PARTICIPANT_DISTINCTNESS",
            "status": "ROLES_ANOTHER_DECLARED_PARTICIPANT_IS_PINNED_TO_EXCLUDED",
            "excluded_roles": sorted(taken),
            "remaining_roles": sorted(narrowed),
        })

    result = {}
    for rid, role in roles.items():
        domain_values = set(domains[rid])
        # ``domains`` already had the contradictory kinds taken out; intersecting
        # here keeps a preferred role of the wrong kind from being restored as
        # the single answer below.
        preferred = set(functions[rid].preferred_roles) & domain_values
        structural = constrained[rid]
        # The declared entity kind is structural evidence in the same sense the
        # relation and operation text are: it comes from the contract's own
        # shape rather than from reading a function phrase.  So when it is what
        # ruled the weak alias out, that counts as the alias being overridden,
        # and the resulting type is bindable where a merely-guessed one is not.
        structurally_evidenced = structural or (
            {"FM_DECLARED_ENTITY_KIND"} if rid in declared_kind_narrowed else set())
        override = bool(structurally_evidenced and weak_mapped[rid] is not None
                        and weak_mapped[rid] not in domain_values and domain_values)
        if override:
            evidences[rid].append({"source": "JOINT_TYPING", "status": "STRUCTURAL_OVERRIDE_OF_WEAK_FUNCTION_ALIAS"})
        if len(preferred) == 1 and (
            functions[rid].evidence_strength == "EXPLICIT_FAMILY" or not structural
        ):
            domain_values = preferred
        values = tuple(sorted(domain_values))
        if not values:
            status = "CONTRADICTORY_ROLE_TYPE" if rid in structural_conflicts else "UNREPRESENTED_SEMANTIC"
        elif len(values) > 1:
            status = "AMBIGUOUS_ROLE_TYPE" if structural else "UNCONSTRAINED_ROLE_TYPE"
        elif override:
            status = "STRUCTURAL_OVERRIDE_OF_WEAK_FUNCTION_ALIAS"
        elif functions[rid].evidence_source == "GLOBAL_GRAPH_CONSISTENCY_OVERRIDE" and len(values) == 1:
            status = "GLOBAL_GRAPH_CONSISTENCY_OVERRIDE"
            evidences[rid].append({
                "source": "GLOBAL_GRAPH_CONSISTENCY_OVERRIDE",
                "status": "UNIQUE_MINIMAL_REPAIR",
                "raw_function": str(role.get("function", "")),
                "resolved_role": values[0],
            })
        elif rid in runtime_authored: status = str(role.get("provenance"))
        elif len(values) == 1 and weak_mapped[rid] == values[0]: status = "DIRECT_FUNCTION_MATCH"
        elif structural == {"RELATION_TEXT"}: status = "RELATION_ASSISTED"
        elif structural == {"OPERATION_TEXT"}: status = "OPERATION_ASSISTED"
        elif structural: status = "JOINT_SEMANTIC_RESOLUTION"
        else: status = "DIRECT_FUNCTION_MATCH"
        result[rid] = RoleTypeHypothesis(
            raw_role_id=rid, raw_function=str(role.get("function", "")),
            raw_description=str(role.get("description", "")), entity_kind=str(role.get("entity_kind", "OBJECT")),
            canonical_role_candidates=values, status=status, evidence=tuple(evidences[rid]),
            runtime_context_only=functions[rid].runtime_context_only,
        )
    return result


def operation_participant_satisfiable(
    domain: str,
    operation: dict[str, Any],
    hypotheses: dict[str, RoleTypeHypothesis],
) -> bool:
    """Check whether some semantic-family/type interpretation fits an operation."""
    rs = operation.get("source_role") or operation.get("tool_role")
    rt = operation.get("target_role")
    ra = operation.get("anchor_role") or operation.get("context_role")
    capabilities = extract_operation_semantic_candidates(domain, str(operation.get("operation") or operation.get("function") or ""))
    if not capabilities: return True
    source_h, target_h = hypotheses[rs], hypotheses[rt]
    source_types, target_types = set(source_h.canonical_role_candidates), set(target_h.canonical_role_candidates)
    source_families = {e for row in source_h.evidence for e in row.get("explicit_families", [])}
    target_families = {e for row in target_h.evidence for e in row.get("explicit_families", [])}
    anchor_types = set(hypotheses[ra].canonical_role_candidates) if ra else set()
    anchor_families = {e for row in hypotheses[ra].evidence for e in row.get("explicit_families", [])} if ra else set()
    for cap in capabilities:
        orientations = [(set(cap.allowed_source_roles), set(cap.allowed_target_roles))]
        if domain == "living_room": orientations.append((orientations[0][1], orientations[0][0]))
        for allowed_s, allowed_t in orientations:
            sf = {canonical_role_family(domain, x) for x in allowed_s}
            tf = {canonical_role_family(domain, x) for x in allowed_t}
            source_ok = bool(source_types & allowed_s) or bool(source_families & sf)
            target_ok = bool(target_types & allowed_t) or bool(target_families & tf)
            if not (source_ok and target_ok): continue
            if ra:
                af = {canonical_role_family(domain, x) for x in cap.allowed_anchor_roles}
                if not (anchor_types & set(cap.allowed_anchor_roles) or anchor_families & af): continue
            return True
    return False
