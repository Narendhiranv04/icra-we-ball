"""Robot Capability Registry and Explicit Operation Semantic Interpreter.

Defines the physical robot capabilities available in the runtime execution layer.
Separates FM task-level operation expression from backend physical feasibility requirements:
- The FM expresses that an operation is needed (e.g. 'stir beverage in cups').
- The capability registry matches the operation to an explicit robot capability.
- Physical feasibility preconditions (e.g. INSERTABLE_IN, REACHES_BOTTOM) are attached
  only after an explicit operation maps to a capability.
- Endpoint roles alone CANNOT infer or synthesize an operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


def _phrase(s: str) -> str:
    """Normalize a phrase for robust semantic matching."""
    s = s.lower().strip()
    s = re.sub(r"[_\-\/\\]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


_NON_PHYSICAL_LEADING_ACTIONS = frozenset({
    "choose", "chooses", "choosing", "chose", "chosen",
    "detect", "detects", "detecting", "detected",
    "find", "finds", "finding", "found",
    "identify", "identifies", "identifying", "identified",
    "inspect", "inspects", "inspecting", "inspected",
    "inspector",
    "locate", "locates", "locating", "located",
    "recognize", "recognizes", "recognizing", "recognized",
    "recognise", "recognises", "recognising", "recognised",
    "search", "searches", "searching", "searched",
    "select", "selects", "selecting", "selected",
    # Getting hold of something that is stowed away is the search the runtime
    # already performs: it opens the regions it was told about and looks in
    # them.  Read as a final capability instead, "retrieve_from_storage" became
    # an operation nothing in the runtime could represent, and the trial failed
    # for lacking a capability whose work the runtime does elsewhere.  Opening
    # and examining are the same phase.
    "retrieve", "retrieves", "retrieving", "retrieved",
    "collect", "collects", "collecting", "collected",
    "fetch", "fetches", "fetching", "fetched",
    "gather", "gathers", "gathering", "gathered",
    "obtain", "obtains", "obtaining", "obtained",
    "acquire", "acquires", "acquiring", "acquired",
    "open", "opens", "opening", "opened",
    "examine", "examines", "examining", "examined",
    "check", "checks", "checking", "checked",
    "verify", "verifies", "verifying", "verified",
    "scan", "scans", "scanning", "scanned",
    "look", "looks", "looking", "looked",
    "confirm", "confirms", "confirming", "confirmed",
    "determine", "determines", "determining", "determined",
    "assess", "assesses", "assessing", "assessed",
})


# The model often writes the actor into the operation phrase.  The actor is
# always this robot, so the words naming it are not part of the action, and
# leaving them in front made the leading-action tests read "the" as the verb.
_LEADING_AGENT_PHRASE = re.compile(
    r"^(?:the\s+)?(?:robot(?:ic)?(?:\s+(?:arm|manipulator|hand|gripper))?|arm|"
    r"manipulator|gripper|end\s+effector|system|agent)\s+", re.I)


# A coordinator, then a physical action word: the mark of a phrase that names a
# perception step and a physical change together.
_COORDINATED_PHYSICAL_ACTION = re.compile(
    r"\b(?:and|then|before|after|so as to|in order to|to)\b[^.;]{0,40}?"
    r"\b(?:place\w*|put\w*|insert\w*|pour\w*|fill\w*|transfer\w*|move\w*|"
    r"stir\w*|mix\w*|fasten\w*|screw\w*|driv\w*|tighten\w*|attach\w*|"
    r"return\w*|deposit\w*|position\w*|set down|lay\w*|combin\w*|"
    r"secur\w*|instal\w*|join\w*|assembl\w*|mount\w*|affix\w*|connect\w*|"
    r"relocat\w*|bolt\w*|rivet\w*|decant\w*|dispens\w*)\b",
    re.I,
)


def is_non_physical_operation_phrase(
    raw_phrase: str, participant_roles: Sequence[str] = ()
) -> bool:
    """Return whether the phrase leads with an explicitly non-physical action.

    Only the leading action token is considered.  This deliberately permits
    physical operations containing later adjectival forms, such as
    ``place selected component``.
    """
    normalized = _LEADING_AGENT_PHRASE.sub("", _phrase(raw_phrase))
    if not normalized:
        return False
    if normalized.split(maxsplit=1)[0] not in _NON_PHYSICAL_LEADING_ACTIONS:
        return False
    # The participants' own names are not the action.  "Search for and verify
    # the fastening_component" contains the word "fastening" only because that
    # is what the model called the part, and reading it as a fastening turned a
    # search directive into a requirement the runtime could not represent.
    action_only = operation_action_phrase(raw_phrase, participant_roles)
    # Leading with a perception verb does not make the whole phrase perception.
    # "Locate the parts and perform the fastening", "find the screw then drive
    # it into the joint": the model has named an acquisition step and a physical
    # change in one breath, and dismissing the phrase on its first word threw
    # the change away with the search.  The acquisition half is the runtime's
    # own business; the physical half is an operation it must not lose.
    #
    # What makes such a phrase compound is the coordination, and that is what is
    # tested -- not the mere presence of a physical word.  "Select the
    # compatible fastening component" names one act, selection, and the word
    # "fastening" in it modifies a noun; nothing is coordinated with it.
    return not _COORDINATED_PHYSICAL_ACTION.search(action_only or normalized)


# Words that state what must be true when the task is done, rather than a
# motion the robot makes.  The robot's own actions are motions: place, transfer,
# insert, stir, fasten, return.  "Serve the soup", "prepare the coffee",
# "provide a refreshment" are not motions, and inventing a physical endpoint for
# one -- moving a bowl into a person -- puts words in the model's mouth.
#
# The same words are also legitimate cues for real capabilities: "serve soup" is
# how the model usually asks for the eating utensil.  So this is consulted only
# after capability matching has already failed, never instead of it.
_ABSTRACT_TASK_DIRECTIVE_ACTIONS = frozenset({
    "serve", "serves", "serving", "served",
    "prepare", "prepares", "preparing", "prepared",
    "provide", "provides", "providing", "provided",
    "arrange", "arranges", "arranging", "arranged",
    "distribute", "distributes", "distributing", "distributed",
    "offer", "offers", "offering", "offered",
    "assign", "assigns", "assigning", "assigned",
    "ensure", "ensures", "ensuring", "ensured",
    "hand", "hands", "handing", "handed",
    "deliver", "delivers", "delivering", "delivered",
})

# A motion the runtime recognises anywhere in the phrase keeps it physical, so a
# directive that also says how it is to be carried out is not reinterpreted.
_PHYSICAL_MOTION_WORD = re.compile(
    r"\b(place\w*|put\w*|insert\w*|pour\w*|fill\w*|transfer\w*|move\w*|"
    r"stir\w*|mix\w*|fasten\w*|screw\w*|driv\w*|tighten\w*|attach\w*|"
    r"return\w*|deposit\w*|position\w*|set down|lay\w*|combin\w*|add\w*|"
    r"secur\w*|instal\w*|join\w*|assembl\w*|mount\w*|affix\w*|connect\w*|"
    r"relocat\w*|bolt\w*|rivet\w*|decant\w*|dispens\w*)\b",
    re.I,
)


def leads_with_abstract_task_directive(raw_phrase: str) -> bool:
    """Whether the phrase states a desired end state instead of a motion."""
    normalized = _phrase(raw_phrase)
    if not normalized:
        return False
    if _PHYSICAL_MOTION_WORD.search(normalized):
        return False
    return normalized.split(maxsplit=1)[0] in _ABSTRACT_TASK_DIRECTIVE_ACTIONS


@dataclass(frozen=True)
class RobotCapability:
    """A physical capability executable by the robot runtime in a domain."""

    domain: str
    capability_id: str
    semantic_description: str
    allowed_source_roles: tuple[str, ...]
    allowed_target_roles: tuple[str, ...]
    allowed_anchor_roles: tuple[str, ...]
    required_relation_templates: tuple[tuple[str, str, str], ...]
    planner_operation: str
    semantic_cues: tuple[str, ...] = ()
    # What carrying this capability out brings about, as (predicate, subject
    # slot, object slot) over the slots "source", "target" and "anchor".  This
    # is knowledge about the robot's own actions, in the same category as the
    # preconditions above, and it is only ever used to recognize that an end
    # state the FM stated is the end state of an operation the FM independently
    # expressed.  It never creates an operation from a stated end state.
    achieved_effects: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "capability_id": self.capability_id,
            "semantic_description": self.semantic_description,
            "allowed_source_roles": list(self.allowed_source_roles),
            "allowed_target_roles": list(self.allowed_target_roles),
            "allowed_anchor_roles": list(self.allowed_anchor_roles),
            "required_relation_templates": [list(t) for t in self.required_relation_templates],
            "planner_operation": self.planner_operation,
            "semantic_cues": list(self.semantic_cues),
            "achieved_effects": [list(row) for row in self.achieved_effects],
        }


@dataclass(frozen=True)
class OperationInterpretationResult:
    """Outcome of interpreting free-form operation text against robot capabilities."""

    raw_operation: str
    canonical_source: str
    canonical_target: str
    canonical_anchor: str | None
    capability: RobotCapability | None
    status: str  # EXACT_CAPABILITY_MATCH, LEXICAL_SEMANTIC_MATCH, UNMAPPABLE_OPERATION, AMBIGUOUS_CAPABILITY
    reason: str
    physical_preconditions: tuple[tuple[str, str, str], ...] = ()
    planner_operation: str | None = None
    required_relations: tuple[str, ...] = ()
    context_relations: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status in ("EXACT_CAPABILITY_MATCH", "LEXICAL_SEMANTIC_MATCH") and self.capability is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_operation": self.raw_operation,
            "canonical_source": self.canonical_source,
            "canonical_target": self.canonical_target,
            "canonical_anchor": self.canonical_anchor,
            "capability_id": self.capability.capability_id if self.capability else None,
            "status": self.status,
            "reason": self.reason,
            "physical_preconditions": [list(t) for t in self.physical_preconditions],
            "planner_operation": self.planner_operation,
            "required_relations": list(self.required_relations),
            "context_relations": list(self.context_relations),
        }


# Canonical domain robot capabilities
CANONICAL_ROBOT_CAPABILITIES: dict[str, tuple[RobotCapability, ...]] = {
    "kitchen": (
        RobotCapability(
            domain="kitchen",
            capability_id="STIR_COFFEE",
            semantic_description="Stir or agitate liquid in a beverage container using a stirring implement.",
            allowed_source_roles=("stirrer", "coffee_stirrer"),
            allowed_target_roles=("prepared_cup_target", "coffee_container", "beverage_cup", "cup", "coffee_cup"),
            allowed_anchor_roles=(),
            required_relation_templates=(
                ("source", "INSERTABLE_IN", "target"),
                ("source", "REACHES_BOTTOM", "target"),
            ),
            planner_operation="STIR_COFFEE",
            semantic_cues=(
                "stir", "stirring", "stir coffee", "stir beverage", "mix", "mixing",
                "blend", "blending", "agitate", "swirl", "coffee stirring",
                "stir beverage in cups", "stir coffee in mug", "stir drink",
                "stir contents", "stir liquid", "stir or agitate liquid",
            ),
        ),
        RobotCapability(
            domain="kitchen",
            capability_id="PROVIDE_SOUP_EATING_UTENSIL",
            semantic_description="Provide or insert a suitable eating utensil into a soup or food container.",
            allowed_source_roles=("eating_utensil", "soup_eating_utensil"),
            allowed_target_roles=("soup_bowl_target", "soup_container", "soup_bowl", "bowl"),
            allowed_anchor_roles=(),
            required_relation_templates=(
                ("source", "INSERTABLE_IN", "target"),
                ("source", "REACHES_BOTTOM", "target"),
            ),
            planner_operation="PROVIDE_SOUP_EATING_UTENSIL",
            # A utensil going into a bowl is an insertion, which this capability's own
            # preconditions already check; calling it a placement-on would need
            # a carrier reading the endpoint families do not admit.
            achieved_effects=(),
            semantic_cues=(
                "eating utensil", "provide utensil", "place utensil", "serve soup",
                "spoon soup", "eat soup",
                "utensil for soup", "provide eating utensil for each soup bowl",
                "provide soup utensil", "soup eating utensil", "utensil in soup",
                "place associated utensil", "associate eating utensil with soup",
                "provide eating utensil",
            ),
        ),
        RobotCapability(
            domain="kitchen",
            capability_id="TRANSFER_CONTENT_TO_CONTAINER",
            semantic_description="Transfer or pour granular or liquid material from a source vessel into a destination container.",
            allowed_source_roles=("coffee_source", "water_source", "source", "ingredient"),
            allowed_target_roles=("coffee_container", "prepared_cup_target", "beverage_cup", "cup", "target_container"),
            allowed_anchor_roles=(),
            required_relation_templates=(),
            planner_operation="POUR",
            achieved_effects=(("CONTAINS", "target", "source"),),
            semantic_cues=(
                "pour", "pouring", "transfer", "transferring", "pour coffee", "pour water",
                "dispense", "dispensing", "fill", "filling", "fill cup", "pour liquid",
                "transfer coffee", "transfer water", "transfer content to container",
                "transfer material into container", "transfer material to container",
                "transfer material", "dispense coffee into cups", "pour water into cups",
                "pour hot water into cups", "pour coffee grounds into cups",
                "pour into container", "pour into cup", "pour beverage",
            ),
        ),
    ),
    "living_room": (
        RobotCapability(
            domain="living_room",
            capability_id="SUPPORT_DRINKWARE",
            semantic_description="Support personal drinkware on an adjacent surface near a seating location.",
            allowed_source_roles=("PERSONAL_CUP_SAUCER_REGION", "DRINKWARE_SUPPORT"),
            allowed_target_roles=("CUP_SAUCER_SET", "DRINKWARE"),
            # NEAR_SEAT is checked against one seating position.  Listing the
            # pair as well made the two anchor forms look interchangeable, and
            # the predicate that has to verify the result does not accept it.
            allowed_anchor_roles=("SEATING_POSITION",),
            required_relation_templates=(
                ("source", "FITS_SET_ON", "target"),
                ("source", "NEAR_SEAT", "anchor"),
            ),
            planner_operation="SUPPORT_DRINKWARE",
            achieved_effects=(("PLACED_ON", "target", "source"),),
            semantic_cues=(
                "drinkware", "support drinkware", "personal support",
                "personal support group", "cup and saucer", "cup saucer",
                "place drinkware", "support cup", "refreshment", "personal refreshment",
                "drinkware support", "support drink", "support beverage",
                "place refreshment setting on personal support", "place payload on support",
                "support drinkware set beside seat", "support drinkware set",
                "transfer container to surface", "transfer refreshment pair to table",
                "transfer refreshment setting", "distribute refreshment sets",
                "distribute refreshment settings", "place refreshment item on surface",
            ),
        ),
        RobotCapability(
            domain="living_room",
            capability_id="SUPPORT_ENTERTAINMENT_CONTROL",
            semantic_description="Support shared remote control on a central surface accessible from multiple seating positions.",
            allowed_source_roles=("SHARED_REMOTE_REGION", "REMOTE_SUPPORT"),
            allowed_target_roles=("REMOTE", "REMOTE_CONTROL"),
            # Being reachable from both seats is a property of the pair.  One
            # individual seating position cannot witness it.
            allowed_anchor_roles=("SEATING_PAIR",),
            required_relation_templates=(
                ("source", "FITS_ON", "target"),
                ("source", "ACCESSIBLE_FROM_BOTH_SEATS", "anchor"),
            ),
            planner_operation="SUPPORT_ENTERTAINMENT_CONTROL",
            achieved_effects=(("PLACED_ON", "target", "source"),),
            semantic_cues=(
                "entertainment", "remote", "remote control", "control",
                "shared entertainment", "shared entertainment group",
                "support remote", "place remote", "television control",
                "shared control", "entertainment control", "support controller",
                "place entertainment control on shared support",
                "support television remote control", "support remote control",
                "transfer device to surface", "move entertainment control to surface",
                "relocate entertainment control", "relocate device to shared spot",
                "transport control device to central area", "transfer entertainment controller",
            ),
        ),
    ),
    "workshop": (
        RobotCapability(
            domain="workshop",
            capability_id="FASTEN_JOINT",
            semantic_description="Fasten or secure a frame joint hole using a compatible screw driven by a driving tool.",
            allowed_source_roles=("driver", "fastening_tool"),
            allowed_target_roles=("fastener",),
            allowed_anchor_roles=("repair_target",),
            required_relation_templates=(
                ("source", "COMPATIBLE_WITH", "target"),
                ("source", "REACHES_TARGET", "anchor"),
                ("target", "COMPATIBLE_WITH_TARGET", "anchor"),
            ),
            planner_operation="DRIVE_FASTENER_INTO_TARGET",
            achieved_effects=(("INSTALLED_AT", "target", "anchor"),
                                ("CONNECTED_TO", "target", "anchor"),),
            semantic_cues=(
                "fasten", "fastening", "drive screw", "drive fastener", "fasten joint",
                "secure joint", "repair joint", "tighten screw",
                "drive screw into joint target", "drive fastener into target",
                "drive fastener into joint", "screw joint", "drive screw into joint",
                "fasten the joint", "drive screw into workpiece",
                "install component at target", "fasten component at target",
                "install component at anchor", "drive fastener into target",
                "execute fastening", "perform fastening", "install fastener into target",
                "fasten component", "install component", "install fastener",
                "install/fasten component at target",
                "tighten", "tightening", "apply fastening action", "apply fastening force",
                "complete fastening", "secure fastener", "secure connection",
            ),
        ),
        RobotCapability(
            domain="workshop",
            capability_id="RETURN_REUSABLE_ITEM_TO_SUPPORT",
            semantic_description="Return a reusable tool or implement back to a workbench surface or support region after use.",
            allowed_source_roles=("driver", "tool", "fastening_tool"),
            allowed_target_roles=("MAIN_WORKBENCH_ZONE", "workbench_surface", "workbench"),
            allowed_anchor_roles=(),
            required_relation_templates=(),
            planner_operation="PLACE",
            achieved_effects=(("PLACED_ON", "source", "target"),),
            semantic_cues=(
                "return", "returning", "return driver", "return tool", "place driver",
                "return to workbench", "put back", "return reusable item to support",
                "return equipment", "return reusable equipment to workbench",
                "place tool on workbench", "place reusable tool", "restore tool",
                "deposit tool", "store equipment", "set tool down", "settle on surface",
                "place equipment onto workbench", "place tool down",
            ),
        ),
    ),
}


def get_robot_capabilities(domain: str) -> tuple[RobotCapability, ...]:
    """Retrieve all canonical robot capabilities registered for domain."""
    d_norm = domain.strip().lower()
    return CANONICAL_ROBOT_CAPABILITIES.get(d_norm, ())


def get_robot_capability_registry_hash() -> str:
    """Compute a deterministic SHA256 fingerprint of the robot capability registry."""
    serialized = []
    for dom in sorted(CANONICAL_ROBOT_CAPABILITIES.keys()):
        for cap in CANONICAL_ROBOT_CAPABILITIES[dom]:
            serialized.append(cap.to_dict())
    canonical_json = json.dumps(serialized, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def operation_action_phrase(raw_phrase: str, participant_roles: Sequence[str] = ()) -> str:
    """The operation phrase with the participant names it repeats taken out.

    A capability is identified by the action; the participants are given
    separately.  Leaving their names in the text let "place the fastening_tool
    on the workbench" read as a fastening, because a cue matched inside the
    tool's own name -- and the runtime then induced a second fastening for it.
    """
    normalized = _phrase(raw_phrase)
    for role in sorted(participant_roles or (), key=len, reverse=True):
        role_words = _phrase(role)
        if not role_words:
            continue
        normalized = re.sub(rf"\b{re.escape(role_words)}\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_operation_semantic_candidates(
    domain: str,
    raw_phrase: str,
    participant_roles: Sequence[str] = (),
) -> tuple[RobotCapability, ...]:
    """Extract physical capability meanings from text without endpoint filtering."""
    norm_text = operation_action_phrase(raw_phrase, participant_roles)
    if not norm_text or is_non_physical_operation_phrase(raw_phrase, participant_roles):
        return ()
    matched: list[RobotCapability] = []
    for capability in get_robot_capabilities(domain):
        if norm_text in (_phrase(capability.capability_id), _phrase(capability.planner_operation)):
            matched.append(capability)
            continue
        # Word boundaries, not raw substrings: "fasten" does not occur in
        # "fastening tool" as a word, and treating it as a hit made every phrase
        # naming that tool look like a fastening.
        if any(
            (cue_norm := _phrase(cue)) == norm_text
            or (len(cue_norm) >= 4 and re.search(rf"\b{re.escape(cue_norm)}\b", norm_text))
            for cue in capability.semantic_cues
        ):
            matched.append(capability)
    if not matched:
        # Fallback on the leading action word, by stem rather than by exact
        # token.  The model inflects and nominalizes freely -- "tightened",
        # "inserted_into", "placement" -- and an exact match read every one of
        # those as naming no action at all.  A stem that several capabilities
        # share proposes all of them; which one can actually seat the named
        # participants is settled by the slot resolution, and an operation two
        # capabilities could seat stays ambiguous rather than being guessed.
        matched.extend(_capabilities_by_leading_action_stem(domain, norm_text))
    return tuple(sorted(set(matched), key=lambda item: item.capability_id))


# Action stems, per domain, mapped to the capabilities they could name.  Stems
# rather than words, so inflections and nominalizations of the same action are
# one entry; ambiguous stems deliberately name more than one capability.
_LEADING_ACTION_STEMS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "living_room": (
        (r"transfer|mov|relocat|plac|position|transport|set|plce", ("__ALL__",)),
    ),
    "workshop": (
        (r"secur|fasten|tighten|driv|instal|screw|bolt|insert|affix|rivet",
         ("FASTEN_JOINT",)),
        (r"return|deposit|restor|stor|put|leav|rest|stow|park",
         ("RETURN_REUSABLE_ITEM_TO_SUPPORT",)),
        # Genuinely ambiguous in this domain: a component may be placed into the
        # joint, and a tool may be placed back on the bench.
        (r"plac|attach|join|appl|mount|assembl|fit|connect",
         ("FASTEN_JOINT", "RETURN_REUSABLE_ITEM_TO_SUPPORT")),
    ),
    "kitchen": (
        (r"pour|transfer|dispens|fill|decant|combin|add",
         ("TRANSFER_CONTENT_TO_CONTAINER",)),
        # Genuinely ambiguous in this domain: ingredients are mixed *into* a cup,
        # and the drink already in the cup is mixed *with* an implement.  Naming
        # both leaves the endpoints to say which one a phrase meant, instead of
        # reading "use the stirrer to mix the coffee in the mug" as a transfer.
        (r"mix|blend", ("TRANSFER_CONTENT_TO_CONTAINER", "STIR_COFFEE")),
        (r"stir|agitat|swirl|whisk", ("STIR_COFFEE",)),
        (r"provid|accompan|pair", ("PROVIDE_SOUP_EATING_UTENSIL",)),
    ),
}


# Words that can stand in front of the action without being it.
_PRE_ACTION_FUNCTION_WORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "get", "gets", "got",
    "must", "should", "shall", "will", "would", "can", "could", "may", "might",
    "to", "the", "a", "an", "then", "also", "now", "next", "first", "finally",
    "and", "or", "it", "its", "their", "them", "that", "this", "which", "who",
    "carefully", "safely", "gently", "firmly", "securely", "properly", "neatly",
})


def _capabilities_by_leading_action_stem(domain: str, norm_text: str) -> list[RobotCapability]:
    """Capabilities named by the first word in the phrase that names an action.

    Only the very first token used to be examined, which read the passive voice
    as actionless: "fastening_tool is placed on workbench" leads with "is", and
    the tool return the model plainly expressed matched nothing at all.  Leading
    auxiliaries, determiners, modals and manner adverbs are skipped, and the
    first token that does name an action decides -- so a later noun cannot
    reinterpret an operation whose verb was already found.
    """
    available = get_robot_capabilities(domain)
    stem_rules = _LEADING_ACTION_STEMS.get(domain, ())
    for token in norm_text.split():
        if token in _PRE_ACTION_FUNCTION_WORDS:
            continue
        found: list[RobotCapability] = []
        for stems, capability_ids in stem_rules:
            if not re.fullmatch(rf"(?:{stems})\w*", token):
                continue
            if capability_ids == ("__ALL__",):
                found.extend(available)
            else:
                found.extend(cap for cap in available if cap.capability_id in capability_ids)
        if found:
            return found
        # A content word that names no action at all is a participant the model
        # wrote into the phrase; keep looking rather than stopping on it.
    return []


def interpret_operation(
    domain: str,
    raw_phrase: str,
    source_role: str,
    target_role: str,
    anchor_role: str | None = None,
    capability_hint: str | None = None,
) -> OperationInterpretationResult:
    """Safely map free-form operation text to a robot capability.

    RULE: Endpoint roles are filters, NEVER the source of operation semantics.
    1. Extract semantic candidate capabilities from raw_phrase text.
    2. Intersect with capabilities valid for canonical endpoints (source, target, anchor).
    3. If 0 semantic candidates or 0 valid in intersection: fail closed (UNMAPPABLE_OPERATION).
    4. If unique match: compile capability and instantiate physical preconditions.
    5. If multiple matches without clear winner: fail closed (AMBIGUOUS_CAPABILITY).
    """
    d_norm = domain.strip().lower()
    norm_text = _phrase(raw_phrase)
    capabilities = get_robot_capabilities(d_norm)

    if not norm_text:
        return OperationInterpretationResult(
            raw_operation=raw_phrase,
            canonical_source=source_role,
            canonical_target=target_role,
            canonical_anchor=anchor_role,
            capability=None,
            status="UNMAPPABLE_OPERATION",
            reason="Empty operation phrase; operations must not be inferred from endpoints alone",
        )

    if is_non_physical_operation_phrase(raw_phrase):
        return OperationInterpretationResult(
            raw_operation=raw_phrase,
            canonical_source=source_role,
            canonical_target=target_role,
            canonical_anchor=anchor_role,
            capability=None,
            status="UNMAPPABLE_OPERATION",
            reason=(
                "NON_PHYSICAL_OPERATION: operation phrase leads with a cognitive, "
                "selection, search, or inspection action"
            ),
        )

    # 1. Semantic candidates based on text evidence
    semantic_matches = set(extract_operation_semantic_candidates(d_norm, raw_phrase))

    # A caller that already identified the capability from the participant
    # signature supplies it here.  The model frequently names an operation with a
    # bare noun -- "manipulation", "placement" -- which carries no text evidence,
    # while its participants determine exactly one capability.  The hint is only
    # honoured when the phrase itself yields nothing, so text evidence still wins
    # wherever it exists, and the hint must name a real capability of the domain.
    if not semantic_matches and capability_hint:
        semantic_matches = {
            capability for capability in capabilities
            if capability.capability_id == capability_hint
        }

    if not semantic_matches:
        return OperationInterpretationResult(
            raw_operation=raw_phrase,
            canonical_source=source_role,
            canonical_target=target_role,
            canonical_anchor=anchor_role,
            capability=None,
            status="UNMAPPABLE_OPERATION",
            reason=f"No semantic evidence supporting any robot capability in domain {domain!r} for phrase {raw_phrase!r}",
        )

    # 2. Endpoint filtering
    endpoint_matches: set[RobotCapability] = set()
    for cap in capabilities:
        if source_role not in cap.allowed_source_roles:
            continue
        if target_role not in cap.allowed_target_roles:
            continue
        if anchor_role is not None and cap.allowed_anchor_roles:
            if anchor_role not in cap.allowed_anchor_roles:
                continue
        endpoint_matches.add(cap)

    # 3. Intersect semantic candidates with endpoint-valid capabilities
    intersection = semantic_matches.intersection(endpoint_matches)

    # A wording the endpoints refuse is not a reading of this operation.  The
    # model coordinates two acts in one phrase -- "place soup and add utensil"
    # -- and the action word the phrase offers belongs to the other act, so
    # insisting on it discarded an operation whose participants identify exactly
    # one capability.  The hint is only consulted once the text's own reading has
    # been found unusable here, and only if it is itself endpoint-valid, so text
    # evidence still wins wherever the endpoints can carry it.
    if not intersection and capability_hint:
        hinted = {
            capability for capability in endpoint_matches
            if capability.capability_id == capability_hint
        }
        if len(hinted) == 1:
            intersection = hinted

    if not intersection:
        return OperationInterpretationResult(
            raw_operation=raw_phrase,
            canonical_source=source_role,
            canonical_target=target_role,
            canonical_anchor=anchor_role,
            capability=None,
            status="UNMAPPABLE_OPERATION",
            reason=(
                f"Operation semantic evidence {sorted(c.capability_id for c in semantic_matches)} "
                f"is incompatible with endpoints ({source_role}, {target_role}, {anchor_role})"
            ),
        )

    if len(intersection) > 1:
        # Check if one is an exact match for capability_id or planner_operation
        exact = [c for c in intersection if norm_text in (_phrase(c.capability_id), _phrase(c.planner_operation))]
        if len(exact) == 1:
            selected = exact[0]
        else:
            return OperationInterpretationResult(
                raw_operation=raw_phrase,
                canonical_source=source_role,
                canonical_target=target_role,
                canonical_anchor=anchor_role,
                capability=None,
                status="AMBIGUOUS_CAPABILITY",
                reason=f"Operation phrase {raw_phrase!r} matches multiple valid capabilities: {sorted(c.capability_id for c in intersection)}",
            )
    else:
        selected = next(iter(intersection))

    # 4. Instantiate physical preconditions
    preconditions: list[tuple[str, str, str]] = []
    req_rels: list[str] = []
    ctx_rels: list[str] = []

    for subj_key, pred, obj_key in selected.required_relation_templates:
        s_val = source_role if subj_key == "source" else (target_role if subj_key == "target" else anchor_role)
        o_val = target_role if obj_key == "target" else (anchor_role if obj_key == "anchor" else source_role)

        if s_val is None or o_val is None:
            continue

        preconditions.append((s_val, pred, o_val))
        if subj_key == "source" and obj_key == "target":
            if pred not in req_rels:
                req_rels.append(pred)
        elif obj_key == "anchor" or subj_key == "anchor":
            if pred not in ctx_rels:
                ctx_rels.append(pred)
        else:
            if pred not in req_rels:
                req_rels.append(pred)

    status = "EXACT_CAPABILITY_MATCH" if norm_text in (_phrase(selected.capability_id), _phrase(selected.planner_operation)) else "LEXICAL_SEMANTIC_MATCH"

    return OperationInterpretationResult(
        raw_operation=raw_phrase,
        canonical_source=source_role,
        canonical_target=target_role,
        canonical_anchor=anchor_role,
        capability=selected,
        status=status,
        reason=f"Successfully mapped to capability {selected.capability_id}",
        physical_preconditions=tuple(preconditions),
        planner_operation=selected.planner_operation,
        required_relations=tuple(req_rels),
        context_relations=tuple(ctx_rels),
    )


# ---------------------------------------------------------------------------
# Operation-induced abstract role requirements
# ---------------------------------------------------------------------------
# When the FM explicitly expresses a physical operation but enumerates its
# participants imperfectly, the operation's own semantics still tell the runtime
# which functional participants must exist.  A fastening needs an instrument, a
# joining component and a fixed receiving target whether or not the FM named all
# three.  Those requirements may be represented existentially -- as typed slots
# to be resolved later against the observed scene -- but never bound here to a
# concrete object, instance or scene category.  Search and grounding resolve
# them; this module only states that they must exist.
#
# This is gated on the FM having expressed the operation.  The runtime never
# invents an operation in order to invent its participants.

ABSTRACT_ROLE_PROVENANCE: Final[str] = "OPERATION_INDUCED_ABSTRACT_ROLE_REQUIREMENT"


@dataclass(frozen=True)
class AbstractRoleRequirement:
    """An existential functional participant implied by an expressed operation."""

    slot: str
    semantic_family: str
    candidate_canonical_roles: tuple[str, ...]
    capability_id: str
    provenance: str = ABSTRACT_ROLE_PROVENANCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "semantic_family": self.semantic_family,
            "candidate_canonical_roles": list(self.candidate_canonical_roles),
            "capability_id": self.capability_id,
            "provenance": self.provenance,
        }


def _dominant_role_family(domain: str, roles: tuple[str, ...]) -> str:
    """Most common declared family among a capability slot's admissible roles."""
    from .semantic_typing import canonical_role_family
    counts: dict[str, int] = {}
    for role in roles:
        family = canonical_role_family(domain, role)
        if family != "OTHER":
            counts[family] = counts.get(family, 0) + 1
    if not counts:
        return "UNSPECIFIED"
    return max(sorted(counts), key=lambda family: counts[family])


def abstract_slots_for_capability(capability: RobotCapability) -> tuple[AbstractRoleRequirement, ...]:
    """The existential participants a capability requires, independent of any binding."""
    slots: list[AbstractRoleRequirement] = []
    for slot, allowed in (
        ("source", capability.allowed_source_roles),
        ("target", capability.allowed_target_roles),
        ("anchor", capability.allowed_anchor_roles),
    ):
        if not allowed:
            continue
        slots.append(AbstractRoleRequirement(
            slot=slot,
            semantic_family=_dominant_role_family(capability.domain, allowed),
            candidate_canonical_roles=tuple(allowed),
            capability_id=capability.capability_id,
        ))
    return tuple(slots)


def abstract_slots_for_operation(
    domain: str,
    raw_phrase: str,
) -> tuple[AbstractRoleRequirement, ...]:
    """Existential participants implied by an operation the FM actually expressed.

    Returns an empty tuple when the phrase supports no capability, so a
    non-physical directive or an unrecognised operation induces nothing.  When
    several capabilities match, only requirements common to all of them are
    returned, since anything else would presuppose a choice not yet justified.
    """
    capabilities = extract_operation_semantic_candidates(domain, raw_phrase)
    if not capabilities:
        return ()
    per_capability = [abstract_slots_for_capability(cap) for cap in capabilities]
    if len(per_capability) == 1:
        return per_capability[0]
    shared_slots = set.intersection(*({s.slot for s in slots} for slots in per_capability))
    merged: list[AbstractRoleRequirement] = []
    for slot in ("source", "target", "anchor"):
        if slot not in shared_slots:
            continue
        entries = [s for slots in per_capability for s in slots if s.slot == slot]
        families = {s.semantic_family for s in entries}
        merged.append(AbstractRoleRequirement(
            slot=slot,
            semantic_family=next(iter(families)) if len(families) == 1 else "UNSPECIFIED",
            candidate_canonical_roles=tuple(dict.fromkeys(
                role for s in entries for role in s.candidate_canonical_roles)),
            capability_id="|".join(sorted({s.capability_id for s in entries})),
        ))
    return tuple(merged)


def effect_achieved_by_compiled_operation(
    domain: str,
    predicate: str,
    canonical_subject: str,
    canonical_object: str,
    groups: Sequence[Any],
) -> str | None:
    """The compiled operation whose execution brings this end state about, if any.

    Matched against the *compiled* interpretation rather than the raw wire
    slots.  V3 participants are unordered and the group they end up in has
    since been through id normalization, joint role typing, slot completion,
    context elision and composite lowering, so the raw layout is not what the
    runtime is going to execute -- and testing against it made corroboration
    depend on how the model happened to order its participants.

    This recognizes direction only.  It never creates an operation from a
    stated end state: the group has to already exist because the FM expressed
    the operation independently.
    """
    for group in groups:
        capability = getattr(group, "capability_id", None) or ""
        slots = {
            "source": getattr(group, "tool_role", None),
            "target": getattr(group, "target_role", None),
            "anchor": getattr(group, "context_role", None),
        }
        for declared in get_robot_capabilities(domain):
            if declared.capability_id != capability:
                continue
            for achieved, subject_slot, object_slot in declared.achieved_effects:
                if achieved != predicate:
                    continue
                if (slots.get(subject_slot) == canonical_subject
                        and slots.get(object_slot) == canonical_object):
                    return str(getattr(group, "id", "")) or capability
    return None
