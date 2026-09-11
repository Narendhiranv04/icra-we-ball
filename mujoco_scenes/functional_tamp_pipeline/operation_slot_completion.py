"""Complete the capability slots an expressed operation needs but the FM did not name.

The FM is asked for the task's meaning, not for the runtime's call signatures.
It routinely expresses an operation while naming only some of the participants
that operation structurally involves: a fastening without saying what receives
the fastener, a personal refreshment placement without naming the drinkware.

When the FM has expressed such an operation, the operation's own capability
signature says which functional participants must exist.  This module states
those requirements existentially -- as typed slots carrying a canonical role
name and a cardinality -- and never binds a concrete object, instance, scene
category or observed identity.  Search and grounding resolve them.

Two kinds of missing slot are treated differently, and the difference is the
whole point:

SELECTABLE functional assets are things the robot must find and choose.  A
missing one becomes an ordinary functional role, goes to search, and fails as an
object-discovery or assignment problem if the scene cannot supply it.

SYSTEM_FIXED_FUNCTIONAL_ANCHORS are calibrated references the scene already
owns.  Supplying one asserts something, so each anchor declares why it may be
supplied at all.  A fastening has no meaning without something to fasten into,
so its receiving target is entailed by the operation itself.  A placement, by
contrast, is a placement whether or not it must end up near a seat: that
spatial requirement is task content, so the anchor is supplied only when the FM
expressed the corresponding semantic somewhere in its own graph.

Causality runs one way only.  An expressed operation induces participants; an
available participant never induces an operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, Sequence

from .predicate_registry import get_predicate_signature
from .robot_capability_registry import (
    ABSTRACT_ROLE_PROVENANCE,
    RobotCapability,
    extract_operation_semantic_candidates,
)
from .role_semantic_ontology import role_may_be_reused_across_applications
from .semantic_typing import canonical_role_family, role_text_scopes
from .system_context_registry import (
    get_domain_selectable_roles,
    get_domain_system_fixed_anchors,
)


SLOT_ORDER: tuple[str, ...] = ("source", "target", "anchor")

STRUCTURALLY_ENTAILED = "STRUCTURALLY_ENTAILED_BY_CAPABILITY"
REQUIRES_EXPRESSED_SEMANTIC = "REQUIRES_FM_EXPRESSED_SEMANTIC"


@dataclass(frozen=True)
class FixedAnchorSemantics:
    """Why a calibrated scene reference may stand in for an expressed semantic."""

    domain: str
    canonical_role: str
    family: str
    quantification: str  # INDIVIDUAL or SET
    justification: str
    semantic_cue: str = ""

    def cue_matches(self, text: str) -> bool:
        return bool(self.semantic_cue) and bool(re.search(self.semantic_cue, text, re.I))


# Individual-proximity language: a support belonging to, or set beside, one person.
_INDIVIDUAL_SEAT_CUE = (
    r"\b(personal|individual|own|per person|per[- ]?person|for (?:a|one|each|every) person|"
    r"each person|each of the (?:two )?(?:people|persons|viewers|occupants)|"
    r"near ?by|nearby|beside|adjacent|next to|alongside|"
    r"within reach|close to|convenient to|"
    r"near (?:the |a |their |its )?(?:seat|chair|sofa|couch|armchair|person|viewer|occupant|seating))\b"
)

# Shared-access language: one thing deliberately reachable from both places.
_SHARED_SEAT_CUE = (
    r"\b(accessible (?:to|from|by) (?:both|all|each|every|either)|"
    r"reachable (?:by|from) (?:both|all|each|either)|"
    r"within reach of (?:both|all|each|either)|"
    r"(?:shared|common|communal|mutual) (?:access|support|surface|spot|location|between)|"
    r"between (?:both|the two|two)|"
    r"for (?:both|two) (?:people|persons|viewers|occupants)|"
    r"(?:both|two|several|multiple|all|each of the two) (?:seats|seating positions|chairs)|"
    r"pair of seats|seating pair|accessible to both|"
    r"shared|both people|both viewers|either seat)\b"
)


# Seating described as serving more than one person.
_COLLECTIVE_OCCUPANCY = re.compile(
    r"\b(people|persons|viewers|occupants|guests|users|both|two|pair|each|every|all)\b"
)

# Access or reach asserted about something, without saying toward how many places.
_ACCESS_CUE = re.compile(
    r"\b(accessib\w*|reachab\w*|within reach|reach(?:able|ed|es)? (?:by|from)|"
    r"gettable|usable from)\b"
)


FIXED_ANCHOR_SEMANTICS: tuple[FixedAnchorSemantics, ...] = (
    FixedAnchorSemantics(
        domain="workshop", canonical_role="repair_target", family="FIXED_TARGET",
        quantification="INDIVIDUAL", justification=STRUCTURALLY_ENTAILED,
    ),
    FixedAnchorSemantics(
        domain="living_room", canonical_role="SEATING_POSITION", family="SEATING",
        quantification="INDIVIDUAL", justification=REQUIRES_EXPRESSED_SEMANTIC,
        semantic_cue=_INDIVIDUAL_SEAT_CUE,
    ),
    FixedAnchorSemantics(
        domain="living_room", canonical_role="SEATING_PAIR", family="SEATING",
        quantification="SET", justification=REQUIRES_EXPRESSED_SEMANTIC,
        semantic_cue=_SHARED_SEAT_CUE,
    ),
)


def anchor_semantics(domain: str, canonical_role: str) -> FixedAnchorSemantics | None:
    for item in FIXED_ANCHOR_SEMANTICS:
        if item.domain == domain and item.canonical_role == canonical_role:
            return item
    return None


def capability_anchor_roles(domain: str, capability: RobotCapability) -> tuple[str, ...]:
    """Anchors the capability's own physical preconditions can actually verify.

    A capability that needs its anchor checked by a predicate accepting only one
    role form does not really admit the others, however permissively its slot
    list is written.  Reading the answer off the predicate registry keeps the two
    from drifting apart.
    """
    admissible = []
    for role in capability.allowed_anchor_roles:
        ok = True
        for subject_key, predicate, object_key in capability.required_relation_templates:
            if "anchor" not in (subject_key, object_key):
                continue
            signature = get_predicate_signature(domain, predicate)
            if signature is None:
                ok = False
                break
            if subject_key == "anchor" and signature.allowed_subject_roles and role not in signature.allowed_subject_roles:
                ok = False
                break
            if object_key == "anchor" and signature.allowed_object_roles and role not in signature.allowed_object_roles:
                ok = False
                break
        if ok:
            admissible.append(role)
    return tuple(admissible)


def capability_slots(domain: str, capability: RobotCapability) -> dict[str, tuple[str, ...]]:
    """The slots a capability requires, with the canonical roles each admits."""
    slots = {
        "source": tuple(capability.allowed_source_roles),
        "target": tuple(capability.allowed_target_roles),
    }
    anchors = capability_anchor_roles(domain, capability)
    if anchors:
        slots["anchor"] = anchors
    return {slot: roles for slot, roles in slots.items() if roles}


# ---------------------------------------------------------------------------
# Whole-graph semantic evidence
# ---------------------------------------------------------------------------


@dataclass
class ExpressedSemantics:
    """What the FM said about spatial task context, and what each phrase is about.

    Evidence is gathered across the whole graph, because the model may state a
    setting's proximity in a role's function, in a relation, or in the operation
    phrase itself.  But a phrase is evidence *for* an operation only when it is
    connected to it: a shared-access requirement written about one placement
    must not license a seating reference for an unrelated one.  So every phrase
    carries the roles it is about, and an anchor completion asks only for the
    phrases connected to the operation it is completing.

    The model's own restatement of the task is the exception.  It is about the
    task rather than about any one participant, so it counts when the anchor it
    licenses is admissible for only one of the operations the model expressed --
    which is the "connection is unique" condition, read off the capabilities
    rather than guessed at.
    """

    domain: str
    # (origin, normalized text, the role ids the phrase is about)
    texts: tuple[tuple[str, str, frozenset[str]], ...] = ()
    seating_roles: tuple[dict[str, Any], ...] = ()   # the FM's seating-typed roles
    # anchor canonical role -> ids of the expressed operations whose capability admits it
    anchor_admitting_operations: dict[str, frozenset[str]] = field(default_factory=dict)
    # the model's own restatement of the task, and what each operation is about
    task_summary: str = ""
    operation_participant_words: dict[str, frozenset[str]] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def _summary_names_only(self, operation_id: str) -> bool:
        """Whether the summary names this operation's participants and no other's.

        The connection has to be unique for a task-level phrase to be evidence
        about one operation, and there are two ways for it to be: the anchor is
        admissible for only this operation, or the summary's own words pick out
        this operation's participants and nobody else's.  "Position the remote
        control for shared access" names the control, so it is about the
        placement the control takes part in even where both operations are
        worded alike.
        """
        words = _content_words(self.task_summary)
        if not words:
            return False
        named = {
            other for other, participant_words in self.operation_participant_words.items()
            if words & participant_words
        }
        return named == {operation_id}

    def _in_scope(self, origin: str, about: frozenset[str], semantics: FixedAnchorSemantics,
                  operation_id: str, participants: frozenset[str]) -> bool:
        if origin.startswith("OPERATION:"):
            # An operation's phrase describes that operation and no other.  It
            # used to reach any operation sharing one of its participants, and
            # two placements naturally share the seating they are arranged
            # around: "place the control where it is accessible to both people"
            # then licensed a two-seat anchor for the *drinkware* placement, a
            # second capability completed for it on that strength, and the tie
            # made slot completion refuse an operation it could otherwise seat.
            # The origin already says which operation the phrase belongs to.
            return origin == f"OPERATION:{operation_id}"
        if about & participants:
            return True
        if origin == "TASK_SUMMARY":
            admitting = self.anchor_admitting_operations.get(semantics.canonical_role, frozenset())
            if admitting == frozenset({operation_id}):
                return True
            return self._summary_names_only(operation_id)
        return False

    @property
    def _denotes_several_seats(self) -> bool:
        """Whether the FM's own seating references stand for more than one seat.

        Three ways the model can say so: two seat roles, one role declared with
        two instances, or one shared role described collectively -- "the area
        where people sit".  All three are the model enumerating more than one
        seat; none of them is the runtime assuming a second person.
        """
        if len(self.seating_roles) >= 2:
            return True
        return any(
            role["count"] >= 2
            or (role["binding_policy"] in {"SHARED", "REUSABLE"}
                and _COLLECTIVE_OCCUPANCY.search(role["text"]))
            for role in self.seating_roles
        )

    def witnesses(
        self,
        semantics: FixedAnchorSemantics,
        *,
        operation_id: str = "",
        participants: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        scope = frozenset(participants)

        def connected(origin: str, about: frozenset[str]) -> bool:
            return self._in_scope(origin, about, semantics, operation_id, scope)

        found = [
            {"origin": origin, "text": text}
            for origin, text, about in self.texts
            if semantics.cue_matches(text) and connected(origin, about)
        ]
        if semantics.quantification != "SET":
            return found
        if self._denotes_several_seats:
            # Accessibility asserted toward a seating reference that stands for
            # several seats *is* the shared-access requirement, however the model
            # worded it.  Insisting on the word "both" would demand our own
            # phrasing of a semantic the model already gave.
            found.extend(
                {"origin": origin, "text": text}
                for origin, text, about in self.texts
                if _ACCESS_CUE.search(text) and connected(origin, about)
            )
            if found:
                # Cardinality says the pair exists; it never says on its own that
                # anything must be reachable from both of its seats.
                found.append({
                    "origin": "FM_ROLE_CARDINALITY",
                    "text": "the FM's seating reference stands for two or more seats",
                })
        return found


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", str(value)).lower()).strip()


# Words too common to establish that two phrases are about the same thing.
_UNDISTINGUISHING_WORDS = frozenset({
    "the", "a", "an", "and", "or", "for", "with", "to", "of", "on", "in", "at",
    "each", "every", "both", "all", "two", "one", "their", "its", "it", "is",
    "be", "must", "should", "place", "places", "put", "set", "area", "areas",
    "position", "positions", "item", "items", "thing", "things", "surface",
    "surfaces", "people", "person", "task", "room", "shared", "access",
})


def _content_words(text: Any) -> frozenset[str]:
    """Words distinctive enough to say two phrases are about the same participant."""
    return frozenset(
        word for word in re.findall(r"[a-z]+", _normalized(text))
        if len(word) > 3 and word not in _UNDISTINGUISHING_WORDS
    )


def collect_expressed_semantics(
    domain: str,
    contract: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
) -> ExpressedSemantics:
    """Gather every FM-authored phrase and seating cardinality in the contract.

    Evidence is looked for across the whole graph rather than in one binary
    relation: the model may state a personal setting's proximity in a role's
    function, in a relation, or in the operation phrase itself, and any of those
    is the model saying it.
    """
    texts: list[tuple[str, str, frozenset[str]]] = []
    seating: list[dict[str, Any]] = []
    roles_by_id = {str(role.get("id")): dict(role) for role in contract.get("functional_roles", ()) or ()}
    for role in contract.get("functional_roles", ()) or ():
        scopes = role_text_scopes(dict(role))
        about = frozenset({str(role.get("id"))})
        for scope, text in scopes.items():
            if text:
                texts.append((f"ROLE:{role.get('id')}:{scope}", text, about))
        hypothesis = hypotheses.get(str(role.get("id")))
        candidates = set(getattr(hypothesis, "canonical_role_candidates", ()) or ())
        if candidates and all(
            canonical_role_family(domain, candidate) == "SEATING" for candidate in candidates
        ):
            try:
                count = int(role.get("required_count", 1))
            except (TypeError, ValueError):
                count = 1
            seating.append({
                "role_id": str(role.get("id")),
                "count": count,
                "binding_policy": str(role.get("binding_policy", "SHARED")),
                "text": " ".join(value for value in scopes.values() if value),
            })
    summary = _normalized(contract.get("task_summary", ""))
    if summary:
        # The model's own restatement of the task is its wording too, and it is
        # often where it says a thing must be reachable from both seats.  It is
        # about the task rather than any one participant, so it carries no roles
        # and is admitted only under the uniqueness condition above.
        texts.append(("TASK_SUMMARY", summary, frozenset()))
    for relation in contract.get("functional_relations", ()) or ():
        texts.append((
            f"RELATION:{relation.get('id')}",
            _normalized(relation.get("relation", "")),
            frozenset(str(p) for p in relation.get("participant_roles", ()) or ()),
        ))
    operations = list(contract.get("operation_pairings", ()) or contract.get("interaction_groups", ()) or ())
    for operation in operations:
        texts.append((
            f"OPERATION:{operation.get('id')}",
            _normalized(operation.get("operation") or operation.get("function") or ""),
            frozenset(str(p) for p in operation.get("participant_roles", ()) or ()),
        ))
    for row in contract.get("explicit_context_sets", ()) or ():
        texts.append((
            f"CONTEXT_SET:{row.get('source_id')}",
            _normalized(row.get("runtime_role", "")),
            frozenset({str(row.get("source_id"))}),
        ))
    # Which expressed operations could take each anchor at all.  This is what
    # makes a task-level statement's connection unique or ambiguous.
    admitting: dict[str, set[str]] = {}
    for operation in operations:
        phrase = str(operation.get("operation") or operation.get("function") or "")
        participants = [str(p) for p in operation.get("participant_roles", ()) or ()]
        for capability in extract_operation_semantic_candidates(domain, phrase, participants):
            for anchor in capability_anchor_roles(domain, capability):
                admitting.setdefault(anchor, set()).add(str(operation.get("id")))
    return ExpressedSemantics(
        domain=domain, texts=tuple(texts), seating_roles=tuple(seating),
        anchor_admitting_operations={
            anchor: frozenset(ids) for anchor, ids in admitting.items()
        },
        task_summary=summary,
        operation_participant_words={
            str(operation.get("id")): frozenset().union(*[
                _content_words(f"{participant} "
                               f"{(roles_by_id.get(str(participant)) or {}).get('function', '')}")
                for participant in operation.get("participant_roles", ()) or ()
            ] or [frozenset()])
            for operation in operations
        },
    )


# ---------------------------------------------------------------------------
# Slot completion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletedOperation:
    """One capability reading whose unnamed slots the runtime supplied."""

    capability_id: str
    options: tuple[dict[str, Any], ...]
    synthesized_roles: tuple[dict[str, Any], ...]
    trace: dict[str, Any]


def _slot_role_kind(domain: str, canonical_role: str) -> str:
    if canonical_role in set(get_domain_system_fixed_anchors(domain)):
        return "FIXED_TARGET"
    if canonical_role in {"PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION", "MAIN_WORKBENCH_ZONE"}:
        return "REGION"
    return "OBJECT"


def _slot_cardinality(
    domain: str, slot: str, canonical_role: str, operation_count: int,
) -> tuple[int, str]:
    """How many instances an induced slot needs, and whether they must differ.

    How many times the task applies the operation is ``operation_count``.  How
    many separate physical things that needs is a different question, and its
    answer belongs to the role, not to the slot: the runtime declares, per
    canonical role, whether one instance may serve several applications.  A
    material source or an implement may; a serving container, a personal
    support, or the seat that makes a support personal may not.

    Deciding this from the slot name alone was wrong in both directions.  Every
    anchor was made a single shared reference, so two personal placements were
    anchored to the same seat and the second one's proximity requirement had
    nothing left to check.  Every repeated non-anchor slot was made distinct, so
    one coffee source poured twice was recorded as a demand for two jars.
    """
    reusable = role_may_be_reused_across_applications(domain, canonical_role)
    if slot == "anchor":
        # A reference standing for one thing per application must be supplied
        # once per application; one standing for all of them is shared.
        if not reusable and operation_count > 1:
            return operation_count, "DISTINCT"
        return 1, "SHARED"
    if reusable:
        # Needed for every application, satisfiable by a single instance.
        return max(1, operation_count), "REUSABLE"
    if operation_count > 1:
        return operation_count, "DISTINCT"
    return 1, "REUSABLE"


def _synthesized_role(
    domain: str,
    operation: Mapping[str, Any],
    capability: RobotCapability,
    slot: str,
    canonical_role: str,
    operation_count: int,
) -> dict[str, Any]:
    count, policy = _slot_cardinality(domain, slot, canonical_role, operation_count)
    family = canonical_role_family(domain, canonical_role)
    return {
        "id": f"op_slot__{operation.get('id')}__{slot}",
        "entity_kind": _slot_role_kind(domain, canonical_role),
        "function": (
            f"{family.lower().replace('_', ' ')} participant the expressed "
            f"{capability.capability_id} operation requires"
        ),
        "description": (
            "Existential functional requirement induced by an operation the FM "
            "expressed; no concrete object, instance or scene category is bound here."
        ),
        "required_count": count,
        "binding_policy": policy,
        "candidate_categories": [],
        "required_properties": [],
        "visible_candidates": [],
        "canonical_role": canonical_role,
        "provenance": ABSTRACT_ROLE_PROVENANCE,
        "induced_by_operation": operation.get("id"),
        "induced_capability": capability.capability_id,
        "induced_slot": slot,
    }


@dataclass(frozen=True)
class SlotFit:
    """How a named participant can occupy one capability slot."""

    canonical_role: str
    how: str
    # True when the FM named a reference of the right kind but in a different
    # quantified form, so the slot is held by the registered anchor rather than
    # by the raw role: an individual seat where the verifier needs the pair.
    realized_by_system_anchor: bool = False
    witnesses: tuple[dict[str, Any], ...] = ()


def _own_wording_preference(hypotheses: Mapping[str, Any], participant: str) -> tuple[str, ...]:
    """The role, if any, the participant's own description singled out.

    A role the model described specifically enough to name -- a remote control,
    a coffee source -- is not available to be re-read as a different member of
    its family.  Only a generically worded participant, whose current type came
    from the rest of the graph rather than from its own description, may be.
    """
    for row in getattr(hypotheses.get(participant), "evidence", ()) or ():
        if isinstance(row, dict) and "preferred_roles" in row:
            return tuple(row.get("preferred_roles") or ())
    return ()


def _participant_slot_fit(
    domain: str,
    participant: str,
    slot: str,
    allowed: Sequence[str],
    hypotheses: Mapping[str, Any],
    evidence: ExpressedSemantics,
    operation_id: str = "",
    scope_participants: Sequence[str] = (),
) -> SlotFit | None:
    """How, if at all, a named participant can occupy this slot.

    A participant whose own type is admissible fills the slot directly.  A
    participant that names the same *kind* of fixed reference in a different
    quantified form realizes the registered anchor instead, but only when the FM
    expressed the quantification that anchor stands for.
    """
    candidates = set(getattr(hypotheses.get(participant), "canonical_role_candidates", ()) or ())
    direct = sorted(candidates & set(allowed))
    if len(direct) == 1:
        return SlotFit(direct[0], "FM_PARTICIPANT_TYPE")
    if direct:
        return None
    families = {canonical_role_family(domain, candidate) for candidate in candidates}
    named_its_own_role = len(_own_wording_preference(hypotheses, participant)) == 1
    if slot == "source":
        # The model named a participant of the right kind for this slot but not
        # the specific form the capability needs -- one "surface" role standing
        # for both the personal support and the shared one, which the prompt
        # asks it to keep apart.  Each operation says which form it needs, so
        # the slot is held by that role and the raw participant is recorded as
        # the witness.  Only admissible if the kind picks out exactly one form:
        # a kitchen transfer admits two sources and stays ambiguous.
        #
        # The target slot is deliberately excluded.  This substitutes a sibling
        # of the same family for what the model named, which is a reasonable way
        # to settle which of several receiving places an operation meant, and not
        # a reasonable way to decide what the operation acts on.  Allowing it
        # there read a drinkware set as the remote control, on the strength of
        # both being things one carries: two capabilities could then seat the
        # placement, slot completion refused the tie, and the operation the task
        # is mostly about was dropped as unrepresentable.
        bindable = set(get_domain_selectable_roles(domain)) | set(get_domain_system_fixed_anchors(domain))
        same_kind = [
            role for role in allowed
            if role in bindable
            and canonical_role_family(domain, role) in families
            and canonical_role_family(domain, role) != "OTHER"
        ]
        if len(same_kind) == 1 and not named_its_own_role:
            return SlotFit(
                same_kind[0], "FM_PARTICIPANT_KIND_REALIZED_BY_CAPABILITY_SLOT",
                realized_by_system_anchor=True,
                witnesses=({"origin": f"ROLE:{participant}",
                            "text": "a participant of this slot's kind in a different form"},),
            )
        return None
    for role in allowed:
        semantics = anchor_semantics(domain, role)
        if semantics is None or semantics.family not in families:
            continue
        if semantics.justification == STRUCTURALLY_ENTAILED:
            return SlotFit(role, "SYSTEM_ANCHOR_STRUCTURALLY_ENTAILED_BY_CAPABILITY",
                           realized_by_system_anchor=True)
        witnesses = evidence.witnesses(
            semantics, operation_id=operation_id, participants=scope_participants)
        if witnesses:
            return SlotFit(
                role, "FM_EXPRESSED_QUANTIFIED_SEMANTIC_REALIZED_BY_SYSTEM_ANCHOR",
                realized_by_system_anchor=True,
                witnesses=tuple(witnesses + [{
                    "origin": f"ROLE:{participant}",
                    "text": "a seating reference of the right kind in a different quantified form",
                }]),
            )
    return None


def _complete_slot(
    domain: str,
    slot: str,
    allowed: Sequence[str],
    evidence: ExpressedSemantics,
    operation_id: str = "",
    scope_participants: Sequence[str] = (),
) -> tuple[str, str, list[dict[str, Any]]] | None:
    """Supply an unnamed slot, or refuse to."""
    selectable = [role for role in allowed if role in set(get_domain_selectable_roles(domain))]
    fixed = [role for role in allowed if role in set(get_domain_system_fixed_anchors(domain))]
    if len(selectable) == 1 and not fixed:
        return selectable[0], "SELECTABLE_FUNCTIONAL_ASSET_SENT_TO_GROUNDING", []
    if not selectable:
        supplied: list[tuple[str, list[dict[str, Any]]]] = []
        for role in fixed:
            semantics = anchor_semantics(domain, role)
            if semantics is None:
                continue
            if semantics.justification == STRUCTURALLY_ENTAILED:
                supplied.append((role, [{"origin": "CAPABILITY_STRUCTURE",
                                         "text": semantics.justification}]))
                continue
            witnesses = evidence.witnesses(
                semantics, operation_id=operation_id, participants=scope_participants)
            if witnesses:
                supplied.append((role, witnesses))
        if len(supplied) == 1:
            role, witnesses = supplied[0]
            return role, "SYSTEM_FIXED_FUNCTIONAL_ANCHOR_FOR_EXPRESSED_SEMANTIC", witnesses
    return None


def probe_named_participant_slots(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Seat only the participants the FM named, leaving the rest of the slots open.

    Used to ask which capability an under-specified operation could be, without
    introducing anything.  Nothing is synthesized and no anchor is supplied, so a
    caller can look for the missing context in the FM's own relations first and
    keep that better provenance when it finds it.
    """
    phrase = str(operation.get("operation") or operation.get("function") or "")
    participants = list(dict.fromkeys(operation.get("participant_roles", ()) or ()))
    if any(p not in hypotheses or p not in roles_by_id for p in participants):
        return []
    options: list[dict[str, Any]] = []
    for capability in extract_operation_semantic_candidates(domain, phrase, participants):
        slots = capability_slots(domain, capability)
        if len(participants) >= len(slots) or not participants:
            continue
        assignment: dict[str, str] = {}
        types: dict[str, str] = {}
        used: set[str] = set()
        for slot in SLOT_ORDER:
            if slot not in slots:
                continue
            fits = [
                (participant, sorted(
                    set(getattr(hypotheses.get(participant), "canonical_role_candidates", ()) or ())
                    & set(slots[slot])))
                for participant in participants if participant not in used
            ]
            viable = [(participant, kinds) for participant, kinds in fits if len(kinds) == 1]
            if len(viable) != 1:
                continue
            participant, kinds = viable[0]
            used.add(participant)
            assignment[slot] = participant
            types[slot] = kinds[0]
        if "source" not in assignment or "target" not in assignment or len(used) != len(participants):
            continue
        source_role = roles_by_id.get(assignment["source"], {})
        binding = str(source_role.get("binding_policy", "REUSABLE"))
        options.append({
            "capability_id": capability.capability_id,
            "planner_operation": capability.planner_operation,
            "source_role": assignment["source"],
            "target_role": assignment["target"],
            "anchor_role": assignment.get("anchor"),
            "source_type": types["source"],
            "target_type": types["target"],
            "anchor_type": types.get("anchor"),
            "usage_policy": (
                "SEQUENTIAL_REUSE_ALLOWED" if binding in {"REUSABLE", "SHARED"}
                else "DEDICATED_PER_TARGET"
            ),
            "capability_provenance": "NAMED_PARTICIPANT_PROBE",
        })
    return options


def _seat_named_participants_without_inventing(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
    capability: Any,
) -> dict[str, tuple[str, str, str]] | None:
    """Seat every named participant in this capability, inventing nothing.

    An operation the model stated over participants it already named should be
    read as being about those participants.  Where one capability can seat all
    of them and another can only do so by synthesizing a participant nobody
    named, the first is what the model said.

    "The fastening tool is placed on the workbench" names exactly two things and
    the tool return takes exactly two.  It was read as a *fastening* instead: a
    fastener was synthesized for the target slot, the workbench became the site
    driven into, and the invented participant then had no runtime role, so a
    contract that was complete failed on an operation the model expressed plainly.

    A participant whose resolved type does not fit the slot may still be seated
    when its own functional family picks out exactly one of the slot's roles --
    the model writes one "workbench" role for the site fastened into and the
    surface tools are left on, and its families say both.  That is the same
    reading the operation-scoped projection performs later; doing it here keeps
    the wrong capability from being chosen first and hiding the need for it.
    """
    participants = list(dict.fromkeys(operation.get("participant_roles", ()) or ()))
    slots = capability_slots(domain, capability)
    if not participants or len(participants) != len(slots):
        return None
    assignment: dict[str, tuple[str, str, str]] = {}
    used: set[str] = set()
    for slot in SLOT_ORDER:
        if slot not in slots:
            continue
        allowed = set(slots[slot])
        seated = None
        for participant in participants:
            if participant in used:
                continue
            candidates = set(getattr(
                hypotheses.get(participant), "canonical_role_candidates", ()) or ())
            direct = candidates & allowed
            if len(direct) == 1:
                seated = (participant, sorted(direct)[0], "FM_PARTICIPANT_TYPE")
                break
        if seated is None:
            # Only a participant whose own resolved type fits the slot.  Reading
            # one in a *different* form here was tried and reverted: it decides
            # the participant's canonical meaning earlier than the projection
            # does, and everything else the model wrote about it then follows the
            # new reading.  One workshop contract's "workbench" was re-read as
            # the surface tools are left on, and the required relation saying the
            # component must fit the fastening *site* lost its endpoint, costing
            # a variant that had been succeeding on all three trials.  Choosing a
            # form per operation is the projection's job, after the operations
            # are seated.
            return None
        used.add(seated[0])
        assignment[slot] = seated
    if len(used) != len(participants):
        return None
    return assignment


def complete_operation_slots(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
    evidence: ExpressedSemantics,
) -> CompletedOperation | None:
    """Seat an under-specified operation by supplying the slots the FM omitted.

    Refuses whenever more than one reading survives: choosing between them would
    invent semantics the model never gave.  Refuses too when a named participant
    cannot be seated at all, since that is over-specification rather than
    omission and is handled elsewhere.
    """
    phrase = str(operation.get("operation") or operation.get("function") or "")
    participants = list(dict.fromkeys(operation.get("participant_roles", ()) or ()))
    capabilities = extract_operation_semantic_candidates(domain, phrase, participants)
    if not capabilities:
        return None
    if not participants or any(
        p not in hypotheses or p not in roles_by_id for p in participants
    ):
        return None
    try:
        operation_count = int(operation.get("operation_count", 1) or 1)
    except (TypeError, ValueError):
        operation_count = 1

    # Prefer a reading that invents nothing.  See the helper above: where
    # exactly one nominated capability seats every participant the model named,
    # that is the operation it described, and no placeholder is created.
    uninvented = {}
    for capability in capabilities:
        seated = _seat_named_participants_without_inventing(
            domain, operation, roles_by_id, hypotheses, capability)
        if seated is not None:
            uninvented[capability.capability_id] = (capability, seated)
    if len(uninvented) == 1:
        capability, assignment = next(iter(uninvented.values()))
        source_role = roles_by_id.get(assignment["source"][0], {})
        binding = str(source_role.get("binding_policy", "REUSABLE"))
        option = {
            "capability_id": capability.capability_id,
            "planner_operation": capability.planner_operation,
            "source_role": assignment["source"][0],
            "target_role": assignment["target"][0],
            "anchor_role": assignment.get("anchor", (None,))[0],
            "source_type": assignment["source"][1],
            "target_type": assignment["target"][1],
            "anchor_type": assignment.get("anchor", (None, None))[1],
            "usage_policy": ("SEQUENTIAL_REUSE_ALLOWED"
                             if binding in {"REUSABLE", "SHARED"} else "DEDICATED_PER_TARGET"),
            "capability_provenance": "NAMED_PARTICIPANTS_SEATED_WITHOUT_INDUCTION",
        }
        return CompletedOperation(
            capability_id=capability.capability_id,
            options=(option,),
            synthesized_roles=(),
            trace={
                "code": "OPERATION_SEATED_ON_ITS_NAMED_PARTICIPANTS",
                "operation_id": operation.get("id"),
                "capability_id": capability.capability_id,
                "named_participants": participants,
                "slot_resolution": {
                    slot: {"role": value[0], "canonical_role": value[1], "how": value[2]}
                    for slot, value in sorted(assignment.items())
                },
                "induced_roles": [],
                "rejected_readings_that_would_have_invented_a_participant": sorted(
                    capability.capability_id for capability in capabilities
                    if capability.capability_id not in uninvented),
                "provenance": "FM_EXPLICIT_OPERATION",
            },
        )

    completions: list[CompletedOperation] = []
    for capability in capabilities:
        slots = capability_slots(domain, capability)
        # No count precondition: an operation may name as many participants as
        # there are slots and still leave one unseated, because the model wrote
        # a reference in a different quantified form -- an individual seat where
        # the verifier needs the pair.  Over-specification is still refused, by
        # the requirement below that every named participant be used.
        assignment: dict[str, tuple[str | None, str, str]] = {}
        used: set[str] = set()
        conflict = False
        synthesized: list[dict[str, Any]] = []
        witnesses: dict[str, list[dict[str, Any]]] = {}
        for slot in SLOT_ORDER:
            if slot not in slots:
                continue
            fits = [
                (participant, fit)
                for participant in participants
                if participant not in used
                and (fit := _participant_slot_fit(
                    domain, participant, slot, slots[slot], hypotheses, evidence,
                    operation_id=str(operation.get("id") or ""),
                    scope_participants=participants)) is not None
            ]
            if len(fits) > 1:
                conflict = True
                break
            if fits:
                participant, fit = fits[0]
                used.add(participant)
                if fit.realized_by_system_anchor:
                    # The FM named a reference of the right kind in the wrong
                    # quantified form.  Hold the slot with the registered anchor
                    # so downstream role identity stays consistent, and record
                    # the raw role as the witness that licensed it.
                    role = _synthesized_role(
                        domain, operation, capability, slot, fit.canonical_role, operation_count)
                    role["fm_witness_role"] = participant
                    synthesized.append(role)
                    assignment[slot] = (role["id"], fit.canonical_role, fit.how)
                    witnesses[slot] = list(fit.witnesses)
                else:
                    assignment[slot] = (participant, fit.canonical_role, fit.how)
        if conflict or len(used) != len(participants):
            continue
        for slot in SLOT_ORDER:
            if slot not in slots or slot in assignment:
                continue
            supplied = _complete_slot(
                domain, slot, slots[slot], evidence,
                operation_id=str(operation.get("id") or ""),
                scope_participants=participants)
            if supplied is None:
                break
            canonical_role, how, slot_witnesses = supplied
            role = _synthesized_role(
                domain, operation, capability, slot, canonical_role, operation_count)
            synthesized.append(role)
            assignment[slot] = (role["id"], canonical_role, how)
            witnesses[slot] = slot_witnesses
        if any(slot in slots and slot not in assignment for slot in SLOT_ORDER):
            continue
        source_participant = assignment["source"][0]
        source_role = roles_by_id.get(source_participant) or next(
            (item for item in synthesized if item["id"] == source_participant), {})
        binding = str(source_role.get("binding_policy", "REUSABLE"))
        if binding == "DISTINCT" and int(source_role.get("required_count", 1)) < operation_count:
            continue
        option = {
            "capability_id": capability.capability_id,
            "planner_operation": capability.planner_operation,
            "source_role": assignment["source"][0],
            "target_role": assignment["target"][0],
            "anchor_role": assignment.get("anchor", (None,))[0],
            "source_type": assignment["source"][1],
            "target_type": assignment["target"][1],
            "anchor_type": assignment.get("anchor", (None, None))[1],
            "usage_policy": (
                "SEQUENTIAL_REUSE_ALLOWED" if binding in {"REUSABLE", "SHARED"}
                else "DEDICATED_PER_TARGET"
            ),
            "capability_provenance": "OPERATION_INDUCED_SLOT_COMPLETION",
        }
        completions.append(CompletedOperation(
            capability_id=capability.capability_id,
            options=(option,),
            synthesized_roles=tuple(synthesized),
            trace={
                "code": "OPERATION_INDUCED_ABSTRACT_SLOT_COMPLETION",
                "operation_id": operation.get("id"),
                "capability_id": capability.capability_id,
                "named_participants": participants,
                "slot_resolution": {
                    slot: {"role": value[0], "canonical_role": value[1], "how": value[2]}
                    for slot, value in sorted(assignment.items())
                },
                "induced_roles": [item["id"] for item in synthesized],
                "fm_semantic_witnesses": witnesses,
                "provenance": ABSTRACT_ROLE_PROVENANCE,
            },
        ))
    if len(completions) != 1:
        return None
    return completions[0]
