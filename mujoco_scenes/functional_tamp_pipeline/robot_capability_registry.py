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
    "locate", "locates", "locating", "located",
    "recognize", "recognizes", "recognizing", "recognized",
    "recognise", "recognises", "recognising", "recognised",
    "search", "searches", "searching", "searched",
    "select", "selects", "selecting", "selected",
})


def is_non_physical_operation_phrase(raw_phrase: str) -> bool:
    """Return whether the phrase leads with an explicitly non-physical action.

    Only the leading action token is considered.  This deliberately permits
    physical operations containing later adjectival forms, such as
    ``place selected component``.
    """
    normalized = _phrase(raw_phrase)
    if not normalized:
        return False
    return normalized.split(maxsplit=1)[0] in _NON_PHYSICAL_LEADING_ACTIONS


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
            semantic_cues=(
                "soup", "soup serving", "eating utensil", "provide utensil",
                "place utensil", "serve soup", "spoon soup", "eat soup",
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
            allowed_anchor_roles=("SEATING_POSITION", "SEATING_PAIR"),
            required_relation_templates=(
                ("source", "FITS_SET_ON", "target"),
                ("source", "NEAR_SEAT", "anchor"),
            ),
            planner_operation="SUPPORT_DRINKWARE",
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
            allowed_anchor_roles=("SEATING_PAIR", "SEATING_POSITION"),
            required_relation_templates=(
                ("source", "FITS_ON", "target"),
                ("source", "ACCESSIBLE_FROM_BOTH_SEATS", "anchor"),
            ),
            planner_operation="SUPPORT_ENTERTAINMENT_CONTROL",
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
            allowed_anchor_roles=("repair_target", "MAIN_WORKBENCH_ZONE"),
            required_relation_templates=(
                ("source", "COMPATIBLE_WITH", "target"),
                ("source", "REACHES_TARGET", "anchor"),
                ("target", "COMPATIBLE_WITH_TARGET", "anchor"),
            ),
            planner_operation="DRIVE_FASTENER_INTO_TARGET",
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


def interpret_operation(
    domain: str,
    raw_phrase: str,
    source_role: str,
    target_role: str,
    anchor_role: str | None = None,
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
    semantic_matches: set[RobotCapability] = set()
    for cap in capabilities:
        cap_id_norm = _phrase(cap.capability_id)
        plan_op_norm = _phrase(cap.planner_operation)
        if norm_text == cap_id_norm or norm_text == plan_op_norm:
            semantic_matches.add(cap)
            continue
        for cue in cap.semantic_cues:
            c_norm = _phrase(cue)
            if c_norm == norm_text or (len(c_norm) >= 4 and c_norm in norm_text):
                semantic_matches.add(cap)
                break

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
