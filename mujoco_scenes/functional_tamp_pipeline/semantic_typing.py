"""Pure FM semantic hypothesis construction, independent of compilation and G_O."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from .models import RoleTypeHypothesis
from .predicate_registry import get_predicate_signature
from .relation_interpreter import extract_relation_semantic_candidates
from .robot_capability_registry import extract_operation_semantic_candidates
from .system_context_registry import (
    get_domain_planner_context_constants,
    get_domain_selectable_roles,
    get_domain_system_fixed_anchors,
)


@dataclass(frozen=True)
class FunctionSemanticEvidence:
    preferred_roles: tuple[str, ...] = ()
    excluded_roles: tuple[str, ...] = ()
    explicit_families: tuple[str, ...] = ()
    evidence_strength: str = "UNKNOWN"
    provenance: str = "FM_FUNCTION_TEXT"
    evidence_source: str = "FUNCTION_TEXT"


def _text(role: dict[str, Any]) -> str:
    fields = (
        ("FUNCTION_TEXT", str(role.get("function", ""))),
        ("ROLE_ID_TEXT", str(role.get("id", ""))),
        ("CANDIDATE_CATEGORY_TEXT", " ".join(role.get("candidate_categories", []))),
        ("DESCRIPTION_TEXT", str(role.get("description", ""))),
    )
    signal = re.compile(
        r"\b(source|provider|ingredient|receiv|destination|tool|implement|instrument|utensil|driver|"
        r"table|surface|support|platform|seat|chair|component|fastener|target|fixture|refreshment|"
        r"drinkware|remote|control|television|display|cup|plate|saucer)\w*\b", re.I
    )
    for _, value in fields:
        normalized = re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value).lower()).strip()
        if signal.search(normalized):
            return normalized
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", " ".join(value for _, value in fields)).lower()).strip()


def _evidence_source(role: dict[str, Any], selected_text: str) -> str:
    for source, value in (
        ("FUNCTION_TEXT", str(role.get("function", ""))),
        ("ROLE_ID_TEXT", str(role.get("id", ""))),
        ("CANDIDATE_CATEGORY_TEXT", " ".join(role.get("candidate_categories", []))),
        ("DESCRIPTION_TEXT", str(role.get("description", ""))),
    ):
        normalized = re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value).lower()).strip()
        if normalized == selected_text:
            return source
    return "COMBINED_FALLBACK_TEXT"


def canonical_role_family(domain: str, role: str) -> str:
    families = {
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
    return families.get(domain, {}).get(role, "OTHER")


def function_semantic_evidence(
    domain: str,
    role: dict[str, Any],
    runtime_roles: set[str],
    weak_preference: str | None = None,
) -> FunctionSemanticEvidence:
    """Extract positive preferences and explicit family exclusions without scores."""
    text = _text(role)
    evidence_source = _evidence_source(role, text)
    families: set[str] = set()
    # Keep a legacy lexical hit available as a fallback, but do not mix it
    # with preferences derived from explicit causal/function language.  Mixing
    # the two made phrases such as "provide water for coffee" prefer both the
    # water source and the legacy coffee alias.
    preferred: set[str] = set()

    if re.search(r"\b(source|suppl(?:y|ies)|provider|ingredient|raw material|holding (?:dry|drinking|raw))\b", text):
        families.add("SOURCE")
    if re.search(r"\b(receiv(?:e|es|ing)|destination|prepared (?:coffee|soup)|served soup)\b", text):
        families.add("DESTINATION")
    if re.search(r"\b(tool|implement|instrument|utensil|driver|wrench|drill|applicator|drive|tighten)\b", text):
        families.add("INSTRUMENT")
    if re.search(r"\b(table|tabletop|surface|platform|support|placement area|central area|staging area|storage area|storage region)\b", text):
        families.add("SUPPORT")
    if re.search(r"\b(chair|armchair|seat|seating (?:fixture|position|context)|occupant support)\b", text):
        families.add("SEATING")
    if re.search(r"\b(screw|bolt|fastener|joining element|connecting element|connector|installed component|component to be installed)\b", text):
        families.add("COMPONENT")
    if re.search(r"\b(fixed (?:workpiece|point|target)|repair target|fixture|marked (?:target|joint|site|location)|target joint|joint hole|fastening site|location[^.]*requiring fastening|assembly receiv(?:ing|er)|workpiece assembly|object to be secured|primary object to be secured)\b", text):
        families.add("FIXED_TARGET")
    if re.search(r"\b(refreshment set|cup and saucer|drinkware|remote control|media control|entertainment control(?:ler)?|device (?:used to|for controlling|to control)|controlling (?:television|tv|display)|(?:remote|media|entertainment|television|tv) controller?|control(?:ler)? (?:television|tv|display)|consumables)\b", text):
        families.add("PAYLOAD")
    if re.search(r"\b(television|tv screen|display|monitor)\b", text):
        families.add("DISPLAY_CONTEXT")

    if domain == "kitchen":
        if "SOURCE" in families:
            if re.search(r"\b(?:provide|supply|source|holding)\s+(?:the\s+)?water\b|\bwater\s+source\b", text):
                preferred.add("water_source")
            elif re.search(r"\b(?:provide|supply|source|holding)\s+(?:the\s+)?coffee\b|\bcoffee\s+(?:ingredient\s+)?source\b", text):
                preferred.add("coffee_source")
            elif re.search(r"\bwater\b", text) and not re.search(r"\bcoffee\b", text):
                preferred.add("water_source")
            elif re.search(r"\bcoffee\b", text):
                preferred.add("coffee_source")
        if "DESTINATION" in families:
            if re.search(r"\bcoffee\b", text): preferred.add("coffee_container")
            if re.search(r"\bsoup\b", text): preferred.add("soup_container")
        if "INSTRUMENT" in families:
            if re.search(r"\b(stir|mix|agitat)\w*\b", text): preferred.add("coffee_stirrer")
            if re.search(r"\b(eat|consum|soup)\w*\b", text): preferred.add("soup_eating_utensil")
        if "SUPPORT" in families:
            if re.search(r"\b(serv(?:e|ing|ice)|delivery|handoff)\b", text):
                preferred.add("serving_area")
            elif re.search(r"\b(dining|meal)\b", text):
                preferred.add("dining_table")
            else:
                preferred.add("countertop")
    elif domain == "living_room":
        categories = " ".join(role.get("candidate_categories", [])).lower()
        description = str(role.get("description", "")).lower()
        if "SOURCE" in families and re.search(r"\b(cup|plate|saucer|drinkware)\b", categories):
            families.discard("SOURCE")
            families.add("PAYLOAD")
            evidence_source = "CANDIDATE_CATEGORY_TEXT"
        if "COMPONENT" in families and re.search(r"\b(electronic|remote|controller|device)\b", categories + " " + description) and re.search(r"\b(control|remote|television|tv)\b", description):
            families.discard("COMPONENT")
            families.add("PAYLOAD")
            evidence_source = "CANDIDATE_CATEGORY_TEXT"
        if "SUPPORT" in families:
            if re.search(r"\b(shared|central|common|coffee table|both|remote)\b", text):
                preferred.add("SHARED_REMOTE_REGION")
            else:
                preferred.add("PERSONAL_CUP_SAUCER_REGION")
        if "SEATING" in families: preferred.add("SEATING_POSITION")
        if "PAYLOAD" in families:
            preferred.add("REMOTE" if re.search(r"\b(remote|control)\b", text) else "CUP_SAUCER_SET")
    elif domain == "workshop":
        if "INSTRUMENT" in families: preferred.add("driver")
        if "COMPONENT" in families: preferred.add("fastener")
        if "FIXED_TARGET" in families: preferred.add("repair_target")
        if "SUPPORT" in families: preferred.add("MAIN_WORKBENCH_ZONE")

    if not preferred and weak_preference in runtime_roles:
        preferred.add(weak_preference)

    # Multiple explicit families can be legitimate (for example a source stored
    # in a vessel). Causal SOURCE/DESTINATION and Workshop participant families
    # take precedence over physical-shape words.
    authoritative = set(families)
    if "SUPPORT" in authoritative and "SEATING" in authoritative and re.search(
        r"\b(table|surface|support|platform)\b.*\b(near|beside|adjacent)\b|\b(near|beside|adjacent)\b.*\b(seat|chair|seating)\b", text
    ):
        authoritative.discard("SEATING")
    if "INSTRUMENT" in authoritative and re.search(r"\b(tool|implement|instrument|driver|wrench|drill|applicator)\b", text):
        authoritative.discard("COMPONENT")
        authoritative.discard("FIXED_TARGET")
    if (
        "COMPONENT" in authoritative
        and re.search(r"\b(joining element|connecting element|connector|installed component|component to be installed)\b", text)
        and not re.search(r"\b(assembly receiving|workpiece|fixture)\b", text)
    ):
        authoritative.discard("FIXED_TARGET")
    if "FIXED_TARGET" in authoritative and re.search(r"\b(assembly receiving|workpiece|fixture)\b", text):
        authoritative.discard("COMPONENT")
    if "PAYLOAD" in authoritative:
        authoritative.discard("DISPLAY_CONTEXT")
    if domain == "workshop" and "FIXED_TARGET" in authoritative:
        authoritative.discard("DESTINATION")
    if "SOURCE" in authoritative: authoritative.discard("SUPPORT")
    if "DESTINATION" in authoritative: authoritative.discard("SUPPORT")
    excluded = {
        candidate for candidate in runtime_roles
        if authoritative and canonical_role_family(domain, candidate) not in authoritative
    }
    if domain == "kitchen" and "SOURCE" in authoritative and re.search(r"\bsoup\b", text):
        # The current runtime has coffee and water sources, but no soup-material
        # source role. Preserve that as representability failure; never relabel it.
        excluded |= {candidate for candidate in runtime_roles
                     if canonical_role_family(domain, candidate) == "SOURCE"}
    elif domain == "kitchen" and "SOURCE" in authoritative:
        if preferred == {"water_source"}:
            excluded.add("coffee_source")
        elif preferred == {"coffee_source"}:
            excluded.add("water_source")
    return FunctionSemanticEvidence(
        preferred_roles=tuple(sorted(preferred & runtime_roles)),
        excluded_roles=tuple(sorted(excluded)),
        explicit_families=tuple(sorted(authoritative)),
        evidence_strength="EXPLICIT_FAMILY" if authoritative else ("WEAK_ALIAS" if preferred else "UNKNOWN"),
        evidence_source=evidence_source,
    )


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
    return set()


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

    for rid, role in roles.items():
        mapped, rule = (None, "NONE")
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
        })

    binary_constraints: list[tuple[str, str, set[tuple[str, str]], dict[str, Any]]] = []
    for relation in document.get("functional_relations", ()):
        rs, ro = relation.get("subject_role"), relation.get("object_role")
        if rs not in roles or ro not in roles: continue
        phrase = str(relation.get("relation", relation.get("predicate", "")))
        pairs: set[tuple[str, str]] = set()
        meanings = extract_relation_semantic_candidates(domain, phrase)
        if any(meaning.category == "TASK_EFFECT_SEMANTICS" for meaning in meanings):
            # Desired/resulting state is not a static endpoint-type precondition.
            continue
        for meaning in meanings:
            if meaning.category == "PHYSICAL_VERIFIER":
                current = relation_canonical_role_pairs(domain, meaning.predicate_name, meaning.category)
                if meaning.predicate_name in {"NEAR_SEAT", "ACCESSIBLE_FROM_BOTH_SEATS", "COMPATIBLE_WITH"}:
                    current |= {(o, s) for s, o in current}
            elif meaning.category == "TASK_CAUSAL_SEMANTICS":
                current = relation_canonical_role_pairs(domain, meaning.predicate_name, meaning.category)
            else: continue
            if meaning.direction == "REVERSE": current = {(o, s) for s, o in current}
            if relation.get("unordered_participants"):
                current |= {(o, s) for s, o in current}
            pairs |= current
        if pairs:
            binary_constraints.append((rs, ro, pairs, {"source": "RELATION_TEXT", "raw_phrase": phrase}))

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

    result = {}
    for rid, role in roles.items():
        domain_values = set(domains[rid])
        preferred = set(functions[rid].preferred_roles) & domain_values
        structural = constrained[rid]
        override = bool(structural and weak_mapped[rid] is not None
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
        elif len(values) == 1 and weak_mapped[rid] == values[0]: status = "DIRECT_FUNCTION_MATCH"
        elif structural == {"RELATION_TEXT"}: status = "RELATION_ASSISTED"
        elif structural == {"OPERATION_TEXT"}: status = "OPERATION_ASSISTED"
        elif structural: status = "JOINT_SEMANTIC_RESOLUTION"
        else: status = "DIRECT_FUNCTION_MATCH"
        result[rid] = RoleTypeHypothesis(
            raw_role_id=rid, raw_function=str(role.get("function", "")),
            raw_description=str(role.get("description", "")), entity_kind=str(role.get("entity_kind", "OBJECT")),
            canonical_role_candidates=values, status=status, evidence=tuple(evidences[rid]),
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
