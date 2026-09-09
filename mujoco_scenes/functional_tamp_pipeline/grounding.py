"""Canonical constraint-aware graph grounding boundary phi : G_F -> G_O."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, permutations, product
from typing import Any, Sequence

from .models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    GraphGroundingResult,
    NumericConstraint,
    OperationGroup,
)
from . import role_semantic_ontology as semantic_ontology
from .scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph


def project_functional_graph_to_roles(
    graph_f: FunctionalRequirementGraph,
    keep: set[str] | frozenset[str] | Sequence[str],
) -> FunctionalRequirementGraph:
    """Return the valid semantic subgraph induced by retained physical roles.

    The authoritative graph is never mutated. Graph-level configuration and
    metadata remain unchanged because this projection is only a temporary
    grounding candidate, not a rewritten FM contract.
    """
    retained = frozenset(keep)
    unknown = retained.difference(graph_f.nodes)
    if unknown:
        raise ValueError(f"Cannot project unknown functional roles: {sorted(unknown)}")

    projected = replace(
        graph_f,
        nodes={name: node for name, node in graph_f.nodes.items() if name in retained},
        relations=tuple(
            relation for relation in graph_f.relations
            if {relation.subject_role, relation.object_role} <= retained
        ),
        task_causal_relations=tuple(
            relation for relation in graph_f.task_causal_relations
            if {relation.subject_role, relation.object_role} <= retained
        ),
        task_effect_relations=tuple(
            effect for effect in graph_f.task_effect_relations
            if effect.subject_role in retained
            and (effect.object_is_literal or effect.object_value in retained)
        ),
        operation_groups=tuple(
            group for group in graph_f.operation_groups
            if group.tool_role in retained
            and group.target_role in retained
            and (group.context_role is None or group.context_role in retained)
        ),
        provisional_relation_constraints=tuple(
            constraint for constraint in graph_f.provisional_relation_constraints
            if constraint.subject_node in retained and constraint.object_node in retained
        ),
        provisional_operation_constraints=tuple(
            constraint for constraint in graph_f.provisional_operation_constraints
            if constraint.source_node in retained and constraint.target_node in retained
            and (constraint.anchor_node is None or constraint.anchor_node in retained)
        ),
    )
    projected.validate()
    return projected


def _extract_property_value(node: ObservedNode, property_name: str) -> float | None:
    """Extract a numerical property value from an observed node."""
    val = node.unary_properties.get(property_name)
    if val is not None:
        if isinstance(val, dict) and "value" in val:
            val = val["value"]
        if isinstance(val, (int, float)):
            return float(val)

    geom = node.geometry
    if property_name in geom:
        val = geom[property_name]
        if isinstance(val, dict) and "value" in val:
            val = val["value"]
        if isinstance(val, (int, float)):
            return float(val)

    props = geom.get("properties", {})
    if property_name in props:
        val = props[property_name]
        if isinstance(val, dict) and "value" in val:
            val = val["value"]
        if isinstance(val, (int, float)):
            return float(val)

    return None


def _check_unary_predicate(node: ObservedNode, predicate_name: str) -> str:
    """Check if an observed node satisfies a unary predicate. Returns 'TRUE', 'FALSE', or 'UNKNOWN'."""
    record = node.unary_predicates.get(predicate_name)
    if record is not None:
        if isinstance(record, dict):
            status = record.get("status", "UNKNOWN")
            if status in {"TRUE", "FALSE", "UNKNOWN"}:
                return status
            if "value" in record and record["value"] is not None:
                return "TRUE" if bool(record["value"]) else "FALSE"
        elif isinstance(record, bool):
            return "TRUE" if record else "FALSE"
        elif isinstance(record, str) and record.upper() in {"TRUE", "FALSE", "UNKNOWN"}:
            return record.upper()

    record = node.unary_properties.get(predicate_name)
    if record is not None:
        if isinstance(record, dict):
            status = record.get("status", "UNKNOWN")
            if status in {"TRUE", "FALSE", "UNKNOWN"}:
                return status
            if "value" in record and record["value"] is not None:
                return "TRUE" if bool(record["value"]) else "FALSE"
        elif isinstance(record, bool):
            return "TRUE" if record else "FALSE"
        elif isinstance(record, str) and record.upper() in {"TRUE", "FALSE", "UNKNOWN"}:
            return record.upper()

    geom_predicates = node.geometry.get("predicates", {})
    if predicate_name in geom_predicates:
        record = geom_predicates[predicate_name]
        if isinstance(record, dict):
            status = record.get("status", "UNKNOWN")
            if status in {"TRUE", "FALSE", "UNKNOWN"}:
                return status
        elif isinstance(record, bool):
            return "TRUE" if record else "FALSE"
        elif isinstance(record, str) and record.upper() in {"TRUE", "FALSE", "UNKNOWN"}:
            return record.upper()

    return "UNKNOWN"


def extract_plausible_labels(belief: dict[str, Any] | None) -> list[str]:
    """Extract credible semantic candidate hypotheses H(o) from semantic belief."""
    if not belief:
        return []

    validated_dict = belief.get("validated") if isinstance(belief.get("validated"), dict) else {}
    latest = belief.get("latest_observation") if isinstance(belief.get("latest_observation"), dict) else {}

    reasons = (
        belief.get("reason_codes")
        or validated_dict.get("reason_codes")
        or latest.get("reason_codes")
        or []
    )
    lack_of_evidence_codes = {
        "NO_ASSOCIATED_DETECTION",
        "INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT",
        "INSUFFICIENT_DETECTOR_CONFIDENCE",
        "SEMANTIC_LABEL_UNKNOWN",
        "SEMANTIC_REGION_LABEL_UNKNOWN",
    }
    if any(code in reasons for code in lack_of_evidence_codes):
        return []

    # Direct fields from semantic fusion
    if "plausible_labels" in belief and isinstance(belief["plausible_labels"], list) and belief["plausible_labels"]:
        return [str(lbl) for lbl in belief["plausible_labels"] if lbl]
    if "ambiguity_hypotheses" in belief and isinstance(belief["ambiguity_hypotheses"], list) and belief["ambiguity_hypotheses"]:
        return [str(lbl) for lbl in belief["ambiguity_hypotheses"] if lbl]
    if "plausible_labels" in validated_dict and isinstance(validated_dict["plausible_labels"], list) and validated_dict["plausible_labels"]:
        return [str(lbl) for lbl in validated_dict["plausible_labels"] if lbl]
    if "plausible_labels" in latest and isinstance(latest["plausible_labels"], list) and latest["plausible_labels"]:
        return [str(lbl) for lbl in latest["plausible_labels"] if lbl]
    if "ambiguity_hypotheses" in latest and isinstance(latest["ambiguity_hypotheses"], list) and latest["ambiguity_hypotheses"]:
        return [str(lbl) for lbl in latest["ambiguity_hypotheses"] if lbl]

    status = (
        belief.get("status")
        or validated_dict.get("status")
        or latest.get("status")
    )
    if status == "SUPPORTED" or status is None:
        canonical = (
            validated_dict.get("canonical_label")
            or belief.get("evaluated_label")
            or belief.get("canonical_label")
            or latest.get("canonical_label")
        )
        if canonical:
            return [str(canonical)]

    if status == "UNKNOWN" and "CONFLICTING_MULTI_VIEW_LABELS" in reasons:
        alts = belief.get("alternatives") or validated_dict.get("alternatives") or latest.get("alternatives")
        if isinstance(alts, list) and alts:
            winner = alts[0].get("label") if isinstance(alts[0], dict) else None
            candidates = [str(winner)] if winner else []
            for a in alts[1:]:
                lbl = a.get("label") if isinstance(a, dict) else None
                if lbl and str(lbl) not in candidates and int(a.get("supporting_view_count", 0)) >= 2:
                    candidates.append(str(lbl))
            return candidates

    return []


def _resolve_accepted_vocabulary(accepted_categories) -> tuple[set[str], set[str], bool]:
    """Return (literal_forms, canonical_forms, fully_known) for a role's acceptance set."""
    from .role_semantic_ontology import normalize_semantic_label, _normalize_label_text
    literal: set[str] = set()
    canonical: set[str] = set()
    fully_known = True
    for category in accepted_categories:
        literal.add(_normalize_label_text(category))
        resolved = normalize_semantic_label(category)
        if resolved is None:
            fully_known = False
        else:
            canonical.add(resolved)
    return literal, canonical, fully_known


def _observed_label_verdict(
    observed_label: str,
    accepted_categories,
    *,
    confident: bool,
) -> tuple[str, str | None]:
    """Tri-state semantic verdict for one observed label against a role's categories.

    Open-world rule.  A label matches when it is literally accepted or when it
    normalizes, through the runtime ontology's synonym groups, onto an accepted
    canonical category.  A mismatch is only FALSE when the runtime can actually
    *justify* incompatibility: the observed label and every accepted category must
    resolve inside the known vocabulary, and be disjoint there.  If either side is
    open-vocabulary the runtime has no grounds to reject, so the verdict is
    UNKNOWN and later physical/relational evidence decides.  UNKNOWN is never
    promoted to TRUE without such evidence.
    """
    from .role_semantic_ontology import normalize_semantic_label, _normalize_label_text
    literal, canonical, accepted_fully_known = _resolve_accepted_vocabulary(accepted_categories)
    observed_norm = _normalize_label_text(observed_label)
    if observed_norm in literal:
        return "TRUE", str(observed_label)
    observed_canonical = normalize_semantic_label(observed_label)
    if observed_canonical is not None and observed_canonical in canonical:
        return "TRUE", str(observed_label)
    if not confident:
        return "UNKNOWN", None
    if observed_canonical is not None and accepted_fully_known and canonical:
        # Both sides are inside the runtime's known vocabulary and do not intersect:
        # this is a justified incompatibility rather than a vocabulary gap.
        return "FALSE", None
    return "UNKNOWN", None


def check_semantic_role_compatibility(
    node_or_belief: ObservedNode | dict[str, Any] | None,
    accepted_categories: Sequence[str],
) -> tuple[str, str | None]:
    """Check if an observed node or semantic belief matches accepted categories under role-compatible ambiguity.

    Returns (status, matched_category) where status is 'TRUE', 'FALSE', or 'UNKNOWN'.
    """
    if not accepted_categories:
        if isinstance(node_or_belief, ObservedNode):
            return "TRUE", node_or_belief.canonical_category
        return "TRUE", None

    accepted_set = {cat.strip().lower().replace(" ", "_") for cat in accepted_categories}

    if isinstance(node_or_belief, ObservedNode):
        node = node_or_belief
        entity_kind = getattr(node, "entity_kind", "OBJECT")
        belief = node.semantic_labels if isinstance(node.semantic_labels, dict) else None

        if entity_kind == "OBJECT":
            # For OBJECT:
            # If explicit semantic belief with status/reasons exists, evaluate belief directly
            has_belief_contract = bool(
                belief and (
                    "status" in belief or "reason_codes" in belief or "plausible_labels" in belief
                    or "validated" in belief or "latest_observation" in belief
                )
            )
            if has_belief_contract:
                pass
            elif node.canonical_category:
                # Synthetic/legacy graph input without explicit belief contract
                return _observed_label_verdict(
                    node.canonical_category, accepted_categories, confident=True
                )
        else:
            # For REGION / FIXED_TARGET:
            if node.canonical_category:
                verdict, matched = _observed_label_verdict(
                    node.canonical_category, accepted_categories, confident=True
                )
                if verdict != "UNKNOWN":
                    return verdict, matched
                # fall through to instance-id containment for regions/fixed targets
    else:
        node = None
        entity_kind = "OBJECT"
        belief = node_or_belief if isinstance(node_or_belief, dict) else None

    # Check belief if available
    if belief:
        validated_dict = belief.get("validated") if isinstance(belief.get("validated"), dict) else {}
        latest = belief.get("latest_observation") if isinstance(belief.get("latest_observation"), dict) else {}

        status = (
            belief.get("status")
            or validated_dict.get("status")
            or latest.get("status")
        )
        reasons = (
            belief.get("reason_codes")
            or validated_dict.get("reason_codes")
            or latest.get("reason_codes")
            or []
        )
        lack_of_evidence_codes = {
            "NO_ASSOCIATED_DETECTION",
            "INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT",
            "INSUFFICIENT_DETECTOR_CONFIDENCE",
            "SEMANTIC_LABEL_UNKNOWN",
            "SEMANTIC_REGION_LABEL_UNKNOWN",
        }
        has_lack_of_evidence = any(code in reasons for code in lack_of_evidence_codes)

        if status == "SUPPORTED" and not has_lack_of_evidence:
            canonical = (
                belief.get("canonical_label")
                or validated_dict.get("canonical_label")
                or (node.canonical_category if node is not None else None)
                or latest.get("canonical_label")
            )
            if canonical:
                return _observed_label_verdict(
                    str(canonical), accepted_categories, confident=True
                )

        # If status == UNKNOWN or no supported canonical:
        if has_lack_of_evidence:
            return "UNKNOWN", None

        plausible = extract_plausible_labels(belief)
        if plausible:
            # S_sem(o, r) = TRUE iff H(o) != empty and every hypothesis is accepted.
            verdicts = [
                _observed_label_verdict(lbl, accepted_categories, confident=True)[0]
                for lbl in plausible
            ]
            if verdicts and all(v == "TRUE" for v in verdicts):
                return "TRUE", plausible[0]
            # S_sem(o, r) = FALSE only when every hypothesis is *justifiably*
            # incompatible.  A hypothesis the runtime cannot resolve leaves the
            # verdict open rather than rejecting the candidate outright.
            if verdicts and all(v == "FALSE" for v in verdicts):
                return "FALSE", None
            # Partial support, or open-vocabulary hypotheses: undecided.
            return "UNKNOWN", None

        return "UNKNOWN", None

    # Only check instance_id if entity_kind is NOT OBJECT (e.g. REGION, FIXED_TARGET)
    if node is not None and entity_kind != "OBJECT":
        norm_id = node.instance_id.strip().lower().replace(" ", "_")
        for cat in accepted_set:
            if cat in norm_id:
                return "TRUE", cat

    return "UNKNOWN", None


_check_semantic_category = check_semantic_role_compatibility


def evaluate_node_for_role(node: ObservedNode, role: FunctionalRole) -> tuple[str, dict[str, Any]]:
    """Evaluate whether an observed node satisfies a single functional role's local requirements."""
    details: dict[str, Any] = {
        "instance_id": node.instance_id,
        "role_name": role.name,
        "checks": {},
    }

    # 1. Entity kind check
    if role.entity_kind and node.entity_kind != role.entity_kind:
        # If node instance_id matches role name directly (e.g. fixed targets or regions)
        if node.instance_id != role.name:
            details["checks"]["entity_kind"] = {
                "expected": role.entity_kind,
                "actual": node.entity_kind,
                "status": "FALSE",
            }
            return "FALSE", details

    # 2. Semantic category check
    if role.semantic_categories:
        sem_status, matched_cat = _check_semantic_category(node, role.semantic_categories)
        details["checks"]["semantic_categories"] = {
            "expected": list(role.semantic_categories),
            "matched": matched_cat,
            "status": sem_status,
        }
        if sem_status == "FALSE":
            return "FALSE", details

    # 3. Unary predicates check
    for pred in role.unary_predicates:
        pred_status = _check_unary_predicate(node, pred)
        details["checks"][f"unary_predicate_{pred}"] = {
            "predicate": pred,
            "status": pred_status,
        }
        if pred_status == "FALSE":
            return "FALSE", details

    # 4. Numeric constraints check
    for constraint in role.numeric_constraints:
        val = _extract_property_value(node, constraint.property_name)
        if val is None:
            details["checks"][f"numeric_{constraint.property_name}"] = {
                "constraint": constraint.to_dict(),
                "actual_value": None,
                "status": "UNKNOWN",
            }
        else:
            passed = constraint.matches(val)
            status = "TRUE" if passed else "FALSE"
            details["checks"][f"numeric_{constraint.property_name}"] = {
                "constraint": constraint.to_dict(),
                "actual_value": val,
                "status": status,
            }
            if not passed:
                return "FALSE", details

    statuses = [c["status"] for c in details["checks"].values()]
    if "FALSE" in statuses:
        return "FALSE", details
    if "UNKNOWN" in statuses:
        return "UNKNOWN", details
    return "TRUE", details


def _evaluate_operation_group(
    grp: OperationGroup,
    selected_tools: list[str],
    selected_targets: list[str],
    graph_o: ObservedSceneGraph,
    selected_contexts: list[str] | None = None,
    required_distinct_tools: int = 1,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate an operation group pairing between selected tools and targets (and optional context).
    Returns (status, diagnostics, matching) where status in {'TRUE', 'FALSE', 'UNKNOWN'}
    and matching is a list of binding dicts: [{'tool_id': u, 'target_id': t, 'context': {...}}].
    """
    required_relations = grp.required_relations or ()
    diagnostics = []

    # Helper to check if tool u satisfies all required relations with target t and optional context c
    def check_pair(u_id: str, t_id: str, c_id: str | None = None) -> tuple[str, dict[str, Any]]:
        statuses = []
        relation_evidence = []
        checks: list[tuple[str, str, str]] = []
        if grp.physical_preconditions:
            role_to_instance = {grp.tool_role: u_id, grp.target_role: t_id}
            if grp.context_role and c_id is not None:
                role_to_instance[grp.context_role] = c_id
            checks.extend(
                (role_to_instance[s_role], pred, role_to_instance[o_role])
                for s_role, pred, o_role in grp.physical_preconditions
                if s_role in role_to_instance and o_role in role_to_instance
            )
        else:
            checks.extend((u_id, pred, t_id) for pred in required_relations)
            if grp.context_role and c_id is not None:
                checks.extend((u_id, pred, c_id) for pred in grp.context_relations)

        for subject_id, pred, object_id in checks:
            rel = graph_o.get_relation(pred, subject_id, object_id)
            rel_status = rel.status if rel else "UNKNOWN"
            statuses.append(rel_status)
            relation_evidence.append({"predicate": pred, "subject_id": subject_id, "object_id": object_id, "status": rel_status})
            if rel_status != "TRUE":
                diagnostics.append({
                    "group": grp.id,
                    "tool": u_id, "target": t_id,
                    "subject_id": subject_id, "object_id": object_id,
                    "predicate": pred,
                    "status": rel_status,
                })

        context_dict = {grp.context_role: c_id} if (grp.context_role and c_id is not None) else {}
        binding = {"tool_id": u_id, "target_id": t_id, "context": context_dict}
        if grp.capability_id:
            binding["relation_evidence"] = relation_evidence

        if "FALSE" in statuses:
            return "FALSE", binding
        if "UNKNOWN" in statuses:
            return "UNKNOWN", binding
        return "TRUE", binding

    if grp.usage_policy == "DEDICATED_PER_TARGET":
        num_targets = len(selected_targets)
        if len(selected_tools) < num_targets:
            return "FALSE", diagnostics, []

        has_unknown_matching = False
        sorted_tools = sorted(selected_tools)

        for tool_perm in permutations(sorted_tools, num_targets):
            perm_checks = []
            for i in range(num_targets):
                u = tool_perm[i]
                t = selected_targets[i]
                c = selected_contexts[i] if (selected_contexts and i < len(selected_contexts)) else None
                perm_checks.append(check_pair(u, t, c))

            perm_statuses = [chk[0] for chk in perm_checks]
            if all(s == "TRUE" for s in perm_statuses):
                matching = [chk[1] for chk in perm_checks]
                return "TRUE", [], matching
            if all(s in {"TRUE", "UNKNOWN"} for s in perm_statuses) and "UNKNOWN" in perm_statuses:
                has_unknown_matching = True

        if has_unknown_matching:
            return "UNKNOWN", diagnostics, []
        return "FALSE", diagnostics, []

    else:  # SEQUENTIAL_REUSE_ALLOWED
        sorted_tools = sorted(selected_tools)
        required_distinct_tools = min(required_distinct_tools, len(selected_targets))
        if grp.same_tool_must_cover_all_targets:
            if required_distinct_tools > 1:
                return "FALSE", [{
                    "group": grp.id,
                    "status": "ROLE_OPERATION_BINDING_CONFLICT",
                    "required_distinct_tools": required_distinct_tools,
                    "same_tool_must_cover_all_targets": True,
                }], []
            # Must find a single tool that satisfies all targets
            has_unknown_single = False
            for u in sorted_tools:
                tool_checks = []
                for i, t in enumerate(selected_targets):
                    c = selected_contexts[i] if (selected_contexts and i < len(selected_contexts)) else None
                    tool_checks.append(check_pair(u, t, c))
                statuses = [chk[0] for chk in tool_checks]
                if all(s == "TRUE" for s in statuses):
                    matching = [chk[1] for chk in tool_checks]
                    return "TRUE", [], matching
                if all(s in {"TRUE", "UNKNOWN"} for s in statuses) and "UNKNOWN" in statuses:
                    has_unknown_single = True
            if has_unknown_single:
                return "UNKNOWN", diagnostics, []
            return "FALSE", diagnostics, []
        else:
            # Reuse is optional: first satisfy explicit DISTINCT source-role
            # participation, then reuse any selected source for extra targets.
            per_target_checks: list[list[tuple[str, dict[str, Any]]]] = []
            for i, t in enumerate(selected_targets):
                c = selected_contexts[i] if (selected_contexts and i < len(selected_contexts)) else None
                checks = []
                for u in sorted_tools:
                    checks.append(check_pair(u, t, c))
                per_target_checks.append(checks)

            def covers_distinct_minimum(
                matching: tuple[tuple[str, dict[str, Any]], ...]
            ) -> bool:
                return len({item[1]["tool_id"] for item in matching}) >= required_distinct_tools

            true_options = [
                [item for item in checks if item[0] == "TRUE"]
                for checks in per_target_checks
            ]
            if all(true_options):
                for matching in product(*true_options):
                    if covers_distinct_minimum(matching):
                        return "TRUE", [], [item[1] for item in matching]

            viable_options = [
                [item for item in checks if item[0] in {"TRUE", "UNKNOWN"}]
                for checks in per_target_checks
            ]
            if all(viable_options):
                for matching in product(*viable_options):
                    if covers_distinct_minimum(matching):
                        return "UNKNOWN", diagnostics, []
            return "FALSE", diagnostics, []


def _materialize_type_hypothesis(
    graph_f: FunctionalRequirementGraph,
    selected_types: dict[str, str],
    relation_choices: dict[int, tuple[str, str, str, str]] | None = None,
) -> FunctionalRequirementGraph | None:
    """Materialize one finite tau hypothesis as a conventional canonical G_F."""
    rename = {name: selected_types.get(name, name) for name in graph_f.nodes}
    if len(set(rename.values())) != len(rename):
        return None
    nodes = {}
    for old_name, role in graph_f.nodes.items():
        new_name = rename[old_name]
        nodes[new_name] = replace(
            role,
            name=new_name,
            semantic_categories=semantic_ontology.get_system_role_semantic_categories(
                graph_f.domain, new_name
            ),
            canonical_role_candidates=(new_name,),
            role_resolution_status=(
                "SCENE_RELATION_RESOLVED"
                if len(role.canonical_role_candidates) > 1 else role.role_resolution_status
            ),
        )

    relations = [replace(r, subject_role=rename[r.subject_role], object_role=rename[r.object_role])
                 for r in graph_f.relations]
    causal = [replace(r, subject_role=rename[r.subject_role], object_role=rename[r.object_role])
              for r in graph_f.task_causal_relations]
    groups = [replace(g, tool_role=rename[g.tool_role], target_role=rename[g.target_role],
                      context_role=rename[g.context_role] if g.context_role else None,
                      physical_preconditions=tuple(
                          (rename.get(s, s), p, rename.get(o, o))
                          for s, p, o in g.physical_preconditions
                      )) for g in graph_f.operation_groups]

    relation_choices = relation_choices or {}
    for constraint_index, constraint in enumerate(graph_f.provisional_relation_constraints):
        s_type = rename[constraint.subject_node]
        o_type = rename[constraint.object_node]
        viable = [row for row in constraint.allowed_canonical_role_pairs
                  if row[0] == s_type and row[1] == o_type]
        if not viable:
            return None
        if constraint_index in relation_choices:
            chosen = relation_choices[constraint_index]
            if chosen not in viable:
                return None
        elif len(viable) == 1:
            chosen = viable[0]
        else:
            # The caller must enumerate predicate interpretations; lexical sort
            # order is never semantic evidence.
            return None
        rel = FunctionalRelation(
            subject_role=s_type, predicate=chosen[2], object_role=o_type,
            expected=constraint.expected, provenance=constraint.provenance,
            category=chosen[3],
        )
        (relations if chosen[3] == "PHYSICAL_VERIFIER" else causal).append(rel)

    for constraint in graph_f.provisional_operation_constraints:
        viable = []
        if constraint.slot_assignments:
            capabilities = {cap["capability_id"]: cap for cap in constraint.capability_candidates}
            for assignment in constraint.slot_assignments:
                cap = capabilities.get(assignment["capability_id"])
                source_type = rename[assignment["source_node"]]
                target_type = rename[assignment["target_node"]]
                anchor_node = assignment.get("anchor_node")
                anchor_type = rename[anchor_node] if anchor_node else None
                if (
                    cap
                    and source_type == assignment["source_type"]
                    and target_type == assignment["target_type"]
                    and anchor_type == assignment.get("anchor_type")
                ):
                    viable.append((cap, source_type, target_type, anchor_type, assignment["usage_policy"]))
        else:
            s_type = rename[constraint.source_node]
            t_type = rename[constraint.target_node]
            a_type = rename[constraint.anchor_node] if constraint.anchor_node else None
            for cap in constraint.capability_candidates:
                orientations = [(s_type, t_type)]
                if graph_f.domain == "living_room":
                    orientations.append((t_type, s_type))
                for effective_source, effective_target in orientations:
                    if (effective_source in cap["allowed_source_roles"]
                            and effective_target in cap["allowed_target_roles"]
                            and (a_type is None or a_type in cap["allowed_anchor_roles"])):
                        viable.append((cap, effective_source, effective_target, a_type, constraint.reuse_policy))
        if len(viable) != 1:
            return None
        cap, source_type, target_type, a_type, reuse_policy = viable[0]
        endpoint = {"source": source_type, "target": target_type, "anchor": a_type}
        templates = tuple(
            (endpoint[s], predicate, endpoint[o])
            for s, predicate, o in cap["required_relation_templates"]
            if endpoint.get(s) and endpoint.get(o)
        )
        groups.append(OperationGroup(
            id=constraint.raw_operation_id,
            function=cap["planner_operation"], tool_role=source_type,
            target_role=target_type, context_role=a_type,
            required_target_count=constraint.required_count,
            usage_policy=reuse_policy,
            capability_id=cap["capability_id"], physical_preconditions=templates,
            preconditions_provenance=({"source": "ROBOT_CAPABILITY_REGISTRY",
                                       "capability_id": cap["capability_id"]},),
        ))

    return replace(
        graph_f, nodes=nodes, relations=tuple(relations),
        task_causal_relations=tuple(causal), operation_groups=tuple(groups),
        provisional_relation_constraints=(), provisional_operation_constraints=(),
    )


def _ground_provisional_graph(
    graph_f: FunctionalRequirementGraph,
    graph_o: ObservedSceneGraph,
    domain_context: dict[str, Any] | None,
) -> GraphGroundingResult:
    provisional = [
        (name, tuple(role.canonical_role_candidates))
        for name, role in sorted(graph_f.nodes.items())
        if len(role.canonical_role_candidates) > 1
    ]
    candidates = []
    failures = []
    for types in product(*[domain for _, domain in provisional]):
        tau_nodes = dict(zip((name for name, _ in provisional), types))
        rename = {name: tau_nodes.get(name, name) for name in graph_f.nodes}
        relation_options = []
        for constraint in graph_f.provisional_relation_constraints:
            viable = tuple(row for row in constraint.allowed_canonical_role_pairs
                           if row[0] == rename[constraint.subject_node]
                           and row[1] == rename[constraint.object_node])
            relation_options.append(viable)
        if any(not choices for choices in relation_options):
            continue
        choice_products = product(*relation_options) if relation_options else [()]
        for chosen_rows in choice_products:
            choices = dict(enumerate(chosen_rows))
            resolved = _materialize_type_hypothesis(graph_f, tau_nodes, choices)
            if resolved is None:
                continue
            result = ground_graph(resolved, graph_o, domain_context)
            raw_tau = {
                role.raw_role_id or old_name: rename_type
                for old_name, role in graph_f.nodes.items()
                for rename_type in [tau_nodes.get(old_name, old_name)]
            }
            interpretation = tuple((row[0], row[2], row[1], row[3]) for row in chosen_rows)
            if result.complete:
                candidates.append((raw_tau, interpretation, resolved, result))
            else:
                failures.append(result)
    distinct_interpretations = {
        (tuple(sorted(tau.items())), interpretation)
        for tau, interpretation, _, _ in candidates
    }
    if len(distinct_interpretations) > 1:
        return GraphGroundingResult(
            status="INFEASIBLE" if (domain_context or {}).get("search_exhausted", True) else "INCOMPLETE",
            complete=False,
            unresolved_constraints=("AMBIGUOUS_FUNCTIONAL_ASSIGNMENT",),
            evidence={
                "valid_type_relation_assignments": [
                    {"role_types": dict(tau), "relations": [list(row) for row in relations]}
                    for tau, relations in sorted(distinct_interpretations)
                ]
            },
            failure_kind="FUNCTIONAL_ASSIGNMENT_FAILURE",
        )
    if candidates:
        tau, _, resolved, result = candidates[0]
        evidence = dict(result.evidence)
        evidence["type_resolution_provenance"] = "G_O_RELATION_AND_OPERATION_CONSTRAINTS"
        return replace(result, resolved_role_types=tau, resolved_graph=resolved.to_dict(), evidence=evidence)
    discovery_only = bool(failures) and all(
        failure.failure_kind == "OBJECT_DISCOVERY_FAILURE" for failure in failures
    )
    return GraphGroundingResult(
        status="INFEASIBLE" if (domain_context or {}).get("search_exhausted", True) else "INCOMPLETE",
        complete=False,
        unresolved_constraints=("NO_VALID_TYPE_OBJECT_ASSIGNMENT",),
        evidence={"type_hypotheses_evaluated": len(failures)},
        failure_kind="OBJECT_DISCOVERY_FAILURE" if discovery_only else "FUNCTIONAL_ASSIGNMENT_FAILURE",
    )


def resolved_functional_graph(
    graph_f: FunctionalRequirementGraph,
    grounding: GraphGroundingResult,
) -> FunctionalRequirementGraph:
    """Return the fully materialized G_F* selected by grounding, if present."""
    if grounding.resolved_graph is None:
        return graph_f
    resolved = FunctionalRequirementGraph.from_dict(grounding.resolved_graph)
    resolved.validate()
    return resolved


def ground_graph(
    graph_f: FunctionalRequirementGraph,
    graph_o: ObservedSceneGraph,
    domain_context: dict[str, Any] | None = None,
) -> GraphGroundingResult:
    """Perform constraint-aware graph grounding phi : G_F -> G_O."""
    graph_f.validate()
    if any(len(role.canonical_role_candidates) > 1 for role in graph_f.nodes.values()):
        return _ground_provisional_graph(graph_f, graph_o, domain_context)
    context = domain_context or {}
    roles = graph_f.nodes
    search_exhausted = bool(context.get("search_exhausted", True))

    # Step 1: Find candidate nodes for each role (both TRUE and UNKNOWN)
    role_candidates_true: dict[str, list[str]] = {}
    role_candidates_unknown: dict[str, list[str]] = {}
    evaluations: dict[tuple[str, str], dict[str, Any]] = {}

    for role_name, role in roles.items():
        cands_true: list[str] = []
        cands_unk: list[str] = []
        blocked_property = any(item['role'] == role_name for item in graph_f.metadata.get('unverified_required_properties', []))
        for instance_id, node in sorted(graph_o.nodes.items()):
            if blocked_property:
                continue
            status, details = evaluate_node_for_role(node, role)
            evaluations[(role_name, instance_id)] = details
            if status == "TRUE":
                cands_true.append(instance_id)
            elif status == "UNKNOWN":
                cands_unk.append(instance_id)

        role_candidates_true[role_name] = cands_true
        role_candidates_unknown[role_name] = cands_unk

    # Check minimum required candidate availability
    missing_roles_definitive: list[str] = []
    missing_roles_potential: list[str] = []

    for role_name, role in roles.items():
        true_cnt = len(role_candidates_true[role_name])
        unk_cnt = len(role_candidates_unknown[role_name])
        if true_cnt < role.minimum_count:
            if true_cnt + unk_cnt >= role.minimum_count:
                missing_roles_potential.append(role_name)
            else:
                missing_roles_definitive.append(role_name)

    # Per-role individual plausibility, independent of any joint assignment.
    # A role is individually satisfiable when the number of observed individuals that
    # are semantically TRUE or UNKNOWN for it meets its minimum count.  This is the
    # signal that separates OBJECT_DISCOVERY_FAILURE (too few plausible individuals)
    # from FUNCTIONAL_ASSIGNMENT_FAILURE (enough individuals, no consistent joint choice).
    role_plausibility: dict[str, dict[str, Any]] = {}
    for role_name, role in roles.items():
        n_true = len(role_candidates_true[role_name])
        n_unknown = len(role_candidates_unknown[role_name])
        role_plausibility[role_name] = {
            "true": n_true,
            "unknown": n_unknown,
            "plausible": n_true + n_unknown,
            "minimum_count": role.minimum_count,
            "sufficient": (n_true + n_unknown) >= role.minimum_count,
        }
    individual_candidates_sufficient = all(
        entry["sufficient"] for entry in role_plausibility.values()
    )
    plausibility_evidence: dict[str, Any] = {
        "role_plausibility": role_plausibility,
        "individual_candidates_sufficient": individual_candidates_sufficient,
    }

    if missing_roles_definitive:
        return GraphGroundingResult(
            status="INFEASIBLE" if search_exhausted else "INCOMPLETE",
            complete=False,
            assignment=None,
            operation_bindings={},
            missing_roles=tuple(missing_roles_definitive),
            unsatisfied_relations=(),
            unresolved_constraints=tuple(missing_roles_definitive),
            evidence={"candidate_evaluations": {f"{k[0]}:{k[1]}": v for k, v in evaluations.items()},
                      **plausibility_evidence},
            failure_kind="OBJECT_DISCOVERY_FAILURE" if search_exhausted else None,
        )

    # Pigeonhole principle capacity check for distinct objects
    total_distinct_objs_required = sum(
        role.minimum_count for role in roles.values()
        if role.entity_kind == "OBJECT" and not role.shared
    )
    available_candidate_objects = set().union(
        *[role_candidates_true[r] + role_candidates_unknown[r]
          for r, role in roles.items()
          if role.entity_kind == "OBJECT" and not role.shared]
    ) if roles else set()
    if total_distinct_objs_required > len(available_candidate_objects):
        return GraphGroundingResult(
            status="INFEASIBLE" if search_exhausted else "INCOMPLETE",
            complete=False,
            assignment=None,
            operation_bindings={},
            missing_roles=tuple(missing_roles_definitive),
            unsatisfied_relations=(),
            unresolved_constraints=("INSUFFICIENT_SCENE_OBJECTS_FOR_ROLES",),
            evidence={"candidate_evaluations": {f"{k[0]}:{k[1]}": v for k, v in evaluations.items()},
                      **plausibility_evidence},
            failure_kind="OBJECT_DISCOVERY_FAILURE" if search_exhausted else None,
        )

    # Collect operation-managed relation signatures to avoid double-enforcing with Cartesian semantics
    operation_managed_edges: set[tuple[str, str, str]] = set()
    for grp in graph_f.operation_groups:
        req_rels = grp.required_relations or ()
        for pred in req_rels:
            operation_managed_edges.add((grp.tool_role, pred, grp.target_role))
        if grp.context_role:
            for pred in grp.context_relations:
                operation_managed_edges.add((grp.tool_role, pred, grp.context_role))

    # Role count schedules respecting preferences (e.g. minimize_distinct)
    role_names = sorted(roles.keys())
    role_count_options: dict[str, list[int]] = {}
    for r_name in role_names:
        role = roles[r_name]
        min_c = role.minimum_count
        max_c = role.maximum_count
        if role.preference == "minimize_distinct":
            role_count_options[r_name] = list(range(min_c, max_c + 1))
        else:
            role_count_options[r_name] = list(range(max_c, min_c - 1, -1)) if max_c != min_c else [min_c]

    # Generate all role count configurations
    all_count_configs = list(product(*[role_count_options[r] for r in role_names]))

    unsatisfied_relations_recorded: list[dict[str, Any]] = []
    unresolved_relations_recorded: list[dict[str, Any]] = []
    valid_assignments: list[tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]] = []
    valid_unknown_assignments: list[tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]] = []
    has_unknown_combination = bool(missing_roles_potential)

    for count_config in all_count_configs:
        config_map = dict(zip(role_names, count_config))

        # Build candidate combos for this count configuration
        role_combos_for_config = []
        for r_name in role_names:
            req_c = config_map[r_name]
            true_cands = role_candidates_true[r_name]
            unk_cands = role_candidates_unknown[r_name]
            all_cands = [(c, True) for c in true_cands] + [(c, False) for c in unk_cands]

            if len(all_cands) < req_c:
                role_combos_for_config.append([])
            else:
                role_combos_for_config.append(list(combinations(all_cands, req_c)))

        if any(len(combos) == 0 for combos in role_combos_for_config):
            continue

        for combo in product(*role_combos_for_config):
            assignment_map: dict[str, Any] = {}
            used_instances: set[str] = set()
            conflict = False
            selected_unknown_nodes: list[tuple[str, str]] = []

            for r_idx, r_name in enumerate(role_names):
                role = roles[r_name]
                selected_tagged = combo[r_idx]
                selected_ids = [item[0] for item in selected_tagged]
                if any(not item[1] for item in selected_tagged):
                    selected_unknown_nodes.extend((r_name, item[0]) for item in selected_tagged if not item[1])

                if role.entity_kind == "OBJECT" and not role.shared:
                    for inst_id in selected_ids:
                        if inst_id in used_instances:
                            conflict = True
                            break
                        used_instances.add(inst_id)
                if conflict:
                    break

                assignment_map[r_name] = (
                    selected_ids[0] if len(selected_ids) == 1 else list(selected_ids)
                )

            if conflict:
                continue

            # Check cross-group reuse policy
            if not graph_f.cross_group_reuse_allowed and len(graph_f.operation_groups) > 1:
                group_tool_instances: list[set[str]] = []
                cross_group_conflict = False
                for grp in graph_f.operation_groups:
                    t_insts = assignment_map.get(grp.tool_role, [])
                    t_set = set([t_insts] if isinstance(t_insts, str) else t_insts)
                    for prev_set in group_tool_instances:
                        if prev_set & t_set:
                            cross_group_conflict = True
                            break
                    if cross_group_conflict:
                        break
                    group_tool_instances.append(t_set)
                if cross_group_conflict:
                    continue

            # Evaluate Operation Groups
            combo_status = "TRUE"
            combo_op_bindings: dict[str, list[dict[str, Any]]] = {}
            incident_evidence: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for grp in graph_f.operation_groups:
                tools = assignment_map.get(grp.tool_role, [])
                targets = assignment_map.get(grp.target_role, [])
                contexts = assignment_map.get(grp.context_role, []) if grp.context_role else None
                tool_list = [tools] if isinstance(tools, str) else list(tools)
                target_list = [targets] if isinstance(targets, str) else list(targets)
                context_list = ([contexts] if isinstance(contexts, str) else list(contexts)) if contexts is not None else None
                source_role = roles[grp.tool_role]
                required_distinct_tools = (
                    min(source_role.minimum_count, len(target_list))
                    if source_role.binding_policy == "DISTINCT"
                    else 1
                )

                grp_stat, grp_diags, grp_matching = _evaluate_operation_group(
                    grp, tool_list, target_list, graph_o, context_list,
                    required_distinct_tools=required_distinct_tools,
                )
                if grp_stat == "FALSE":
                    combo_status = "FALSE"
                    unsatisfied_relations_recorded.extend(grp_diags)
                    break
                elif grp_stat == "UNKNOWN":
                    if combo_status != "FALSE":
                        combo_status = "UNKNOWN"
                    unresolved_relations_recorded.extend(grp_diags)
                else:
                    for binding in grp_matching:
                        if grp.capability_id:
                            binding["capability_id"] = grp.capability_id
                            binding["source_operation_id"] = grp.id
                            binding["planner_operation"] = grp.function
                        binding_relation_evidence = binding.get("relation_evidence", [])
                        if not binding_relation_evidence:
                            role_to_instance = {grp.tool_role: binding["tool_id"], grp.target_role: binding["target_id"]}
                            role_to_instance.update(binding.get("context", {}))
                            templates = grp.physical_preconditions or tuple(
                                [(grp.tool_role, pred, grp.target_role) for pred in grp.required_relations]
                                + [(grp.tool_role, pred, grp.context_role) for pred in grp.context_relations if grp.context_role]
                            )
                            binding_relation_evidence = [
                                {"predicate": pred, "subject_id": role_to_instance[s_role],
                                 "object_id": role_to_instance[o_role],
                                 "status": (graph_o.get_relation(pred, role_to_instance[s_role], role_to_instance[o_role]).status
                                            if graph_o.get_relation(pred, role_to_instance[s_role], role_to_instance[o_role]) else "UNKNOWN")}
                                for s_role, pred, o_role in templates
                                if s_role in role_to_instance and o_role in role_to_instance
                            ]
                        for relation_evidence in binding_relation_evidence:
                            s_id = relation_evidence["subject_id"]
                            o_id = relation_evidence["object_id"]
                            role_instances = [(grp.tool_role, tool_list), (grp.target_role, target_list)]
                            if grp.context_role and context_list:
                                role_instances.append((grp.context_role, context_list))
                            for endpoint_role, endpoint_ids in role_instances:
                                for endpoint_id in endpoint_ids:
                                    if endpoint_id in {s_id, o_id}:
                                        incident_evidence.setdefault((endpoint_role, endpoint_id), []).append(relation_evidence)
                    combo_op_bindings[grp.id] = grp_matching

            if combo_status == "FALSE":
                continue

            # Evaluate explicit relations in G_F (skipping operation-managed edges)
            for relation in graph_f.relations:
                subj_role = relation.subject_role
                obj_role = relation.object_role
                predicate = relation.predicate

                if (subj_role, predicate, obj_role) in operation_managed_edges:
                    continue

                if subj_role not in assignment_map or obj_role not in assignment_map:
                    continue

                subj_instances = (
                    [assignment_map[subj_role]]
                    if isinstance(assignment_map[subj_role], str)
                    else assignment_map[subj_role]
                )
                obj_instances = (
                    [assignment_map[obj_role]]
                    if isinstance(assignment_map[obj_role], str)
                    else assignment_map[obj_role]
                )

                for s_id in subj_instances:
                    for o_id in obj_instances:
                        if s_id == o_id:
                            continue
                        obs_rel = graph_o.get_relation(predicate, s_id, o_id)
                        rel_status = obs_rel.status if obs_rel else "UNKNOWN"
                        rel_evidence = {
                            "predicate": predicate, "subject_id": s_id,
                            "object_id": o_id, "status": rel_status,
                            "provenance": relation.provenance,
                        }
                        incident_evidence.setdefault((subj_role, s_id), []).append(rel_evidence)
                        incident_evidence.setdefault((obj_role, o_id), []).append(rel_evidence)

                        # Determine satisfaction based on relation.expected
                        if rel_status == "UNKNOWN":
                            inst_status = "UNKNOWN"
                        elif relation.expected:
                            inst_status = "TRUE" if rel_status == "TRUE" else "FALSE"
                        else:
                            inst_status = "TRUE" if rel_status == "FALSE" else "FALSE"

                        if inst_status == "FALSE":
                            combo_status = "FALSE"
                            unsatisfied_relations_recorded.append({
                                "subject_role": subj_role,
                                "subject_id": s_id,
                                "predicate": predicate,
                                "object_role": obj_role,
                                "object_id": o_id,
                                "status": rel_status,
                                "expected": relation.expected,
                            })
                            break
                        elif inst_status == "UNKNOWN":
                            if combo_status != "FALSE":
                                combo_status = "UNKNOWN"
                            unresolved_relations_recorded.append({
                                "subject_role": subj_role,
                                "subject_id": s_id,
                                "predicate": predicate,
                                "object_role": obj_role,
                                "object_id": o_id,
                                "status": rel_status,
                                "expected": relation.expected,
                            })
                    if combo_status == "FALSE":
                        break
                if combo_status == "FALSE":
                    break

            binding_provenance: dict[str, Any] = {}
            if combo_status != "FALSE":
                for role_name, instance_id in selected_unknown_nodes:
                    detail = evaluations[(role_name, instance_id)]
                    unknown_checks = [name for name, check in detail["checks"].items() if check["status"] == "UNKNOWN"]
                    relation_evidence = incident_evidence.get((role_name, instance_id), [])
                    recoverable = (
                        unknown_checks == ["semantic_categories"]
                        and bool(relation_evidence)
                        and all(item["status"] == "TRUE" for item in relation_evidence)
                    )
                    if not recoverable:
                        combo_status = "UNKNOWN"
                    binding_provenance[f"{role_name}:{instance_id}"] = {
                        "role": role_name,
                        "observed_instance": instance_id,
                        "role_resolution": roles[role_name].role_resolution_status,
                        "grounding_mode": "RELATIONALLY_VERIFIED_GROUNDING" if recoverable else "UNRESOLVED_SEMANTIC_UNKNOWN",
                        "semantic_status": "UNKNOWN",
                        "relation_evidence": relation_evidence,
                        "operation_evidence": [
                            {"capability": grp.capability_id, "source_operation_id": grp.id}
                            for grp in graph_f.operation_groups
                            if grp.tool_role == role_name or grp.target_role == role_name or grp.context_role == role_name
                        ],
                    }
                for role_name, assigned in assignment_map.items():
                    instance_ids = [assigned] if isinstance(assigned, str) else assigned
                    for instance_id in instance_ids:
                        key = f"{role_name}:{instance_id}"
                        binding_provenance.setdefault(key, {
                            "role": role_name,
                            "observed_instance": instance_id,
                            "role_resolution": roles[role_name].role_resolution_status,
                            "grounding_mode": "FUNCTION_AND_RELATION",
                            "semantic_status": evaluations[(role_name, instance_id)]["checks"].get("semantic_categories", {}).get("status", "TRUE"),
                            "relation_evidence": incident_evidence.get((role_name, instance_id), []),
                            "operation_evidence": [
                                {"capability": grp.capability_id, "source_operation_id": grp.id}
                                for grp in graph_f.operation_groups
                                if grp.tool_role == role_name or grp.target_role == role_name or grp.context_role == role_name
                            ],
                        })

            if combo_status == "TRUE":
                operations_complete = all(
                    grp.id in combo_op_bindings
                    and len(combo_op_bindings[grp.id]) >= grp.required_target_count
                    for grp in graph_f.operation_groups
                )
                if operations_complete:
                    valid_assignments.append((assignment_map, combo_op_bindings, binding_provenance))
                    break
                else:
                    combo_status = "UNKNOWN"
            if combo_status == "UNKNOWN":
                has_unknown_combination = True
                valid_unknown_assignments.append((assignment_map, combo_op_bindings))

        if valid_assignments:
            break

    if valid_assignments:
        valid_assignments.sort(key=lambda a: (
            sum(item.get("semantic_status") == "UNKNOWN" for item in a[2].values()),
            tuple(sorted(str(v) for v in a[0].values())),
        ))
        chosen_assignment, chosen_bindings, chosen_provenance = valid_assignments[0]
        return GraphGroundingResult(
            status="COMPLETE",
            complete=True,
            assignment=chosen_assignment,
            operation_bindings=chosen_bindings,
            missing_roles=(),
            unsatisfied_relations=(),
            unresolved_constraints=(),
            evidence={"valid_assignment_count": len(valid_assignments), "binding_provenance": chosen_provenance,
                      "operation_binding_complete": True, **plausibility_evidence},
        )

    if not search_exhausted:
        # During active search, uninspected regions can bring new candidates
        unres = list(missing_roles_potential) + list(missing_roles_definitive)
        unres.extend(r.get("predicate", "UNSATISFIED") for r in unsatisfied_relations_recorded)
        unres.extend(r.get("predicate", "UNKNOWN") for r in unresolved_relations_recorded)
        return GraphGroundingResult(
            status="INCOMPLETE",
            complete=False,
            assignment=None,
            operation_bindings={},
            missing_roles=tuple(dict.fromkeys(missing_roles_potential + missing_roles_definitive)),
            unsatisfied_relations=tuple(unsatisfied_relations_recorded),
            unresolved_constraints=tuple(dict.fromkeys(unres)),
            evidence={
                "search_exhausted": False,
                "unresolved_relations": unresolved_relations_recorded,
                "unsatisfied_relations": unsatisfied_relations_recorded,
                **plausibility_evidence,
            },
            failure_kind=None,
        )

    # Search is exhausted: check if failure was due to unresolved UNKNOWN evidence vs definitive FALSE
    if has_unknown_combination or missing_roles_potential or unresolved_relations_recorded:
        unres = list(missing_roles_potential)
        unres.extend(r.get("predicate", "UNKNOWN") for r in unresolved_relations_recorded)
        return GraphGroundingResult(
            status="INCOMPLETE",
            complete=False,
            assignment=None,
            operation_bindings={},
            missing_roles=tuple(missing_roles_potential),
            unsatisfied_relations=tuple(unsatisfied_relations_recorded),
            unresolved_constraints=tuple(dict.fromkeys(unres)),
            evidence={
                "search_exhausted": True,
                "unresolved_relations": unresolved_relations_recorded,
                **plausibility_evidence,
            },
            failure_kind="FUNCTIONAL_ASSIGNMENT_FAILURE",
        )

    return GraphGroundingResult(
        status="INFEASIBLE",
        complete=False,
        assignment=None,
        operation_bindings={},
        missing_roles=tuple(missing_roles_definitive),
        unsatisfied_relations=tuple(unsatisfied_relations_recorded),
        unresolved_constraints=tuple(
            dict.fromkeys(
                list(missing_roles_definitive)
                + [r.get("predicate", "UNSATISFIED") for r in unsatisfied_relations_recorded]
            )
        ),
        evidence={
            "search_exhausted": True,
            "unsatisfied_relations": unsatisfied_relations_recorded,
            **plausibility_evidence,
        },
        failure_kind=(
            "FUNCTIONAL_ASSIGNMENT_FAILURE"
            if individual_candidates_sufficient
            else "OBJECT_DISCOVERY_FAILURE"
        ),
    )



def ground_verified_candidate_subgraph(graph_f, graph_o, context=None):
    """Retain the largest verified role subset after observation exhaustion.

    Every trial uses the same semantic/geometric verifier and all relations
    induced by its retained roles. This is grounding, never an A* replan.
    The original requested graph/cardinalities remain unchanged.
    """
    from itertools import combinations
    context = dict(context or {}, search_exhausted=True)
    full = ground_graph(graph_f, graph_o, context)
    if full.complete or not getattr(
        graph_f, "online_executable_contract_complete", getattr(graph_f, "required_contract_complete", True)
    ):
        return full
    names = sorted(graph_f.nodes)
    for size in range(len(names) - 1, 0, -1):
        for selected in combinations(names, size):
            keep = set(selected)
            candidate = project_functional_graph_to_roles(graph_f, keep)
            verified = ground_graph(candidate, graph_o, context)
            if verified.complete and verified.assignment:
                return replace(verified, complete=False, status='PARTIAL_VERIFIED_GROUNDING',
                    missing_roles=tuple(sorted(set(names) - keep)),
                    failure_kind=full.failure_kind,
                    evidence={**verified.evidence, 'candidate_roles': list(selected),
                              'original_grounding_status': full.status, 'search_exhausted': True})
    return full
