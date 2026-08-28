"""Canonical constraint-aware graph grounding boundary phi : G_F -> G_O."""

from __future__ import annotations

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
from .scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph


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
                norm_canonical = node.canonical_category.strip().lower().replace(" ", "_")
                if norm_canonical in accepted_set:
                    return "TRUE", node.canonical_category
                return "FALSE", None
        else:
            # For REGION / FIXED_TARGET:
            if node.canonical_category:
                norm_canonical = node.canonical_category.strip().lower().replace(" ", "_")
                if norm_canonical in accepted_set:
                    return "TRUE", node.canonical_category
                return "FALSE", None
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
                norm_canonical = str(canonical).strip().lower().replace(" ", "_")
                if norm_canonical in accepted_set:
                    return "TRUE", str(canonical)
                return "FALSE", None

        # If status == UNKNOWN or no supported canonical:
        if has_lack_of_evidence:
            return "UNKNOWN", None

        plausible = extract_plausible_labels(belief)
        if plausible:
            norm_plausible = {lbl.strip().lower().replace(" ", "_") for lbl in plausible}
            # S_sem(o, r) = TRUE iff H(o) != ∅ and H(o) ⊆ C(r)
            if norm_plausible.issubset(accepted_set):
                return "TRUE", plausible[0]
            # S_sem(o, r) = FALSE iff H(o) != ∅ and H(o) ∩ C(r) = ∅
            if norm_plausible.isdisjoint(accepted_set):
                return "FALSE", None
            # Mixed overlap (some in C(r), some not)
            return "UNKNOWN", None

        if status == "SUPPORTED":
            return "FALSE", None
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
        if sem_status == "UNKNOWN":
            return "UNKNOWN", details

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
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate an operation group pairing between selected tools and targets (and optional context).
    Returns (status, diagnostics, matching) where status in {'TRUE', 'FALSE', 'UNKNOWN'}
    and matching is a list of binding dicts: [{'tool_id': u, 'target_id': t, 'context': {...}}].
    """
    required_relations = grp.required_relations or ("INSERTABLE_IN", "REACHES_BOTTOM")
    diagnostics = []

    # Helper to check if tool u satisfies all required relations with target t and optional context c
    def check_pair(u_id: str, t_id: str, c_id: str | None = None) -> tuple[str, dict[str, Any]]:
        statuses = []
        for pred in required_relations:
            rel = graph_o.get_relation(pred, u_id, t_id)
            rel_status = rel.status if rel else "UNKNOWN"
            statuses.append(rel_status)
            if rel_status != "TRUE":
                diagnostics.append({
                    "group": grp.id,
                    "tool": u_id,
                    "target": t_id,
                    "predicate": pred,
                    "status": rel_status,
                })

        if grp.context_role and c_id is not None:
            for pred in grp.context_relations:
                rel = graph_o.get_relation(pred, u_id, c_id)
                rel_status = rel.status if rel else "UNKNOWN"
                statuses.append(rel_status)
                if rel_status != "TRUE":
                    diagnostics.append({
                        "group": grp.id,
                        "tool": u_id,
                        "context_id": c_id,
                        "predicate": pred,
                        "status": rel_status,
                    })

        context_dict = {grp.context_role: c_id} if (grp.context_role and c_id is not None) else {}
        binding = {"tool_id": u_id, "target_id": t_id, "context": context_dict}

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
        if grp.same_tool_must_cover_all_targets:
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
            # Each target must be satisfied by at least one selected tool
            target_matching: list[dict[str, Any]] = []
            target_statuses: list[str] = []
            for i, t in enumerate(selected_targets):
                c = selected_contexts[i] if (selected_contexts and i < len(selected_contexts)) else None
                found_true_tool = False
                has_unknown_tool = False
                chosen_binding: dict[str, Any] | None = None
                for u in sorted_tools:
                    st, b = check_pair(u, t, c)
                    if st == "TRUE":
                        found_true_tool = True
                        chosen_binding = b
                        break
                    elif st == "UNKNOWN":
                        has_unknown_tool = True
                        if chosen_binding is None:
                            chosen_binding = b

                if found_true_tool and chosen_binding is not None:
                    target_statuses.append("TRUE")
                    target_matching.append(chosen_binding)
                elif has_unknown_tool:
                    target_statuses.append("UNKNOWN")
                    if chosen_binding is not None:
                        target_matching.append(chosen_binding)
                else:
                    target_statuses.append("FALSE")

            if all(s == "TRUE" for s in target_statuses):
                return "TRUE", [], target_matching
            if all(s in {"TRUE", "UNKNOWN"} for s in target_statuses) and "UNKNOWN" in target_statuses:
                return "UNKNOWN", diagnostics, []
            return "FALSE", diagnostics, []


def ground_graph(
    graph_f: FunctionalRequirementGraph,
    graph_o: ObservedSceneGraph,
    domain_context: dict[str, Any] | None = None,
) -> GraphGroundingResult:
    """Perform constraint-aware graph grounding phi : G_F -> G_O."""
    graph_f.validate()
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
        for instance_id, node in sorted(graph_o.nodes.items()):
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

    if missing_roles_definitive:
        return GraphGroundingResult(
            status="INFEASIBLE" if search_exhausted else "INCOMPLETE",
            complete=False,
            assignment=None,
            operation_bindings={},
            missing_roles=tuple(missing_roles_definitive),
            unsatisfied_relations=(),
            unresolved_constraints=tuple(missing_roles_definitive),
            evidence={"candidate_evaluations": {f"{k[0]}:{k[1]}": v for k, v in evaluations.items()}},
        )

    # Collect operation-managed relation signatures to avoid double-enforcing with Cartesian semantics
    operation_managed_edges: set[tuple[str, str, str]] = set()
    for grp in graph_f.operation_groups:
        req_rels = grp.required_relations or ("INSERTABLE_IN", "REACHES_BOTTOM")
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
    valid_assignments: list[tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]] = []
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
            combo_has_unknown_node = False

            for r_idx, r_name in enumerate(role_names):
                role = roles[r_name]
                selected_tagged = combo[r_idx]
                selected_ids = [item[0] for item in selected_tagged]
                if any(not item[1] for item in selected_tagged):
                    combo_has_unknown_node = True

                if not role.reusable and not role.shared:
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
            combo_status = "UNKNOWN" if combo_has_unknown_node else "TRUE"
            combo_op_bindings: dict[str, list[dict[str, Any]]] = {}
            for grp in graph_f.operation_groups:
                tools = assignment_map.get(grp.tool_role, [])
                targets = assignment_map.get(grp.target_role, [])
                contexts = assignment_map.get(grp.context_role, []) if grp.context_role else None
                tool_list = [tools] if isinstance(tools, str) else list(tools)
                target_list = [targets] if isinstance(targets, str) else list(targets)
                context_list = ([contexts] if isinstance(contexts, str) else list(contexts)) if contexts is not None else None

                grp_stat, grp_diags, grp_matching = _evaluate_operation_group(
                    grp, tool_list, target_list, graph_o, context_list
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

            if combo_status == "TRUE":
                valid_assignments.append((assignment_map, combo_op_bindings))
                break
            elif combo_status == "UNKNOWN":
                has_unknown_combination = True

        if valid_assignments:
            break

    if valid_assignments:
        valid_assignments.sort(key=lambda a: sorted(str(v) for v in a[0].values()))
        chosen_assignment, chosen_bindings = valid_assignments[0]
        return GraphGroundingResult(
            status="COMPLETE",
            complete=True,
            assignment=chosen_assignment,
            operation_bindings=chosen_bindings,
            missing_roles=(),
            unsatisfied_relations=(),
            unresolved_constraints=(),
            evidence={"valid_assignment_count": len(valid_assignments)},
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
            },
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
            },
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
        },
    )

