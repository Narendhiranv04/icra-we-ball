"""Frozen, evaluation-only scoring of raw V1 and V2 FM contracts.

The matcher intentionally does not import the sanitizer, semantic compiler,
runtime ontologies, domain adapters, or capability registry. Reference data is
permitted here because this module is used only after a run has completed.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable, Mapping

from .evaluation_contract_adapter import EvaluationContract, EvaluationOperation, extract_evaluation_contract


_ROLE_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "kitchen": {
        "coffee_container": (r"coffee (cup|mug|container|receptacle)", r"(contain|hold).*coffee", r"receive.*coffee"),
        "soup_container": (r"soup (bowl|cup|container|receptacle)", r"contain.*soup", r"receive.*soup"),
        "coffee_stirrer": (r"stir(rer|ring)?", r"mix.*coffee"),
        "soup_eating_utensil": (r"eat(ing)?.*utensil.*soup", r"soup.*(eating utensil|spoon|fork)", r"provide.*utensil.*soup"),
        "coffee_source": (r"coffee.*(source|ingredient|powder|grounds)", r"source of coffee", r"provide.*coffee"),
        "water_source": (r"water.*(source|ingredient|bottle|jug)", r"source of water", r"provide.*water"),
    },
    "living_room": {
        "PERSONAL_CUP_SAUCER_REGION": (r"(personal|individual|one person|refreshment).*(support|surface|region|setting)", r"support.*(drinkware|cup|saucer|refreshment)"),
        "SHARED_REMOTE_REGION": (r"(shared|both).*(remote|control).*(support|surface|region|accessible)", r"support.*entertainment control"),
        "CUP_SAUCER_SET": (r"(cup|drinkware|refreshment).*(saucer|set|item)", r"contain refreshment item"),
        "REMOTE": (r"remote control|media controller|entertainment control",),
        "SEATING_POSITION": (r"seating (position|location|area)|armchair|seat for",),
        "SEATING_PAIR": (r"seating pair|both seats|two seating|both.*seating",),
    },
    "workshop": {
        "driver": (r"(driving|fastening) tool", r"screwdriver|power drill", r"tool.*(drive|fasten|manipulat)"),
        "fastener": (r"fasten(ing|er)|screw|bolt", r"component.*(secure|thread)"),
        "repair_target": (r"(repair|fastening|marked).*(target|location|hole|joint)", r"target.*(workpiece|workbench)"),
    },
}

_PREDICATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "INSERTABLE_IN": (r"fit(s)? (inside|in|into)", r"insert(able|s|ed)? (in|into)", r"enter(s)? (the )?(opening|cavity)"),
    "REACHES_BOTTOM": (r"reach(es)? (the )?(bottom|interior)", r"long enough.*(bottom|interior)"),
    "FITS_ON": (r"fit(s)? on", r"(support|hold)(s|ing)? .*remote|remote.*(support|hold)"),
    "ACCESSIBLE_FROM_BOTH_SEATS": (r"accessible.*both", r"within reach.*(both|two).*seat"),
    "FITS_SET_ON": (r"fit(s)? .*set.*on", r"(support|hold)(s|ing)? .*(drinkware|cup|saucer|refreshment)", r"place.*(drinkware|refreshment).*surface"),
    "NEAR_SEAT": (r"near (a |the )?seat", r"beside.*seat", r"adjacent.*seat"),
    "COMPATIBLE_WITH": (r"compatible with", r"(drive|manipulat|apply).*component|component.*tool"),
    "REACHES_TARGET": (r"reach(es)? (the )?target", r"tool.*(target|location|hole)", r"capable.*(apply|fasten)"),
    "COMPATIBLE_WITH_TARGET": (r"compatible with", r"component.*(applied|installed|inserted).*target"),
}

_RELATION_SIGNATURES: dict[str, set[tuple[str, str]]] = {
    "INSERTABLE_IN": {("coffee_stirrer", "coffee_container"), ("soup_eating_utensil", "soup_container")},
    "REACHES_BOTTOM": {("coffee_stirrer", "coffee_container"), ("soup_eating_utensil", "soup_container")},
    "FITS_ON": {("SHARED_REMOTE_REGION", "REMOTE")},
    "ACCESSIBLE_FROM_BOTH_SEATS": {("SHARED_REMOTE_REGION", "SEATING_PAIR")},
    "FITS_SET_ON": {("PERSONAL_CUP_SAUCER_REGION", "CUP_SAUCER_SET")},
    "NEAR_SEAT": {("PERSONAL_CUP_SAUCER_REGION", "SEATING_POSITION")},
    "COMPATIBLE_WITH": {("driver", "fastener")},
    "REACHES_TARGET": {("driver", "repair_target")},
    "COMPATIBLE_WITH_TARGET": {("fastener", "repair_target")},
}

_OPERATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "STIR_COFFEE": (r"stir", r"mix.*coffee", r"prepare coffee"),
    "PROVIDE_SOUP_EATING_UTENSIL": (r"serve soup", r"provide.*(soup|eating).*utensil", r"place.*utensil"),
    "SUPPORT_DRINKWARE": (r"support.*(drinkware|refreshment|cup|saucer)", r"place.*(refreshment|drinkware|cup|saucer).*surface"),
}


def prf(predicted: Iterable[Any], reference: Iterable[Any]) -> dict[str, float | None]:
    predicted_counter, reference_counter = Counter(predicted), Counter(reference)
    matches = sum((predicted_counter & reference_counter).values())
    precision = matches / sum(predicted_counter.values()) if predicted_counter else 0.0
    recall = matches / sum(reference_counter.values()) if reference_counter else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else (0.0 if recall is not None else None)
    return {"precision": precision, "recall": recall, "f1": f1}


def _text(item: Mapping[str, Any]) -> str:
    categories = item.get("candidate_categories", ())
    properties = item.get("required_properties", ())
    return " ".join([
        str(item.get("function", "")), str(item.get("description", "")),
        *(str(value) for value in categories if isinstance(categories, list)),
        *(str(value) for value in properties if isinstance(properties, list)),
    ]).lower()


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _map_role(domain: str, role: Mapping[str, Any]) -> str | None:
    text = _text(role)
    candidates = [name for name, patterns in _ROLE_PATTERNS.get(domain, {}).items() if _matches(text, patterns)]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        scores = {
            name: max((len(match.group(0)) for pattern in _ROLE_PATTERNS[domain][name] for match in re.finditer(pattern, text, re.I)), default=0)
            for name in candidates
        }
        best = max(scores.values())
        winners = [name for name, score in scores.items() if score == best]
        return winners[0] if len(winners) == 1 else None
    return None


def _map_phrase(phrase: str, patterns: Mapping[str, tuple[str, ...]]) -> str | None:
    text = phrase.lower()
    scores = {
        name: max((len(match.group(0)) for pattern in choices for match in re.finditer(pattern, text, re.I)), default=0)
        for name, choices in patterns.items()
    }
    best = max(scores.values(), default=0)
    winners = [name for name, score in scores.items() if score == best and score > 0]
    return winners[0] if len(winners) == 1 else None


def _map_relation_phrase(phrase: str, subject: str | None, target: str | None) -> str | None:
    """Map explicit relation language, using endpoint types only to filter supported meanings."""
    text = phrase.lower()
    semantic_candidates = {
        name
        for name, choices in _PREDICATE_PATTERNS.items()
        if any(re.search(pattern, text, re.I) for pattern in choices)
    }
    endpoint_valid = {
        name for name, pairs in _RELATION_SIGNATURES.items()
        if subject is not None and target is not None and (subject, target) in pairs
    }
    matched = semantic_candidates & endpoint_valid
    return next(iter(matched)) if len(matched) == 1 else None


def _normalize_reuse(value: Any) -> str | None:
    aliases = {"REUSABLE_ACROSS_TARGETS": "REUSABLE", "SEQUENTIAL_REUSE_ALLOWED": "REUSABLE", "DEDICATED_PER_TARGET": "DISTINCT"}
    text = str(value) if value is not None else None
    return aliases.get(text, text)


def _binding_equivalent(predicted: Any, expected: Any, count: int) -> bool:
    predicted_norm, expected_norm = _normalize_reuse(predicted), _normalize_reuse(expected)
    # With exactly one physical instance there is no cross-target identity
    # choice, so DISTINCT and REUSABLE denote the same realized binding.
    if count == 1 and {predicted_norm, expected_norm} <= {"DISTINCT", "REUSABLE"}:
        return True
    return predicted_norm == expected_norm


def _reference_operation(operation: Any) -> tuple[Any, ...]:
    return (str(operation.function), str(operation.tool_role), str(operation.target_role), int(operation.required_target_count), _normalize_reuse(operation.usage_policy))


def _predicted_operation(operation: EvaluationOperation, mapped: Mapping[str, str | None]) -> tuple[Any, ...]:
    return (_map_phrase(operation.phrase, _OPERATION_PATTERNS), mapped.get(operation.source_role or ""), mapped.get(operation.target_role or ""), operation.count, _normalize_reuse(operation.reuse_policy))


def evaluate_raw_semantics(domain: str, task: str, raw: Any, reference: Any = None) -> dict[str, Any]:
    """Evaluate semantic content as emitted by the FM, before production repair."""
    from .gt_spec_provider import GTSpecProvider

    if reference is None:
        reference = GTSpecProvider().provide(domain, task)
    contract: EvaluationContract = extract_evaluation_contract(raw)
    mapped: dict[str, str | None] = {}
    predicted_roles: list[str] = []
    for index, role in enumerate(contract.roles):
        semantic_role = _map_role(domain, role)
        mapped[str(role.get("id", f"invalid_{index}"))] = semantic_role
        predicted_roles.append(semantic_role or f"__unmatched_role_{index}")

    predicted_relations: list[tuple[Any, ...]] = []
    for index, relation in enumerate(contract.relations):
        subject = mapped.get(str(relation.get("subject_role", "")))
        target = mapped.get(str(relation.get("object_role", "")))
        predicate = _map_relation_phrase(
            str(relation.get("relation", relation.get("predicate", ""))), subject, target
        )
        predicted_relations.append((subject, predicate, target) if subject and predicate and target else ("__unmatched", str(index), ""))

    predicted_operations = [_predicted_operation(operation, mapped) for operation in contract.operations]
    reference_roles = list(reference.nodes)
    reference_relations = [(relation.subject_role, relation.predicate, relation.object_role) for relation in reference.relations]
    # Relations owned by an operation group are task requirements too, even
    # when the runtime graph stores them on the group rather than as top-level
    # relation records.
    for operation in reference.operation_groups:
        reference_relations.extend(
            (operation.tool_role, predicate, operation.target_role)
            for predicate in operation.required_relations
        )
        if operation.context_role:
            reference_relations.extend(
                (operation.tool_role, predicate, operation.context_role)
                for predicate in operation.context_relations
            )
    reference_relations = list(dict.fromkeys(reference_relations))
    reference_operations = [_reference_operation(operation) for operation in reference.operation_groups]
    role_metrics = prf(predicted_roles, reference_roles)
    relation_metrics = prf(predicted_relations, reference_relations)
    operation_metrics = prf(predicted_operations, reference_operations)

    count_checks, binding_checks = [], []
    for role in contract.roles:
        semantic_role = mapped.get(str(role.get("id", "")))
        if semantic_role in reference.nodes:
            expected = reference.nodes[semantic_role]
            count_checks.append(role.get("required_count") == expected.count)
            binding_checks.append(_binding_equivalent(role.get("binding_policy"), expected.binding_policy, expected.count))
    count_correct = bool(count_checks) and len(count_checks) == len(reference.nodes) and all(count_checks)
    binding_correct = bool(binding_checks) and len(binding_checks) == len(reference.nodes) and all(binding_checks)
    required_scores = (role_metrics["recall"], relation_metrics["recall"], operation_metrics["recall"])
    complete = bool(all(score is None or score == 1.0 for score in required_scores) and count_correct and binding_correct)
    return {
        "schema": contract.schema,
        "role": role_metrics, "relation": relation_metrics, "operation": operation_metrics,
        "group": operation_metrics,
        "count_correct": count_correct, "binding_correct": binding_correct,
        "complete_task_contract": complete,
        "matching_method": "frozen_evaluation_only_semantic_matcher_v2",
        "role_count": len(contract.roles), "relation_count": len(contract.relations), "operation_count": len(contract.operations),
        "raw_id_to_semantic_role": mapped,
    }
