"""Final-only benchmark context loading and generated-goal evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import GeneratedPDDLProblem, SerializableContract
from .evaluation import HiddenBenchmarkContext, TerminalStateSnapshot
from .evaluation.base import physical_on


class LiveEvaluationError(ValueError):
    """Raised when a final evaluation input is malformed or unsupported."""


class FileHiddenContextProvider:
    """Load privileged benchmark truth from a final-evaluation-only directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def load(self, domain: str, variant: str) -> HiddenBenchmarkContext:
        domain_key = _safe_component(domain, "domain").replace("-", "_")
        variant_key = _safe_component(variant, "variant")
        path = (self.root / domain_key / f"{variant_key}.json").resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise LiveEvaluationError("hidden context path escapes its root") from error
        if not path.is_file():
            raise LiveEvaluationError(f"hidden benchmark context is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LiveEvaluationError(f"unable to read hidden context: {path}") from error
        if not isinstance(payload, Mapping):
            raise LiveEvaluationError("hidden benchmark context must be a mapping")
        if set(payload).difference(
            {"domain", "variant", "ground_truth_feasibility", "requirements", "evidence_artifacts"}
        ):
            raise LiveEvaluationError("hidden benchmark context contains unknown fields")
        feasibility = payload.get("ground_truth_feasibility")
        if not isinstance(feasibility, bool):
            raise LiveEvaluationError(
                "hidden ground_truth_feasibility must be boolean"
            )
        evidence = payload.get("evidence_artifacts", ())
        if not isinstance(evidence, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in evidence
        ):
            raise LiveEvaluationError("hidden evidence_artifacts must be a string list")
        context = HiddenBenchmarkContext(
            domain=str(payload.get("domain", "")),
            variant=str(payload.get("variant", "")),
            ground_truth_feasibility=feasibility,
            requirements=_mapping(payload.get("requirements"), "requirements"),
            evidence_artifacts=tuple(evidence),
        )
        if _domain_key(context.domain) != domain_key or context.variant != variant_key:
            raise LiveEvaluationError("hidden context identity does not match the request")
        return context


class PhysicalGeneratedGoalEvaluator:
    """Evaluate the generated goal against physical state and certified effects."""

    def evaluate(
        self,
        *,
        problem: GeneratedPDDLProblem,
        terminal_state: TerminalStateSnapshot,
        effect_ledger: Sequence[Mapping[str, Any] | SerializableContract],
    ) -> Mapping[str, Any]:
        if not isinstance(problem, GeneratedPDDLProblem):
            raise LiveEvaluationError("generated-goal evaluation requires a generated problem")
        if not isinstance(terminal_state, TerminalStateSnapshot):
            raise LiveEvaluationError("generated-goal evaluation requires terminal physical state")
        effects = tuple(_effect_payload(entry) for entry in effect_ledger)
        checks = []
        for expression in problem.goal_atoms:
            atom = _parse_atom(expression)
            passed, evidence = _evaluate_atom(atom, terminal_state, effects)
            checks.append(
                {
                    "atom": expression,
                    "predicate": atom[0],
                    "arguments": list(atom[1:]),
                    "passed": passed,
                    "physical_evidence": evidence,
                }
            )
        satisfied = bool(checks) and all(check["passed"] for check in checks)
        return {
            "status": "SATISFIED" if satisfied else "NOT_SATISFIED",
            "satisfied": satisfied,
            "problem_sha256": problem.problem_sha256,
            "attempt_index": problem.attempt_index,
            "goal_checks": checks,
            "evaluation_source": "TERMINAL_PHYSICAL_STATE_AND_VERIFIED_EFFECT_LEDGER",
        }


def _evaluate_atom(
    atom: tuple[str, ...],
    state: TerminalStateSnapshot,
    effects: Sequence[Mapping[str, Any]],
) -> tuple[bool, Mapping[str, Any]]:
    predicate, arguments = atom[0], atom[1:]
    objects = state.objects
    relations = state.relations
    if predicate == "handempty" and not arguments:
        return not state.held_objects, {"held_objects": list(state.held_objects)}
    if predicate == "holding" and len(arguments) == 1:
        return arguments[0] in state.held_objects, {"held_objects": list(state.held_objects)}
    if predicate in {"at", "supports"} and len(arguments) == 2:
        payload, support = arguments if predicate == "at" else (arguments[1], arguments[0])
        record = objects.get(payload, {})
        physical_support = objects.get(support, {}).get("entity_name", support)
        contained = _relation_members(relations, "contained_in", support)
        insertion = relations.get("insertion", {})
        physical_payload = objects.get(payload, {}).get("entity_name", payload)
        physical_target = objects.get(support, {}).get("entity_name", support)
        inserted = bool(
            isinstance(insertion, Mapping)
            and insertion.get("fastener") in {payload, physical_payload}
            and insertion.get("target") in {support, physical_target}
        )
        support_passed = any(
            physical_on(record, candidate)
            for candidate in {support, str(physical_support)}
        )
        passed = bool(support_passed or payload in contained or inserted)
        return passed, {"object": payload, "support": record.get("support"), "contained": payload in contained, "inserted": inserted}
    if predicate == "open" and len(arguments) == 1:
        articulation = relations.get("articulation", {})
        record = articulation.get(arguments[0], {}) if isinstance(articulation, Mapping) else {}
        return record.get("open") is True, dict(record) if isinstance(record, Mapping) else {}
    if predicate == "inside" and len(arguments) == 2:
        members = _relation_members(relations, "contained_in", arguments[1])
        record = objects.get(arguments[0], {})
        passed = bool(
            arguments[0] in members
            and record.get("released") is True
            and record.get("stable") is True
        )
        return passed, {"container": arguments[1], "members": list(members), "object_state": dict(record)}
    if predicate == "contains" and len(arguments) == 2:
        matches = _matching_effects(effects, "POUR_COMPLETED", target=arguments[0], tail=arguments[1])
        return bool(matches), {"verified_effects": matches}
    if predicate == "stirred" and len(arguments) == 1:
        matches = _matching_effects(effects, "STIR_COMPLETED", target=arguments[0])
        return bool(matches), {"verified_effects": matches}
    if predicate == "inserted" and len(arguments) == 2:
        insertion = relations.get("insertion", {})
        fastener = objects.get(arguments[0], {}).get("entity_name", arguments[0])
        target = objects.get(arguments[1], {}).get("entity_name", arguments[1])
        passed = bool(
            isinstance(insertion, Mapping)
            and insertion.get("fastener") in {arguments[0], fastener}
            and insertion.get("target") in {arguments[1], target}
            and insertion.get("verified") is True
        )
        return passed, dict(insertion) if isinstance(insertion, Mapping) else {}
    if predicate == "fastened" and len(arguments) == 2:
        matches = [
            effect for effect in effects
            if effect.get("effect") == "DRIVE_COMPLETED"
            and tuple(effect.get("symbolic_arguments", ()))[1:] == arguments
        ]
        return bool(matches), {"verified_effects": matches}
    raise LiveEvaluationError(f"unsupported generated-goal atom: {atom!r}")


def _matching_effects(
    effects: Sequence[Mapping[str, Any]],
    name: str,
    *,
    target: str,
    tail: str | None = None,
) -> list[Mapping[str, Any]]:
    matches = []
    for effect in effects:
        arguments = tuple(effect.get("symbolic_arguments", ()))
        if effect.get("effect") != name or len(arguments) < 2 or arguments[1] != target:
            continue
        if tail is not None and (len(arguments) < 3 or arguments[2] != tail):
            continue
        matches.append(dict(effect))
    return matches


def _relation_members(relations: Mapping[str, Any], name: str, target: str) -> tuple[str, ...]:
    relation = relations.get(name, {})
    if not isinstance(relation, Mapping):
        return ()
    value = relation.get(target, ())
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _effect_payload(entry: Mapping[str, Any] | SerializableContract) -> Mapping[str, Any]:
    if isinstance(entry, SerializableContract):
        payload = entry.to_dict()
    elif isinstance(entry, Mapping):
        payload = dict(entry)
    else:
        raise LiveEvaluationError("effect ledger entry must be serializable")
    if not isinstance(payload.get("effect"), str) or not isinstance(
        payload.get("symbolic_arguments"), (list, tuple)
    ):
        raise LiveEvaluationError("effect ledger entry is malformed")
    return payload


def _parse_atom(expression: str) -> tuple[str, ...]:
    tokens = re.findall(r"[^\s()]+", expression.strip().lower())
    if not tokens or expression.count("(") != 1 or expression.count(")") != 1:
        raise LiveEvaluationError(f"goal atom is malformed: {expression!r}")
    return tuple(tokens)


def _safe_component(value: str, label: str) -> str:
    normalized = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", normalized):
        raise LiveEvaluationError(f"unsafe hidden-context {label}")
    return normalized


def _domain_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveEvaluationError(f"hidden context {label} must be a mapping")
    return dict(value)
