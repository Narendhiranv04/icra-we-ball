"""Common contracts for independent, post-terminal benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import BenchmarkGoalEvaluation, SerializableContract


class EvaluationContractError(ValueError):
    """Raised when terminal evaluation input is incomplete or inconsistent."""


@dataclass(frozen=True)
class TerminalStateSnapshot(SerializableContract):
    """Neutral physical state captured after baseline execution terminates."""

    domain: str
    predicted_infeasible: bool
    objects: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    relations: Mapping[str, Any] = field(default_factory=dict)
    held_objects: tuple[str, ...] = ()
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise EvaluationContractError("terminal snapshot domain must not be empty")
        if any(not str(object_id).strip() for object_id in self.objects):
            raise EvaluationContractError("terminal snapshot object IDs must not be empty")


@dataclass(frozen=True)
class HiddenBenchmarkContext(SerializableContract):
    """Privileged task truth available only to the terminal evaluator."""

    domain: str
    variant: str
    ground_truth_feasibility: bool
    requirements: Mapping[str, Any]
    evidence_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.domain.strip() or not self.variant.strip():
            raise EvaluationContractError(
                "hidden benchmark domain and variant must not be empty"
            )
        if not isinstance(self.requirements, Mapping):
            raise EvaluationContractError("hidden requirements must be a mapping")


def evaluate_hidden_benchmark(
    terminal_state: TerminalStateSnapshot,
    effect_ledger: Sequence[Mapping[str, Any] | SerializableContract],
    hidden_context: HiddenBenchmarkContext,
) -> BenchmarkGoalEvaluation:
    """Evaluate actual task truth without accepting a generated goal."""
    if not isinstance(terminal_state, TerminalStateSnapshot):
        raise EvaluationContractError(
            "evaluator requires a TerminalStateSnapshot"
        )
    if not isinstance(hidden_context, HiddenBenchmarkContext):
        raise EvaluationContractError(
            "evaluator requires a HiddenBenchmarkContext"
        )
    domain = hidden_context.domain.strip().lower().replace("-", "_")
    if terminal_state.domain.strip().lower().replace("-", "_") != domain:
        raise EvaluationContractError("terminal and hidden-context domains differ")
    normalized_ledger = tuple(_normalize_effect(entry) for entry in effect_ledger)

    if domain == "kitchen":
        from .kitchen import evaluate_kitchen_requirements

        checks = evaluate_kitchen_requirements(
            terminal_state, normalized_ledger, hidden_context
        )
    elif domain == "living_room":
        from .living_room import evaluate_living_room_requirements

        checks = evaluate_living_room_requirements(
            terminal_state, normalized_ledger, hidden_context
        )
    elif domain == "workshop":
        from .workshop import evaluate_workshop_requirements

        checks = evaluate_workshop_requirements(
            terminal_state, normalized_ledger, hidden_context
        )
    else:
        raise EvaluationContractError(f"unsupported benchmark domain: {domain!r}")

    return finalize_evaluation(
        terminal_state=terminal_state,
        hidden_context=hidden_context,
        requirement_checks=checks,
    )


def finalize_evaluation(
    *,
    terminal_state: TerminalStateSnapshot,
    hidden_context: HiddenBenchmarkContext,
    requirement_checks: Sequence[Mapping[str, Any]],
) -> BenchmarkGoalEvaluation:
    """Apply common feasible/infeasible outcome accounting."""
    checks = tuple(dict(check) for check in requirement_checks)
    if any("name" not in check or "passed" not in check for check in checks):
        raise EvaluationContractError(
            "each benchmark requirement check needs name and passed fields"
        )
    requirements_passed = bool(checks) and all(
        check["passed"] is True for check in checks
    )
    feasible = hidden_context.ground_truth_feasibility
    actual_success = feasible and requirements_passed
    predicted_infeasible = terminal_state.predicted_infeasible
    correct_infeasibility = (not feasible) and predicted_infeasible
    benchmark_correct = (
        correct_infeasibility
        if not feasible
        else actual_success and not predicted_infeasible
    )
    return BenchmarkGoalEvaluation(
        domain=hidden_context.domain,
        variant=hidden_context.variant,
        ground_truth_feasibility=feasible,
        requirement_checks=checks,
        actual_task_success=actual_success,
        predicted_infeasible=predicted_infeasible,
        correct_infeasibility_recognition=correct_infeasibility,
        benchmark_outcome_correct=benchmark_correct,
        evidence_artifacts=hidden_context.evidence_artifacts,
    )


def effect_exists(
    effect_ledger: Sequence[Mapping[str, Any]],
    effect: str,
    arguments: Sequence[str],
) -> bool:
    """Return whether a controller-certified effect has the exact symbolic tuple."""
    expected = tuple(arguments)
    return any(
        entry["effect"] == effect
        and tuple(entry["symbolic_arguments"]) == expected
        for entry in effect_ledger
    )


def physical_on(state: Mapping[str, Any], support: str) -> bool:
    """Apply the benchmark's method-independent physical ON criteria."""
    return bool(
        state.get("present", True)
        and state.get("support") == support
        and state.get("released") is True
        and state.get("stable") is True
        and state.get("inside_support_footprint") is True
        and state.get("support_contact") is True
        and state.get("floor_contact") is False
        and state.get("invalid_penetration") is False
    )


def check(name: str, passed: bool, **evidence: Any) -> Mapping[str, Any]:
    """Create one serialization-friendly requirement result."""
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def required_sequence(
    requirements: Mapping[str, Any], key: str
) -> tuple[str, ...]:
    value = requirements.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise EvaluationContractError(
            f"hidden requirement {key!r} must be a non-empty sequence"
        )
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result):
        raise EvaluationContractError(
            f"hidden requirement {key!r} contains an empty ID"
        )
    return result


def required_string(requirements: Mapping[str, Any], key: str) -> str:
    value = requirements.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationContractError(
            f"hidden requirement {key!r} must be a non-empty string"
        )
    return value


def _normalize_effect(
    entry: Mapping[str, Any] | SerializableContract,
) -> Mapping[str, Any]:
    if isinstance(entry, SerializableContract):
        payload = entry.to_dict()
    elif isinstance(entry, Mapping):
        payload = dict(entry)
    else:
        raise EvaluationContractError("effect ledger entries must be mappings")
    effect = payload.get("effect")
    arguments = payload.get("symbolic_arguments")
    if not isinstance(effect, str) or not effect.strip():
        raise EvaluationContractError("effect ledger entry has no effect name")
    if not isinstance(arguments, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in arguments
    ):
        raise EvaluationContractError(
            "effect ledger entry has invalid symbolic arguments"
        )
    return {**payload, "effect": effect, "symbolic_arguments": tuple(arguments)}
