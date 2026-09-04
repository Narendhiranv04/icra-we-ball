"""Bounded, history-preserving corrective planning for ViLaIn-TAMP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from .artifacts import atomic_write_json, atomic_write_text, sha256_text
from .contracts import (
    GeneratedPDDLProblem,
    ObjectEstimate,
    ProblemSource,
    SerializableContract,
)
from .domains.registry import DomainDefinition
from .fm import (
    FMCallRecord,
    FMCallType,
    FMRequest,
    FMTransportError,
    RecordedFMClient,
)
from .pddl import validate_problem
from .planner import (
    NoPlanError,
    PlanFormatError,
    PlannerInfrastructureError,
    PlannerTimeoutError,
    PlanValidationError,
    TranslatorError,
)
from .prompts import build_corrective_planning_prompt


class CorrectivePlanningContractError(ValueError):
    """Raised when CP is given data outside its independent baseline contract."""


class CorrectiveFailureKind(str, Enum):
    PDDL_INVALID = "PDDL_INVALID"
    TRANSLATOR = "TRANSLATOR"
    NO_PLAN = "NO_PLAN"
    PLAN_VAL = "PLAN_VAL"
    ENTITY_RESOLUTION = "ENTITY_RESOLUTION"
    REFINEMENT = "REFINEMENT"
    INVALID_CORRECTION = "INVALID_CORRECTION"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    SYMBOLIC_TIMEOUT = "SYMBOLIC_TIMEOUT"

    @property
    def cp_eligible(self) -> bool:
        return self not in {
            CorrectiveFailureKind.INFRASTRUCTURE,
            CorrectiveFailureKind.SYMBOLIC_TIMEOUT,
        }


class CorrectiveRunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    EXHAUSTED = "EXHAUSTED"
    REPEATED_REVISION = "REPEATED_REVISION"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    SYMBOLIC_TIMEOUT = "SYMBOLIC_TIMEOUT"


@dataclass(frozen=True)
class CorrectiveFailure(SerializableContract):
    kind: CorrectiveFailureKind
    summary: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("corrective failure summary must not be empty")


@dataclass(frozen=True)
class TAMPAttemptOutcome(SerializableContract):
    attempt_index: int
    success: bool
    failure: CorrectiveFailure | None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    result_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        if self.success == (self.failure is not None):
            raise ValueError("attempt success and failure must be mutually exclusive")


@dataclass(frozen=True)
class CorrectionAttemptRecord(SerializableContract):
    correction_index: int
    initial_problem_sha256: str
    prior_problem_sha256: str
    trigger_failure: CorrectiveFailure
    history_problem_hashes: tuple[str, ...]
    history_error_hashes: tuple[str, ...]
    model: str
    request_artifact: str
    raw_response_artifact: str
    revised_problem_sha256: str | None
    validation_diagnostics: tuple[str, ...]
    status: str
    latency_and_usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrectivePlanningResult(SerializableContract):
    status: CorrectiveRunStatus
    selected_problem: GeneratedPDDLProblem | None
    tamp_attempts: tuple[TAMPAttemptOutcome, ...]
    corrections: tuple[CorrectionAttemptRecord, ...]
    terminal_failure: CorrectiveFailure | None


class TAMPAttemptRunner(Protocol):
    """Injected symbolic-planning plus geometric-refinement attempt."""

    def attempt(
        self,
        problem: GeneratedPDDLProblem,
        output_root: Path,
    ) -> TAMPAttemptOutcome: ...


class CorrectivePlanningLoop:
    """Run one initial attempt followed by at most three whole-problem revisions."""

    def __init__(
        self,
        *,
        fm_client: RecordedFMClient,
        attempt_runner: TAMPAttemptRunner,
        model: str,
        model_revision: str | None = None,
        max_corrections: int = 3,
    ) -> None:
        if not 0 <= max_corrections <= 3:
            raise ValueError("max_corrections must be between zero and three")
        if not model.strip():
            raise ValueError("corrective-planning model must not be empty")
        self.fm_client = fm_client
        self.attempt_runner = attempt_runner
        self.model = model
        self.model_revision = model_revision
        self.max_corrections = max_corrections

    def run(
        self,
        *,
        task_instruction: str,
        domain: DomainDefinition,
        object_estimates: Sequence[ObjectEstimate],
        initial_problem: GeneratedPDDLProblem,
        output_root: str | Path,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> CorrectivePlanningResult:
        if external_method_artifacts is not None:
            raise CorrectivePlanningContractError(
                "external method artifacts are not valid corrective-planning input"
            )
        if not task_instruction.strip():
            raise CorrectivePlanningContractError("task instruction must not be empty")
        if not all(isinstance(item, ObjectEstimate) for item in object_estimates):
            raise CorrectivePlanningContractError(
                "corrective planning requires original ObjectEstimate contracts"
            )
        if initial_problem.domain_sha256 != domain.sha256:
            raise CorrectivePlanningContractError("initial problem domain hash mismatch")
        if initial_problem.domain_name != domain.name:
            raise CorrectivePlanningContractError("initial problem domain name mismatch")
        if initial_problem.problem_sha256 != sha256_text(initial_problem.problem_text):
            raise CorrectivePlanningContractError("initial problem content hash mismatch")
        destination = Path(output_root)
        current_problem = initial_problem
        attempts: list[TAMPAttemptOutcome] = []
        corrections: list[CorrectionAttemptRecord] = []
        history: list[dict[str, Any]] = []
        seen_problem_hashes = {_canonical_problem_hash(initial_problem.problem_text)}

        outcome = self._attempt(current_problem, destination)
        attempts.append(outcome)
        if outcome.success:
            return self._finish(
                CorrectiveRunStatus.SUCCESS,
                current_problem,
                attempts,
                corrections,
                None,
                destination,
            )
        failure = outcome.failure
        assert failure is not None

        while True:
            terminal = _terminal_status(failure)
            if terminal is not None:
                return self._finish(
                    terminal, None, attempts, corrections, failure, destination
                )
            if len(corrections) >= self.max_corrections:
                return self._finish(
                    CorrectiveRunStatus.EXHAUSTED,
                    None,
                    attempts,
                    corrections,
                    failure,
                    destination,
                )

            correction_index = len(corrections) + 1
            correction_root = (
                destination / "corrective_planning" / f"attempt_{correction_index:02d}"
            )
            history_problem_hashes = tuple(
                str(entry["raw_revision_sha256"]) for entry in history
            )
            history_error_hashes = tuple(
                str(entry["trigger_failure_sha256"]) for entry in history
            )
            prompt = build_corrective_planning_prompt(
                task_instruction=task_instruction,
                domain=domain,
                object_estimates=object_estimates,
                initial_problem=initial_problem.problem_text,
                current_problem=current_problem.problem_text,
                current_failure=failure.to_dict(),
                correction_history=history,
                prior_problem_hashes=history_problem_hashes,
                prior_error_summaries=tuple(
                    str(entry["trigger_failure"]["summary"]) for entry in history
                ),
            )
            atomic_write_json(
                correction_root / "history_manifest.json",
                {
                    "correction_index": correction_index,
                    "initial_problem_sha256": initial_problem.problem_sha256,
                    "current_problem_sha256": current_problem.problem_sha256,
                    "current_failure": failure.to_dict(),
                    "complete_history": history,
                },
            )
            try:
                response, call = self.fm_client.invoke(
                    FMRequest(
                        call_type=FMCallType.CORRECTIVE_PLANNING,
                        model=self.model,
                        revision=self.model_revision,
                        messages=prompt.messages(),
                        response_format="pddl",
                        metadata={"correction_index": correction_index},
                    ),
                    correction_root,
                )
            except FMTransportError as error:
                infrastructure = CorrectiveFailure(
                    CorrectiveFailureKind.INFRASTRUCTURE,
                    f"corrective-planning model unavailable: {error}",
                )
                return self._finish(
                    CorrectiveRunStatus.INFRASTRUCTURE_ERROR,
                    None,
                    attempts,
                    corrections,
                    infrastructure,
                    destination,
                )

            stripped_problem = response.raw_text.strip()
            raw_problem = stripped_problem + "\n" if stripped_problem else ""
            raw_hash = sha256_text(raw_problem)
            validation = validate_problem(
                raw_problem,
                domain,
                expected_domain_sha256=initial_problem.domain_sha256,
            )
            failure_hash = _failure_hash(failure)
            common_record = {
                "correction_index": correction_index,
                "initial_problem_sha256": initial_problem.problem_sha256,
                "prior_problem_sha256": current_problem.problem_sha256,
                "trigger_failure": failure,
                "history_problem_hashes": history_problem_hashes,
                "history_error_hashes": history_error_hashes,
                "model": call.model,
                "request_artifact": call.request_artifact,
                "raw_response_artifact": call.raw_response_artifact,
                "latency_and_usage": _call_metrics(call),
            }

            if not validation.valid:
                correction = CorrectionAttemptRecord(
                    **common_record,
                    revised_problem_sha256=None,
                    validation_diagnostics=validation.diagnostics,
                    status="INVALID_CORRECTION",
                )
                corrections.append(correction)
                history.append(
                    _history_entry(
                        correction_index=correction_index,
                        prior_problem=current_problem,
                        failure=failure,
                        failure_hash=failure_hash,
                        raw_revision=raw_problem,
                        raw_hash=raw_hash,
                        status=correction.status,
                        diagnostics=validation.diagnostics,
                    )
                )
                failure = CorrectiveFailure(
                    CorrectiveFailureKind.INVALID_CORRECTION,
                    "corrective response is not a valid replacement PDDL problem",
                    {"diagnostics": validation.diagnostics, "raw_response_sha256": raw_hash},
                )
                continue

            canonical_hash = _canonical_problem_hash(raw_problem)
            if canonical_hash in seen_problem_hashes:
                correction = CorrectionAttemptRecord(
                    **common_record,
                    revised_problem_sha256=raw_hash,
                    validation_diagnostics=(),
                    status="REPEATED_REVISION",
                )
                corrections.append(correction)
                repeated = CorrectiveFailure(
                    CorrectiveFailureKind.INVALID_CORRECTION,
                    "corrective response repeats an earlier problem without new information",
                    {"repeated_problem_sha256": raw_hash},
                )
                return self._finish(
                    CorrectiveRunStatus.REPEATED_REVISION,
                    None,
                    attempts,
                    corrections,
                    repeated,
                    destination,
                )

            revised_path = atomic_write_text(
                correction_root / "revised_problem.pddl", raw_problem
            )
            revised = _generated_problem(
                raw_problem,
                correction_index=correction_index,
                domain=domain,
                raw_response_artifact=call.raw_response_artifact,
            )
            correction = CorrectionAttemptRecord(
                **common_record,
                revised_problem_sha256=revised.problem_sha256,
                validation_diagnostics=(),
                status="ACCEPTED",
            )
            corrections.append(correction)
            history.append(
                _history_entry(
                    correction_index=correction_index,
                    prior_problem=current_problem,
                    failure=failure,
                    failure_hash=failure_hash,
                    raw_revision=raw_problem,
                    raw_hash=raw_hash,
                    status=correction.status,
                    diagnostics=(),
                    revised_problem_artifact=str(revised_path),
                )
            )
            seen_problem_hashes.add(canonical_hash)
            current_problem = revised
            outcome = self._attempt(current_problem, destination)
            attempts.append(outcome)
            if outcome.success:
                return self._finish(
                    CorrectiveRunStatus.SUCCESS,
                    current_problem,
                    attempts,
                    corrections,
                    None,
                    destination,
                )
            failure = outcome.failure
            assert failure is not None

    def _attempt(
        self,
        problem: GeneratedPDDLProblem,
        destination: Path,
    ) -> TAMPAttemptOutcome:
        try:
            outcome = self.attempt_runner.attempt(
                problem,
                destination / "attempts" / f"{problem.attempt_index:02d}",
            )
        except PlannerTimeoutError as error:
            return TAMPAttemptOutcome(
                problem.attempt_index,
                False,
                CorrectiveFailure(CorrectiveFailureKind.SYMBOLIC_TIMEOUT, str(error)),
            )
        except TranslatorError as error:
            return TAMPAttemptOutcome(
                problem.attempt_index,
                False,
                CorrectiveFailure(
                    CorrectiveFailureKind.TRANSLATOR,
                    str(error),
                    {"diagnostics": error.result.diagnostics},
                ),
            )
        except NoPlanError as error:
            return TAMPAttemptOutcome(
                problem.attempt_index,
                False,
                CorrectiveFailure(CorrectiveFailureKind.NO_PLAN, str(error)),
            )
        except PlanValidationError as error:
            return TAMPAttemptOutcome(
                problem.attempt_index,
                False,
                CorrectiveFailure(
                    CorrectiveFailureKind.PLAN_VAL,
                    str(error),
                    {"diagnostics": error.result.diagnostics},
                ),
            )
        except PlanFormatError as error:
            return TAMPAttemptOutcome(
                problem.attempt_index,
                False,
                CorrectiveFailure(CorrectiveFailureKind.PDDL_INVALID, str(error)),
            )
        except (PlannerInfrastructureError, OSError) as error:
            return TAMPAttemptOutcome(
                problem.attempt_index,
                False,
                CorrectiveFailure(CorrectiveFailureKind.INFRASTRUCTURE, str(error)),
            )
        except Exception as error:
            return TAMPAttemptOutcome(
                problem.attempt_index,
                False,
                CorrectiveFailure(
                    CorrectiveFailureKind.INFRASTRUCTURE,
                    f"attempt backend raised {type(error).__name__}: {error}",
                ),
            )
        if not isinstance(outcome, TAMPAttemptOutcome):
            raise CorrectivePlanningContractError(
                "attempt runner must return a TAMPAttemptOutcome"
            )
        if outcome.attempt_index != problem.attempt_index:
            raise CorrectivePlanningContractError("attempt runner returned wrong index")
        return outcome

    @staticmethod
    def _finish(
        status: CorrectiveRunStatus,
        selected_problem: GeneratedPDDLProblem | None,
        attempts: Sequence[TAMPAttemptOutcome],
        corrections: Sequence[CorrectionAttemptRecord],
        terminal_failure: CorrectiveFailure | None,
        destination: Path,
    ) -> CorrectivePlanningResult:
        result = CorrectivePlanningResult(
            status,
            selected_problem,
            tuple(attempts),
            tuple(corrections),
            terminal_failure,
        )
        atomic_write_json(
            destination / "corrective_planning_result.json", result.to_dict()
        )
        return result


def _terminal_status(failure: CorrectiveFailure) -> CorrectiveRunStatus | None:
    if failure.kind is CorrectiveFailureKind.SYMBOLIC_TIMEOUT:
        return CorrectiveRunStatus.SYMBOLIC_TIMEOUT
    if not failure.kind.cp_eligible:
        return CorrectiveRunStatus.INFRASTRUCTURE_ERROR
    return None


def _generated_problem(
    text: str,
    *,
    correction_index: int,
    domain: DomainDefinition,
    raw_response_artifact: str,
) -> GeneratedPDDLProblem:
    objects_section = _section(text, ":objects")
    init_section = _section(text, ":init")
    goal_section = _section(text, ":goal")
    return GeneratedPDDLProblem(
        attempt_index=correction_index,
        source=ProblemSource.CP,
        domain_name=domain.name,
        domain_sha256=domain.sha256,
        problem_text=text,
        declared_objects=_declared_object_names(objects_section),
        initial_atoms=_atom_strings(init_section),
        goal_atoms=_atom_strings(goal_section),
        raw_response_artifact=raw_response_artifact,
        problem_sha256=sha256_text(text),
    )


def _section(text: str, head: str) -> str:
    match = re.search(r"\(\s*" + re.escape(head) + r"\b", text, re.IGNORECASE)
    if match is None:
        raise CorrectivePlanningContractError(f"validated problem has no {head} section")
    depth = 0
    for index in range(match.start(), len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise CorrectivePlanningContractError(f"validated problem has unbalanced {head} section")


def _declared_object_names(section: str) -> tuple[str, ...]:
    inner = re.sub(r"^\s*\(\s*:objects\b|\)\s*$", "", section, flags=re.I)
    tokens = re.findall(r"[^\s()]+", inner)
    names: list[str] = []
    pending: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == "-":
            names.extend(pending)
            pending = []
            index += 2
        else:
            pending.append(tokens[index].lower())
            index += 1
    names.extend(pending)
    return tuple(names)


def _atom_strings(section: str) -> tuple[str, ...]:
    atoms = []
    for match in re.finditer(r"\([^()]+\)", section):
        atom = " ".join(match.group(0).lower().split())
        head = atom[1:].split(maxsplit=1)[0].rstrip(")")
        if head not in {":init", ":goal", "and", "or", "not"}:
            atoms.append(atom)
    return tuple(atoms)


def _failure_hash(failure: CorrectiveFailure) -> str:
    return sha256_text(
        json.dumps(failure.to_dict(), sort_keys=True, separators=(",", ":"))
    )


def _canonical_problem_hash(problem_text: str) -> str:
    return sha256_text(problem_text.strip())


def _call_metrics(call: FMCallRecord) -> dict[str, Any]:
    return {
        "call_id": call.call_id,
        "latency_seconds": call.latency_seconds,
        "usage": dict(call.usage),
    }


def _history_entry(
    *,
    correction_index: int,
    prior_problem: GeneratedPDDLProblem,
    failure: CorrectiveFailure,
    failure_hash: str,
    raw_revision: str,
    raw_hash: str,
    status: str,
    diagnostics: Sequence[str],
    revised_problem_artifact: str | None = None,
) -> dict[str, Any]:
    return {
        "correction_index": correction_index,
        "prior_problem": {
            "sha256": prior_problem.problem_sha256,
            "text": prior_problem.problem_text,
        },
        "trigger_failure": failure.to_dict(),
        "trigger_failure_sha256": failure_hash,
        "raw_revision_text": raw_revision,
        "raw_revision_sha256": raw_hash,
        "revised_problem_artifact": revised_problem_artifact,
        "status": status,
        "validation_diagnostics": list(diagnostics),
    }
