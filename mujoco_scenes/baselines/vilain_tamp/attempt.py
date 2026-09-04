"""Concrete ViLaIn symbolic-to-geometric TAMP attempt orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .artifacts import atomic_write_json, sha256_text
from .contracts import (
    BaselineExecutionPlan,
    ExecutionProjection,
    GeneratedPDDLProblem,
    ObjectEstimate,
    PDDLValidationResult,
    SerializableContract,
    SymbolicAction,
)
from .corrective_planning import (
    CorrectiveFailure,
    CorrectiveFailureKind,
    TAMPAttemptOutcome,
)
from .domains.registry import DomainDefinition
from .execution.base import ProjectionError, project_plan, required_binding_ids
from .identity import (
    BaselineIdentityResolver,
    EntityBinding,
    EntityCandidate,
    IdentityInputError,
    IdentityResolutionError,
)
from .pddl import initial_goal_is_satisfied, validate_problem
from .planner import (
    NoPlanError,
    PlanFormatError,
    PlannerInfrastructureError,
    PlannerTimeoutError,
    PlanningResult,
    PlanValidationError,
    TranslatorError,
)
from .refinement import SequenceRefinementResult


class AttemptContractError(ValueError):
    """Raised when a concrete attempt is configured with invalid inputs."""


class SymbolicPlanner(Protocol):
    def plan(
        self,
        *,
        problem: GeneratedPDDLProblem,
        domain: DomainDefinition,
        output_root: str | Path,
        initial_goal_satisfied: bool = False,
    ) -> PlanningResult: ...


class SequenceRefiner(Protocol):
    def refine(
        self,
        *,
        attempt_index: int,
        actions: Sequence[SymbolicAction],
        projections: Sequence[ExecutionProjection],
        planning_scene_factory: Callable[[], Any],
        output_root: str | Path | None = None,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> SequenceRefinementResult: ...


class CandidateProvider(Protocol):
    def __call__(
        self,
        problem: GeneratedPDDLProblem,
        output_root: Path,
    ) -> Sequence[EntityCandidate]: ...


@dataclass(frozen=True)
class AttemptInputs(SerializableContract):
    domain: str
    problem_sha256: str
    object_estimate_ids: tuple[str, ...]
    fixed_binding_ids: tuple[str, ...]


class TAMPAttemptRunner:
    """Run one complete, independent ViLaIn-TAMP attempt."""

    def __init__(
        self,
        *,
        domain: DomainDefinition,
        object_estimates: Sequence[ObjectEstimate],
        planner: SymbolicPlanner,
        identity_resolver: BaselineIdentityResolver,
        candidate_provider: CandidateProvider,
        fixed_bindings: Mapping[str, EntityBinding],
        refiner: SequenceRefiner,
        planning_scene_factory: Callable[[], Any],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not all(isinstance(item, ObjectEstimate) for item in object_estimates):
            raise AttemptContractError("object estimates must use baseline contracts")
        estimate_ids = [item.object_id for item in object_estimates]
        if len(estimate_ids) != len(set(estimate_ids)):
            raise AttemptContractError("object estimates contain duplicate IDs")
        if any(
            not isinstance(binding, EntityBinding) or key != binding.object_id
            for key, binding in fixed_bindings.items()
        ):
            raise AttemptContractError("fixed bindings must match their symbolic IDs")
        self.domain = domain
        self.object_estimates = tuple(object_estimates)
        self.planner = planner
        self.identity_resolver = identity_resolver
        self.candidate_provider = candidate_provider
        self.fixed_bindings = dict(fixed_bindings)
        self.refiner = refiner
        self.planning_scene_factory = planning_scene_factory
        self.clock = clock

    def attempt(
        self,
        problem: GeneratedPDDLProblem,
        output_root: Path,
    ) -> TAMPAttemptOutcome:
        if not isinstance(problem, GeneratedPDDLProblem):
            raise AttemptContractError("attempt requires GeneratedPDDLProblem")
        destination = Path(output_root)
        destination.mkdir(parents=True, exist_ok=True)
        started = self.clock()
        input_path = atomic_write_json(
            destination / "attempt_inputs.json",
            AttemptInputs(
                domain=self.domain.key,
                problem_sha256=problem.problem_sha256,
                object_estimate_ids=tuple(
                    sorted(item.object_id for item in self.object_estimates)
                ),
                fixed_binding_ids=tuple(sorted(self.fixed_bindings)),
            ).to_dict(),
        )
        artifacts: dict[str, str] = {"attempt_inputs": str(input_path)}
        metrics: dict[str, Any] = {
            "translation_valid": False,
            "plannable": False,
            "val_plan_valid": False,
            "symbolic_plan_length": None,
            "identity_binding_count": 0,
            "refinement_success": False,
        }

        contract_failure = self._problem_contract_failure(problem)
        internal = validate_problem(
            problem.problem_text,
            self.domain,
            expected_domain_sha256=problem.domain_sha256,
        )
        internal_path = atomic_write_json(
            destination / "internal_validation.json", internal.to_dict()
        )
        artifacts["internal_validation"] = str(internal_path)
        if contract_failure is not None or not internal.valid:
            details = {
                "stage": "INTERNAL",
                "diagnostics": tuple(internal.diagnostics)
                + ((contract_failure,) if contract_failure else ()),
            }
            return self._finish(
                problem,
                destination,
                started,
                artifacts,
                metrics,
                failure=CorrectiveFailure(
                    CorrectiveFailureKind.PDDL_INVALID,
                    "generated problem failed internal validation",
                    details,
                ),
            )

        try:
            planning = self.planner.plan(
                problem=problem,
                domain=self.domain,
                output_root=destination,
                initial_goal_satisfied=initial_goal_is_satisfied(problem.problem_text),
            )
        except TranslatorError as error:
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.TRANSLATOR, str(error), error.result,
            )
        except NoPlanError as error:
            metrics["translation_valid"] = True
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.NO_PLAN, str(error), None,
            )
        except PlanValidationError as error:
            metrics["translation_valid"] = True
            metrics["plannable"] = True
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.PLAN_VAL, str(error), error.result,
            )
        except PlanFormatError as error:
            metrics["translation_valid"] = True
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.PDDL_INVALID, str(error), None,
            )
        except PlannerTimeoutError as error:
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.SYMBOLIC_TIMEOUT, str(error), None,
            )
        except (PlannerInfrastructureError, OSError) as error:
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.INFRASTRUCTURE, str(error), None,
            )
        except Exception as error:
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.INFRASTRUCTURE,
                f"symbolic planner raised {type(error).__name__}: {error}",
                None,
            )

        if not isinstance(planning, PlanningResult):
            raise AttemptContractError("planner returned an invalid result contract")
        if not planning.translation.valid:
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.TRANSLATOR,
                "planner returned an invalid translation result",
                planning.translation,
            )
        if not planning.plan_validation.valid:
            metrics["translation_valid"] = True
            metrics["plannable"] = True
            return self._planner_failure(
                problem, destination, started, artifacts, metrics,
                CorrectiveFailureKind.PLAN_VAL,
                "planner returned an invalid VAL result",
                planning.plan_validation,
            )
        if planning.plan.attempt_index != problem.attempt_index:
            raise AttemptContractError("symbolic plan attempt index mismatch")
        normalization_failure = self._normalization_failure(
            planning.plan.actions, problem
        )
        if normalization_failure is not None:
            metrics["translation_valid"] = True
            metrics["plannable"] = True
            metrics["val_plan_valid"] = True
            return self._finish(
                problem, destination, started, artifacts, metrics,
                failure=CorrectiveFailure(
                    CorrectiveFailureKind.PDDL_INVALID,
                    normalization_failure,
                    {
                        "stage": "SYMBOLIC_NORMALIZATION",
                        "reason_code": "INVALID_SYMBOLIC_ACTION",
                    },
                ),
            )
        metrics.update(
            {
                "translation_valid": planning.translation.valid,
                "plannable": True,
                "val_plan_valid": planning.plan_validation.valid,
                "symbolic_plan_length": len(planning.plan.actions),
                "symbolic_planning_seconds": planning.plan.planner_time_seconds,
            }
        )
        symbolic_plan_path = destination / "planner" / "symbolic_plan.json"
        if not symbolic_plan_path.is_file():
            raise AttemptContractError("planner did not persist symbolic_plan.json")
        artifacts["symbolic_plan"] = str(symbolic_plan_path)

        try:
            required_ids = set(
                required_binding_ids(self.domain.key, planning.plan.actions)
            )
        except ProjectionError as error:
            return self._finish(
                problem, destination, started, artifacts, metrics,
                failure=CorrectiveFailure(
                    CorrectiveFailureKind.REFINEMENT,
                    str(error),
                    {
                        "stage": "EXECUTION_PROJECTION",
                        "reason_code": type(error).__name__,
                    },
                ),
            )

        try:
            estimates_by_id = {item.object_id: item for item in self.object_estimates}
            movable_ids = required_ids.difference(self.fixed_bindings)
            missing = sorted(movable_ids.difference(estimates_by_id))
            if missing:
                raise IdentityResolutionError(
                    "UNRESOLVED_ENTITY",
                    f"no object estimate exists for required IDs {missing!r}",
                    object_ids=missing,
                )
            try:
                candidates = (
                    tuple(self.candidate_provider(problem, destination / "identity"))
                    if movable_ids
                    else ()
                )
            except Exception as error:
                return self._finish(
                    problem, destination, started, artifacts, metrics,
                    failure=CorrectiveFailure(
                        CorrectiveFailureKind.INFRASTRUCTURE,
                        f"entity candidate provider raised {type(error).__name__}: {error}",
                        {
                            "stage": "ENTITY_RESOLUTION",
                            "reason_code": "CANDIDATE_PROVIDER_ERROR",
                        },
                    ),
                )
            if not all(isinstance(item, EntityCandidate) for item in candidates):
                raise IdentityInputError(
                    "candidate provider must return EntityCandidate contracts"
                )
            identity = self.identity_resolver.resolve(
                tuple(estimates_by_id[item] for item in sorted(movable_ids)),
                candidates,
            )
            identity_path = atomic_write_json(
                destination / "identity" / "identity_resolution.json",
                {
                    "required_binding_ids": sorted(required_ids),
                    "movable": identity.to_dict(),
                    "fixed_bindings": [
                        self.fixed_bindings[key].to_dict()
                        for key in sorted(required_ids.intersection(self.fixed_bindings))
                    ],
                },
            )
            artifacts["identity_resolution"] = str(identity_path)
            metrics["identity_binding_count"] = len(identity.bindings)
        except IdentityResolutionError as error:
            return self._finish(
                problem, destination, started, artifacts, metrics,
                failure=CorrectiveFailure(
                    CorrectiveFailureKind.ENTITY_RESOLUTION,
                    str(error),
                    {
                        "stage": "ENTITY_RESOLUTION",
                        "reason_code": error.reason_code,
                        "object_ids": error.object_ids,
                        "candidate_entities": error.candidate_entities,
                    },
                ),
            )
        except IdentityInputError as error:
            return self._finish(
                problem, destination, started, artifacts, metrics,
                failure=CorrectiveFailure(
                    CorrectiveFailureKind.ENTITY_RESOLUTION,
                    str(error),
                    {"stage": "ENTITY_RESOLUTION", "reason_code": type(error).__name__},
                ),
            )

        try:
            projections = project_plan(
                self.domain.key,
                planning.plan.actions,
                identity.by_object_id(),
                fixed_bindings=self.fixed_bindings,
            )
        except ProjectionError as error:
            unresolved = "UNRESOLVED_ENTITY" in str(error)
            return self._finish(
                problem, destination, started, artifacts, metrics,
                failure=CorrectiveFailure(
                    (
                        CorrectiveFailureKind.ENTITY_RESOLUTION
                        if unresolved
                        else CorrectiveFailureKind.REFINEMENT
                    ),
                    str(error),
                    {
                        "stage": "EXECUTION_PROJECTION",
                        "reason_code": type(error).__name__,
                    },
                ),
            )

        projections_path = atomic_write_json(
            destination / "execution_projections.json",
            {"projections": [item.to_dict() for item in projections]},
        )
        artifacts["execution_projections"] = str(projections_path)
        try:
            refinement = self.refiner.refine(
                attempt_index=problem.attempt_index,
                actions=planning.plan.actions,
                projections=projections,
                planning_scene_factory=self.planning_scene_factory,
                output_root=destination,
            )
        except Exception as error:
            return self._finish(
                problem, destination, started, artifacts, metrics,
                failure=CorrectiveFailure(
                    CorrectiveFailureKind.INFRASTRUCTURE,
                    f"sequence refiner raised {type(error).__name__}: {error}",
                    {
                        "stage": "REFINEMENT",
                        "reason_code": "REFINER_ERROR",
                    },
                ),
            )
        refinement_path = destination / "refinement.json"
        if not refinement_path.is_file():
            raise AttemptContractError("refiner did not persist refinement.json")
        artifacts["refinement"] = str(refinement_path)
        if not isinstance(refinement, SequenceRefinementResult):
            raise AttemptContractError("refiner returned an invalid result contract")
        if not refinement.success:
            assert refinement.failure is not None
            return self._finish(
                problem, destination, started, artifacts, metrics,
                failure=CorrectiveFailure(
                    CorrectiveFailureKind.REFINEMENT,
                    refinement.failure.summary,
                    {
                        "stage": refinement.failure.stage.value,
                        "reason_code": refinement.failure.reason_code,
                        "refinement_failure": refinement.failure.to_dict(),
                    },
                ),
            )

        assert refinement.certificate is not None
        metrics["refinement_success"] = True
        execution_plan = BaselineExecutionPlan(
            selected_attempt_index=problem.attempt_index,
            domain=self.domain.key,
            symbolic_plan=planning.plan,
            refinement_certificate=refinement.certificate.to_dict(),
            normalized_actions=planning.plan.actions,
        )
        execution_plan_path = atomic_write_json(
            destination / "execution_plan.json", execution_plan.to_dict()
        )
        artifacts["execution_plan"] = str(execution_plan_path)
        return self._finish(
            problem,
            destination,
            started,
            artifacts,
            metrics,
            result_payload={
                "execution_plan": execution_plan,
                "execution_projections": projections,
            },
        )

    def _problem_contract_failure(
        self, problem: GeneratedPDDLProblem
    ) -> str | None:
        if problem.domain_name != self.domain.name:
            return "problem domain name differs from the configured domain"
        if problem.domain_sha256 != self.domain.sha256:
            return "problem domain hash differs from the configured domain"
        if problem.problem_sha256 != sha256_text(problem.problem_text):
            return "problem content hash differs from its contract"
        return None

    def _normalization_failure(
        self,
        actions: Sequence[SymbolicAction],
        problem: GeneratedPDDLProblem,
    ) -> str | None:
        instance_ids: set[str] = set()
        declared = set(problem.declared_objects)
        for expected_index, action in enumerate(actions):
            if not isinstance(action, SymbolicAction):
                return "symbolic plan contains a non-SymbolicAction value"
            if action.action_index != expected_index:
                return "symbolic action indices are not contiguous from zero"
            if not action.action_instance_id.strip():
                return "symbolic action instance ID is empty"
            if action.action_instance_id in instance_ids:
                return "symbolic action instance IDs are not unique"
            instance_ids.add(action.action_instance_id)
            operator = action.operator.strip()
            if operator != operator.lower().replace("_", "-"):
                return "symbolic action operator is not normalized"
            signature = self.domain.action_signatures.get(operator)
            if signature is None or len(action.arguments) != len(signature):
                return "symbolic action does not match the fixed domain signature"
            if any(
                not argument.strip()
                or argument != argument.lower()
                or argument not in declared
                for argument in action.arguments
            ):
                return "symbolic action arguments are not normalized declared objects"
        return None

    def _planner_failure(
        self,
        problem: GeneratedPDDLProblem,
        destination: Path,
        started: float,
        artifacts: Mapping[str, str],
        metrics: Mapping[str, Any],
        kind: CorrectiveFailureKind,
        summary: str,
        validation: PDDLValidationResult | None,
    ) -> TAMPAttemptOutcome:
        details: dict[str, Any] = {"stage": kind.value}
        if validation is not None:
            details["validation"] = validation.to_dict()
        return self._finish(
            problem,
            destination,
            started,
            artifacts,
            metrics,
            failure=CorrectiveFailure(kind, summary, details),
        )

    def _finish(
        self,
        problem: GeneratedPDDLProblem,
        destination: Path,
        started: float,
        artifacts: Mapping[str, str],
        metrics: Mapping[str, Any],
        *,
        failure: CorrectiveFailure | None = None,
        result_payload: Mapping[str, Any] | None = None,
    ) -> TAMPAttemptOutcome:
        payload = dict(result_payload or {})
        payload["metrics"] = {
            **dict(metrics),
            "attempt_seconds": self.clock() - started,
        }
        outcome = TAMPAttemptOutcome(
            attempt_index=problem.attempt_index,
            success=failure is None,
            failure=failure,
            artifacts=dict(artifacts),
            result_payload=payload,
        )
        atomic_write_json(destination / "attempt_outcome.json", outcome.to_dict())
        return outcome
