"""Baseline-native orchestration with explicit runtime dependency injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .artifacts import (
    artifact_manifest_entry,
    atomic_write_json,
    build_manifest,
    repository_provenance,
    verify_artifact_manifest,
    verify_repository_provenance,
)
from .config import BaselineConfig, Domain, ModelCondition, ObservationMode
from .contracts import (
    BaselineExecutionPlan,
    BaselineRunResult,
    BenchmarkGoalEvaluation,
    ExecutionProjection,
    GeneratedPDDLProblem,
    SerializableContract,
)
from .corrective_planning import CorrectivePlanningResult, CorrectiveRunStatus
from .domains.registry import DomainDefinition, load_domain
from .evaluation import (
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    evaluate_hidden_benchmark,
)
from .interpreter import InterpretationResult
from .observations import ObservationAcquisitionResult


TARGET_BRANCH = "naren/ViLaIn-TAMP"
LOCAL_PLAN_PATH = "vilain-tamp.md"


class RunnerContractError(ValueError):
    """Raised when orchestration inputs violate the isolated runner contract."""


@dataclass(frozen=True)
class RunOptions(SerializableContract):
    domain: Domain
    variant: str
    observation_mode: ObservationMode
    model_condition: ModelCondition
    output_directory: Path
    cp_limit: int
    execute: bool = False
    offline_fixture_root: Path | None = None
    fast_downward_path: Path | None = None
    val_path: Path | None = None
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not self.variant.strip():
            raise RunnerContractError("variant must not be empty")
        if not 0 <= self.cp_limit <= 3:
            raise RunnerContractError("CP limit must be between zero and three")
        if not str(self.output_directory):
            raise RunnerContractError("output directory must not be empty")
        for label, path in (
            ("offline fixture root", self.offline_fixture_root),
            ("Fast Downward path", self.fast_downward_path),
            ("VAL path", self.val_path),
        ):
            if path is not None and not path.is_absolute():
                raise RunnerContractError(f"{label} must be absolute")


@dataclass(frozen=True)
class ExecutionStageResult(SerializableContract):
    status: str
    success: bool
    terminal_state: TerminalStateSnapshot
    effect_ledger: tuple[Mapping[str, Any] | SerializableContract, ...] = ()
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.status.strip():
            raise RunnerContractError("execution status must not be empty")


class ObservationStage(Protocol):
    domain: Domain
    observation_mode: ObservationMode
    output_root: Path

    def acquire(self) -> ObservationAcquisitionResult: ...


class InterpretationStage(Protocol):
    def interpret(
        self,
        *,
        task_instruction: str,
        domain: DomainDefinition,
        observations: Sequence[Any],
        observation_root: str | Path,
        output_root: str | Path,
    ) -> InterpretationResult: ...


class CorrectivePlanningStage(Protocol):
    max_corrections: int

    def run(
        self,
        *,
        task_instruction: str,
        domain: DomainDefinition,
        object_estimates: Sequence[Any],
        initial_problem: GeneratedPDDLProblem,
        output_root: str | Path,
        external_method_artifacts: Mapping[str, Any] | None = None,
    ) -> CorrectivePlanningResult: ...


class ExecutionStage(Protocol):
    def execute(
        self,
        *,
        domain: str,
        variant: str,
        execution_plan: BaselineExecutionPlan,
        projections: Sequence[ExecutionProjection],
        output_root: Path,
    ) -> ExecutionStageResult: ...

    def terminal_without_execution(
        self,
        *,
        domain: str,
        variant: str,
        predicted_infeasible: bool,
    ) -> TerminalStateSnapshot: ...


class HiddenContextProvider(Protocol):
    def load(self, domain: str, variant: str) -> HiddenBenchmarkContext: ...


class GeneratedGoalEvaluator(Protocol):
    def evaluate(
        self,
        *,
        problem: GeneratedPDDLProblem,
        terminal_state: TerminalStateSnapshot,
        effect_ledger: Sequence[Mapping[str, Any] | SerializableContract],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RunnerComponents:
    """Baseline-owned components configured by a runtime or offline fixture."""

    task_instruction: str
    observation: ObservationStage
    interpreter: InterpretationStage
    corrective_planning: CorrectivePlanningStage
    execution: ExecutionStage | None = None
    hidden_context: HiddenContextProvider | None = None
    generated_goal_evaluator: GeneratedGoalEvaluator | None = None

    def __post_init__(self) -> None:
        if not self.task_instruction.strip():
            raise RunnerContractError("task instruction must not be empty")


class BaselineRunner:
    """Run observation through metrics without accepting prior-run handoffs."""

    def __init__(
        self,
        *,
        config: BaselineConfig,
        config_path: str | Path,
        repository_root: str | Path,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.repository_root = Path(repository_root).resolve()
        self.clock = clock
        if not self.config_path.is_file():
            raise RunnerContractError(f"configuration is missing: {self.config_path}")

    def run(
        self,
        options: RunOptions,
        components: RunnerComponents,
    ) -> BaselineRunResult:
        self._validate_options(options, components)
        run_root = options.output_directory.resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        started = self.clock()
        initial_provenance = repository_provenance(self.repository_root)
        domain = load_domain(options.domain.value)
        run_config_path = atomic_write_json(
            run_root / "run_config.json",
            self._run_configuration(options),
        )
        locked_repository_artifacts = self._locked_repository_artifacts(domain)
        manifest_path = atomic_write_json(
            run_root / "baseline_manifest.json",
            {
                "repository": initial_provenance,
                "configuration": self._run_configuration(options),
                "locked_repository_artifacts": locked_repository_artifacts,
                "adaptation": (
                    "MuJoCo cloned-scene sequence preflight adapted from "
                    "MoveIt Task Constructor refinement"
                ),
                "material_artifacts": [],
            },
        )

        stage_times: dict[str, float] = {}
        stage_started = self.clock()
        observation = components.observation.acquire()
        stage_times["observation_seconds"] = self.clock() - stage_started
        self._validate_observation_result(observation, run_root)

        stage_started = self.clock()
        interpretation = components.interpreter.interpret(
            task_instruction=components.task_instruction,
            domain=domain,
            observations=observation.observations,
            observation_root=run_root / "observations",
            output_root=run_root,
        )
        stage_times["interpretation_seconds"] = self.clock() - stage_started
        self._validate_interpretation(interpretation, domain)

        stage_started = self.clock()
        planning = components.corrective_planning.run(
            task_instruction=components.task_instruction,
            domain=domain,
            object_estimates=interpretation.object_estimates,
            initial_problem=interpretation.problem,
            output_root=run_root,
        )
        stage_times["planning_refinement_cp_seconds"] = (
            self.clock() - stage_started
        )
        if not isinstance(planning, CorrectivePlanningResult):
            raise RunnerContractError(
                "corrective-planning stage returned an invalid result"
            )

        artifact_paths: dict[str, str] = {
            "baseline_manifest": str(manifest_path),
            "run_config": str(run_config_path),
            "observation_manifest": str(observation.manifest_path),
            "inspection_trace": str(observation.inspection_trace_path),
            "initial_problem": str(
                run_root / "interpreter" / "problem_initial.pddl"
            ),
            "interpretation_artifacts": interpretation.problem.raw_response_artifact,
            "corrective_planning_result": str(
                run_root / "corrective_planning_result.json"
            ),
        }
        execution_plan: BaselineExecutionPlan | None = None
        projections: tuple[ExecutionProjection, ...] = ()
        selected_problem = planning.selected_problem
        if planning.status is CorrectiveRunStatus.SUCCESS:
            execution_plan, projections = _selected_attempt_payload(planning)
            final_plan_path = atomic_write_json(
                run_root / "final_action_plan.json", execution_plan.to_dict()
            )
            projection_path = atomic_write_json(
                run_root / "execution_projection.json",
                {"actions": [projection.to_dict() for projection in projections]},
            )
            artifact_paths.update(
                final_action_plan=str(final_plan_path),
                execution_projection=str(projection_path),
            )

        generated_goal_status = "NOT_EVALUATED"
        benchmark_status = "NOT_EVALUATED"
        execution_status = "NOT_REQUESTED_PLANNING_ONLY"
        benchmark_evaluation: BenchmarkGoalEvaluation | None = None
        generated_goal_evaluation: Mapping[str, Any] | None = None
        execution_result: ExecutionStageResult | None = None

        if options.execute:
            if components.execution is None or components.hidden_context is None:
                raise RunnerContractError(
                    "scored execution requires execution and hidden-context adapters"
                )
            if planning.status is CorrectiveRunStatus.SUCCESS:
                if components.generated_goal_evaluator is None:
                    raise RunnerContractError(
                        "scored execution requires a separate generated-goal evaluator"
                    )
                assert execution_plan is not None and selected_problem is not None
                material_entries = self._material_artifacts(run_root, artifact_paths)
                self._verify_execution_boundary(
                    initial_provenance,
                    locked_repository_artifacts,
                    material_entries,
                    run_root,
                )
                stage_started = self.clock()
                execution_result = components.execution.execute(
                    domain=options.domain.value,
                    variant=options.variant,
                    execution_plan=execution_plan,
                    projections=projections,
                    output_root=run_root / "execution",
                )
                stage_times["execution_seconds"] = self.clock() - stage_started
                _validate_execution_result(execution_result, options.domain)
                execution_status = execution_result.status
                artifact_paths.update(execution_result.artifact_paths)
                generated_goal_evaluation = (
                    components.generated_goal_evaluator.evaluate(
                        problem=selected_problem,
                        terminal_state=execution_result.terminal_state,
                        effect_ledger=execution_result.effect_ledger,
                    )
                )
                if not isinstance(generated_goal_evaluation, Mapping):
                    raise RunnerContractError(
                        "generated-goal evaluator returned an invalid result"
                    )
                generated_goal_status = str(
                    generated_goal_evaluation.get("status", "UNKNOWN")
                )
                terminal_state = execution_result.terminal_state
                effect_ledger = execution_result.effect_ledger
            else:
                predicted_infeasible = planning.status in {
                    CorrectiveRunStatus.EXHAUSTED,
                    CorrectiveRunStatus.REPEATED_REVISION,
                }
                terminal_state = components.execution.terminal_without_execution(
                    domain=options.domain.value,
                    variant=options.variant,
                    predicted_infeasible=predicted_infeasible,
                )
                effect_ledger = ()
                execution_status = "NOT_RUN_NO_SELECTED_PLAN"

            hidden_context = components.hidden_context.load(
                options.domain.value, options.variant
            )
            benchmark_evaluation = evaluate_hidden_benchmark(
                terminal_state, effect_ledger, hidden_context
            )
            benchmark_status = (
                "SUCCESS"
                if benchmark_evaluation.benchmark_outcome_correct
                else "FAILURE"
            )
            benchmark_root = run_root / "benchmark"
            if generated_goal_evaluation is not None:
                generated_path = atomic_write_json(
                    benchmark_root / "generated_goal_evaluation.json",
                    generated_goal_evaluation,
                )
                artifact_paths["generated_goal_evaluation"] = str(generated_path)
            hidden_path = atomic_write_json(
                benchmark_root / "benchmark_goal_evaluation.json",
                benchmark_evaluation.to_dict(),
            )
            artifact_paths["benchmark_goal_evaluation"] = str(hidden_path)

        metrics = self._metrics(
            interpretation=interpretation,
            planning=planning,
            execution=execution_result,
            benchmark=benchmark_evaluation,
            generated_goal=generated_goal_evaluation,
            stage_times=stage_times,
            total_seconds=self.clock() - started,
        )
        metrics_path = atomic_write_json(run_root / "metrics.json", metrics)
        artifact_paths["metrics"] = str(metrics_path)
        result_path = run_root / "baseline_run_result.json"
        artifact_paths["baseline_run_result"] = str(result_path)
        run_result = BaselineRunResult(
            run_status=_run_status(
                planning,
                options.execute,
                execution_result,
                benchmark_evaluation,
            ),
            selected_attempt_index=(
                execution_plan.selected_attempt_index
                if execution_plan is not None
                else None
            ),
            planning_status=planning.status.value,
            refinement_status=(
                "SUCCESS" if execution_plan is not None else "NOT_SELECTED"
            ),
            execution_status=execution_status,
            generated_goal_status=generated_goal_status,
            benchmark_status=benchmark_status,
            artifact_paths=artifact_paths,
            metrics=metrics,
        )
        atomic_write_json(result_path, run_result.to_dict())
        material_entries = self._material_artifacts(run_root, artifact_paths)
        atomic_write_json(
            manifest_path,
            {
                "repository": initial_provenance,
                "configuration": self._run_configuration(options),
                "locked_repository_artifacts": locked_repository_artifacts,
                "adaptation": (
                    "MuJoCo cloned-scene sequence preflight adapted from "
                    "MoveIt Task Constructor refinement"
                ),
                "material_artifacts": material_entries,
            },
        )
        return run_result

    def _validate_options(
        self, options: RunOptions, components: RunnerComponents
    ) -> None:
        if options.domain is not self.config.domain:
            raise RunnerContractError("CLI domain differs from resolved configuration")
        if options.observation_mode is not self.config.observation_mode:
            raise RunnerContractError(
                "CLI observation mode differs from resolved configuration"
            )
        if options.model_condition is not self.config.model_condition:
            raise RunnerContractError(
                "CLI model condition differs from resolved configuration"
            )
        if options.cp_limit != self.config.max_cp_corrections:
            raise RunnerContractError("CLI CP limit differs from resolved configuration")
        if components.observation.domain is not options.domain:
            raise RunnerContractError("observation adapter domain mismatch")
        if components.observation.observation_mode is not options.observation_mode:
            raise RunnerContractError("observation adapter mode mismatch")
        expected_observation_root = (
            options.output_directory.resolve() / "observations"
        )
        if components.observation.output_root.resolve() != expected_observation_root:
            raise RunnerContractError("observation adapter output root mismatch")
        if components.corrective_planning.max_corrections != options.cp_limit:
            raise RunnerContractError("corrective-planning CP limit mismatch")

    def _locked_repository_artifacts(
        self, domain: DomainDefinition
    ) -> list[Mapping[str, Any]]:
        paths = (self.config_path, domain.domain_path, domain.knowledge_path)
        return [
            artifact_manifest_entry(path, root=self.repository_root)
            for path in paths
        ]

    def _verify_execution_boundary(
        self,
        initial_provenance: Mapping[str, Any],
        locked_repository_artifacts: Sequence[Mapping[str, Any]],
        material_entries: Sequence[Mapping[str, Any]],
        run_root: Path,
    ) -> None:
        if self.config.require_clean_execution_provenance:
            current = repository_provenance(self.repository_root)
            verify_repository_provenance(
                initial_provenance,
                current,
                required_branch=TARGET_BRANCH,
                allowed_untracked_paths=(LOCAL_PLAN_PATH,),
            )
        verify_artifact_manifest(
            locked_repository_artifacts, root=self.repository_root
        )
        verify_artifact_manifest(material_entries, root=run_root)

    @staticmethod
    def _validate_observation_result(
        result: ObservationAcquisitionResult, run_root: Path
    ) -> None:
        if not isinstance(result, ObservationAcquisitionResult):
            raise RunnerContractError("observation stage returned an invalid result")
        expected_root = run_root / "observations"
        for path in (result.manifest_path, result.inspection_trace_path):
            if not path.is_file() or expected_root not in path.resolve().parents:
                raise RunnerContractError(
                    "observation artifact is missing or outside its stage root"
                )

    @staticmethod
    def _validate_interpretation(
        result: InterpretationResult, domain: DomainDefinition
    ) -> None:
        if not isinstance(result, InterpretationResult):
            raise RunnerContractError("interpreter returned an invalid result")
        if not result.validation.valid:
            raise RunnerContractError("interpreter returned invalid PDDL")
        if result.problem.domain_sha256 != domain.sha256:
            raise RunnerContractError("generated problem domain hash mismatch")

    @staticmethod
    def _material_artifacts(
        run_root: Path, artifact_paths: Mapping[str, str]
    ) -> list[Mapping[str, Any]]:
        files = []
        for label, path in artifact_paths.items():
            candidate = Path(path)
            if not candidate.is_file():
                raise RunnerContractError(
                    f"required run artifact is missing ({label}): {candidate}"
                )
            if candidate.resolve() != (
                run_root / "baseline_manifest.json"
            ).resolve():
                files.append(candidate)
        return build_manifest(files, root=run_root)["artifacts"]

    def _run_configuration(self, options: RunOptions) -> Mapping[str, Any]:
        return {
            **options.to_dict(),
            "object_estimator_model": self.config.object_estimator_model,
            "reasoning_model": self.config.reasoning_model,
            "symbolic_planner": self.config.symbolic_planner,
            "search_configuration": self.config.search_configuration,
            "fast_downward_version": (
                self.config.external_tools.fast_downward_version
            ),
            "val_version": self.config.external_tools.val_version,
            "independent_model_calls": self.config.independent_model_calls,
            "require_clean_execution_provenance": (
                self.config.require_clean_execution_provenance
            ),
            "timeouts": {
                "symbolic_seconds": self.config.timeouts.symbolic_seconds,
                "model_seconds": self.config.timeouts.model_seconds,
                "refinement_seconds": self.config.timeouts.refinement_seconds,
            },
            "planning_only": not options.execute,
        }

    @staticmethod
    def _metrics(
        *,
        interpretation: InterpretationResult,
        planning: CorrectivePlanningResult,
        execution: ExecutionStageResult | None,
        benchmark: BenchmarkGoalEvaluation | None,
        generated_goal: Mapping[str, Any] | None,
        stage_times: Mapping[str, float],
        total_seconds: float,
    ) -> Mapping[str, Any]:
        usage: dict[str, float] = {}
        for call in interpretation.calls:
            for key, value in call.usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0.0) + float(value)
        return {
            **stage_times,
            "end_to_end_seconds": total_seconds,
            "pddl_valid": interpretation.validation.valid,
            "planning_status": planning.status.value,
            "cp_calls": len(planning.corrections),
            "attempt_count": len(planning.tamp_attempts),
            "model_call_count": len(interpretation.calls) + len(planning.corrections),
            "model_usage": usage,
            "execution_success": execution.success if execution else None,
            "generated_goal_status": (
                generated_goal.get("status") if generated_goal else None
            ),
            "generated_goal_satisfied": (
                generated_goal.get("satisfied") if generated_goal else None
            ),
            "actual_benchmark_success": (
                benchmark.actual_task_success if benchmark else None
            ),
            "correct_infeasibility_recognition": (
                benchmark.correct_infeasibility_recognition
                if benchmark
                else None
            ),
            "execution_metrics": dict(execution.metrics) if execution else {},
        }


def _selected_attempt_payload(
    planning: CorrectivePlanningResult,
) -> tuple[BaselineExecutionPlan, tuple[ExecutionProjection, ...]]:
    successful = [outcome for outcome in planning.tamp_attempts if outcome.success]
    if len(successful) != 1:
        raise RunnerContractError(
            "successful corrective result must select exactly one attempt"
        )
    payload = successful[0].result_payload
    execution_plan = payload.get("execution_plan")
    projections = payload.get("execution_projections")
    if not isinstance(execution_plan, BaselineExecutionPlan):
        raise RunnerContractError(
            "successful attempt has no BaselineExecutionPlan payload"
        )
    selected_attempt = successful[0].attempt_index
    if execution_plan.selected_attempt_index != selected_attempt:
        raise RunnerContractError("selected execution-plan attempt index mismatch")
    if (
        planning.selected_problem is None
        or planning.selected_problem.attempt_index != selected_attempt
        or execution_plan.symbolic_plan.attempt_index != selected_attempt
    ):
        raise RunnerContractError("selected problem/plan attempt index mismatch")
    if not isinstance(projections, (tuple, list)) or not all(
        isinstance(item, ExecutionProjection) for item in projections
    ):
        raise RunnerContractError(
            "successful attempt has no execution projection payload"
        )
    projection_tuple = tuple(projections)
    if len(execution_plan.normalized_actions) != len(projection_tuple):
        raise RunnerContractError("execution plan/projection length mismatch")
    for action, projection in zip(
        execution_plan.normalized_actions, projection_tuple
    ):
        if action.action_instance_id != projection.action_instance_id:
            raise RunnerContractError("execution plan/projection identity mismatch")
    return execution_plan, projection_tuple


def _validate_execution_result(
    result: ExecutionStageResult, expected_domain: Domain
) -> None:
    if not isinstance(result, ExecutionStageResult):
        raise RunnerContractError("execution stage returned an invalid result")
    if result.terminal_state.domain != expected_domain.value:
        raise RunnerContractError("execution terminal-state domain mismatch")
    if result.terminal_state.predicted_infeasible:
        raise RunnerContractError(
            "executed plans cannot report predicted-infeasible terminal state"
        )


def _run_status(
    planning: CorrectivePlanningResult,
    execute: bool,
    execution: ExecutionStageResult | None,
    benchmark: BenchmarkGoalEvaluation | None,
) -> str:
    if planning.status is not CorrectiveRunStatus.SUCCESS:
        if benchmark and benchmark.correct_infeasibility_recognition:
            return "INFEASIBLE_CORRECT"
        return planning.status.value
    if not execute:
        return "PLANNING_COMPLETE"
    if execution is not None and not execution.success:
        return "EXECUTION_FAILURE"
    if benchmark is None:
        return "EVALUATION_MISSING"
    return "SUCCESS" if benchmark.benchmark_outcome_correct else "BENCHMARK_FAILURE"
