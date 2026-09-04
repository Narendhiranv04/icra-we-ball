from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.artifacts import sha256_text
from mujoco_scenes.baselines.vilain_tamp.contracts import (
    GeneratedPDDLProblem,
    ObjectEstimate,
    ObjectEstimateStatus,
    ProblemSource,
)
from mujoco_scenes.baselines.vilain_tamp.corrective_planning import (
    CorrectiveFailure,
    CorrectiveFailureKind,
    CorrectivePlanningContractError,
    CorrectivePlanningLoop,
    CorrectiveRunStatus,
    TAMPAttemptOutcome,
)
from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.fm import (
    FMRequest,
    FMTransportResponse,
    RecordedFMClient,
)
from mujoco_scenes.baselines.vilain_tamp.planner import (
    NoPlanError,
    PlannerInfrastructureError,
    PlannerTimeoutError,
)
from mujoco_scenes.baselines.vilain_tamp.prompts import (
    build_corrective_planning_prompt,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures"
INITIAL_PROBLEM_PATH = (
    FIXTURE_ROOT / "interpreter" / "kitchen" / "expected_problem.pddl"
)
CORRECTION_ROOT = FIXTURE_ROOT / "corrective_planning"


class FakeTransport:
    def __init__(self, responses: list[str], *, fail: bool = False) -> None:
        self.responses = list(responses)
        self.fail = fail
        self.requests: list[FMRequest] = []

    def complete(self, request: FMRequest) -> FMTransportResponse:
        self.requests.append(request)
        if self.fail:
            raise OSError("mock model transport is offline")
        response = self.responses[len(self.requests) - 1]
        return FMTransportResponse(
            raw_text=response,
            call_id=f"cp-{len(self.requests)}",
            model=request.model,
            revision=request.revision,
            usage={"input_tokens": 20, "output_tokens": 10},
        )


class FakeTAMPAttemptRunner:
    def __init__(
        self,
        *,
        success_attempt: int | None = None,
        raised_error: Exception | None = None,
    ) -> None:
        self.success_attempt = success_attempt
        self.raised_error = raised_error
        self.problems: list[GeneratedPDDLProblem] = []
        self.planner_attempts: list[int] = []
        self.refiner_attempts: list[int] = []

    def attempt(
        self, problem: GeneratedPDDLProblem, output_root: Path
    ) -> TAMPAttemptOutcome:
        del output_root
        if self.raised_error is not None:
            raise self.raised_error
        self.problems.append(problem)
        self.planner_attempts.append(problem.attempt_index)
        self.refiner_attempts.append(problem.attempt_index)
        if problem.attempt_index == self.success_attempt:
            return TAMPAttemptOutcome(
                attempt_index=problem.attempt_index,
                success=True,
                failure=None,
                result_payload={"planner": "mocked", "refiner": "mocked"},
            )
        return TAMPAttemptOutcome(
            attempt_index=problem.attempt_index,
            success=False,
            failure=CorrectiveFailure(
                CorrectiveFailureKind.REFINEMENT,
                f"mock IK failure on attempt {problem.attempt_index}",
                {
                    "stage": "IK",
                    "reason_code": "TARGET_UNREACHABLE",
                    "failed_trace": ["pick-from", "pour"],
                },
            ),
        )


def _read_correction(index: int) -> str:
    return (CORRECTION_ROOT / f"correction_{index}.pddl").read_text(
        encoding="utf-8"
    )


def _object_estimates() -> tuple[ObjectEstimate, ...]:
    return (
        ObjectEstimate(
            object_id="mug_1",
            label="mug",
            pddl_type="vessel",
            description="observed white mug",
            detections=({"camera_id": "front", "xyxy": [1, 2, 10, 12]},),
            estimated_centroid_m=(0.1, 0.2, 0.3),
            centroid_covariance=None,
            observation_stage_ids=("000_initial",),
            status=ObjectEstimateStatus.OBSERVED,
        ),
    )


def _initial_problem() -> GeneratedPDDLProblem:
    domain = load_domain("kitchen")
    text = INITIAL_PROBLEM_PATH.read_text(encoding="utf-8")
    return GeneratedPDDLProblem(
        attempt_index=0,
        source=ProblemSource.INITIAL,
        domain_name=domain.name,
        domain_sha256=domain.sha256,
        problem_text=text,
        declared_objects=(
            "mug_1",
            "coffee_source_1",
            "spoon_1",
            "coffee",
            "counter",
        ),
        initial_atoms=("(handempty)",),
        goal_atoms=("(contains mug_1 coffee)", "(stirred mug_1)"),
        raw_response_artifact="fixtures/initial_problem.pddl",
        problem_sha256=sha256_text(text),
    )


def _run(
    tmp_path: Path,
    *,
    responses: list[str],
    runner: FakeTAMPAttemptRunner,
    max_corrections: int = 3,
    transport_failure: bool = False,
):
    transport = FakeTransport(responses, fail=transport_failure)
    loop = CorrectivePlanningLoop(
        fm_client=RecordedFMClient(transport),
        attempt_runner=runner,
        model="gpt-4o-2024-08-06",
        max_corrections=max_corrections,
    )
    result = loop.run(
        task_instruction="Make coffee and stir it.",
        domain=load_domain("kitchen"),
        object_estimates=_object_estimates(),
        initial_problem=_initial_problem(),
        output_root=tmp_path,
    )
    return result, transport


@pytest.mark.parametrize("success_attempt", [0, 1, 2, 3])
def test_success_is_supported_on_attempt_zero_through_three(
    tmp_path: Path, success_attempt: int
) -> None:
    runner = FakeTAMPAttemptRunner(success_attempt=success_attempt)
    responses = [_read_correction(index) for index in range(1, 4)]

    result, transport = _run(tmp_path, responses=responses, runner=runner)

    assert result.status is CorrectiveRunStatus.SUCCESS
    assert result.selected_problem is not None
    assert result.selected_problem.attempt_index == success_attempt
    assert len(result.tamp_attempts) == success_attempt + 1
    assert len(result.corrections) == success_attempt
    assert len(transport.requests) == success_attempt
    assert runner.planner_attempts == list(range(success_attempt + 1))
    assert runner.refiner_attempts == list(range(success_attempt + 1))


def test_three_corrections_and_four_tamp_attempts_exhaust_the_budget(
    tmp_path: Path,
) -> None:
    runner = FakeTAMPAttemptRunner()
    result, transport = _run(
        tmp_path,
        responses=[_read_correction(index) for index in range(1, 4)],
        runner=runner,
    )

    assert result.status is CorrectiveRunStatus.EXHAUSTED
    assert result.selected_problem is None
    assert len(result.corrections) == 3
    assert len(result.tamp_attempts) == 4
    assert len(transport.requests) == 3


def test_identical_revision_terminates_without_another_tamp_attempt(
    tmp_path: Path,
) -> None:
    runner = FakeTAMPAttemptRunner()
    result, transport = _run(
        tmp_path,
        responses=[INITIAL_PROBLEM_PATH.read_text(encoding="utf-8")],
        runner=runner,
    )

    assert result.status is CorrectiveRunStatus.REPEATED_REVISION
    assert len(result.tamp_attempts) == 1
    assert len(result.corrections) == 1
    assert result.corrections[0].status == "REPEATED_REVISION"
    assert len(transport.requests) == 1


def test_invalid_revision_consumes_a_correction_without_rerunning_tamp(
    tmp_path: Path,
) -> None:
    runner = FakeTAMPAttemptRunner()
    invalid = (CORRECTION_ROOT / "invalid_correction.pddl").read_text(
        encoding="utf-8"
    )
    result, _ = _run(
        tmp_path,
        responses=[invalid],
        runner=runner,
        max_corrections=1,
    )

    assert result.status is CorrectiveRunStatus.EXHAUSTED
    assert len(result.tamp_attempts) == 1
    assert len(result.corrections) == 1
    assert result.corrections[0].status == "INVALID_CORRECTION"
    assert result.corrections[0].revised_problem_sha256 is None
    assert any(
        "unknown predicate" in diagnostic
        for diagnostic in result.corrections[0].validation_diagnostics
    )


def test_each_prompt_contains_complete_prior_problem_and_error_history(
    tmp_path: Path,
) -> None:
    runner = FakeTAMPAttemptRunner(success_attempt=2)
    result, _ = _run(
        tmp_path,
        responses=[_read_correction(1), _read_correction(2)],
        runner=runner,
    )

    request = json.loads(
        (tmp_path / "corrective_planning" / "attempt_02" / "request.json").read_text(
            encoding="utf-8"
        )
    )
    prompt = request["messages"][1]["content"]
    assert _read_correction(1).strip() in prompt
    assert "mock IK failure on attempt 0" in prompt
    assert "failed_trace" in prompt
    assert result.status is CorrectiveRunStatus.SUCCESS
    manifest = json.loads(
        (
            tmp_path
            / "corrective_planning"
            / "attempt_02"
            / "history_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert len(manifest["complete_history"]) == 1


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            PlannerInfrastructureError("Fast Downward missing"),
            CorrectiveRunStatus.INFRASTRUCTURE_ERROR,
        ),
        (
            PlannerTimeoutError("symbolic timeout"),
            CorrectiveRunStatus.SYMBOLIC_TIMEOUT,
        ),
    ],
)
def test_infrastructure_and_timeout_failures_are_terminal_without_cp(
    tmp_path: Path, error: Exception, expected_status: CorrectiveRunStatus
) -> None:
    runner = FakeTAMPAttemptRunner(raised_error=error)
    result, transport = _run(tmp_path, responses=[], runner=runner)

    assert result.status is expected_status
    assert len(result.tamp_attempts) == 1
    assert not result.corrections
    assert not transport.requests


def test_model_transport_failure_is_terminal_infrastructure_error(
    tmp_path: Path,
) -> None:
    runner = FakeTAMPAttemptRunner()
    result, transport = _run(
        tmp_path,
        responses=[],
        runner=runner,
        transport_failure=True,
    )

    assert result.status is CorrectiveRunStatus.INFRASTRUCTURE_ERROR
    assert len(result.tamp_attempts) == 1
    assert not result.corrections
    assert len(transport.requests) == 1


def test_cp_eligibility_is_explicit() -> None:
    eligible = {
        CorrectiveFailureKind.PDDL_INVALID,
        CorrectiveFailureKind.TRANSLATOR,
        CorrectiveFailureKind.NO_PLAN,
        CorrectiveFailureKind.PLAN_VAL,
        CorrectiveFailureKind.ENTITY_RESOLUTION,
        CorrectiveFailureKind.REFINEMENT,
        CorrectiveFailureKind.INVALID_CORRECTION,
    }
    assert {kind for kind in CorrectiveFailureKind if kind.cp_eligible} == eligible


def test_evaluator_data_is_rejected_before_prompt_construction() -> None:
    domain = load_domain("kitchen")
    initial = _initial_problem()
    with pytest.raises(ValueError, match="evaluator data is forbidden"):
        build_corrective_planning_prompt(
            task_instruction="Make coffee.",
            domain=domain,
            object_estimates=_object_estimates(),
            initial_problem=initial.problem_text,
            current_problem=initial.problem_text,
            current_failure={"summary": "failed", "benchmark_outcome": True},
        )


def test_external_method_artifacts_and_mutated_domain_hash_are_rejected(
    tmp_path: Path,
) -> None:
    runner = FakeTAMPAttemptRunner(success_attempt=0)
    loop = CorrectivePlanningLoop(
        fm_client=RecordedFMClient(FakeTransport([])),
        attempt_runner=runner,
        model="gpt-4o-2024-08-06",
    )
    common = {
        "task_instruction": "Make coffee.",
        "domain": load_domain("kitchen"),
        "object_estimates": _object_estimates(),
        "output_root": tmp_path,
    }
    with pytest.raises(CorrectivePlanningContractError, match="external method"):
        loop.run(
            **common,
            initial_problem=_initial_problem(),
            external_method_artifacts={"opaque": "forbidden"},
        )
    with pytest.raises(CorrectivePlanningContractError, match="domain hash"):
        loop.run(
            **common,
            initial_problem=replace(_initial_problem(), domain_sha256="0" * 64),
        )


def test_initial_problem_content_hash_must_match_text(tmp_path: Path) -> None:
    loop = CorrectivePlanningLoop(
        fm_client=RecordedFMClient(FakeTransport([])),
        attempt_runner=FakeTAMPAttemptRunner(success_attempt=0),
        model="gpt-4o-2024-08-06",
    )
    with pytest.raises(CorrectivePlanningContractError, match="content hash"):
        loop.run(
            task_instruction="Make coffee.",
            domain=load_domain("kitchen"),
            object_estimates=_object_estimates(),
            initial_problem=replace(_initial_problem(), problem_sha256="0" * 64),
            output_root=tmp_path,
        )


def test_malformed_initial_problem_can_reach_cp_eligible_attempt_runner(
    tmp_path: Path,
) -> None:
    malformed_text = "(define (problem malformed) (:domain vilain-kitchen)"
    malformed = replace(
        _initial_problem(),
        problem_text=malformed_text,
        problem_sha256=sha256_text(malformed_text),
    )
    runner = FakeTAMPAttemptRunner(success_attempt=0)
    loop = CorrectivePlanningLoop(
        fm_client=RecordedFMClient(FakeTransport([])),
        attempt_runner=runner,
        model="gpt-4o-2024-08-06",
    )

    result = loop.run(
        task_instruction="Make coffee.",
        domain=load_domain("kitchen"),
        object_estimates=_object_estimates(),
        initial_problem=malformed,
        output_root=tmp_path,
    )

    assert result.status is CorrectiveRunStatus.SUCCESS
    assert runner.problems == [malformed]


def test_planner_no_plan_error_is_cp_eligible(tmp_path: Path) -> None:
    runner = FakeTAMPAttemptRunner(raised_error=NoPlanError("search found no plan"))
    result, transport = _run(
        tmp_path,
        responses=[_read_correction(1)],
        runner=runner,
        max_corrections=1,
    )

    assert result.status is CorrectiveRunStatus.EXHAUSTED
    assert len(result.tamp_attempts) == 2
    assert all(
        attempt.failure is not None
        and attempt.failure.kind is CorrectiveFailureKind.NO_PLAN
        for attempt in result.tamp_attempts
    )
    assert len(transport.requests) == 1
