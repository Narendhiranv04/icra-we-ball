from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from mujoco_scenes.baselines.vilain_tamp.artifacts import atomic_write_json, sha256_text
from mujoco_scenes.baselines.vilain_tamp.attempt import TAMPAttemptRunner
from mujoco_scenes.baselines.vilain_tamp.contracts import (
    GeneratedPDDLProblem,
    ObjectEstimate,
    ObjectEstimateStatus,
    ProblemSource,
    RefinementFailure,
    RefinementStage,
)
from mujoco_scenes.baselines.vilain_tamp.corrective_planning import (
    CorrectiveFailureKind,
    CorrectivePlanningLoop,
    CorrectiveRunStatus,
)
from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.identity import (
    BaselineIdentityResolver,
    EntityCandidate,
    fixed_entity_binding,
)
from mujoco_scenes.baselines.vilain_tamp.pddl import initial_goal_is_satisfied
from mujoco_scenes.baselines.vilain_tamp.planner import (
    FastDownwardPlanner,
    NoPlanError,
    VALAdapter,
)
from mujoco_scenes.baselines.vilain_tamp.refinement import (
    ADAPTATION_LABEL,
    SequenceRefinementCertificate,
    SequenceRefinementResult,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "planner"
PROBLEM_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "interpreter"
    / "kitchen"
    / "expected_problem.pddl"
)


def _problem() -> GeneratedPDDLProblem:
    domain = load_domain("kitchen")
    text = PROBLEM_PATH.read_text(encoding="utf-8")
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
        initial_atoms=(),
        goal_atoms=(),
        raw_response_artifact="fixture",
        problem_sha256=sha256_text(text),
    )


def _estimate(
    object_id: str, pddl_type: str, centroid: tuple[float, float, float]
) -> ObjectEstimate:
    return ObjectEstimate(
        object_id=object_id,
        label=object_id,
        pddl_type=pddl_type,
        description="independently observed test object",
        detections=(
            {
                "stage_id": "000_initial",
                "camera_id": "front",
                "xyxy": [0, 0, 4, 4],
                "confidence": 0.95,
            },
        ),
        estimated_centroid_m=centroid,
        centroid_covariance=None,
        observation_stage_ids=("000_initial",),
        status=ObjectEstimateStatus.OBSERVED,
    )


def _candidate(
    entity_name: str, pddl_type: str, centroid: tuple[float, float, float]
) -> EntityCandidate:
    return EntityCandidate(
        entity_name=entity_name,
        broad_class=pddl_type,
        compatible_pddl_types=(pddl_type,),
        centroid_m=centroid,
        aabb_min_m=tuple(value - 0.01 for value in centroid),
        aabb_max_m=tuple(value + 0.01 for value in centroid),
        visible_stage_ids=("000_initial",),
        movable=True,
        evidence_artifacts=(f"observations/{entity_name}.json",),
    )


class CandidateProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[GeneratedPDDLProblem, Path]] = []

    def __call__(self, problem: GeneratedPDDLProblem, output_root: Path):
        self.calls.append((problem, output_root))
        return (
            _candidate("scene_coffee", "source", (0.0, 0.0, 0.0)),
            _candidate("scene_mug", "vessel", (1.0, 0.0, 0.0)),
            _candidate("scene_spoon", "utensil", (2.0, 0.0, 0.0)),
        )


class FakeRefiner:
    def __init__(self, failure: RefinementFailure | None = None) -> None:
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    def refine(self, **kwargs: Any) -> SequenceRefinementResult:
        self.calls.append(kwargs)
        kwargs["planning_scene_factory"]()
        if self.failure is not None:
            result = SequenceRefinementResult(False, None, self.failure)
        else:
            result = SequenceRefinementResult(
                True,
                SequenceRefinementCertificate(
                    attempt_index=kwargs["attempt_index"],
                    adaptation_label=ADAPTATION_LABEL,
                    actions=(),
                    final_predicted_state={"preflight": "complete"},
                    elapsed_seconds=0.01,
                ),
                None,
            )
        atomic_write_json(
            Path(kwargs["output_root"]) / "refinement.json", result.to_dict()
        )
        return result


class NoPlanPlanner:
    def plan(self, **kwargs: Any):
        del kwargs
        raise NoPlanError("synthetic search found no plan")


def _runner(
    candidate_provider: CandidateProvider,
    refiner: FakeRefiner,
    *,
    planner: Any | None = None,
) -> TAMPAttemptRunner:
    symbolic_planner = planner or FastDownwardPlanner(
        FIXTURE_ROOT / "fake_fast_downward.py",
        VALAdapter(FIXTURE_ROOT / "fake_val.py", expected_version="4.2.09"),
        expected_version="24.06",
        search_alias="lama-first",
        timeout_seconds=2,
    )
    return TAMPAttemptRunner(
        domain=load_domain("kitchen"),
        object_estimates=(
            _estimate("coffee_source_1", "source", (0.0, 0.0, 0.0)),
            _estimate("mug_1", "vessel", (1.0, 0.0, 0.0)),
            _estimate("spoon_1", "utensil", (2.0, 0.0, 0.0)),
            _estimate("unused_mug", "vessel", (9.0, 0.0, 0.0)),
        ),
        planner=symbolic_planner,
        identity_resolver=BaselineIdentityResolver(
            maximum_distance_m=0.20,
            ambiguity_margin_m=0.01,
        ),
        candidate_provider=candidate_provider,
        fixed_bindings={
            "counter": fixed_entity_binding(
                "counter", "kitchen_counter", broad_class="surface"
            )
        },
        refiner=refiner,
        planning_scene_factory=lambda: {"copy": True},
    )


def test_concrete_attempt_runs_full_pipeline_and_builds_execution_payload(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_PLAN_FIXTURE", str(FIXTURE_ROOT / "multiple"))
    candidates = CandidateProvider()
    refiner = FakeRefiner()
    runner = _runner(candidates, refiner)

    outcome = runner.attempt(_problem(), tmp_path)

    assert outcome.success
    assert outcome.failure is None
    execution_plan = outcome.result_payload["execution_plan"]
    projections = outcome.result_payload["execution_projections"]
    assert execution_plan.symbolic_plan.planner_name == "Fast Downward"
    assert execution_plan.symbolic_plan.search_configuration == "lama-first"
    assert len(execution_plan.normalized_actions) == 5
    assert [item.controller_operator for item in projections] == [
        "PICK", "POUR", "PLACE", "PICK", "STIR"
    ]
    assert projections[0].controller_arguments == ("scene_coffee",)
    assert projections[1].controller_arguments == ("scene_coffee", "scene_mug")
    assert projections[2].controller_arguments == (
        "scene_coffee", "kitchen_counter"
    )
    assert len(candidates.calls) == 1
    assert len(refiner.calls) == 1
    metrics = outcome.result_payload["metrics"]
    assert metrics["translation_valid"] is True
    assert metrics["plannable"] is True
    assert metrics["val_plan_valid"] is True
    assert metrics["symbolic_plan_length"] == 5
    assert metrics["identity_binding_count"] == 3
    assert metrics["refinement_success"] is True
    assert metrics["attempt_seconds"] >= 0
    for path in outcome.artifacts.values():
        assert Path(path).is_file()
    persisted = json.loads((tmp_path / "attempt_outcome.json").read_text())
    assert persisted["success"] is True
    assert persisted["result_payload"]["execution_plan"]["domain"] == "kitchen"
    identity = json.loads(
        (tmp_path / "identity/identity_resolution.json").read_text()
    )
    assert len(identity["movable"]["bindings"]) == 3
    assert identity["fixed_bindings"][0]["object_id"] == "counter"


def test_concrete_runner_is_consumed_directly_by_corrective_loop(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_PLAN_FIXTURE", str(FIXTURE_ROOT / "multiple"))
    runner = _runner(CandidateProvider(), FakeRefiner())

    result = CorrectivePlanningLoop(
        fm_client=object(),  # The successful initial attempt makes no CP call.
        attempt_runner=runner,
        model="gpt-4o-2024-08-06",
    ).run(
        task_instruction="Prepare coffee.",
        domain=load_domain("kitchen"),
        object_estimates=runner.object_estimates,
        initial_problem=_problem(),
        output_root=tmp_path,
    )

    assert result.status is CorrectiveRunStatus.SUCCESS
    assert result.tamp_attempts[0].success
    assert result.tamp_attempts[0].result_payload["execution_plan"].domain == "kitchen"


def test_internal_invalid_problem_stops_before_planner(tmp_path: Path) -> None:
    problem = _problem()
    invalid = replace(
        problem,
        problem_text=problem.problem_text.replace("(:goal", "(:unknown"),
    )
    candidates = CandidateProvider()
    outcome = _runner(candidates, FakeRefiner(), planner=NoPlanPlanner()).attempt(
        invalid, tmp_path
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert outcome.failure.kind is CorrectiveFailureKind.PDDL_INVALID
    assert outcome.failure.details["stage"] == "INTERNAL"
    assert candidates.calls == []


def test_no_plan_is_a_structured_cp_eligible_outcome(tmp_path: Path) -> None:
    candidates = CandidateProvider()
    outcome = _runner(candidates, FakeRefiner(), planner=NoPlanPlanner()).attempt(
        _problem(), tmp_path
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert outcome.failure.kind is CorrectiveFailureKind.NO_PLAN
    assert outcome.failure.kind.cp_eligible
    assert outcome.result_payload["metrics"]["plannable"] is False
    assert candidates.calls == []


def test_missing_required_estimate_is_an_entity_resolution_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_PLAN_FIXTURE", str(FIXTURE_ROOT / "multiple"))
    candidates = CandidateProvider()
    runner = _runner(candidates, FakeRefiner())
    runner.object_estimates = tuple(
        item for item in runner.object_estimates if item.object_id != "spoon_1"
    )

    outcome = runner.attempt(_problem(), tmp_path)

    assert not outcome.success
    assert outcome.failure is not None
    assert outcome.failure.kind is CorrectiveFailureKind.ENTITY_RESOLUTION
    assert outcome.failure.details["reason_code"] == "UNRESOLVED_ENTITY"
    assert candidates.calls == []


def test_refinement_failure_preserves_stage_and_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_PLAN_FIXTURE", str(FIXTURE_ROOT / "multiple"))
    failure = RefinementFailure(
        attempt_index=0,
        action_index=0,
        action_instance_id="vilain_00_001_pick_from",
        operator="pick-from",
        arguments=("coffee_source_1", "counter"),
        stage=RefinementStage.IK,
        reason_code="NO_IK_SOLUTION",
        summary="no collision-free IK solution",
        robot_or_arm="arm",
        involved_entities=("scene_coffee",),
        collision_pair=None,
        numeric_evidence={"candidates": 0.0},
        backend_trace_artifact="traces/ik.json",
        recoverable_by_problem_revision=True,
    )
    outcome = _runner(CandidateProvider(), FakeRefiner(failure)).attempt(
        _problem(), tmp_path
    )

    assert not outcome.success
    assert outcome.failure is not None
    assert outcome.failure.kind is CorrectiveFailureKind.REFINEMENT
    assert outcome.failure.details["stage"] == "IK"
    assert outcome.failure.details["refinement_failure"]["reason_code"] == (
        "NO_IK_SOLUTION"
    )
    assert outcome.result_payload["metrics"]["refinement_success"] is False


def test_initial_goal_evaluation_supports_positive_negative_and_disjunction() -> None:
    problem = _problem().problem_text
    satisfied = problem.replace(
        "(:goal (and (contains mug_1 coffee) (stirred mug_1)))",
        "(:goal (and (handempty) (not (stirred mug_1))))",
    )
    disjunction = problem.replace(
        "(:goal (and (contains mug_1 coffee) (stirred mug_1)))",
        "(:goal (or (stirred mug_1) (handempty)))",
    )
    assert initial_goal_is_satisfied(satisfied)
    assert initial_goal_is_satisfied(disjunction)
    assert not initial_goal_is_satisfied(problem)
