from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from mujoco_scenes.baselines.vilain_tamp.artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    sha256_text,
)
from mujoco_scenes.baselines.vilain_tamp.config import (
    BaselineConfig,
    Domain,
    ModelCondition,
    ObservationMode,
)
from mujoco_scenes.baselines.vilain_tamp.contracts import (
    BaselineExecutionPlan,
    CameraFrameArtifacts,
    ExecutionProjection,
    GeneratedPDDLProblem,
    ObjectEstimate,
    ObjectEstimateStatus,
    PDDLValidationResult,
    ProblemSource,
    SymbolicAction,
    SymbolicPlan,
    ValidationStage,
    ViLaInObservation,
)
from mujoco_scenes.baselines.vilain_tamp.corrective_planning import (
    CorrectionAttemptRecord,
    CorrectiveFailure,
    CorrectiveFailureKind,
    CorrectivePlanningResult,
    CorrectiveRunStatus,
    TAMPAttemptOutcome,
)
from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.evaluation import (
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
)
from mujoco_scenes.baselines.vilain_tamp.fm import FMCallRecord, FMCallType
from mujoco_scenes.baselines.vilain_tamp.interpreter import InterpretationResult
from mujoco_scenes.baselines.vilain_tamp.observations import (
    ObservationAcquisitionResult,
)
from mujoco_scenes.baselines.vilain_tamp.runner import (
    BaselineRunner,
    ExecutionStageResult,
    RunOptions,
    RunnerComponents,
)
import mujoco_scenes.baselines.vilain_tamp.runner as runner_module


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "offline_pipeline"
CONFIG_PATH = PACKAGE_ROOT / "configs" / "paper_faithful.yaml"
SCENARIOS = tuple(sorted(FIXTURE_ROOT.glob("*.json")))


class TickClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class MockObservation:
    def __init__(self, domain: Domain, output_root: Path) -> None:
        self.domain = domain
        self.observation_mode = ObservationMode.INITIAL_ONLY
        self.output_root = output_root

    def acquire(self) -> ObservationAcquisitionResult:
        camera_root = self.output_root / "stages/000_initial/cameras/front"
        rgb = atomic_write_text(camera_root / "rgb.png", "offline-rgb\n")
        depth = atomic_write_text(camera_root / "depth.npy", "offline-depth\n")
        calibration = atomic_write_json(
            camera_root / "camera.json",
            {"intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "unit": "meter"},
        )
        frame = CameraFrameArtifacts(
            "front",
            "public offline view",
            "stages/000_initial/cameras/front/rgb.png",
            "stages/000_initial/cameras/front/depth.npy",
            "stages/000_initial/cameras/front/camera.json",
            sha256_file(rgb),
            sha256_file(depth),
            sha256_file(calibration),
        )
        observation = ViLaInObservation(
            self.domain.value,
            self.observation_mode.value,
            "000_initial",
            (frame,),
            None,
            "2026-09-05T00:00:00+00:00",
            None,
            sha256_text(self.domain.value + frame.rgb_sha256),
        )
        trace = atomic_write_json(
            self.output_root / "inspection_trace.json",
            {"domain": self.domain.value, "openings": []},
        )
        manifest = atomic_write_json(
            self.output_root / "observation_manifest.json",
            {"observations": [observation.to_dict()]},
        )
        return ObservationAcquisitionResult((observation,), (), manifest, trace)


class MockInterpreter:
    def interpret(self, *, task_instruction, domain, observations, observation_root, output_root):
        del task_instruction, observations, observation_root
        root = Path(output_root)
        estimate = ObjectEstimate(
            "object_1", "object", _movable_type(domain.key), "offline estimate",
            ({"camera_id": "front", "xyxy": [0, 0, 1, 1]},),
            (0.1, 0.2, 0.3), None, ("000_initial",), ObjectEstimateStatus.OBSERVED,
        )
        call_specs = (
            (FMCallType.OBJECT_ESTIMATION, "perception/request.json", "perception/raw_response.txt", "perception/model_metadata.json", "Qwen2.5-VL-7B-Instruct"),
            (FMCallType.INITIAL_STATE, "interpreter/initial_state_request.json", "interpreter/initial_state_raw.txt", "interpreter/initial_state_model_metadata.json", "gpt-4o-2024-08-06"),
            (FMCallType.GOAL_STATE, "interpreter/goal_request.json", "interpreter/goal_raw.txt", "interpreter/goal_model_metadata.json", "gpt-4o-2024-08-06"),
        )
        calls = []
        for index, (call_type, request_name, response_name, metadata_name, model) in enumerate(call_specs):
            request = atomic_write_json(root / request_name, {"call_type": call_type.value})
            response = atomic_write_text(root / response_name, "offline fixture response\n")
            metadata = atomic_write_json(root / metadata_name, {"model": model})
            calls.append(FMCallRecord(call_type, f"offline-{index}", model, "fixture", str(request), str(response), str(metadata), {"input_tokens": 4, "output_tokens": 2}, 0.01))
        atomic_write_json(root / "perception/object_estimates.json", {"objects": [estimate.to_dict()]})
        problem_text = _problem_text(domain.name, domain.key)
        atomic_write_text(root / "interpreter/initial_state.pddlfrag", "(:init)\n")
        atomic_write_text(root / "interpreter/goal.pddlfrag", "(:goal (and))\n")
        atomic_write_text(root / "interpreter/domain.pddl", domain.text)
        atomic_write_text(root / "interpreter/problem_initial.pddl", problem_text)
        generation = atomic_write_json(root / "interpreter/generation_artifacts.json", {"offline": True})
        problem = GeneratedPDDLProblem(0, ProblemSource.INITIAL, domain.name, domain.sha256, problem_text, ("object_1",), (), ("(offline-goal)",), str(generation), sha256_text(problem_text))
        validation = PDDLValidationResult(True, ValidationStage.INTERNAL, (), 0, None, None, 0.01)
        return InterpretationResult((estimate,), problem, validation, tuple(calls))


class MockCorrectivePlanning:
    def __init__(self, scenario: Mapping[str, Any]) -> None:
        self.scenario = scenario
        self.max_corrections = 3

    def run(self, *, task_instruction, domain, object_estimates, initial_problem, output_root, external_method_artifacts=None):
        del task_instruction, object_estimates, external_method_artifacts
        root = Path(output_root)
        count = int(self.scenario["expected_attempts"])
        corrections = []
        attempts = []
        selected = None
        for index in range(count):
            label = "00_initial" if index == 0 else f"{index:02d}_cp"
            attempt_root = root / "attempts" / label
            problem = replace(initial_problem, attempt_index=index, source=ProblemSource.INITIAL if index == 0 else ProblemSource.CP)
            success = self.scenario["outcome"] != "exhausted_infeasible" and index == count - 1
            _write_attempt_artifacts(attempt_root, problem, success)
            failure = None if success else CorrectiveFailure(CorrectiveFailureKind.REFINEMENT, f"offline refinement failure {index}", {"stage": "IK"})
            payload = _successful_payload(domain.key, index) if success else {"metrics": {"translation_valid": True, "plannable": True, "val_plan_valid": True, "symbolic_plan_length": 1}}
            attempts.append(TAMPAttemptOutcome(index, success, failure, result_payload=payload))
            if success:
                selected = problem
            if index < count - 1:
                cp_root = root / "corrective_planning" / f"attempt_{index + 1:02d}"
                request = atomic_write_json(cp_root / "request.json", {"correction_index": index + 1})
                history = atomic_write_json(cp_root / "history_manifest.json", {"failures": index + 1})
                raw = atomic_write_text(cp_root / "raw_response.txt", problem.problem_text)
                revised = atomic_write_text(cp_root / "revised_problem.pddl", problem.problem_text)
                del history, revised
                corrections.append(CorrectionAttemptRecord(index + 1, initial_problem.problem_sha256, problem.problem_sha256, failure, (), (), "gpt-4o-2024-08-06", str(request), str(raw), problem.problem_sha256, (), "ACCEPTED", {"latency_seconds": 0.02, "input_tokens": 3, "output_tokens": 2, "cost": 0.001}))
        status = CorrectiveRunStatus.SUCCESS if selected else CorrectiveRunStatus.EXHAUSTED
        result = CorrectivePlanningResult(status, selected, tuple(attempts), tuple(corrections), None if selected else attempts[-1].failure)
        atomic_write_json(root / "corrective_planning_result.json", result.to_dict())
        return result


class MockExecution:
    def execute(self, *, domain, variant, execution_plan, projections, output_root):
        del variant, execution_plan
        output_root.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, value in (
            ("entity_resolution.json", {"resolved": True}),
            ("execution_trace.json", {"actions": len(projections)}),
            ("effect_ledger.json", {"effects": _ledger(domain)}),
            ("execution_result.json", {"success": True}),
        ):
            paths[name] = str(atomic_write_json(output_root / name, value))
        return ExecutionStageResult("SUCCESS", True, _terminal_state(domain), tuple(_ledger(domain)), paths, {"controller_action_count": len(projections), "execution_action_failures": 0, "physical_postcondition_failures": 0})

    def terminal_without_execution(self, *, domain, variant, predicted_infeasible):
        del variant
        return TerminalStateSnapshot(domain=domain, predicted_infeasible=predicted_infeasible)


class MockHiddenContext:
    def load(self, domain: str, variant: str) -> HiddenBenchmarkContext:
        if domain == "kitchen":
            requirements = {"coffee_vessels": ["coffee_left", "coffee_right"], "soup_vessels": ["soup_left", "soup_right"], "water_sources": ["kettle"], "coffee_sources": ["coffee_jar"], "suitable_stirrers": ["stirrer"], "suitable_soup_utensils": ["spoon_left", "spoon_right"], "serving_support": "serving_area", "water_content": "water", "coffee_content": "coffee"}
            feasible = True
        elif domain == "living_room":
            requirements = {"left_payloads": ["left_cup", "left_saucer"], "right_payloads": ["right_cup", "right_saucer"], "remote": "remote", "left_support": "left_table", "right_support": "right_table", "shared_support": "coffee_table"}
            feasible = True
        else:
            requirements = {"compatible_drivers": ["manual_driver", "power_driver"], "compatible_fasteners": ["screw"], "target": "repair_joint", "workbench": "main_workbench", "inspection_order": ["left_drawer", "right_drawer", "tool_cabinet"], "storage_contents": {"left_drawer": ["screw"], "right_drawer": ["hammer"], "tool_cabinet": []}, "minimum_insertion_depth_m": 0.012, "maximum_insertion_depth_m": 0.018, "radial_tolerance_m": 0.004, "orientation_tolerance_rad": 0.03}
            feasible = False
        return HiddenBenchmarkContext(domain, variant, feasible, requirements)


class MockGeneratedGoalEvaluator:
    def evaluate(self, *, problem, terminal_state, effect_ledger):
        del problem, terminal_state, effect_ledger
        return {"status": "SATISFIED", "satisfied": True}


def _movable_type(domain: str) -> str:
    return {"kitchen": "vessel", "living_room": "cup", "workshop": "driver"}[domain]


def _problem_text(domain_name: str, domain: str) -> str:
    objects = {"kitchen": "object_1 - vessel", "living_room": "object_1 - cup", "workshop": "object_1 - driver"}[domain]
    return f"(define (problem offline-{domain}) (:domain {domain_name}) (:objects {objects}) (:init) (:goal (and (offline-goal))))\n"


def _successful_payload(domain: str, attempt: int) -> Mapping[str, Any]:
    action = SymbolicAction(0, f"vilain_{attempt:02d}_001_mock", "mock-action", ("object_1",))
    plan = SymbolicPlan(attempt, "Fast Downward", "24.06", "lama-first", (action,), 1.0, 0.03, ("sas_plan",), sha256_text(action.action_instance_id))
    execution_plan = BaselineExecutionPlan(attempt, domain, plan, {"adaptation": "MUJOCO_SEQUENCE_PREFLIGHT"}, (action,))
    projection = ExecutionProjection(action.action_instance_id, action.operator, action.arguments, "MOCK", ("mock_entity",), ("mock_entity",), "OFFLINE_FIXTURE", 1.0, ("offline/evidence.json",))
    return {"execution_plan": execution_plan, "execution_projections": (projection,), "metrics": {"translation_valid": True, "plannable": True, "val_plan_valid": True, "symbolic_plan_length": 1}}


def _write_attempt_artifacts(root: Path, problem: GeneratedPDDLProblem, success: bool) -> None:
    atomic_write_text(root / "problem.pddl", problem.problem_text)
    atomic_write_json(root / "pddl_validation.json", {"valid": True})
    for name, value in (("command.json", "{}\n"), ("stdout.txt", "offline\n"), ("stderr.txt", ""), ("sas_plan", "(mock-action object_1)\n"), ("symbolic_plan.json", "{}\n"), ("plan_validation.json", "{\"valid\": true}\n")):
        atomic_write_text(root / "planner" / name, value)
    atomic_write_json(root / "refinement/refinement.json", {"success": success})
    atomic_write_json(root / "refinement/failures.json", {"failures": [] if success else ["IK"]})
    atomic_write_json(root / "refinement/traces/trace.json", {"offline": True})
    atomic_write_json(root / "execution_projection.json", {"actions": []})


def _on(support: str) -> Mapping[str, Any]:
    return {"present": True, "support": support, "released": True, "stable": True, "inside_support_footprint": True, "support_contact": True, "floor_contact": False, "invalid_penetration": False}


def _terminal_state(domain: str) -> TerminalStateSnapshot:
    if domain == "kitchen":
        objects = {name: _on("serving_area") for name in ("coffee_left", "coffee_right", "soup_left", "soup_right")}
        objects.update({"spoon_left": {"present": True, "contained_stably": True}, "spoon_right": {"present": True, "contained_stably": True}})
        return TerminalStateSnapshot(domain=domain, predicted_infeasible=False, objects=objects, relations={"contained_in": {"soup_left": ["spoon_left"], "soup_right": ["spoon_right"]}})
    objects = {name: _on(support) for name, support in (("left_cup", "left_table"), ("left_saucer", "left_table"), ("right_cup", "right_table"), ("right_saucer", "right_table"), ("remote", "coffee_table"))}
    objects.update({"left_table": {"present": True}, "right_table": {"present": True}, "coffee_table": {"present": True}})
    return TerminalStateSnapshot(domain=domain, predicted_infeasible=False, objects=objects)


def _ledger(domain: str) -> list[Mapping[str, Any]]:
    if domain != "kitchen":
        return []
    rows = []
    for vessel in ("coffee_left", "coffee_right"):
        rows.extend(({"effect": "POUR_COMPLETED", "symbolic_arguments": [source, vessel, content]} for source, content in (("kettle", "water"), ("coffee_jar", "coffee"))))
        rows.append({"effect": "STIR_COMPLETED", "symbolic_arguments": ["stirrer", vessel]})
    return rows


def _expected_files(scenario: Mapping[str, Any]) -> set[str]:
    common = {"baseline_manifest.json", "run_config.json", "events.jsonl", "observations/inspection_trace.json", "observations/observation_manifest.json", "observations/stages/000_initial/cameras/front/rgb.png", "observations/stages/000_initial/cameras/front/depth.npy", "observations/stages/000_initial/cameras/front/camera.json", "perception/request.json", "perception/raw_response.txt", "perception/model_metadata.json", "perception/object_estimates.json", "interpreter/initial_state_request.json", "interpreter/initial_state_raw.txt", "interpreter/initial_state_model_metadata.json", "interpreter/goal_request.json", "interpreter/goal_raw.txt", "interpreter/goal_model_metadata.json", "interpreter/initial_state.pddlfrag", "interpreter/goal.pddlfrag", "interpreter/domain.pddl", "interpreter/problem_initial.pddl", "interpreter/generation_artifacts.json", "corrective_planning_result.json", "benchmark/benchmark_goal_evaluation.json", "metrics.json", "baseline_run_result.json"}
    for index in range(int(scenario["expected_attempts"])):
        label = "00_initial" if index == 0 else f"{index:02d}_cp"
        common.update({f"attempts/{label}/problem.pddl", f"attempts/{label}/pddl_validation.json", f"attempts/{label}/planner/command.json", f"attempts/{label}/planner/stdout.txt", f"attempts/{label}/planner/stderr.txt", f"attempts/{label}/planner/sas_plan", f"attempts/{label}/planner/symbolic_plan.json", f"attempts/{label}/planner/plan_validation.json", f"attempts/{label}/refinement/refinement.json", f"attempts/{label}/refinement/failures.json", f"attempts/{label}/refinement/traces/trace.json", f"attempts/{label}/execution_projection.json"})
        if index:
            common.update({f"corrective_planning/attempt_{index:02d}/request.json", f"corrective_planning/attempt_{index:02d}/history_manifest.json", f"corrective_planning/attempt_{index:02d}/raw_response.txt", f"corrective_planning/attempt_{index:02d}/revised_problem.pddl"})
    if scenario["outcome"] != "exhausted_infeasible":
        common.update({"final_action_plan.json", "execution_projection.json", "execution/entity_resolution.json", "execution/execution_trace.json", "execution/effect_ledger.json", "execution/execution_result.json", "benchmark/generated_goal_evaluation.json"})
    return common


@pytest.mark.parametrize("fixture_path", SCENARIOS, ids=lambda path: path.stem)
def test_fully_mocked_pipeline_artifacts_metrics_and_isolation(monkeypatch, tmp_path: Path, fixture_path: Path) -> None:
    scenario = json.loads(fixture_path.read_text(encoding="utf-8"))
    domain = Domain(scenario["domain"])
    run_root = tmp_path / fixture_path.stem
    config = replace(BaselineConfig.from_yaml(CONFIG_PATH), domain=domain)
    provenance = {"repository_root": str(REPOSITORY_ROOT), "head": "offline-fixture-head", "branch": "naren/ViLaIn-TAMP", "tracked_changes": [], "untracked_paths": ["vilain-tamp.md"], "dirty": True}
    monkeypatch.setattr(runner_module, "repository_provenance", lambda _: provenance)
    result = BaselineRunner(config=config, config_path=CONFIG_PATH, repository_root=REPOSITORY_ROOT, clock=TickClock()).run(
        RunOptions(domain, scenario["variant"], ObservationMode.INITIAL_ONLY, ModelCondition.PAPER_FAITHFUL, run_root, 3, execute=True),
        RunnerComponents("Complete the offline benchmark.", MockObservation(domain, run_root / "observations"), MockInterpreter(), MockCorrectivePlanning(scenario), MockExecution(), MockHiddenContext(), MockGeneratedGoalEvaluator()),
    )
    assert result.run_status == scenario["expected_run_status"]
    actual_files = {path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_file()}
    assert actual_files == _expected_files(scenario)
    manifest = json.loads((run_root / "baseline_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["material_artifacts"]
    assert {entry["path"] for entry in entries} == actual_files - {"baseline_manifest.json"}
    assert all(sha256_file(run_root / entry["path"]) == entry["sha256"] for entry in entries)
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
    required_metrics = {
        "actual_benchmark_success",
        "correct_infeasibility_recognition",
        "generated_goal_satisfied",
        "pddl_extraction_valid",
        "translation_valid",
        "plannable",
        "val_plan_valid",
        "refinement_success",
        "failure_stage_distribution",
        "cp_calls",
        "success_by_cp_iteration",
        "model_calls_by_type",
        "model_usage",
        "api_cost",
        "fm_latency_seconds",
        "symbolic_planning_seconds",
        "geometric_refinement_seconds",
        "execution_seconds",
        "end_to_end_seconds",
        "execution_action_failures",
        "physical_postcondition_failures",
        "entity_resolution_failures",
        "inspected_region_count",
        "inspection_opening_seconds",
        "inspection_travel_seconds",
        "symbolic_plan_length",
        "controller_action_count",
    }
    assert required_metrics <= metrics.keys()
    assert metrics["attempt_count"] == scenario["expected_attempts"]
    assert metrics["cp_calls"] == scenario["expected_cp_calls"]
    assert metrics["model_call_count"] == 3 + scenario["expected_cp_calls"]
    assert metrics["pddl_extraction_valid"] is True
    assert metrics["failure_stage_distribution"] == ({"IK": scenario["expected_attempts"]} if scenario["outcome"] == "exhausted_infeasible" else ({"IK": 1} if scenario["outcome"] == "cp_success" else {}))
    assert manifest["repository"]["branch"] == "naren/ViLaIn-TAMP"
    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert events[0]["event"] == "RUN_STARTED"
    assert events[-1] == {
        "elapsed_seconds": events[-1]["elapsed_seconds"],
        "event": "RUN_COMPLETE",
        "event_index": len(events) - 1,
        "status": scenario["expected_run_status"],
    }
    rendered = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in run_root.rglob("*") if path.is_file())
    assert "functional_tamp_pipeline" not in rendered
    assert "Phase3Handoff" not in rendered
