from __future__ import annotations

from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.artifacts import sha256_text
from mujoco_scenes.baselines.vilain_tamp.config import BaselineConfig
from mujoco_scenes.baselines.vilain_tamp.contracts import (
    GeneratedPDDLProblem,
    ProblemSource,
    ValidationStage,
)
from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.planner import (
    FastDownwardPlanner,
    NoPlanError,
    PlanFormatError,
    PlannerInfrastructureError,
    PlannerTimeoutError,
    PlanValidationError,
    TranslatorError,
    VALAdapter,
    parse_plan_file,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "planner"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROBLEM_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "interpreter"
    / "kitchen"
    / "expected_problem.pddl"
)


def generated_problem() -> GeneratedPDDLProblem:
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


def planner(*, timeout_seconds: float = 2.0) -> FastDownwardPlanner:
    return FastDownwardPlanner(
        FIXTURE_ROOT / "fake_fast_downward.py",
        VALAdapter(FIXTURE_ROOT / "fake_val.py", expected_version="4.2.09"),
        expected_version="24.06",
        search_alias="lama-first",
        timeout_seconds=timeout_seconds,
    )


def test_translate_search_multiple_plans_and_val(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_PLAN_FIXTURE", str(FIXTURE_ROOT / "multiple"))
    result = planner().plan(
        problem=generated_problem(),
        domain=load_domain("kitchen"),
        output_root=tmp_path,
    )

    assert result.translation.valid
    assert result.translation.stage is ValidationStage.TRANSLATOR
    assert result.plan_validation.valid
    assert result.plan_validation.stage is ValidationStage.PLAN_VAL
    assert result.plan.plan_cost == 5
    assert result.plan.planner_version == "Fast Downward 24.06"
    assert result.plan.search_configuration == "lama-first"
    assert len(result.plan.raw_plan_artifacts) == 2
    assert [action.action_instance_id for action in result.plan.actions] == [
        "vilain_00_001_pick_from",
        "vilain_00_002_pour",
        "vilain_00_003_place_on",
        "vilain_00_004_pick_from",
        "vilain_00_005_stir",
    ]
    assert result.plan.actions[0].to_dict() == {
        "action_index": 0,
        "action_instance_id": "vilain_00_001_pick_from",
        "operator": "pick-from",
        "arguments": ["coffee_source_1", "counter"],
    }
    assert (tmp_path / "planner" / "translate" / "command.json").is_file()
    assert (tmp_path / "planner" / "search" / "command.json").is_file()
    assert (tmp_path / "planner" / "val" / "run" / "command.json").is_file()


def test_translator_failure_is_structured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_FD_MODE", "translator_error")
    with pytest.raises(TranslatorError) as captured:
        planner().plan(
            problem=generated_problem(),
            domain=load_domain("kitchen"),
            output_root=tmp_path,
        )
    assert captured.value.result.stage is ValidationStage.TRANSLATOR
    assert captured.value.result.exit_code == 2
    assert not captured.value.result.valid
    assert Path(captured.value.result.stderr_artifact).read_text(encoding="utf-8") == (
        FIXTURE_ROOT / "translator_error.txt"
    ).read_text(encoding="utf-8")


def test_completed_search_without_plan_is_reported(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_FD_MODE", "no_plan")
    with pytest.raises(NoPlanError, match="no sas_plan"):
        planner().plan(
            problem=generated_problem(),
            domain=load_domain("kitchen"),
            output_root=tmp_path,
        )
    assert (tmp_path / "planner" / "search" / "stdout.txt").read_text(
        encoding="utf-8"
    ) == (FIXTURE_ROOT / "search_no_plan.txt").read_text(encoding="utf-8")


def test_search_process_failure_is_reported(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_FD_MODE", "search_error")
    with pytest.raises(NoPlanError, match="exited with 3"):
        planner().plan(
            problem=generated_problem(),
            domain=load_domain("kitchen"),
            output_root=tmp_path,
        )
    assert "search: failure" in (
        tmp_path / "planner" / "search" / "stderr.txt"
    ).read_text(encoding="utf-8")


def test_search_timeout_is_terminal_and_recorded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_FD_MODE", "timeout")
    with pytest.raises(PlannerTimeoutError, match="timed out"):
        planner(timeout_seconds=0.2).plan(
            problem=generated_problem(),
            domain=load_domain("kitchen"),
            output_root=tmp_path,
        )
    command = (tmp_path / "planner" / "search" / "command.json").read_text(
        encoding="utf-8"
    )
    assert '"status": "TIMEOUT"' in command


def test_val_invalid_plan_is_structured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_PLAN_FIXTURE", str(FIXTURE_ROOT / "multiple"))
    monkeypatch.setenv("FAKE_VAL_MODE", "invalid")
    with pytest.raises(PlanValidationError) as captured:
        planner().plan(
            problem=generated_problem(),
            domain=load_domain("kitchen"),
            output_root=tmp_path,
        )
    assert captured.value.result.stage is ValidationStage.PLAN_VAL
    assert captured.value.result.exit_code == 0
    assert not captured.value.result.valid
    assert Path(captured.value.result.stdout_artifact).read_text(encoding="utf-8") == (
        FIXTURE_ROOT / "val_invalid.txt"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "line, message",
    [
        ("0: (unknown mug_1)\n", "unknown action"),
        ("0: (pick-from ghost counter)\n", "undeclared objects"),
        ("0: (pick-from mug_1)\n", "expects 2 arguments"),
        (
            "0: (pick-from mug_1 counter)\n0: (pick-from spoon_1 counter)\n",
            "duplicate action index",
        ),
    ],
)
def test_plan_parser_rejects_invalid_actions(
    tmp_path: Path, line: str, message: str
) -> None:
    path = tmp_path / "sas_plan"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(PlanFormatError, match=message):
        parse_plan_file(
            path,
            domain=load_domain("kitchen"),
            declared_objects=generated_problem().declared_objects,
        )


def test_empty_plan_requires_explicit_goal_satisfaction(tmp_path: Path) -> None:
    path = tmp_path / "sas_plan"
    path.write_text("; cost = 0\n", encoding="utf-8")
    with pytest.raises(PlanFormatError, match="initial goal"):
        parse_plan_file(
            path,
            domain=load_domain("kitchen"),
            declared_objects=generated_problem().declared_objects,
        )
    candidate = parse_plan_file(
        path,
        domain=load_domain("kitchen"),
        declared_objects=generated_problem().declared_objects,
        allow_empty=True,
    )
    assert candidate.actions == ()


def test_missing_and_wrong_version_tools_are_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing-fast-downward"
    with pytest.raises(PlannerInfrastructureError, match="missing or not executable"):
        FastDownwardPlanner(missing, VALAdapter(FIXTURE_ROOT / "fake_val.py")).plan(
            problem=generated_problem(),
            domain=load_domain("kitchen"),
            output_root=tmp_path / "missing",
        )
    with pytest.raises(PlannerInfrastructureError, match="mismatch"):
        FastDownwardPlanner(
            FIXTURE_ROOT / "fake_fast_downward.py",
            VALAdapter(FIXTURE_ROOT / "fake_val.py"),
            expected_version="99.99",
        ).plan(
            problem=generated_problem(),
            domain=load_domain("kitchen"),
            output_root=tmp_path / "version",
        )


def test_stored_val_and_no_plan_outputs_are_explicit() -> None:
    assert (FIXTURE_ROOT / "val_valid.txt").read_text(encoding="utf-8") == "Plan valid\n"
    assert "without a plan" in (FIXTURE_ROOT / "search_no_plan.txt").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("name", ["paper_faithful.yaml", "model_matched.yaml"])
def test_stage_six_tool_configuration_is_explicit(name: str) -> None:
    config = BaselineConfig.from_yaml(PACKAGE_ROOT / "configs" / name)
    assert config.external_tools.fast_downward_version == "24.06"
    assert config.external_tools.val_version is None
    assert config.search_configuration == "lama-first"
    assert config.timeouts.symbolic_seconds == 200


def test_planner_does_not_import_common_astar() -> None:
    source = (PACKAGE_ROOT / "planner.py").read_text(encoding="utf-8").lower()
    assert "symbolic_planning_core" not in source
    assert "functional_tamp_pipeline" not in source
    assert "astar" not in source
