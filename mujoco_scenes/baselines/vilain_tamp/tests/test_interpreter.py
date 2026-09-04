from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mujoco_scenes.baselines.vilain_tamp.config import Domain, ObservationMode
from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.fm import (
    FMCallType,
    FMRequest,
    FMTransportResponse,
    RecordedFMClient,
)
from mujoco_scenes.baselines.vilain_tamp.interpreter import (
    InterpreterModels,
    InterpreterOutputError,
    ViLaInInterpreter,
    normalize_object_estimates,
)
from mujoco_scenes.baselines.vilain_tamp.observations import (
    CameraFrameCapture,
    ObservationProtocol,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interpreter"
INTRINSICS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
EXTRINSICS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FakeCaptureBackend:
    def capture(self, camera_id: str, stage_id: str) -> CameraFrameCapture:
        del stage_id
        return CameraFrameCapture(
            camera_id=camera_id,
            view_description="Synthetic public front view",
            rgb_png=b"\x89PNG\r\n\x1a\nsynthetic",
            depth_m=np.full((4, 4), 2.0, dtype=np.float32),
            intrinsics=INTRINSICS,
            extrinsics=EXTRINSICS,
        )


class FixtureTransport:
    def __init__(self, fixture_dir: Path, *, initial_override: str | None = None):
        self.fixture_dir = fixture_dir
        self.initial_override = initial_override
        self.calls: list[FMCallType] = []

    def complete(self, request: FMRequest) -> FMTransportResponse:
        self.calls.append(request.call_type)
        if request.call_type is FMCallType.OBJECT_ESTIMATION:
            raw_text = (self.fixture_dir / "objects.json").read_text(encoding="utf-8")
        elif request.call_type is FMCallType.INITIAL_STATE:
            raw_text = self.initial_override or (
                self.fixture_dir / "initial.pddlfrag"
            ).read_text(encoding="utf-8")
        elif request.call_type is FMCallType.GOAL_STATE:
            raw_text = (self.fixture_dir / "goal.pddlfrag").read_text(encoding="utf-8")
        else:
            raise AssertionError(f"unexpected call type: {request.call_type}")
        return FMTransportResponse(
            raw_text=raw_text,
            call_id=f"fixture-{len(self.calls)}",
            model=request.model,
            revision=request.revision,
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def acquire_observation(tmp_path: Path, domain_key: str):
    protocol = ObservationProtocol(
        domain=Domain(domain_key),
        observation_mode=ObservationMode.INITIAL_ONLY,
        camera_ids=("front",),
        output_root=tmp_path,
        capture_backend=FakeCaptureBackend(),
        clock=lambda: "2026-09-04T00:00:00+00:00",
    )
    return protocol.acquire().observations


def make_interpreter(transport: FixtureTransport) -> ViLaInInterpreter:
    client = RecordedFMClient(transport)
    return ViLaInInterpreter(
        object_client=client,
        reasoning_client=client,
        models=InterpreterModels(
            object_estimator_model="Qwen2.5-VL-7B-Instruct",
            object_estimator_revision="fixture-revision",
            reasoning_model="gpt-4o-2024-08-06",
            reasoning_model_revision=None,
        ),
    )


@pytest.mark.parametrize(
    "domain_key, expected_ids",
    [
        ("kitchen", ("coffee_source_1", "mug_1", "spoon_1")),
        ("living_room", ("cup_1", "side_table_1")),
        ("workshop", ("screw_1", "screwdriver_1")),
    ],
)
def test_fixture_pipeline_builds_deterministic_valid_problem(
    tmp_path: Path,
    domain_key: str,
    expected_ids: tuple[str, ...],
) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, domain_key)
    fixture_dir = FIXTURE_ROOT / domain_key
    transport = FixtureTransport(fixture_dir)
    result = make_interpreter(transport).interpret(
        task_instruction="Complete the fixed benchmark task.",
        domain=load_domain(domain_key),
        observations=observations,
        observation_root=observation_root,
        output_root=tmp_path / "run",
    )

    assert result.validation.valid
    assert tuple(item.object_id for item in result.object_estimates) == expected_ids
    assert result.problem.problem_text == (
        fixture_dir / "expected_problem.pddl"
    ).read_text(encoding="utf-8")
    assert len(result.problem.problem_sha256) == 64
    assert result.problem.domain_sha256 == load_domain(domain_key).sha256
    assert [call.call_type for call in result.calls] == [
        FMCallType.OBJECT_ESTIMATION,
        FMCallType.INITIAL_STATE,
        FMCallType.GOAL_STATE,
    ]
    assert transport.calls == [
        FMCallType.OBJECT_ESTIMATION,
        FMCallType.INITIAL_STATE,
        FMCallType.GOAL_STATE,
    ]
    assert all(item.estimated_centroid_m is not None for item in result.object_estimates)
    assert (tmp_path / "run" / "perception" / "object_estimates.json").is_file()
    assert (tmp_path / "run" / "interpreter" / "problem_initial.pddl").is_file()


def test_same_label_ids_follow_spatial_order_not_response_order(tmp_path: Path) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    raw = """
    {"objects": [
      {"label": "mug", "pddl_type": "vessel", "description": "right",
       "detections": [{"stage_id": "000_initial", "camera_id": "front", "xyxy": [2, 0, 4, 2], "confidence": 0.9}]},
      {"label": "Mug", "pddl_type": "vessel", "description": "left",
       "detections": [{"stage_id": "000_initial", "camera_id": "front", "xyxy": [0, 0, 2, 2], "confidence": 0.9}]}
    ]}
    """
    estimates = normalize_object_estimates(
        raw,
        domain=load_domain("kitchen"),
        observations=observations,
        observation_root=observation_root,
    )
    assert [(item.object_id, item.description) for item in estimates] == [
        ("mug_1", "left"),
        ("mug_2", "right"),
    ]


def test_unknown_object_type_is_rejected(tmp_path: Path) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    raw = """
    {"objects": [{"label": "mug", "pddl_type": "secret_type",
      "detections": [{"stage_id": "000_initial", "camera_id": "front", "xyxy": [0, 0, 2, 2], "confidence": 0.9}]}]}
    """
    with pytest.raises(InterpreterOutputError, match="unknown PDDL type"):
        normalize_object_estimates(
            raw,
            domain=load_domain("kitchen"),
            observations=observations,
            observation_root=observation_root,
        )


def test_invalid_generated_predicate_is_rejected_after_artifacts_are_saved(
    tmp_path: Path,
) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    fixture_dir = FIXTURE_ROOT / "kitchen"
    invalid_initial = (fixture_dir / "initial.pddlfrag").read_text(
        encoding="utf-8"
    ).replace("(handempty)", "(unknown-state)")
    transport = FixtureTransport(fixture_dir, initial_override=invalid_initial)
    output_root = tmp_path / "run"
    with pytest.raises(InterpreterOutputError, match="unknown predicate"):
        make_interpreter(transport).interpret(
            task_instruction="Complete the fixed benchmark task.",
            domain=load_domain("kitchen"),
            observations=observations,
            observation_root=observation_root,
            output_root=output_root,
        )
    assert (output_root / "interpreter" / "initial_state.pddlfrag").is_file()
    assert (output_root / "interpreter" / "goal_state.pddlfrag").is_file()
    assert (output_root / "interpreter" / "problem_initial.pddl").is_file()


def test_unobserved_movable_declaration_is_rejected(tmp_path: Path) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    fixture_dir = FIXTURE_ROOT / "kitchen"
    invalid_initial = (fixture_dir / "initial.pddlfrag").read_text(
        encoding="utf-8"
    ).replace("mug_1 - vessel", "mug_1 ghost_mug - vessel")
    transport = FixtureTransport(fixture_dir, initial_override=invalid_initial)
    with pytest.raises(InterpreterOutputError, match="unobserved movable object"):
        make_interpreter(transport).interpret(
            task_instruction="Complete the fixed benchmark task.",
            domain=load_domain("kitchen"),
            observations=observations,
            observation_root=observation_root,
            output_root=tmp_path / "run",
        )
