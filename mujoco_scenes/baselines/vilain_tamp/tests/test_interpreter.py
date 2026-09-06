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
    _declared_object_types,
    _extract_initial_fragments,
    normalize_object_estimates,
)
from mujoco_scenes.baselines.vilain_tamp.symbolic_contract import (
    build_variant_action_contract,
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
            if self.initial_override is not None:
                raw_text = self.initial_override
            else:
                raw_text = self._selection(request, "initial.pddlfrag", "true_fact_ids")
        elif request.call_type is FMCallType.GOAL_STATE:
            raw_text = self._selection(request, "goal.pddlfrag", "goal_fact_ids")
        else:
            raise AssertionError(f"unexpected call type: {request.call_type}")
        return FMTransportResponse(
            raw_text=raw_text,
            call_id=f"fixture-{len(self.calls)}",
            model=request.model,
            revision=request.revision,
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    def _selection(self, request: FMRequest, fixture: str, field: str) -> str:
        candidates = request.metadata["fact_candidates"]
        source = (self.fixture_dir / fixture).read_text(encoding="utf-8").lower()
        selected = [
            fact_id
            for fact_id, literal in candidates.items()
            if literal in source
        ]
        return __import__("json").dumps({field: selected})


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
    domain = load_domain(transport.fixture_dir.name)
    initial = (transport.fixture_dir / "initial.pddlfrag").read_text(encoding="utf-8")
    objects, _ = _extract_initial_fragments(initial)
    declared = _declared_object_types(objects)
    observed = __import__("json").loads(
        (transport.fixture_dir / "objects.json").read_text(encoding="utf-8")
    )["objects"]
    def is_movable(type_name: str) -> bool:
        current: str | None = type_name
        while current is not None:
            if current == "movable":
                return True
            current = domain.type_hierarchy.get(current)
        return False

    observed_ids = {
        f"{row['label'].lower().replace(' ', '_')}_1"
        for row in observed
        if is_movable(row["pddl_type"])
    }
    structural = {
        object_id: object_type
        for object_id, object_type in declared.items()
        if object_id not in observed_ids
    }
    return ViLaInInterpreter(
        object_client=client,
        reasoning_client=client,
        models=InterpreterModels(
            object_estimator_model="Qwen2.5-VL-7B-Instruct",
            object_estimator_revision="fixture-revision",
            reasoning_model="gpt-4o-2024-08-06",
            reasoning_model_revision=None,
        ),
        symbolic_contract=build_variant_action_contract(
            domain, "fixture", structural_inventory=structural
        ),
    )


@pytest.mark.parametrize(
    "domain_key, expected_ids",
    [
        ("kitchen", ("coffee_source_1", "mug_1", "spoon_1")),
        ("living_room", ("cup_1",)),
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
    assert result.problem.problem_text.startswith("(define (problem vilain-")
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


def test_raw_pddl_selection_is_rejected_after_one_bounded_retry(
    tmp_path: Path,
) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    fixture_dir = FIXTURE_ROOT / "kitchen"
    invalid_initial = (fixture_dir / "initial.pddlfrag").read_text(encoding="utf-8")
    transport = FixtureTransport(fixture_dir, initial_override=invalid_initial)
    output_root = tmp_path / "run"
    with pytest.raises(InterpreterOutputError, match="RAW_PDDL_NOT_ALLOWED"):
        make_interpreter(transport).interpret(
            task_instruction="Complete the fixed benchmark task.",
            domain=load_domain("kitchen"),
            observations=observations,
            observation_root=observation_root,
            output_root=output_root,
        )
    assert (
        output_root / "interpreter" / "initial_state_validation_errors.json"
    ).is_file()
    assert (
        output_root / "interpreter" / "initial_state_regeneration_call" / "raw_response.txt"
    ).is_file()
    assert transport.calls.count(FMCallType.INITIAL_STATE) == 2
    assert FMCallType.GOAL_STATE not in transport.calls


def test_one_schema_regeneration_can_recover_initial_state(tmp_path: Path) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    fixture_dir = FIXTURE_ROOT / "kitchen"
    class RepairOnceTransport(FixtureTransport):
        def complete(self, request: FMRequest) -> FMTransportResponse:
            if request.call_type is FMCallType.INITIAL_STATE:
                initial_count = self.calls.count(FMCallType.INITIAL_STATE)
                if initial_count == 0:
                    self.calls.append(request.call_type)
                    return FMTransportResponse(
                        '{"true_fact_ids":["unknown"]}',
                        f"fixture-{len(self.calls)}", request.model,
                        request.revision, {"input_tokens": 1, "output_tokens": 1},
                    )
            return super().complete(request)

    transport = RepairOnceTransport(fixture_dir)
    result = make_interpreter(transport).interpret(
        task_instruction="Complete the fixed benchmark task.",
        domain=load_domain("kitchen"),
        observations=observations,
        observation_root=observation_root,
        output_root=tmp_path / "run",
    )

    assert result.validation.valid
    assert transport.calls.count(FMCallType.INITIAL_STATE) == 2
    assert [call.call_type for call in result.calls] == [
        FMCallType.OBJECT_ESTIMATION,
        FMCallType.INITIAL_STATE,
        FMCallType.INITIAL_STATE,
        FMCallType.GOAL_STATE,
    ]


def test_normalized_bbox_is_converted_to_image_pixels_and_outer_fence_is_allowed(
    tmp_path: Path,
) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    raw = """```json
{"objects":[{"label":"mug","pddl_type":"vessel","description":"",
"detections":[{"stage_id":"000_initial","camera_id":"front",
"bbox_1000":[250,0,750,500],"confidence":0.9}]}]}
```"""

    estimates = normalize_object_estimates(
        raw,
        domain=load_domain("kitchen"),
        observations=observations,
        observation_root=observation_root,
    )

    assert estimates[0].detections[0]["xyxy"] == (1.0, 0.0, 3.0, 2.0)


@pytest.mark.parametrize(
    "box",
    ([0, 0, 1001, 500], [500, 0, 500, 500], [-1, 0, 500, 500]),
)
def test_invalid_normalized_bbox_is_rejected(tmp_path: Path, box: list[int]) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    raw = (
        '{"objects":[{"label":"mug","pddl_type":"vessel",'
        '"detections":[{"stage_id":"000_initial","camera_id":"front",'
        f'"bbox_1000":{box},"confidence":0.9}}]}}]}}'
    )
    with pytest.raises(InterpreterOutputError, match="normalized bounds"):
        normalize_object_estimates(
            raw,
            domain=load_domain("kitchen"),
            observations=observations,
            observation_root=observation_root,
        )


def test_truncated_object_json_is_rejected_explicitly(tmp_path: Path) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    with pytest.raises(InterpreterOutputError, match="truncated JSON"):
        normalize_object_estimates(
            '{"objects":[{"label":"mug"}',
            domain=load_domain("kitchen"),
            observations=observations,
            observation_root=observation_root,
        )


def test_reasoning_fragments_accept_only_a_single_outer_fence() -> None:
    from mujoco_scenes.baselines.vilain_tamp.interpreter import (
        _extract_goal_fragment,
        _extract_initial_fragments,
    )

    assert _extract_initial_fragments(
        "```pddl\n(:objects mug - vessel)\n(:init (handempty))\n```"
    ) == ("(:objects mug - vessel)", "(:init (handempty))")
    assert _extract_goal_fragment("```pddl\n(:goal (handempty))\n```") == (
        "(:goal (handempty))"
    )


def test_model_never_controls_compiled_object_declarations(tmp_path: Path) -> None:
    observation_root = tmp_path / "observations"
    observations = acquire_observation(observation_root, "kitchen")
    fixture_dir = FIXTURE_ROOT / "kitchen"
    transport = FixtureTransport(fixture_dir)
    result = make_interpreter(transport).interpret(
        task_instruction="Complete the fixed benchmark task.",
        domain=load_domain("kitchen"),
        observations=observations,
        observation_root=observation_root,
        output_root=tmp_path / "run",
    )
    assert "ghost_mug" not in result.problem.problem_text
