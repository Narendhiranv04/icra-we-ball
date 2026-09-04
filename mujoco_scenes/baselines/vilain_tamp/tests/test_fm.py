from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    CameraFrameArtifacts,
    ObjectEstimate,
    ObjectEstimateStatus,
    ViLaInObservation,
)
from mujoco_scenes.baselines.vilain_tamp.domains import load_domain
from mujoco_scenes.baselines.vilain_tamp.fm import (
    FMCallType,
    FMRequest,
    FMTransportError,
    FMTransportResponse,
    RecordedFMClient,
)
from mujoco_scenes.baselines.vilain_tamp.prompts import (
    PromptBundle,
    build_corrective_planning_prompt,
    build_goal_state_prompt,
    build_initial_state_prompt,
    build_object_estimation_prompt,
)


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[FMRequest] = []

    def complete(self, request: FMRequest) -> FMTransportResponse:
        self.requests.append(request)
        return FMTransportResponse(
            raw_text=f"synthetic response for {request.call_type.value}",
            call_id=f"call-{len(self.requests)}",
            model=request.model,
            revision=request.revision,
            usage={"input_tokens": 10, "output_tokens": 5},
        )


def observation() -> ViLaInObservation:
    frame = CameraFrameArtifacts(
        camera_id="front",
        view_description="Front room view",
        rgb_path="stages/000_initial/cameras/front/rgb.png",
        depth_path="stages/000_initial/cameras/front/depth.npy",
        calibration_path="stages/000_initial/cameras/front/camera.json",
        rgb_sha256="a" * 64,
        depth_sha256="b" * 64,
        calibration_sha256="c" * 64,
    )
    return ViLaInObservation(
        domain="kitchen",
        observation_mode="initial_observation_only",
        stage_id="000_initial",
        camera_frames=(frame,),
        opened_region_id=None,
        capture_timestamp="2026-09-04T00:00:00+00:00",
        inspection_ordinal=None,
        content_hash="d" * 64,
    )


def object_estimate() -> ObjectEstimate:
    return ObjectEstimate(
        object_id="mug_1",
        label="mug",
        pddl_type="vessel",
        description="white mug",
        detections=({"camera_id": "front", "xyxy": [1, 2, 3, 4]},),
        estimated_centroid_m=None,
        centroid_covariance=None,
        observation_stage_ids=("000_initial",),
        status=ObjectEstimateStatus.OBSERVED,
    )


def bundles() -> list[tuple[FMCallType, PromptBundle]]:
    domain = load_domain("kitchen")
    estimate = object_estimate()
    task = "Prepare the requested meal."
    initial = "(:objects mug_1 - vessel) (:init (handempty))"
    problem = "(define (problem synthetic) (:domain vilain-kitchen))"
    return [
        (
            FMCallType.OBJECT_ESTIMATION,
            build_object_estimation_prompt(
                task_instruction=task,
                domain=domain,
                observations=(observation(),),
            ),
        ),
        (
            FMCallType.INITIAL_STATE,
            build_initial_state_prompt(
                task_instruction=task, domain=domain, objects=(estimate,)
            ),
        ),
        (
            FMCallType.GOAL_STATE,
            build_goal_state_prompt(
                task_instruction=task,
                domain=domain,
                objects=(estimate,),
                initial_state_fragment=initial,
            ),
        ),
        (
            FMCallType.CORRECTIVE_PLANNING,
            build_corrective_planning_prompt(
                task_instruction=task,
                domain=domain,
                object_estimates=(estimate,),
                initial_problem=problem,
                current_problem=problem,
                current_failure={"stage": "IK", "summary": "target unreachable"},
                prior_problem_hashes=("1" * 64,),
                prior_error_summaries=("previous syntax error",),
            ),
        ),
    ]


def test_all_four_call_types_are_recorded_without_external_calls(tmp_path: Path) -> None:
    transport = FakeTransport()
    client = RecordedFMClient(transport)
    records = []
    for index, (call_type, bundle) in enumerate(bundles()):
        model = (
            "Qwen2.5-VL-7B-Instruct"
            if call_type is FMCallType.OBJECT_ESTIMATION
            else "gpt-4o-2024-08-06"
        )
        request = FMRequest(
            call_type=call_type,
            model=model,
            revision="checkpoint-1" if call_type is FMCallType.OBJECT_ESTIMATION else None,
            messages=bundle.messages(),
            image_artifacts=bundle.image_artifacts,
            metadata={"api_key": "must-not-persist", "sample": index},
        )
        response, record = client.invoke(request, tmp_path / f"call_{index}")
        assert response.raw_text.startswith("synthetic response")
        records.append(record)

    assert [request.call_type for request in transport.requests] == list(FMCallType)
    assert [record.call_type for record in records] == list(FMCallType)
    for index, record in enumerate(records):
        request_data = json.loads(
            (tmp_path / f"call_{index}" / "request.json").read_text(encoding="utf-8")
        )
        assert request_data["metadata"]["api_key"] == "[REDACTED]"
        assert Path(record.raw_response_artifact).read_text(encoding="utf-8").startswith(
            "synthetic response"
        )
        metadata = json.loads(Path(record.metadata_artifact).read_text(encoding="utf-8"))
        assert metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert metadata["latency_seconds"] >= 0


def test_pddl_stages_request_pddl_text_and_no_action_sequence() -> None:
    prompt_by_type = dict(bundles())
    assert "Return only PDDL" in prompt_by_type[FMCallType.INITIAL_STATE].user_text
    assert "PDDL `:goal`" in prompt_by_type[FMCallType.GOAL_STATE].user_text
    corrective = prompt_by_type[FMCallType.CORRECTIVE_PLANNING]
    assert "one complete replacement PDDL problem" in corrective.user_text
    assert "never output an action sequence" in corrective.system_text
    assert "return json" not in corrective.user_text.lower()


def test_prompts_do_not_reference_protected_method_artifacts() -> None:
    rendered = "\n".join(
        bundle.system_text + "\n" + bundle.user_text for _, bundle in bundles()
    )
    forbidden = (
        "functional_tamp_pipeline",
        "ground_graph",
        "GraphGroundingResult",
        "G_F",
        "G_O",
        "Phase3Handoff",
        "graph_grounding_result",
        "plan_grounding_audit",
    )
    assert not any(item in rendered for item in forbidden)


def test_transport_errors_are_typed_and_do_not_create_fake_response(tmp_path: Path) -> None:
    class BrokenTransport:
        def complete(self, request: FMRequest) -> FMTransportResponse:
            del request
            raise OSError("offline")

    request = FMRequest(
        call_type=FMCallType.GOAL_STATE,
        model="gpt-4o-2024-08-06",
        revision=None,
        messages=({"role": "user", "content": "test"},),
    )
    with pytest.raises(FMTransportError, match="transport failed"):
        RecordedFMClient(BrokenTransport()).invoke(request, tmp_path)
    assert (tmp_path / "request.json").is_file()
    assert not (tmp_path / "raw_response.txt").exists()


def test_response_model_mismatch_is_rejected(tmp_path: Path) -> None:
    class WrongModelTransport:
        def complete(self, request: FMRequest) -> FMTransportResponse:
            return FMTransportResponse("text", "call-1", "moving-alias", None, {})

    request = FMRequest(
        call_type=FMCallType.INITIAL_STATE,
        model="gpt-4o-2024-08-06",
        revision=None,
        messages=({"role": "user", "content": "test"},),
    )
    with pytest.raises(FMTransportError, match="model mismatch"):
        RecordedFMClient(WrongModelTransport()).invoke(request, tmp_path)
