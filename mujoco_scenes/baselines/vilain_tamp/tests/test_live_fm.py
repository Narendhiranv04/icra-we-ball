from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from mujoco_scenes.baselines.vilain_tamp.config import BaselineConfig
from mujoco_scenes.baselines.vilain_tamp.fm import (
    FMCallType,
    FMRequest,
    FMTransportError,
    RecordedFMClient,
)
from mujoco_scenes.baselines.vilain_tamp.live_fm import (
    DEFAULT_VLLM_BASE_URL,
    OpenAIReasoningTransport,
    PAPER_QWEN_MODEL,
    PAPER_REASONING_MODEL,
    QwenGeneration,
    QwenVLTransport,
    VLLMQwenTransport,
    build_qwen_only_clients,
    build_paper_faithful_clients,
    main,
    require_vilain_environment,
    validate_standalone_object_estimation,
)


REVISION = "a" * 40


class FakeQwenBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> QwenGeneration:
        self.calls.append(kwargs)
        return QwenGeneration(
            text='{"objects": []}',
            input_tokens=123,
            output_tokens=7,
            device="cpu",
            dtype="torch.float32",
        )


@dataclass
class FakeUsage:
    prompt_tokens: int = 11
    completion_tokens: int = 5
    total_tokens: int = 16


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        message = type("Message", (), {"content": "(:goal (handempty))"})()
        choice = type("Choice", (), {"message": message})()
        return type(
            "Response",
            (),
            {
                "id": "chatcmpl-test",
                "model": PAPER_REASONING_MODEL,
                "choices": [choice],
                "usage": FakeUsage(),
                "system_fingerprint": "fp_test",
            },
        )()


def _object_request() -> FMRequest:
    return FMRequest(
        call_type=FMCallType.OBJECT_ESTIMATION,
        model=PAPER_QWEN_MODEL,
        revision=REVISION,
        messages=(
            {"role": "system", "content": "estimate objects"},
            {"role": "user", "content": "return JSON"},
        ),
        image_artifacts=("stages/000_initial/cameras/front/rgb.png",),
        response_format="json",
    )


def test_qwen_transport_uses_exact_revision_images_and_records_metadata(
    tmp_path: Path,
) -> None:
    image = tmp_path / "stages/000_initial/cameras/front/rgb.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    backend = FakeQwenBackend()
    transport = QwenVLTransport(
        image_root=tmp_path,
        backend=backend,
        require_dedicated_venv=False,
    )

    response, record = RecordedFMClient(transport).invoke(
        _object_request(), tmp_path / "call"
    )

    assert response.model == PAPER_QWEN_MODEL
    assert response.revision == REVISION
    assert response.usage == {"input_tokens": 123, "output_tokens": 7}
    assert backend.calls[0]["revision"] == REVISION
    assert backend.calls[0]["image_paths"] == (image.resolve(),)
    qwen_messages = backend.calls[0]["messages"]
    assert qwen_messages == _object_request().messages
    metadata = json.loads(Path(record.metadata_artifact).read_text(encoding="utf-8"))
    assert metadata["provider_metadata"]["model_source"] == (
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    assert metadata["provider_metadata"]["resolved_revision"] == REVISION


@pytest.mark.parametrize("revision", [None, "main", "a" * 39])
def test_qwen_transport_rejects_non_exact_revision(
    tmp_path: Path, revision: str | None
) -> None:
    request = FMRequest(
        **{**_object_request().__dict__, "revision": revision}
    )
    transport = QwenVLTransport(
        image_root=tmp_path,
        backend=FakeQwenBackend(),
        require_dedicated_venv=False,
    )
    with pytest.raises(FMTransportError, match="exact 40-character commit"):
        transport.complete(request)


def test_qwen_transport_rejects_image_outside_observation_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"png")
    request = FMRequest(
        **{**_object_request().__dict__, "image_artifacts": (str(outside),)}
    )
    transport = QwenVLTransport(
        image_root=tmp_path,
        backend=FakeQwenBackend(),
        require_dedicated_venv=False,
    )
    with pytest.raises(FMTransportError, match="escapes observation root"):
        transport.complete(request)
    outside.unlink()


def test_openai_transport_uses_pinned_snapshot_and_no_images(tmp_path: Path) -> None:
    fake = FakeOpenAIClient()
    transport = OpenAIReasoningTransport(
        client=fake, require_dedicated_venv=False
    )
    request = FMRequest(
        call_type=FMCallType.GOAL_STATE,
        model=PAPER_REASONING_MODEL,
        revision=PAPER_REASONING_MODEL,
        messages=({"role": "user", "content": "return a goal"},),
        metadata={"api_key": "never-write-this"},
    )
    response, record = RecordedFMClient(transport).invoke(request, tmp_path / "call")

    assert fake.calls == [
        {
            "model": PAPER_REASONING_MODEL,
            "messages": [{"role": "user", "content": "return a goal"}],
            "temperature": 0,
        }
    ]
    assert response.call_id == "chatcmpl-test"
    request_artifact = json.loads(Path(record.request_artifact).read_text())
    assert request_artifact["metadata"]["api_key"] == "[REDACTED]"
    assert "never-write-this" not in "".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "call").iterdir()
    )


def test_openai_transport_rejects_moving_alias_and_object_calls() -> None:
    transport = OpenAIReasoningTransport(
        client=FakeOpenAIClient(), require_dedicated_venv=False
    )
    moving = FMRequest(
        call_type=FMCallType.INITIAL_STATE,
        model="gpt-4o",
        revision=None,
        messages=({"role": "user", "content": "test"},),
    )
    with pytest.raises(FMTransportError, match="paper-faithful reasoning model"):
        transport.complete(moving)
    with pytest.raises(FMTransportError, match="does not estimate objects"):
        transport.complete(_object_request())


def test_live_factory_is_lazy_and_returns_exact_model_identities(tmp_path: Path) -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "paper_faithful.yaml"
    )
    clients = build_paper_faithful_clients(
        config=BaselineConfig.from_yaml(config_path),
        image_root=tmp_path,
        qwen_revision=REVISION,
        qwen_backend=FakeQwenBackend(),
        openai_client=FakeOpenAIClient(),
    )
    assert clients.object_estimator_model == PAPER_QWEN_MODEL
    assert clients.object_estimator_revision == REVISION
    assert clients.reasoning_model == PAPER_REASONING_MODEL
    assert clients.reasoning_model_revision == PAPER_REASONING_MODEL


def test_live_environment_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/not-the-baseline-environment")
    with pytest.raises(FMTransportError, match=".venv-vilain-tamp"):
        require_vilain_environment()
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/.venv-vilain-tamp")
    require_vilain_environment()


def test_vllm_transport_is_localhost_only_and_normalizes_multimodal_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = FakeOpenAIClient()
    fake.create = lambda **kwargs: (
        fake.calls.append(kwargs)
        or type(
            "Response",
            (),
            {
                "id": "chatcmpl-vllm",
                "model": "qwen35-9b",
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": '{"objects": []}'})()},
                    )()
                ],
                "usage": FakeUsage(),
            },
        )()
    )
    transport = VLLMQwenTransport(
        image_root=tmp_path,
        served_model_id="qwen35-9b",
        base_url=DEFAULT_VLLM_BASE_URL,
        client=fake,
    )
    request = FMRequest(
        call_type=FMCallType.OBJECT_ESTIMATION,
        model="qwen35-9b",
        revision=None,
        messages=(
            {"role": "system", "content": "estimate"},
            {"role": "user", "content": "return JSON"},
        ),
        image_artifacts=("frame.png",),
    )
    response = transport.complete(request)

    assert response.model == "qwen35-9b"
    content = fake.calls[0]["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1] == {"type": "text", "text": "return JSON"}
    assert response.provider_metadata["base_url"] == DEFAULT_VLLM_BASE_URL
    assert "OPENAI_API_KEY" not in response.provider_metadata
    with pytest.raises(ValueError, match="approved local tunnel"):
        VLLMQwenTransport(
            image_root=tmp_path,
            served_model_id="qwen35-9b",
            base_url="https://api.openai.com/v1",
        )


def test_vllm_transport_packs_many_views_per_observation_stage(
    tmp_path: Path,
) -> None:
    fake = FakeOpenAIClient()
    fake.create = lambda **kwargs: (
        fake.calls.append(kwargs)
        or type(
            "Response",
            (),
            {
                "id": "chatcmpl-packed",
                "model": "qwen35-9b",
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": '{"objects": []}'})()},
                    )()
                ],
                "usage": FakeUsage(),
            },
        )()
    )
    from PIL import Image

    artifacts = []
    for stage in ("000_initial", "001_d1", "002_d2"):
        for camera in range(5):
            path = Path("stages") / stage / "cameras" / f"camera_{camera}" / "rgb.png"
            absolute = tmp_path / path
            absolute.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (camera * 20, 0, 0)).save(absolute)
            artifacts.append(str(path))
    transport = VLLMQwenTransport(
        image_root=tmp_path,
        served_model_id="qwen35-9b",
        client=fake,
    )
    response = transport.complete(
        FMRequest(
            call_type=FMCallType.OBJECT_ESTIMATION,
            model="qwen35-9b",
            revision=None,
            messages=({"role": "user", "content": "return JSON"},),
            image_artifacts=tuple(artifacts),
            response_format="json",
        )
    )

    content = fake.calls[0]["messages"][0]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 3
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in images)
    assert response.provider_metadata["source_image_count"] == 15
    assert response.provider_metadata["model_image_count"] == 3
    assert response.provider_metadata["packed_stage_ids"] == [
        "000_initial",
        "001_d1",
        "002_d2",
    ]


def test_vllm_transport_supports_text_reasoning_and_qwen_only_clients(
    tmp_path: Path,
) -> None:
    fake = FakeOpenAIClient()
    fake.create = lambda **kwargs: (
        fake.calls.append(kwargs)
        or type(
            "Response",
            (),
            {
                "id": "chatcmpl-vllm-text",
                "model": "qwen35-9b",
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": "(:goal (handempty))"})()},
                    )()
                ],
                "usage": FakeUsage(),
            },
        )()
    )
    clients = build_qwen_only_clients(
        image_root=tmp_path,
        served_model_id="qwen35-9b",
        client=fake,
    )
    response = clients.reasoning_client.transport.complete(
        FMRequest(
            call_type=FMCallType.GOAL_STATE,
            model="qwen35-9b",
            revision=None,
            messages=({"role": "user", "content": "goal"},),
        )
    )

    assert response.raw_text == "(:goal (handempty))"
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "goal"}]
    assert clients.object_estimator_model == clients.reasoning_model == "qwen35-9b"
    assert clients.object_estimator_revision is None
    assert clients.reasoning_model_revision is None


def test_standalone_object_call_uses_manifest_and_writes_raw_artifacts(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "observations/stages/000_initial/cameras/front/rgb.png"
    rgb.parent.mkdir(parents=True)
    rgb.write_bytes(b"png")
    frame = {
        "camera_id": "front",
        "view_description": "front view",
        "rgb_path": "stages/000_initial/cameras/front/rgb.png",
        "depth_path": "stages/000_initial/cameras/front/depth.npy",
        "calibration_path": "stages/000_initial/cameras/front/camera.json",
        "rgb_sha256": "a" * 64,
        "depth_sha256": "b" * 64,
        "calibration_sha256": "c" * 64,
    }
    manifest = tmp_path / "observations/observation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "domain": "kitchen",
                        "observation_mode": "initial_observation_only",
                        "stage_id": "000_initial",
                        "camera_frames": [frame],
                        "opened_region_id": None,
                        "capture_timestamp": "2026-09-05T00:00:00+00:00",
                        "inspection_ordinal": None,
                        "content_hash": "d" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    transport = QwenVLTransport(
        image_root=manifest.parent,
        backend=FakeQwenBackend(),
        require_dedicated_venv=False,
    )

    record = validate_standalone_object_estimation(
        observation_manifest=manifest,
        task_instruction="Prepare the meal.",
        domain="kitchen",
        model_source="Qwen/Qwen2.5-VL-7B-Instruct",
        revision=REVISION,
        output_directory=tmp_path / "standalone",
        transport=transport,
    )

    assert Path(record.request_artifact).is_file()
    assert Path(record.raw_response_artifact).read_text() == '{"objects": []}'


def test_cli_help_does_not_make_a_model_call(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "standalone" in capsys.readouterr().out
