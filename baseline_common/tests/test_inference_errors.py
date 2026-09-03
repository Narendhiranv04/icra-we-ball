from __future__ import annotations

from types import SimpleNamespace
import urllib.error

import pytest

from baseline_common.inference import (
    InvalidCompletionError,
    ModelTransportError,
    OpenAITransport,
    response_content,
)


def test_invalid_completion_content_is_not_a_transport_failure() -> None:
    response = {
        "choices": [
            {"message": {"content": "not json"}, "finish_reason": "stop"}
        ]
    }

    with pytest.raises(InvalidCompletionError, match="not valid JSON"):
        response_content(response)


def test_unreachable_server_is_a_transport_failure(monkeypatch) -> None:
    def unreachable(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)
    transport = OpenAITransport(
        SimpleNamespace(
            base_url="http://127.0.0.1:1/v1",
            api_key="",
            timeout_seconds=1.0,
        )
    )

    with pytest.raises(ModelTransportError, match="Cannot reach model server"):
        transport.complete({"model": "test"})
