"""An unreachable endpoint must never be scored as the model failing.

A tunnel died mid-experiment and 94 trials were recorded as VLM_SPEC_FAILED.
Nothing distinguished them in the aggregate from the model genuinely failing to
specify the task, so the run silently deflated instead of stopping.  These tests
pin the separation at every layer it has to hold at.
"""
from __future__ import annotations

import json
import urllib.error
import pytest

from mujoco_scenes.evaluation_outcome import (
    is_infrastructure_failure, summarize, trial_is_scorable,
)
from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
    FM_RESPONSE_FAILURE_CATEGORIES,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMEndpointUnavailableError, FMTransportError, OpenAICompletionTransport,
)


class _Refusing:
    """Every call fails at the socket, as a dead endpoint does."""

    def __init__(self, exc=None):
        self.calls = 0
        self.exc = exc or urllib.error.URLError("Connection refused")

    def __call__(self, request, timeout=None):
        self.calls += 1
        raise self.exc


def _transport(monkeypatch, opener, **kwargs):
    monkeypatch.setattr(urllib.request, "urlopen", opener)
    return OpenAICompletionTransport("http://127.0.0.1:8000/v1", "", 5.0, **kwargs)


# --- positive: a refused connection is an infrastructure fault ---------------

def test_refused_connection_raises_infrastructure_not_spec_failure(monkeypatch):
    import urllib.request
    transport = _transport(monkeypatch, _Refusing(), max_attempts=1)
    with pytest.raises(FMEndpointUnavailableError) as caught:
        transport.complete({})
    assert caught.value.category == "INFRASTRUCTURE_UNAVAILABLE"


def test_infrastructure_category_counts_as_unusable_fm_response():
    assert "INFRASTRUCTURE_UNAVAILABLE" in FM_RESPONSE_FAILURE_CATEGORIES


def test_refused_connection_is_retried_and_retries_are_bounded(monkeypatch):
    import urllib.request
    opener = _Refusing()
    transport = _transport(monkeypatch, opener, max_attempts=3, retry_backoff_seconds=0)
    with pytest.raises(FMEndpointUnavailableError):
        transport.complete({})
    assert opener.calls == 3, "connection-level failure must be retried up to the bound"


def test_server_error_is_treated_as_unavailable(monkeypatch):
    import io, urllib.request
    err = urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"overloaded"))
    transport = _transport(monkeypatch, _Refusing(err), max_attempts=2, retry_backoff_seconds=0)
    with pytest.raises(FMEndpointUnavailableError):
        transport.complete({})


# --- adversarial: a real model failure must NOT be excused or retried --------

def test_client_error_is_a_real_failure_and_is_not_retried(monkeypatch):
    """HTTP 4xx means the server read the request and rejected it.

    Retrying that would resample until the benchmark agreed with us, and
    excusing it would hide a genuine defect behind an 'outage'.
    """
    import io, urllib.request
    err = urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"schema violation"))
    opener = _Refusing(err)
    transport = _transport(monkeypatch, opener, max_attempts=3, retry_backoff_seconds=0)
    with pytest.raises(FMTransportError) as caught:
        transport.complete({})
    assert not isinstance(caught.value, FMEndpointUnavailableError)
    assert opener.calls == 1, "a rejected request is the datum; it must not be retried"


def test_malformed_json_is_a_real_failure_not_an_outage(monkeypatch):
    import urllib.request
    opener = _Refusing(json.JSONDecodeError("bad", "", 0))
    transport = _transport(monkeypatch, opener, max_attempts=3, retry_backoff_seconds=0)
    with pytest.raises(FMTransportError) as caught:
        transport.complete({})
    assert not isinstance(caught.value, FMEndpointUnavailableError)
    assert opener.calls == 1


def test_spec_failure_status_remains_scorable():
    """Only the outage status is excluded; a genuine spec failure still counts."""
    assert trial_is_scorable("VLM_SPEC_FAILED")
    assert not trial_is_scorable("INFRASTRUCTURE_UNAVAILABLE")
    assert is_infrastructure_failure("INFRASTRUCTURE_UNAVAILABLE")


# --- the scoring consequence -------------------------------------------------

def test_outage_rows_leave_the_denominator_rather_than_scoring_wrong():
    rows = [
        {"feasible": True, "gt_full_task_satisfied": True,
         "pipeline_status": "ACTION_SEQUENCE_READY"},
        {"feasible": True, "gt_full_task_satisfied": False,
         "pipeline_status": "INFRASTRUCTURE_UNAVAILABLE"},
    ]
    out = summarize(rows)
    assert out["trials"] == 1 and out["trials_attempted"] == 2
    assert out["infrastructure_excluded"] == 1
    assert out["feasible_outcome_correct"] == 1, "1/1, not 1/2"


def test_an_outage_cannot_manufacture_a_perfect_score():
    """Excluding outages must not let an all-outage run report success."""
    rows = [{"feasible": True, "gt_full_task_satisfied": False,
             "pipeline_status": "INFRASTRUCTURE_UNAVAILABLE"} for _ in range(10)]
    out = summarize(rows)
    assert out["trials"] == 0 and out["infrastructure_excluded"] == 10
    assert out["overall_outcome_correct"] == 0
