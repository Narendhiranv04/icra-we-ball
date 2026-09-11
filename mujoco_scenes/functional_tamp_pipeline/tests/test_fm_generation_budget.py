"""The generation budget is a frozen decision, not an environment accident.

The archived distribution that every measurement in this work rests on was
produced with a 24000-token budget supplied through an environment variable,
while the code default was 8192.  Across the 91 responses that finished, 39 are
longer than 8192, so anyone running the pipeline without that variable set
would have had 43% of the trials truncated and would have read the result as
the model failing to specify the task.

Truncation stays a reported failure of the response.  There is no semantic
retry: exactly one semantic request per trial is the scientific rule, and a
second request would change what is being measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter


FROZEN_DEFAULT_MAX_TOKENS = 24000
ARCHIVE = (Path(__file__).resolve().parents[3]
           / "benchmark_reports" / "v3_qwen_distribution_3x32_20260910T053937")


def test_the_default_budget_is_the_one_the_archives_were_produced_with(monkeypatch):
    monkeypatch.delenv("TAMP_FM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("FM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("TAMP_FM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("TAMP_FM_MODEL", "unused-in-this-test")
    assert FMAdapter().max_tokens == FROZEN_DEFAULT_MAX_TOKENS


def test_the_default_budget_covers_every_archived_response_that_finished():
    """Read off the archive rather than asserted: the budget must clear the longest."""
    records = sorted(ARCHIVE.glob("*/*/trial_*/fm_call_record.json"))
    if not records:
        return  # the archive is not part of a minimal checkout
    finished = []
    truncated = []
    for path in records:
        record = json.loads(path.read_text())
        tokens = (record.get("usage") or {}).get("completion_tokens") or 0
        (truncated if record.get("finish_reason") == "length" else finished).append(tokens)
    assert finished, records[:1]
    assert max(finished) < FROZEN_DEFAULT_MAX_TOKENS, max(finished)
    # And the runaway generations are genuinely runaway: they are not responses
    # a slightly larger budget would have completed.
    assert all(count >= FROZEN_DEFAULT_MAX_TOKENS for count in truncated), truncated


def test_a_truncated_response_is_reported_as_a_response_failure():
    from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
        FM_RESPONSE_FAILURE_CATEGORIES,
        classify_pipeline_outcome,
    )

    outcome = classify_pipeline_outcome(
        task_specification_valid=False, graph_compiled=False,
        fm_response_usable="TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE"
        not in FM_RESPONSE_FAILURE_CATEGORIES,
        reason="Completion truncated due to token limit (finish_reason='length')",
    )
    assert outcome.category == "FM_RESPONSE_FAILURE"
