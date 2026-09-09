import json
from pathlib import Path

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    LIVE_V2_EQUIVALENT,
    normalize_and_validate_v2_contract,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "benchmark_reports/live_v2_thinking_3view_full32_20260909T105016IST"


def _raw(domain, variant):
    return json.loads((MATRIX / domain / variant / "raw_v2.json").read_text())


def test_live_and_replay_use_identical_normalize_validate_boundary():
    raw = _raw("living_room", "L1")
    live, live_trace = normalize_and_validate_v2_contract(
        raw, domain="living_room", validation_mode=LIVE_V2_EQUIVALENT
    )
    replay, replay_trace = normalize_and_validate_v2_contract(
        raw, domain="living_room", validation_mode=LIVE_V2_EQUIVALENT
    )
    graph = VLMSpecProvider().provide("living_room", "task", raw_document=raw)
    assert live == replay
    assert live_trace == replay_trace == graph.metadata["v2_boundary_normalization_trace"]
    assert graph.metadata["v2_validation_mode"] == LIVE_V2_EQUIVALENT


@pytest.mark.parametrize("domain,variant,reason", [
    ("kitchen", "K1", "INCONSISTENT_OPERATION_REUSE_CARDINALITY"),
    ("kitchen", "K2", "serving_surface"),
    ("workshop", "W9", "INCOMPLETE_OPERATION_PARTICIPANT_STRUCTURE"),
])
def test_archived_invalid_contract_stops_at_task_specification(domain, variant, reason):
    with pytest.raises(VLMSpecificationError) as caught:
        VLMSpecProvider().provide(domain, "task", raw_document=_raw(domain, variant))
    assert reason in str(caught.value)
    assert caught.value.category in {"MALFORMED_VLM_SPECIFICATION", "TASK_SPECIFICATION_FAILURE"}
