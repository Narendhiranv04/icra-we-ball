"""The prompt cannot be changed on the strength of the frozen replay alone.

The replay feeds archived FM responses through the pipeline.  It is structurally
incapable of detecting a prompt change: the prompt is not read.  A sentence
added to the OPERATIONS paragraph after the distribution was collected therefore
shipped with no evidence at all, and the first live run after it showed the
model dropping an instruction-required operation -- "Place soup_ingredient into
soup_bowl" became "Place soup bowl on table" with no filling step -- so the
contract omitted an outcome, the planner satisfied everything it did state, and
the run announced completion of a half-finished task.

These tests do not freeze the prompt.  They pin the specific regression and
require that a diff against the collection the numbers came from is deliberate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import SYSTEM_PROMPT_V3

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "benchmark_reports/v3_qwen_distribution_3x32_20260910T053937/collection_manifest.json"


def _paragraph(text: str, prefix: str) -> str:
    return next(p for p in text.split("\n") if p.startswith(prefix))


def test_operations_wording_matches_the_collection_it_is_compared_against():
    if not MANIFEST.is_file():
        pytest.skip("archived collection manifest not present")
    archived = json.loads(MANIFEST.read_text())["system_prompt_text"]
    assert _paragraph(SYSTEM_PROMPT_V3, "OPERATIONS") == _paragraph(archived, "OPERATIONS")


def test_the_motion_decomposition_instruction_is_not_reintroduced():
    """It cost an instruction-required operation, observed live."""
    for fragment in ("single concrete physical action",
                     "not two actions bundled into one phrase",
                     "that is two operations"):
        assert fragment not in SYSTEM_PROMPT_V3, (
            f"the OPERATIONS motion-decomposition wording is back: {fragment!r}")


def test_the_deliberate_roles_change_is_kept():
    """required_count counts occasions, not instances.

    This one *is* a deliberate divergence from the collection: the compiler's
    binding-policy handling is built on it.  Pinned so a blanket revert to the
    archived prompt cannot quietly undo it.
    """
    assert "not how many of them must exist" in SYSTEM_PROMPT_V3
