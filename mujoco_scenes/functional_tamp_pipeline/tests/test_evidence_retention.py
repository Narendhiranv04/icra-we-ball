"""A belief that was checked and confirmed must not be overridden by a weaker one.

Precedence between an authoritative cached observation and a later, weaker
re-observation was written as `a.get(k) or b.get(k) or c.get(k)`.  For a
container-valued field that is wrong: an observation validated with
`reason_codes == []` is reporting that it found no problems, and an empty list
is falsy, so the chain fell through to a later observation's complaints.  A
confirmed belief could therefore be discarded by an unconfirmed one.
"""
from __future__ import annotations

from mujoco_scenes.functional_tamp_pipeline.grounding import (
    _first_declared,
    check_semantic_role_compatibility,
    extract_plausible_labels,
)


def test_an_empty_finding_is_an_answer_not_an_absence():
    assert _first_declared("reason_codes", {"reason_codes": []}, {"reason_codes": ["BAD"]}) == []
    assert _first_declared("reason_codes", {}, {"reason_codes": ["BAD"]}) == ["BAD"]
    assert _first_declared("reason_codes", {}, {}) is None


def test_a_validated_clean_belief_is_not_overridden_by_a_later_complaint():
    """The defect: validated says no problems, latest complains, latest won."""
    belief = {
        "validated": {
            "status": "SUPPORTED",
            "canonical_label": "mug",
            "plausible_labels": ["mug"],
            "reason_codes": [],
        },
        "latest_observation": {
            "status": "UNKNOWN",
            "canonical_label": None,
            "reason_codes": ["SEMANTIC_LABEL_UNKNOWN"],
        },
    }
    # Before the fix the empty validated reason_codes fell through to
    # SEMANTIC_LABEL_UNKNOWN, which is a lack-of-evidence code, and the
    # confirmed hypotheses were dropped entirely.
    assert extract_plausible_labels(belief) == ["mug"]


def test_the_confirmed_label_still_satisfies_its_role():
    belief = {
        "validated": {
            "status": "SUPPORTED",
            "canonical_label": "mug",
            "plausible_labels": ["mug"],
            "reason_codes": [],
        },
        "latest_observation": {
            "status": "UNKNOWN",
            "reason_codes": ["INSUFFICIENT_DETECTOR_CONFIDENCE"],
        },
    }
    verdict, matched = check_semantic_role_compatibility(belief, ["mug", "cup"])
    assert verdict == "TRUE", (verdict, matched)


def test_a_genuine_lack_of_evidence_is_still_respected():
    """The fix must not turn a real absence of evidence into a confirmation."""
    belief = {
        "validated": {"status": "UNKNOWN", "reason_codes": ["NO_ASSOCIATED_DETECTION"]},
        "latest_observation": {"status": "UNKNOWN", "reason_codes": []},
    }
    assert extract_plausible_labels(belief) == []


def test_absent_reason_codes_still_fall_through():
    """Only a declared empty list is authoritative; a missing key is not."""
    belief = {
        "validated": {"status": "SUPPORTED", "plausible_labels": ["mug"]},
        "latest_observation": {"reason_codes": ["NO_ASSOCIATED_DETECTION"]},
    }
    assert extract_plausible_labels(belief) == []
