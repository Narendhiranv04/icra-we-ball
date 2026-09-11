"""An earlier stage may rank semantic hypotheses; it may not delete them.

The size prior is right and belongs to the *fused* cloud, where how big a thing
is can actually be known.  It used to also run at association time against the
raw proposal cloud, which takes in surrounding surface: a 4 cm fastener measured
13 cm there, every "screw" hypothesis was deleted as physically impossible on a
measurement wrong by a factor of three, and six archived trials then failed for
a fastener that had been detected and accepted in all five views.

These tests pin the invariant and the constraint that still has to hold.
"""
from __future__ import annotations

import numpy as np

from mujoco_scenes.workshop_phase1.tracking import (
    PersistentInstanceTracker, _physical_prior_multipliers,
)

PRIOR = {
    "enabled": True,
    "small_object_max_dimension_m": 0.08,
    "small_object": {"screw": 2.0, "screwdriver": 0.0, "power_driver": 0.0},
    "large_object": {"screw": 0.0, "screwdriver": 2.0},
}
FUSION = {"winner_policy": "weighted_score_then_supporting_views",
          "minimum_supporting_views": 1, "proposal_crop_score_multiplier": 20.0}


def _observation(camera, label, confidence, alternatives=()):
    return {
        "camera_id": camera, "canonical_label": label, "raw_label": label,
        "confidence": confidence, "physical_support_quality": 1.0,
        "inference_source": "proposal_crop",
        "semantic_alternatives": [
            {"canonical_label": alt, "raw_label": alt, "confidence": conf}
            for alt, conf in alternatives
        ],
    }


def test_the_fused_size_promotes_a_retained_hypothesis():
    """The W2 case.

    Every view's primary label is a driver, as the proposal crops had it, and
    every view also carries a screw hypothesis.  At 4 cm the prior zeroes the
    driver labels, so the retained hypothesis is what survives -- which is only
    possible because the hypothesis was not deleted earlier.
    """
    observations = [
        _observation(camera, "screwdriver", 0.10, [("screw", 0.05)])
        for camera in ("LEFT", "RIGHT", "TOP", "FRONT", "CLOSE")
    ]
    belief = PersistentInstanceTracker._compute_consensus_semantic_belief(
        observations, fusion_config=FUSION, physical_prior=PRIOR,
        maximum_dimension_m=0.0398)
    assert belief["canonical_label"] == "screw"


def test_the_same_evidence_at_driver_size_stays_a_driver():
    """The adversarial half: a retained hypothesis is not promoted on its own.

    Identical observations, a 20 cm object.  The prior now zeroes the screw and
    the driver labels win, so nothing here prefers the smaller reading.
    """
    observations = [
        _observation(camera, "screwdriver", 0.10, [("screw", 0.05)])
        for camera in ("LEFT", "RIGHT", "TOP", "FRONT", "CLOSE")
    ]
    belief = PersistentInstanceTracker._compute_consensus_semantic_belief(
        observations, fusion_config=FUSION, physical_prior=PRIOR,
        maximum_dimension_m=0.20)
    assert belief["canonical_label"] == "screwdriver"


def test_geometry_never_invents_a_label_nobody_proposed():
    """A small object whose only hypothesis is a driver does not become a screw.

    The prior scales what the detector proposed and can propose nothing itself,
    so with every available label zeroed the answer is not a fastener.
    """
    observations = [
        _observation(camera, "screwdriver", 0.10)
        for camera in ("LEFT", "RIGHT", "TOP")
    ]
    belief = PersistentInstanceTracker._compute_consensus_semantic_belief(
        observations, fusion_config=FUSION, physical_prior=PRIOR,
        maximum_dimension_m=0.0398)
    assert belief["canonical_label"] != "screw"


def test_an_unmeasured_extent_leaves_the_detector_s_own_answer():
    observations = [_observation("LEFT", "screwdriver", 0.10, [("screw", 0.05)])]
    belief = PersistentInstanceTracker._compute_consensus_semantic_belief(
        observations, fusion_config=FUSION, physical_prior=PRIOR,
        maximum_dimension_m=None)
    assert belief["canonical_label"] == "screwdriver"
    assert _physical_prior_multipliers(PRIOR, None) == {}


def test_association_no_longer_deletes_hypotheses_the_prior_zeroes():
    """Guards the defect directly, at the line that caused it."""
    source = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "workshop_phase1" / "inspection_controller.py").read_text()
    assert "om.semantic_alternatives = list(alternatives)" in source
    assert "filtered_alts" not in source, (
        "the association-time size filter that deleted alternatives is back")
