"""Why the size prior reads the fused extent, and why a per-view median cannot.

The prior tells a small fastener from a long driver: below the declared
threshold the fastener's label is boosted and the driver's zeroed, above it the
reverse.  It is only as good as the extent it is given, and it is given the
bounding extent of the fused cloud -- the union of every view's mask.

That has a known cost.  One mask straying onto a neighbour makes the whole track
measure large, which is how a drawer holding a screw and a driver came back
holding two drivers and no screw, leaving the fastener role nothing to bind.

The obvious repair is to measure per view and take the middle, so one straying
mask cannot decide.  It was implemented and replayed over all thirty archived
workshop trials, and it is wrong: workshop successes fell from ten to six and
*drivers* began disappearing.  These tests pin the reason, so the next attempt
starts from the real constraint rather than repeating this one.
"""
from __future__ import annotations

import numpy as np

from mujoco_scenes.workshop_phase1.tracking import (
    PersistentInstanceTracker, _physical_prior_multipliers,
)

WORKSHOP_PRIOR = {
    "enabled": True,
    "small_object_max_dimension_m": 0.08,
    "small_object": {"screw": 2.0, "screwdriver": 0.0, "power_driver": 0.0},
    "large_object": {"screw": 0.0, "screwdriver": 2.0},
}


def _cloud(length_m: float, points: int = 40) -> np.ndarray:
    """A thin cloud of the given length along x."""
    along = np.linspace(0.0, length_m, points)
    return np.stack([along, np.zeros(points), np.zeros(points)], axis=1)


def test_a_view_that_sees_part_of_a_long_object_measures_it_short():
    """The constraint that defeats a per-view estimator.

    A driver lying in a drawer is partly occluded, and a close inspection view
    frames only the shank it can see.  That view measures well under the
    threshold on its own, so a median over views can call a driver small -- and
    the prior then zeroes the only labels a driver has.  Under the threshold the
    driver's own labels score zero, which is what made drivers vanish.
    """
    partial_view = _cloud(0.035)
    measured = PersistentInstanceTracker._maximum_dimension_m(partial_view)
    assert measured is not None and measured < 0.08
    multipliers = _physical_prior_multipliers(WORKSHOP_PRIOR, measured)
    assert multipliers["screwdriver"] == 0.0
    assert multipliers["power_driver"] == 0.0


def test_the_fused_extent_keeps_a_partly_seen_long_object_large():
    """Why the union is used: any view that does see the whole thing rescues it.

    The same driver, with one view that frames all of it.  The fused extent is
    large, the driver's labels keep their weight, and this is the behaviour the
    ten passing workshop trials depend on.
    """
    per_camera = {"close": _cloud(0.035), "left": _cloud(0.040),
                  "front": _cloud(0.038), "right": _cloud(0.195)}
    fused = np.vstack(list(per_camera.values()))
    measured = PersistentInstanceTracker._maximum_dimension_m(fused)
    assert measured is not None and measured > 0.08
    assert _physical_prior_multipliers(WORKSHOP_PRIOR, measured)["screwdriver"] == 2.0


def test_the_prior_buckets_on_the_declared_threshold():
    small = _physical_prior_multipliers(WORKSHOP_PRIOR, 0.030)
    large = _physical_prior_multipliers(WORKSHOP_PRIOR, 0.200)
    assert (small["screw"], small["screwdriver"]) == (2.0, 0.0)
    assert (large["screw"], large["screwdriver"]) == (0.0, 2.0)


def test_no_prior_and_unmeasured_extent_change_nothing():
    """Wherever the constraint is unknown the fusion is left exactly as it was."""
    assert _physical_prior_multipliers(None, 0.03) == {}
    assert _physical_prior_multipliers(WORKSHOP_PRIOR, None) == {}
    assert _physical_prior_multipliers({"enabled": False}, 0.03) == {}
    assert PersistentInstanceTracker._maximum_dimension_m(np.zeros((1, 3))) is None
