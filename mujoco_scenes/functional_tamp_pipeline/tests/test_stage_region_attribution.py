"""A stage may not claim the contents of the region next door.

Points are gated to the volume of the region being inspected with a boundary
margin, which exists to tolerate calibration error at the edge of a drawer.
The declared volumes are close enough together that the margin reaches into a
neighbour's interior: in the workshop's right-drawer volume, y stops at 0.45
and the tool cabinet's own volume starts at y 0.30, so an 8 cm margin overlaps
it outright.

That is how a scene whose storage contains no fastener at all came to contain
one.  A small blob sitting in the tool cabinet was admitted as an observation
of the right drawer, believed to be a screw, bound as the fastener, and the
task the benchmark declares impossible was reported complete.

Nothing here reads ground truth, a variant identifier, or a scene definition.
The volumes are the ones the controller already drives its cameras to.
"""

from __future__ import annotations

import numpy as np
import pytest

from mujoco_scenes.workshop_phase1.inspection_controller import WorkshopPhase1InspectionController
from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker


RIGHT_DRAWER = (np.array([0.10, -0.20, 0.35]), np.array([0.65, 0.45, 0.75]))
TOOL_CABINET = (np.array([0.10, 0.30, 0.60]), np.array([0.80, 0.95, 1.25]))


def test_an_object_inside_the_next_region_is_not_evidence_from_this_stage():
    tracker = PersistentInstanceTracker()
    verdict = tracker._misattributed_to_this_stage(
        np.array([0.3389, 0.5260, 0.7854]), *RIGHT_DRAWER,
        {"TOOL_CABINET": TOOL_CABINET})
    assert verdict == "TOOL_CABINET"


def test_an_object_inside_this_stage_region_is_kept():
    tracker = PersistentInstanceTracker()
    assert tracker._misattributed_to_this_stage(
        np.array([0.27, -0.0755, 0.5063]), *RIGHT_DRAWER,
        {"TOOL_CABINET": TOOL_CABINET}) is None


def test_the_boundary_margin_still_does_its_job():
    """The adversarial half: just outside this volume, and inside no other."""
    tracker = PersistentInstanceTracker()
    assert tracker._misattributed_to_this_stage(
        np.array([0.27, 0.47, 0.50]), *RIGHT_DRAWER,
        {"TOOL_CABINET": TOOL_CABINET}) is None


def test_with_no_other_declared_regions_nothing_is_rejected():
    tracker = PersistentInstanceTracker()
    assert tracker._misattributed_to_this_stage(
        np.array([9.0, 9.0, 9.0]), *RIGHT_DRAWER, None) is None
    assert tracker._misattributed_to_this_stage(
        np.array([9.0, 9.0, 9.0]), *RIGHT_DRAWER, {}) is None


def test_the_declared_volumes_actually_overlap_once_the_margin_is_applied():
    """The measurement behind the repair, so the reason is not taken on trust."""
    margin = 0.08
    right_max_y = RIGHT_DRAWER[1][1] + margin
    assert right_max_y > TOOL_CABINET[0][1], (right_max_y, TOOL_CABINET[0][1])
    right_max_z = RIGHT_DRAWER[1][2] + margin
    assert right_max_z > TOOL_CABINET[0][2], (right_max_z, TOOL_CABINET[0][2])


def test_every_declared_region_reports_its_neighbours():
    others = WorkshopPhase1InspectionController._other_declared_region_volumes("RIGHT_DRAWER")
    assert set(others) == {"LEFT_DRAWER", "TOOL_CABINET"}
    for minimum, maximum in others.values():
        assert np.all(np.asarray(maximum) > np.asarray(minimum))
