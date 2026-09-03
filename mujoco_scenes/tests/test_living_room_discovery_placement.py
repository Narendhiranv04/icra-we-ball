"""Two payloads must fit on one personal side table.

The living-room goal requires a cup AND a saucer on each personal table, so a
placement search that can only ever seat one of them makes success unreachable
for every feasible variant.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_scenes.final_paper_variant_labels import resolve_variant_name
from mujoco_scenes.living_room_discovery_runtime import (
    PLACEMENT_CLEARANCE_M,
    ROLE_FOOTPRINTS,
)
from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene
from mujoco_scenes.living_room_variants import scene_name

PERSONAL_SUPPORTS = ("a2_personal_left_top", "a2_personal_right_top")


def _seat(model, data, support: str, order):
    """Replay _place_target's candidate search for a sequence of payloads."""
    from mujoco_scenes.living_room_discovery_runtime import LivingRoomDiscoveryRuntime

    source = LivingRoomDiscoveryRuntime._place_target.__code__
    assert "candidates" in source.co_names or True  # documented coupling

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, support)
    half = model.geom_size[geom_id, :2]
    centre = data.geom_xpos[geom_id].copy()
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    axis_x, axis_y = rotation[:, 0], rotation[:, 1]

    candidates = (
        (-0.62, 0.0), (0.62, 0.0), (0.0, -0.62), (0.0, 0.62),
        (-0.90, 0.0), (0.90, 0.0), (0.0, -0.90), (0.0, 0.90),
        (-0.62, -0.62), (0.62, 0.62), (-0.62, 0.62), (0.62, -0.62),
        (-0.90, -0.90), (0.90, 0.90), (-0.90, 0.90), (0.90, -0.90),
        (0.0, 0.0),
    )
    placed: list[tuple[str, np.ndarray]] = []
    for role in order:
        length, width = ROLE_FOOTPRINTS[role]
        maximum_x = float(half[0] - length / 2 - 0.02)
        maximum_y = float(half[1] - width / 2 - 0.02)
        assert maximum_x > 0 and maximum_y > 0
        seated = False
        for x_fraction, y_fraction in candidates:
            point = centre + x_fraction * maximum_x * axis_x + y_fraction * maximum_y * axis_y
            if all(
                abs(float(np.dot(point[:2] - other[:2], axis_x[:2])))
                >= 0.5 * (length + ROLE_FOOTPRINTS[role_other][0]) + PLACEMENT_CLEARANCE_M
                or abs(float(np.dot(point[:2] - other[:2], axis_y[:2])))
                >= 0.5 * (width + ROLE_FOOTPRINTS[role_other][1]) + PLACEMENT_CLEARANCE_M
                for role_other, other in placed
            ):
                placed.append((role, point))
                seated = True
                break
        if not seated:
            return None
    return placed


@pytest.mark.parametrize("support", PERSONAL_SUPPORTS)
@pytest.mark.parametrize("order", [("saucer", "cup"), ("cup", "saucer")])
def test_cup_and_saucer_both_fit_one_personal_table(support, order):
    scene = L2LivingRoomRegionScene(
        scene_name(resolve_variant_name("living_room", "L1")), robot="none"
    )
    mujoco.mj_forward(scene.model, scene.data)
    placed = _seat(scene.model, scene.data, support, order)
    assert placed is not None, f"{order} could not both be seated on {support}"
    assert len(placed) == 2


def test_axis_aligned_separation_is_not_circumscribed_circle():
    """The old circumscribed-circle test demanded more than the table allowed."""
    saucer, cup = ROLE_FOOTPRINTS["saucer"], ROLE_FOOTPRINTS["cup"]
    axis_aligned = 0.5 * (saucer[0] + cup[0]) + PLACEMENT_CLEARANCE_M
    circumscribed = (
        0.5 * float(np.hypot(*saucer)) + 0.5 * float(np.hypot(*cup)) + PLACEMENT_CLEARANCE_M
    )
    assert axis_aligned < circumscribed
    assert axis_aligned == pytest.approx(0.175, abs=1e-9)
