import numpy as np
import pytest

from mujoco_scenes.kitchen_phase_c_execution import KitchenPhaseCExecutionDispatcher
from mujoco_scenes.sequential_inspection import INTERFERING_OPEN_REGIONS


def test_c2_then_b1_keeps_c2_open_by_contract():
    # Opening C2 may repair the unsupported reverse order by closing B1 first,
    # but opening B1 after C2 must not imply CLOSE(C2).
    assert INTERFERING_OPEN_REGIONS == {"C2": "B1"}
    assert INTERFERING_OPEN_REGIONS.get("B1") is None


@pytest.mark.parametrize("observed_length_m", [0.16, 0.20, 0.24])
def test_serving_utensil_orientation_is_vertical_first_insertion(observed_length_m):
    normal = np.array((0.0, 0.0, 1.0))
    local_axis = np.array((0.0, 0.0, 1.0))
    candidates = KitchenPhaseCExecutionDispatcher._serving_utensil_orientation_family(
        np.eye(3),
        local_axis,
        normal,
        np.array((1.0, 0.0, 0.0)),
        observed_length_m=observed_length_m,
    )

    assert candidates
    assert candidates[0]["inclination_deg"] == 0.0
    assert {
        row["inclination_deg"] for row in candidates
    } == {0.0, 3.0, 5.0}
    assert {
        row["provenance"] for row in candidates
    } == {"SERVING_UTENSIL_INSERTION_AXIS_FAMILY"}

    for row in candidates:
        axis = row["rotation"] @ local_axis
        expected = np.cos(np.deg2rad(row["inclination_deg"]))
        assert float(np.dot(axis, normal)) == pytest.approx(expected, abs=1e-7)
        # No candidate may encode the old 12-45 degree rim-resting family.
        assert row["inclination_deg"] <= 5.0
