import tempfile
import unittest

from mujoco_scenes.observed_geometry import ObservedGeometryState


def object_record(*, width=None, length=None, opening=None, depth=None, cavity=True):
    def value(number):
        return {"value": number, "status": "DERIVED", "unit": "m"}

    return {
        "geometric_properties": {
            "maximum_cross_section_m": value(width),
            "usable_length_m": value(length),
            "opening_width_m": value(opening),
            "cavity_depth_m": value(depth),
        },
        "geometric_predicates": {
            "OPEN_CAVITY": {"status": "TRUE" if cavity else "FALSE"}
        },
        "measurement_cloud_path": "stage/fused.ply",
    }


class ObservedGeometryTests(unittest.TestCase):
    def test_ranked_selection_stops_at_first_compatible_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            state = ObservedGeometryState(root, scene_name="test")
            state.registry["objects"] = {
                "wide": object_record(width=0.10, length=0.20),
                "spoon": object_record(width=0.02, length=0.20),
                "jar": object_record(opening=0.06, depth=0.10),
            }
            selected = state.select_first_compatible(
                ["wide", "spoon"],
                target_id="jar",
                required_relations=["INSERTABLE_IN", "REACHES_BOTTOM"],
            )
            self.assertEqual(selected.status, "COMPLETE")
            self.assertEqual(selected.selected_object_id, "spoon")
            self.assertEqual(len(selected.evaluations), 2)


if __name__ == "__main__":
    unittest.main()
