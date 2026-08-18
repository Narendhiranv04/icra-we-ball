import unittest

import numpy as np

from mujoco_scenes.perception import SegmentedInstance, validate_segmentations


class PerceptionTests(unittest.TestCase):
    def test_validates_and_normalizes_backend_masks(self):
        result = validate_segmentations(
            [
                SegmentedInstance(
                    "object_1", np.array([[0, 1], [1, 0]]), " spoon ", 0.8
                )
            ],
            image_shape=(2, 2),
        )
        self.assertEqual(result[0].label, "spoon")
        self.assertEqual(result[0].mask.dtype, np.bool_)

    def test_rejects_duplicate_ids_and_wrong_shapes(self):
        duplicate = [
            SegmentedInstance("x", np.ones((2, 2))),
            SegmentedInstance("x", np.ones((2, 2))),
        ]
        with self.assertRaisesRegex(ValueError, "repeated"):
            validate_segmentations(duplicate, image_shape=(2, 2))
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_segmentations(
                [SegmentedInstance("x", np.ones((1, 2)))],
                image_shape=(2, 2),
            )


if __name__ == "__main__":
    unittest.main()
