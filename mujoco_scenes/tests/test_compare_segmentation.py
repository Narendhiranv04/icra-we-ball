import unittest
from types import SimpleNamespace

import numpy as np

from mujoco_scenes.compare_segmentation import compare_runs, mask_iou


def run_with_masks(masks_by_camera):
    captures = {
        camera: SimpleNamespace(instance_masks=masks)
        for camera, masks in masks_by_camera.items()
    }
    return SimpleNamespace(inspection=SimpleNamespace(cameras=captures))


class CompareSegmentationTests(unittest.TestCase):
    def test_mask_iou(self):
        left = np.array([[True, True], [False, False]])
        right = np.array([[True, False], [True, False]])
        self.assertAlmostEqual(mask_iou(left, right), 1 / 3)

    def test_comparison_matches_masks_one_to_one(self):
        mask = np.array([[True, False], [False, False]])
        empty = np.array([[False, False], [False, True]])
        oracle = run_with_masks({"front": {"hidden_name": mask}})
        learned = run_with_masks(
            {"front": {"object_0001": mask, "object_0002": empty}}
        )
        report = compare_runs(oracle, learned)
        self.assertEqual(report["aggregate"]["matched_masks"], 1)
        self.assertEqual(report["aggregate"]["recall"], 1.0)
        self.assertEqual(report["aggregate"]["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
