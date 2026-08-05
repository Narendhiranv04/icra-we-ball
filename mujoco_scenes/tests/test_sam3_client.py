import unittest

import numpy as np

from mujoco_scenes.sam3_client import Sam3HttpSegmenter, decode_rle


class Sam3ClientTests(unittest.TestCase):
    def test_decode_rle(self):
        result = decode_rle({"height": 2, "width": 3, "counts": [1, 2, 2, 1]})
        expected = np.array([[False, True, True], [False, False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_client_sends_pixels_and_prompts(self):
        received = {}

        def transport(path, payload):
            received.update(payload)
            self.assertEqual(path, "/v1/segment")
            return {
                "instances": [
                    {
                        "instance_id": "spoon_001",
                        "label": "spoon",
                        "score": 0.9,
                        "mask": {"height": 2, "width": 3, "counts": [1, 2, 3]},
                    }
                ]
            }

        segmenter = Sam3HttpSegmenter("http://unused", transport=transport)
        instances = segmenter.segment(
            np.zeros((2, 3, 3), dtype=np.uint8),
            camera_id="front",
            prompts=["spoon"],
        )
        self.assertEqual(received["camera_id"], "front")
        self.assertEqual(received["prompts"], ["spoon"])
        self.assertIn("image_png_base64", received)
        self.assertEqual(instances[0].label, "spoon")
        self.assertEqual(instances[0].mask.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
