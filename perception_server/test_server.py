import unittest

import numpy as np

from perception_server.server import (
    Detection,
    deduplicate,
    encode_rle,
    normalize_prompts,
)


def decode_rle(payload):
    flat = np.zeros(payload["height"] * payload["width"], dtype=bool)
    offset = 0
    value = False
    for count in payload["counts"]:
        if value:
            flat[offset : offset + count] = True
        offset += count
        value = not value
    return flat.reshape(payload["height"], payload["width"])


class ServerTests(unittest.TestCase):
    def test_rle_round_trip(self):
        mask = np.array([[False, True, True], [False, False, True]])
        np.testing.assert_array_equal(decode_rle(encode_rle(mask)), mask)

    def test_duplicate_prompt_masks_keep_higher_score(self):
        mask = np.array([[True, False], [False, False]])
        detections = [
            Detection("tool", 0.7, mask),
            Detection("utensil", 0.9, mask),
        ]
        kept = deduplicate(detections, threshold=0.85)
        self.assertEqual([(item.label, item.score) for item in kept], [("utensil", 0.9)])

    def test_prompt_contract_rejects_string_as_array(self):
        with self.assertRaisesRegex(ValueError, "array"):
            normalize_prompts("spoon")

    def test_prompt_contract_strips_and_rejects_non_strings(self):
        self.assertEqual(normalize_prompts([" spoon ", "", "mug"]), ["spoon", "mug"])
        with self.assertRaisesRegex(ValueError, "only strings"):
            normalize_prompts(["spoon", 3])


if __name__ == "__main__":
    unittest.main()
