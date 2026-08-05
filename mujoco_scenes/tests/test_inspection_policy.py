import unittest

from mujoco_scenes.inspection_policy import (
    FixedInspectionPolicy,
    RandomInspectionPolicy,
    RankedInspectionPolicy,
)


class InspectionPolicyTests(unittest.TestCase):
    def test_fixed_policy_skips_unavailable_regions(self):
        policy = FixedInspectionPolicy(("D1", "D2", "C1"))
        self.assertEqual(policy.choose(("D2", "C1"), ("D1",)), "D2")

    def test_random_policy_is_reproducible(self):
        first = RandomInspectionPolicy(seed=9)
        second = RandomInspectionPolicy(seed=9)
        regions = ("D1", "D2", "C1")
        self.assertEqual(first.choose(regions, ()), second.choose(regions, ()))

    def test_ranked_policy_rejects_invented_regions(self):
        policy = RankedInspectionPolicy(
            lambda available, inspected: ("hidden_region", *available)
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            policy.choose(("D1", "D2"), ())


if __name__ == "__main__":
    unittest.main()
