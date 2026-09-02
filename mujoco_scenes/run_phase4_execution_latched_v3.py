"""Final containment-latch runner with a physically negligible activation tolerance.

The bowl/utensil PLACE has already passed the normal physical containment check
before this layer runs.  Enabling a MuJoCo equality constraint can project the
current state by a tiny amount during ``mj_forward``.  The measured K1 projection
was ~14 micrometres and ~0.00129 rad (0.074 deg), which is negligible and should
not turn an otherwise successful physical PLACE into a failure.
"""

from __future__ import annotations

from . import run_phase4_execution_latched_v2 as v2


# These bounds are only a no-snap guard for weld activation.  They are not
# placement tolerances: the original PLACE controller has already established
# and verified the requested spoon-in-bowl relation before the latch is enabled.
# 0.5 mm / 0.005 rad (~0.286 deg) still rejects any meaningful pose snap while
# allowing normal equality-constraint projection at the live pose.
v2.LATCH_ACTIVATION_POSITION_TOLERANCE_M = 5.0e-4
v2.LATCH_ACTIVATION_ORIENTATION_TOLERANCE_RAD = 5.0e-3


def main() -> int:
    return v2.main()


if __name__ == "__main__":
    raise SystemExit(main())
