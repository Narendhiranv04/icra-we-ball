# Three-view standardization evaluation

This directory records the final cross-environment experiment. The code-level
three-view contract and Workshop controlled geometry gate are valid, but the
candidate is **not a frozen production baseline** because Kitchen, Living Room,
and Workshop production perception regressions remain.

## Results

| Environment / gate | Previous | Three-view result |
|---|---:|---:|
| Kitchen canonical feasibility | 15/16 | 11/16 (0.6875) |
| Living Room integrated region function | 13/13 | 6/13 (0.4615) |
| Workshop controlled oracle masks + semantics | 14/14 | 14/14 |
| Workshop YOLO semantics + oracle masks | 0/14 | 0/14 |
| Workshop full production | 1/14 | 0/14 |

Kitchen retains 1.0 infeasible recall and 1.0 distinct soup-assignment
validity, but feasible recall falls to 0.5. Living Room retains 1.0 infeasible
recall and valid three-view measurement provenance, but feasible recall is
0.0. Those are major regressions and are not silently accepted.

## Workshop calibration outcome

The initial target distance falls from a five-view mean of 1.781 m to a
three-view mean of 1.596 m. Drawer oblique views are 0.957 m from their aimed
targets and the downward detail view is 1.234 m away. Cabinet views range from
1.323 to 1.384 m. The robot repair-target distance falls from 1.862 m to
1.402 m with no neutral-pose robot contacts in F0 or F6.

The post-change detector study selected the small World-v2 checkpoint,
confidence 0.075, and NMS IoU 0.35. On the development split, critical-object
recall is 0.25, small-object recall is 0.0, and generic tiling gives no small-
object benefit. In the held-out 14-variant runs, screw, bolt, wrench, and all
other requested Workshop categories have 0.0 matched recall under the frozen
operating point. This shows source framing was not the sole bottleneck.

## Artifact map

- `calibration_summary.json`: pose, resolution, and robot-standoff decisions.
- `workshop_detector/detector_calibration/`: checkpoint, threshold, NMS, and
  tiling comparisons.
- `workshop_controlled_gate.json`: exact 14-row controlled gate.
- `compact/`: full compact benchmark JSON/CSV summaries without multi-gigabyte
  raw RGB-D stages.

The raw untracked RGB-D run trees were removed after their compact summaries
were preserved; they can be regenerated with the commands in the repository
documentation.
