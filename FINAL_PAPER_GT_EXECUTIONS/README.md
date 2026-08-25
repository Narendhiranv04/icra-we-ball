# Final paper GT executions

This is the single packaged evidence tree for Kitchen, Living Room, and
Workshop. Each variant contains a natural-speed merged five-camera MP4, the GT
action sequence, function/object assignments, initial object/region placement,
execution summary, and wall-clock timings with and without recording.

Each directory also contains the predeclared expected GT actions, the actions
observed in the physical execution trace, and an exact sequence-comparison
report. Packaging fails if those GT task sequences differ. Living Room permits
planner-inserted MOVE actions between its frozen PICK/PLACE task actions; those
navigation actions remain visible in the executed trace and are excluded only
from the exact GT task-order comparison.

Paper-facing directories and manifest entries use K1-K12, L1-L10, and W1-W10.
The descriptive implementation identifier is retained as `source_variant` in
the manifest and `internal_variant` in timing metadata for reproducibility.

Videos are captured directly during physics execution at the configured FPS.
No frame dropping for speed-up, FFmpeg `setpts`, or postprocessing time scaling
is applied. Feasible variants execute their complete GT task. Infeasible
Kitchen and Workshop variants execute their available inspection/rejection
sequence. Living Room infeasible variants are rejected before Phase-2 physical
planning, so their evidence explicitly records a no-manipulation termination
instead of fabricating robot actions.
