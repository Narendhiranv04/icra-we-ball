# Final paper GT executions

This is the single packaged evidence tree for Kitchen, Living Room, and
Workshop. Each variant contains a natural-speed merged five-camera MP4, the GT
action sequence, function/object assignments, initial object/region placement,
execution summary, and wall-clock timings with and without recording.

Videos are captured directly during physics execution at the configured FPS.
No frame dropping for speed-up, FFmpeg `setpts`, or postprocessing time scaling
is applied. Feasible variants execute their complete GT task. Infeasible
Kitchen and Workshop variants execute their available inspection/rejection
sequence. Living Room infeasible variants are rejected before Phase-2 physical
planning, so their evidence explicitly records a no-manipulation termination
instead of fabricating robot actions.
