# GT_everything

Ground-truth evidence for the Kitchen, Living Room, and Workshop environments.

Current counts are 16 Kitchen variants, 13 Living Room variants, and 10 redesigned Workshop variants: 39 total. Kitchen and Living Room retain their existing MP4/action/assignment bundles.

Workshop was restructured on 2026-08-23. Its old 14-variant folders and videos were retired. Every current Workshop variant contains:

- `robot_action_sequence.txt`
- `function_object_assignments.txt`
- `five_camera_scene_snapshot.png`
- `execution_summary.json`

The Workshop physical validation suite passes all 10 variants and 266/266 actions. The eight feasible variants execute complete manual- or power-driver insertion logic; the two infeasible variants exhaustively inspect and terminate with the exact missing-role reason. New MP4 rendering is deliberately separate from this validation pass and has not been substituted with the obsolete videos.

See `workshop/manifest.json` for the redesigned Workshop index and the root `manifest.json` for the combined environment index.
