# Cross-environment three-view RGB-D standard

## Frozen observation contract

Each observation stage acquires three calibrated RGB-D viewpoints: symmetric
left- and right-isometric views and a region-oriented detail view. The roles
are `ISO_LEFT`, `ISO_RIGHT`, and `DETAIL`. They are observations, not a
requirement for three physical cameras; one movable calibrated sensor could
acquire them sequentially.

The same aligned RGB-D observations support both perception branches. RGB is
passed to YOLO and semantic consensus, while accepted masked depth pixels are
back-projected into the world frame and fused for geometry. An object may
contribute valid semantic evidence from one view. Geometry that depends on
multi-view coverage is trusted only when at least two viewpoints contribute;
otherwise the corresponding property remains `UNKNOWN`. No object is required
to appear in all three views.

`DETAIL` depends only on access geometry:

- horizontal/open-top region: high, downward view;
- vertical/front-access region: elevated frontal view;
- general initial work area: top-front hybrid.

These are fixed relative templates. The inspection policy selects what region
to inspect; it does not learn or plan viewpoints.

## Environment schedules

| Environment | Initial stage | Fresh hidden-region stages |
|---|---|---|
| Kitchen | three work-area views | three region-centred views after D1, D2, C2, B1, or C1 is opened |
| Living Room | three room views | none in the canonical integrated region-function task |
| Workshop | three work-area views | three region-centred views after LEFT_DRAWER, RIGHT_DRAWER, or TOOL_CABINET is opened |

Opening a Kitchen or Workshop storage region triggers exactly one fresh
region-centred observation stage. Earlier evidence remains in the persistent
observed-state graph. Closed hidden storage is never inspected with a
region-specific rig.

## Workshop calibration

Camera calibration uses the development scenes F0, F1, and F5, with F3/F4
used only where drawer distribution or local interface visibility is needed.
Privileged segmentation and scene geometry are evaluator-only signals. The
frozen YAML is applied unchanged to every variant; no pose depends on contents
or the expected outcome.

The Workshop robot uses the Workshop-owned base pose
`[0, -0.75, 0.06205]`. The shared `GOOGLE_BASE_POSE` remains unchanged. The
inspection viewpoints are virtual calibrated sensor poses and are not inferred
from the robot base or claimed to be products of motion planning.

## Resolution and detector policy

The canonical Workshop source resolution remains 1280×720 after the viewpoint
change. Kitchen and the integrated Living Room benchmark retain 1280×960 to
avoid changing their established 4:3 measurement calibration in the same
experiment. YOLO inference size is distinct from source capture resolution.

The FM/manual contract runs once. At `INITIAL`, its ranked object and region
categories are used. Hidden-storage stages preserve that ranking but select
only object categories; there is no second FM request. The historical
`SINGLE_FRONT_VIEW` spelling is retained as a CLI compatibility alias and now
means the explicit canonical `DETAIL` view.

The post-viewpoint development calibration freezes the small World-v2
checkpoint at confidence 0.075 and NMS IoU 0.35. Its critical-object recall is
0.25 and its small screw/bolt recall remains 0.0; this is a measured limitation,
not a successful resolution of the detector bottleneck. Generic 2×2 tiling did
not improve small-object recall and remains disabled.

## Reproduction

The canonical configs are:

- `mujoco_scenes/configs/inspection_rigs.yaml`
- `mujoco_scenes/configs/l2_integrated_region_function_rig.yaml`
- `mujoco_scenes/configs/workshop_inspection_rigs.yaml`

Calibration and benchmark evidence is recorded under
`mujoco_scenes/benchmark_reports/three_view_standardization/`.
