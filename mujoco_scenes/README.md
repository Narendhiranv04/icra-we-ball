# MuJoCo Kitchen Scene S1

This package builds the `S1_coffee_missing_mug` scene dynamically from the
shared kitchen XML, object library, and YAML configuration.

## Object meshes

The scene uses textured YCB meshes for the coffee can, sugar box, mug, cup,
plate, bowl, fork, spoon, knife, and tea-box proxy. Google Scanned Objects
provide the kettle/teapot, coffee jar, sugar jar, and two S1 distractors.
Stirrer and folded-napkin meshes are authored locally. Because the GSO catalog
does not contain a tong scan, `tongs` uses a purpose-built local mesh instead
of an incorrectly relabelled scanned object.

Prepared OBJ/PNG files are included in the repository and Docker build
context. To recreate them and refresh the provenance hashes:

```bash
python mujoco_scenes/scripts/prepare_object_assets.py --force
```

Visual scans are paired with simple invisible collision proxies. Drinkware,
plates, bowls, and the coffee jar use segmented hollow shells rather than
solid cylinders, so utensils can physically enter and rest in them. The
Nescafe scan is prepared as an open jar with a visible coffee-powder surface;
its powder is visual-only so later symbolic `pour` actions are not obstructed.
Exact dataset IDs, URLs, and SHA-256 hashes are recorded in
`assets/objects/meshes/manifest.json`; licensing and attribution are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Drawer contents remain graspable free bodies. Smoother, damped drawer motion
and a flat high-friction stirrer proxy keep their reset arrangement nearly
fixed while a drawer opens, without introducing weld constraints that a later
grasp action would have to disable.

## Six camera views

- `left_shoulder_camera`
- `right_shoulder_camera`
- `overhead_camera`
- `side_camera`
- `wrist_camera`
- `front_camera`

Google Robot contributes `head_camera_rgb`. It is available from the same
`--camera` CLI option whenever the robot is enabled.

The interactive viewer starts with MuJoCo's free camera by default. Pass an
explicit `--camera NAME` to start from one of the fixed or robot-mounted
cameras instead.

With Google Robot enabled, `wrist_camera` is attached to its gripper
link and `head_camera_rgb` is robot-mounted.
The kitchen-only `--no-robot` mode uses a fixed placeholder with the same name.

Google Robot assets are loaded from an external MuJoCo Menagerie checkout. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licenses.

## Run natively with uv

Create the Python environment from the repository root:

```bash
uv venv --python 3.11
uv pip install -r mujoco_scenes/requirements.txt
```

Sparse-clone the official Google Robot Menagerie directory beside the repository:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git \
  ../third_party/mujoco_menagerie
git -C ../third_party/mujoco_menagerie sparse-checkout set google_robot
```

Then launch the scene. Google Robot is the default backend:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot google --viewer
```

Set `MUJOCO_MENAGERIE_PATH` when the checkout lives elsewhere. The Google
backend supports scene loading, cameras, joint targets, collision-checked base
navigation, and an S1-calibrated vertical sugar-jar pick/place at
`serving_spot`. It also ports main's far-tip spoon pick, passive bowl-down
hang, and secured carry; spoon placement remains gated. Other scene poses plus
coffee-jar and kettle actions remain gated. See
[ROBOT_CALIBRATION.md](ROBOT_CALIBRATION.md) for the calibration and acceptance
process used for this and future robot backends.

## Run the interactive viewer in Docker (Linux/X11)

From the repository root:

```bash
docker build -t mujoco-kitchen-s1 -f mujoco_scenes/Dockerfile .
xhost +local:docker
docker run --rm -it \
  -e DISPLAY="$DISPLAY" \
  -e MUJOCO_GL=glfw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug --viewer --camera front_camera
xhost -local:docker
```

The same viewer can be started with Compose after allowing the local Docker
X11 connection:

```bash
xhost +local:docker
docker compose up --build kitchen-s1
```

Use the MuJoCo viewer's camera selector to switch between all six cameras.
To start with C1 already open, append `--open-container C1` to the `docker run`
command. The actuator controls in the viewer UI can also move every door,
drawer, and the box lid.

With Google Robot enabled, `--viewer` also opens the companion
`Actions` panel.
Choose `Actions` → `Move` → `Home`, `Cupboard 1`, `Cupboard 2`, or `Box` to run
one collision-checked RRT* base motion. `Cupboard 2` and `Box` are symbolic
aliases of the same right-side pose.
Google uses the same navigation controls and exposes its validated sugar-jar
pick/place plus far-tip spoon pick/carry. The spoon uses a live rotational
pivot to settle bowl-down before its transport weld is restored; its Place
control remains disabled. Unsupported objects are deliberately not shown as
actionable. Google parks on a farther navigation line, automatically
approaches the S1 manipulation stance for pick/place, and retracts the base and
arm before enabling another Move. The held-object carry state is included in
RRT*/rotation collision checks. Google IK additionally validates dense joint
segments using signed visual-geometry distances, catching arm/body clipping
that MuJoCo's filtered contact solver does not expose, with a live execution
guard for tracking deviations. Arm, base, and gripper targets are slew-limited
to avoid abrupt position-control commands and reduce visible wobble.
Add `--no-actions-panel` for the original viewer without the companion controls.

## Render without opening a GUI

```bash
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD:/output" \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug \
  --camera overhead_camera \
  --render /output/s1_overhead.png
```

## Five-view object point clouds and timing

The geometry checker uses the five fixed cameras (`left_shoulder`, `right_shoulder`,
`overhead`, `side`, and `front`). For every currently catalogued object instance
it renders RGB, metric depth, and MuJoCo geometry segmentation; back-projects
masked pixels; transforms them through the live camera pose into the MuJoCo world
frame; and voxel-fuses the five observations.

To load a deterministic perception snapshot with C1, C2, D1, D2, and B1 all
open, run the complete reconstruction, print stage timings, and export colored
PLY files:

```bash
mkdir -p point_cloud_runs
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD/point_cloud_runs:/output" \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug \
  --no-robot \
  --open-all \
  --point-cloud /output/all_open
```

`manifest.json` contains the total time, per-stage times, point count per
object, and the number of contributing pixels from each camera.
`all_visible_objects.ply` is the combined colored cloud; each object also gets
its own PLY. Open these files in MeshLab, CloudCompare, or Open3D.

The interactive `Actions` panel has a `Geometry` button for benchmarking at
the initial region-facing pose. A fresh region-facing reconstruction is also
run automatically after a physical box, drawer, or cupboard opening.
Interactive persistent observed-state outputs are written under `runs/`.
The legacy `--open-all --point-cloud` benchmark above remains scene-wide and
does not feed the property or task evaluator.

## Persistent registry, observed graph, and task witness

Sequential inspection begins with a fresh virtual five-camera observation
while every region is closed. It then directly actuates exactly one requested
container, settles the scene, positions the virtual rig toward that open
interior, validates all views, and captures fresh evidence. It does not load a
robot model or execute base, gripper, IK, navigation, or manipulation motion:

```bash
mkdir -p runs
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD/runs:/output" \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug \
  --no-robot \
  --task-requirements configs/s1_find_open_receptacle.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root /output \
  --run-id open_receptacle_region_evidence_demo \
  --point-cloud-width 320 \
  --point-cloud-height 240
```

The fixed default order is `D1 → D2 → C2 → B1 → C1`. With
`--stop-on-complete`, both `INCOMPLETE` and `INDETERMINATE` continue;
`COMPLETE` stops immediately. Completion searches the cached validated
properties of every object in the global registry. A valid object measured at
stage 0 may therefore complete the task; the match is not required to be new
or to come from the most recently opened region.

The pipeline intentionally keeps three point-cloud concepts separate:

- `MeasurementEvidence`: fresh, region-gated points fused from only the
  current five views. This is the sole property-extractor input.
- Global object memory: all generic object IDs legitimately observed so far,
  with the most recent validated property record cached per ID.
- `cumulative_visualization.ply`: historical points retained only for display
  and debugging. Its purpose marker is
  `CUMULATIVE_VISUALIZATION_NOT_MEASUREMENT`.

`MeasurementEvidence` is a required typed API input. Raw arrays and paths
named `cumulative.ply`, `cumulative_visualization.ply`,
`combined_cloud.ply`, or `all_visible_objects.ply` are explicitly rejected.
Objects visible outside the current region volume are saved as rejected debug
evidence but are not discovered, counted, merged, or re-measured in that
stage.

Each run contains atomically updated current state, append-only events,
historical visualization clouds, and one immutable directory per observation:

```text
runs/open_receptacle_region_evidence_demo/
├── run_config.json
├── events.jsonl
├── object_registry.json
├── observed_graph.json
├── latest_witness.json
├── graph_growth.gif
├── graph_growth.mp4                  # when FFmpeg is available
├── objects/<object_id>/
│   ├── cumulative_visualization.ply  # never measurement input
│   ├── cumulative.ply                # compatibility alias, same restriction
│   └── properties.json
└── stages/
    ├── 000_initial/
    │   ├── inspection_metadata.json
    │   ├── inspection_quality.json
    │   ├── region_combined_cloud.ply # stage-local accepted evidence
    │   ├── combined_cloud.ply        # cumulative visualization snapshot
    │   ├── evidence/<object_id>/
    │   │   ├── fused.ply             # valid measurement input
    │   │   ├── properties.json
    │   │   └── quality.json
    │   ├── cameras/<camera_id>/
    │   │   ├── rgb.png
    │   │   ├── depth.png             # uint16 millimetres
    │   │   ├── segmentation.png
    │   │   ├── cloud.ply
    │   │   └── camera_metadata.json
    │   ├── properties.json
    │   ├── graph.json
    │   ├── witness.json
    │   ├── pointcloud.png
    │   ├── graph.png
    │   └── overview.png
    └── 001_after_D1/ ...
```

`configs/inspection_rigs.yaml` defines `INITIAL`, `D1`, `D2`, `C2`, `B1`,
and `C1` target/rig poses, five deterministic relative views, near/far depth,
inspection AABB and margin, settle steps, view-quality thresholds, erosion,
depth-edge rejection, voxel/outlier filtering, and evidence acceptance.
`inspection_metadata.json` records the resolved camera poses, intrinsics,
capture resolution, volume, and accepted/rejected camera/object diagnostics.

For geometry-only task documents, witness inference remains strictly
geometry-only. Simulator categories, object families, semantic function
tables, and category-bearing instance names are absent from that inference
path. Persistent IDs are generic (`object_0001`, ...), and raw simulator
instance names are stored only as one-way association hashes.

Every valid stage-local evidence cloud receives the same property and
predicate schema. It includes
robust OBB dimensions, length and cross-section, extent ratios, planarity,
support area/thickness/normal, and conservative visible rim/opening/cavity
measurements. Structural predicates are `OPEN_CAVITY`, `ELONGATED_OBJECT`, and
`PLANAR_SUPPORT`. `ELONGATED_OBJECT` is scale-independent: it checks whether
the largest robust principal extent dominates the next-largest extent and
does not impose an absolute length. `OPEN_CAVITY` checks the structural
conjunction of an enclosed rim, an open centre, and observed interior surfaces
below the rim. Absolute resolution floors yield `UNKNOWN`; they do not define
a small resolved object as `FALSE`. Their thresholds and relation margins are in
`configs/geometry_inference.yaml`; the file contains no category mappings.
Unavailable evidence remains `UNKNOWN`.
Every record also carries `source_stage`, `source_region`,
`measurement_cloud_path`, contributing camera IDs, point count, method,
extractor version, and the `MEASUREMENT_EVIDENCE` purpose marker.

Task roles declare qualitative unary predicates and task-specific numeric or
relational requirements. Absolute tool/container suitability is expressed by
binary relations rather than folded into `ELONGATED_OBJECT` or `OPEN_CAVITY`.
The graph records `SATISFIES_GEOMETRY` edges with the complete
measurement evidence. `INSERTABLE_IN` and `REACHES_BOTTOM` use generic
cross-section, opening, usable-length, and cavity-depth measurements. The
solver returns `COMPLETE` only for a globally distinct assignment whose role
checks and pairwise checks are all `TRUE`. A possible assignment containing
any unknown geometric evidence is `INDETERMINATE`.

The universal `geometric_properties` keys are:

```text
total_length_m             usable_length_m
maximum_cross_section_m    elongation_ratio
flatness_ratio             dominant_plane_normal_world
planarity_score            support_length_m
support_width_m            support_thickness_m
support_area_m2            opening_width_m
opening_length_m           cavity_depth_m
```

A role requirement contains no class name:

```yaml
roles:
  planar_support:
    count: 1
    geometric_requirements:
      - predicate: PLANAR_SUPPORT
        required_status: TRUE
      - property: support_area_m2
        minimum: 0.008
        unit: m2
        allowed_statuses: [DERIVED]
```

A reliable scene-level geometry-only early-stop demonstration searches for a
horizontal planar support:

```bash
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD/runs:/output" \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug \
  --no-robot \
  --task-requirements configs/s1_find_planar_support.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root /output \
  --run-id s1_geometry_planar_support
```

At 320×240 or the default 640×480 resolution, the C2 support object satisfies
the configured measured planarity, upward-normal, thickness, and area checks;
the run becomes `COMPLETE` after C2 and leaves B1/C1 unopened.

`configs/s1_find_open_receptacle.yaml` demonstrates conservative cavity
reasoning. `OPEN_CAVITY=TRUE` requires adequate multi-view support, vertical
extent, a gravity-aligned and mostly enclosed upper rim, low central top
occupancy, observed interior points below the rim from multiple cameras,
positive cavity depth, and planarity/noise rejection. Missing central depth
alone is never an opening. If these observations are inadequate, the result is
`UNKNOWN` even when a simulator-private asset name would identify the object.
No semantic label is allowed to rescue that geometry-only result.

## Joint RGB semantic and point-cloud geometric grounding

The joint-grounding milestone adds an independent RGB detector path without
changing the measurement-evidence guarantees above. YOLO-World processes only
rendered RGB pixels and the configurable open vocabulary. MuJoCo instance
masks are used only after detection to associate boxes with generic persistent
object IDs. Body, geom, mesh, asset, scene-instance, and simulator category
names are never supplied to the detector.

The primary task is “find a suitable tool for stirring the contents of the
bowl.” Its declarative, FM-ready manual specification is
`configs/stir_contents_joint.yaml`:

| Role | Semantic gate | Unary geometry | Relations |
|---|---|---|---|
| `mixing_container` | bowl, rank 1 | `OPEN_CAVITY` | relation target |
| `mixing_tool` | spoon rank 1; fork rank 2; spatula rank 3 | `ELONGATED_OBJECT` | `INSERTABLE_IN` and `REACHES_BOTTOM` |

The resolver enumerates distinct role assignments. It first rejects any
assignment with a false/unknown required semantic, unary, or relational check.
Only complete valid assignments are ranked: semantic rank, then detector
confidence within the same rank, then persistent object ID. It stops at the
highest-ranked currently observed valid assignment; it does not search for a
hypothetical unseen higher-ranked object.

Three separate scene variants make the claims measurable:

| Scene | Initial counter | D1 | D2 | Expected joint result |
|---|---|---|---|---|
| `S1_joint_stir_counterexamples` | mixing bowl + YCB marker | physically oversized YCB spoon | normal YCB fork | reject marker semantically, reject spoon geometrically, select fork at D2 |
| `S1_joint_stir_initial_preference` | mixing bowl + normal spoon + normal fork | distractor | distractor | both tools valid; select rank-1 spoon at stage 0 and open nothing |
| `S1_joint_stir_exhaustion` | mixing bowl + YCB marker | oversized spoon | knife | inspect the complete fixed order and emit exhaustion |

The oversized spoon is a visibly anisotropically scaled YCB mesh. No runtime
code reads that scale: its observed cross-section naturally exceeds the
measured bowl opening. Likewise, the marker and fork relations come solely
from their fresh fused evidence clouds.

The pinned detector is Ultralytics `8.4.112` with
`yolov8m-worldv2.pt`. The medium 55 MB checkpoint remains CPU-capable and is
more reliable on thin synthetic utensils than the small checkpoint. Its CLIP
text encoder is pinned to Ultralytics CLIP commit
`c4b6ea0932a2c0f39a0fa528af5ec4982ff15cab`. Download and checksum both
artifacts once:

```bash
python mujoco_scenes/scripts/prepare_semantic_models.py \
  --output semantic_model_cache
```

For a local (non-Docker) run, create an ignored virtual environment and put
the same checksum-verified cache in the repository paths used by Ultralytics:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r mujoco_scenes/requirements.txt pytest
.venv/bin/python mujoco_scenes/scripts/prepare_semantic_models.py --output .

MUJOCO_GL=egl .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_joint_stir_counterexamples \
  --no-robot \
  --task-requirements mujoco_scenes/configs/stir_contents_joint.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root runs \
  --run-id joint_stir_counterexamples_local \
  --point-cloud-width 1280 \
  --point-cloud-height 960 \
  --semantic-detector yolo_world \
  --semantic-model yolov8m-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/semantic_vocabulary.yaml \
  --grounding-mode joint \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

The image sets `MUJOCO_SEMANTIC_PROCESS_ISOLATION=1`. YOLO-World therefore
runs in one persistent clean worker process while MuJoCo rendering remains in
the main process. This avoids native OpenGL/PyTorch conflicts observed as
container exit code 139 on some Mesa hosts, without reloading the detector for
each view or changing the captured evidence. The execution mode is recorded as
`semantic_detector.process_isolation` in `run_config.json`. It can be disabled
for diagnosis with
`-e MUJOCO_SEMANTIC_PROCESS_ISOLATION=0`.

Build the image from the repository root:

```bash
docker build -t mujoco-kitchen-s1 \
  -f mujoco_scenes/Dockerfile .
```

Run the primary actual-detector demonstration:

```bash
mkdir -p runs
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD/runs:/output" \
  -v "$PWD/semantic_model_cache/yolov8m-worldv2.pt:/models/yolov8m-worldv2.pt:ro" \
  -v "$PWD/semantic_model_cache/weights:/workspace/weights:ro" \
  mujoco-kitchen-s1 \
  --scene S1_joint_stir_counterexamples \
  --no-robot \
  --task-requirements configs/stir_contents_joint.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root /output \
  --run-id joint_stir_counterexamples \
  --point-cloud-width 1280 \
  --point-cloud-height 960 \
  --semantic-detector yolo_world \
  --semantic-model /models/yolov8m-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/semantic_vocabulary.yaml \
  --grounding-mode joint \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

`auto` is the CLI grounding default: it selects production `joint` evaluation
for joint-role task documents and preserves `geometry-only` evaluation for
legacy geometric task documents. Explicit `geometry-only` and `semantic-only`
are diagnostic ablations. Every joint stage evaluates all three modes from
the exact same saved graph, point-cloud evidence, and semantic evidence in
`grounding_mode_comparison.json`; no capture is repeated.

Ground-truth annotations live only in the offline evaluation configuration.
Runtime detector, registry, graph, and resolver modules never import it. To
produce the machine-readable comparison:

```bash
docker run --rm \
  -v "$PWD/runs:/output" \
  --entrypoint python \
  mujoco-kitchen-s1 \
  -m mujoco_scenes.evaluate_joint_grounding_run \
  /output/joint_stir_counterexamples \
  --evaluation-config \
  /workspace/mujoco_scenes/configs/joint_grounding_evaluation.yaml
```

Joint runs add:

```text
verified_task_handoff.json
candidate_evaluations.json
ablation_summary.json
offline_ablation_evaluation.json       # after the offline command
stages/<stage>/semantic_overview.png
stages/<stage>/semantics/detections.json
stages/<stage>/semantics/associations.json
stages/<stage>/semantics/<object_id>/semantic_evidence.json
stages/<stage>/semantics/cameras/<camera_id>/overlay.png
```

Semantic records preserve alternative labels, multi-view support, raw
confidence, association quality, detector/checkpoint/version, RGB/crop paths,
stage, and region. `run_config.json` records the Python, MuJoCo, NumPy,
Pillow, Torch, Ultralytics, and CLIP versions. Each camera summary in
`detections.json` records full-frame, crop, and total detector inference time,
while the file-level `inference_seconds` records the stage total. Multi-view
fusion selects the label supported by the most independent views, uses
weighted detector confidence only to break equal-view support, and returns
`UNKNOWN` for inadequate or ambiguous evidence. A weak re-observation is
recorded but cannot overwrite a stronger validated cached semantic result.

`verified_task_handoff.json` is emitted only for production joint completion.
It contains the role-to-object assignment, semantic rank and provenance,
unary and relational geometric evidence, stage-local evidence paths,
`verified: true`, and `ready_for_tamp: true`. This milestone does not execute
TAMP, robot motion, navigation, IK, manipulation, adaptive search, a
foundation model, training, or fine-tuning.

Each selected pairwise relation records its measured operands and signed
`pass_margin_m`. For `INSERTABLE_IN` this is the measured opening width minus
the tool cross-section and configured clearance; for `REACHES_BOTTOM` it is
usable tool length minus the configured grip allowance and measured cavity
depth. Positive margins pass (zero also passes for reach), negative margins
fail, and unavailable operands produce `UNKNOWN` with a null margin.

### One-command scene, ablations, and visual report

Build the current image once:

```bash
docker build -t mujoco-kitchen-s1 -f mujoco_scenes/Dockerfile .
```

Then run the complete primary experiment and report workflow:

```bash
./mujoco_scenes/scripts/run_joint_ablation_demo.sh
```

An optional first argument supplies a stable run ID:

```bash
./mujoco_scenes/scripts/run_joint_ablation_demo.sh my_joint_report
```

The script runs the actual MuJoCo scene once in production joint mode. It then
evaluates geometry-only, semantic-only, and joint acceptance from the exact
same saved observations; the ablations do not rerender the scene. Docker runs
with the host UID/GID so the reports remain writable from the host.

Open the generated interactive report:

```bash
xdg-open reports/my_joint_report/presentation_report.html
```

The report directory contains:

```text
presentation_report.html
ablation_report.html
ablation_report.md
ablation_comparison.png
offline_ablation_evaluation.json
report_data.json
ablations/
  geometry_only/geometry_only.gif
  semantic_only/semantic_only.gif
  joint/joint.gif
stages/
  000_initial/
  001_after_D1/
  002_after_D2/
```

Each ablation animation includes the rendered multi-view scene, RGB detections,
the cumulative point-cloud/graph overview, current stage, assignment, status,
and reason. The HTML report also includes measured object geometry, signed
pairwise margins, detector provenance, stage point clouds, graph snapshots,
all five individual camera overlays per stage, and the evidence-isolation
safeguards. The presentation HTML embeds every image, GIF, and MP4 as a data
URI, so it remains viewable as one self-contained local file even when browser
local-file access rules would otherwise block relative assets. Separate media
files remain beside it for presentation software and direct playback.

The synthetic RGB detector is not perfect: individual views can call a fork
or spoon a knife, so the fused result requires independent multi-view support
and a winning-label margin. The documented 1280×960 capture is required for
the counterexample demonstrations; 640×480 can lose thin-fork evidence and is
not the validated semantic setting. `READY_FOR_TAMP` is a handoff artifact
only and does not execute planning or manipulation.

The Actions panel creates `000_initial` automatically. Every automatic
post-opening capture and the manual `Geometry` button update that same
persistent run and re-evaluates the configured witness. Mount the interactive
output directory with:

```bash
-v "$PWD/runs:/workspace/runs"
```

## Ablation 2: function-aware cardinality and utensil reuse

Ablation 2 tests a separate claim from the semantic-versus-geometry
counterexample above: a raw count of detected utensils does not determine
whether a multi-target task is feasible. The number of required physical
objects depends on the usage policy of the function being performed.

The manually declared integrated task is:

> Prepare coffee and soup for two people. Stir both coffees using a reusable
> stirring utensil, and provide one dedicated utensil for each soup serving.

No foundation model generates this requirement. No global
`REUSABLE_OBJECT` property is inferred. Reuse belongs to the operation group:

| Function group | Accepted tools | Targets | Policy | Distinct tools required |
|---|---|---:|---|---:|
| `coffee_stirring` | spoon or fork | two cups | `sequential_reuse_allowed` | 1 |
| `soup_serving` | spoon or fork | two bowls | `dedicated_per_target` | 2 |

Cross-group reuse is allowed. The utensil used for coffee may later be
assigned to one soup bowl, so the integrated requirement derives a total of
two valid physical objects rather than three. The validated witness uses the
initial spoon for coffee and one soup bowl, then the D2 fork for the other
soup bowl. The task file contains no scene instance IDs, region names, or
stage-specific conditions.

Candidate eligibility is still production joint grounding:

```text
RGB semantic compatibility
AND ELONGATED_OBJECT from fresh point-cloud evidence
AND target-specific INSERTABLE_IN
AND target-specific REACHES_BOTTOM
AND operation-group assignment policy
```

Counts are recorded separately for raw observed utensils, semantically
eligible utensils, geometrically eligible utensils, functionally assignable
utensils, satisfied target slots, and distinct assigned physical objects. A
spoon or fork is counted only after it passes the selected function's
semantic, unary, and target-specific geometric checks. A detected utensil is
not counted if its point-cloud geometry fails. A single persistent instance
cannot appear twice in a dedicated-per-target matching.

### Scene family

| Scene | Purpose | Expected production result |
|---|---|---|
| `S1_ablation2_count_reuse_primary` | Integrated coffee and soup task | Complete after D2 with one spoon and one fork |
| `S1_ablation2_coffee_reuse` | Sequential-reuse isolation | Complete at INITIAL with one utensil assigned to two cups |
| `S1_ablation2_soup_dedicated` | Dedicated-per-target isolation | Complete after D2 with distinct spoon and fork IDs |
| `S1_ablation2_count_reuse_exhaustion` | No second valid utensil | Inspect full order and terminate `EXHAUSTED` without a verified handoff |

The primary scene initially shows two separate cups, two separate bowls, one
normal spoon, and no second valid utensil. D1 contains an oversized spoon that
is recognized semantically but rejected by measured insertability. D2
contains a normal fork. The production progression is:

```text
INITIAL: coffee 2/2 by reuse; soup 1/2; INCOMPLETE
D1:      oversized spoon excluded by geometry; INCOMPLETE
D2:      valid fork supplies the second dedicated assignment; COMPLETE
```

C2, B1, and C1 remain closed in the production early-stop run.

### Task schema

The operation-group extension is represented in
`configs/ablation2_count_reuse.yaml`:

```yaml
operation_groups:
  coffee_stirring:
    function: STIR_COFFEE
    tool_role: coffee_stirrer
    target_role: coffee_cup
    required_target_count: 2
    usage_policy:
      mode: sequential_reuse_allowed
      distinct_within_group: false
    relations: [INSERTABLE_IN, REACHES_BOTTOM]

  soup_serving:
    function: SERVE_SOUP
    tool_role: soup_utensil
    target_role: soup_bowl
    required_target_count: 2
    usage_policy:
      mode: dedicated_per_target
      distinct_within_group: true

cross_group_reuse:
  allowed: true
```

The solver first resolves target objects, evaluates semantic/unary/pairwise
compatibility for every tool-target pair, builds the compatibility graph, and
then applies reuse or one-to-one matching. It does not infer feasibility from
`valid_utensil_count >= target_count`, because compatibility is
target-specific and distinctness is policy-specific.

### Same-evidence policy modes

All three policy modes are evaluated from the same saved RGB, detections,
associations, point clouds, properties, and persistent registry:

| Mode | Integrated outcome | Interpretation |
|---|---|---|
| `always-reusable` | Incorrect COMPLETE at INITIAL using one utensil | False positive: ignores dedicated soup utensils |
| `always-distinct` | Incorrect EXHAUSTED while requiring four utensils | False negative: ignores valid sequential coffee reuse |
| `function-aware` | Correct COMPLETE at D2 using a spoon and fork | Applies the declared policy of each function group |

Diagnostic modes never control production early stopping and never rerender
the scene. Only assignment constraints change.

### Local actual-detector run

The validated detector setting remains YOLO-World
`yolov8m-worldv2.pt`, Ultralytics `8.4.112`, 1280×960 rendered RGB, and
multi-view process isolation:

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
.venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_ablation2_count_reuse_primary \
  --no-robot \
  --task-requirements mujoco_scenes/configs/ablation2_count_reuse.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root runs \
  --run-id ablation2_count_reuse_local \
  --point-cloud-width 1280 \
  --point-cloud-height 960 \
  --semantic-detector yolo_world \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --semantic-vocabulary \
    mujoco_scenes/configs/ablation2_semantic_vocabulary.yaml \
  --grounding-mode joint \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

### Docker production run

Prepare the persistent model cache once:

```bash
python mujoco_scenes/scripts/prepare_semantic_models.py \
  --output semantic_model_cache
```

Build and run with the host UID/GID so generated files remain writable:

```bash
docker build -t mujoco-kitchen-s1 -f mujoco_scenes/Dockerfile .

mkdir -p runs
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e YOLO_CONFIG_DIR=/tmp \
  -e MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 \
  -e MALLOC_ARENA_MAX=2 \
  -v "$PWD/runs:/output" \
  -v "$PWD/semantic_model_cache/yolov8m-worldv2.pt:/models/yolov8m-worldv2.pt:ro" \
  -v "$PWD/semantic_model_cache/weights:/workspace/weights:ro" \
  mujoco-kitchen-s1 \
  --scene S1_ablation2_count_reuse_primary \
  --no-robot \
  --task-requirements configs/ablation2_count_reuse.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root /output \
  --run-id ablation2_count_reuse_docker \
  --point-cloud-width 1280 \
  --point-cloud-height 960 \
  --semantic-detector yolo_world \
  --semantic-model /models/yolov8m-worldv2.pt \
  --semantic-vocabulary \
    mujoco_scenes/configs/ablation2_semantic_vocabulary.yaml \
  --grounding-mode joint \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

The cache is mounted read-only and reused between runs; neither checkpoint is
downloaded inside the demonstration container.

### One-command shared-evidence report

The presentation workflow deliberately runs the primary scene through the
full fixed order once. This supplies later evidence needed to demonstrate the
`always-distinct` exhaustion diagnostic. The production function-aware
first-completion stage remains D2.

```bash
./mujoco_scenes/scripts/run_ablation2_count_reuse_demo.sh \
  ablation2_count_reuse_demo
```

Open:

```bash
xdg-open \
  reports/ablation2_count_reuse_demo/presentation_report.html
```

The report package includes:

```text
presentation_report.html
ablation_report.html
ablation_report.md
README.md
report_data.json
offline_policy_ablation_evaluation.json
policy_ablation_comparison.png
ablations/
  always_reusable/always_reusable.gif
  always_reusable/always_reusable.mp4
  always_distinct/always_distinct.gif
  always_distinct/always_distinct.mp4
  function_aware/function_aware.gif
  function_aware/function_aware.mp4
stages/<stage>/
  semantic_overview.png
  pointcloud.png
  graph.png
  overview.png
```

The report contains the rendered scene, detector overlays, generic persistent
IDs, candidate counts, target assignments, numeric relation margins,
point-cloud views, graph growth, and per-policy timelines. Ground-truth
expectations in `ablation2_count_reuse_evaluation.yaml` are marked
offline-only and are never imported by runtime perception or assignment.

This milestone performs no FM/LLM/VLM requirement generation, no adaptive
search, no robot operation, and no TAMP action execution.

## Ablation 3: multi-target semantic–geometric assignment

Ablation 3 tests a stricter claim: an object that is individually plausible
for a function is not necessarily compatible with every target of that
function. The production predicate is therefore:

```text
VALID_FOR(tool, function, target)
```

and not a global `VALID_TOOL(tool)` flag. For every observed utensil–container
pair the runtime records semantic compatibility, scale-independent unary
geometry, `INSERTABLE_IN`, `REACHES_BOTTOM`, both numeric operands, signed
margins, source stages, and stage-local evidence paths. The assignment solver
then applies task-level reuse and distinctness to that measured compatibility
graph.

### Function-scoped reuse—not an object property

Reuse is declared on each task's functional requirement. It is neither a
property of the word `spoon` nor a permanent property of the abstract
function:

| Function group | Tool semantics | Targets | Usage constraint |
|---|---|---:|---|
| `coffee_stirring` | spoon | cup + mug | `sequential_reuse_allowed`; the same physical ID must pass both targets |
| `soup_serving` | spoon | two bowls | `dedicated_per_target`; different physical IDs are required |

`cross_group_reuse.allowed: true` means a coffee spoon may also occupy one
soup slot, but the two soup assignments must still be distinct. The solver
derives the required physical-object count from these constraints. It does
not attach a Boolean `reusable` property to a spoon or fork, and it does not
hard-code a final count.

Both functions use manually declared spoon semantics in this ablation. The
same schema could accept a broader vocabulary in another task; usage-policy
logic is independent of object category.

### Semantic-first production pairing and exhaustive ablation

Unary geometry is always computed for every legitimately observed object.
Binary geometry has two explicit strategies.

Production (`semantic_role_scoped`) performs:

```text
all observed objects
→ evaluate every unary role requirement for every object
→ evaluate semantic compatibility with every declared role
→ for each relation, form only the semantically compatible subject-role ×
  object-role pairs declared in the task
→ evaluate binary geometry for those pairs
→ bind objects to functions/roles
→ solve reuse, distinctness, and target coverage
```

For example, the configuration declares `INSERTABLE_IN` from
`coffee_stirrer` to `coffee_container`. Only objects with reliable semantic
support for those respective roles enter that directional relation check.
A future `NESTABLE_IN` declaration may use the same role at both ends, such as
`container` to `container`; the pairing engine does not hard-code a
tool/target distinction. Self-pairs are always forbidden. Semantic `UNKNOWN`
defers the binary check rather than treating the object as FALSE.

The timing ablation (`exhaustive_all_pairs`) retains the previous behavior:
semantic and unary checks still cover every object, and each required binary
relation is evaluated over all `N × (N - 1)` directed non-self pairs. Select
it with `--pairing-strategy exhaustive-all-pairs`. Production can be selected
explicitly with `--pairing-strategy semantic-role-scoped`; it is also the
default in this task configuration.

Every stage saves `pair_relation_evaluations.json` with strategy, possible
pair count, executed relation count, pruned count, and binary-geometry elapsed
time. Exhaustive runs additionally retain the compatibility artifact
`all_observed_pair_relations.json` for the existing ablation report.

The declaration is in `configs/ablation3_multi_target.yaml`. It contains no
object IDs, source regions, stage names, simulator names, hidden poses, or
asset dimensions. No FM generated it.

### Separate scene family

| Scene | Purpose | Expected production result |
|---|---|---|
| `S1_ablation3_multi_target_primary` | Four targets and target-specific counterexamples | INCOMPLETE at INITIAL and D1; COMPLETE after D2 |
| `S1_ablation3_multi_target_initial_complete` | Long narrow spoon visible initially | COMPLETE at INITIAL; no region opened |
| `S1_ablation3_multi_target_exhaustion` | Many useful spoons but none covers both coffee targets | Full order exhausted; no verified handoff |

The primary scene initially presents a narrow deep cup, a medium deep mug, a
shallow bowl, a deep bowl, a short spoon, a medium spoon, a long wide spoon,
and a long narrow fork. D1 adds another useful-but-partial spoon. D2 adds the
first long narrow spoon compatible with both coffee targets. C2, B1, and C1
remain closed in the early-stop production run.

Scene construction uses visibly scaled YCB meshes to obtain controlled,
noise-robust measurements. Runtime inference never reads those scale values,
body names, asset names, or hidden dimensions. It receives rendered RGB,
metric depth, visible instance masks, and camera calibration only.

### Complete compatibility matrix and assignment

For every tool–target cell the saved matrix contains:

```text
tool and target persistent IDs
fused RGB labels and tri-state semantic statuses
ELONGATED_OBJECT and OPEN_CAVITY statuses
maximum_cross_section_m and usable_length_m
opening_width_m and cavity_depth_m
clearance and grip margins
INSERTABLE_IN and REACHES_BOTTOM signed pass margins/statuses
final target-specific tri-state compatibility
source stages, regions, semantic records, and MeasurementEvidence paths
```

Coffee's `same_tool_must_cover_all_targets: true` rule accepts only an
assignment in which one persistent spoon ID has a `TRUE` edge to both the cup
and mug. Two partial spoons cannot be combined to fake reusable coverage.
Soup uses deterministic one-to-one matching: two spoon IDs compatible only
with the same bowl do not satisfy two bowl targets. Required `UNKNOWN`
evidence never becomes a compatibility edge.

### Four same-evidence modes

Perception runs exactly once. Every mode consumes the same saved RGB, YOLO
detections, mask associations, semantic fusion, point clouds, object
properties, relation measurements, and generic IDs. Only the acceptance and
assignment logic changes:

| Mode | Primary outcome | Why |
|---|---|---|
| `semantic-only` | Incorrect COMPLETE at INITIAL | Ignores unary and pairwise geometry |
| `geometry-only` | Incorrect COMPLETE at INITIAL | Evaluates every object and pair geometrically but cannot establish task-specific tool/container roles |
| `joint-target-agnostic-count` | Incorrect COMPLETE at INITIAL | Counts candidates valid somewhere without proving full target coverage |
| `joint-target-specific` | Correct COMPLETE after D2 | Requires every selected edge and declared usage constraint |

Diagnostic modes never control opening or stopping. Only
`joint-target-specific` may emit `verified_task_handoff.json`.

### Local actual-detector command

The validated configuration uses actual Ultralytics YOLO-World,
`yolov8m-worldv2.pt`, Ultralytics 8.4.112, CPU execution, an isolated detector
worker, and 1280×960 rendered RGB:

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
.venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_ablation3_multi_target_primary \
  --no-robot \
  --task-requirements mujoco_scenes/configs/ablation3_multi_target.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root runs \
  --run-id ablation3_multi_target_local \
  --point-cloud-width 1280 \
  --point-cloud-height 960 \
  --semantic-detector yolo_world \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --semantic-vocabulary \
    mujoco_scenes/configs/ablation3_semantic_vocabulary.yaml \
  --grounding-mode joint \
  --pairing-strategy semantic-role-scoped \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

### Docker build and execution

Prepare the persistent model cache once and build the pinned image:

```bash
python mujoco_scenes/scripts/prepare_semantic_models.py \
  --output semantic_model_cache
docker build -t mujoco-kitchen-s1 -f mujoco_scenes/Dockerfile .
```

Then run headless perception as the host UID/GID:

```bash
mkdir -p runs
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e YOLO_CONFIG_DIR=/tmp \
  -e MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 \
  -e MALLOC_ARENA_MAX=2 \
  -v "$PWD/runs:/output" \
  -v "$PWD/semantic_model_cache/yolov8m-worldv2.pt:/models/yolov8m-worldv2.pt:ro" \
  -v "$PWD/semantic_model_cache/weights:/workspace/weights:ro" \
  mujoco-kitchen-s1 \
  --scene S1_ablation3_multi_target_primary \
  --no-robot \
  --task-requirements configs/ablation3_multi_target.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root /output \
  --run-id ablation3_multi_target_docker \
  --point-cloud-width 1280 \
  --point-cloud-height 960 \
  --semantic-detector yolo_world \
  --semantic-model /models/yolov8m-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/ablation3_semantic_vocabulary.yaml \
  --grounding-mode joint \
  --pairing-strategy semantic-role-scoped \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

Both checkpoints are mounted read-only and are not downloaded per run.

For a controlled pairing-cost ablation, run the same command twice with fresh
run IDs and change only this option:

```bash
# Actual architecture
--pairing-strategy semantic-role-scoped --run-id pairing_semantic_first

# Exhaustive diagnostic baseline
--pairing-strategy exhaustive-all-pairs --run-id pairing_exhaustive
```

Compare `pair_relation_evaluations.json` in the two run roots. Its
`relation_evaluation_count`, `skipped_relation_pair_count`, and
`elapsed_seconds` fields isolate binary geometric relation work; the detector,
unary point-cloud measurements, task, and fixed inspection order are otherwise
unchanged.

Generate a compact timing/count report after the two runs:

```bash
python -m mujoco_scenes.generate_pairing_strategy_report \
  runs/pairing_exhaustive \
  runs/pairing_semantic_first \
  reports/pairing_strategy_ablation
xdg-open reports/pairing_strategy_ablation/pairing_strategy_ablation.html
```

### One-command presentation package

```bash
./mujoco_scenes/scripts/run_ablation3_multi_target_demo.sh \
  ablation3_multi_target_demo exhaustive-all-pairs
xdg-open \
  reports/ablation3_multi_target_demo/presentation_report.html
```

The published package contains:

```text
presentation_report.html       # self-contained images/GIFs/MP4s
ablation_report.html
README.md
report_data.json
offline_assignment_ablation_evaluation.json
assignment_ablation_comparison.png
compatibility_matrix.png
ablations/<mode>/<mode>.gif
ablations/<mode>/<mode>.mp4
ablations/<mode>/stage_<n>.png
stages/<stage>/semantic_overview.png
stages/<stage>/overview.png
stages/<stage>/pointcloud.png
stages/<stage>/graph.png
stages/<stage>/cameras/*_overlay.png
```

The run directory separately preserves `compatibility_matrix.json/.csv`,
`all_observed_pair_relations.json`,
`assignment_evaluations.json`, `assignment_ablation_summary.json`,
`verified_task_handoff.json`, `candidate_evaluations.json`, graph snapshots,
events, stage-local clouds, semantic records, and all RGB-D observations.

This milestone performs no FM/LLM/VLM task generation, no adaptive search,
no robot or navigation, no IK, no grasping, no task planning, and no TAMP
execution. `ready_for_tamp` is only a verified grounding handoff. The
reconstruction is complete only with respect to visible five-view evidence;
genuinely occluded surfaces remain unobserved, and detector reliability still
depends on visible scale, pose, lighting, vocabulary, and multi-view support.

## Integrated Scene 1 kitchen object–function benchmark

This benchmark is an integrated stress test of the capabilities isolated by
Ablations 1–3; it is not “Ablation 4.” It runs a deterministic no-robot
perception and grounding horizon and stops at a verified assignment for future
planning. It performs no navigation, IK, grasping, stirring, serving, action
sequencing, or TAMP execution.

The exact stored goal is:

> Prepare and serve coffee and soup for three people using the available
> kitchenware. Stir all three coffees and provide each soup bowl with a
> suitable utensil. Search the closed kitchen storage for anything still
> required.

The goal is not parsed by an FM. The manual specification is
`configs/s1_integrated_kitchen_object_function.yaml`: three cup/mug targets
need one persistent spoon that independently fits and reaches all three
(`sequential_reuse_allowed`), while three bowl targets need a one-to-one
matching with three distinct compatible spoon IDs (`dedicated_per_target`).
Cross-group reuse is enabled, so the reusable coffee spoon may also occupy one
soup slot. Forks, markers, tongs, and arbitrary objects are not admitted to the
spoon roles merely because their geometry fits.

### Scene family and validated progression

| Scene | Purpose | Result |
|---|---|---|
| `S1_integrated_kitchen_object_function_primary` | Full stress test | INITIAL, D1, D2 incomplete; soup complete at C2; B1 globally incomplete; global COMPLETE at C1 |
| `S1_integrated_kitchen_object_function_initial_complete` | Early-stop guard | COMPLETE at INITIAL; zero storage regions opened |
| `S1_integrated_kitchen_object_function_exhaustion` | No all-coffee spoon | All five regions inspected; EXHAUSTED; no verified handoff |

The validated primary run registered 20 persistent objects. Its final
assignment uses one C1 spoon for all three coffees and one soup bowl, plus two
other distinct spoons for the remaining bowls: three physical spoon IDs
overall, derived by the allocator rather than hard-coded.

Unary geometry is extracted for every accepted object from typed,
stage-local, region-gated `MeasurementEvidence`. Production evaluates binary
geometry only across reliable semantic role domains. The run also rebuilds
exhaustive pairing from identical cached evidence. At the validated final
stage, exhaustive pairing executed 760 relation checks and semantic-first
pairing executed 96, pruning 664 (87.4%); both produced the same required-edge
matrix, status, and selected assignments. These timings cover cached binary
relation evaluation only—not rendering, YOLO, reconstruction, or unary
extraction.

### Local actual-detector run

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MALLOC_ARENA_MAX=2 \
.venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_integrated_kitchen_object_function_primary \
  --no-robot \
  --task-requirements \
    mujoco_scenes/configs/s1_integrated_kitchen_object_function.yaml \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --stop-on-complete \
  --runs-root runs \
  --run-id s1_integrated_kitchen_local \
  --point-cloud-width 1280 \
  --point-cloud-height 960 \
  --semantic-detector yolo_world \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --semantic-config \
    mujoco_scenes/configs/s1_integrated_semantic_grounding.yaml \
  --semantic-vocabulary \
    mujoco_scenes/configs/s1_integrated_semantic_vocabulary.yaml \
  --grounding-mode joint \
  --pairing-strategy semantic-role-scoped \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

Generate the self-contained report from that one evidence stream:

```bash
.venv/bin/python -m mujoco_scenes.generate_target_assignment_report \
  runs/s1_integrated_kitchen_local \
  reports/s1_integrated_kitchen_local \
  --evaluation-config \
    mujoco_scenes/configs/s1_integrated_kitchen_object_function_evaluation.yaml
xdg-open reports/s1_integrated_kitchen_local/presentation_report.html
```

### Docker and one-command demonstration

```bash
python mujoco_scenes/scripts/prepare_semantic_models.py \
  --output semantic_model_cache
docker build -t mujoco-kitchen-s1 -f mujoco_scenes/Dockerfile .
./mujoco_scenes/scripts/run_s1_integrated_kitchen_demo.sh \
  s1_integrated_kitchen_demo
xdg-open reports/s1_integrated_kitchen_demo/presentation_report.html
```

The wrapper runs Docker as the host UID/GID, mounts the YOLO-World and CLIP
caches read-only, uses detector process isolation, captures every region
through C1 once, evaluates all diagnostics and both pairing strategies from
that saved evidence, validates expected outcomes, and generates matrices,
GIFs, MP4s, HTML, and README. It refuses to overwrite an existing run ID.

### Outputs and limits

The run root includes `goal_instruction.json`, `task_requirements.json`,
`object_registry.json`, `observed_graph.json`, `events.jsonl`,
`compatibility_matrix.json/.csv`, `function_assignments.json`,
`assignment_evaluations.json`, `usage_policy_evaluations.json`,
`pair_relation_evaluations.json`, `pairing_strategy_comparison.json`,
`diagnostic_summary.json`, and, only after verified production completion,
`verified_task_handoff.json`. Each stage retains five RGB/depth/segmentation
captures, semantic detections and associations, per-object fused evidence,
properties, pair evaluations, graph/point-cloud views, and an overview.

`reports/s1_integrated_kitchen_demo/presentation_report.html` embeds stage
progression, detector overlays, point clouds, graph evolution, four
same-evidence diagnostic animations, measurements, signed margins, assignment
matrices, the pairing-strategy comparison, GIFs, and MP4s.

Functional requirements and semantic vocabulary remain manually authored;
the broad goal is not converted into predicates by an FM; region order is
fixed; geometry covers only visible five-view evidence; detector reliability
depends on appearance and framing; and `ready_for_tamp` is a data handoff, not
executed manipulation.

## L2 living-room Region Ablation 1

Kitchen grounding selects functional objects. L2 selects a functional spatial
region for:

> Place the refreshment tray on a suitable living-room surface within easy
> reach of the sofa.

The three candidates intentionally separate the evidence:

- the rug passes `PLANAR_SUPPORT`, `FITS_ON`, and
  `NEAR_SEATING_AREA`, but fails serving-region semantics;
- the side table passes semantics and planar/context checks, but the measured
  tray footprint does not fit;
- the coffee table passes semantic compatibility and all measured geometry.

Perception is captured once, and geometry-only, semantic-only, and joint modes
reuse a SHA-256 manifest of the same RGB, depth, segmentation, masks,
detections, associations, and stage-local point clouds. Only joint mode is
authoritative.

```bash
docker build -f mujoco_scenes/Dockerfile -t mujoco-kitchen-s1 .
./mujoco_scenes/scripts/run_l2_region_ablation1_demo.sh \
  l2_living_room_region_ablation1_demo
xdg-open reports/l2_living_room_region_ablation1_demo/presentation_report.html
```

The new scene family, exact local/Docker commands, typed region evidence,
persistent generic region registry, full output layout, and limitations are
documented in
[LIVING_ROOM_ENVIRONMENT.md](LIVING_ROOM_ENVIRONMENT.md#l2-region-functional-grounding-region-ablation-1).

## Observed kitchen state to symbolic task plan

`S1_integrated_kitchen_object_function_primary` is the authoritative no-robot
three-serving experiment. Its pipeline keeps four interfaces separate:

1. `configs/s1_integrated_kitchen_object_function.yaml` declares the
   ground-truth functional task (the future FM replacement), counts, causal
   goals, and reuse policies, but contains neither physical object IDs nor a
   complete action order.
2. The fixed inspection controller observes `INITIAL`, then
   `D1 -> D2 -> C2 -> B1 -> C1`, constructing the persistent registry and
   measured functional witness from fresh RGB-D evidence.
3. `symbolic_planning.py` compiles only observed generic IDs, region-gated
   location evidence, and verified witness relations into a PDDL problem.
4. A deterministic classical state-space planner chooses the task action
   order, after which a replay validator checks every precondition, effect,
   holding/location invariant, reuse constraint, and final goal.

Coffee stirring permits one verified utensil to be used sequentially with all
three coffee vessels. Soup serving requires three target-specific, distinct
utensils. Source roles (kettle and coffee source) are grounded
in a separate YOLO-World pass over saved stage-0 mask-bounded RGB crops using
`configs/symbolic_source_vocabulary.yaml`. Simulator body names, hidden poses,
mesh dimensions, and legacy source-region metadata are not compiler inputs.
The three soup targets are visibly pre-filled; the task specification compiles
those observed targets with initial `has_content(..., soup)` facts, so the
primary scene has no redundant handled soup pot and the planner emits no
artificial soup-pouring action.
Planner-visible location comes only from `last_evidence_source_region`:
`INITIAL` evidence maps to the known countertop, while region-gated evidence
maps to the inspected region.

Phase 2 begins only after all of that evidence is frozen. It performs no RGB,
depth, point-cloud, detector, inspection, or simulator call. Run it on a
COMPLETE Phase-1 directory from the repository root:

```bash
.venv/bin/python -m mujoco_scenes.run_kitchen_symbolic_pipeline \
  --phase1-run-dir runs/feasibility_benchmarks/<benchmark-id>/<variant> \
  --output-root runs/phase2_symbolic \
  --run-id phase2_single_variant
```

The output includes the compact witness contract, compiled problem, initial
state, goal, generated plan, independent replay trace, validation, and matching
`domain.pddl`/`problem.pddl` exports. The Phase-1 evidence directory is not
mutated.

The planner exposes exactly `PICK`, `PLACE`, `POUR`, and `STIR`. Serving and
utensil provision are goal interpretations of generic `PLACE`; transferred
content is resolved from source facts by generic `POUR`. These symbolic state
transitions do not invoke robot
navigation, IK, grasping, collision motion planning, PDDLStream, or execution.
The task specification and source-role vocabulary remain manually configured;
object identities, geometry, semantics, compatibility, assignments, locations,
PDDL objects/facts, action order, and validation are produced from each run.

The tracked Phase-2 report and reproduction command are in
`mujoco_scenes/benchmark_reports/kitchen_symbolic_phase2/`.

## No-FM kitchen task-feasibility benchmark

The controlled feasibility family keeps the three-serving goal text byte-for-
byte identical while changing only the physical roster and deterministic
layout. It ends after observed functional grounding: it does not import the
symbolic planner, generate PDDL, execute actions, use a robot, or call an FM.

Coffee feasibility requires a valid semantic-and-geometric edge for every one
of the three coffee vessels. A one-tool cover is preferred, but two- and
three-tool collective covers are also feasible; complete covers are ranked by
minimum distinct physical tools and then by the existing deterministic
semantic/object ordering. Soup feasibility remains a global bipartite
matching: all three bowls need target-specific compatible utensils and those
three physical tools must be distinct. After the fixed
`INITIAL -> D1 -> D2 -> C2 -> B1 -> C1` evidence sequence, any non-complete
witness maps to terminal `INFEASIBLE`.

`configs/kitchen_feasibility_variants.yaml` defines F0--F7 feasible controls,
I0--I5 infeasible count/coverage/matching/geometry counterexamples, and P0/P1
deterministic layout perturbations. The independent oracle loads the same
instantiated MuJoCo model and derives dimensions from its actual mesh and
primitive geometry, using thresholds from `configs/geometry_inference.yaml`.
Its artifacts are marked
`PRIVILEGED_ORACLE_EVALUATION_ONLY`; no oracle result or exact property is
passed to the RGB-D prediction function.

Run all curated variants with real five-view RGB-D and YOLO-World:

```bash
BENCHMARK_ID="kitchen_feasibility_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MALLOC_ARENA_MAX=2 \
.venv/bin/python -m mujoco_scenes.run_kitchen_feasibility_benchmark \
  --all-core-variants \
  --no-robot \
  --output-root runs/feasibility_benchmarks \
  --benchmark-id "$BENCHMARK_ID" \
  --width 1280 --height 960 \
  --semantic-detector yolo_world \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/semantic_vocabulary.yaml \
  --semantic-confidence-threshold 0.03 \
  --semantic-min-supporting-views 2 \
  --save-semantic-overlays
```

For one debugging case, replace `--all-core-variants` with, for example,
`--variant F3_DISTRIBUTED_COFFEE_THREE`. The benchmark root contains
`benchmark_summary.json`, `benchmark_summary.csv`, and `README.md`; every
variant directory contains its config, stage-wise privileged oracle label,
observed prediction, comparison, final witness, grounded assignments, and the
normal RGB/depth/segmentation, semantic overlay, graph, and stage-local
point-cloud evidence. Exact scene geometry is intentionally available to the
offline oracle, so these variants evaluate controlled simulator evidence and
are not a claim of open-world ground-truth availability.

The frozen Phase-1 review package is tracked at
`mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1/`. It contains a
compact JSON/CSV manifest for all 16 variants, scientific guard assertions,
environment/version provenance, per-variant records, and representative
compressed initial/terminal overview and semantic images. Raw runs remain
outside Git under `runs/`.

## Living-room region-function Phase 1

The integrated living-room benchmark complements kitchen object-function
grounding with REGION-only functional grounding. Every F0--F6 and I0--I5
variant uses the exact same movie-night instruction and six initially observed
payloads: two mugs, two snack bowls, a TV remote, and a game controller. One
deterministic five-view RGB-D observation creates evidence records for both
seating targets and all five neutral candidate-region proposals. There is no
inspection sequence,
hidden-region search, robot, object-function predicate, symbolic planning, or
action execution.

The integrated family uses a sparse furniture-scale room made from documented
Poly Haven chair and table visuals with separate analytic collision and RGB-D
measurement proxies. Six payloads are arranged on a dedicated staging console
rather than cluttering candidate surfaces. Exact source, licence, author,
scale, transform, and hash provenance is stored in
`assets/living_room_realistic/manifest.json`; visual mesh dimensions are not
production evidence.

The future-FM contract is
`configs/l2_integrated_region_function_task.yaml`. It asks for two distinct
`PERSONAL_REFRESHMENT_REGION` assignments, one per observed seat and complete
drink/snack set, plus one `SHARED_CONTROLS_REGION` holding the remote and
controller together. Cross-function region sharing is disabled. Compatibility
requires RGB semantic support, measured `PLANAR_SUPPORT`, measured two-object
`FITS_SET_ON`, and either target-specific `NEAR_SEAT` or
`ACCESSIBLE_FROM_BOTH_SEATS`. Required UNKNOWN evidence never forms an edge.
An exhaustive deterministic solver evaluates the complete three-slot
allocation, preventing greedy target coverage and cross-function conflicts.

Run one scene:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MALLOC_ARENA_MAX=2 \
.venv/bin/python -m mujoco_scenes.run_living_room_region_function \
  --scene L2_integrated_living_room_region_function_F0_BASE \
  --no-robot --runs-root runs --run-id living_region_f0 \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --width 1280 --height 960
```

Run the authoritative fixed-goal family and create the compact report:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
MALLOC_ARENA_MAX=2 \
.venv/bin/python -m mujoco_scenes.run_living_room_region_benchmark \
  --runs-root runs/living_room_region_phase1 \
  --run-id living_room_region_phase1_reproduction \
  --report-dir mujoco_scenes/benchmark_reports/living_room_region_feasibility_phase1 \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --width 1280 --height 960
```

Raw variant directories contain camera RGB/depth/segmentation, association
overlays, stage-local region/payload/seat clouds, registries, the full
compatibility matrix, diagnostics, and final region witness. The tracked
compact report retains representative images and result records. The
independent oracle is explicitly marked
`PRIVILEGED_ORACLE_EVALUATION_ONLY` and is never imported by production
grounding. Per-variant `evaluation_order.json` proves that the production
result was persisted before the privileged oracle ran. Guard and allocation
metrics are derived from saved runtime artifacts rather than declared
constants. “INFEASIBLE” means no complete allocation exists in this controlled
fully observed candidate set; it is not an open-world impossibility claim.

Candidate support regions are supplied by neutral simulator-derived spatial
proposal volumes. These proposals provide only region localization/evidence
gating. Functional validity, semantic role, support dimensions, geometry, and
target-relative suitability are inferred from rendered RGB/RGB-D evidence.
Open-world support-region proposal/discovery is outside the scope of this
benchmark. Resolved rig, observation, registry, and run metadata mark the
proposal source as `SIMULATOR_DERIVED_NEUTRAL_SPATIAL_GATE` and explicitly
state that it encodes no function, semantic class, or expected validity;
functional dimensions come from `OBSERVED_RGBD_POINT_CLOUD` evidence.

The F0 Google Robot model is compiled only for spawn validation. The saved
`robot_spawn_validation.json` checks its canonical pose behind the staging
console, workspace-facing orientation, static-furniture clearance, and initial
contacts. No navigation, IK, grasp, manipulation, planning, or execution is
performed.

Current limitations are manually specified functional requirements,
controlled simulator layouts, neutral privileged spatial proposals rather
than free-space region discovery, no robot execution, no task planning, no FM
requirement generation, and no photorealism claim.

## Living-room mobile execution Phase 3A

The frozen Phase-2 living-room PICK/PLACE plan can now be refined into
condition-driven mobile Google Robot execution. The refiner uses observed
generic IDs and point-cloud support geometry, tests the current base with the
execution IK/collision path, generates target-facing stances, plans base paths
with deterministic RRT*, and physically verifies each dynamic placement. See
[`benchmark_reports/living_room_mobile_execution_phase3/README.md`](benchmark_reports/living_room_mobile_execution_phase3/README.md)
for the exact commands and research boundaries. The preceding limitations
describe the frozen Phase-1 report itself; Phase 3 does not alter it.
