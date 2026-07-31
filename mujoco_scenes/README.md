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

Fetch also contributes `head_camera_rgb`. It is available from the same
`--camera` CLI option whenever the robot is enabled.

With the Fetch robot enabled, `wrist_camera` is attached to its gripper link.
The kitchen-only `--no-robot` mode uses a fixed placeholder with the same name.

The Fetch MJCF and mesh assets are supplied at runtime by the MIT-licensed
`gymnasium-robotics` package. Its Fetch model is based on Fetch Robotics assets
and was adapted by OpenAI/Farama. See [S1_ENVIRONMENT.md](S1_ENVIRONMENT.md) for
the region map, robot controls, and intended S1 episode.

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

With Fetch enabled, `--viewer` also opens the companion `Actions` panel.
Choose `Actions` → `Move` → `Home`, `Cupboard 1`, `Cupboard 2`, or `Box` to run
one collision-checked RRT* base motion. `Cupboard 2` and `Box` are symbolic
aliases of the same right-side pose. `Actions` → `Pick` provides staged
vertical grasps for the kettle handle, both jar upper bodies, and the spoon
handle. Jar picks add a compliant 90-degree in-hand pitch—with the rigid weld
released and only soft upright/centring assistance—followed by a horizontal
carry pose. Objects placed in a left/right table strip can be selected and
picked again from the matching base pose. Side-picked jars first move into a
base-relative clear corridor, then perform the same compliant 90-degree slip,
so every jar pick ends horizontal. `Actions` → `Place` samples
an object-aware buffered point on the
serving table or on the counter strip facing the robot, moves through hover
and release poses, lets physics settle the object, and returns the empty hand
to its carry pose.
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
`PLANAR_SUPPORT`. Their thresholds and relation margins are in
`configs/geometry_inference.yaml`; the file contains no category mappings.
Unavailable evidence remains `UNKNOWN`.
Every record also carries `source_stage`, `source_region`,
`measurement_cloud_path`, contributing camera IDs, point count, method,
extractor version, and the `MEASUREMENT_EVIDENCE` purpose marker.

Task roles declare explicit geometric predicate and numeric-property
requirements. The graph records `SATISFIES_GEOMETRY` edges with the complete
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
| `mixing_container` | bowl, rank 1 | `OPEN_CAVITY`, opening ≥ 0.05 m, cavity ≥ 0.015 m | relation target |
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
