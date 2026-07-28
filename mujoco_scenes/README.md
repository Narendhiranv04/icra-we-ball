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
the robot's current pose. A reconstruction is also run automatically after a
physical box or drawer opening. The cupboard buttons use the scene's direct
actuator-based search/debug opening and likewise reconstruct immediately.
Interactive persistent observed-state outputs are written under `runs/`.

## Persistent registry and growing observed graph

Sequential inspection begins with a five-view observation while every region
is closed, then moves and opens exactly one requested region before each later
observation:

```bash
mkdir -p runs
docker run --rm \
  -e MUJOCO_GL=osmesa \
  -v "$PWD/runs:/output" \
  mujoco-kitchen-s1 \
  --scene S1_coffee_missing_mug \
  --inspect-sequence D1 D2 C2 B1 C1 \
  --runs-root /output \
  --run-id s1_observed_demo
```

The robot-enabled command reuses the existing deterministic mobile-base move
executor and the scene controller's opening action. Add `--no-robot` for a
faster perception-only smoke run that skips base movement while retaining the
same closed/open observation stages.

Each run contains an atomically updated `object_registry.json`, an
`observed_graph.json`, append-only `events.jsonl`, cumulative per-object PLY
clouds and properties, and one immutable directory per observation:

```text
runs/s1_observed_demo/
├── run_config.json
├── events.jsonl
├── object_registry.json
├── observed_graph.json
├── graph_growth.gif
├── objects/<object_id>/{cumulative.ply,properties.json}
└── stages/
    ├── 000_initial/
    │   ├── combined_cloud.ply
    │   ├── properties.json
    │   ├── graph.json
    │   ├── pointcloud.png
    │   ├── graph.png
    │   └── overview.png
    └── 001_after_D1/ ...
```

Geometric values come only from fused observed points. Dimensions use a
2nd–98th percentile PCA-oriented bounding box and retain explicit measurement
status and method provenance. Unavailable geometry remains `UNKNOWN`.
Category-to-family and category-to-function mappings, plus relation margins,
are configured in `configs/observed_state_semantics.yaml`.

The Actions panel creates `000_initial` automatically. Every automatic
post-opening capture and the manual `Geometry` button update that same
persistent run. Mount the interactive output directory with:

```bash
-v "$PWD/runs:/workspace/runs"
```
