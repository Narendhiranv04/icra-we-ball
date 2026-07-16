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

The interactive viewer starts with MuJoCo's free camera by default. Pass an
explicit `--camera NAME` to start from one of the fixed or robot-mounted
cameras instead.

With Fetch or Google Robot enabled, `wrist_camera` is attached to its gripper
link and `head_camera_rgb` is robot-mounted.
The kitchen-only `--no-robot` mode uses a fixed placeholder with the same name.

Fetch assets are supplied by `gymnasium-robotics`. Google Robot assets are
loaded from an external MuJoCo Menagerie checkout. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licenses.

## Run natively with uv

Create the Python environment from the repository root:

```bash
uv venv --python 3.11
uv pip install -r mujoco_scenes/requirements.txt
```

Fetch remains the default robot:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot fetch --viewer
```

For Google Robot, sparse-clone its official Menagerie directory beside `V1`:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git \
  ../third_party/mujoco_menagerie
git -C ../third_party/mujoco_menagerie sparse-checkout set google_robot
```

Then launch the same scene with the alternative backend:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot google --viewer
```

Set `MUJOCO_MENAGERIE_PATH` when the checkout lives elsewhere. The Google
backend supports scene loading, cameras, joint targets, collision-checked base
navigation, and an S1-calibrated vertical sugar-jar pick/place at
`serving_spot`. Other scene poses plus coffee-jar, kettle, and spoon actions
remain gated. See
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

With Fetch or Google Robot enabled, `--viewer` also opens the companion
`Actions` panel.
Choose `Actions` → `Move` → `Home`, `Cupboard 1`, `Cupboard 2`, or `Box` to run
one collision-checked RRT* base motion. `Cupboard 2` and `Box` are symbolic
aliases of the same right-side pose. For Fetch, `Actions` → `Pick` provides
staged vertical grasps for the kettle handle, both jar upper bodies, and the
spoon handle. Fetch jar picks add a compliant 90-degree in-hand pitch—with the
rigid weld released and only soft upright/centring assistance—followed by a
horizontal carry pose.
Google uses the same navigation controls and exposes its validated sugar-jar
pick and serving-area place controls. Unsupported objects are deliberately not
shown as actionable. Google parks on a farther navigation line, automatically
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
