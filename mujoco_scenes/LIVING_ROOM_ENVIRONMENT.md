# Google Robot Living-Room Environment

The `L1_living_room` scene is a separate, rigid-only MuJoCo environment for
Google Robot navigation and manipulation experiments. It does not replace or
modify the S1 kitchen. The room contains a fixed L-shaped couch along the west
and south sides, a fixed coffee table, rigid tabletop objects, a
wall-mounted flat-screen TV, an attached book ledge, a media console with two
rigid sliding drawers, and a rigid hand duster on its own rest. Full-height
north and west walls form the visible room corner; the south and east sides
remain open so free and orbiting cameras retain a clear view.

There are no cloth, flex, deformable, particle, liquid, or other soft-body
objects in this scene. The couch cushions, duster head, mug, coasters, and all
other props are ordinary rigid MuJoCo geometry.

## Install and run natively with uv

Run these commands from the repository root (`V1`):

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r mujoco_scenes/requirements.txt
```

Google Robot is loaded from MuJoCo Menagerie. If it is not already available,
sparse-clone it beside `V1`:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git \
  ../third_party/mujoco_menagerie
git -C ../third_party/mujoco_menagerie sparse-checkout set google_robot
```

If Menagerie is elsewhere, set `MUJOCO_MENAGERIE_PATH` to either the checkout
root or its `google_robot` directory.

Launch the room, MuJoCo viewer, and companion Actions window through the
shared scene selector:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --environment living-room --robot google --viewer
```

Calibration mode also exposes guarded candidate object actions:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene L1_living_room --robot google --viewer --calibration-mode
```

Launch only the MuJoCo viewer, without the Actions window:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene L1_living_room --robot google --viewer --no-actions-panel
```

Render one frame without opening a GUI:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python \
  -m mujoco_scenes.scene_loader \
  --scene L1_living_room --no-robot \
  --camera room_corner_camera --render captures/living_room.png
```

The static room can be inspected without a robot using either `--robot none`
or `--no-robot`. The default is `--robot google`; Fetch is not composed into
either production environment.

Run the same five-view point-cloud reconstruction used by the kitchen:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python \
  -m mujoco_scenes.scene_loader \
  --scene L1_living_room --no-robot \
  --point-cloud runs/living_room_cloud \
  --point-cloud-width 640 --point-cloud-height 480
```

Run persistent stage-local inspection for the initially visible coffee-table
objects and then the two media-console drawers:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python \
  -m mujoco_scenes.scene_loader \
  --scene L1_living_room --no-robot \
  --task-requirements configs/living_room_find_planar_support.yaml \
  --inspect-sequence LEFT_DRAWER RIGHT_DRAWER \
  --runs-root runs --run-id living_room_inspection
```

This uses `configs/living_room_inspection_rigs.yaml`. The room and kitchen
have independent calibrated camera poses and region volumes while sharing the
same typed `MeasurementEvidence`, registry, graph, property, semantic, role,
pairing, witness, and visualization implementation.

### Lost-remote scenario

The compact lost-remote variant places the remote inside the sofa's real
under-seat clearance:

```bash
MUJOCO_GL=glfw uv run python -m mujoco_scenes.living_room_scene \
  --scenario lost_remote \
  --sofa-perception oracle \
  --robot-debug-view \
  --viewer
```

Select **Move → Couch**, then **Inspect beneath sofa with foot cameras**. The
remote is absent from observed state before this action. Oracle mode is an
explicit ground-truth testing backend; it still saves the same two low-camera
images and bounded RGB-D evidence used by the learned path. Detection requires
support from both views.

`--robot-debug-view` opens the two ground-camera and five top-camera RGB streams
in one grid. `--sofa-debug-view` remains a compatible alias. The same window can
also be opened at any time with **Open live robot-camera view** in the Actions
panel. Enable **Show latest inspection masks** to display the exact masks from
the most recent ground-camera inspection while the five top feeds remain live.
This does not repeatedly call SAM 3 while the live RGB feeds are running.

For SAM 3.1, start the separate perception server, set `SAM3_BASE_URL`, and
replace the perception option:

```bash
MUJOCO_GL=glfw uv run python -m mujoco_scenes.living_room_scene \
  --scenario lost_remote \
  --sofa-perception sam3 \
  --viewer
```

The SAM service receives the two ground-camera RGB frames and the prompt
`remote control`.
Depth, camera poses, sofa bounds, and MuJoCo IDs stay local. Results are saved
under `runs/living_room_sofa/<mode>/`, including each RGB frame, mask overlay,
and `inspection.json`. Physical extraction is a separate tool-use action; the
ordinary top-down pick remains disabled while the remote is beneath the sofa.

## Layout and rigid contents

```text
                              NORTH
        +------------------------------------------------+
        | book ledge  wall TV + two-drawer console      |
        |                                  duster rest   |
        |                                                |
        |  west couch        fixed coffee table          |
        |  +---------+       +----------------+          |
        |  |         |       | rigid objects  |   east   |
        |  |         |       +----------------+ corridor |
        |  |                                            |
        |  +---------- south couch --------+      home   |
        +------------------------------------------------+
                              SOUTH
```

The couch is fixed and uses a dedicated square corner section joined to west
and south extensions. Its bases, seats, and backrests meet at exact seams
without interpenetrating modules, producing one continuous L-shaped couch. The
coffee table is also fixed: it has no free, slide, hinge, or actuator joints.
The TV has no stand or console contact. Warm/cool fill lights, light walls, a
large bordered area rug centered on the coffee table, and differentiated
drawer faces provide visual depth without adding navigation collisions. The
rug is non-colliding visual geometry.

The tabletop initially carries:

- a TV remote;
- a ceramic mug;
- a hardback book;
- two rigid coasters; and
- a rigid game controller.

These objects use a staggered multi-row layout rather than a single line. The
four pickable tabletop objects start at least 20 cm apart by centre distance,
with the coasters separated along the south row of the table.

The remote, mug, book, coasters, and controller are independent free bodies.
They move through ordinary contact and friction and are not welded to the
tabletop. The duster is also a single free rigid body, with a capsule handle
and a box-shaped head.

Available starting cameras are:

- `free` (default);
- `overhead_camera`;
- `room_corner_camera`;
- `tv_camera`;
- `table_camera`;
- `wrist_camera`;
- `left_foot_camera`;
- `right_foot_camera`;
- `top_front_camera`;
- `top_front_left_camera`;
- `top_rear_left_camera`;
- `top_rear_right_camera`; and
- `top_front_right_camera`.

The wrist, two foot, and five top cameras require `--robot google`. The Google
model has a mobile base rather than articulated feet, so the foot cameras sit
on its lower front corners and look into the real clearance beneath the raised
rigid sofa. The five-camera upper rig is mounted above the fixed head shell,
independent of the arm. Its overlapping 90-degree views span 360 degrees so the
robot can observe room regions while navigating to them. The wrist camera is
retained for close manipulation views but is not part of the region-observation
rig.

## Actions window

Only one action runs at a time. Buttons that would conflict with an active
motion, an extended arm, or a held object remain disabled. Status text and the
progress bar report planning, contact confirmation, settling, completion, or
the guard that stopped a failed attempt.

The **Functional task** group also exposes **Store game controller**. It sends
only visible storage candidates to the configured foundation-model server,
then executes and verifies the selected alternative. The right drawer, left
drawer, and wall shelf are supported targets. Use `TAMP_FM_BACKEND=fixed` for
an offline integration smoke test; see
[TAMP_PIPELINE.md](TAMP_PIPELINE.md) for the full data flow.

The **Grounded action file** runs exact symbolic actions for testing without
foundation-model selection. Edit `configs/living_room_actions.txt`, placing
one action on each line, then press **Reload and run**. Examples include
`move drawer_left`, `open left`, `pick game_controller`, and
`place media_console_left_drawer`. The runner waits for each action to finish.
`state observed` prints the executive's bounded observation; `gt` prints the
simulator's symbolic ground truth. These commands still execute through the
guarded physical controllers. Set `LIVING_ROOM_ACTION_FILE` to select another
text file.

### Move

The named destinations are:

- **Home / coffee table**: the south-side starting and tabletop-manipulation
  position;
- **Table south**: a close view of the table's south side;
- **Table north**: the opposite side of the table, facing south;
- **Table east** and **Table west**: side views and routes around the table;
- **Book shelf**: the wall-ledged book placement/retrieval stance;
- **Media-console drawer**: the original shared console stance;
- **Media console - left drawer** and **right drawer**: aligned storage
  stances used by autonomous placement;
- **Couch**: a navigation pose near the L-shaped couch;
- **TV**: the dusting stance in front of the screen; and
- **Duster rack**: the pickup/return stance beside the duster rest.

The duster rest is on the east side of the room at world X = 1.45 m. The
corresponding collision-checked navigation stance is at X = 1.28 m (Y =
0.64 m), close enough for the arm to reach while leaving clearance for the
robot and held tool.

Each request builds a collision-checked RRT* route around the fixed furniture.
The planner checks the compact arm, robot base, and any welded carried object.
Live robot/furniture contact stops execution.

The direction names use world coordinates: **north** is positive Y, toward the
TV/media console, and **south** is negative Y, toward the starting side. Table
north/south/east/west refer to the corresponding sides of the fixed table.

### Pick and place rigid objects

All currently advertised living-room object actions are calibrated in normal
mode. Calibration mode remains available for future guarded candidates:

| Object | Pick status | Place target | Notes |
|---|---|---|---|
| TV remote (`remote_control`) | Calibrated | Remote marker on coffee table | Bilateral contact on the rigid remote body |
| Ceramic mug (`living_room_mug`) | Calibrated | Right coaster | Vertical body pinch; explicit stiff contact keeps the released mug physically supported by the coaster |
| Hardback book (`hardback_book`) | Calibrated | Wall ledge or table | Reversible edge-grasp storage workflow |
| Game controller (`game_controller`) | Calibrated | Right media-console drawer or table | Reversible multi-geom grasp and storage workflow |
| Rigid TV duster (`rigid_duster`) | Calibrated | Duster return rest | Pickup, compact carry, navigation, and 15-cell TV sweep verified end to end |
| Coasters | Physical clutter only | None | No coaster pick action is advertised yet |

The production calibrated-object set is the TV remote, ceramic mug, hardback
book, game controller, and rigid TV duster.

Remote, mug, book, and controller manipulation initially starts from **Home /
coffee table**; stored book/controller picks start at their storage location.
Duster pickup and return start from **Duster rack**. A pick plans dense
collision-checked IK segments, closes gradually, requires persistent contact
from both finger sides on the named object collision geom, and only then
enables the object's transport weld. The weld captures the live relative pose,
so an object does not snap to a prerecorded grasp. The arm returns to compact
carry before navigation is allowed.

Place uses the object's dedicated marker. The remote marker is on the fixed
table, while the mug target is a child of the free right coaster and follows
its live pose. A book picked at Home is placed on the wall ledge; one
picked from the shelf returns to the table. The controller similarly
alternates between the table and the open right media-console drawer. Release is
followed by a settling period and a compact-arm retreat.

### Use the TV remote

1. At **Home**, pick the TV remote.
2. Move to **TV** in compact carry.
3. Select **Aim remote and toggle TV**.

The controller verifies the remote transport weld, solves a collision-checked
aim pose seeded from the compact arm branch, holds the receiver-facing pose,
toggles the TV screen material/emission, and follows a checked retract path.
The action can be repeated to switch the TV off. Reset always restores the TV
to off. The power command is semantic—the single gripper cannot physically
press a button on an object it is already holding—but the remote itself is a
physically picked and carried rigid body.

Powering on changes the otherwise near-black screen to an emissive blue panel
and changes the small lower-frame status LED from dim red to green. It does
not play a video, load media, or simulate television content. If dust remains,
the translucent dust layer is still rendered over either powered state.

### Store a book on the shelf

1. Pick the hardback book at **Home**.
2. Move to **Book shelf**.
3. Select **Place book on shelf**.

To retrieve it, pick it at **Book shelf**, move Home, and select **Return book
to table**. The Actions panel tracks its symbolic storage location while all
pick, carry, navigation, collision, release, and support behavior remains
physical.

### Use the media-console drawers

Move to **Media-console drawer** before selecting a left/right open or close
command. Each drawer is an independent rigid body on a bounded 27 cm slide
joint. A slow position actuator drives each one; completion requires both
position and velocity settling.

The manual controller workflow continues to use the right drawer. The
functional task can instead choose either drawer or the wall shelf. It opens a
selected drawer before placement and closes it after the arm returns to
compact carry. Closed-drawer occupancy is treated as unknown until inspection,
and a newly discovered occupied target causes the executive to try the next
ranked alternative.

### Dust the TV

The dust action uses the rigid duster and the Google Robot arm; it does not
animate a detached tool or directly erase the whole screen.

The screen's central 0.78 m-wide band is divided internally into a 3-by-5 grid
of 15 invisible coverage cells. Cell-center X coordinates run from -0.32 m to
+0.32 m. A coverage cell is a measurement region, not a physical target or
obstacle.
The arm supplies the calibrated row height and tool orientation while the
holonomic base moves laterally through three serpentine horizontal passes. The
nominal head-to-screen gap is 15 mm. The visible dust uses fifteen
zero-thickness, edge-aligned regions that form one seamless translucent layer
at reset. No grid is visible before cleaning. When a cell is newly verified,
only its matching region eases smoothly to transparent while the other dust
remains. A cell contributes only after the live rigid head satisfies every
coverage check for eight consecutive simulation ticks:

- head-position error no greater than 3 cm;
- head-orientation error no greater than 6 degrees;
- measured head-to-screen gap between -1.5 mm and 26 mm; and
- measured rigid-duster head speed no greater than 0.10 m/s.

Advancing the state machine or arriving at a base waypoint alone cannot clean
a cell. The Actions window reports measured progress out of 15 cells.

The intended sequence is:

1. Move to **Duster rack** and pick the rigid duster.
2. Move to **TV** in compact carry.
3. Select **Dust TV**.
4. The arm follows a held-tool collision-checked joint-space clearance path
   that orients the rigid head toward the screen. It holds the calibrated row
   pose while the base makes guarded lateral coverage passes, then follows a
   checked detour back to carry.
5. Move back to **Duster rack** and place the tool on its rest if desired.

Only the duster head may enter the screen's small allowed proximity/contact
band; penetration deeper than 1.5 mm is rejected. The controller also rejects
forward base drift greater than 6 mm or yaw drift greater than 0.6 degrees
during a sweep. Handle, gripper, arm, TV-frame, console, and
self-collisions remain failures. Coverage is retained until **Reset living
room** is used. Reset restores the robot, fixed furniture, drawer, TV power,
clutter, equality constraints, cleaned-cell set, and the original dust-film
opacity in every region.

## Calibration workflow

Use `--calibration-mode` only for developing a candidate. It does not disable
IK, joint-limit, bilateral-contact, self-collision, furniture-collision,
timeout, or settling guards.

1. Start a fresh calibration scene:

   ```bash
   MUJOCO_GL=glfw uv run python -m mujoco_scenes.living_room_scene \
     --viewer --calibration-mode
   ```

2. Move to the stance named in the Actions window before selecting a
   candidate.

3. Run one pick or dust attempt. Record the final status and the stage at
   which a guard stopped it.

4. Use **Reset living room** between attempts. This matters because free-body
   poses, drawer position, TV state, active equalities, and dust-patch alpha
   are simulation or model state.

5. Tune the object's grasp site, collision proxy, per-object grasp offset,
   wrist orientation/IK seed, and carry pose in the living-room MJCF and
   controller. Do not weaken global collision tolerances to force one object
   to pass.

6. Repeat the action headlessly where a repeatable check exists, then perform
   the human visual checks below.

7. Promote a candidate to the calibrated-object list only after both automated
   and visual acceptance pass from a clean reset.

The shared robot-profile and general calibration rationale are documented in
[ROBOT_CALIBRATION.md](ROBOT_CALIBRATION.md). Living-room object definitions
are in `assets/living_room_base.xml`; scene-specific pick specifications are
in `living_room_manipulation.py`; and the screen coverage guards and sweep
controller are in `living_room_dusting.py`.

## Acceptance checklist

For every navigation destination:

- RRT* finds a path around the fixed table and couch.
- The robot does not touch the couch, table, loose objects, TV, console, or
  duster rest.
- Final base position, yaw, and velocity settle before the action completes.
- A carried object is included in planning and remains clear during turns.

For every promoted object:

- Dense IK and live guards report no robot self-clipping or furniture contact.
- Both fingers maintain contact with the required object geom before and after
  transport-weld activation.
- Pick, compact carry, navigation, and placement all complete from reset.
- Placement settles near the correct table-relative or duster-rest marker.
- The viewer shows a credible grasp with no visible gap or mesh penetration.

For storage and TV power:

- The book can be placed on and picked from the shelf.
- The controller can be placed in and picked from the open right drawer.
- Both drawer open/close actions reach their joint targets and settle.
- Drawer-dependent object buttons remain disabled while the right drawer is
  closed.
- A held remote at TV reaches a checked aim pose, toggles visible power state,
  and retracts to compact carry.
- Reset closes both drawers and switches the TV off.

For TV dusting:

- Duster pickup has bilateral physical handle contact.
- The base is settled at **TV** before the arm unfolds.
- All 15 cells are credited by actual duster-head proximity, not elapsed time
  or direct UI state changes.
- The duster head remains aligned with the screen and within its allowed gap;
  deep penetration is rejected.
- No handle, gripper, arm, frame, console, or self-collision occurs.
- The arm retracts to compact carry, and reset restores the seamless dust
  layer.

Run the fast headless structural and guard checks with:

```bash
uv run python -m unittest mujoco_scenes.tests.test_living_room -v
```

## Current limitations

- Only Google Robot and the no-robot static view are supported.
- The base is an ideal position-controlled holonomic X/Y/yaw adaptation of the
  Menagerie model, not a wheel/ground drivetrain simulation.
- Navigation uses known MuJoCo geometry rather than online perception or
  SLAM.
- The coffee table is fixed and has no movement action.
- The couch is entirely rigid; it does not deform when contacted.
- Dust is visual coverage state, not particles or a material-removal model.
- Both drawers use lightweight position actuators; robot contact-based handle
  pulling is not implemented.
- The TV power command is semantic after physical remote pickup/aiming because
  the one-handed gripper cannot press a button on the object it is holding.
- Coasters have no pick controller.
- There is no sitting, door, pouring, cloth-wiping, or soft-object interaction
  in this scene.

## L2 region-functional grounding: Region Ablation 1

L1 remains the Google Robot interaction demonstration described above. The
separate L2 family asks a different perception question: **which observed
spatial region can perform a requested placement function?**

The exact benchmark goal is:

> Place the refreshment tray on a suitable living-room surface within easy
> reach of the sofa.

The manually authored future-FM contract is
`configs/l2_region_ablation1_task.yaml`. It proposes the fixed inspection order
`RUG_PATCH → SMALL_SIDE_TABLE → COFFEE_TABLE` and requires:

- parent-furniture semantics compatible with
  `refreshment_serving_region`;
- `PLANAR_SUPPORT` from a fresh region-local point cloud;
- `FITS_ON(refreshment_tray, region)` at 0° or 90° with measured edge margins;
- `NEAR_SEATING_AREA(region, observed_sofa_context)`.

Ranking is a proposal, not proof. A high-ranked candidate is still rejected
when any required verifier returns `FALSE` or `UNKNOWN`.

### Scene variants

- `L2_living_room_region_ablation1_primary`: the rug is the geometry-only
  false positive, the small side table is the semantic-only false positive,
  and the coffee table is the joint solution.
- `L2_living_room_region_ablation1_initial_complete`: the valid coffee table
  is inspected first and joint grounding completes without later inspection.
- `L2_living_room_region_ablation1_exhaustion`: the coffee table is also too
  small, so joint grounding exhausts the fixed order without a handoff.

All variants compile with `--robot google`, `--robot none`, and `--no-robot`.
Scientific runs use no robot.

### Evidence and persistent regions

Each stage positions five deterministic virtual cameras, renders aligned RGB,
metric depth, and instance segmentation, validates every view, back-projects
depth into the MuJoCo world frame, gates the current candidate volume, and
extracts its upper support plane. The property extractor accepts only typed
`RegionMeasurementEvidence`. Inspection volumes select evidence; their
configured extents never become measured support dimensions.

The fixed payload is associated as the scene's single independently movable
rigid segmentation instance. Runtime capture does not look up a tray body,
geom, material, or asset name; its footprint is measured only from the
selected current-view RGB-D points.

Observed patches receive generic IDs such as `region_0001`. Identity is
maintained using measured 3-D location, footprint overlap, size consistency,
and independently detected semantic context. Public inference uses separate
semantic, geometric, functional, and provenance namespaces. Configured labels
such as `RUG_PATCH` remain provenance only.

YOLO-World receives rendered RGB and the vocabulary in
`configs/l2_region_semantic_vocabulary.yaml`. Projected observed masks
associate detections with the support, fixed payload, and seating context.
Simulator body, geom, material, asset, and region names are not detector
inputs or semantic decisions.

### Same-evidence modes

Perception runs once. Offline evaluation then hashes and reuses the identical
RGB, depth, segmentation, masks, detections, associations, region clouds,
payload cloud, and seating evidence:

| Mode | Acceptance logic | Primary outcome |
|---|---|---|
| Geometry-only | planar AND fits AND near | incorrectly selects the rug |
| Semantic-only | serving semantics AND seating semantics | incorrectly selects the small side table |
| Joint | semantics AND planar AND fits AND near | selects the coffee table |

Only joint mode is production-authoritative. Diagnostic modes never stop
inspection and never emit a verified handoff.

### Local command

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 \
MALLOC_ARENA_MAX=2 \
python -m mujoco_scenes.run_l2_region_ablation \
  --scene L2_living_room_region_ablation1_primary \
  --no-robot \
  --runs-root runs \
  --run-id l2_living_room_region_ablation1_demo \
  --width 1280 \
  --height 960 \
  --semantic-detector yolo_world \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/l2_region_semantic_vocabulary.yaml \
  --semantic-confidence-threshold 0.03

python -m mujoco_scenes.generate_region_ablation_report \
  runs/l2_living_room_region_ablation1_demo \
  --report-dir reports/l2_living_room_region_ablation1_demo
```

Open the presentation:

```bash
xdg-open reports/l2_living_room_region_ablation1_demo/presentation_report.html
```

### Docker and one-command demonstration

Build the existing shared image:

```bash
docker build -f mujoco_scenes/Dockerfile -t mujoco-kitchen-s1 .
```

The wrapper uses host UID/GID, EGL, detector process isolation, writable
run/report mounts, and the persistent model cache:

```bash
./mujoco_scenes/scripts/run_l2_region_ablation1_demo.sh \
  l2_living_room_region_ablation1_demo
```

The equivalent perception container invocation is:

```bash
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
  --entrypoint python \
  -v "$PWD/runs:/output" \
  -v "$PWD/semantic_model_cache/yolov8m-worldv2.pt:/models/yolov8m-worldv2.pt:ro" \
  -v "$PWD/semantic_model_cache/weights:/workspace/weights:ro" \
  mujoco-kitchen-s1 \
  -m mujoco_scenes.run_l2_region_ablation \
  --scene L2_living_room_region_ablation1_primary \
  --no-robot \
  --runs-root /output \
  --run-id l2_living_room_region_ablation1_demo \
  --width 1280 \
  --height 960 \
  --semantic-detector yolo_world \
  --semantic-model /models/yolov8m-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/l2_region_semantic_vocabulary.yaml \
  --semantic-confidence-threshold 0.03
```

### Outputs and limitations

The run contains `region_registry.json`, `payload_registry.json`,
`region_function_evaluations.json`, JSON/CSV compatibility matrices,
`observed_graph.json`, event history, hashed offline evaluation, and (only for
joint success) `verified_region_handoff.json`. Every stage contains five
camera captures, masks, semantic associations, fresh region and payload
clouds, properties, and evaluations.

The report package contains HTML, README, numeric data, matrix and graph
images, per-stage semantic/point-cloud/mask views, GIF, and MP4.

Function requirements and ranking remain manual. Region occupancy and
arbitrary free-space decomposition are outside this first controlled
benchmark. No FM parsing, mobile navigation, manipulation, placement action,
planning, or TAMP execution occurs. The output is a verified
destination-region handoff only.
