# Calibrating a New Robot Backend

This guide defines when a robot is ready for kitchen navigation, picking, and
placing. A model loading successfully is only the first step: every action must
also have named interfaces, reachable poses, stable control, physical contact,
and repeatable checks.

## Current calibration status

| Robot/action | Automated status | Human visual status |
|---|---|---|
| Fetch navigation | Supported | Recheck after scene geometry changes |
| Fetch coffee jar, sugar jar, and spoon pick/carry | Current S1 headless picks pass | Recheck after object/robot changes |
| Fetch kettle pick/carry | Exposed, but current S1 bilateral-contact check fails | Calibration repair required |
| Fetch place | Not implemented | Not applicable |
| Google navigation | Empty/held collision monitors pass | User validation required |
| Google S1 sugar-jar pick/carry | Bilateral-contact check passes | User validation required |
| Google S1 sugar-jar place at `serving_spot` | Position check passes | User validation required |
| Google S1 spoon pick/carry | Far-tip contact, vertical hang, final bilateral re-grasp, and both held routes pass | User validation required |
| Google spoon place | Gated as uncalibrated | Required after implementation |
| Google coffee jar and kettle | Guarded calibration candidates; current attempts fail | Required after calibration |

Only actions listed as supported are exposed in the Actions panel. Do not add
an object to a profile merely because IK returns a solution.

### Current action and interaction surface

| Interaction | Fetch | Google Robot | How it is exposed |
|---|---|---|---|
| Move to home, cupboard 1, cupboard 2, or box | Supported | Supported | Actions panel |
| Pick/carry sugar jar | Supported; rotates jar horizontal | Supported; vertical carry | Actions panel |
| Place sugar jar at serving area | Not implemented | Supported | Actions panel |
| Pick/carry spoon | Supported | Supported | Actions panel |
| Place spoon | Not implemented | Not implemented | Gated |
| Pick/carry coffee jar | Supported; rotates jar horizontal | Candidate; provisional IK currently fails | Calibration mode only |
| Pick/carry kettle by handle | Exposed, but current S1 check fails | Candidate; provisional return does not settle | Calibration mode only |
| Place coffee jar or kettle | Not implemented | Not implemented | Gated |
| Open/close cupboards and drawers | Scene actuator helper only | Scene actuator helper only | Python API / CLI, not a robot manipulation |
| Pour, stir, or complete the coffee task | Not implemented | Not implemented | Not exposed |

An object being visible, included in the task description, or having a grasp
constraint in the composed MJCF does not mean its interaction has been
calibrated. This table describes executable behavior, not scene contents.

### Interactive candidate calibration

The ordinary Actions window executes validated actions. It shows Google
coffee-jar and kettle candidates as disabled so their absence is explicit, but
it will not run them. The serving-area Place button remains disabled until the
robot is holding a place-calibrated object, currently only the sugar jar.
Launch the opt-in calibration window with `uv`:

```bash
uv run python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot google --viewer --calibration-mode
```

This labels each pick as `[calibrated]` or `[candidate]`, enables present
candidate objects, displays planning/contact/collision failures in the status
area and terminal, and provides **Reset simulation attempt**. Candidate motion
still passes through the normal IK path checks, self/environment collision
guards, bilateral-contact requirement, and a bounded attempt timeout.

For repeatable measurements without the viewer, run one candidate headlessly:

```bash
uv run python -m mujoco_scenes.calibration_check \
  --robot google --calibration-mode --pick coffee_jar
uv run python -m mujoco_scenes.calibration_check \
  --robot google --calibration-mode --pick kettle
```

Tune provisional definitions in `GOOGLE_CALIBRATION_PICK_SPECS` in
`generic_manipulation.py`, restart the scene, and repeat. The current starting
coffee-jar definition stops at an inaccurate IK approach, while the starting
kettle definition reaches its return trajectory but does not settle. These are
diagnostics, not supported grasps. After a candidate passes the full checklist
in section 5, move its definition into `GOOGLE_PICK_SPECS`, add it to the
robot/scene supported-object declarations, and add a regression check. The UI
does not automatically promote a candidate or weaken a failed guard.

## Where calibration lives

- `robot_profiles.py` declares joint, actuator, body, site, contact-geometry,
  jaw-direction, IK-seed, and supported-object conventions.
- `scene_loader.py` adapts source MJCF: planar base joints, actuator gains,
  stable fingertip names, cameras, and inactive grasp constraints.
- `generic_manipulation.py` supplies profile-driven vertical pick/place.
- `mobile_motion.py` supplies profile-driven base planning and the Actions UI.
- `calibration_check.py` is the repeatable headless acceptance check.

Keep upstream robot assets unmodified. Put kitchen-specific adaptations in the
scene composer so updating MuJoCo Menagerie does not overwrite calibration.

## 1. Confirm provenance and model composition

Before motion work:

1. Record the upstream repository, revision, and license in
   `THIRD_PARTY_NOTICES.md`.
2. Namespace imported names to avoid collisions with the kitchen model.
3. Load every scene with the new backend and check for warnings, invalid
   `qpos`, contacts at reset, and missing textures/meshes.
4. Add deterministic home joint positions and actuator commands.

## 2. Declare the mobile interface

Add a `MobileRobotProfile` containing the ordered forward, lateral, and yaw
joints and actuators plus the robot body-name prefix. The current kitchen uses
an ideal holonomic base abstraction; the conversion in `mobile_motion.py`
assumes both robots start at the same world pose and orientation.

Run a destination independently:

```bash
.venv/bin/python -m mujoco_scenes.calibration_check \
  --robot google --move cupboard1
```

Repeat for `cupboard2`, `box`, and the return to `home` when changing the base
collision proxy, kitchen geometry, home pose, or carry pose.

Treat navigation home and the manipulation stance as separate calibrated
states. Google Robot parks at `Y=-1.25 m`, approaches 15 cm for S1 pick/place,
and returns to the navigation line before Move is enabled. A robot must expose:

- an empty-gripper compact navigation joint configuration;
- a held-object carry configuration whose live geometry is collision-free;
- a base approach/retract transition that preserves object and table clearance.

The collision checker must classify robot geoms through their owning bodies as
well as geom names. Menagerie models commonly contain anonymous collision geoms.
Validate every sampled yaw during base rotation, including attached objects.

## 3. Measure the manipulation interface

Create a `ManipulationProfile` with:

- arm joint and actuator names in matching order;
- both finger joints and actuators;
- gripper body and public end-effector site;
- separately named right/left contact geometries;
- the end-effector's approach and finger-closing axes;
- open/closed commands and closing direction;
- a reachable carry position and deterministic IK seed.

Never infer open/closed semantics from a joint range. Sweep the gripper in
simulation and measure fingertip separation. For Google Robot in this model,
`0.01 rad` is about 145 mm open and `1.30 rad` is about 9 mm closed.

## 4. Tune tracking before contact

First validate end-effector tracking with an open gripper. Record final
position and orientation error at carry, pre-grasp, grasp, pre-place, and place.
Low demonstration gains can produce centimetres of error under gravity even
when joint-space IK is correct.

Tune only the composed-model actuator gains and force limits. Recheck for:

- oscillation or numerical instability;
- excessive impact at a waypoint;
- actuator saturation;
- self-collision and table collision;
- steady-state Cartesian bias.

Simulation gains are not approved real-hardware gains.

## 4a. Validate the whole robot, not only the end effector

Pose IK is not collision avoidance. Every accepted trajectory must check:

- every non-adjacent robot visual-geometry pair for self-clipping;
- robot visual geometry against physical environment collision geometry;
- dense interpolation between IK solutions, not only waypoint endpoints;
- the live simulated configuration while tracking the planned commands;
- the held object during base translation and every sampled yaw.

Menagerie disables some detailed base collision meshes in this kitchen adapter
and MuJoCo normally filters parent/child contacts. Use signed
`mj_geomDistance` queries on visual meshes to catch clipping that the contact
solver cannot report. Establish explicit allowances only for real mechanical
mounting interfaces. Google Robot's shoulder rotates partly inside its base
housing; that one interface has a bounded allowance, while forearm/base,
elbow/base, other self pairs, and environment penetration are rejected.

The collision checker must reject a path as soon as any interpolated segment is
unsafe and report the two bodies involved. A runtime guard must stop motion if
controller tracking leaves the collision-checked path.

## 4b. Smooth position-controlled motion

Do not send a distant joint target to a stiff position actuator in one
simulation update. That produces a large command discontinuity, high joint
velocity, and visible wobble even when the geometric path itself is safe. The
Google controller applies a lightweight slew-rate limit before every arm and
base command:

- arm targets change by at most `1.20 rad/s`;
- base translation targets change by at most `0.25 m/s`;
- base yaw targets change by at most `0.60 rad/s`;
- finger closure advances by `0.003 rad` per simulation update.

Intermediate arm waypoints may advance after tracking within `0.05 rad`; the
final grasp, carry, or navigation pose must settle within `0.01 rad`. Each
accepted waypoint is held for four updates. These limits smooth the existing
collision-checked path without adding a heavier time-parameterization library.
Mobile navigation is not reported complete until the final actuator command
has arrived, translation is below `0.003 m/s`, yaw is below `0.01 rad/s`, and
those conditions remain stable for 50 simulation updates. Apply the same
settling check before transitioning between base approach and arm motion.
Google's composed planar translation joints use damping `750`; this reduces
servo overshoot without over-damping a full navigation route, and makes the
manipulation approach settle faster than merely waiting for the original
underdamped base to stop oscillating. Short pick/place base approaches
temporarily use damping `2000`, then restore `750` before normal navigation.

When tuning a new robot, log peak joint velocity and the largest per-update
actuator-command change before and after altering these values. Lower the rate
limits if the robot still oscillates. Raise them only after the full collision,
contact, carry, and place checks pass, because a faster target ramp increases
tracking lag and impact energy. The calibration check's `--max-steps` may need
to be increased for deliberately slow profiles.

## 5. Calibrate one object at a time

Start with a regular cylinder or box. Add an explicit grasp site and support
height, then tune the gripper-site Z offset so the pad contact band—not the
palm or fingertip body origin—aligns with that site.

Calibration is a `(scene, object pose, robot)` result. A grasp validated for an
S1 countertop slot is not automatically enabled for a different scene/slot.

An object becomes supported only after all of these pass:

1. Continuous IK from carry to pre-grasp and grasp.
2. Open descent without moving the object.
3. Contact from named geometries on both fingers for multiple simulation ticks.
4. Lift without table contact, slipping, or explosive constraint activation.
5. Stable carry without cabinet, table, or robot self-collision.
6. Base retreat to navigation home while maintaining the grasp.
7. RRT* start validity and full translation/rotation without collision.
8. Repeatability after a fresh scene reset.

Run the current Google acceptance check:

```bash
.venv/bin/python -m mujoco_scenes.calibration_check \
  --robot google --pick sugar_jar \
  --move-while-holding cupboard1 --place \
  --move-after-place cupboard1
```

The checker fails on IK error, timeout, missing bilateral contact, missing
grasp constraints, or placement more than 5 cm from the serving-site centre.

Run the main-branch spoon behavior on Google Robot in both navigation
directions:

```bash
.venv/bin/python -m mujoco_scenes.calibration_check \
  --robot google --pick spoon --move-while-holding cupboard1
.venv/bin/python -m mujoco_scenes.calibration_check \
  --robot google --pick spoon --move-while-holding cupboard2
```

The spoon check additionally requires contact on the named handle proxy, a
passive connect-constraint phase, a bowl-down hang within five degrees, a
successful handoff back to the transport weld, a final physical re-grasp with
both fingers, and collision-free held-object rotation. The pivot temporarily
relaxes the fingers so gravity can orient the spoon; after the weld handoff,
the controller closes them gradually and does not report success until named
handle contact persists on both sides. Google uses a 180-degree-equivalent
wrist branch to preserve the jaw axis without hitting its wrist limit, plus a
carry point four centimetres closer to the base to clear B1. Spoon placement is
deliberately not exposed.

Irregular objects need separate rules. A kettle handle requires alignment to
the live handle axis. The spoon implementation demonstrates thin-handle
contact and controlled hanging around a live pivot; its offsets and wrist
branch are not suitable for the kettle. Do not copy a cylindrical-object
offset to either one.

## 6. Calibrate placing and release

For every destination:

1. Define a named support site and object support height.
2. Preserve the live gripper-to-object transform captured at grasp time.
3. Solve carry to pre-place and vertical descent.
4. Disable the transport constraint without snapping the object.
5. Open the fingers, allow contact to settle, and retreat vertically.
6. Fold to the empty compact pose and retreat the base to navigation home.
7. Check final horizontal error, height, linear/angular motion, and containment
   within the intended support surface.

## What requires a person

Codex can inspect MJCF, compute kinematics, run headless physics, render still
frames, measure contacts/errors, and implement repeatable tests. You must still:

- watch the full native viewer motion from several camera angles and confirm
  that clearances and behavior look physically reasonable;
- confirm the Actions window, mouse input, timing, and camera controls work on
  your desktop/display server;
- decide whether visible wobble, sliding, mesh interpenetration, or motion style
  is acceptable for the experiment;
- provide video/screenshots and observations when a problem appears only in
  your interactive session;
- measure and validate any real robot's frames, payload, limits, controllers,
  emergency stops, collision behavior, and workspace safety with qualified
  on-site supervision.

Never transfer simulation controller gains or trajectories directly to real
hardware. Treat real-robot commissioning as a separate safety-reviewed task.

## Human acceptance checklist

Launch:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot google --viewer
```

Then verify:

- `Move -> Cupboard 1`, return `Home`, and repeat for the right side.
- `Pick -> Sugar jar` descends centrally and contacts both sides.
- `Pick -> Spoon` pinches the far handle tip, releases into a passive pivot,
  settles bowl-down, and remains vertical during left and right navigation.
- The jar clears the countertop and remains stable at carry.
- `Place -> Serving area` releases onto the surface without a snap or tip-over.
- The gripper retreats without touching the released jar.
- The arm folds before Move becomes available after placement.
- While holding, Move reaches each side and rotates without an RRT start error
  or contact with the cupboard/table.
- Free, head, and wrist cameras remain usable throughout.

Record the date, Menagerie commit, MuJoCo version, scene, action, and any visual
issue when reporting calibration results.
