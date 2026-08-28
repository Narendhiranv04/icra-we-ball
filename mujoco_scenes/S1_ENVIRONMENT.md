# S1 Environment: Regions, Objects, and Google Robot Usage

## World frame

- `+X`: to the right when looking at the workstation from the front.
- `+Y`: from the robot toward the workstation and upper cabinets.
- `+Z`: upward.
- The floor is `Z = 0`.

## Physical regions

| Region | Position / extent | Initial S1 contents | Interaction |
|---|---|---|---|
| Navigation floor | In front of the workstation | Google Robot starts at approximately `(0, -1.25, 0)`, behind the serving table | Home manipulation pose plus symmetric left/right routes |
| Countertop | `X=[-0.70,0.70]`, `Y=[-0.40,0.40]`, top at `Z=0.58` | kettle, coffee jar, sugar jar, spoon | Always visible work surface |
| C1 | Upper-left cabinet, centered near `(-0.35,0.45,0.96)`; usable opening about `0.40 m` wide by `0.41 m` high and `0.31 m` deep | mug, glass | Enlarged hinged-door cabinet; primary S1 search region |
| C2 | Upper-right cabinet, centered near `(0.35,0.45,0.96)`; same enlarged opening as C1 | plate, bowl, scanned canister distractor | Enlarged hinged-door cabinet |
| D1 | Lower-left drawer, mounted beneath the tabletop near `(-0.44,-0.30,0.46)`; usable tray about `0.332 m` wide | fork, knife, stirrer | Centered level U-handle; front grasp from Home |
| D2 | Lower-right drawer, mounted beneath the tabletop near `(0.44,-0.30,0.46)`; usable tray about `0.332 m` wide | tongs, napkin, scanned spatula distractor | Mirrored centered U-handle and identical front-grasp motion |
| B1 | Lidded box at the far-right edge of the countertop, centered near `(0.52,0.18,0.58)`; inner footprint about `0.344 x 0.174 m` | tea box | Set rearward for carried-object clearance; requires a rightward base reposition |
| Serving area | Green table centered near `(0,-0.56,0.56)` with top at `Z=0.58`; its `0.50 m` width fits between the drawer fronts | empty | Goal destination and central navigation obstacle |

Opening a container reveals and catalogs all objects in that region. An
already-inspected region remains known even if it is later closed.

## S1 visible objects

| Object | Initial location | Role |
|---|---|---|
| kettle | `counter_spot_1`, near `(-0.35,-0.32)` | Hot-water appliance; picked by its handle |
| coffee jar | `counter_spot_2`, near `(-0.15,-0.30)` | Coffee ingredient |
| sugar jar | `counter_spot_3`, near `(0.05,-0.30)` | Optional coffee ingredient |
| spoon | `counter_spot_5`, near `(0.25,-0.34)` | Stirring utensil; picked near the far handle tip |

The mug is the only unresolved required object at reset. C1 contains the exact
required mug and a glass that can act as a configured substitute.

The cabinet shelf splits each opening into an approximately `0.17 m` lower
bay and `0.24 m` upper bay. Both exceed the robot gripper's outer span, while
the `0.40 m` cabinet width and the open
top of B1 leave room for an approach, grasp, and withdrawal.

B1 is intentionally close to C2. Either mechanism opens normally on its own,
but the B1 lid and C2 door have negative geometric clearance when both are at
their configured open targets (`B1=1.8 rad`, `C2=1.4 rad`). Physics therefore
blocks the second mechanism. A later task planner must close or move one
before fully opening the other; this is deliberate ordering logic, not an XML
placement error.

The thick front edge of the B1 lid has a collidable U-shaped metal handle
projecting toward the robot. Its crossbar runs horizontally along `X`, and the
named `B1_lid_handle_grasp` site supports a later frontal grasp-and-lift motion.

The drawer slides use damped, synchronized position control to keep the arm
and physical tray together during the pull. The flat stirrer also has a matching
flat, high-friction collision proxy instead of its former rolling capsule.
The contents remain ordinary free bodies—not welded fixtures—so a gripper can
still pick them up. In the S1 opening test, the stirrer moves about `0.1 mm`
relative to D1 with no measurable rotation; the fork and knife each move about
`1.2 mm`.

## Mesh provenance in S1

| Semantic object | Visual source |
|---|---|
| mug, plate, bowl, fork, spoon, knife, tea box | YCB (`025`, `029`, `024`, `030`, `031`, `032`, `008`) |
| kettle | GSO `Threshold_Porcelain_Teapot_White` |
| coffee jar | GSO Nescafe Taster's Choice jar |
| sugar jar | GSO Wilton sugar-sprinkles jar |
| canister and spatula distractors | Google Scanned Objects |
| stirrer, folded napkin, tongs | Local simple meshes |

The object library also exposes YCB `002_master_chef_can` as `coffee_can`, YCB
`004_sugar_box` as `sugar_box`, and YCB `065-a_cups` as `cup` for later scenes.
Textured scans use invisible primitive collision proxies so their contact
behavior remains stable enough for manipulation planning. Receptacles are a
special case: the mug, cup, glass, plate, small plate, bowl, and coffee jar
use hollow bottom-and-wall proxies, allowing a spoon to occupy the visible
interior instead of being ejected by a solid proxy.

## Google Robot actions

Google Robot is the only production robot backend. Its model comes from the
pinned MuJoCo Menagerie checkout. The kitchen adapter adds planar base joints,
cameras, collision-checked RRT* navigation, and calibrated manipulation.

The active planar joints are `google:base_forward_joint`,
`google:base_lateral_joint`, and `google:base_yaw_joint`. The calibrated home
line is `Y=-1.25 m`. `mobile_motion.py` maps the named UI destinations to
collision-checked base routes and uses `generic_manipulation.py` for supported
pick/place operations. Container buttons call the scene's articulated
open/close API. `calibration_check.py` exercises the supported Google Robot
navigation, grasp, carry, and placement paths.

Running the viewer with the robot opens the companion `Actions` panel. Use
`--no-actions-panel` when only the standard MuJoCo viewer is desired.

## Intended S1 episode

1. Reset Google Robot in front of the workstation with all containers closed.
2. Observe the countertop and determine that kettle, coffee, sugar, and spoon
   are available but the required mug is missing.
3. Rank C1 as the first search region using the mug-location prior.
4. Navigate or reposition the mobile base to a C1 manipulation pose.
5. Open C1 and inspect its complete contents: `{mug, glass}`.
6. Select the mug, grasp it, and place it on the countertop.
7. Hand the resolved state to the later coffee-making TAMP pipeline.
8. Place the completed coffee in the separate serving area.

The current `open_container()` API moves a container actuator directly and is
useful for search-state experiments. Once manipulation planning is connected,
the Google Robot gripper should physically operate the same door, drawer, and lid
joints; the direct API then serves as an oracle/debug action.

## Observed-resource witness boundary

The observed-state pipeline can now check counted task roles and measured
pairwise compatibility after every closed/open observation stage. It uses only
the persistent visible-object graph: unopened contents never become candidates.
`COMPLETE` is an assignment certificate, not task execution, retrieval, motion
refinement, PDDLStream invocation, or ROBUSTTAMP invocation.

Witness inference does not use the scene's semantic object names. Each stable
instance name is hashed for association, objects are exported as
`object_0001`-style IDs, every cloud receives the same geometry schema, and
roles are satisfied only through configured measured predicates and numeric
thresholds. A missing rim or interior measurement remains `INDETERMINATE`.

`configs/s1_find_planar_support.yaml` is the deterministic early-stop fixture.
With order `D1 D2 C2 B1 C1`, the measured planar support in C2 completes the
task and leaves B1/C1 closed. `configs/s1_find_open_receptacle.yaml` is the
conservative cavity fixture. At the documented 320×240 region-facing setup,
the validated C2 evidence contains a mostly enclosed rim and interior points
below it, so the task completes at C2; inadequate coverage in another scene
still remains `INDETERMINATE`.

Sequential observed-state inspection is intentionally no-robot. Each direct
container actuator action is followed by the configured settle interval and a
fresh virtual five-camera rig pose from `configs/inspection_rigs.yaml`. Only
points inside that stage's region volume can update the registry. Properties
are extracted from `stages/<stage>/evidence/<object_id>/fused.ply`, never from
the cumulative visualization or combined scene clouds.

## Joint-grounding scene variants

The original S1 distributions above are unchanged. Three additional
evaluation scenes reuse the workstation and deterministic no-robot inspection
path:

| Scene | Counter placement | Closed-region placement |
|---|---|---|
| `S1_joint_stir_counterexamples` | `counter_spot_2`: mixing bowl; `counter_spot_4`: YCB marker | D1: oversized YCB spoon; D2: normal YCB fork; C2/B1/C1: ordinary distractors |
| `S1_joint_stir_initial_preference` | `counter_spot_1`: mixing bowl; `counter_spot_3`: normal YCB spoon; `counter_spot_5`: normal YCB fork | ordinary distractors |
| `S1_joint_stir_exhaustion` | `counter_spot_2`: mixing bowl; `counter_spot_4`: YCB marker | D1: oversized spoon; D2: knife; no valid later alternative |

The initial-preference fork is rotated 90 degrees in the physical scene so
its tines remain visible beside the spoon. The D1 spoon uses the same
recognizable YCB spoon mesh with a deliberately obvious anisotropic scale;
while D1 is closed it is held to the drawer by a scene-construction weld and
released after opening/settling so it cannot be ejected by contact. Neither
the transform, scale, weld, body name, nor region assignment enters semantic
or geometric inference.

`configs/stir_contents_joint.yaml` requires a bowl-like RGB detection plus
measured open-cavity geometry for `mixing_container`, and a ranked
spoon/fork/spatula RGB detection plus measured elongation, insertion, and
reach relations for `mixing_tool`. Detection uses rendered RGB only; point
cloud measurement remains category-free. See `README.md` for the exact
pretrained-model cache and Docker commands.

## Run

```bash
MUJOCO_GL=glfw /home/naren/miniconda3/bin/python \
  -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug \
  --viewer --camera front_camera
```

Use `--no-robot` for the virtual sequential-inspection milestone. The
interactive viewer uses Google Robot by default.

To start directly in Google Robot's head-camera view, replace `front_camera` with
`head_camera_rgb`.
