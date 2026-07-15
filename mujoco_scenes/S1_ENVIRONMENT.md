# S1 Environment: Regions, Objects, and Fetch Usage

## World frame

- `+X`: to the right when looking at the workstation from the front.
- `+Y`: from the robot toward the workstation and upper cabinets.
- `+Z`: upward.
- The floor is `Z = 0`.

## Physical regions

| Region | Position / extent | Initial S1 contents | Interaction |
|---|---|---|---|
| Navigation floor | In front of the workstation | Fetch starts at approximately `(0, -1.05, 0)` | Planar base motion |
| Countertop | `X=[-0.60,0.60]`, `Y=[-0.40,0.40]`, top at `Z=0.77` | kettle, coffee jar, sugar jar, spoon | Always visible work surface |
| C1 | Upper-left cabinet, centered near `(-0.35,0.35,1.15)`; usable opening about `0.40 m` wide by `0.41 m` high and `0.31 m` deep | mug, glass | Enlarged hinged-door cabinet; primary S1 search region |
| C2 | Upper-right cabinet, centered near `(0.35,0.35,1.15)`; same enlarged opening as C1 | plate, bowl, scanned canister distractor | Enlarged hinged-door cabinet |
| D1 | Lower-left drawer, centered near `(-0.35,-0.30,0.55)`; usable tray about `0.332 m` wide | fork, knife, stirrer | Wide tray; damped slide toward the robot |
| D2 | Lower-right drawer, centered near `(0.35,-0.30,0.55)`; usable tray about `0.332 m` wide | tongs, napkin, scanned spatula distractor | Wide tray; damped slide toward the robot |
| B1 | Lidded box on the right of the countertop, centered near `(0.44,0.01,0.77)`; inner footprint about `0.344 x 0.174 m` | tea box | Enlarged hinged lid and gripper-accessible opening |
| Serving area | Separate green table centered near `(0.90,0,0.75)` | empty | Goal destination |

Opening a container reveals and catalogs all objects in that region. An
already-inspected region remains known even if it is later closed.

## S1 visible objects

| Object | Initial location | Role |
|---|---|---|
| kettle | `counter_spot_1`, near `(-0.35,-0.08)` | Hot-water appliance |
| coffee jar | `counter_spot_2`, near `(-0.15,-0.08)` | Coffee ingredient |
| sugar jar | `counter_spot_3`, near `(0.05,-0.08)` | Optional coffee ingredient |
| spoon | `counter_spot_5`, near `(-0.25,-0.22)` | Stirring utensil |

The mug is the only unresolved required object at reset. C1 contains the exact
required mug and a glass that can act as a configured substitute.

The cabinet shelf splits each opening into an approximately `0.17 m` lower
bay and `0.24 m` upper bay. Both exceed the Fetch finger assembly's roughly
`0.13 m` maximum outer span, while the `0.40 m` cabinet width and the open
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

The drawer slides use lower-gain position control and additional damping to
avoid jerking loose objects when opened. The flat stirrer also has a matching
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

## Fetch mobile manipulator

The Fetch kinematic tree and meshes come from Gymnasium-Robotics. The original
benchmark model fixes the base; this scene adapts it to three planar controlled
joints:

- `robot0:base_forward_joint`: motion along Fetch's forward direction.
- `robot0:base_lateral_joint`: sideways motion.
- `robot0:base_yaw_joint`: rotation about the vertical axis.

At reset Fetch faces the workstation. The model also exposes the torso lift,
head pan/tilt, seven arm joints, and two gripper-finger joints through named
position actuators. The base is holonomic at this stage: it represents a
mobile-manipulator planning pose, not differential wheel dynamics. Wheel-level
drive dynamics can be introduced later without changing the scene regions.

Robot-mounted sensors:

- `head_camera_rgb`: Fetch head camera.
- `wrist_camera`: attached to the real Fetch gripper link.

Environment cameras remain:

- `left_shoulder_camera`
- `right_shoulder_camera`
- `overhead_camera`
- `front_camera`

## Intended S1 episode

1. Reset Fetch in front of the workstation with all containers closed.
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
the Fetch gripper should physically operate the same door, drawer, and lid
joints; the direct API then serves as an oracle/debug action.

## Run

```bash
MUJOCO_GL=glfw /home/naren/miniconda3/bin/python \
  -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug \
  --viewer --camera front_camera
```

Use `--no-robot` only when debugging the kitchen by itself.

To start directly in Fetch's head-camera view, replace `front_camera` with
`head_camera_rgb`.
