# Workshop (W1) Domain Specification

## Overview

The **Workshop (W1) Domain** is the third canonical robotics benchmark domain in `icra-we-ball`, joining the **Kitchen (S1)** object-function domain and the **Living Room (L2)** region-function domain.

While Kitchen explores *object-function discovery* ("which tools satisfy required task capabilities?") and Living Room explores *region-function grounding* ("which spatial regions satisfy placement conditions?"), the Workshop domain investigates **joint object-function and region-function grounding**:

> **Core Research Question**: Which observed functional objects (tools and fasteners) AND functional regions (work surfaces and parts containers) jointly satisfy task semantics, physical geometric mating, and relational set packing?

---

## Canonical Task Specification

```text
Repair the loose frame joint using an appropriate tool and fastener.
Arrange the required tool and hardware on a suitable nearby work surface,
and keep loose small parts in a suitable container.
```

### Functional Roles

1. **Object Function 1 (`CAN_DRIVE_SCREW`)**:
   - Requires a driver tool capable of driving threaded fasteners into the target joint.
   - Candidates: `workshop_long_phillips_driver` (valid), `workshop_stubby_phillips_driver` (reach failure), `workshop_flathead_screwdriver` (tip profile failure), `workshop_power_driver` (valid alternative in F1/F4), decoys (`pliers`, `combination_wrench`, `ratchet_wrench`, `wooden_mallet`).
2. **Object Function 2 (`CAN_FASTEN`)**:
   - Requires a compatible threaded fastener that fits the target hole (diameter 7mm, depth 30mm).
   - Candidates: `workshop_medium_phillips_screw` (valid), `workshop_short_phillips_screw` (depth failure), `workshop_long_phillips_screw` (depth failure), `workshop_hex_bolt` (recess profile failure).
3. **Region Function 1 (`WORK_SURFACE`)**:
   - Requires a planar work surface within reach of the repair fixture that can accommodate the bounding set of selected driver + fastener (`FITS_SET(work_surface, {driver, fastener})`).
   - Candidates: `MAIN_WORKBENCH_ZONE` (canonical), `TOOL_CART_TOP` (alternative), `NARROW_WALL_SHELF` (packing failure).
4. **Region Function 2 (`SMALL_PARTS_CONTAINER`)**:
   - Requires an open container with non-zero cavity volume to hold loose small parts and the removed seal cover (`FITS_IN(container, {parts})`).
   - Candidates: `PARTS_TRAY` (canonical shallow staging tray), `HARDWARE_BIN` (small bin), `TOOLBOX_COMPARTMENT` (deep tool compartment).

---

## Benchmark Architecture Flow

The Workshop domain enforces a strict scientific boundary between simulation truth, physical interaction, observation, and perception-based grounding:

```text
                  TASK INSTRUCTION
                         ↓
                   Physical Scene
                         ↓
                Inspect / Open Region
                         ↓
                     RGB / RGB-D
                         ↓
  Generic Object Instances + Neutral Region Proposals
                         ↓
                 [FUTURE PHASE 1]
           Semantic / Geometric Grounding
```

### Key Principles

1. **`open_container()` Never Returns Inventory**:
   - Opening a container only commands the actuator, simulates physics, and updates the physical open state.
   - It returns a neutral status dictionary (`{"region_id": "...", "opened": True, "newly_opened": True}`).
   - It never returns object lists, backend body names, semantic classes, or hidden counts.
   - Object discovery occurs strictly through subsequent visual observation (RGB-D point cloud perception).

2. **Neutral Region Proposals Are Based on Physical Existence, Not Benchmark Validity**:
   - Production `get_candidate_regions()` proposes candidate spatial patches based solely on whether the underlying physical bodies exist in the compiled MuJoCo model.
   - It does NOT filter by benchmark configuration (`active_surfaces` / `active_containers`).
   - Obstructed regions (such as the main workbench in `F2_REGION_ALTERNATIVE` or `I2_NO_WORK_SURFACE`) are still emitted as neutral spatial proposals; Phase 1 perception is responsible for inferring whether the candidate is usable or obstructed.
   - Physically absent bodies (e.g. removed containers in `I3_NO_PARTS_CONTAINER`) are not proposed.

3. **Production Candidate Regions Do Not Encode Function**:
   - The production API returns only `{"region_instance_id": "region_XXXX", "proposal_bounds_m": {...}, "observation_source": "workbench|cart|shelf"}`.
   - `proposal_type` ("surface" vs "container") is intentionally omitted to prevent leaking functional classifications.

4. **Privileged APIs Exist Only for Benchmark Evaluation and Oracle Testing**:
   - All ground-truth parameters, semantic affordances, exact dimensions, and oracle feasibility search algorithms are confined to explicit `privileged_*` methods.

---

## Controlled Benchmark Variant Matrix

The Workshop domain provides 14 standardized benchmark variants across 7 feasible configurations and 7 controlled infeasible failure modes:

| Variant | Outcome | Key Characteristic | Rejection Reason |
|---|---|---|---|
| `F0_BASE` | FEASIBLE | Canonical baseline: long Phillips driver + medium screw + main workbench zone + parts tray. | N/A |
| `F1_TOOL_ALTERNATIVE` | FEASIBLE | Tool alternative: both manual long driver and cordless power driver are valid. | N/A |
| `F2_REGION_ALTERNATIVE` | FEASIBLE | Region alternative: main workbench zone obstructed; tool cart top is valid. | N/A |
| `F3_DISTRIBUTED_OBJECTS` | FEASIBLE | Distributed search: valid driver is in `LEFT_DRAWER`, valid fastener in `TOOL_CABINET`. | N/A |
| `F4_OBJECT_REGION_COUPLING` | FEASIBLE | Relational packing: power driver requires large tool cart; manual driver fits auxiliary shelf. | N/A |
| `F5_DECOY_HEAVY` | FEASIBLE | Decoy heavy: multiple wrenches, mallets, pliers, bolts distributed across all regions. | N/A |
| `F6_LAYOUT_SWAPPED` | FEASIBLE | Layout rearranged: the tool cabinet, tool cart, shelf, and parts-storage locations are spatially swapped while drawer fixtures remain fixed. | N/A |
| `I0_NO_VALID_DRIVER` | INFEASIBLE | Semantic deficit: only wrenches, pliers, and mallets available; no screw driver. | `NO_VALID_DRIVER` |
| `I1_NO_VALID_FASTENER` | INFEASIBLE | Semantic deficit: only hex bolts available; no compatible screw fastener. | `NO_VALID_FASTENER` |
| `I2_NO_WORK_SURFACE` | INFEASIBLE | Region deficit: candidate work surfaces obstructed or missing. | `NO_WORK_SURFACE` |
| `I3_NO_PARTS_CONTAINER` | INFEASIBLE | Region deficit: no open container with containment volume available. | `NO_PARTS_CONTAINER` |
| `I4_TOOL_GEOMETRY_FAILURE` | INFEASIBLE | Tool geometry failure: only stubby driver available; shaft reach < required joint depth. | `TOOL_GEOMETRY_FAILURE` |
| `I5_OBJECT_REGION_PACKING_FAILURE` | INFEASIBLE | Relational packing failure: only narrow shelf available; set area exceeds usable surface. | `OBJECT_REGION_PACKING_FAILURE` |
| `I6_GLOBAL_CONFLICT` | INFEASIBLE | Global conflict: valid driver, fastener, and surface exist individually, but no compatible triplet exists. | `GLOBAL_CONFLICT` |

---

## Production-Safe vs Privileged Simulation Boundary

The Workshop domain enforces an absolute architectural separation between production-facing perception/planning interfaces and privileged simulation oracle interfaces:

### Production-Safe Interface (Zero Cheating / Zero Leakage)

- **Natural Language Task**: High-level task prompt (`"Repair the loose frame joint using an appropriate tool and fastener..."`).
- **Generic Object Instances (`get_observed_instances`)**: Deterministic generic IDs (`object_0001`, `object_0002`, ...) assigned deterministically during scene initialization. The production API returns only `{"instance_id": "object_XXXX", "source_region": "..."}`. It contains NO simulator backend body names, NO semantic classes, NO functional affordances (`can_drive_screw`), and NO physical dimensions.
- **Generic Region Proposals (`get_candidate_regions`)**: Region proposals return generic IDs (`region_0001`, `region_0002`), neutral proposal bounds (`proposal_bounds_m`) for RGB-D cropping, and observation handles. They contain NO ground-truth classes (`WORK_SURFACE`, `SMALL_PARTS_CONTAINER`), NO usable surface area (`usable_area_m2`), and NO cavity containment volume (`cavity_volume_m3`).
- **Neutral Target Workpiece (`get_target_workpiece_specification`)**: Workpiece observation exposes only workpiece position and localization handle (`target_instance_id`). It contains NO hole diameter, NO hole depth, NO driver/fastener function requirements, and NO recess profiles (`PH2`).
- **Container Articulation (`open_container`, `close_container`)**: Commands physical actuation without revealing contained inventory.
- **Task Observation State (`get_task_scene_state`)**: Exposes only known container open states and access status without benchmark variant metadata or hidden seal identities.
- **Raw Sensor Data**: RGB images, calibrated depth maps, and fused colored point clouds.

### Privileged Simulation Oracle (Benchmark Construction, Testing & Evaluation Only)

- **Simulator Backend Names**: MuJoCo body names (`workshop_long_phillips_driver`, `workshop_medium_phillips_screw`, etc.).
- **Privileged Object Metadata (`PRIVILEGED_WORKSHOP_ORACLE_SPECS`)**: Semantic classes, functional affordance labels (`can_drive_screw`, `can_fasten`), true shaft reach, tip profiles, fastener recess profiles, exact diameters, lengths, and bounding areas.
- **Privileged Storage Contents (`privileged_get_storage_contents`)**: True declared backend contents of a storage container.
- **Privileged Region Specs (`privileged_get_work_surface_specs`, `privileged_get_parts_container_specs`)**: Exact ground-truth planar dimensions, usable surface area, and open cavity containment volume.
- **Privileged Workpiece Ground Truth (`privileged_get_target_joint_specification`)**: Exact target hole diameter (7mm), hole depth (30mm), and required tool/fastener mating properties.
- **Physical Region Inference (`privileged_actual_work_surface_regions`, `privileged_actual_parts_container_regions`)**: Privileged audit routines that inspect compiled `MjModel` bodies and collision geoms directly, independent of YAML configuration metadata.
- **Privileged Scene Oracle Validator (`privileged_validate_variant_feasibility`)**: Oracle feasibility validator evaluating true 4-tuple physical constraints.
- **Privileged Task State (`privileged_get_task_scene_state`)**: Evaluation-grade task state including variant name, true seal location, and repair state.
- **Oracle Segmentation (`EXPLICIT_MUJOCO_ORACLE_DEBUG`)**: Labeled multi-camera point cloud exports used exclusively for ground-truth inspection and debug visualization.

---

## 3D Asset Provenance and Licensing

All 3D visual models and albedo textures are CC0 1.0 Universal Public Domain from Poly Haven:

- **Tools & Hardware**:
  - Long & Stubby Phillips Screwdrivers, Phillips Screws, Hex Bolt: derived from *Screwdrivers 02* (Martin Klekner, CC0).
  - Flathead Screwdriver: derived from *Screwdriver* (Martin Klekner, CC0).
  - Cordless Power Drill/Driver: derived from *Drill 01* (Mike van der Valk, CC0).
  - Wrenches & Decoys: *Combination Wrench*, *Ratchet Wrench*, *Pliers*, *Wooden Hammer 01* (Martin Klekner, CC0).
- **Furniture & Storage**:
  - Workbench: derived from *Wooden Table 02* (Fran Calvente, CC0).
  - Rolling Tool Cart: derived from *Tool Cart* (Mike van der Valk, CC0).
  - Parts Tray: derived from *Seedling Tray 01* (Mike van der Valk, CC0).
  - Hardware Storage Bin: derived from *Plastic Container* (Mike van der Valk, CC0).
  - Tool Compartment: derived from *Metal Toolbox* (Martin Klekner, CC0).

All asset downloads, coordinate frame conversions (Poly Haven Y-up to MuJoCo Z-up), dimension normalization, and hash generation are automated idempotently in `mujoco_scenes/scripts/prepare_workshop_assets.py` with provenance recorded in `mujoco_scenes/assets/workshop_realistic/manifest.json`.

---

## Scene Construction Architecture

The Workshop domain uses a modular, template-driven, and physically truthful variant generation architecture:

1. **Clean Structural Foundation (`assets/workshop_base.xml`)**:
   - Contains only structural geometry: walls, floor, workbench, pegboard, fixture workpiece, tool cart, narrow wall shelf, metal toolbox, parts tray, and hardware bin.
   - Articulated storage containers:
     - `left_tool_drawer` (prismatic slide along Y axis)
     - `right_tool_drawer` (prismatic slide along Y axis)
     - `tool_cabinet` (tabletop metal cabinet with hinged door)
   - 5 calibrated cameras.
   - All tools, fasteners, and decoys are excluded from the base XML.

2. **Template-Based Object Realization (`WORKSHOP_SIM_OBJECT_TEMPLATES`)**:
   - All 12 manipulable tool/fastener types are instantiated dynamically based on `workshop_variants.yaml`.
   - **Dynamic Free-Body Policy**: Every instantiated tool and fastener is a top-level `<body name="..."><freejoint name="..."/>...</body>` in `<worldbody>`, giving each object independent 6-DOF dynamic freedom.
   - **Deterministic Placement Slots**: Pre-computed collision-free resting positions and canonical orientations for `LEFT_DRAWER`, `RIGHT_DRAWER`, and `TOOL_CABINET`.
   - **Inspection Storage Welds**: Named equality constraints (`storage_weld_<object>`) ensure objects move smoothly with drawer/door articulation during inspection without popping or falling through, and are detached for robotic manipulation.

3. **Active Surface & Container Physics**:
   - `MAIN_WORKBENCH_ZONE`: In `F2_REGION_ALTERNATIVE`, `I2`, `I5`, and `I6`, a physical obstruction crate is instantiated on the workbench, visibly and physically blocking the surface.
   - `TOOL_CART_TOP` / `NARROW_WALL_SHELF`: Removed or obstructed in variants where they are inactive.
   - Candidate parts containers (`PARTS_TRAY`, `HARDWARE_BIN`, `TOOLBOX_COMPARTMENT`): Physically present only when declared active in the variant configuration.

4. **Physical Layout Profiles (`F6_LAYOUT_SWAPPED`)**:
   - In `F6_LAYOUT_SWAPPED`, the spatial layout is physically swapped:
     - `tool_cabinet`: moved from center (`x = 0.0`) to left workbench (`x = -0.40`).
     - `workshop_tool_cart`: moved from right (`x = 0.92`) to left (`x = -0.92`).
     - `workshop_narrow_shelf`: moved from left (`x = -0.70`) to right wall (`x = 0.70`).
     - `workshop_parts_tray`: moved from left (`x = -0.44`) to right (`x = 0.44`).
     - `workshop_hardware_bin`: moved from right (`x = 0.44`) to left (`x = -0.44`).

5. **Privileged Oracle Feasibility Validator (`privileged_validate_variant_feasibility`)**:
   - Performs a true scene-level search over all present bodies and active regions derived directly from compiled `MjModel` state.
   - Evaluates: `can_drive_screw`, `can_fasten`, `fits_hole` (radial clearance <= 7mm), `reaches_joint` (length >= 30mm), `driver_reaches` (reach >= 25mm), `tip_mates` (PH2 == PH2), `fits_work_surface` (usable area >= 1.2 * set area), `fits_parts_container` (open cavity > 0).
   - Produces exact oracle status (`FEASIBLE` vs `INFEASIBLE`) and exact failure codes (`NO_VALID_DRIVER`, `NO_VALID_FASTENER`, `NO_WORK_SURFACE`, `NO_PARTS_CONTAINER`, `TOOL_GEOMETRY_FAILURE`, `OBJECT_REGION_PACKING_FAILURE`, `GLOBAL_CONFLICT`).

---

## Multi-Camera Observation Rigs

The scene features 5 calibrated cameras calibrated across 4 sequential inspection stages:

```text
Stages:
  1. INITIAL: Full workbench, fixture, pegboard, and tool cart overview.
  2. LEFT_DRAWER: Close inspection of left sliding drawer contents.
  3. RIGHT_DRAWER: Close inspection of right sliding drawer contents.
  4. TOOL_CABINET: Inspection of tabletop cabinet interior and shelves.

Cameras:
  - workshop_camera_left
  - workshop_camera_right
  - workshop_camera_top
  - workshop_camera_front
  - workshop_camera_close
```

---

## CLI Usage

```bash
# 1. List all 14 benchmark variants
PYTHONPATH=. /home/naren/miniconda3/bin/python -m mujoco_scenes.workshop_scene --list-variants

# 2. Launch interactive viewer on variant F0_BASE
PYTHONPATH=. /home/naren/miniconda3/bin/python -m mujoco_scenes.workshop_scene --robot google --variant F0_BASE --viewer

# 3. Inspect specific opened regions and render front overview
MUJOCO_GL=egl PYTHONPATH=. /home/naren/miniconda3/bin/python -m mujoco_scenes.workshop_scene --variant F1_TOOL_ALTERNATIVE --open LEFT_DRAWER --open TOOL_CABINET --render overview.png

# 4. Run 5-camera RGB-D point cloud reconstruction across all 4 inspection stages
MUJOCO_GL=egl PYTHONPATH=. /home/naren/miniconda3/bin/python -m mujoco_scenes.workshop_pointcloud --variant F0_BASE --output runs/workshop_pc_f0

# 5. Run full 14-variant physical & oracle suite audit
MUJOCO_GL=egl PYTHONPATH=. /home/naren/miniconda3/bin/python -m mujoco_scenes.audit_workshop_scene --variant all --output-dir runs/workshop_scene_audit
```
