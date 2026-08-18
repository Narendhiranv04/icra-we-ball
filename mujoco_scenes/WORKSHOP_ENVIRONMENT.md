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
   - Candidates: `MAIN_WORKBENCH_ZONE` (canonical), `TOOL_CART_TOP` (alternative), `NARROW_WALL_SHELF` (packing failure), `HIGH_CABINET_TOP` (inaccessible/contextual failure).
4. **Region Function 2 (`SMALL_PARTS_CONTAINER`)**:
   - Requires an open container with non-zero cavity volume to hold loose small parts and the removed seal cover (`FITS_IN(container, {parts})`).
   - Candidates: `PARTS_TRAY` (canonical shallow staging tray), `HARDWARE_BIN` (small bin), `TOOLBOX_COMPARTMENT` (deep tool compartment).

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
| `F6_LAYOUT_SWAPPED` | FEASIBLE | Swapped layout: storage regions and staging areas mirrored. | N/A |
| `I0_NO_VALID_DRIVER` | INFEASIBLE | Semantic deficit: only wrenches, pliers, and mallets available; no screw driver. | `NO_VALID_DRIVER` |
| `I1_NO_VALID_FASTENER` | INFEASIBLE | Semantic deficit: only hex bolts available; no compatible screw fastener. | `NO_VALID_FASTENER` |
| `I2_NO_WORK_SURFACE` | INFEASIBLE | Region deficit: candidate work surfaces obstructed or missing. | `NO_WORK_SURFACE` |
| `I3_NO_PARTS_CONTAINER` | INFEASIBLE | Region deficit: no open container with containment volume available. | `NO_PARTS_CONTAINER` |
| `I4_TOOL_GEOMETRY_FAILURE` | INFEASIBLE | Tool geometry failure: only stubby driver available; shaft reach < required joint depth. | `TOOL_GEOMETRY_FAILURE` |
| `I5_OBJECT_REGION_PACKING_FAILURE` | INFEASIBLE | Relational packing failure: only narrow shelf available; set area exceeds usable surface. | `OBJECT_REGION_PACKING_FAILURE` |
| `I6_GLOBAL_CONFLICT` | INFEASIBLE | Global conflict: valid driver, fastener, and surface exist individually, but no compatible triplet exists. | `GLOBAL_CONFLICT` |

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
# List all 14 benchmark variants
PYTHONPATH=. /home/naren/miniconda3/bin/python mujoco_scenes/workshop_scene.py --list-variants

# Launch interactive viewer with Google Robot on variant F0_BASE
PYTHONPATH=. /home/naren/miniconda3/bin/python mujoco_scenes/workshop_scene.py --robot google --variant F0_BASE --viewer

# Inspect specific opened regions and render front overview
MUJOCO_GL=egl PYTHONPATH=. /home/naren/miniconda3/bin/python mujoco_scenes/workshop_scene.py --variant F1_TOOL_ALTERNATIVE --open LEFT_DRAWER --open TOOL_CABINET --render overview.png

# Run 5-camera RGB-D point cloud reconstruction across all 4 inspection stages
MUJOCO_GL=egl PYTHONPATH=. /home/naren/miniconda3/bin/python mujoco_scenes/workshop_pointcloud.py --variant F0_BASE --output runs/workshop_pc_f0
```
