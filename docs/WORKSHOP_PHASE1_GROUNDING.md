# Workshop W1 Phase-1 Functional Grounding

Workshop Phase 1 composes the functional-grounding structure already used in
the other project domains. Kitchen grounds object functions from semantic
candidates, point-cloud properties, and geometric relations. Living Room
grounds region functions from semantic region candidates, region-local
point-cloud measurements, and object-region relations. Workshop uses the same
separation jointly: semantic object and region candidates are verified by
object-target, object-object, and object-region relations before a complete
assignment is accepted.

Phase 1 stops at a verified functional witness or a grounded rejection. It does
not perform PDDL/PDDLStream planning, robot motion, picking, placing, fastening,
or physical execution. The Workshop meshes, textures, poses, cameras, variant
definitions, and inspection rigs are frozen.

## Architecture

The task is converted once into `ManualWorkshopFMContract`, a deterministic
surrogate for the single FM response that would occur at episode initialization.
No live foundation model is called. `FMRequirementProvider` implements the same
boundary and fails clearly until a real backend exists. The provider exposes
requirements, ranked canonical categories, one detector display label per
canonical category, normalization aliases, and required relations. It does not
contain millimetre thresholds or point-cloud verifier parameters.

YOLO-World is the only production semantic detector. Its `set_classes(...)`
vocabulary comes exclusively from the active provider. Exactly one detector
class is installed per canonical category. Aliases such as `manual screwdriver`,
`cordless drill`, and `shallow tray` normalize external labels but are never
independent YOLO classes. It detects broad categories such as
`screwdriver`, `power_driver`, `screw`, `bolt`, `workbench`, `tool_cart`,
`shelf`, `parts_tray`, and `hardware_bin`. It is not asked functional-affordance
sentences and is not supplemented by CLIP, another VLM, or a point-cloud
semantic classifier.

The active calibrated stage volume is projected to a conservative image crop so
tiny drawer hardware receives adequate detector resolution. Detections are then
depth-refined inside YOLO boxes and backprojected to world-frame points.
Before tracking, same-camera duplicates are suppressed using canonical label,
box IoU, mask overlap, 3-D centroid distance, and cloud-AABB overlap. Competing
cross-category hypotheses are retained as uncertainty metadata on one physical
proposal rather than creating duplicate objects. Proposals are fused across
views and associated with persistent generic IDs (`object_XXXX`). Each track
also retains a typed, stage-local
`MeasurementEvidence` record with its source stage/region, camera contributors,
point count, measurement method, and quality metadata. Persistence accumulates
semantic/entity history; property extraction uses the latest valid local
measurement rather than simulator body geometry.

Neutral calibrated region volumes select evidence and are represented by
stable generic IDs (`region_XXXX`). YOLO remains the semantic source. Refined
YOLO region pixels are backprojected and assigned to at most one neutral region
using point-in-volume fraction, 3-D centroid proximity, projected 2-D overlap,
and multi-view confidence consensus. Evidence accumulates across views and
inspection stages. Proposal/backend names are not semantic evidence. Configured
bounds are never treated as
measured support or cavity dimensions. A dominant observed support plane gives
the usable oriented footprint. The shared Kitchen open-cavity extractor checks
rim enclosure, open centre, observed interior, depth, and camera coverage.
Missing or unreliable geometry remains `UNKNOWN`.

## Configuration separation

The manual future-FM contract is
`mujoco_scenes/configs/workshop_phase1_fm_contract.yaml`. It defines exactly:

- `CAN_DRIVE_SCREW(driver)`: `REACHES_TARGET`, `COMPATIBLE_WITH`
- `CAN_FASTEN(fastener)`: `COMPATIBLE_WITH_TARGET`
- `WORK_SURFACE(region)`: `PLANAR_SUPPORT`, `FITS_SET_ON`
- `SMALL_PARTS_CONTAINER(region)`: `OPEN_CAVITY`, `FITS_IN`

The same file is loaded for F0-F6 and I0-I6. Flathead, stubby, and long manual
drivers all remain broad semantic `screwdriver` candidates. Screw and bolt both
remain broad `CAN_FASTEN` candidates. Physical rejection happens downstream.

Deterministic, category-independent settings live in
`mujoco_scenes/configs/workshop_geometry_inference.yaml`: robust percentiles,
point-count/quality gates, plane and cavity resolution, local-interface
measurement settings, grasp/clearance allowances, and packing margins. Runtime
detector, tracker, voxel, and inspection settings remain in
`mujoco_scenes/configs/workshop_phase1.yaml`.

The current detector classes are the provider-owned display strings for the
eleven canonical categories: `hand screwdriver`, `power drill`, `fastener`,
`hex bolt`, `adjustable wrench`, `pliers`, `workbench`, `tool cart`, `shelf`,
`parts tray`, and `hardware bin`. The serialized canonical output records both
this ordered list and the display-label-to-canonical map.

## Generic measurements and relations

Object measurements use robust PCA and local distal subsets. They record a
centroid, principal axes, robust dimensions, total/usable length, maximum cross
section, oriented horizontal footprint, local transverse interface dimensions,
quality, cameras, stage, and inspection region. Local interfaces are geometric
descriptors (`CROSS_LIKE`, `SLOT_LIKE`, `HEX_LIKE`, or `UNKNOWN`), never object
names. No label/category branch sets a reach or dimension, and there is no
stubby cap or power-driver floor.

Target calibration only localizes a target ROI. Multi-view RGB-D points provide
the front-plane statistic, recess cluster, opening footprint, and recess depth.
There are no 7 mm/30 mm fallback measurements: unresolved target evidence makes
target-dependent relations `UNKNOWN`.

The verifier exposes a compact relation registry:

- `REACHES_TARGET`: observed usable length versus observed recess depth and a
  configured generic grasp allowance, with a signed margin.
- `COMPATIBLE_WITH`: measured driver working-end descriptor versus measured
  fastener head/interface descriptor.
- `COMPATIBLE_WITH_TARGET`: measured fastener length, shaft cross section, local
  interface evidence, and observed target opening/depth.
- `PLANAR_SUPPORT`: observed support-plane normal, planarity, thickness, and
  footprint.
- `FITS_SET_ON`: 0/90-degree oriented two-object arrangements with edge and
  inter-object clearances, non-overlap, tested arrangements, and signed margins.
- `OPEN_CAVITY`: shared structural rim/open-centre/interior evidence.
- `FITS_IN`: observed fastener dimensions versus observed opening and cavity.

Every predicate returns `TRUE`, `FALSE`, or `UNKNOWN`. Semantic and geometry
combine as: PASS+PASS=PASS; PASS+UNKNOWN and UNKNOWN+PASS=UNKNOWN; any FAIL
combination is FAIL. Only PASS enters verified witness pools. No detections after
inspection exhaustion is `INSUFFICIENT_EVIDENCE`, not infeasibility.
The active requirement's `required_relations` list selects which unary and joint
registry entries execute. Unary relations are `REACHES_TARGET`,
`COMPATIBLE_WITH_TARGET`, `PLANAR_SUPPORT`, and `OPEN_CAVITY`; joint relations
are `COMPATIBLE_WITH`, `FITS_SET_ON`, and `FITS_IN`. Removing a relation from a
provider contract changes verification without a source-code edit.

## Joint search and diagnoses

The authoritative path searches every verified
`(driver, fastener, work_surface, parts_container)` tuple. Ranking proposes; the
relations decide. F4 selects the cart because the power-driver/screw set fails
`FITS_SET_ON` for the shelf but passes on the cart. In I5 the same shelf remains
a semantic work-surface with `PLANAR_SUPPORT=TRUE`; the same packing relation
fails, producing `OBJECT_REGION_PACKING_FAILURE`. I1 retains semantic screw and
bolt candidates but none passes `COMPATIBLE_WITH_TARGET`, yielding
`NO_VALID_FASTENER`. I4 retains the stubby semantic screwdriver but
`REACHES_TARGET=FALSE`, yielding `TOOL_GEOMETRY_FAILURE`. I6 retains broad local
candidates, while interface and reach relations eliminate every complete tuple,
yielding `GLOBAL_CONFLICT`.

Semantic-only selects solely by contract membership and does not invoke the
point-cloud verifier. No-joint-coupling selects
independently verified unary roles without compatibility or packing. No-
persistence replaces the tracker, semantic cache, and current object evidence at
each inspection stage. Single-front-view explicitly selects
`workshop_camera_front`.

Oracle-mask and oracle-semantics experiments are separate. Oracle masks provide
only object instance pixels; YOLO labels are associated independently using
2-D overlap and calibrated 3-D proximity,
and raw YOLO furniture detections remain available to region grounding. Oracle
semantics is an explicitly privileged upper bound and maps simulator subtypes
back to the same broad contract taxonomy.

## Reproduction

The repository-owned runner generates oracle-semantics/perfect-mask,
YOLO-semantics/oracle-mask, full-production, semantic diagnostics, and two
separately labelled ablation suites. Production ablations retain real YOLO
perception. Controlled grounding ablations use oracle semantics/masks so
upstream detector misses do not hide geometry, persistence, or coupling effects:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/naren/miniconda3/bin/python -m mujoco_scenes.run_workshop_phase1 \
  --canonical --output outputs/workshop_phase1_final
```

Focused and full regressions:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/naren/miniconda3/bin/python -m pytest \
  mujoco_scenes/tests/test_workshop_phase1.py \
  mujoco_scenes/tests/test_workshop_phase1_no_privileged_leaks.py -v

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/naren/miniconda3/bin/python -m pytest mujoco_scenes/tests -q
```

The controlled oracle-semantics/perfect-mask gate must be 14/14 exact before
freeze. Real YOLO accuracy is reported honestly and does not change the frozen
architecture.

## Canonical freeze result

The frozen canonical run records 14/14 for controlled full grounding, 1/14 for
YOLO semantics with oracle object masks, and 2/14 for full production. The two
production passes are I3 (`NO_PARTS_CONTAINER`) and I5
(`OBJECT_REGION_PACKING_FAILURE`). These production results are a perception
limitation, not a closed production benchmark: screws have zero aggregate
recall, bolt detections have zero correct associations, and shelf detections
have a high false-positive rate in the simulated RGB domain. The complete
variant tables, post-hoc semantic diagnostics, and failure-layer breakdown are
stored under `outputs/workshop_phase1_final/`.

The production ablation scores are full 2/14, semantic-only 1/14, no-joint
1/14, no-persistence 0/14, and single-front-view 2/14. With controlled oracle
perception, the corresponding scientific grounding scores are full 14/14,
semantic-only 4/14, no-geometry 4/14, no-joint 8/14, no-persistence 6/14, and
single-front-view 7/14.

## Limitations

The current requirement contract is manual rather than a live FM output.
Candidate region proposal volumes are calibrated. Observations are simulated
RGB-D, zero-shot YOLO recall is sensitive to the MuJoCo rendering domain, and
small local interface structures approach the available depth resolution.
