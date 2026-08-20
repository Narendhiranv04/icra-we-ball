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
boundary and fails clearly until a real backend exists. Future FM output supplies
broad functional requirements, explicitly ranked semantic candidate categories,
and required relations. The provider exposes those requirements, the ranked
detector vocabulary, one detector label per canonical category, normalization
aliases, and required relations. It does not contain millimetre thresholds or
point-cloud verifier parameters. If the manual contract is absent or malformed,
initialization fails; there is no hidden Workshop dictionary in Python.

YOLO-World is the only production semantic detector. Its `set_classes(...)`
vocabulary comes exclusively from the active provider. Exactly one detector
class is installed per canonical category. Aliases such as `manual screwdriver`,
`cordless drill`, and `shallow tray` normalize external labels but are never
independent YOLO classes. It detects broad categories such as
`screwdriver`, `power drill`, `screw`, `bolt`, `workbench`, `tool cart`,
`shelf`, `parts tray`, and `hardware bin`. It is not asked functional-affordance
sentences and is not supplemented by CLIP, another VLM, or a point-cloud
semantic classifier.

FM rank controls vocabulary selection and serialization order. The runtime
takes the first `max_detector_vocabulary_size` ranked entries (currently all 11
fit within the budget of 32) and passes those labels to
`YOLOWorld.set_classes(...)`. Rank never alters YOLO's visual-model confidence.

Each camera runs the same model and vocabulary on the full frame and the
calibration-derived active-stage crop. Boxes are mapped back to global image
coordinates. A generic overlapping 2x2 tile pass was evaluated but not selected:
it did not recover screws or bolts and materially increased false positives and
runtime. Detections are depth-refined inside YOLO boxes and backprojected to
world-frame points. Before tracking, same-camera duplicates across inference
scales are suppressed using box IoU, refined-mask overlap, 3-D centroid distance,
and cloud-AABB overlap. Competing
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
using point-in-volume fraction, 3-D centroid proximity, and projected 2-D
overlap. Association quality is `0.55*inside + 0.25*centroid + 0.20*overlap`;
semantic support is detector confidence times that quality. Only the strongest
hypothesis per region/camera/stage contributes to multi-view consensus. Evidence accumulates across views and
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

The ranked detector classes are the provider-owned broad physical strings:
`screwdriver`, `power drill`, `screw`, `bolt`, `wrench`, `pliers`, `workbench`, `tool cart`, `shelf`,
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
- `COMPATIBLE_WITH_TARGET`: measured fastener length and shaft cross section
  versus observed target opening/depth and generic clearances. Fastener head
  interface type belongs only to driver-fastener `COMPATIBLE_WITH`.
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
persistence clears object tracking, semantic caches, region semantic history,
region fused points, and accumulated region geometry at each inspection stage;
neutral calibrated proposal identities remain available for rediscovery.
Single-front-view explicitly selects
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
  --calibrate-detector --output outputs/workshop_phase1_final

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

The final canonical run records 14/14 for controlled full grounding, 0/14 for
YOLO semantics with oracle object masks, and 1/14 for full production. The sole
production pass is I4 (`TOOL_GEOMETRY_FAILURE`). This is an architecture freeze,
not semantic closure: screw and bolt each remain at zero recall, screwdriver
recall is 0.5833, wrench recall is zero, and shelf remains an uncontrolled
zero-shot confusion class despite a far cleaner global operating point. Missing
semantics propagate through strict tri-state reasoning rather than being silently
converted to physical absence.

The selected same-family checkpoint is YOLO-World medium-v2. The global detector
operating point is confidence 0.01, NMS IoU 0.35, inference size 1280, and at
most 50 proposals per inference pass. It was selected only on F0/F1/F5 using a
fixed detector-quality criterion; the other 11 variants were held out from
tuning. The final production aggregate is 6.4536 accepted physical proposals
per camera-stage, 4.5714 tracks per variant, and 964 cross-scale/cross-label
duplicates suppressed. Complete calibration and category metrics are stored in
the canonical artifacts.

The production ablation scores are full 1/14, semantic-only 1/14, no-joint
1/14, no-persistence 0/14, and single-front-view 1/14. With controlled oracle
perception, the corresponding scores are full 14/14, semantic-only 4/14,
no-joint 8/14, no-persistence 7/14, and single-front-view 8/14. `NO_GEOMETRY`
remains backward-compatible internally but is not published as a duplicate arm.

## Limitations

The current requirement contract is manual rather than a live FM output.
Candidate region volumes and the target ROI are calibrated spatial proposals,
not automatically discovered semantic regions. Production never uses simulator
semantic identities; privileged mappings exist only in post-hoc evaluation.
Observations are simulated RGB-D, zero-shot YOLO recall is sensitive to the
MuJoCo rendering domain, and small local interface structures approach the
available RGB/depth resolution. The honest final verdict is: **PHASE 1
ARCHITECTURE FROZEN — YOLO PERCEPTION LIMITATION REMAINS**.
