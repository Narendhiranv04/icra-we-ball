# Workshop YOLO-World L five-view production profile

This is a separate Workshop-only configuration. The canonical Workshop scene,
three-view rig, and canonical geometry configuration remain selectable.

- Runtime: `mujoco_scenes/configs/workshop_phase1_yoloworld_l_five_view_close.yaml`
- Five-view rig: `mujoco_scenes/configs/workshop_inspection_rigs_yoloworld_l_five_view_close.yaml`
- Visual profile: `mujoco_scenes/configs/workshop_visual_profile_yoloworld_l.yaml`
- L geometry tolerances: `mujoco_scenes/configs/workshop_geometry_inference_yoloworld_l.yaml`
- Vocabulary: `mujoco_scenes/configs/workshop_phase1_fm_contract_yoloworld_l.yaml`
- Views per inspected region: `LEFT`, `RIGHT`, `TOP`, `FRONT`, `CLOSE`
- Detector: `yolov8l-worldv2.pt`, CUDA device 0, 1280 px
- Decision path: production YOLO boxes plus RGB-D mask refinement; no oracle masks

## Frozen result

The final single-configuration run passed all 14 canonical Workshop variants.
Every feasible case passed exact witness mapping and every infeasible case passed
exact rejection-code matching.

| Variant | Observed result | Evaluator |
|---|---|---|
| `F0_BASE` | FEASIBLE | PASS |
| `F1_TOOL_ALTERNATIVE` | FEASIBLE | PASS |
| `F2_REGION_ALTERNATIVE` | FEASIBLE | PASS |
| `F3_DISTRIBUTED_OBJECTS` | FEASIBLE | PASS |
| `F4_OBJECT_REGION_COUPLING` | FEASIBLE | PASS |
| `F5_DECOY_HEAVY` | FEASIBLE | PASS |
| `F6_LAYOUT_SWAPPED` | FEASIBLE | PASS |
| `I0_NO_VALID_DRIVER` | `NO_VALID_DRIVER` | PASS |
| `I1_NO_VALID_FASTENER` | `NO_VALID_FASTENER` | PASS |
| `I2_NO_WORK_SURFACE` | `NO_WORK_SURFACE` | PASS |
| `I3_NO_PARTS_CONTAINER` | `NO_PARTS_CONTAINER` | PASS |
| `I4_TOOL_GEOMETRY_FAILURE` | `TOOL_GEOMETRY_FAILURE` | PASS |
| `I5_OBJECT_REGION_PACKING_FAILURE` | `OBJECT_REGION_PACKING_FAILURE` | PASS |
| `I6_GLOBAL_CONFLICT` | `GLOBAL_CONFLICT` | PASS |

Frozen evidence is in:

`outputs/workshop_yoloworld_l_five_view_close/final_14_bright_profile_frozen_v5/`

Each variant directory contains raw RGB frames, full bounding-box predictions,
annotated overlays, observed evidence graphs, privileged post-hoc mappings,
`episode_result.json`, and `evaluation_metrics.json`.

## Profile decisions

- Shadows are disabled; headlight/fill lighting and matte storage colors make
  the dark tools and fasteners separable from the cabinet and drawer interiors.
- Cabinet and drawer rigs retain five complete viewpoints. Drawer images are
  rolled so long tools remain horizontal and fully inside the detector frame.
- `F5` uses a closer cabinet detail view for its dense decoy inventory.
- `F3` and `F6` use a collision-aligned continuous render silhouette for the
  long Phillips driver. Collision geometry, pose, and functional dimensions are
  unchanged.
- The base scene's incorrect screwdriver texture on screw meshes is hidden in
  this profile and replaced by metallic Phillips-fastener render geometry.
- Small fasteners require independent support from at least two cameras.
- Endpoint classification uses measured size, anisotropy, and radial symmetry;
  unresolved sub-resolution endpoints cannot override a resolved endpoint.
- Driver reach supports slender manual tools and a bounded full-size compact
  power-driver branch. The intermediate stubby geometry remains invalid.
- Conclusive five-view surface obstruction and container evidence takes
  precedence over incidental small-object detector misses during diagnosis.

## Reproduction

Run each canonical variant with production proposals and evaluation:

```bash
MUJOCO_GL=egl /home/naren/miniconda3/bin/python \
  mujoco_scenes/run_workshop_phase1.py \
  --variant F0_BASE \
  --mask-backend production \
  --semantic-backend production \
  --proposal-mode yolo_only \
  --config mujoco_scenes/configs/workshop_phase1_yoloworld_l_five_view_close.yaml \
  --output outputs/workshop_yoloworld_l_five_view_close/reproduction \
  --evaluate
```

Replace `F0_BASE` with each variant listed in the table for the complete suite.
