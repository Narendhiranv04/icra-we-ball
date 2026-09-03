# Ground-truth execution and evidence ablations

Ground-truth execution and functional grounding are intentionally separate.
The first validates real MuJoCo action primitives. The second measures which
evidence is allowed to ground functional roles before symbolic planning.

## Ground-truth execution

Run a single variant interactively:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.run_kitchen_ground_truth_execution \
  --variant F0_ALL_VISIBLE --show --strict-robot-execution
```

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.run_living_room_execution \
  --variant F0_ALL_OBJECTS_IN_STAGING --show
```

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.run_workshop_ground_truth_execution \
  --variant F0_MANUAL_FIRST_ONE_REGION --show
```

Use `--dry-run` to generate and symbolically preflight actions without moving
the robot. Kitchen supports `--variant all --dry-run`; Living Room and Workshop
also accept `--variant all`.

## Functional grounding ablations

`joint` uses semantic labels and geometric predicates. `semantic_only` ignores
geometric predicates and relations. `geometric_only` ignores semantic labels
but retains entity kinds, unary geometry, numeric constraints, and relations.

For the component ablation, the three independently removable evidence
components are:

- `semantic`: semantic role/category compatibility;
- `unary`: unary predicates and numeric geometric properties; and
- `binary`: tool-target, object-region, and other directed geometric relations.

Use `--evidence-components` to override the aggregate mode. These are the four
primary leave-one-out conditions:

```bash
# Full pipeline
--evidence-components semantic,unary,binary
# Remove semantic / unary geometry / binary geometry respectively
--evidence-components unary,binary
--evidence-components semantic,binary
--evidence-components semantic,unary
```

For example, this runs the normal GT pipeline with semantic evidence and
binary relations but no unary geometry. It performs the usual region-search
and produces an action sequence; `--dry-run` means that OPEN uses direct scene
articulation rather than robot motion.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.functional_tamp_pipeline.run \
  --domain kitchen --variant F0_ALL_VISIBLE --mode gt --dry-run \
  --evidence-components semantic,binary \
  --output-root runs/component_ablation/no_unary
```

This pipeline entry point currently stops at an action sequence. It does not
yet execute the final pick/place/pour/stir sequence under each mask. Therefore
use it for grounding, search, and planning ablations—not a physical-execution
success claim.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.functional_tamp_pipeline.run \
  --domain kitchen --variant K1 --mode gt --dry-run \
  --evidence-mode joint --output-root runs/functional_joint
```

Use the same command with `semantic_only` and `geometric_only`. For a variant
sweep, use `scripts/evaluate_functional_tamp_variants.py` with the same flag.
These runs produce grounding assignments and symbolic plans, not physical
execution. The later VLM interface will replace the GT functional graph while
keeping the grounding and planning interface unchanged.

## Privileged GT-evidence ablation

For the controlled semantic/geometric experiment, use the dedicated offline
runner below. It gives every condition the same complete simulator-derived
scene graph and disables only the selected evidence type. It does **not** run
YOLO, a VLM, search, symbolic planning, or robot execution.

```bash
.venv/bin/python -m mujoco_scenes.run_gt_evidence_ablation \
  --domains kitchen,living_room,workshop \
  --output-root runs/gt_evidence_ablation/final
```

To run all seven non-empty evidence masks with the same privileged scene graph:

```bash
.venv/bin/python -m mujoco_scenes.run_gt_evidence_ablation \
  --domains kitchen,living_room,workshop \
  --component-masks all \
  --output-root runs/gt_evidence_ablation/components_all
```

The primary paper comparison should normally use `full`, `no_semantic`,
`no_unary`, and `no_binary`. The single-component masks are diagnostic rather
than principal results.

It writes one artifact per `(domain, variant, evidence mode)`, plus
`summary.json` and `summary.csv`. Report these metrics:

- `outcome_agreement_pct`: whether feasible/infeasible matches the GT label.
- `feasible_completion_pct` and `infeasible_rejection_pct`: the two classwise
  parts of outcome agreement.
- `false_completion_pct`: infeasible variants incorrectly accepted.
- `gt_valid_selection_pct`: among accepted variants, the selected roles and
  operation bindings pass the full GT semantic-and-geometric graph.
- `semantic_role_validity_pct`, `geometric_role_validity_pct`, and
  `operation_binding_validity_pct`: validity of accepted bindings under the
  withheld GT evidence. `mean_grounding_ms` is recorded separately.

The three conditions use complete privileged evidence only for this ablation;
they should never be compared to RGB/VLM perception performance.

Workshop binary oracle evidence is computed from the privileged construction
geometry of the instantiated scene. `REACHES_TARGET` compares driver reach
with target depth; `COMPATIBLE_WITH` compares driver-tip and fastener-recess
profiles and widths; `COMPATIBLE_WITH_TARGET` checks screw length and shaft
diameter against the target depth, opening, and radial clearance. Semantic
categories are not read by these relation evaluators. The production pipeline
continues to estimate the analogous quantities from RGB-D point clouds.

The current ten Workshop variants contain only compatible drivers, one
compatible screw, a hammer distractor, and missing-object infeasible cases.
Consequently, removing binary evidence can still produce the same outcomes as
the full oracle. Add wrong-interface and wrong-dimension variants before using
Workshop to quantify the benefit of geometric verification.
