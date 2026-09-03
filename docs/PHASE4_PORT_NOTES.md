# Phase-4 execution-layer port notes

Date: 2026-09-02
Target branch: `phase4_integration` (branched from `baseline_setup` with the
existing dirty worktree preserved).

Source: `origin/phase4/execution-integration-replay-contract` at `863611c`
("Phase 4: resolve Kitchen park specs by backend"), which is Naren's Phase-3
line plus 21 Phase-4 commits. Merge base with `baseline_setup` is `b4dcbd6`
(2026-08-05); the shared execution modules were added independently on both
lines with no common commit, so this was a reviewed file-level port, not a
`git merge`.

## Added (new files, taken verbatim)

```text
mujoco_scenes/phase4_execution.py
mujoco_scenes/phase4_kitchen.py
mujoco_scenes/phase4_living_room.py
mujoco_scenes/phase4_workshop.py
mujoco_scenes/phase4_workshop_entities.py
mujoco_scenes/run_phase4_execution.py
mujoco_scenes/run_phase4_gt_suite.py
mujoco_scenes/run_workshop_phase4_controller_development.py
mujoco_scenes/audit_workshop_scene.py          # required by the ported test_workshop.py
mujoco_scenes/tests/test_phase4_execution.py
docs/PHASE3_FREEZE_AND_PHASE4_HANDOFF.md
docs/PHASE4_SYNC_CURRENT_HANDOFF.md
docs/PHASE4_SYNC1_STRICT_BOUNDARY.md
docs/PHASE4_P4_SYNC2_STRICT_GRASP_PROVENANCE.md
docs/PHASE4_P4_SYNC2_PROVENANCE.json
docs/PHASE4_P4C1_WORKSHOP_CONTROLLER_BRINGUP.md
docs/PHASE4_P4C1_PROVENANCE.json
docs/PHASE4_P4C1_1_ACCESS_PICK_CLOSURE.md
docs/PHASE4_P4C1_1_PROVENANCE.json
docs/PHASE4_CERTIFICATION_AUDIT.md             # branch runs/phase4_certification_audit.md
```

## Replaced with the branch version

These carry the calibrated Phase-4 controllers and add symbols without
removing any that this tree used:

```text
mujoco_scenes/kitchen_ground_truth_execution.py      (+serving_utensil_containment_evidence)
mujoco_scenes/workshop_ground_truth_execution.py     (+9 strict grasp/insert/place verifiers)
mujoco_scenes/live_mosaic_viewer.py                  (+LiveMosaicViewerClosed)
mujoco_scenes/configs/kitchen_execution_semantics.json
mujoco_scenes/functional_tamp_pipeline/audit.py      (+prompt-leakage and provenance auditing)
mujoco_scenes/tests/test_kitchen_ground_truth_execution.py
mujoco_scenes/tests/test_workshop_ground_truth_execution.py
mujoco_scenes/tests/test_workshop.py
```

## Merged two ways

`mujoco_scenes/generic_manipulation.py` — branch base (its
`allowed_environment_geoms` collision API is threaded through
`MuJoCoBaseCollisionChecker` and used by `workshop_ground_truth_execution.py`),
plus these local additions restored:

- `transport_upright_axis_local` / `transport_max_tilt_rad` on `SimplePickSpec`
- `upright_preserving_gripper_rotation`
- `ProfiledIK = CanonicalProfiledIK`, i.e. the name is rebound to
  `mujoco_scenes.ik.ProfiledIK`. That class is a backend *dispatcher*, not a
  different solver: with `MUJOCO_IK_BACKEND` unset it defaults to `legacy` and
  runs `ik.DampedLeastSquaresIK`, the same damped-least-squares algorithm as
  the in-file class the branch uses, refactored onto a shared
  `_ProfileIKBase` (`_validate_target` / `_initial_qpos` helpers, module-level
  `rotation_vector`) with the same 1200-iteration default. So under the default
  environment the Phase-4 controllers run equivalent IK.
  The behaviour only diverges if `MUJOCO_IK_BACKEND` is set to `mink` or
  `auto`, which selects the Mink solver (installed here, opt-in per
  `mujoco_scenes/ROBOT_CALIBRATION.md`). Kept because this tree's comment
  states the consolidation is intentional and repo-wide.

`mujoco_scenes/living_room_mobile_execution.py` — branch base (execution
evidence join against `compatibility_matrix.json`, progress callbacks, strict
`post_release_dynamics_modification_enabled`, wider placement spread, legacy
`observed_grounding` variant tolerance), with one local fix restored: absent
region backends are filtered observationally instead of raising, so infeasible
variants such as `I0_NO_SHARED_TABLE` resolve and let the planner reject them.
Covered by
`vlm_tamp_baseline/tests/test_living_room.py::test_missing_region_variant_resolves_without_backend_probe_failure`.

`mujoco_scenes/kitchen_object_manipulation.py` — branch base (bowl grasp
candidates, held-payload geom exemptions for base collision checking), plus the
local `KITCHEN_ARM_COMMAND_SPEED` constant retained as the named call site.
**Resolved:** the constant is set to the branch's calibrated `0.60`, not this
tree's previous `0.85`, on the author's instruction. Slower actuator tracking
prevents overshoot across the unchanged shoulder-mount collision boundary, and
the Phase-4 kitchen controllers were tuned at `0.60`.
`mujoco_scenes/tests/test_robot_profiles.py` asserts the new value.

## Restored to the local version after replacement

Phase-4 commits never touched these, their divergence is Phase-3-era, and the
local files are strict symbol supersets:

```text
mujoco_scenes/scene_loader.py               (local-only: _apply_robot_home_pose, _validate_step_count, _validate_render_dimensions)
mujoco_scenes/kitchen_execution_entities.py (local-only: apply_within_region_execution_calibration + 5 helpers)
```

## Local changes made for the port

- `mujoco_scenes/functional_tamp_pipeline/run.py`: the Workshop branch now
  writes `plan_grounding_audit.json` (`home_region=SURFACE`). Phase 4 rejects a
  handoff without it.
- `mujoco_scenes/functional_tamp_pipeline/domains/living_room.py`: same, with
  `home_region="staging_tray"`.
- `mujoco_scenes/run_phase4_execution.py`: `SUPPORTED_MUJOCO_VERSION = "3.3.5"`
  became `CALIBRATED_MUJOCO_VERSION` plus
  `SUPPORTED_MUJOCO_VERSIONS = ("3.3.5", "3.3.6")`, so this environment's 3.3.6
  runs. Every execution artifact now carries a `mujoco_runtime` record with the
  installed version and `runs_on_calibrated_runtime`, and the runner prints a
  warning when off the calibrated runtime.

## Workshop expected-GT actions: both vocabularies kept

Nothing was switched. The active references remain the 28-action vocabulary in
`EXPECTED_GT_ACTIONS/workshop/`; the branch's 6-action vocabulary is a
reference-only copy in `EXPECTED_GT_ACTIONS/_phase4_branch_workshop/`, which no
loader reads. Kitchen and Living Room references are byte-identical between the
two lines.

## Deliberately not ported

- The branch's newer Phase-3 core (`role_semantic_ontology.py`,
  `search_contract.py`, and 6 additional P3-I test files). This tree's Phase 3
  carries the evidence-mask ablation work (`oracle_evidence.py`,
  `test_oracle_evidence.py`, `test_evidence_modes.py`, the extended
  `search_order.py`) that the branch does not have.
- `mujoco_scenes/benchmark_reports/` (151 files) and the branch's remaining
  Phase-3 documents.
- Anything under `runs/`.

## Status

Phase 4 is plumbing plus an honest self-audit, not a working result grid. Per
`docs/PHASE4_CERTIFICATION_AUDIT.md` and `docs/PHASE4_SYNC_CURRENT_HANDOFF.md`,
on the source branch: K1 was blocked upstream by an `INCOMPLETE` Phase 3, L1
completed 9 of 10 actions, W1 3 of 5, and W6's first PICK missed by 0.4292 m.
Kitchen `POUR`/`STIR` have no modelled task effect and Kitchen terminal
verification only counts actions, so Kitchen sits at certification E0.

## Open calibration questions

The kitchen storage arm speed was resolved to `0.60` (see above). One item
remains:

2. **Which `ProfiledIK` runs.** Low risk under the default environment: the
   rebound `mujoco_scenes.ik.ProfiledIK` dispatches to
   `DampedLeastSquaresIK`, the same algorithm as the branch's in-file class.
   The open question is only whether Phase-4 execution should ever run with
   `MUJOCO_IK_BACKEND=mink`, which would swap the solver under controllers
   that were never tuned for it. Leave the variable unset for Phase-4 runs
   unless you are deliberately comparing backends.

## Repository cleanup 2026-09-02

Removed byproducts (all previously gitignored, no tracked change): 20
`__pycache__` directories and 664 `.pyc` files, `.pytest_cache/`,
`txt_file.txt` (24,886 lines of PDDLStream statistics logging), `statistics/`,
`temp/`, `visualizations/`, and `.paper_deps/ccache/` — about 102 MB.

Relocated `weights/clip/ViT-B-32.pt` (338 MB, sha256 verified
`40d3657…950af`) from the repo-root Ultralytics download path to the canonical
cache location `semantic_model_cache/weights/clip/ViT-B-32.pt` declared by
`mujoco_scenes/scripts/prepare_semantic_models.py`. That directory was empty,
so six `mujoco_scenes/scripts/run_*_demo.sh` scripts were failing their own
weight-existence check. A symlink remains at the old path so Ultralytics does
not re-download.

Deleted `EXPECTED_GT_ACTIONS.zip`, verified byte-for-byte identical to
`EXPECTED_GT_ACTIONS/` (0 files differing in either direction). The directory
is now the single source of truth for private evaluator references.

Deleted `mujoco_scenes/workshop_phase1/serialization.py`. It exported
`FORBIDDEN_SIMULATOR_SUBSTRINGS`, `sanitize_production_data`,
`assert_no_backend_names`, and `write_production_json`, each with zero
references anywhere in the repository and no dynamic import. Dead code that
implements a privacy boundary is a hazard: it reads as protection while nothing
calls it. **If Workshop artifacts need backend-name sanitizing, it must be
implemented at the actual write sites, not restored as an unused module.**

Deliberately kept: `runs/` (3.8 GB — its PNGs are the real model inputs for
retained trials and are not losslessly regenerable under current annotation
behaviour), `mujoco_scenes/benchmark_reports/` (read by tests and runtime),
the FastDownward build under `.paper_deps` (430 MB of object files plus the
138 MB unstripped `bin/downward`), and `mujoco_scenes/assets/`.
