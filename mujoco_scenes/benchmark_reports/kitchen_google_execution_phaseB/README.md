# Kitchen Google Robot Execution — Phase B

Phase B closes the perception-grounded `PICK → CARRY → PLACE` boundary for
the integrated kitchen. Generic observed IDs are resolved one-to-one to
execution-only MuJoCo bodies, then manipulated with source-aware workspace,
stance, grasp, extraction, live bilateral-contact, and held-state checks.

Final physical coverage includes tabletop vessel, bowl, utensil, kettle and
coffee-source families; D1 and D2 drawers; C1 and C2 cupboards; and B1 box.
The final C2 vessel and C2 utensil validations each pass three independent
fresh-reset trials. C1 bowl and C2 vessel use outward-through-aperture
extraction; B1 uses vertical-above-rim extraction.

The grasp predictor considers collision-active target geometry only and is a
ranking signal. It cannot authorize attachment. A weld is enabled only after
live left/right contact with the intended target persists. Pre-close Cartesian
pose is measured and corrected before closure. Target free-joint qpos is not
written by normal execution.

Phase C remains deliberately outside this milestone: `POUR` and `STIR` return
`UNSUPPORTED_PHASE_C_OPERATOR`, and no symbolic effects are fabricated.

## Reproduce

Run the focused and full checks:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /home/naren/miniconda3/bin/python \
  -m pytest mujoco_scenes/tests/test_kitchen_execution_entities.py \
  mujoco_scenes/tests/test_kitchen_phase_b_execution.py \
  mujoco_scenes/tests/test_manipulation_stance.py -q

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /home/naren/miniconda3/bin/python \
  -m pytest mujoco_scenes/tests -q

/home/naren/miniconda3/bin/python \
  -m mujoco_scenes.generate_kitchen_phase_b_closure_report
```

Physical run directories are intentionally kept under `runs/` rather than
committed. The JSON files here index each authoritative artifact and preserve
its exact generic ID, source provenance, selected grasp, contacts, attachment
snap, extraction, and carry result.
