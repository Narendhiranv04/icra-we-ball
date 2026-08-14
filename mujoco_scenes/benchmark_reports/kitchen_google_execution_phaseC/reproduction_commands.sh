#!/bin/sh
set -eu

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
.venv/bin/python -m pytest -q \
  mujoco_scenes/tests/test_kitchen_phase_c_execution.py \
  mujoco_scenes/tests/test_kitchen_phase_b_execution.py
.venv/bin/python -m compileall -q \
  mujoco_scenes/generic_manipulation.py \
  mujoco_scenes/kitchen_object_manipulation.py \
  mujoco_scenes/kitchen_phase_c_execution.py \
  mujoco_scenes/kitchen_pour_stir_manipulation.py \
  mujoco_scenes/run_kitchen_phase_c_freeze_evidence.py

# Individual physical evidence gates (each creates a fresh scene reset).
.venv/bin/python -m mujoco_scenes.run_kitchen_phase_c_freeze_evidence --pair-coverage POUR
.venv/bin/python -m mujoco_scenes.run_kitchen_phase_c_freeze_evidence --repeatability POUR
.venv/bin/python -m mujoco_scenes.run_kitchen_phase_c_freeze_evidence --sequential POUR
.venv/bin/python -m mujoco_scenes.run_kitchen_phase_c_freeze_evidence --pair-coverage STIR
.venv/bin/python -m mujoco_scenes.run_kitchen_phase_c_freeze_evidence --repeatability STIR
.venv/bin/python -m mujoco_scenes.run_kitchen_phase_c_freeze_evidence --sequential STIR
