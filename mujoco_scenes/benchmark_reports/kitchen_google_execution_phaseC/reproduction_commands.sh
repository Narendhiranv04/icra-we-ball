#!/bin/sh
set -eu

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
/home/naren/miniconda3/bin/python -m pytest mujoco_scenes/tests/test_kitchen_phase_c_execution.py -q
/home/naren/miniconda3/bin/python -m compileall -q \
  mujoco_scenes/generic_manipulation.py \
  mujoco_scenes/kitchen_object_manipulation.py \
  mujoco_scenes/kitchen_phase_c_execution.py \
  mujoco_scenes/kitchen_pour_stir_manipulation.py \
  mujoco_scenes/run_kitchen_phase_c_freeze_evidence.py
