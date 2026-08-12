#!/usr/bin/env sh
set -eu
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
PYTHON=/home/naren/miniconda3/bin/python
$PYTHON -m pytest mujoco_scenes/tests/test_kitchen_execution_entities.py \
  mujoco_scenes/tests/test_kitchen_phase_b_execution.py \
  mujoco_scenes/tests/test_manipulation_stance.py -q
$PYTHON -m pytest mujoco_scenes/tests -q
$PYTHON -m compileall -q mujoco_scenes
$PYTHON -m mujoco_scenes.generate_kitchen_phase_b_closure_report
