#!/usr/bin/env sh
set -eu

PYTHON=${PYTHON:-/home/naren/miniconda3/bin/python}
SCENE=S1_integrated_kitchen_object_function_primary

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PYTHON" \
  -m mujoco_scenes.run_kitchen_google_execution_phaseA \
  --scene "$SCENE" \
  --sequence container_validation \
  --execute \
  --output-dir runs/kitchen_phaseA_combined_reproduction

# Interactive visual inspection (close only after the result is saved).
# MUJOCO_GL=glfw "$PYTHON" \
#   -m mujoco_scenes.run_kitchen_google_execution_phaseA \
#   --scene "$SCENE" --sequence container_validation --execute --viewer \
#   --output-dir runs/kitchen_phaseA_combined_viewer

# Regenerate this compact report from saved evidence.
"$PYTHON" -m mujoco_scenes.generate_kitchen_phase_a_report
