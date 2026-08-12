#!/usr/bin/env sh
set -eu
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
PYTHON=/home/naren/miniconda3/bin/python
MODE=${1:-quick}

focused() {
  "$PYTHON" -m pytest \
    mujoco_scenes/tests/test_kitchen_execution_entities.py \
    mujoco_scenes/tests/test_kitchen_phase_b_execution.py \
    mujoco_scenes/tests/test_manipulation_stance.py \
    mujoco_scenes/tests/test_kitchen_phase_b_closure_report.py -q
}

if [ "$MODE" = full ]; then
  for family in VESSEL BOWL UTENSIL KETTLE JAR_SOURCE; do
    "$PYTHON" -m mujoco_scenes.run_kitchen_phase_b_freeze_evidence \
      --carry-family "$family" \
      --output-dir "runs/phaseB_freeze_carried_move/$family"
  done
  "$PYTHON" -m mujoco_scenes.run_kitchen_phase_b_freeze_evidence \
    --multi-object --output-dir runs/phaseB_freeze_multi_object_authoritative
elif [ "$MODE" != quick ]; then
  echo "usage: $0 [quick|full]" >&2
  exit 2
fi

focused
if [ "$MODE" = full ]; then
  "$PYTHON" -m pytest mujoco_scenes/tests -q
fi
"$PYTHON" -m compileall -q mujoco_scenes
"$PYTHON" -m mujoco_scenes.generate_kitchen_phase_b_closure_report
