#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYTHON="${PYTHON:-/home/naren/miniconda3/bin/python}"
PHASE1="${PHASE1:-runs/living_room_region_phase1/living_room_region_phase1_final_closure_v3_20260809/F0_BASE}"
PHASE2="${PHASE2:-mujoco_scenes/benchmark_reports/living_room_symbolic_phase2/variants/F0_BASE}"
OUTPUT="${OUTPUT:-runs/living_room_mobile_execution_phase3/f0_full_execution}"

"$PYTHON" -m mujoco_scenes.run_living_room_mobile_execution \
  --phase1-dir "$PHASE1" \
  --phase2-dir "$PHASE2" \
  --output-dir "$OUTPUT" \
  --execute
