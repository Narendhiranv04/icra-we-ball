#!/bin/sh
set -eu
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2 \
.venv/bin/python -m mujoco_scenes.run_living_room_region_benchmark \
  --runs-root runs/living_room_region_phase1 \
  --run-id living_room_region_phase1_reproduction \
  --report-dir mujoco_scenes/benchmark_reports/living_room_region_feasibility_phase1 \
  --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
  --width 1280 --height 960
