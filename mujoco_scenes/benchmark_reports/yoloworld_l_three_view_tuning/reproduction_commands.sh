#!/bin/sh

# Kitchen (returns 2 while the measured result is below 16/16).
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python -m mujoco_scenes.run_kitchen_feasibility_benchmark \
  --all-core-variants --no-robot \
  --output-root runs/yoloworld_l_three_view \
  --benchmark-id kitchen_l_frozen \
  --semantic-model yolov8l-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/semantic_vocabulary_yoloworld_l_kitchen.yaml \
  --semantic-confidence-threshold 0.01 \
  --semantic-min-supporting-views 1 \
  --semantic-minimum-mean-confidence 0.01 \
  --width 1280 --height 960

# Living Room (returns 2 while the measured result is below 13/13).
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2 \
.venv/bin/python -m mujoco_scenes.run_living_room_region_benchmark \
  --runs-root runs/yoloworld_l_three_view \
  --run-id living_room_l_frozen \
  --semantic-model yolov8l-worldv2.pt \
  --semantic-confidence-threshold 0.01 \
  --semantic-min-supporting-views 1 \
  --semantic-minimum-mean-confidence 0.01 \
  --semantic-minimum-winning-label-margin 0.015 \
  --width 1280 --height 960
