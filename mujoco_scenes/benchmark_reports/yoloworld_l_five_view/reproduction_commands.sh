#!/bin/sh

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2 \
.venv/bin/python -m mujoco_scenes.run_living_room_region_benchmark \
  --runs-root runs/yoloworld_l_five_view \
  --run-id living_room_l_frozen \
  --semantic-model yolov8l-worldv2.pt \
  --semantic-confidence-threshold 0.01 \
  --semantic-min-supporting-views 2 \
  --semantic-minimum-mean-confidence 0.01 \
  --semantic-minimum-winning-label-margin 0.015 \
  --width 1280 --height 960

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp \
MUJOCO_SKIP_GRAPH_MEDIA=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 \
.venv/bin/python -m mujoco_scenes.run_kitchen_feasibility_benchmark \
  --all-core-variants --no-robot \
  --output-root runs/yoloworld_l_five_view \
  --benchmark-id kitchen_l_frozen \
  --semantic-model yolov8l-worldv2.pt \
  --semantic-vocabulary mujoco_scenes/configs/semantic_vocabulary_yoloworld_l_five_view_kitchen_nojar.yaml \
  --semantic-confidence-threshold 0.01 \
  --semantic-min-supporting-views 1 \
  --semantic-minimum-mean-confidence 0.01 \
  --semantic-maximum-conflicting-view-fraction 1.0 \
  --semantic-winner-policy supporting_views_then_weighted_score \
  --width 1280 --height 960
