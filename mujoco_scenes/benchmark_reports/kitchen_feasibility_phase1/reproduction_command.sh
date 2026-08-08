#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
BID="kitchen_feasibility_phase1_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl YOLO_CONFIG_DIR=/tmp MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
.venv/bin/python -m mujoco_scenes.run_kitchen_feasibility_benchmark --all-core-variants --no-robot --output-root runs/feasibility_benchmarks --benchmark-id "$BID" --width 1280 --height 960 --semantic-detector yolo_world --semantic-model semantic_model_cache/yolov8m-worldv2.pt --semantic-vocabulary mujoco_scenes/configs/semantic_vocabulary.yaml --semantic-confidence-threshold 0.03 --semantic-min-supporting-views 2 --save-semantic-overlays
