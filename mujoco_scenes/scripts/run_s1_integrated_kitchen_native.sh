#!/bin/sh
set -eu

run_id="${1:-s1_integrated_kitchen_native}"
root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
python="$root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "Missing $python; follow THREE_SCENE_BENCHMARKS.md first." >&2
    exit 1
fi
if [ -e "$root/runs/$run_id" ]; then
    echo "Run already exists: runs/$run_id" >&2
    exit 1
fi

cd "$root"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/yolo-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

exec "$python" -m mujoco_scenes.scene_loader \
    --scene S1_integrated_kitchen_object_function_primary \
    --no-robot \
    --task-requirements \
      mujoco_scenes/configs/s1_integrated_kitchen_object_function.yaml \
    --inspect-sequence D1 D2 C2 B1 C1 \
    --stop-on-complete \
    --runs-root runs \
    --run-id "$run_id" \
    --point-cloud-width 1280 \
    --point-cloud-height 960 \
    --semantic-detector yolo_world \
    --semantic-model semantic_model_cache/yolov8m-worldv2.pt \
    --semantic-config \
      mujoco_scenes/configs/s1_integrated_semantic_grounding.yaml \
    --semantic-vocabulary \
      mujoco_scenes/configs/s1_integrated_semantic_vocabulary.yaml \
    --grounding-mode joint \
    --pairing-strategy semantic-role-scoped \
    --semantic-confidence-threshold 0.03 \
    --semantic-min-supporting-views 2 \
    --save-semantic-overlays
