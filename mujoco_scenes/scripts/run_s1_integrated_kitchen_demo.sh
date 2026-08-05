#!/bin/sh
set -eu

run_id="${1:-s1_integrated_kitchen_demo}"
image="${MUJOCO_KITCHEN_IMAGE:-mujoco-kitchen-s1}"
repository_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

cd "$repository_root"
mkdir -p runs reports

for weight in \
    semantic_model_cache/yolov8m-worldv2.pt \
    semantic_model_cache/weights/clip/ViT-B-32.pt
do
    if [ ! -r "$weight" ]; then
        echo "Missing model cache file: $weight" >&2
        echo "Run: python -m mujoco_scenes.scripts.prepare_semantic_models" >&2
        exit 1
    fi
done
if [ -e "runs/$run_id" ] || [ -e "reports/$run_id" ]; then
    echo "Run or report directory already exists for: $run_id" >&2
    echo "Choose a fresh run ID; existing evidence is never overwritten." >&2
    exit 1
fi

echo "Running integrated Scene 1 RGB-D + YOLO-World benchmark: $run_id"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    -e YOLO_CONFIG_DIR=/tmp \
    -e MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
    -e OMP_NUM_THREADS=2 \
    -e MKL_NUM_THREADS=2 \
    -e OPENBLAS_NUM_THREADS=2 \
    -e MALLOC_ARENA_MAX=2 \
    -v "$repository_root/runs:/output" \
    -v "$repository_root/semantic_model_cache/yolov8m-worldv2.pt:/models/yolov8m-worldv2.pt:ro" \
    -v "$repository_root/semantic_model_cache/weights:/workspace/weights:ro" \
    "$image" \
    --scene S1_integrated_kitchen_object_function_primary \
    --no-robot \
    --task-requirements configs/s1_integrated_kitchen_object_function.yaml \
    --inspect-sequence D1 D2 C2 B1 C1 \
    --stop-on-complete \
    --runs-root /output \
    --run-id "$run_id" \
    --point-cloud-width 1280 \
    --point-cloud-height 960 \
    --semantic-detector yolo_world \
    --semantic-model /models/yolov8m-worldv2.pt \
    --semantic-config mujoco_scenes/configs/s1_integrated_semantic_grounding.yaml \
    --semantic-vocabulary mujoco_scenes/configs/s1_integrated_semantic_vocabulary.yaml \
    --grounding-mode joint \
    --pairing-strategy semantic-role-scoped \
    --semantic-confidence-threshold 0.03 \
    --semantic-min-supporting-views 2 \
    --save-semantic-overlays

echo "Generating diagnostics, matrices, GIFs, MP4s, and HTML"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    --entrypoint python \
    -v "$repository_root/runs:/runs:ro" \
    -v "$repository_root/reports:/reports" \
    "$image" \
    -m mujoco_scenes.generate_target_assignment_report \
    "/runs/$run_id" \
    "/reports/$run_id" \
    --evaluation-config \
    mujoco_scenes/configs/s1_integrated_kitchen_object_function_evaluation.yaml

docker run --rm \
    --entrypoint python \
    -v "$repository_root/reports:/reports:ro" \
    "$image" -c \
    "import json,sys; p=json.load(open('/reports/$run_id/report_data.json')); sys.exit(0 if p['offline_evaluation']['all_expected_results_matched'] else 2)"

echo
echo "Complete."
echo "Run evidence: $repository_root/runs/$run_id"
echo "Presentation: $repository_root/reports/$run_id/presentation_report.html"
echo "Open with: xdg-open \"$repository_root/reports/$run_id/presentation_report.html\""
