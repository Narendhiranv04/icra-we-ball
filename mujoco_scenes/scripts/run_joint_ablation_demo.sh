#!/bin/sh
set -eu

run_id="${1:-joint_stir_report_$(date +%Y%m%d_%H%M%S)}"
image="${MUJOCO_KITCHEN_IMAGE:-mujoco-kitchen-s1}"
repository_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

cd "$repository_root"
mkdir -p runs reports

if [ ! -r semantic_model_cache/yolov8m-worldv2.pt ]; then
    echo "Missing semantic_model_cache/yolov8m-worldv2.pt" >&2
    echo "Run: python -m mujoco_scenes.scripts.prepare_semantic_models" >&2
    exit 1
fi
if [ ! -r semantic_model_cache/weights/clip/ViT-B-32.pt ]; then
    echo "Missing semantic_model_cache/weights/clip/ViT-B-32.pt" >&2
    echo "Run: python -m mujoco_scenes.scripts.prepare_semantic_models" >&2
    exit 1
fi
if [ -e "runs/$run_id" ]; then
    echo "Run directory already exists: runs/$run_id" >&2
    echo "Choose a new run ID." >&2
    exit 1
fi

echo "Running actual MuJoCo scene and joint inspection: $run_id"
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
    --scene S1_joint_stir_counterexamples \
    --no-robot \
    --task-requirements configs/stir_contents_joint.yaml \
    --inspect-sequence D1 D2 C2 B1 C1 \
    --stop-on-complete \
    --runs-root /output \
    --run-id "$run_id" \
    --point-cloud-width 1280 \
    --point-cloud-height 960 \
    --semantic-detector yolo_world \
    --semantic-model /models/yolov8m-worldv2.pt \
    --semantic-vocabulary mujoco_scenes/configs/semantic_vocabulary.yaml \
    --grounding-mode joint \
    --semantic-confidence-threshold 0.03 \
    --semantic-min-supporting-views 2 \
    --save-semantic-overlays

echo "Generating same-evidence ablation report and visualizations"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    --entrypoint python \
    -v "$repository_root/runs:/runs:ro" \
    -v "$repository_root/reports:/reports" \
    "$image" \
    -m mujoco_scenes.generate_grounding_report \
    "/runs/$run_id" \
    --output-dir "/reports/$run_id"

echo
echo "Complete."
echo "Run evidence: $repository_root/runs/$run_id"
echo "Visual report: $repository_root/reports/$run_id/ablation_report.html"
echo "Open with: xdg-open \"$repository_root/reports/$run_id/ablation_report.html\""
