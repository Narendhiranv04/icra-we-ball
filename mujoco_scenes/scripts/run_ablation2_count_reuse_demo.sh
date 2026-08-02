#!/bin/sh
set -eu

run_id="${1:-ablation2_count_reuse_demo}"
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
if [ -e "runs/$run_id" ] || [ -e "reports/$run_id" ]; then
    echo "Run or report directory already exists for: $run_id" >&2
    echo "Choose a fresh run ID." >&2
    exit 1
fi

echo "Running one full actual-perception Ablation 2 scene: $run_id"
echo "The full fixed order is retained as shared evidence for the"
echo "always-distinct exhaustion diagnostic; production first completes at D2."
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
    --scene S1_ablation2_count_reuse_primary \
    --no-robot \
    --task-requirements configs/ablation2_count_reuse.yaml \
    --inspect-sequence D1 D2 C2 B1 C1 \
    --runs-root /output \
    --run-id "$run_id" \
    --point-cloud-width 1280 \
    --point-cloud-height 960 \
    --semantic-detector yolo_world \
    --semantic-model /models/yolov8m-worldv2.pt \
    --semantic-vocabulary mujoco_scenes/configs/ablation2_semantic_vocabulary.yaml \
    --grounding-mode joint \
    --semantic-confidence-threshold 0.03 \
    --semantic-min-supporting-views 2 \
    --save-semantic-overlays

echo "Generating same-evidence policy evaluation, GIFs, MP4s, and HTML"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    --entrypoint python \
    -v "$repository_root/runs:/runs:ro" \
    -v "$repository_root/reports:/reports" \
    "$image" \
    -m mujoco_scenes.generate_usage_policy_report \
    "/runs/$run_id" \
    "/reports/$run_id"

echo
echo "Complete."
echo "Run evidence: $repository_root/runs/$run_id"
echo "Presentation: $repository_root/reports/$run_id/presentation_report.html"
echo "Open with: xdg-open \"$repository_root/reports/$run_id/presentation_report.html\""
