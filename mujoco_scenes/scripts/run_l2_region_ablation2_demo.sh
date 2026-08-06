#!/bin/sh
set -eu

run_id="${1:-l2_living_room_region_ablation2_demo}"
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
        echo "Prepare the persistent detector cache before running." >&2
        exit 1
    fi
done
if [ -e "runs/$run_id" ] || [ -e "reports/$run_id" ]; then
    echo "Run or report directory already exists for: $run_id" >&2
    echo "Use a fresh run ID; saved evidence is never overwritten." >&2
    exit 1
fi

echo "Running one actual RGB-D + YOLO-World observation: $run_id"
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
    --entrypoint python \
    -v "$repository_root/runs:/output" \
    -v "$repository_root/semantic_model_cache/yolov8m-worldv2.pt:/models/yolov8m-worldv2.pt:ro" \
    -v "$repository_root/semantic_model_cache/weights:/workspace/weights:ro" \
    "$image" \
    -m mujoco_scenes.run_l2_region_ablation2 \
    --scene L2_living_room_region_ablation2_primary \
    --no-robot \
    --runs-root /output \
    --run-id "$run_id" \
    --width 1280 \
    --height 960 \
    --semantic-detector yolo_world \
    --semantic-model /models/yolov8m-worldv2.pt \
    --semantic-vocabulary mujoco_scenes/configs/l2_region_ablation2_semantic_vocabulary.yaml \
    --semantic-confidence-threshold 0.03

echo "Generating matrices, policy graphs, GIF, MP4, and self-contained HTML"
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    --entrypoint python \
    -v "$repository_root/runs:/runs:ro" \
    -v "$repository_root/reports:/reports" \
    "$image" \
    -m mujoco_scenes.generate_region_ablation2_report \
    "/runs/$run_id" \
    --report-dir "/reports/$run_id"

docker run --rm \
    --entrypoint python \
    -v "$repository_root/runs:/runs:ro" \
    -v "$repository_root/reports:/reports:ro" \
    "$image" -c \
    "import json,sys; v=json.load(open('/runs/$run_id/region_ablation2_validation.json')); r=json.load(open('/reports/$run_id/report_data.json')); sys.exit(0 if v['passed'] and r['animation']['mp4'] else 2)"

echo
echo "Complete."
echo "Run evidence: $repository_root/runs/$run_id"
echo "Presentation: $repository_root/reports/$run_id/presentation_report.html"
echo "Open with: xdg-open \"$repository_root/reports/$run_id/presentation_report.html\""
