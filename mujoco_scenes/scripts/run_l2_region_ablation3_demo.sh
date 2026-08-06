#!/bin/sh
set -eu

base="${1:-l2_living_room_region_ablation3_demo_$(date +%Y%m%d_%H%M%S)}"
image="${MUJOCO_KITCHEN_IMAGE:-mujoco-kitchen-s1}"
root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$root"
mkdir -p runs reports

for weight in semantic_model_cache/yolov8m-worldv2.pt semantic_model_cache/weights/clip/ViT-B-32.pt; do
    [ -r "$weight" ] || { echo "Missing model cache: $weight" >&2; exit 1; }
done
for suffix in primary matching_trap valid permuted; do
    [ ! -e "runs/${base}_${suffix}" ] || { echo "Run exists: ${base}_${suffix}" >&2; exit 1; }
done
[ ! -e "reports/$base" ] || { echo "Report exists: reports/$base" >&2; exit 1; }

for suffix in primary matching_trap valid permuted; do
    scene="L2_living_room_region_ablation3_${suffix}"
    echo "Running actual one-observation scene: $scene"
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
        -e YOLO_CONFIG_DIR=/tmp -e MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
        -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
        -e OPENBLAS_NUM_THREADS=2 -e MALLOC_ARENA_MAX=2 \
        --entrypoint python \
        -v "$root/runs:/output" \
        -v "$root/semantic_model_cache/yolov8m-worldv2.pt:/models/yolov8m-worldv2.pt:ro" \
        -v "$root/semantic_model_cache/weights:/workspace/weights:ro" \
        "$image" -m mujoco_scenes.run_l2_region_ablation3 \
        --scene "$scene" --no-robot --runs-root /output \
        --run-id "${base}_${suffix}" --width 1280 --height 960 \
        --semantic-detector yolo_world \
        --semantic-model /models/yolov8m-worldv2.pt \
        --semantic-vocabulary mujoco_scenes/configs/l2_region_ablation3_semantic_vocabulary.yaml \
        --semantic-confidence-threshold 0.03
done

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    --entrypoint python -v "$root/runs:/runs:ro" -v "$root/reports:/reports" \
    "$image" -m mujoco_scenes.generate_region_ablation3_report \
    "/runs/${base}_primary" \
    --matching-trap "/runs/${base}_matching_trap" \
    --valid "/runs/${base}_valid" \
    --permuted "/runs/${base}_permuted" \
    --report-dir "/reports/$base"

docker run --rm --entrypoint python -v "$root:/workspace" "$image" \
    -m pytest -q mujoco_scenes/tests/test_region_ablation3.py

echo "Complete: $root/reports/$base/presentation_report.html"
