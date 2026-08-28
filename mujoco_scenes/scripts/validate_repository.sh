#!/bin/sh
set -eu

root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
python="$root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "Missing $python; follow EXECUTION_AND_TESTING.md first." >&2
    exit 1
fi

cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

git diff --check

for module in \
    mujoco_scenes.scene_loader \
    mujoco_scenes.living_room_scene \
    mujoco_scenes.workshop_scene \
    mujoco_scenes.workshop_pointcloud \
    mujoco_scenes.run_kitchen_goal_execution \
    mujoco_scenes.run_kitchen_planner_execution \
    mujoco_scenes.run_kitchen_symbolic_pipeline \
    mujoco_scenes.run_living_room_region_function \
    mujoco_scenes.run_living_room_mobile_execution \
    llm3_baseline.client \
    llm3_baseline.run_kitchen \
    vlm_tamp_baseline.client \
    vlm_tamp_baseline.run_kitchen
do
    "$python" -m "$module" --help >/dev/null
done

"$python" -m pytest -q -p no:cacheprovider
