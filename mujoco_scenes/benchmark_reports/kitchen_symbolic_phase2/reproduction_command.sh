#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
.venv/bin/python -m mujoco_scenes.run_phase2_symbolic_benchmark \
  --phase1-report mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1 \
  --benchmark-id "${1:-phase2_kitchen_reproduction}"
