# VLM-TAMP and OWL-TAMP camera-ablation runbook

This benchmark is planning-only. It compares each predicted grounded action
sequence with `EXPECTED_GT_ACTIONS`; it does not score MuJoCo execution.

## Experiment grid

- Methods: VLM-TAMP and OWL-TAMP.
- Kitchen: K1--K12.
- Living Room: L1--L10.
- Camera conditions: 1, 3, and 5 fixed views.
- Trials: seeds 0--9 for every variant and camera condition.

The fixed subsets are nested: 1 uses top; 3 uses left, right, and top; 5 adds
front and close. Thus the only changed input factor is camera coverage.

## Server smoke test

On gvlab2, in Fish:

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish

vllm serve Qwen/Qwen3.5-9B \
  --served-model-name qwen35-9b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":8}' \
  --enable-prefix-caching \
  --generation-config vllm \
  --reasoning-parser qwen3
```

In a separate server shell:

```fish
curl --max-time 5 http://127.0.0.1:8000/v1/models
```

Keep the server shell open. On the experiment PC, keep this tunnel open:

```bash
ssh -N -L 18000:127.0.0.1:8000 long-horizon@gvlab2.iiit.ac.in
```

In another local shell:

```bash
cd ~/Documents/RRC/LH_Extension/V1
source .venv/bin/activate

export VLM_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export VLM_TAMP_PROFILE=qwen35-9b
export VLM_TAMP_TIMEOUT_SECONDS=1200
export OWL_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export OWL_TAMP_PROFILE=qwen35-9b
export OWL_TAMP_TIMEOUT_SECONDS=1200

curl --max-time 5 http://127.0.0.1:18000/v1/models
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python -m pytest \
  vlm_tamp_baseline/tests owl_tamp_baseline/tests \
  mujoco_scenes/tests/test_baseline_kitchen_runtime.py -q
```

## One-episode model smoke tests

Use fresh output directories:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  vlm_tamp_baseline.run_kitchen \
  --planning-only --variant K1 --camera-count 1 --seed 0 \
  --output-dir runs/smoke/vlm_tamp_k1_c1_s0

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  owl_tamp_baseline.run_kitchen \
  --variant K1 --camera-count 1 --seed 0 \
  --output-dir runs/smoke/owl_tamp_k1_c1_s0

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  vlm_tamp_baseline.run_living_room \
  --variant L1 --camera-count 1 --seed 0 \
  --output-dir runs/smoke/vlm_tamp_l1_c1_s0

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  owl_tamp_baseline.run_living_room \
  --variant L1 --camera-count 1 --seed 0 \
  --output-dir runs/smoke/owl_tamp_l1_c1_s0
```

Each successful run must contain `episode_result.json`, `model_trace.json`,
and its selected annotated camera images. Confirm `camera_count`, seed,
variant, outcome comparison, and ordered F1 in `episode_result.json`.

## Full 10-trial benchmark

Run Kitchen and Living Room in two local terminals so vLLM can batch them.
The two output roots are separate.

Kitchen:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  baseline_common.run_plan_gt_batch \
  --environment kitchen \
  --methods vlm_tamp,owl_tamp \
  --camera-counts 1,3,5 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-root runs/baseline_camera_ablation/kitchen \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --max-tokens 8192 \
  --continue-on-error --resume
```

Living Room:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  baseline_common.run_plan_gt_batch \
  --environment living_room \
  --methods vlm_tamp,owl_tamp \
  --camera-counts 1,3,5 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-root runs/baseline_camera_ablation/living_room \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --max-tokens 8192 \
  --continue-on-error --resume
```

`--resume` skips completed episodes. An interrupted, nonempty episode without
`episode_result.json` is deliberately not overwritten; move that one episode
directory aside, then rerun the same command.

## Generate Table 4

```bash
.venv/bin/python -m baseline_common.summarize_plan_gt_batch \
  runs/baselines/current/kitchen \
  runs/baselines/current/living_room \
  --output-dir runs/baselines/current/summary
```

This prints the Markdown table and writes `table4.csv` and `table4.json`.
The main table uses goal-level measures rather than sequence F1:

- **Goal completion:** feasible trials where every required final relation holds.
- **Placement correctness:** correct goal-related placements divided by all
  goal-related placements proposed by the planner (precision).
- **Required placement coverage:** required final relations established by the
  plan divided by all required final relations (recall).
- **Infeasibility detection:** infeasible trials correctly rejected.

The evaluator symbolically applies planned `PLACE` effects, so these metrics do
not require physical execution. Exact sequence and LCS fields remain in each
`episode_result.json` as secondary diagnostics.
