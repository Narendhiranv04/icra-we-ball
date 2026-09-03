# Benchmark protocols

Use two separately reported conditions. Never pool their trials.

## Native protocol

Each method keeps its documented algorithm:

- Proposed framework: one functional VLM request, then non-VLM search,
  geometric checks, and PDDLStream sequencing.
- VLM-TAMP: one indivisible two-stage planning round (two raw VLM requests) in
  the planning-to-GT benchmark.
- OWL-TAMP: one sketch request plus its native per-action constraint requests.
- Discovery replanning: one initial request plus bounded requests after genuine
  discoveries, recoverable failures, or an incomplete observed goal.

## Single-call ablation

- Proposed framework and discovery replanning receive one raw VLM request.
- OWL-TAMP receives one raw sketch request; auxiliary constraint generation is
  disabled and the artifact records that the native constraint stage is
  incomplete. Its output allowance is capped at 2,048 tokens so the large
  relaxed-grounding prompt fits the model's 16,384-token context.
- VLM-TAMP retains one minimum viable two-stage round. This necessarily costs
  two raw requests and must not be described as an equal one-request result.

Always report both `planning_rounds` and `raw_vlm_requests`.

## Planning-to-GT batches

These runs are planning-only. They compare final symbolic relations and
feasibility decisions; they do not claim physical execution success.

```bash
SEEDS=0,1,2,3,4,5,6,7,8,9

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  baseline_common.run_plan_gt_batch \
  --environment workshop \
  --methods vlm_tamp,owl_tamp \
  --variants W1,W2,W3,W4,W5,W6,W7,W8,W9,W10 \
  --camera-counts 1,3,5 \
  --seeds "$SEEDS" \
  --protocol native \
  --output-root runs/benchmark/native/planning/workshop \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --max-tokens 8192 \
  --continue-on-error
```

Replace `--protocol native` and the output root with `single_call` for the
minimum-call ablation. Use a fresh directory.

Summarize one or more planning roots with:

```bash
.venv/bin/python -m baseline_common.summarize_plan_gt_batch \
  runs/benchmark/native/planning/workshop \
  runs/benchmark/single_call/planning/workshop \
  --output-dir runs/benchmark/summary/planning
```

## Discovery execution batches

These runs physically execute MuJoCo skills. Task failures still produce valid
result files and are included as failed trials.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  baseline_common.run_discovery_execution_batch \
  --environment living_room \
  --protocol native \
  --variants L1,L2,L3,L4,L5,L6 \
  --camera-counts 1,3,5 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-root runs/benchmark/native/execution/discovery/living_room \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --max-tokens 8192 \
  --max-replans 5 \
  --continue-on-error
```

Run the same command with `--protocol single_call` and a separate output root.
The runner automatically caps the episode at one raw model request.

Summarize execution roots with:

```bash
.venv/bin/python -m baseline_common.summarize_execution_batch \
  runs/benchmark/native/execution \
  runs/benchmark/single_call/execution \
  --output-dir runs/benchmark/summary/execution
```

The proposed framework can join the execution summary by writing
`benchmark_execution_result.json` with: `scene`, `method`, `protocol`,
`camera_count`, `success`, `executed_actions`, `model_calls`,
`raw_vlm_requests`, `replans`, `planning_latency_s`, and `elapsed_seconds`.

## Kitchen baseline physical execution

VLM-TAMP and OWL-TAMP can now execute their own Kitchen actions through the
same Google-robot skill dispatcher. The private expected-action file is used
only to compile terminal evaluator relations; it is never part of the model
prompt or action construction.

Run a one-episode VLM-TAMP physical check:

```bash
MUJOCO_GL=glfw .venv/bin/python -m vlm_tamp_baseline.run_kitchen \
  --physical-variant K1 \
  --output-dir runs/physical_smoke/vlm_tamp/K1 \
  --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
  --camera-count 5 --protocol native
```

Run OWL-TAMP with its receding-horizon execution policy:

```bash
MUJOCO_GL=glfw .venv/bin/python -m owl_tamp_baseline.run_kitchen \
  --variant K1 --physical-execution \
  --output-dir runs/physical_smoke/owl_tamp/K1 \
  --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
  --camera-count 5 --protocol receding_horizon
```

For a headless batch, use a fresh output root:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  baseline_common.run_baseline_execution_batch \
  --methods vlm_tamp,owl_tamp --variants K1,K2 \
  --camera-counts 1,3,5 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --protocol native \
  --output-root runs/benchmark/native/execution/kitchen_baselines \
  --base-url http://127.0.0.1:18000/v1 --model qwen35-9b \
  --continue-on-error
```

`receding_horizon` is currently OWL-TAMP-only. Each completed episode writes
`benchmark_execution_result.json`, which `summarize_execution_batch` accepts.
Living Room and Workshop baseline runtimes remain planning-only: their
rendered/proxy observations are not yet generated from the exact live physical
executor instance, so they are intentionally excluded from this batch runner.

## Primary reporting

Planning table: **symbolic** goal completion, required relation coverage,
correct feasibility decision, raw VLM requests, and planning latency.  These
values come only from `run_plan_gt_batch` and must not be captioned as
physical or end-to-end task success.

Execution table: physical task success, executed actions, raw VLM requests,
replans, planning latency, and total elapsed time.

## Table-readiness boundary

The repository currently supports three independently valid result classes:

- **Baseline planning table:** VLM-TAMP and OWL-TAMP runners accept the common
  annotated RGB observations in Kitchen, Living Room, and Workshop. The
  retained final corpus currently contains Kitchen and Living Room episodes.
  Living Room has an explicit final-relation scorer. Kitchen and Workshop
  currently expose feasibility and ordered-action comparison, but their
  `goal_completion_percent` must not be reported until domain-specific final
  relation replay replaces the pre-execution `goal_satisfied` snapshot field.
- **Proposed-framework grounding ablation:** semantic-only, geometric-only,
  and joint evidence, scored by `run_gt_evidence_ablation`.  This is an
  offline privileged-GT grounding result, not the live VLM, learned semantic
  detector, point-cloud search, PDDLStream, or execution result. Existing
  artifacts produced before the `gt_valid_selection_pct` denominator fix must
  be discarded and regenerated.
- **Physical execution table:** Discovery replanning has a shared physical
  execution batch for Kitchen and Living Room. VLM-TAMP and OWL-TAMP now have
  a shared Kitchen-only physical path. The final proposed functional framework
  and all Living-Room/Workshop baseline paths still lack a completed common
  physical action adapter; do not populate their end-to-end rows until they
  write `benchmark_execution_result.json` and pass a final-state evaluator.

As of the 2026-08-31 critical audit, `runs/` contains no
`benchmark_execution_result.json` or `discovery_replanning_result.json`.
Therefore no physical/end-to-end row is presently supported by retained run
artifacts, even though physical adapters and their tests exist.

Before generating a paper table, run the focused suite below.  It checks the
batch/summarizer contracts, all baseline scene adapters, and functional
grounding contracts:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python -m pytest \
  baseline_common/tests vlm_tamp_baseline/tests owl_tamp_baseline/tests \
  mujoco_scenes/functional_tamp_pipeline/tests \
  mujoco_scenes/tests/test_baseline_kitchen_runtime.py \
  mujoco_scenes/tests/test_living_room_execution.py \
  mujoco_scenes/tests/test_workshop_ground_truth_execution.py -q
```
