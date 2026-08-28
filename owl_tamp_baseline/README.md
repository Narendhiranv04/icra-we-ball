# OWL-TAMP planning baseline

This folder is an independent, paper-derived implementation of OWL-TAMP for
the Kitchen and Living Room benchmarks. It is **not** an official author code
release. It follows *Open-World Task and Motion Planning with Vision-Language
Models* ([arXiv:2411.08253v4](https://arxiv.org/abs/2411.08253)). No public
author repository was available when this adapter was written.

## Implemented protocol

1. Use the same alias-annotated RGB images, goal, and observable alias-to-ID
   map used by the visual comparison baselines.
2. Relax-ground scene actions with optimistic continuous values.
3. Ask the VLM for a partial discrete sketch and grounded goal literals.
4. Compile the sketch as an ordered subsequence with `Executed(i)` facts.
5. Ask separately for every sketched action's continuous constraint.
6. Validate expressions against a restricted geometric helper DSL.
7. Run discrete search, then at most 500 samples per action and five skeletons.
8. Compare with `EXPECTED_GT_ACTIONS` only after planning.

Expected actions, backend names, functional roles, hidden contents, and
intended feasibility never enter a prompt. This implementation does not use
YOLO, SAM, the proposed functional ranker/search, VLM-TAMP's PDDLStream
refiner, or physical execution.

## Protocol boundary

The paper's simulation protocol is single-shot and assumes relevant objects
are represented initially. Kitchen variants with closed storage are partially
observable under this benchmark's shared input contract. `OPEN` is grounded
for a visible closed region, but its contents are not revealed without a new
observation and are never leaked from MuJoCo. K2-K12 may therefore return an
observation-limited or low-overlap plan. Do not add automatic inspection or a
privileged inventory to improve the baseline. A future closed-loop condition
must be named and reported separately.

Generated constraints are syntactically restricted; the scene adapter supplies
the geometric certificate. Kitchen uses measured visible geometry and Living
Room uses observed payload/support footprints. The experiment remains
planning-only and scores plan-to-GT sequence agreement.

## Start the model server

Server (fish):

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --served-model-name qwen35-9b \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 16384
```

Simulator machine:

```bash
ssh -L 18000:127.0.0.1:8000 long-horizon@gvlab2.iiit.ac.in
export OWL_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export OWL_TAMP_PROFILE=qwen35-9b
export OWL_TAMP_TIMEOUT_SECONDS=1200
```

## Run Living Room

```bash
cd ~/Documents/RRC/LH_Extension/V1
MUJOCO_GL=egl .venv/bin/python -m owl_tamp_baseline.run_living_room \
  --variant L1 \
  --output-dir runs/owl_tamp/living_room/L1/qwen35_seed001
```

Use `L1` through `L10`. A fresh five-view observation is rendered.

## Run Kitchen

```bash
cd ~/Documents/RRC/LH_Extension/V1
MUJOCO_GL=egl .venv/bin/python -m owl_tamp_baseline.run_kitchen \
  --variant K1 \
  --output-dir runs/owl_tamp/kitchen/K1/qwen35_seed001 \
  --seed 1
```

The variant scene is constructed directly. Anonymous IDs come from MuJoCo
instance segmentation; no proposed-framework Phase-1 registry or semantic
detector output is consumed. The variant selects private expected actions only
after planning.

## Evidence

Runs save the shared observation contract and images, semantic-neutral state,
`model_trace.json` (grounding, sketch, `Executed(i)`, constraints, refinement),
and `episode_result.json` (private exact/LCS/F1 comparison). Data below
`_private_evaluation/` is unavailable to the model.
