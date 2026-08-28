# VLM-TAMP baseline

This folder is an algorithm-faithful VLM-TAMP port to the repository's MuJoCo
Google-robot kitchen. It is a benchmark adaptation, not a replay of the
paper's PyBullet/PR2 experiments. The exact/adapted boundary is recorded in
[`BASELINE_FIDELITY.md`](../BASELINE_FIDELITY.md).

The default live protocol is:

```text
goal + alias-annotated images + observable alias-to-ID map
                              |
                 VLM query 1: English goals
                              |
                 VLM query 2: formal subgoals
                              |
       sequential PDDLStream refinement (3 object reducers, 12 skeletons)
                              |
           MuJoCo geometry / IK / collision-checked physical skills
                              |
                    failure-triggered reprompt
```

The model never receives the primitive action catalogue, YOLO detections, or
hidden inventory. Visible objects have unique semantic aliases such as `cup 1`;
internal IDs are not drawn on images. Both VLM stages receive the selected
views and an observable alias-to-ID map, while
PDDLStream receives the private world model required to refine a grounded
subgoal, as in the paper.

The original paper used semantically named/annotated observations, so this
adaptation exposes those names in the RGB annotations. MuJoCo instance
segmentation supplies annotation geometry and persistent ID correspondence;
this oracle annotation source is reported in the benchmark metadata.

## Install the pinned TAMP dependency

From `V1`:

```bash
bash vlm_tamp_baseline/setup_pddlstream.sh
```

This checks out PDDLStream commit
`b38137e47fd4a4116a3e36bc4be691cbe5da6cb0`, initializes its pinned
FastDownward submodule, and builds it under `.paper_deps/`. No system install
or sudo access is required.

## Start the model and run

On the inference server:

```bash
cd inference_server
./serve up qwen35-9b --detach
```

On the simulator PC, tunnel the raw endpoint and run:

```bash
ssh -L 18000:127.0.0.1:8000 user@gpu-server

cd ~/Documents/RRC/LH_Extension/V1
export VLM_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export VLM_TAMP_PROFILE=qwen35-9b

GOAL='Prepare and serve coffee and soup for three people using the available kitchenware. Stir all three coffees and provide each soup bowl with a suitable utensil.'

MUJOCO_GL=glfw .venv/bin/python -m vlm_tamp_baseline.run_kitchen \
  --phase1-run-dir runs/kitchen_live_02 \
  --output-dir runs/vlm_tamp/qwen35_trial_001 \
  --goal "$GOAL" \
  --camera free
```

The default `--decoding paper` uses temperature `0.2` with thinking disabled.
Use `--decoding model-native` only as a reported decoding ablation. The default
refiner is `pddlstream`; `--refiner catalog-ablation` retains the old fixed
template refiner solely as a named ablation.

The terminal prints English goals, formal goals, the internal PDDLStream plan,
and executable skills. Detailed model calls, stream certificates, planner logs,
observations, physical telemetry, and the final result are written below the
chosen output directory. Every episode requires an empty output directory so
traces from separate trials cannot mix.

## Living Room: planning-only GT sequence benchmark

The Living Room condition deliberately stops before physical execution. It
renders the selected L1--L10 initial state through the five fixed project
cameras, annotates only persistent object/region IDs, runs the same two-stage
VLM-TAMP proposer, and refines grounded subgoals with PDDLStream. Refined
PICK/PLACE actions advance a private symbolic rollout; they never move MuJoCo.
The resulting high-level sequence is compared with
`EXPECTED_GT_ACTIONS/living_room/<variant>/expected_gt_actions.json` using an
exact match and ordered LCS precision/recall/F1.
The default is one two-stage VLM call so this is an initial-plan comparison;
`--max-model-calls` greater than one is labeled a symbolic-reprompt ablation.

```bash
cd ~/Documents/RRC/LH_Extension/V1
export VLM_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export VLM_TAMP_PROFILE=qwen35-9b

MUJOCO_GL=egl .venv/bin/python -m vlm_tamp_baseline.run_living_room \
  --variant L1 \
  --output-dir runs/vlm_tamp_living_room/L1/qwen35_seed_001
```

Run another empty output directory for every variant/seed. The concise result
is `gt_sequence_comparison.json`; `episode_result.json` also contains the full
VLM-TAMP result. GT and semantic adapter records are saved only below
`_private_evaluation/` and are not included in either model request. For
infeasible L7--L10 trials, an explicit `NO_VALID_SUBGOALS` model result is
reported as an infeasibility prediction; a timeout or exhausted retry budget
remains `UNRESOLVED` rather than being credited as correct infeasibility.

## Kitchen: planning-only GT sequence benchmark

K1--K12 planning trials construct the requested benchmark scene directly,
assign anonymous persistent IDs using MuJoCo instance segmentation, and render
the five fixed RGB views. They do not consume a proposed-framework Phase-1
registry, YOLO output, hidden object list, or functional assignment. PDDLStream
actions advance a symbolic state without advancing MuJoCo. An `INSPECT` action
reveals that region to the symbolic state, but the single-round condition does
not query the VLM again.

```bash
GOAL='Prepare and serve coffee and soup for two people using the available kitchenware. Stir both coffees and provide each soup bowl with a suitable utensil.'

MUJOCO_GL=egl .venv/bin/python -m vlm_tamp_baseline.run_kitchen \
  --planning-only \
  --variant K1 \
  --output-dir runs/vlm_tamp/kitchen/K1/qwen35_seed_000 \
  --goal "$GOAL" \
  --seed 0
```

Planning-only mode enforces one two-stage planning round. The raw execution
vocabulary comparison is retained, while the primary shared-task comparison
normalizes `OPEN` to `INSPECT`, removes GT-only `CLOSE` cleanup, and normalizes
`PLACE_SERVING_UTENSIL` to `PLACE`. Both comparisons are saved so the
normalization is auditable.

Run both baselines over a variant/seed grid with:

```bash
.venv/bin/python -m baseline_common.run_plan_gt_batch \
  --environment kitchen \
  --methods vlm_tamp,owl_tamp \
  --variants K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12 \
  --seeds 0,1,2 \
  --output-root runs/plan_gt_batch/kitchen
```

## Planning-only client

```bash
.venv/bin/python -m vlm_tamp_baseline.client \
  --goal 'Stir the contents of the visible mug' \
  --observation llm3_baseline/example_observation.json \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png
```

The client makes both VLM stages but does not start PDDLStream or execution.
The example observation may contain human-readable labels for convenience;
both prompt stages strip them before inference.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  vlm_tamp_baseline/tests -q
```

Primary references: [VLM-TAMP paper](https://arxiv.org/abs/2410.02193),
[project page](https://vlm-tamp.github.io/vlm-tamp/), and
[official kitchen-worlds code](https://github.com/Learning-and-Intelligent-Systems/kitchen-worlds).
