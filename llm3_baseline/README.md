# LLM3 baseline

This folder ports LLM3's planning algorithm to the same MuJoCo Google-robot
kitchen used by the other methods. It preserves the paper's full-plan,
continuous-parameter, motion-feedback, resampling, and backtracking contract.
It is a domain/embodiment adaptation rather than an exact replay of the
paper's PyBullet box-packing experiments. See
[`BASELINE_FIDELITY.md`](../BASELINE_FIDELITY.md).

```text
goal + ID-only textual state + five ID-annotated images
                 + action/parameter vocabulary
                              |
       model: brief failure diagnosis + full parameterized action plan
                              |
         MuJoCo motion planning and physical execution, action by action
                              |
             successful prefix + motion failure (trace length 3)
                              |
            resample continuous values or symbolically backtrack
```

Each output action contains grounded discrete arguments and continuous values.
For this kitchen port those values are placement offsets/yaw, pour tilt/outlet
height, and stir radius/depth/cycles. They are range-validated before motion;
the physical controller actually consumes them, and its normal IK, collision,
contact, support, and effect checks remain authoritative.

The action vocabulary includes `INSPECT`, but the runtime never inspects a
region automatically. Object actions may reference only currently visible
IDs. The model does not receive hidden contents, backend body names, the
reference symbolic plan, or the private goal verifier.

The original LLM3 implementation textualizes a symbolic state and is not a
raw-image VLM benchmark. This shared-domain adaptation retains that
textualization, but exposes only persistent IDs, region open/inspected state,
observable locations, and robot state. Semantic object and region labels must
be inferred from the same five ID-annotated RGB views used by VLM-TAMP.
MuJoCo instance segmentation is used only to keep IDs aligned across views;
it is an oracle instance tracker, not a semantic detector.

## Start the model and run

On the inference server:

```bash
cd inference_server
./serve up qwen35-9b --detach
```

On the simulator PC:

```bash
ssh -L 18000:127.0.0.1:8000 user@gpu-server

cd ~/Documents/RRC/LH_Extension/V1
export LLM3_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export LLM3_PROFILE=qwen35-9b

GOAL='Prepare and serve coffee and soup for three people using the available kitchenware. Stir all three coffees and provide each soup bowl with a suitable utensil.'

MUJOCO_GL=glfw .venv/bin/python -m llm3_baseline.run_kitchen \
  --phase1-run-dir runs/kitchen_live_02 \
  --output-dir runs/llm3/qwen35_trial_001 \
  --goal "$GOAL" \
  --camera free
```

The default `--decoding paper` matches the original planner's temperature `0`
and disables thinking. `--decoding model-native` is a separate ablation. Plans
are printed as short lines including their sampled parameters. Model calls,
the three-entry plan/failure trace, physical telemetry, observation frames, and
the final result are written below the chosen output directory. Every episode
requires an empty output directory so traces from separate trials cannot mix.

## Planning-only client

```bash
.venv/bin/python -m llm3_baseline.client \
  --goal 'Stir the contents of the visible mug' \
  --observation llm3_baseline/example_observation.json \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png
```

This validates a full parameterized plan but does not execute it. The example
JSON may contain human-readable labels for convenience; the planner strips
them before constructing `textualized_state`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  llm3_baseline/tests -q
```

Primary references: [LLM3 paper](https://arxiv.org/abs/2403.11552) and
[official implementation](https://github.com/AssassinWS/LLM-TAMP).
