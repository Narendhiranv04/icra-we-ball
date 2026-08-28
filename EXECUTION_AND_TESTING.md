# Execution and testing

This is the canonical runbook for the current repository. Commands are run
from the `V1` repository root unless stated otherwise.

## What is implemented

| Stage | Current implementation |
|---|---|
| FM goal decomposition | The standalone `inference_server` accepts visible images and returns simple functional requirements plus 10--15 concrete type priors. |
| Search and grounding | Kitchen S1 and living-room L2 accumulate observed RGB-D evidence and build semantic/geometric witnesses. |
| Proposed-method sequencing | Kitchen Phase 2 compiles a `COMPLETE` observed witness into generic `PICK`, `PLACE`, `POUR`, and `STIR` actions with deterministic A* and validates the result by independent replay. |
| Kitchen execution | `GroundedPlanExecutive` dispatches generic-ID `PICK`, `PLACE`, `POUR`, and `STIR` actions through the guarded Google Robot Phase-C controller and verifies returned effects. |
| Living-room execution | The Phase-3 runner consumes saved Phase-1 evidence and an externally produced Phase-2 symbolic plan, dynamically refines base/arm poses, and verifies support relations. |
| Workshop | Scene construction and RGB-D/point-cloud capture are implemented. Workshop grounding, sequencing, and manipulation are not. |
| LLM3 comparison | The planning-only client generates observation-bounded actions. `llm3_baseline.run_kitchen` closes the loop through live cameras, physical inspection, and the shared Google Robot dispatcher. |
| VLM-TAMP comparison | Two model calls generate English goals and grounded symbolic subgoals. `vlm_tamp_baseline.run_kitchen` refines them with pinned PDDLStream, executes through the shared live runtime, and reprompts after failures. |

`llm3_baseline/` and `vlm_tamp_baseline/` are independent policy workspaces and
must not import one another. Their live runners and executive histories also
belong in their respective folders. `baseline_common/` is deliberately limited
to neutral observation/action contracts and the identical physical dispatcher
bridge used for controlled comparisons.

The production architecture never sends hidden MuJoCo inventory or backend
body names to the FM. A complete semantic/geometric witness is required before
execution. Recoverable physical failures go back to a deterministic sequencer,
not to the FM. The included command-line kitchen runner consumes one frozen
sequence and therefore uses zero replans; a future PDDLStream sequencer can use
the same `GroundedPlanExecutive` interface.

## 1. Install natively with uv

The supported development version is Python 3.11:

```bash
uv venv --python 3.11
uv pip install --torch-backend cpu --python .venv/bin/python \
  -r mujoco_scenes/requirements-dev.txt
```

Google Robot is loaded from MuJoCo Menagerie beside `V1`:

```bash
mkdir -p ../third_party
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git \
  ../third_party/mujoco_menagerie
git -C ../third_party/mujoco_menagerie sparse-checkout set google_robot
```

If Menagerie is elsewhere, export `MUJOCO_MENAGERIE_PATH` with the path to its
root directory.

Only the learned semantic benchmarks need downloaded model weights:

```bash
.venv/bin/python -m mujoco_scenes.scripts.prepare_semantic_models
```

Use `MUJOCO_GL=glfw` for a desktop window and `MUJOCO_GL=egl` plus
`PYOPENGL_PLATFORM=egl` for headless capture/tests.

## 2. Smoke-test the three scenes

Kitchen with Google Robot and the Actions panel:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot google --viewer
```

Interactive living room, including its text-file action runner:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.living_room_scene \
  --robot google --viewer
```

Edit `mujoco_scenes/configs/living_room_actions.txt`, then press **Reload and
run** in the Actions window. This is the ground-truth/manual skill path; it is
not an FM request.

Workshop scene (observation only at the current project boundary):

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot google --viewer
```

## 3. Run the functional architecture

### A. Decompose the goal once

Start the remote VLM and functional gateway by following
`inference_server/NEW_PC_SETUP.md`. With the SSH tunnel alive, send only the
camera images and goal:

```bash
python3 inference_server/functional_client.py \
  --base-url http://127.0.0.1:18000/v1 \
  --scene kitchen \
  --goal "Make coffee in a suitable container, stir it, and serve it" \
  --image left_shoulder_camera=runs/qwen_kitchen_test/left_shoulder_camera.png \
  --image right_shoulder_camera=runs/qwen_kitchen_test/right_shoulder_camera.png \
  --image overhead_camera=runs/qwen_kitchen_test/overhead_camera.png \
  --image side_camera=runs/qwen_kitchen_test/side_camera.png \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png \
  --image wrist_camera=runs/qwen_kitchen_test/wrist_camera.png \
  --image head_camera_rgb=runs/qwen_kitchen_test/head_camera_rgb.png
```

The response is a functional decomposition and ranked type prior. It does not
claim that an object exists and does not start search, geometry, sequencing,
or execution.

### B. Search and build a kitchen witness

Run the controlled S1 search with a fresh run ID:

```bash
./mujoco_scenes/scripts/run_s1_integrated_kitchen_native.sh kitchen_trial_01
```

This captures the fixed `INITIAL -> D1 -> D2 -> C2 -> B1 -> C1` observation
horizon, persists generic object evidence, and stops on a complete configured
witness. The controlled task YAML is currently the explicit bridge from the FM
functional requirements; the functional API response is not injected into it
automatically.

### C. Send the configured goal and execute it live

The primary kitchen execution task is deliberately fixed to the complete
three-person coffee-and-soup benchmark. With the functional gateway available
through the port-`18080` tunnel, run:

```bash
MUJOCO_GL=glfw .venv/bin/python -m \
  mujoco_scenes.run_kitchen_goal_execution \
  --phase1-run-dir runs/kitchen_trial_01 \
  --base-url http://127.0.0.1:18080/v1 \
  --goal "Prepare and serve coffee and soup for three people using the available kitchenware. Stir all three coffees and provide each soup bowl with a suitable utensil." \
  --camera free
```

This one command:

1. sends the typed goal and only the stage-000 camera images to the functional
   model;
2. verifies that its decomposition contains the configured functional
   requirements;
3. grounds the coffee and water sources from observed RGB crops;
4. compiles the frozen `COMPLETE` witness with the deterministic Phase-2
   symbolic planner;
5. resolves generic IDs one-to-one in a fresh Google-robot scene;
6. opens the MuJoCo viewer and executes the guarded physical plan.

The exact goal check is intentional. A shorter one-coffee goal is not silently
expanded into the configured three-person coffee-and-soup benchmark. Create a
separate task contract before executing a different goal. The runner also
stops before FM inference if `latest_witness.json` is not `COMPLETE`.

The viewer shows the current action and action count. It remains open after
completion so the final scene can be inspected; close the window to exit. Use
`--close-on-complete` for an automatic run or `--headless` to run without a
window. Artifacts and the physical trace are written under
`PHASE1_RUN_DIR/live_execution/`.

To reuse a functional response already saved by the client, add:

```bash
--decomposition path/to/functional_response.json
```

### D. Execute an externally supplied grounded plan

The execution boundary accepts generic-ID actions in LLM3-style, TAMP-style,
or canonical Phase-C JSON. For example:

```json
{
  "actions": [
    {"skill": "PICK", "arguments": {"object_id": "object_0001"}},
    {"skill": "STIR", "arguments": {"tool_id": "object_0001", "target_id": "object_0002"}}
  ]
}
```

Do not substitute guessed IDs. They must come from the frozen observed
registry and complete witness. The current runner requires the Phase-C export
bundle below; raw Phase-1 files alone are insufficient:

```text
execution_inventory.json
execution_entity_resolution.json
object_registry.json
functional_witness.json
planner_output.json
```

Once a planner/export step has produced that bundle, execute it with:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.run_kitchen_planner_execution \
  --scene S1_integrated_kitchen_object_function_primary \
  --inventory RUN/execution_inventory.json \
  --resolution RUN/execution_entity_resolution.json \
  --registry RUN/object_registry.json \
  --witness RUN/functional_witness.json \
  --plan RUN/planner_output.json \
  --goal "Make coffee in a suitable container, stir it, and serve it" \
  --viewer \
  --camera free \
  --output RUN/execution_trace.json
```

This opens the live MuJoCo viewer before the first physical action and keeps it
open after termination for inspection. Add `--close-on-complete` for automated
runs. Use `--goal-file path/to/goal.txt` instead of `--goal` for a repeatable
plain-text goal. A nonzero exit means that a guarded skill or final effect
check failed. It does not silently teleport objects or ask the FM to replan.

The functional model's exact system prompt is
`inference_server/prompts/functional_decomposition.txt`. It deliberately does
not request action sequences. The deterministic Phase-2 planner derives the
sequence after semantic/geometric search has produced a complete witness.

### E. Living-room grounding and execution

Capture the controlled F0 observation/witness:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.run_living_room_region_function \
  --scene L2_integrated_living_room_region_function_F0_BASE \
  --run-id living_f0_observation
```

After an external Phase-2 sequencer has written `plan.json` and
`symbolic_problem.json`, run physical refinement:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.run_living_room_mobile_execution \
  --variant F0_BASE \
  --phase1-dir runs/living_f0_observation \
  --phase2-dir runs/living_f0_phase2 \
  --output-dir runs/living_f0_execution \
  --execute
```

Omit `--execute` to validate/compile the supplied artifacts without running
motion. The repository does not currently generate the Phase-2 plan itself.

## 4. Run the LLM3 comparison baseline

With the raw model endpoint tunnelled to port `18000` as configured for the
baseline:

> The comparison baselines require vLLM's `/v1/chat/completions` endpoint on
> local port `18000`. Port `18080` is the separate functional-decomposition
> gateway and cannot be used for LLM3 or VLM-TAMP planning.

```bash
export LLM3_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export LLM3_PROFILE=qwen35-9b

.venv/bin/python -m llm3_baseline.client \
  --goal "Stir the contents of the visible mug" \
  --observation llm3_baseline/example_observation.json \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png
```

This prints a validated proposed plan and latency. It is a comparison output,
not evidence that actions were executed.

For live end-to-end execution from a completed Phase-1 run:

```bash
GOAL='Prepare and serve coffee and soup for three people using the available kitchenware. Stir all three coffees and provide each soup bowl with a suitable utensil.'

MUJOCO_GL=glfw .venv/bin/python -m llm3_baseline.run_kitchen \
  --phase1-run-dir runs/kitchen_live_02 \
  --output-dir runs/llm3/qwen35_trial_001 \
  --base-url http://127.0.0.1:18000/v1 \
  --goal "$GOAL" \
  --camera free
```

This captures Naren's five configured inspection-rig views, annotates them
with persistent instance/region IDs only, lets `INSPECT` physically open
closed storage, executes validated actions, re-observes successful effects,
and reprompts after inspection or recoverable failure. The textualized LLM3
state contains the same IDs and observable relations, but no semantic names.
The private benchmark goal contract and reference plan are never placed in the
VLM prompt.

## 5. Run the VLM-TAMP comparison baseline

The VLM-TAMP adaptation uses the same raw model endpoint, but the model receives
formal intermediate-goal predicates rather than primitive actions:

```bash
export VLM_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export VLM_TAMP_PROFILE=qwen35-9b

.venv/bin/python -m vlm_tamp_baseline.client \
  --goal "Stir the contents of the visible mug" \
  --observation llm3_baseline/example_observation.json \
  --image front_camera=runs/qwen_kitchen_test/front_camera.png
```

See `vlm_tamp_baseline/README.md` for the ID-only/private-world boundary,
exact prompts, metrics, and live executive contract. This command
makes both VLM calls and validates subgoals only; it does not start PDDLStream
or physical execution.

For live end-to-end subgoal refinement and execution:

```bash
bash vlm_tamp_baseline/setup_pddlstream.sh

MUJOCO_GL=glfw .venv/bin/python -m vlm_tamp_baseline.run_kitchen \
  --phase1-run-dir runs/kitchen_live_02 \
  --output-dir runs/vlm_tamp/qwen35_trial_001 \
  --base-url http://127.0.0.1:18000/v1 \
  --goal "$GOAL" \
  --camera free
```

VLM-TAMP receives its two own prompts and no primitive action catalogue. The
method-specific executors both terminate at the same shared
`MuJoCoSkillDispatcher`, making the low-level controller identical. Each live
episode must use a fresh output directory.

For both baselines, the actual model-visible evidence is auditable in the run
directory: `shared_observation_contract.json`, annotated and raw files below
`observations/`, per-frame `annotations.json`, and the `model_visible_input`
field in each model-call trace. `_private_evaluation/` contains simulator
labels used only by the dispatcher and evaluator. The five-view ID alignment
comes from MuJoCo instance segmentation and must be reported as oracle instance
tracking; neither baseline receives semantic-detector output.

## 6. Validate the repository

Run the canonical check:

```bash
./mujoco_scenes/scripts/validate_repository.sh
```

It checks patch whitespace, imports every public CLI through `--help`, and runs
the full headless test suite without writing bytecode or a pytest cache. On
systems that prohibit local sockets, four inference-gateway transport tests
skip explicitly; they run normally where loopback sockets are allowed.

For a quick execution-only regression:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  mujoco_scenes/tests/test_grounded_execution.py \
  mujoco_scenes/tests/test_physical_dispatcher.py \
  mujoco_scenes/tests/test_kitchen_planner_execution.py \
  mujoco_scenes/tests/test_kitchen_phase_c_execution.py \
  mujoco_scenes/tests/test_living_room_mobile_execution.py \
  llm3_baseline/tests/test_execution.py \
  vlm_tamp_baseline/tests
```

Generated runs, reports, model weights, virtual environments, Python caches,
and `MUJOCO_LOG.TXT` are deliberately ignored by Git. Keep experimental runs
outside commits; preserve source configs, tests, and asset manifests.

## 7. Current controlled variants and GT execution

The active Kitchen suite is `K1`–`K12` (six feasible and six infeasible);
the Living-Room suite is `L1`–`L10` (six feasible and four infeasible).
See `docs/SCENE_VARIANT_CATALOGUE.md` for the mappings.

List Kitchen variants and run symbolic preflight:

```bash
.venv/bin/python -m mujoco_scenes.run_kitchen_ground_truth_execution --list-variants
MUJOCO_GL=egl .venv/bin/python -m mujoco_scenes.run_kitchen_ground_truth_execution \
  --variant K1 --dry-run
```

List Living-Room variants and run all ten symbolic/motion-refinement dry runs:

```bash
.venv/bin/python -m mujoco_scenes.run_living_room_execution --list-variants
MUJOCO_GL=egl .venv/bin/python -m mujoco_scenes.run_living_room_execution \
  --variant all --dry-run
```

The Living-Room runner reads its compact, versioned Phase-1 production inputs
and Phase-2 symbolic outputs from `mujoco_scenes/benchmark_reports/`. Rendered
observations, videos, and diagnostic report bundles are intentionally not
stored in the repository.

Run strict physical execution with the live five-camera mosaic. `--speed`
scales arm command velocity while preserving the same symbolic plan:

```bash
MUJOCO_GL=glfw .venv/bin/python -m +  mujoco_scenes.run_kitchen_ground_truth_execution +  --variant K1 --show --strict-robot-execution --speed 1.5
```

The one-call Qwen functional-graph path currently ends after observed search,
geometric verification, and deterministic symbolic action sequencing. It does
not start physical execution:

```bash
export TAMP_FM_BASE_URL=http://127.0.0.1:18000/v1
export TAMP_FM_MODEL=qwen35-9b
MUJOCO_GL=egl .venv/bin/python -m +  mujoco_scenes.run_kitchen_vlm_pipeline --variant K1
```

Its exact contract and audit files are documented in
`docs/KITCHEN_LIVING_ROOM_VLM_REQUIREMENTS_INTEGRATION.md`.

## 8. Living Room VLM-TAMP planning/GT experiment

This experiment does not execute the Google Robot. It generates the selected
variant's five fixed-camera images, runs both VLM-TAMP model stages, refines
formal subgoals with PDDLStream, advances a symbolic PICK/PLACE rollout, and
compares that high-level sequence with the frozen GT catalogue.

```bash
export VLM_TAMP_MODEL_BASE_URL=http://127.0.0.1:18000/v1
export VLM_TAMP_PROFILE=qwen35-9b

MUJOCO_GL=egl .venv/bin/python -m vlm_tamp_baseline.run_living_room \
  --variant L1 \
  --output-dir runs/vlm_tamp_living_room/L1/qwen35_seed_001
```

Use a fresh directory for every variant and seed. Repeat L1 through L10. Read
`gt_sequence_comparison.json` for `exact_sequence_match`, ordered LCS
precision/recall/F1, and the independently reported outcome match. The run
manifest records `physical_execution: false`. Files under
`_private_evaluation/` are evaluator/adapter records and are never prompt
inputs.
