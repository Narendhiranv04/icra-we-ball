# Baseline fidelity and reporting protocol

This document defines what may be claimed for the paper baselines. The
implementations are algorithm-faithful ports evaluated in a shared new domain;
they are not exact replications of the original papers' robot, simulator,
tasks, or model.

## Frozen upstream references

| Method | Primary paper | Official code used for the audit | Frozen revision |
|---|---|---|---|
| VLM-TAMP | arXiv:2410.02193 | `Learning-and-Intelligent-Systems/kitchen-worlds` and its PDDLStream submodule | kitchen-worlds `1839f5ff4c41f6a6b0cf5abbeb7a1292b1a551c4`; PDDLStream `b38137e47fd4a4116a3e36bc4be691cbe5da6cb0` |
| LLM3 | arXiv:2403.11552 | `AssassinWS/LLM-TAMP` | `aca6f0c1ed5f7319b48b44523e4b317a15b3861f` |
| OWL-TAMP | arXiv:2411.08253v4 | No public author code release found; independent implementation from the paper | Not applicable |

The PDDLStream source is installed outside the repository by
`vlm_tamp_baseline/setup_pddlstream.sh`; the adapter checks its Git revision at
runtime. Small Python 3.11 and logging shims are applied without modifying the
pinned source.

## VLM-TAMP correspondence

Preserved algorithmically:

1. The first VLM query produces ordered intermediate goals in English.
2. A second VLM query translates those goals to grounded formal predicates.
3. The shared-domain adaptation receives the open goal, five RGB views
   annotated with persistent instance/region IDs, a semantic-neutral
   textualized state, executed history, and the last failed subgoal. It does
   not receive simulator semantic names or hidden inventory.
4. Formal subgoals are solved sequentially by PDDLStream.
5. Each subgoal gets three TAMP attempts using goal-related, visible, then all
   manipulable objects, with at most 12 diverse skeletons per attempt.
6. Failed grounding, stream refinement, physical motion, or effect validation
   triggers a new two-stage VLM query.
7. The paper decoding condition is temperature `0.2`, thinking disabled.

Embodiment adaptations:

- PyBullet/PR2 is replaced by the shared MuJoCo/Google-robot scene and skills.
- The PDDL domain and streams represent this kitchen's `INSPECT`, `PICK`,
  `PLACE`, `POUR`, and `STIR` operations.
- RRT* base paths and placement poses are sampled before execution. Pick/task
  geometry uses measured inventory geometry; final arm IK, collision, contact,
  and POUR/STIR trajectory validation remains lazy and is performed by the
  live MuJoCo skill. A failure is returned through the paper's reprompt loop.
- Five ID-annotated camera views replace the paper's semantically annotated
  PyBullet montage. MuJoCo instance segmentation supplies persistent ID
  correspondence without class names; this is an oracle instance tracker and
  must be reported as such.
- Qwen3.5-9B may replace GPT-4o-mini. Report the model substitution explicitly.
- The Living Room sequence experiment is planning-only. PDDLStream-refined
  actions update a symbolic state for sequential refinement and GT comparison,
  but do not move MuJoCo or count as physical execution. Its results therefore
  support plan/GT agreement claims only, not execution-success claims.

The original VLM-TAMP prompt included semantically named objects and annotated
images. The shared-domain protocol deliberately removes those labels so that
both visual baselines receive the same evidence. Therefore results from this
repository are an adapted VLM-TAMP protocol, not an exact input-level
replication of the original benchmark.

The old direct subgoal-to-action templates are not the baseline. They are
available only as `--refiner catalog-ablation`.

## OWL-TAMP correspondence

Preserved algorithmically:

1. Relaxed grounding enumerates reachable discrete actions while deferring
   continuous parameters optimistically.
2. A five-image VLM query produces a partial discrete sketch and goal facts.
3. `Executed(i)` preconditions/effects constrain the symbolic solution to
   contain that sketch as an ordered subsequence.
4. Separate VLM calls translate each sketched action's physical requirement
   into a restricted geometric constraint expression.
5. Refinement uses discrete search followed by bounded continuous sampling:
   500 samples per action and at most five skeletons.
6. The condition is single-shot; it does not inherit VLM-TAMP's reprompt loop.

Domain adaptations and reporting restrictions:

- The paper's simulation domains are replaced by the shared MuJoCo Kitchen and
  Living Room schemas. The model receives the same five ID-only views and
  semantic-neutral state as the visual comparison protocol.
- The implementation is planning-only and supports sequence-to-GT claims, not
  physical execution-success claims.
- Kitchen closed-storage contents are not provided. Hidden-object variants
  expose the single-shot baseline's partial-observability limitation; automatic
  inspection/replanning would be a separately named condition.
- No YOLO/SAM output, functional ranking, proposed search module, expected GT,
  hidden inventory, simulator names, or functional assignment is model input.
- Because the authors have not released code, call this an “OWL-TAMP
  paper-derived reimplementation,” never an official code port or exact
  replication. Keep `model_trace.json` and the method manifest for audit.

For K1--K12 planning-only trials, both visual baselines construct the variant
directly and use MuJoCo instance segmentation solely to assign persistent
anonymous IDs in the five views. They do not consume the proposed framework's
Phase-1 object registry or functional witness. GT/backend ID translation occurs
only after planning. Report raw execution-vocabulary agreement and the shared
task-vocabulary normalization separately.

## LLM3 correspondence

Preserved algorithmically:

1. The model returns a full plan from the current state, not a one-action
   policy or functional decomposition.
2. Every primitive contains discrete arguments and continuous parameters.
3. Motion execution returns success/failure feedback for the attempted plan.
4. The last three plan traces are returned to the model.
5. The model may resample failed continuous parameters or symbolically
   backtrack by changing the full plan.
6. The original decoding condition is temperature `0`, thinking disabled.

Embodiment adaptations:

- The original GPT-4/PyBullet 2-D box arrangement domain is replaced by a
  frozen VLM and the common MuJoCo kitchen.
- Original `place(x,y,theta)` becomes bounded placement offsets/yaw. POUR and
  STIR add scene-relevant continuous values. These values are consumed by the
  physical motion layer; they are not discarded after prompting.
- The original LLM3 release is text-only. In this visual-domain adaptation,
  its textual state is restricted to persistent IDs and observable relations,
  while the same five ID-annotated images provide semantic evidence. Report
  this modality change explicitly.

Do not describe this as the original LLM3 benchmark. Use “LLM3 algorithm port
to the shared kitchen domain” or “LLM3-style task-and-motion baseline.”

## Fair comparison controls

For every reported method and seed, freeze:

- the exact Phase-1 scene/evidence directory and full manipulable registry;
- initial MuJoCo state, five camera poses/resolution, goal, and action budget;
- persistent object/region IDs and the oracle instance-tracking procedure;
- physical skill implementation, IK/collision thresholds, and goal verifier;
- model name/revision, server version, prompt version, decoding mode, token
  limit, and random seed;
- maximum model calls and wall-clock planning limits.

The baseline runtime resolves all 15 manipulable objects in the frozen sample
registry, not just the 11 objects selected by the proposed functional planner.
Non-manipulable markers and ungrounded distractors are excluded consistently
because no method can send them to the physical skill backend.

Report at least task success, partial goal completion, VLM calls/tokens,
subgoal or plan attempts, TAMP time, motion-planning time, executed actions,
inspection count/order, failure codes, and wall-clock runtime. Report
`paper` and `model-native` decoding as separate conditions.

## Required pre-run checks

```bash
cd ~/Documents/RRC/LH_Extension/V1
bash vlm_tamp_baseline/setup_pddlstream.sh

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  llm3_baseline/tests vlm_tamp_baseline/tests \
  mujoco_scenes/tests/test_kitchen_phase_b_execution.py \
  mujoco_scenes/tests/test_kitchen_phase_c_execution.py \
  mujoco_scenes/tests/test_kitchen_execution_entities.py \
  mujoco_scenes/tests/test_physical_dispatcher.py -q
```

Retain each run directory unchanged. `shared_observation_contract.json`, each
observation's `annotations.json`, raw and annotated frames, and each model
call's `model_visible_input` establish the input audit trail. Files below
`_private_evaluation/` contain labels used only by execution/evaluation and
must never be treated as model inputs. Parsed outputs, PDDLStream logs, action
telemetry, and the final evaluator result complete the paper audit trail.

For Living Room planning-only runs, additionally retain
`gt_sequence_comparison.json`. The expected GT file, semantic role map, and
MuJoCo backend resolution are evaluation/adapter-private. The VLM receives the
open-language goal, five ID-only annotated RGB views, semantic-neutral state,
history, and failure feedback—never the expected sequence or functional
assignment.
