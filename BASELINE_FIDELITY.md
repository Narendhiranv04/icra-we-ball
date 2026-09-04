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
7. This baseline's own published decoding is temperature `0.2` with thinking
   disabled, available as `--decoding paper`. It is not the default; see
   "Decoding conditions".

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
- The Workshop W1--W10 experiment is also planning-only. It uses a separate
  symbolic PDDLStream domain for `INSPECT`, `PICK`, `PLACE`, `INSERT`, and
  `FASTEN`; no continuous stream or physical skill is claimed for this
  condition. Storage contents remain hidden from the VLM until a successful
  symbolic inspect update. The default one-call condition is initial-plan
  comparison; any higher call limit is reported as a symbolic-reprompt
  ablation.

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
6. The paper's simulation condition is single-shot; it does not inherit
   VLM-TAMP's reprompt loop. The paper's real-robot appendix separately
   describes a receding-horizon observe--plan--execute policy.

Domain adaptations and reporting restrictions:

- The paper's simulation domains are replaced by the shared MuJoCo Kitchen and
  Living Room schemas. The model receives the same five ID-only views and
  semantic-neutral state as the visual comparison protocol.
- The implementation is planning-only and supports sequence-to-GT claims, not
  physical execution-success claims. `--protocol receding_horizon` is a
  separately named symbolic observe--plan--apply-one-action condition; it is
  not a physical reproduction of the paper's real-robot deployment.
- Receding-horizon episodes apply one action per planning cycle, so
  `--max-replans` caps executed actions at `--max-replans + 1`. Episodes that
  end in `REPLAN_BUDGET_EXHAUSTED` under a budget smaller than the expected
  action count are budget-terminated and must not be reported as the method
  failing the task.
- Kitchen closed-storage contents are not provided. Hidden-object variants
  expose the single-shot baseline's partial-observability limitation; automatic
  inspection/replanning would be a separately named condition.
- Workshop storage contents follow the same rule. The VLM can choose the
  visible `INSPECT` action, but the single-shot condition does not automatically
  inspect storage or reveal its contents.
- No YOLO/SAM output, functional ranking, proposed search module, expected GT,
  hidden inventory, simulator names, or functional assignment is model input.
- The native protocol issues one constraint request per sketch action, so
  `--max-sketch-actions` (default 24) bounds that cost; a sketch that
  degenerates into repetition would otherwise bill one request per repeat.
  Truncation is recorded as `constraint_generation_complete` in
  `model_trace.json`.
- Because the authors have not released code, call this an “OWL-TAMP
  paper-derived reimplementation,” never an official code port or exact
  replication. Keep `model_trace.json` and the method manifest for audit.

For K1--K12 planning-only trials, both visual baselines construct the variant
directly and use MuJoCo instance segmentation solely to assign persistent
anonymous IDs in the five views. They do not consume the proposed framework's
Phase-1 object registry or functional witness. GT/backend ID translation occurs
only after planning. Report raw execution-vocabulary agreement and the shared
task-vocabulary normalization separately.

## Retrieval baseline correspondence

This is not a port of a published TAMP system. It is a deliberately minimal
open-vocabulary grounding baseline that isolates how far similarity-based
retrieval alone gets on the benchmark, with no language model in the loop.

1. The task structure is a fixed role template rather than a planned
   decomposition: two personal supports, one shared support, two drink
   vessels, two under-dishes, one handheld control.
2. Each role is filled by CLIP ViT-B/32 image-text similarity between the
   role's function phrase and a crop taken from the raw, unannotated frames.
3. Role phrases name functions, never category nouns, so the baseline is not
   handed the answer inside its own query.
4. Candidate supports are restricted to the runtime's registered support
   regions. The staging area is where payloads start and is not a placement
   target; electing it would let a missing table masquerade as a usable one.
5. A role that cannot be filled from distinct candidates yields
   `NO_RETRIEVED_ROLE_FILLER` and an infeasible verdict.
6. `uses_language_model` is false and every request counter is structurally
   zero, so this baseline is unaffected by the decoding conditions below and
   its numbers do not move with the served checkpoint.

Physical execution runs the retrieved assignment through the same shared
Living Room skills as the other methods, so a wrong grounding appears as a
physically executed wrong plan rather than a planning-only mismatch. Because
the physical runtime re-settles the scene before observing, its crops are not
bit-identical to the planning path's; render size is held equal across both so
the difference is scene settling only.

Call this an "open-vocabulary retrieval baseline", never a TAMP baseline.

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
   Greedy decoding is not used on this checkpoint; see "Decoding conditions".

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

## Decoding conditions

Two conditions exist and are reported separately. Every method in a table must
run the same one; the batch runners take `--decoding` and record it in
`protocol_manifest.json`.

`model-native` is the default and the main reported condition: thinking
enabled, with Qwen3.5-9B's published thinking-mode sampling for precise coding.

| Parameter | Value | Source |
|---|---|---|
| `temperature` | 0.6 | Qwen3.5-9B card, thinking / precise coding |
| `top_p` | 0.95 | same |
| `top_k` | 20 | same |
| `min_p` | 0.0 | same |
| `presence_penalty` | 0.0 | same |
| `repetition_penalty` | 1.05 | deviation, justified below |
| max output tokens | 24576 | card's planner budget |

The published figures are used unchanged except `repetition_penalty`, which is
the single documented deviation. The card's other thinking profile, for general
tasks, sets `presence_penalty` 1.5 with no repetition guard; on
schema-constrained plan output that made this checkpoint run past its stopping
point. An OWL-TAMP Living Room discrete sketch reached 18 to 64 actions for a
10-action task and emitted mutually contradictory goal literals, and the two
penalties applied together truncated the plan to 4 actions instead.
`repetition_penalty` 1.05 is the smallest value that held the sketch to its
correct length: 1.03 still degenerated to the 64-action parser cap, while 1.05
and 1.10 each returned exactly the 10 correct actions across repeated samples.
Qwen's own guidance warns this checkpoint produces endless repetitions without
a repetition guard and advises against greedy decoding.

`paper` reproduces each baseline's own published condition, temperature 0.2
with `top_p` 1.0 and thinking disabled, selected by `--decoding paper`. That
condition is near-greedy, and it is where the degeneration above was first
observed, so it is reported as an ablation rather than as the headline result.

Thinking is enabled for both visual baselines and is not a free variable. With
thinking disabled, VLM-TAMP's first Living Room subgoal list asked the
one-gripper robot to hold two objects at once and the episode failed; with
thinking enabled the same variant was solved on the first model call. A
per-method difference in thinking would confound the comparison, so the two
baselines must share the flag.

## Physical execution controls

Reported physical success depends on the shared Living Room controller and
verifier, so these are part of the result definition and not incidental tuning:

- A manipulation stance must stay collision-free across the base controller's
  own settle tolerance, not only at the commanded point. A stance validated at
  a single point can settle a centimetre into a support shell, after which
  RRT* refuses to plan from that start and the robot is stranded for the rest
  of the episode.
- `LivingRoomMobileExecutor.plan` retreats to the nearest free pose when the
  base has already settled in collision, rather than failing every later
  action.
- A placed payload is judged at rest by pose drift over a window, not by an
  instantaneous velocity sample. Resting mesh payloads chatter in the contact
  solver: a saucer alternated between 0.06 and 0.48 rad/s on consecutive steps
  while its pose moved 0.03 mm per 500 steps, so a velocity threshold rejected
  placements that were provably static. The drift bounds are tighter than the
  velocity thresholds they replace.
- The gripper's closing twist rotates a payload in the grasp; a cup measured 0
  to 10.8 degrees. That offset is recorded at pick time and cancelled in the
  place command, so a commanded placement yaw is one the controller can
  actually achieve. The 12-degree placement yaw tolerance is then enforced for
  every payload and is not relaxed for round ones.
- `benchmark_execution_result.json` records `mujoco_version`. Physical outcomes
  depend on the contact solver, so episodes from different engine builds are
  different result classes; `summarize_execution_batch` refuses to pool them.
  The repository pins `mujoco==3.3.6`.
- Every method reports a satisfied goal as terminal_status `GOAL_COMPLETE`, so
  failure-mode breakdowns keyed on that column stay comparable.
- A generation cut off by the token ceiling is reported as
  `MODEL_OUTPUT_TRUNCATED` and does not consume the episode's model-call
  budget. No plan was produced, so charging the budget would score the
  harness's ceiling as the method's planning failure: measured on a Living
  Room replan prompt, thinking-mode generation ran to the full 24576 tokens
  nine times in a row without closing its JSON, while every call that did
  complete needed under 7400. Truncation draws on its own bounded retry budget,
  in the same way a transport fault already did. These episodes are recorded,
  not dropped -- runaway reasoning on replan prompts is a real property of the
  checkpoint -- but they are a distinct failure mode and must not be pooled
  into a method's planning-failure count.

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
inspection count/order, failure codes, and wall-clock runtime. Report `paper`
and `model-native` decoding as separate conditions, never pooled; see
"Decoding conditions" above for the exact settings each one fixes.

## Required pre-run checks

```bash
cd ~/Documents/RRC/LH_Extension/V1
bash vlm_tamp_baseline/setup_pddlstream.sh

# `env -u PYTHONPATH` matters when ROS is sourced: its pytest plugin on
# /opt/ros hijacks collection and the run exits 0 having tested nothing.
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  baseline_common/tests vlm_tamp_baseline/tests owl_tamp_baseline/tests \
  retrieval_baseline/tests llm3_baseline/tests \
  mujoco_scenes/tests/test_kitchen_phase_b_execution.py \
  mujoco_scenes/tests/test_kitchen_phase_c_execution.py \
  mujoco_scenes/tests/test_kitchen_execution_entities.py \
  mujoco_scenes/tests/test_physical_dispatcher.py -q
```

Confirm the served checkpoint and the decoding actually sent, rather than the
configured defaults, before trusting a grid:

```bash
curl -s http://127.0.0.1:18000/v1/models | python3 -m json.tool
env -u PYTHONPATH .venv/bin/python -c "
from owl_tamp_baseline.planner import registry_sampling
print('thinking sampling:', registry_sampling('qwen35-9b', True))"
```

Each Living Room episode records what it sent: `method_manifest.json` for
VLM-TAMP and `model_trace.json` for OWL-TAMP. Read those, not the CLI
defaults, when reporting a condition.

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
