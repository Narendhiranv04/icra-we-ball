# Claude Project Handoff

Updated: 2026-09-02 (Asia/Kolkata)

This is the current-state handoff for continuing work on this repository with
Claude or another coding agent. Read this file before changing code. Then read
the current Git status and the narrowly relevant runbook/source files. Do not
assume that every older status statement in `CODEX_HANDOFF.md` is still true.

## 1. Immediate orientation

Repository root:

```text
/home/boreddog/Documents/RRC/LH_Extension/V1
```

Remote:

```text
https://github.com/Narendhiranv04/icra-we-ball.git
```

Current checked-out branch and committed tip at handoff time:

```text
baseline_setup
adfb391 Record final baseline run cleanup
```

The worktree is intentionally very dirty. At handoff time it contained 64
modified tracked files and 82 untracked paths. The diff is roughly 10,627
insertions and 3,171 deletions before counting untracked source. These changes
include major pipeline, baseline, Workshop, GT-oracle, physical-adapter, and
discovery-replanning work. They are not disposable generated edits.

Before doing anything:

```bash
cd /home/boreddog/Documents/RRC/LH_Extension/V1
git status --short --branch
git log -5 --oneline --decorate
git diff --stat
```

Do **not** run `git reset`, `git clean`, `git checkout --`, or delete untracked
files. Do not merge a remote branch wholesale over this worktree. Preserve all
user changes and integrate narrowly.

Useful remote references currently present locally:

```text
origin/main                         ab393eb
origin/baseline_setup               adfb391
origin/naren/pipeline_check         579d2e8
origin/execution_complete_checking  85a91b9
```

The strict P3-G Workshop canonicalizer from `579d2e8` has already been ported
into the dirty worktree. Parts of the newer execution pipeline have also been
integrated. Do not repeat either port without comparing actual files and tests.

## 2. What the researcher is building

This is a MuJoCo research codebase for long-horizon task-and-motion planning
with the MuJoCo Menagerie Google Robot. The central proposed framework reasons
about **functional alternatives** rather than requiring one preselected object
or region.

Examples:

- A spoon, stirrer, or chopstick might satisfy `can_stir`.
- A mug, cup, or other suitable rigid vessel might satisfy `can_hold_liquid`.
- Different visible tables may satisfy a placement-region requirement.
- In the Workshop, a compatible manual or powered driver can drive a screw,
  while geometric checks reject the wrong fastener or driver interface.

The proposed method is intended to:

1. Give a foundation model only the goal and visible RGB observations.
2. Ask once for a functional requirement graph and ranked concrete candidate
   types, not a privileged list of objects known to exist.
3. Search regions and persist newly observed object evidence.
4. Ground roles using semantic and geometric checks.
5. Stop at the first complete feasible witness.
6. Hand the grounded roles to deterministic task/TAMP sequencing.
7. Execute and verify actions without repeatedly asking the FM to redo the
   functional reasoning.

The scientific claim is not that the FM finds a globally optimal object. It
provides commonsense functional priors. Deterministic semantic and geometric
verification decide whether an observed candidate is admissible.

Simple function names are preferred: `can_stir`, `can_clean`,
`can_hold_liquid`, and similar. Avoid subjective functions such as
`can_entertain` or embedding efficiency/preferences into predicate names.

## 3. Non-negotiable research boundaries

These constraints came directly from the user and must remain true:

- The FM/VLM receives visible observations only. Never expose hidden simulator
  inventory, hidden region contents, private variant labels, MuJoCo body names,
  expected GT actions, oracle feasibility, or final evaluator relations.
- Simulator ground truth is allowed only in explicitly named oracle, debug,
  annotation-generation, and evaluator paths.
- Image annotations may provide visible semantic aliases and region names.
  An alias-to-planning-ID map is observable input; private backend bindings are
  not.
- Baselines should receive the same goal and the same selected 1/3/5 annotated
  RGB views. Do not add YOLO-World or proposed-method semantic filtering to a
  baseline unless the original method requires it.
- Search ends at the first fully feasible method. Do not continue searching for
  a hypothetically higher-ranked hidden alternative.
- Baseline planning results, privileged GT grounding ablations, and physical
  execution results are different result classes. Never call a symbolic or
  oracle-only result end-to-end physical success.
- Keep implementation clean and compact. Do not add speculative abstractions,
  verbose generated documentation, or duplicate runners.
- The user prefers `uv`, Python 3.11, and native execution. Docker is optional
  later, not the current requirement.

## 4. The four distinct planning/execution paths

Do not conflate these architectures.

### 4.1 Proposed functional-grounding framework

Primary package:

```text
mujoco_scenes/functional_tamp_pipeline/
```

Canonical high-level flow:

```text
EXPLORE -> SATISFY -> PLAN
```

Important modules:

- `run.py`: CLI and artifact orchestration.
- `models.py`: typed functional graph, roles, relations, operation groups,
  grounding result, and pipeline result.
- `vlm_spec_provider.py`: visible-image functional graph from the remote VLM.
- `gt_spec_provider.py`: explicitly privileged controlled/debug provider.
- `predicate_registry.py`: canonical predicate signatures.
- `system_context_registry.py`: domain/context authority.
- `task_interface_validator.py`: strict graph and interface validation.
- `scene_graph.py`: accumulated observed nodes and directed relations.
- `search.py`: inspect until requirements are satisfied.
- `search_order.py`: fixed, random, provider/FM-ranked, or oracle order as an
  experiment setting.
- `grounding.py`: semantic/unary/binary evidence masks and role assignment.
- `planning.py`: invokes the common deterministic symbolic A* compiler.
- `oracle_evidence.py`: privileged evaluation-only geometry/semantics.
- `gf_reference_evaluator.py`: offline reference comparison.
- `domains/kitchen.py`, `living_room.py`, `workshop.py`: domain adapters and
  planning compilers.

CLI:

```bash
.venv/bin/python -m mujoco_scenes.functional_tamp_pipeline.run \
  --domain kitchen --variant K1 --mode gt --dry-run \
  --evidence-components semantic,unary,binary \
  --output-root runs/functional_tamp_pipeline/example
```

`--mode vlm` uses the FM specification provider. `--mode gt` is an oracle/debug
condition, not the proposed live result.

Evidence components are independently maskable:

- `semantic`: role/category compatibility.
- `unary`: object numeric or unary geometric properties.
- `binary`: directed tool-target, object-region, fit, reach, and compatibility
  relations.

Primary leave-one-out conditions:

```text
full         semantic,unary,binary
no_semantic  unary,binary
no_unary     semantic,binary
no_binary    semantic,unary
```

Current limitation: the canonical functional pipeline reaches a grounded
witness and action sequence, but its evidence-mask runs do not yet execute the
entire final physical sequence. Do not populate physical-success columns from
these runs.

### 4.2 Discovery-based replanning framework

This is an older/different Robust TAMP framework, ported from a CoppeliaSim
project into MuJoCo. It is not the proposed one-call functional framework.

Important files:

```text
mujoco_scenes/tamp/discovery_planner.py
mujoco_scenes/tamp/discovery_replanning.py
mujoco_scenes/tamp/baseline_observation_bridge.py
mujoco_scenes/run_kitchen_discovery_replanning.py
mujoco_scenes/run_living_room_discovery_replanning.py
mujoco_scenes/living_room_discovery_runtime.py
mujoco_scenes/DISCOVERY_REPLANNING.md
```

Flow:

```text
current visible segmented state + goal
  -> VLM action plan
  -> execute safe actions
  -> inspect/open region
  -> if genuinely new goal-relevant objects appear, re-observe and replan
  -> also replan after recoverable failure or observed incomplete goal
```

The old CoppeliaSim system used simulator segmentation; using MuJoCo instance
segmentation as the discovery observation boundary is acceptable for this
separate framework. Hidden object lists remain private.

Kitchen and Living Room discovery runners exist. At handoff time there were no
retained `discovery_replanning_result.json` artifacts, so no paper execution
numbers are established.

### 4.3 VLM-TAMP baseline

Package:

```text
vlm_tamp_baseline/
```

This is an algorithm-focused port to the shared MuJoCo domains. It is not an
exact reproduction of the original robot/simulator/tasks/model.

Prompt version at handoff: `9` in `vlm_tamp_baseline/prompt.py`.

One VLM-TAMP planning round contains two raw VLM requests:

1. `ENGLISH_SYSTEM_PROMPT`: goal + current visible state + images -> ordered
   English intermediate goal states.
2. `GROUNDING_SYSTEM_PROMPT`: English goals + visible aliases/IDs + formal
   predicate catalog -> grounded formal subgoals.

Then the pinned PDDLStream/Fast Downward adapter refines formal subgoals into
actions and continuous stream certificates. In physical Kitchen mode, actions
go through the common Google-Robot dispatcher and effects are verified.

Important files:

```text
vlm_tamp_baseline/planner.py
vlm_tamp_baseline/prompt.py
vlm_tamp_baseline/executive.py
vlm_tamp_baseline/pddlstream_refiner.py
vlm_tamp_baseline/failure_feedback.py
vlm_tamp_baseline/run_kitchen.py
vlm_tamp_baseline/run_living_room.py
vlm_tamp_baseline/run_workshop.py
```

The VLM is aware of `INSPECT` only through the observable action/subgoal
catalog. The neutral task goal does not instruct it to inspect everything.

Compact failure feedback deliberately contains only a typed generic failure
and failed formal subgoal. Controller telemetry and private diagnostics remain
in artifacts and are not sent to the model.

`invalid_vlm_output` means the server returned content that cannot be accepted:
missing/non-JSON content, schema violation, unknown IDs/predicates, or an
invalid completion claim. HTTP errors, connection failures, and timeouts are
separate `inference_failed` transport failures.

Physical execution status:

- Kitchen: adapter exists for K1-K12; only one real K1 VLM episode is retained.
- Living Room: planning-only baseline adapter.
- Workshop: planning-only baseline adapter.

The retained K1 physical smoke is:

```text
runs/execution_smoke/vlm_tamp/K1_20260901_165927
```

It is a valid failed trial: three executed actions, three planning rounds, six
raw VLM requests, two replans, then `NO_VALID_SUBGOALS`/model-call budget
exhaustion. It is not an infrastructure timeout.

### 4.4 OWL-TAMP baseline

Package:

```text
owl_tamp_baseline/
```

There was no public official author code available when this was implemented.
Treat it as a paper-derived best-effort reimplementation, not an official-code
reproduction. Fidelity and allowed claims are documented in
`BASELINE_FIDELITY.md`.

Prompt version at handoff: `3` in `owl_tamp_baseline/prompt.py`.

Flow:

1. Produce a discrete partial action sketch from the observable state and a
   relaxed list of grounded actions.
2. For each sketch action, query the VLM for one continuous constraint in a
   restricted helper DSL.
3. Search over a limited number of plan skeletons and continuous samples.

The permitted helper DSL contains checks such as `inside`, `supported_by`,
`collision_free`, `reachable`, `upright`, and `within_distance`.

Protocols:

- `native`: complete single-shot OWL sketch plus per-action constraint stage.
- `single_call`: sketch-only ablation; constraint generation is disabled and
  the result is explicitly incomplete as a native OWL run.
- `receding_horizon`: local paper-derived adaptation that recomputes a complete
  OWL cycle after each newly observed action state. This is not official author
  code and must be labelled accordingly.

Physical execution status:

- Kitchen: physical adapter exists, including receding-horizon dispatch.
- Living Room and Workshop: planning-only.

Important files:

```text
owl_tamp_baseline/planner.py
owl_tamp_baseline/prompt.py
owl_tamp_baseline/domain.py
owl_tamp_baseline/refinement.py
owl_tamp_baseline/receding_horizon.py
owl_tamp_baseline/run_kitchen.py
owl_tamp_baseline/run_living_room.py
owl_tamp_baseline/run_workshop.py
```

### 4.5 LLM3 baseline status

`llm3_baseline/` remains in the repository for historical comparison work. The
user decided not to use it as a primary baseline because a faithful LLM3-style
textualized-state input would require more complete annotated state than the
chosen shared visual-input protocol. Do not delete it, but do not include it in
current paper tables unless the user explicitly reopens that decision.

## 5. Shared baseline observation and action contracts

Neutral shared code lives in:

```text
baseline_common/
```

Key responsibilities:

- Typed observable entities, regions, actions, plans, and failures.
- OpenAI-compatible inference transport and typed error boundaries.
- Image encoding and camera-labelled client inputs.
- Shared action catalog.
- Common MuJoCo skill-dispatch bridge.
- Planning-to-GT batch execution and summarization.
- Physical benchmark result schema and batch runners.

The baseline RGB frames contain readable semantic aliases such as `spoon`,
`mug`, `countertop`, and `serving_area`. The textualized observation contains
the observable alias-to-internal-ID map. Duplicate visible classes use unique
aliases such as `cup_1` and `cup_2`.

This does not constitute GT-plan leakage. The VLM must know which visible
entity a formal ID denotes. Leakage would be exposing expected actions,
unobserved contents, role assignments, backend object names, or evaluator
relations.

Kitchen static and articulated regions are annotated, including:

```text
countertop, serving_area, B1, C1, C2, D1, D2
```

Annotation code was adjusted to follow masks and avoid labels occluding
objects. Old run images are not retroactively fixed; only newly generated
episodes use current annotation behavior.

The 1/3/5 camera experiment uses nested fixed subsets. Confirm the exact camera
names in the current runtime before publishing them; the intended pattern is
top only, left/right/top, then all five including front and close views.

## 6. Scenes and controlled variants

### 6.1 Google Robot

The active robot is MuJoCo Menagerie's Google Robot, not Fetch.

Expected Menagerie location:

```text
/home/boreddog/Documents/RRC/LH_Extension/third_party/mujoco_menagerie/google_robot
```

Override with `MUJOCO_MENAGERIE_PATH` if needed.

The robot has calibrated navigation/manipulation profiles, compact carry poses,
collision-aware IK, smoother arm interpolation, reduced base wobble, and
liquid-safe carry handling. Calibration guidance is in
`mujoco_scenes/ROBOT_CALIBRATION.md`.

### 6.2 Kitchen: object alternatives

Active benchmark variants: K1-K12.

- K1-K6: feasible.
- K7-K12: single-cause infeasible.
- Fixed task: prepare coffee and soup for two people, stir both coffees, and
  provide each soup bowl with a suitable utensil.
- Hidden resources may be in B1/C1/C2/D1/D2.

Authoritative configuration:

```text
mujoco_scenes/configs/kitchen_feasibility_variants.yaml
```

Main execution/runtime files include:

```text
mujoco_scenes/scene_loader.py
mujoco_scenes/baseline_kitchen_runtime.py
mujoco_scenes/kitchen_ground_truth_planner.py
mujoco_scenes/kitchen_ground_truth_execution.py
mujoco_scenes/kitchen_object_manipulation.py
mujoco_scenes/kitchen_pour_stir_manipulation.py
mujoco_scenes/kitchen_google_execution.py
mujoco_scenes/generic_manipulation.py
mujoco_scenes/ik.py
```

GT dry-run:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.run_kitchen_ground_truth_execution --variant all --dry-run
```

Physical Google-Robot actions include navigation, inspect/open, pick, place,
pour, stir, and placement/terminal verification. K1-K5 had the most upstream
physical tuning; do not assume every K6-K12 trajectory is experimentally
verified merely because symbolic preflight passes.

### 6.3 Living Room: region alternatives

Active benchmark variants: L1-L10.

- L1-L6: feasible object arrangements.
- L7-L10: infeasible because one or more required placement tables are absent.
- Fixed goal: place a cup and saucer near each seat and the remote on a surface
  accessible from both seats.
- These controlled variants have all relevant regions initially visible; no
  closed-region inspection is required in the baseline task.

Authoritative configuration:

```text
mujoco_scenes/configs/living_room_variants.yaml
```

The interactive living room also contains the L-shaped sofa, static coffee
table, rug, two media-console drawers, wall-mounted TV, rigid objects, dusting,
remote use, and low cameras for under-sofa inspection. Those interactive
features are broader than the controlled L1-L10 placement benchmark.

Important files:

```text
mujoco_scenes/living_room_scene.py
mujoco_scenes/living_room_mobile_execution.py
mujoco_scenes/living_room_navigation.py
mujoco_scenes/living_room_manipulation.py
mujoco_scenes/living_room_region_oracle.py
mujoco_scenes/living_room_symbolic_planning.py
```

GT dry-run:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  mujoco_scenes.run_living_room_execution --variant all --dry-run
```

### 6.4 Workshop: joint object and region alternatives

Active expected-action catalogue: W1-W10.

- W1-W8: feasible.
- W9: screw/hammer present but no compatible driver.
- W10: drivers present but required screw absent.

The current task is a compact single-arm frame-joint repair cell. Frame members
are fixture-held. A transparent joint cover/seal must be removed before screw
insertion. Candidate drivers and screws are distributed across storage
regions. The intended functional roles are `driver`, `fastener`, and fixed
`repair_target`.

Canonical geometric relations:

```text
COMPATIBLE_WITH
REACHES_TARGET
COMPATIBLE_WITH_TARGET
```

The exact privileged Workshop oracle calculates reach, tip/recess profile and
width, screw length/diameter, hole depth/opening, and clearances from the
instantiated MuJoCo geometry. It does not derive these relations from semantic
categories.

Important files:

```text
mujoco_scenes/workshop_scene.py
mujoco_scenes/workshop_pointcloud.py
mujoco_scenes/workshop_ground_truth_planner.py
mujoco_scenes/workshop_ground_truth_execution.py
mujoco_scenes/workshop_phase1/
mujoco_scenes/configs/workshop_variants.yaml
docs/PHASE3_P3G_WORKSHOP_CANONICALIZER.md
```

Current physical boundary is incomplete. Scene construction, observations,
region-gated capture, point clouds, strict GT geometry, symbolic planning, and
some GT execution scaffolding exist. Do not claim that the Google Robot has a
fully verified general Workshop grasp/insert/drive execution benchmark.

## 7. Ground truth and evaluator separation

Expected high-level sequences are in:

```text
EXPECTED_GT_ACTIONS/
```

They cover K1-K12, L1-L10, and W1-W10. They are private evaluation references
and must never enter a model prompt.

The evaluator may use expected actions to derive terminal relations, but model
planning and action generation cannot read those relations. Private artifacts
are conventionally written under `_private_evaluation/`.

GT execution and GT evidence ablation are separate:

- GT execution asks whether the predefined action/assignment path can be
  symbolically or physically carried out.
- `run_gt_evidence_ablation` gives every mask the same complete privileged
  scene graph and measures what happens when semantic, unary, or binary
  evidence is removed.

The latter does **not** run VLM perception, YOLO, point-cloud discovery,
PDDLStream, or physical execution. It is a controlled grounding diagnostic.

Run all seven nonempty masks:

```bash
.venv/bin/python -m mujoco_scenes.run_gt_evidence_ablation \
  --domains kitchen,living_room,workshop \
  --component-masks all \
  --output-root runs/gt_evidence_ablation/components_all_new
```

Known experimental weakness: existing Workshop variants do not yet contain a
strong wrong-interface/wrong-dimension case that changes aggregate outcomes
when binary geometry is removed. Add geometry-trap variants before claiming a
Workshop geometry-ablation improvement. The Kitchen suite also needs a clean
standalone geometry trap if semantic-only and full currently behave alike.

## 8. Benchmark protocols

Authoritative protocol document:

```text
docs/BENCHMARK_PROTOCOLS.md
```

### Native

- Proposed method: one functional VLM request, then deterministic search,
  grounding, and sequencing.
- VLM-TAMP: one two-stage round = two raw VLM requests; physical mode may
  reprompt after refinement/execution failure.
- OWL-TAMP: one discrete sketch request plus per-action constraint requests.
- Discovery: initial request plus bounded replans after discovery/failure.

### Single-call/minimum-call ablation

- Proposed/discovery: one raw VLM request.
- OWL-TAMP: one sketch request only; it is not a complete native OWL run.
- VLM-TAMP: retains its indivisible two-stage round, so it still uses two raw
  requests. Never describe that as one raw request.

Always record both `planning_rounds` and `raw_vlm_requests`.

### Planning-to-GT batch

Runner:

```text
baseline_common.run_plan_gt_batch
```

Grid intended for the paper:

- Both `vlm_tamp` and `owl_tamp`.
- Every domain variant.
- 1, 3, and 5 images.
- Seeds 0-9: ten trials per variant/camera/method.

These seeds control model sampling/reproducibility. They do not rearrange the
fixed variant geometry unless a particular runner explicitly uses a separate
scene/search seed.

Summarizer:

```text
baseline_common.summarize_plan_gt_batch
```

### Physical batch

Shared physical artifact:

```text
benchmark_execution_result.json
```

Required fields include scene, variant, method, protocol, camera count, seed,
success, executed action count, model/raw request counts, replans, planning
latency, elapsed time, and terminal failure/status.

At present, the shared baseline physical batch runner supports Kitchen. Do not
silently route Living Room or Workshop through a GT executor and call it a
baseline result.

## 9. Metrics and what they mean

Keep metric names intuitive and do not pool incompatible result types.

Planning metrics:

- **Goal completion / symbolic completion**: whether replaying the predicted
  symbolic effects satisfies every required terminal relation.
- **Required relation coverage**: correct required terminal relations divided
  by all required terminal relations. This can be partial when full completion
  fails.
- **Placement correctness**: correct predicted placement relations divided by
  all predicted placement relations. This is precision; it is optional if
  table space is limited.
- **Correct feasibility decision**: whether the method's feasible/infeasible
  conclusion matches GT.
- **Infeasibility detection**: fraction of GT-infeasible trials correctly
  rejected.
- **Ordered LCS comparison**: diagnostic action-order overlap with one expected
  sequence. Do not make it the primary success metric because multiple action
  sequences may satisfy the same goal.

Physical metrics:

- **Task success**: strict final-state evaluator passes after actual MuJoCo
  execution.
- **Physical plan found**: refinement produced an executable plan; this does
  not itself mean execution succeeded.
- **False completion**: method declares success on a GT-infeasible or
  evaluator-false state.
- **Executed actions**, **replans**, **raw VLM requests**, **planning latency**,
  and **total elapsed time** measure cost and reliability.

Grounding-ablation metrics:

- **Outcome agreement**: accepted/rejected outcome matches GT feasibility.
- **Feasible completion** and **infeasible rejection**: classwise outcome
  accuracy.
- **GT-valid selection**: among accepted cases, selected role and operation
  bindings pass the complete withheld GT semantic/geometric graph.
- **Semantic role validity**, **geometric role validity**, and **operation
  binding validity** isolate failure sources.

## 10. Existing run artifacts: what is usable

`runs/` is about 3.8 GB and contains a mixture of final, superseded, smoke,
partial, and diagnostic runs. Do not summarize the entire directory blindly.

Key locations:

```text
runs/baselines/completed/
runs/baselines/summary/table4.csv
runs/baseline_camera_ablation/workshop/
runs/benchmark/single_call_v2/
runs/gt_evidence_ablation/
runs/execution_smoke/vlm_tamp/K1_20260901_165927/
runs/scenes/workshop_pointcloud/
```

At handoff, the tree contained:

```text
3861 episode_result.json files
1922 gt_sequence_comparison.json files
1 benchmark_execution_result.json
0 discovery_replanning_result.json
```

The single physical benchmark artifact is the failed VLM-TAMP K1 smoke
described above. Therefore, the repository still does not contain a completed
physical result grid suitable for the paper.

`runs/baselines/summary/table4.csv` currently contains Living Room planning
metrics only. The retained results show VLM-TAMP and OWL-TAMP performance for
1/3/5 images, but they are symbolic planning results, not execution.

`runs/benchmark/single_call_v2/summary/table4.csv` contains all three domains.
Some Kitchen VLM-TAMP cells have fewer than 120 completed trials, and Kitchen
and Workshop goal-completion fields are not domain-final-relation replay
metrics. Treat feasibility/request/latency fields as diagnostic until the
summarizer's domain-specific replay is completed and the failed trials are
audited.

The `runs/classical_pddl/` directory is archival from an abandoned experiment.
The user explicitly decided not to use the classical-PDDL baseline. Do not
revive or report it without asking.

## 11. Current remote-model setup

The client repository normally talks to an OpenAI-compatible model server over
an SSH local tunnel. The earlier working model was `Qwen/Qwen3.5-9B`, served as
`qwen35-9b`.

Server host used recently:

```text
long-horizon@gvlab2.iiit.ac.in
```

The remote home resolves to `/home/projects/long-horizon`. Existing server
workspace and environment:

```text
~/SearchTAMP
~/SearchTAMP/.venv-qwen35
```

Client tunnel from the laptop:

```bash
ssh -L 18000:127.0.0.1:8000 long-horizon@gvlab2.iiit.ac.in
```

### Qwen3.8-27B migration diagnosis

The user wants to test official Qwen3.8-27B on one RTX 5090 32 GB GPU. The raw
BF16 checkpoint is already downloaded remotely at approximately:

```text
~/models/Qwen3.8-27B
```

It cannot fit wholly in 32 GB VRAM. Three attempted vLLM routes were ruled out:

1. BitsAndBytes: unsupported by installed vLLM
   `0.27.2rc1.dev122+g8efa13b70`.
2. Online `mxfp4`: quantization and shard loading succeeded, then FlashInfer's
   FP4 `cute-dsl` kernel rejected SM120/RTX 5090.
3. `--linear-backend flashinfer_cutlass --enforce-eager`: ineffective because
   that vLLM MXFP4 implementation hardcodes the incompatible kernel.

Do not retry online MXFP4, BitsAndBytes, `--linear-backend`, or eager-mode
variations.

Recommended next server route:

- Download `RadixArk/Qwen3.8-27B-NVFP4`, a roughly 21.9 GB ModelOpt mixed
  NVFP4/FP8 derivative of the official Qwen checkpoint.
- Keep the existing vLLM environment initially.
- Use FP8 E4M3 KV cache, one request, five-image limit, 32K context.
- Do not use speculative MTP or TurboQuant until correctness is established.
- Do not use online quantization; let vLLM detect `modelopt_mixed`.

Fish commands:

```fish
cd ~/SearchTAMP
source .venv-qwen35/bin/activate.fish

hf download RadixArk/Qwen3.8-27B-NVFP4 \
    --local-dir ~/models/Qwen3.8-27B-NVFP4

set -x CUDA_VISIBLE_DEVICES 0
set -x PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
set -x VLLM_USE_FLASHINFER_SAMPLER 0

vllm serve ~/models/Qwen3.8-27B-NVFP4 \
    --served-model-name qwen38-27b \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 32768 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":5}' \
    --enable-prefix-caching \
    --attention-backend flashinfer \
    --no-enable-flashinfer-autotune \
    --reasoning-parser qwen3 \
    --generation-config vllm
```

Acceptance gate before any Qwen3.8 batch:

1. `/v1/models` returns `qwen38-27b`.
2. Text request succeeds.
3. One-image request succeeds.
4. Five-image request succeeds.
5. Structured JSON schema and `chat_template_kwargs.enable_thinking` work.
6. Repeat the same five-image request three times without malformed or
   repetition-collapsed output.
7. Run one VLM-TAMP and one OWL-TAMP smoke before a batch.

The repository does **not** yet contain a dedicated `qwen38-27b` model profile.
Do not claim the migration is complete. Add a profile to
`inference_server/models.json`, with Qwen's official sampling settings, only
after the server passes the gate. Paper-mode baselines override decoding with
their paper settings, so server viability can be tested first with
`--model qwen38-27b`.

## 12. PDDLStream setup

VLM-TAMP uses the real pinned dependency under:

```text
.paper_deps/pddlstream
```

Pinned revision:

```text
b38137e47fd4a4116a3e36bc4be691cbe5da6cb0
```

Setup/rebuild command:

```bash
bash vlm_tamp_baseline/setup_pddlstream.sh
```

PDDLStream does not physically execute the robot. It refines formal subgoals
into an ordered action skeleton plus sampled stream certificates such as
grasps, placements, and motions. The MuJoCo dispatcher performs execution.

## 13. Validation state and honesty boundary

The most recent documented full-suite pass in the old handoff is:

```text
877 passed, 4 skipped
```

That was a historical code-contract validation, not physical experimental
evidence. The worktree has continued changing since then, and no fresh full
suite was run while writing this handoff. Claude must rerun validation before
claiming the current dirty tree is green.

Canonical validator:

```bash
./mujoco_scenes/scripts/validate_repository.sh
```

Focused experiment-contract suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python -m pytest \
  baseline_common/tests \
  vlm_tamp_baseline/tests \
  owl_tamp_baseline/tests \
  mujoco_scenes/functional_tamp_pipeline/tests \
  mujoco_scenes/tests/test_baseline_kitchen_runtime.py \
  mujoco_scenes/tests/test_living_room_execution.py \
  mujoco_scenes/tests/test_workshop_ground_truth_execution.py -q
```

Tests establish software contracts. They do not prove robust physical
execution over ten trials.

## 14. Known inconsistencies and unfinished work

These should be addressed systematically rather than with isolated patches:

1. `CODEX_HANDOFF.md` is a chronological log with stale branch names and some
   superseded readiness claims. Use this file for current status and the old
   file for history.
2. `docs/SCENE_VARIANT_CATALOGUE.md` still says Workshop is outside the branch,
   while W1-W10 configs, GT actions, baseline runners, and tests now exist.
3. `EXECUTION_AND_TESTING.md` contains old wording that says Workshop grounding
   and sequencing do not exist; newer functional-pipeline and GT scaffolding
   partially supersede it. The physical Workshop limitation remains real.
4. The functional proposed-method pipeline has not been joined to a common
   physical executor for all three domains and evidence masks.
5. Baseline physical execution is only wired for Kitchen. Living Room and
   Workshop remain planning-only for VLM-TAMP/OWL-TAMP.
6. Only one retained baseline physical trial exists, and it failed. No complete
   end-to-end baseline table is defensible yet.
7. Kitchen/Workshop planning summaries need domain-specific terminal-relation
   replay before `goal_completion_percent` is a valid primary metric.
8. Some Qwen3.8 server advice is documented here but not yet represented in
   `inference_server/models.json` or tested against the live server.
9. The worktree is far ahead of committed `baseline_setup`; committing it all
   without first grouping/reviewing changes would be risky.
10. `runs/` mixes final, superseded, partial, diagnostic, and abandoned data.
    Never delete or reorganize it without explicit user approval, but select
    inputs narrowly when summarizing.
11. Workshop needs geometry-trap variants to show a meaningful binary-geometry
    ablation gap.
12. The proposed VLM functional graph is being worked on in parallel by Naren.
    Avoid overwriting his interface or treating the GT provider as the live FM.

## 15. Recommended next steps, in order

### Immediate: finish model-server migration safely

1. Download and serve the prequantized Qwen3.8 checkpoint using the exact
   conservative configuration above.
2. Run the seven-point API acceptance gate.
3. Save server version, model revision SHA, GPU, CUDA, context, quantization,
   and sampling configuration for reproducibility.
4. Add and test the local `qwen38-27b` profile only after server validation.

### Then: validate baseline execution before large runs

1. Run one K1 VLM-TAMP physical episode and inspect every artifact.
2. Run one K1 OWL-TAMP physical episode in its declared protocol.
3. Classify failures as model output, TAMP refinement, motion/execution, or
   final-effect verification.
4. Confirm expected GT/evaluator data is absent from saved model requests.
5. Only then run ten seeds over supported variants.

### Then: complete the paper evaluation plumbing

1. Implement domain-final-relation symbolic replay for Kitchen and Workshop
   planning summaries.
2. Complete shared physical adapters for Living Room and Workshop only if the
   underlying robot skills are genuinely ready.
3. Connect the proposed functional witness/action sequence to the same physical
   artifact contract.
4. Add controlled geometry traps before interpreting evidence-mask ablations.
5. Keep planning, oracle grounding, and physical execution tables separate.

### Git hygiene before any push

1. Run the focused and full validation suites.
2. Review tracked and untracked changes by subsystem.
3. Compare against `origin/baseline_setup`, `origin/naren/pipeline_check`, and
   `origin/execution_complete_checking` without merging blindly.
4. Split commits by coherent subsystem if possible.
5. Never commit the 3.8 GB `runs/` tree, model caches, `.venv`, or generated
   reports unless the user explicitly requests and reviews them.

## 16. Useful commands

Scene smoke tests:

```bash
MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.scene_loader \
  --scene S1_coffee_missing_mug --robot google --viewer

MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.living_room_scene \
  --robot google --viewer

MUJOCO_GL=glfw .venv/bin/python -m mujoco_scenes.workshop_scene \
  --robot google --viewer
```

One VLM-TAMP Kitchen physical smoke:

```bash
MUJOCO_GL=glfw .venv/bin/python -m vlm_tamp_baseline.run_kitchen \
  --physical-variant K1 \
  --output-dir runs/physical_smoke/vlm_tamp/K1_NEW \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --camera-count 5 --protocol native
```

One OWL-TAMP Kitchen physical smoke:

```bash
MUJOCO_GL=glfw .venv/bin/python -m owl_tamp_baseline.run_kitchen \
  --variant K1 --physical-execution \
  --output-dir runs/physical_smoke/owl_tamp/K1_NEW \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --camera-count 5 --protocol receding_horizon
```

Planning batch template:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m \
  baseline_common.run_plan_gt_batch \
  --environment workshop \
  --methods vlm_tamp,owl_tamp \
  --variants W1,W2,W3,W4,W5,W6,W7,W8,W9,W10 \
  --camera-counts 1,3,5 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --protocol native \
  --output-root runs/benchmark/native/planning/workshop_NEW \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen35-9b \
  --max-tokens 8192 \
  --continue-on-error
```

Use a fresh output directory for every experiment. Do not resume into a run
created with a different prompt, model, annotation, protocol, or code revision.

## 17. Files to read for a specific task

- Overall history: `CODEX_HANDOFF.md`
- Current execution commands: `EXECUTION_AND_TESTING.md`
- Protocol/reporting boundaries: `docs/BENCHMARK_PROTOCOLS.md`
- Baseline fidelity claims: `BASELINE_FIDELITY.md`
- Batch procedure: `docs/BASELINE_BENCHMARK_RUNBOOK.md`
- GT and evidence masks: `docs/GT_EXECUTION_AND_ABLATIONS.md`
- Scene/variant intent: `THREE_SCENE_BENCHMARKS.md` and
  `docs/SCENE_VARIANT_CATALOGUE.md`
- Proposed Kitchen VLM interface:
  `docs/KITCHEN_LIVING_ROOM_VLM_REQUIREMENTS_INTEGRATION.md`
- Discovery architecture: `mujoco_scenes/DISCOVERY_REPLANNING.md`
- Robot calibration: `mujoco_scenes/ROBOT_CALIBRATION.md`
- Remote inference: `inference_server/NEW_PC_SETUP.md`
- Exact baseline prompts: `vlm_tamp_baseline/prompt.py` and
  `owl_tamp_baseline/prompt.py`
- Expected evaluation sequences: `EXPECTED_GT_ACTIONS/README.md`

## 18. Instructions for Claude

When continuing:

1. State whether the task concerns the proposed framework, discovery
   replanning, VLM-TAMP, OWL-TAMP, or GT/oracle evaluation.
2. Inspect the current source and artifacts before trusting an old status note.
3. Preserve the visibility/privacy boundary in every prompt and adapter.
4. Do not silently replace a paper algorithm with a generic planner.
5. Label adaptations and unavailable official code honestly.
6. Separate infrastructure failure from model/planning/execution failure.
7. Do not report test passes as experimental success.
8. Do not report planning-only metrics as physical task completion.
9. Use `rg` for search, `apply_patch` for edits, and focused tests before the
   full validator.
10. Update this handoff after every material architecture, experiment-protocol,
    server, or execution change.

The immediate unfinished task at the time of handoff is the Qwen3.8-27B server
migration and exact compatibility smoke testing. No local source change for
that migration has yet been made.

## 19. Update 2026-09-02: pipeline fixes and Phase-4 port

Branch: work continues on `phase4_integration`, branched from `baseline_setup`
with the dirty worktree preserved. Nothing was reset, cleaned, or discarded.

### Pipeline defects fixed

1. `baseline_common/run_baseline_execution_batch.py` raised `NameError` on
   `variants` in `main()` before dispatching any episode; `_validate` now
   returns the validated variants. The physical batch runner could never have
   run. It also derives target modules as `f"{method}_baseline.run_kitchen"`,
   which restores the neutral-layer boundary test.
2. `mujoco_scenes/run_gt_evidence_ablation.py` passed a hard-coded `"joint"`
   mode, so a default `--evidence-modes semantic_only,geometric_only,joint`
   run produced three identical full-evidence rows under three condition
   labels. Runs made with `--component-masks` were unaffected.
3. `baseline_common/summarize_plan_gt_batch.py`: a missing `gt_comparison`
   scored as a correct feasibility decision (`float(None == None)`);
   physically executed episodes were pooled into the planning table; unknown
   baseline names silently reported as VLM-TAMP.
4. `vlm_tamp_baseline/run_living_room.py` and `run_workshop.py` recorded
   `raw_vlm_requests = 2 * model_calls`; they now count transport responses as
   `run_kitchen` already did.
5. `pytest.ini` `testpaths` omitted `baseline_common/tests`,
   `owl_tamp_baseline/tests`, and `functional_tamp_pipeline/tests`, so
   `validate_repository.sh` skipped them.
6. Living-room VLM canonicalization rejected FM output that omits the
   system-fixed seating anchors it is instructed not to declare;
   `canonicalize_living_room_relation` now resolves such a self-relation onto
   the registered anchor (`NORMALIZED_SYSTEM_ANCHOR_ENDPOINT`) and
   `vlm_spec_provider._living_room` materializes `SEATING_POSITION` /
   `SEATING_PAIR` as `FIXED_TARGET` nodes, mirroring the GT provider.
7. The FM system prompt no longer spells out canonical unary property phrases
   in its `required_properties` example. The generic "Robot Verifier
   Capabilities" contract stays, so both the no-leak test and the verifier
   contract test hold.
8. `VLMTAMPExecutive` no longer spends `max_model_calls` on transport faults.
   `ModelTransportError` draws on a separate `max_transport_retries` budget and
   terminates the episode as `INFERENCE_FAILED` when exhausted.

Recorded rather than changed: OWL-TAMP receding horizon executes one action per
planning cycle, so `--max-replans` caps executed actions at `max_replans + 1`
and the default 8 cannot finish Kitchen K1 (24 expected actions). Documented in
`owl_tamp_baseline/README.md` and `BASELINE_FIDELITY.md` as budget termination,
not a capability result.

### Phase-4 execution layer ported

`origin/phase4/execution-integration-replay-contract` (`863611c`) was ported at
file level; see `docs/PHASE4_PORT_NOTES.md` for the exact file list, the two
two-way merges (`generic_manipulation.py`, `kitchen_object_manipulation.py`),
what was restored to local, and what was deliberately left behind.

Key facts:

- Phase 4 imports nothing from `functional_tamp_pipeline`; it consumes the
  persisted run directory and fails closed on the manifest, phi*, plan,
  replay evidence, and `plan_grounding_audit.json`.
- Workshop and Living Room now write `plan_grounding_audit.json`; previously
  only Kitchen did, so Phase 4 would have rejected those handoffs.
- The MuJoCo gate accepts 3.3.5 (calibrated) and 3.3.6 (this environment), and
  every execution artifact records which runtime it ran on.
- Both Workshop expected-GT vocabularies are kept: the active 28-action set in
  `EXPECTED_GT_ACTIONS/workshop/`, the branch's 6-action set as a
  reference-only copy in `EXPECTED_GT_ACTIONS/_phase4_branch_workshop/`.
- Phase 4 is plumbing plus an honest self-audit, not results: Kitchen sits at
  certification E0 with no modelled `POUR`/`STIR` effect and a terminal
  verifier that only counts actions. Do not report Phase-4 execution as task
  success without reading `docs/PHASE4_CERTIFICATION_AUDIT.md`.

Calibration decisions taken during the port: kitchen storage arm speed is the
Phase-4 branch's `0.60` (was `0.85` locally), asserted by
`mujoco_scenes/tests/test_robot_profiles.py`. The `ProfiledIK` consolidation is
behaviourally equivalent under the default environment because
`mujoco_scenes.ik.ProfiledIK` dispatches to the same damped-least-squares
solver; leave `MUJOCO_IK_BACKEND` unset for Phase-4 runs. See
`docs/PHASE4_PORT_NOTES.md`.
