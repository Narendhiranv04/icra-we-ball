# Corrective Recovery Plan — Functional-TAMP VLM Interface, Capability Bridge, Search, and Evaluation

> **Status:** IMPLEMENTATION-READY / EXECUTE END-TO-END
> **Repository:** `Narendhiranv04/icra-we-ball`
> **Development branch:** `vlm-testing-pipeline`
> **Current branch HEAD at plan creation:** `af2dde7341adff911c8a6c2d6b7870797ba87ed3`
> **Purpose:** Recover real end-to-end performance without reintroducing GT leakage, endpoint-only semantic synthesis, extra semantic VLM calls, high-level replanning, or weakened verification.
> **Primary agent target:** fresh Antigravity Flash 3.8 High Reasoning.
> **Do not treat the current 0% final result as a scientifically frozen method.** Preserve it as an immutable historical artifact, then continue corrective development from the current branch.

---

# 0. EXECUTION MODE FOR THE AGENT

You are not being asked to write another plan. You are being asked to **execute this plan completely**.

Work stage-by-stage until all gates pass and the final development and held-out evaluations are complete.

Do **not** stop after:
- writing code,
- passing unit tests,
- fixing one domain,
- generating a small probe,
- obtaining a single successful variant,
- or documenting remaining issues.

Continue to the next stage automatically.

Only stop early for a genuine hard blocker that cannot be resolved from the repository, machine, logs, or existing credentials/configuration, such as:
- the remote GPU host is unreachable for a sustained period,
- the required model weights are unavailable and cannot be recovered from the existing server/cache/config,
- authentication genuinely requires a secret that is not present,
- a destructive repository conflict requires user choice.

When stopping for a hard blocker, report:
1. exact blocker,
2. exact commands already tried,
3. exact state of local and remote processes,
4. last pushed commit,
5. exact stage to resume from.

Do not ask the user to make routine implementation decisions. Use the scientific rules in this file.

---

# 1. NON-NEGOTIABLE SCIENTIFIC CONTRACT

The system architecture MUST remain:

```text
Task instruction + initial RGB/RGB-D
              |
              v
        ONE semantic FM/VLM call
              |
              v
      open-ended task semantics
    roles + relations + operations
              |
              v
  deterministic semantic compiler
              |
              v
              G_F
              |
     current observations
              |
              v
              G_O
              |
    if physical evidence/candidates
    are incomplete and search can help:
          inspect/search
              |
              v
          G_O grows
              |
              v
 semantic + geometric grounding phi*
              |
              v
          ONE A*
              |
              v
 independently validated action sequence
```

Required invariants:

| Invariant | Required |
|---|---|
| Semantic VLM calls per benchmark variant | **Exactly 1** |
| High-level replans | **0** |
| A* invocations | **At most 1 after grounding** |
| Physical execution | **Not required** |
| GT/reference graph online | **Forbidden** |
| Hidden object locations online | **Forbidden** |
| Expected assignment online | **Forbidden** |
| Expected plan online | **Forbidden** |
| Variant-specific logic | **Forbidden** |
| VLM-facing closed predicate enum | **Forbidden** |
| Endpoint-only task relation synthesis | **Forbidden** |
| Endpoint-only task operation synthesis | **Forbidden** |
| Physical verifier weakening | **Forbidden** |
| UNKNOWN physical evidence treated as TRUE | **Forbidden** |
| Per-variant prompt repair / second semantic call | **Forbidden** |

Controlling semantic-boundary rule:

> Neither endpoint identity, canonical role identity, environment ontology, fixed anchors, robot capability registries, nor planner action models may create a task participant, task relation, or task operation that was not semantically expressed by the FM. They may canonicalize an expressed semantic, provide task-explicit fixed context, map an explicitly expressed operation to a robot capability, and attach physical preconditions required by that explicitly expressed capability.

Important consequence:

> **Physical preconditions of an explicitly expressed robot operation are NOT hidden task-semantic synthesis.**

Example:

```text
FM explicitly expresses:
    "stir the beverage in the serving container"

This may map to:
    capability STIR

The capability may legitimately require:
    INSERTABLE_IN(tool, container)
    REACHES_BOTTOM(tool, container)
```

The FM does **not** need to independently rediscover every low-level geometric predicate that the robot already knows is required to execute an explicitly requested operation.

---

# 2. AUTHORITATIVE CURRENT FAILURE SNAPSHOT

The current pushed result at `af2dde73` reports approximately:

```text
Development 32x1:
    feasible success            0.0%
    goal coverage               0.0%
    recovery                    0.0%
    false completion            0.0%
    raw role F1                ~60.2%
    interpreter raw relation F1 ~0.8%
    executable contract          0.0%
    search eligible              0.0%
    A* invoked                  ~6.2%

Held-out 15x1:
    feasible success             0.0%
    goal coverage                0.0%
    executable contract          0.0%
```

Historical pre-corrective baseline:

```text
Outcome correct      6.25%
Feasible success    10.00%  (2/20)
Recovery            15.38%  (2/13)
Goal coverage       20.00%
False completion     0.00%
VLM requests         1.00
Replans              0.00
```

The corrective pass improved safety/strictness but collapsed the useful semantic interface.

Do not restore hidden shortcuts merely to recover these numbers.

The goal is to build the legitimate replacement interface correctly.

---

# 3. VERIFIED ROOT CAUSES TO CORRECT

Treat the following as the starting forensic hypotheses and verify each directly in code and traces before changing it.

## 3.1 Workshop V2 capability preconditions are skipped

In the current `semantic_compiler.py`, the Workshop singleton path only materializes capability physical preconditions inside:

```python
if not is_v2:
    ...
```

then exits the group path.

This means an explicitly expressed V2 fastening operation can map to a robot capability but fail to acquire legitimate physical prerequisites such as:

```text
COMPATIBLE_WITH
REACHES_TARGET
COMPATIBLE_WITH_TARGET
```

Fix the architecture, not the benchmark case.

## 3.2 Online executable contract completeness contains hidden expected-task checklists

Current online completeness logic contains domain-specific requirements such as:

```text
Kitchen:
    must contain STIR_COFFEE
    must contain soup utensil serving op

Living:
    must contain drinkware support
    must contain remote support
    must contain near-seat relation

Workshop:
    must contain driver
    must contain fastener
    must contain repair_target
    must contain exact three physical predicates
    must contain fastening operation
```

That is the wrong online/offline split.

Online runtime may validate **everything the FM expressed**.

Offline evaluation may determine whether the FM omitted something required by the reference task.

The online runtime must not know the hidden expected answer graph.

## 3.3 Search is globally starved by the contract gate

Current output shows:

```text
executable contract complete = 0%
search eligible = 0%
```

The system therefore never exercises its intended partial-observability recovery path.

Search must be blocked only when an **expressed required semantic** is malformed/uninterpretable in a way observation cannot repair.

Search must remain possible when:
- expressed roles are groundable in principle but no candidate is currently visible,
- current candidates are FALSE,
- current candidates are UNKNOWN,
- current joint assignment fails,
- hidden alternative candidates may exist,
- inspectable regions remain.

## 3.4 Capability registry is incomplete relative to planner functionality

Kitchen planner can physically realize material transfer/POUR, but the operation capability layer does not robustly represent the corresponding explicit task-level operation.

This produces the contradiction:

```text
FM explicitly says "transfer coffee/water into serving vessel"
    -> compiler: UNSUPPORTED_OPERATOR

but

planner sees source/container roles
    -> planner may generate POUR anyway
```

Both sides are wrong.

The task operation must come from the FM.
The robot capability layer must own how that operation is realized.

## 3.5 Real Qwen paraphrases are rejected by role canonicalization

Representative current outputs include sensible semantics such as:

```text
"Holds soup ready for consumption."
"Tool used to mix ingredients inside the coffee serving vessel."
"An implement used to manipulate or install the fastening component."
"A physical item capable of connecting or securing elements at the marked location."
```

These have been classified as unresolved or ambiguous by production mappings.

A compiler failure to consume clearly expressed semantics must be reported as a compiler/interface failure, not an FM omission.

## 3.6 Raw semantic evaluation is too narrow for the scientific conclusion currently drawn

The raw evaluator uses frozen regex-style matchers for a small set of predicates/operations.

Do not equate:

```text
raw relation F1 ~= 0
```

with:

```text
the FM expressed no meaningful task relation
```

until the evaluator itself has been audited for semantic coverage and independence.

## 3.7 Current inference configuration has structured-output reliability problems

At least one final case took several minutes and ended in malformed JSON despite structured output.

The current corrective freeze selected a high-output thinking configuration despite earlier evidence that thinking-off was much more stable.

Re-evaluate inference configuration before another full matrix.

---

# 4. REMOTE GPU / VLM INFRASTRUCTURE

All live VLM inference for this corrective pass must use the remote GPU host reachable through:

```bash
ssh -i ~/keyfile -p 23808 long-horizon@0.tcp.in.ngrok.io
```

Do not expose vLLM publicly.

## 4.1 Important ngrok rule

The ngrok TCP process providing:

```text
0.tcp.in.ngrok.io:23808 -> remote SSH :22
```

must remain alive.

Do not kill, restart, reconfigure, or replace the running ngrok process during this work unless the tunnel is genuinely dead and there is no alternative.

If ngrok is running in a remote terminal/tmux, leave it alone.

Before beginning development, verify from the local machine:

```bash
nc -vz -w 10 0.tcp.in.ngrok.io 23808
```

and:

```bash
ssh -i ~/keyfile -p 23808 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  long-horizon@0.tcp.in.ngrok.io \
  'hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv'
```

## 4.2 Remote GPU server inspection

Connect:

```bash
ssh -i ~/keyfile -p 23808 long-horizon@0.tcp.in.ngrok.io
```

Inspect non-destructively:

```bash
hostname
uname -a
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
nvidia-smi topo -m || true
free -h
df -h
python3 --version
which uv || true
which python || true
which git || true
which tmux || true
ps aux | grep -E '[v]llm|[q]wen|[n]grok'
ss -ltnp | grep -E ':22|:8000|:18000' || true
```

If a correct Qwen server is already running, reuse it.
Do not kill it simply to recreate the environment.

## 4.3 Exact model

The expected model is:

```text
Qwen/Qwen3.5-9B
```

historically served as:

```text
qwen35-9b
```

Do not silently substitute a sibling model.

Verify the exact loaded model using:
- `/v1/models`,
- existing vLLM command line,
- model cache,
- server logs,
- previous run provenance.

If the exact model cannot be verified, stop before consuming benchmark calls and report the blocker.

## 4.4 Keep vLLM alive in tmux

If no correct vLLM instance is running, start it in a persistent tmux session such as:

```bash
tmux new -d -s qwen-vllm \
  "bash -lc '<EXACT VERIFIED VLLM START COMMAND> 2>&1 | tee -a ~/qwen-vllm.log'"
```

Do not invent the start command before inspecting the existing environment/config.

Bind the inference server to:

```text
127.0.0.1
```

on the remote host.

Do not bind the vLLM API directly to the public internet.

Verify:

```bash
curl -sS http://127.0.0.1:8000/v1/models | python -m json.tool
```

or include the existing API key if the server requires one.

## 4.5 Persistent local SSH API tunnel

From the laptop, create a persistent local tunnel:

```bash
mkdir -p ~/.cache/tamp-vlm

ssh \
  -i ~/keyfile \
  -p 23808 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  -o TCPKeepAlive=yes \
  -M \
  -S ~/.cache/tamp-vlm/qwen.sock \
  -fN \
  -L 18000:127.0.0.1:8000 \
  long-horizon@0.tcp.in.ngrok.io
```

Verify:

```bash
ss -ltnp | grep ':18000'
curl -sS http://127.0.0.1:18000/v1/models | python -m json.tool
```

If port `18000` is occupied, select another local port and record it in the corrective progress log.

Check the SSH master tunnel:

```bash
ssh \
  -i ~/keyfile \
  -p 23808 \
  -S ~/.cache/tamp-vlm/qwen.sock \
  -O check \
  long-horizon@0.tcp.in.ngrok.io
```

Do not tear this tunnel down until all live evaluation is complete.

At the very end only:

```bash
ssh \
  -i ~/keyfile \
  -p 23808 \
  -S ~/.cache/tamp-vlm/qwen.sock \
  -O exit \
  long-horizon@0.tcp.in.ngrok.io
```

## 4.6 Local inference environment

Use local client variables equivalent to:

```bash
export TAMP_FM_BASE_URL='http://127.0.0.1:18000/v1'
export TAMP_FM_MODEL='qwen35-9b'
```

Recover any existing API key from the user's current non-committed environment/config.
Never print, log, or commit secrets.

Before benchmark probes, perform one non-benchmark smoke call using a generic unrelated image/instruction.

Smoke tests do not count as benchmark VLM calls.

Record only non-secret infrastructure provenance:
- remote hostname,
- GPU model,
- model ID,
- served alias,
- vLLM version,
- remote vLLM port,
- local tunnel port,
- tensor parallel size,
- max model length,
- decoding configuration.

---

# 5. GIT / REPOSITORY RULES

Start from:

```bash
cd /home/naren/RA_iiith
git fetch origin
git checkout vlm-testing-pipeline
git pull --ff-only origin vlm-testing-pipeline
git rev-parse HEAD
git status --short
```

Expected starting HEAD:

```text
af2dde7341adff911c8a6c2d6b7870797ba87ed3
```

If HEAD has advanced, inspect the new commits first and adapt this plan without discarding user work.

Never:
- `git reset --hard`,
- force push,
- rewrite history,
- delete old benchmark reports,
- remove historical final matrices,
- overwrite previous immutable raw outputs.

Create new corrective artifacts under a new root, for example:

```text
benchmark_reports/corrective_recovery_20260908/
```

or a later timestamped equivalent.

## 5.1 Push after every stage

After **every stage that changes code or authoritative documentation**:

```bash
git status --short
git diff --check
git add <intentional files only>
git commit -m "<stage-specific message>"
git push origin vlm-testing-pipeline
```

Do not batch five stages into one commit.

Minimum push cadence:
- Stage 0 baseline/forensics record
- Stage 1 online/offline contract split
- Stage 2 capability layer
- Stage 3 role canonicalization
- Stage 4 relation/operation interface
- Stage 5 search gating
- Stage 6 planner provenance
- Stage 7 evaluator/attribution
- Stage 8 structured-output inference changes
- Stage 9 real-trace regression corpus
- Stage 10 small live probe selection
- Stage 11 full tests/freeze
- Stage 12 development matrix reports
- Stage 13 held-out matrix reports
- Stage 14 final comparative analysis/docs

If a stage requires multiple logically independent fixes, commit and push more frequently.

---

# 6. PROGRESS LEDGER

Create or append this section to:

```text
docs/corrective_recovery.md
```

Do not overwrite the historical content of `docs/final_fix.md`.

Use:

| Stage | Status | Commit | Files changed | Tests | Raw replays | Live calls | Result | Remaining |
|---|---|---|---|---|---|---:|---|---|
| 0 | NOT STARTED | | | | | 0 | | |
| 1 | NOT STARTED | | | | | 0 | | |
| 2 | NOT STARTED | | | | | 0 | | |
| 3 | NOT STARTED | | | | | 0 | | |
| 4 | NOT STARTED | | | | | 0 | | |
| 5 | NOT STARTED | | | | | 0 | | |
| 6 | NOT STARTED | | | | | 0 | | |
| 7 | NOT STARTED | | | | | 0 | | |
| 8 | NOT STARTED | | | | | 0 | | |
| 9 | NOT STARTED | | | | | 0 | | |
| 10 | NOT STARTED | | | | | | | |
| 11 | NOT STARTED | | | | | 0 | | |
| 12 | NOT STARTED | | | | | 32 | | |
| 13 | NOT STARTED | | | | | 15 | | |
| 14 | NOT STARTED | | | | | 0 | | |

Update and commit this ledger at every stage.

---

# 7. STAGE 0 — FORENSIC BASELINE AND SAFETY SNAPSHOT

## Goal

Establish an immutable causal baseline before changing behavior.

## Tasks

1. Confirm branch/HEAD.
2. Verify remote inference path and keep it running.
3. Preserve current final reports.
4. Recompute current metrics from raw records without new live calls.
5. Extract representative raw VLM outputs and compiler traces for at least:
   - K1
   - K2
   - K3
   - L1
   - one Living infeasible
   - W1
   - W2
   - W8
6. Build a table:

```text
variant
raw task roles expressed
raw task operations expressed
raw relations expressed
production role mappings
production operation mappings
production relation mappings
contract-complete result
search state
grounding state
A* invoked?
first-cause label
manual forensic interpretation
```

7. Verify the current suspected issues:
   - Workshop V2 physical-precondition skip
   - hardcoded online domain completeness
   - Kitchen missing material-transfer capability
   - role mapping failures on sensible paraphrases
   - search blocked by contract-incomplete
   - planner action generation from role identity
   - evaluator undercoverage.

## Gate 0

Do not modify semantics until the forensic table proves where each representative variant dies.

## Commit

```text
docs(corrective): record af2dde forensic baseline
```

Push immediately.

---

# 8. STAGE 1 — SEPARATE ONLINE EXECUTABILITY FROM OFFLINE TASK COMPLETENESS

This is the highest-priority architectural correction.

## 8.1 Replace domain-answer checklists in online runtime

Remove any online logic equivalent to:

```python
if domain == "kitchen":
    require exact hidden expected operations...

if domain == "workshop":
    require exact hidden expected roles/predicates...
```

Do not simply move the same hidden checklist to a different online file.

## 8.2 Define generic online executable-contract completeness

The online contract is executable iff:

1. Structural sanitizer succeeded.
2. Every **expressed required role** is either:
   - mapped to a selectable canonical runtime role, or
   - valid explicitly expressed fixed context.
3. Every **expressed required relation** that is intended to constrain physical grounding:
   - maps to a supported physical verifier,
   - or is preserved as a non-physical task/causal relation that does not require G_O proof.
4. Every **expressed required operation** maps to a robot capability.
5. Every mapped robot capability has all required runtime primitives/checkers.
6. Counts and binding policies are internally coherent.
7. No unresolved required FM reference remains.
8. No physical verifier is missing for an executable physical precondition.

This definition may not ask:

> "Did the FM remember every hidden reference requirement?"

That is offline evaluation.

## 8.3 Add separate fields

Use separate names, for example:

```text
online_executable_contract_complete
offline_reference_task_complete
```

Preserve backwards-compatible report aliases only if necessary, and clearly mark them.

## 8.4 First-cause rule

If the FM did not express a reference requirement:

```text
TASK_SPECIFICATION_FAILURE
```

If FM language clearly expresses it but production cannot map it:

```text
GRAPH_COMPILATION_FAILURE
```

If online contract is executable but physical candidate discovery fails:

```text
OBJECT_DISCOVERY_FAILURE
```

etc.

## Gate 1

Synthetic and raw-replay tests must prove:

```text
FM omission
    -> offline task incomplete

but

an otherwise internally executable expressed contract
    -> can still reach search/planning
```

and:

```text
reasonable raw semantic
+ failed mapper
    -> GRAPH_COMPILATION_FAILURE
```

not `TASK_SPECIFICATION_FAILURE`.

## Commit

```text
fix(contract): separate online executability from offline completeness
```

Push immediately.

---

# 9. STAGE 2 — COMPLETE THE EXPLICIT OPERATION → ROBOT CAPABILITY BRIDGE

This stage must not restore endpoint-only task inference.

## 9.1 Define the abstraction correctly

The FM expresses task transformations in free language.

The compiler maps those explicit transformations to robot capabilities.

Robot capabilities provide:
- planner-level operation identity,
- physical feasibility preconditions,
- low-level action templates/primitives.

## 9.2 Audit planner primitives against capability registry

Build a table:

| Domain | Planner can do | Explicit task capability exists? | Fix |
|---|---|---|---|
| Kitchen | transfer/pour coffee | currently incomplete | add |
| Kitchen | transfer/pour water | currently incomplete | add |
| Kitchen | stir | yes | audit |
| Kitchen | associate eating utensil with soup | yes-ish | audit semantics |
| Kitchen | final serving placement | audit provenance | explicit goal/capability or legitimate task effect |
| Living | place refreshment setting on personal support | yes-ish | audit |
| Living | place entertainment control on shared support | yes-ish | audit |
| Workshop | fasten component at target | yes | fix V2 physical preconditions |
| Workshop | return reusable equipment to workbench | audit | add if needed |

Do not add benchmark-specific capability names when a generic physical capability is sufficient.

Potential generic capabilities:

```text
TRANSFER_CONTENT_TO_CONTAINER
STIR_CONTENTS
PLACE_ASSOCIATED_UTENSIL
PLACE_PAYLOAD_ON_SUPPORT
FASTEN_COMPONENT_AT_TARGET
RETURN_REUSABLE_ITEM_TO_SUPPORT
```

Planner-specific names may remain internally, but VLM-facing language remains open.

## 9.3 Workshop V2 bug

Fix the Workshop singleton path so:

```text
explicit V2 fasten operation
    -> capability mapping
    -> physical preconditions instantiated
```

without requiring the FM to separately name all three checker predicates.

Do not gate capability preconditions on `not is_v2`.

## 9.4 Physical-precondition provenance

Every generated physical predicate must record provenance:

```json
{
  "predicate": "REACHES_TARGET",
  "provenance": "ROBOT_CAPABILITY_PRECONDITION",
  "source_operation_id": "...",
  "capability_id": "..."
}
```

This proves the runtime did not invent a task operation.

## 9.5 No operation from endpoints

A pair:

```text
coffee_stirrer + coffee_container
```

must **not** create STIR unless the FM expressed a stirring transformation.

Likewise:

```text
driver + fastener
```

must not create FASTEN unless an explicit fastening operation exists.

## Gate 2

Tests must prove:

```text
explicit FM "stir contents"
    -> STIR capability
    -> INSERTABLE_IN + REACHES_BOTTOM

explicit FM "install/fasten component at target"
    -> FASTEN capability
    -> workshop physical preconditions

explicit FM "transfer material into container"
    -> transfer capability
    -> planner can generate POUR realization

roles only, no explicit operation
    -> no task operation synthesized
```

## Commit

```text
fix(capabilities): complete explicit operation bridge
```

Push immediately.

---

# 10. STAGE 3 — ROBUST ROLE CANONICALIZATION USING REAL QWEN LANGUAGE

## Goal

Map semantically clear FM roles without GT leakage or variant-specific special cases.

## 10.1 Build real-language regression fixtures

Use captured raw outputs from current live records.

Include at minimum paraphrases like:

```text
"Holds soup ready for consumption."
"Tool used to mix ingredients inside the coffee serving vessel."
"Device used to operate the television or media system."
"A collection of items designated for consumption by one person."
"An implement used to manipulate or install the fastening component."
"A physical item capable of connecting or securing elements at the marked location."
```

Do not label the fixture by benchmark variant in production code.

Fixture names can describe the semantic concept.

## 10.2 Role mapping inputs

Role identity may use:
- function
- description
- candidate categories
- required unary properties
- entity kind
- causal position in expressed operations
- causal position in expressed relations
- count/binding policy as a disambiguator

Role identity may not use:
- benchmark variant ID
- GT expected role list
- hidden object location
- expected plan
- scene-specific object IDs.

## 10.3 Fix collision logic

A stirring implement must not collide with the coffee container merely because both mention coffee/container.

Prioritize causal head:
- implement/tool/instrument acting on target,
- source/provider supplying material,
- receptacle/container receiving material,
- component installed,
- fixed target receiving component,
- support region receiving payload.

## 10.4 Semantic category use

Runtime semantic ontology may provide generic canonical categories/synonyms.

Candidate categories from the FM may help disambiguate role identity and detector vocabulary.

They may not create missing task roles.

## 10.5 Living payload roles

Make sure natural phrases for:
- personal refreshment setting payload,
- entertainment control payload,
- personal support region,
- shared support region,
- seating context

map robustly.

## 10.6 Workshop roles

Make sure natural phrases for:
- reusable fastening implement/tool,
- fastening component that remains installed,
- marked fixed receiving target,
- generic workbench support context

remain distinct.

## Gate 3

All captured real-language role fixtures pass.

No endpoint-only operation/relation synthesis tests regress.

## Commit

```text
fix(semantics): harden role canonicalization for natural FM paraphrases
```

Push immediately.

---

# 11. STAGE 4 — REFACTOR RELATION HANDLING: TASK/CAUSAL SEMANTICS VS PHYSICAL VERIFIERS

Current design overburdens `functional_relations`.

Do not force every FM relation into a geometry predicate.

## 11.1 Separate conceptual categories

At compile time, distinguish:

### A. Task/causal semantic relations

Examples:

```text
coffee source provides material to coffee container
water source provides material to coffee container
tool acts on component
component is installed at target
```

These may support operation construction/provenance but are not necessarily direct static G_O predicates.

### B. Physical verifier constraints

Examples:

```text
INSERTABLE_IN
REACHES_BOTTOM
FITS_ON
NEAR_SEAT
COMPATIBLE_WITH
REACHES_TARGET
```

These require TRUE/FALSE/UNKNOWN physical evidence.

### C. Capability-derived physical preconditions

These are generated only after an explicit FM operation maps to a capability.

## 11.2 FM can remain open-ended

Do not expose predicate enum names in the prompt.

Interpret broad natural paraphrases safely.

Endpoint roles may filter/disambiguate candidate physical predicates, never create them.

## 11.3 Do not require redundant FM semantics

If the FM says:

```text
operation = stir contents
```

do not also require the FM to explicitly say:
- implement fits in container,
- implement reaches bottom,

if these are robot capability preconditions.

Likewise a fastening operation need not redundantly enumerate every backend geometry checker if the robot capability already defines them.

## 11.4 Required relation failure

If the FM explicitly declares a required relation that the runtime cannot interpret and it matters for execution:

```text
UNINTERPRETABLE_REQUIRED_RELATION
```

Online executable contract becomes false.

If it is a causal/narrative relation already represented by a mapped operation and it has no physical checker interpretation, preserve it as task semantic provenance instead of silently dropping it.

## Gate 4

Tests cover:
- explicit physical relation -> verifier
- explicit causal relation -> preserved semantic edge
- explicit operation -> capability-derived physical predicates
- unknown required physical relation -> fail closed
- endpoint signature alone -> never create relation

## Commit

```text
refactor(relations): separate task semantics from physical verification
```

Push immediately.

---

# 12. STAGE 5 — FIX SEARCH ELIGIBILITY AND CAUSAL RECOVERY

## 12.1 Search state machine

Implement generic states similar to:

```text
CONTRACT_UNEXECUTABLE
GROUNDING_COMPLETE
NO_CANDIDATE_SEARCHABLE
ONLY_FALSE_CANDIDATES_SEARCHABLE
ONLY_UNKNOWN_CANDIDATES_SEARCHABLE
NO_VALID_JOINT_ASSIGNMENT_SEARCHABLE
SEARCH_EXHAUSTED
```

## 12.2 Search may proceed when

```text
online_executable_contract_complete == true
```

and at least one expressed requirement is not currently grounded due to evidence/candidate insufficiency, and unopened inspectable regions remain.

This includes:
- zero candidates,
- all FALSE,
- all UNKNOWN,
- joint assignment failure.

## 12.3 Search may not proceed merely to repair semantics

If:
- expressed required role cannot be interpreted,
- expressed required operation cannot map,
- required physical relation cannot be interpreted,

then search cannot fix it.

## 12.4 Do not use offline GT completeness as search gate

This is critical.

A hidden reference omission may make the final task fail offline, but the online runtime does not know that and must not use the reference graph to decide search eligibility.

## 12.5 Recovery metric

Count recovery only when:

```text
initial grounding incomplete
AND
at least one inspection occurred
AND
after search grounding became complete enough for the expressed executable contract
```

## Gate 5

Raw replay of K2/K3 and W1/W2 should demonstrate search is no longer globally disabled merely by reference-task incompleteness.

Whether each ultimately earns full task success is a separate offline question.

## Commit

```text
fix(search): restore evidence-driven inspection eligibility
```

Push immediately.

---

# 13. STAGE 6 — PLANNER ACTION PROVENANCE AND GOAL COMPILATION

The planner must not recreate hidden task semantics after the compiler was made strict.

## 13.1 General low-level primitives are allowed

Robot action models may always know generic primitives such as:

```text
PICK
PLACE
POUR
STIR
FASTEN
```

But those primitives may only be instantiated toward task goals derived from explicit FM task semantics/capabilities.

## 13.2 Audit Kitchen

Current candidate planner generates POUR whenever it sees source/container role pairs.

Change to:

```text
mapped TRANSFER_CONTENT operation
    -> enables relevant source-target transfer goal/action schema
```

Do not create transfer task semantics from role pair alone.

Likewise:
- STIR action requires explicit mapped stir operation.
- soup-utensil association requires explicit mapped serve/associate operation.
- final serving goals must be supported by explicit task semantics from the instruction/FΜ contract, not hidden reference recipe.

## 13.3 Audit Living

Personal refreshment placement and shared control placement must follow mapped task operations/relations.

## 13.4 Audit Workshop

Fastening and equipment return goals must follow explicit mapped operations.

## 13.5 Independent replay

Continue independent symbolic replay of every candidate plan.

A non-empty plan is not sufficient.

## Gate 6

Tests prove:
- roles without explicit operation do not create task operation goals,
- explicit operation enables corresponding planner realization,
- planner still generates valid low-level primitives,
- one A* only,
- independent replay validates state transitions.

## Commit

```text
fix(planner): gate task goals on explicit compiled semantics
```

Push immediately.

---

# 14. STAGE 7 — FIX RAW EVALUATION AND FIRST-CAUSE ATTRIBUTION

## 14.1 Keep raw evaluation offline only

It may read GT/reference specifications because it runs after generation.

It must not import/use the production compiler to decide whether a raw semantic exists.

## 14.2 Expand semantic coverage

The raw evaluator must cover all task operation families actually used across:
- Kitchen
- Living Room
- Workshop

Do not leave Workshop operation recall as `None` because the matcher lacks its operation vocabulary.

## 14.3 Evaluate semantic meaning, not exact wording

Use a frozen independent semantic matcher with:
- robust phrase normalization,
- lexical families,
- causal-role context,
- endpoint semantic role matching,
- explicit operation meaning.

Do not score only exact backend predicate names.

## 14.4 First-cause distinction

For each failed feasible case, save:

```text
raw requirement present?
production mapped?
capability mapped?
physical evidence available?
search attempted?
grounding complete?
A* attempted?
plan valid?
```

Then apply:

```text
RAW REQUIRED SEMANTIC ABSENT
    -> TASK_SPECIFICATION_FAILURE

RAW SEMANTIC PRESENT BUT PRODUCTION CANNOT REPRESENT
    -> GRAPH_COMPILATION_FAILURE

EXECUTABLE G_F COMPLETE, VALID OBJECT EXISTS, SEARCH MISSES
    -> OBJECT_DISCOVERY_FAILURE

CANDIDATES OBSERVED, VALID PHYSICAL ASSIGNMENT EXISTS, GROUNDING FAILS
    -> FUNCTIONAL_ASSIGNMENT_FAILURE

VALID COMPLETE G_F + phi*, PLANNER CANNOT PRODUCE VALID PLAN
    -> PLANNING_FAILURE
```

## 14.5 Never attribute compiler rejection to FM omission automatically

A production `UNRESOLVED_SEMANTIC` does not prove raw omission.

## Gate 7

Representative K2/W1 cases must be correctly separable into raw-vs-compiler causes.

## Commit

```text
fix(eval): correct raw semantic scoring and first-cause attribution
```

Push immediately.

---

# 15. STAGE 8 — STRUCTURED OUTPUT AND QWEN INFERENCE RELIABILITY

Do not run another 32-case matrix until this is fixed.

## 15.1 Inspect current request path

Record:
- `response_format`
- guided JSON/schema backend used by vLLM
- Qwen chat template
- thinking mode
- output token cap
- temperature/top-p/top-k
- penalties
- finish reason
- content vs reasoning_content handling
- malformed JSON cases.

## 15.2 Compare controlled inference configs

Use only a small development probe such as:

```text
K1
K2
L1
L2
W1
W2
```

or another balanced six-case set.

Test a small number of configurations.

At minimum compare:

### Config A — stable structured baseline

```text
thinking = false
temperature = 0 or very low
reasonable output cap
strict JSON schema / guided decoding
```

### Config B — thinking

Only if structured decoding is actually compatible and outputs are reliably valid.

Do not use 24k output merely because it is available.

## 15.3 Selection order

Select configuration using:

1. JSON/schema validity
2. zero truncation
3. semantic role/operation completeness
4. production canonicalization
5. executable-contract rate
6. latency

Do not select a configuration with malformed JSON because it marginally improves one semantic metric.

## 15.4 No full-matrix tuning

All config selection happens on the small development probe.

Once selected, freeze it before the final 32.

## Gate 8

Six-case probe must have:
- 100% parse/schema validity,
- no output truncation,
- no transport failures,
- reasonable latency,
- no duplicate semantic VLM calls.

## Commit

```text
fix(vlm): freeze reliable structured Qwen inference config
```

Push immediately.

---

# 16. STAGE 9 — REAL-TRACE REGRESSION SUITE

Before new live evaluation, build a regression suite from captured real raw Qwen outputs.

Include at least:

```text
Kitchen:
    clear role phrasing
    source phrasing
    stirrer phrasing
    soup utensil phrasing
    transfer operations
    stir operation
    serving/association operation

Living:
    refreshment payload
    personal support
    shared support
    entertainment control
    seating context
    personal placement operation
    shared control placement

Workshop:
    reusable tool
    installed component
    marked target
    fastening operation
    equipment return operation
```

For each fixture assert:
- raw semantic interpretation,
- canonical role mapping,
- operation capability mapping,
- relation classification,
- online executable completeness,
- no GT/provider access,
- no variant logic.

Include negative fixtures:
- vague "thing"
- wrong endpoint semantics
- unsupported operation
- empty operation
- self pairing
- ambiguous role.

## Gate 9

Full test suite green.

Real trace fixtures green.

## Commit

```text
test(vlm): add real-output semantic regression corpus
```

Push immediately.

---

# 17. STAGE 10 — SMALL END-TO-END LIVE PROBE

Only now spend new live benchmark calls.

Suggested balanced probe:

```text
K1
K2
K3
L1
L2
W1
W2
W8
```

Each variant:
- exactly one semantic VLM call,
- zero high-level replans,
- max one A*.

For each case save:
- raw response,
- sanitized spec,
- semantic compile trace,
- operation capability trace,
- physical-precondition provenance,
- search trace,
- grounding trace,
- A* trace,
- independent replay result,
- offline first-cause analysis.

## Gate 10

Do not proceed to full 32 unless:

1. structured output is fully reliable,
2. executable-contract rate is no longer globally zero,
3. search occurs on at least some genuinely recovery-requiring cases,
4. W1/W2 do not regress solely because capability preconditions were discarded,
5. K cases with clear transfer/stir semantics can reach appropriate capability mappings,
6. A* failures are no longer dominated by upstream interface defects.

This gate does **not** require perfect success.

It requires evidence the funnel is functioning.

## Commit

```text
eval(probe): validate corrected end-to-end semantic funnel
```

Push immediately.

---

# 18. STAGE 11 — FULL TEST SUITE AND METHOD FREEZE

Run all relevant suites.

At minimum:
- pipeline tests
- scene adapter tests
- evaluator tests
- raw replay tests
- anti-leakage tests
- real-trace tests.

Run `git diff --check`.

Verify production code does not contain:
- benchmark variant IDs in semantic mappings,
- GT provider imports in online path,
- expected assignment lookup,
- expected plan lookup,
- hidden object locations,
- task-specific answer graph used for online completeness.

Freeze hashes:
- active prompt
- response schema
- runtime semantic ontology
- capability registry
- predicate registry
- relevant planner semantic compiler files
- selected inference config.

Document freeze commit.

## Commit

```text
chore(freeze): freeze corrected functional-tamp method
```

Push immediately.

No method changes after this point unless the final matrix reveals an **implementation bug** invalidating the evaluation. If such a bug is found, invalidate the matrix, fix, re-freeze, and rerun from scratch. Never patch mid-matrix.

---

# 19. STAGE 12 — FINAL DEVELOPMENT 32x1

Run exactly the canonical 32 development variants:

```text
Kitchen K1-K12
Living L1-L10
Workshop W1-W10
```

One semantic VLM call each.

No semantic retries.

No high-level replans.

Do not overwrite previous report roots.

New output root example:

```text
benchmark_reports/corrective_recovery_final_32x1_<timestamp>/
```

Validate:
- 32 unique variants
- 20 feasible
- 12 infeasible
- exactly 32 semantic VLM calls total
- zero high-level replans
- no duplicate variant execution
- frozen hashes identical
- git clean/frozen.

Report:

Primary:
- Outcome Correct
- Feasible Success
- Recovery
- Goal Coverage
- False Completion
- VLM Requests
- Replans

Funnel diagnostics:
- JSON/schema validity
- raw role precision/recall/F1
- raw operation precision/recall/F1
- raw relation semantic coverage
- online executable-contract rate
- compiler interpretation rate
- search eligible
- search actually executed
- recovery after search
- any verified grounding
- complete expressed-contract grounding
- A* invocation
- non-empty plan
- independent plan validity
- full-task satisfaction
- first-cause distribution.

## Commit

```text
eval(final): publish corrected 32x1 development matrix
```

Push immediately.

---

# 20. STAGE 13 — HELD-OUT / GENERALIZATION MATRIX

Before running, audit the current held-out definition and provenance.

The previous held-out configuration was modified after an initial freeze for at least one unsupported placement.

Do not falsely call a post-hoc modified set pristine untouched held-out.

Choose one of two scientifically honest paths:

### Path A
Use a newly generated post-freeze held-out matrix that was not viewed during method development.

### Path B
Retain existing set but label it accurately as:
- post-hoc stress test,
- additional generalization matrix,
- or corrected held-out configuration with explicit provenance.

Do not change the method after seeing results.

Run one VLM call per case.

Generate separate reports.

## Commit

```text
eval(heldout): publish post-freeze generalization matrix
```

Push immediately.

---

# 21. STAGE 14 — FINAL ANALYSIS AND PAPER-READY REPORTS

Generate:

```text
benchmark_reports/corrective_recovery_analysis_<timestamp>/
```

with:

1. `primary_metrics.md`
2. `domain_diagnostics.md`
3. `first_cause_failures.md`
4. `semantic_funnel.md`
5. `relation_operation_statistics.md`
6. `search_recovery_statistics.md`
7. `per_variant_trace_summary.json`
8. `leakage_and_provenance_audit.json`
9. `method_freeze.json`
10. `comparative_report.md`

Compare at least:

```text
historical baseline
af2dde strict-but-collapsed result
new corrected method
```

Do not hide negative metrics.

Explicitly separate:
- true FM task omissions,
- compiler/interface failures,
- discovery failures,
- grounding failures,
- planning failures.

Update `docs/corrective_recovery.md` with final status.

Do not rewrite historical `final_fix.md` to pretend the prior result never existed.

## Commit

```text
docs(eval): finalize corrective recovery analysis
```

Push immediately.

---

# 22. REQUIRED IMPLEMENTATION DETAILS

## 22.1 Capability semantics vs primitive realization

Example:

```text
FM:
    "add water to each beverage container"

semantic compiler:
    TRANSFER_CONTENT_TO_CONTAINER

robot capability:
    can realize using POUR primitive

planner:
    PICK source
    POUR source -> target
    PLACE source
```

The planner owns the primitive sequence.
The FM owns the requested task transformation.

## 22.2 Operation-derived physical requirements

Example:

```text
FM:
    "stir contents"

capability:
    STIR_CONTENTS

derived physical checks:
    implement must enter target opening
    implement must reach useful depth
```

These checks are legitimate capability preconditions.

Do not mark the FM incomplete merely because it did not explicitly state both checker names.

## 22.3 Fixed anchors

Environment may provide calibrated non-selectable context only when the FM expressed the relevant contextual role/operation.

Examples:
- marked repair location,
- seating position/pair,
- known workbench support.

Do not create missing selectable task participants.

## 22.4 Candidate categories

Use FM candidate categories for:
- open-vocabulary detector vocabulary,
- semantic role disambiguation where justified.

Do not use them as proof of geometric compatibility.

## 22.5 Physical verification

Maintain:

```text
TRUE
FALSE
UNKNOWN
```

Rules:
- TRUE satisfies
- FALSE rejects
- UNKNOWN does not satisfy
- detectors do not prove geometry
- semantic similarity does not prove geometry.

---

# 23. THINGS THE AGENT MUST NOT DO TO "IMPROVE PERFORMANCE"

Do not:
- re-add endpoint-only STIR inference,
- re-add endpoint-only FASTEN inference,
- add `if variant == K2`,
- add task-specific answer graphs to online compiler,
- silently turn unresolved relation into TRUE,
- mark UNKNOWN geometry as valid,
- load GT/reference YAML in online role ontology,
- run a second VLM semantic repair call,
- use hidden object locations for search,
- inject expected objects in detector vocabulary from GT,
- tune on all 32 repeatedly,
- run final matrix before small probes pass,
- kill the working remote ngrok tunnel unnecessarily,
- restart vLLM repeatedly without cause,
- force-push.

---

# 24. REQUIRED CHECKS AFTER EVERY CODE STAGE

Run:

```bash
git diff --check
git status --short
```

Run focused tests.

Then run the smallest relevant raw replay.

Then commit and push.

Never carry a large uncommitted multi-stage diff.

---

# 25. FINAL SUCCESS CRITERIA

This corrective pass is complete only when all are true:

### Infrastructure
- remote Qwen server stable,
- SSH/ngrok path remained available,
- local API tunnel stable,
- exact model verified.

### Scientific
- exactly one semantic VLM call per benchmark case,
- zero high-level replans,
- no GT/reference online,
- no endpoint-only task semantic synthesis,
- physical verification remains fail-closed,
- capability physical preconditions are correctly allowed from explicit operations.

### Engineering
- online executable completeness is generic,
- offline task completeness is separate,
- real Qwen paraphrases map robustly,
- operation registry covers planner-supported task operations,
- search is no longer globally starved,
- planner task goals are traceable to explicit compiled semantics,
- raw evaluator does not mislabel compiler failures as FM omissions.

### Evaluation
- small live probe demonstrates a functioning funnel,
- full tests pass,
- method frozen before full evaluation,
- final 32x1 complete and valid,
- generalization matrix complete and honestly labeled,
- all reports generated,
- final branch pushed and clean.

---

# 26. START NOW — FIRST COMMAND SEQUENCE

Local repository:

```bash
cd /home/naren/RA_iiith
git fetch origin
git checkout vlm-testing-pipeline
git pull --ff-only origin vlm-testing-pipeline
git rev-parse HEAD
git status --short
```

Connectivity:

```bash
nc -vz -w 10 0.tcp.in.ngrok.io 23808
```

Remote inspection:

```bash
ssh -i ~/keyfile -p 23808 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  long-horizon@0.tcp.in.ngrok.io \
  'hostname; nvidia-smi; ps aux | grep -E "[v]llm|[q]wen|[n]grok"; ss -ltnp | grep -E ":8000|:22" || true'
```

Then establish/reuse the local API tunnel, verify `/v1/models`, create `docs/corrective_recovery.md`, perform Stage 0 forensics, commit, push, and continue through every stage above.

---

# 27. FINAL AGENT REPORT FORMAT

At completion, return:

```text
FINAL STATUS
- Branch:
- Final HEAD:
- Git clean:
- Remote model:
- vLLM version:
- Inference config:
- Prompt/schema hash:
- Tests:
- Development matrix:
- Generalization matrix:

PRIMARY METRICS
- Outcome Correct:
- Feasible Success:
- Recovery:
- Goal Coverage:
- False Completion:
- VLM Requests:
- Replans:

FAILURE FUNNEL
- Task specification failures:
- Graph/compiler failures:
- Discovery failures:
- Grounding failures:
- Planning failures:

FILES
- Corrective implementation ledger:
- Final development report:
- Final generalization report:
- Comparative report:
- Leakage/provenance audit:

PUSHED COMMITS
- Stage 0:
- Stage 1:
- ...
- Stage 14:

REMAINING SCIENTIFIC LIMITATIONS
- ...
```

Do not finish with vague language like "infrastructure works but result is negative" without proving where the remaining bottleneck is.

The purpose of this corrective pass is to produce a scientifically valid, fully traced, performance-oriented implementation in which:
- the FM owns task semantics,
- the robot capability layer legitimately supplies physical execution requirements,
- search is allowed to solve observation incompleteness,
- and the evaluator accurately distinguishes FM failure from compiler failure.

Execute it completely.
