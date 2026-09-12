# FM-TAMP — Final Experiment Freeze

Closure pass on `vlm-testing-pipeline`. This report supersedes
`FM_TAMP_CLOSURE_FINAL.md` for every metric definition and every determinism
claim. Numbers are computed from artifacts by
`mujoco_scenes/evaluation_outcome.py::summarize`, which raises rather than
returns if the decomposition is inconsistent.

> **Interpretation, stated once and applied throughout.** The archived replay
> measures the current downstream pipeline on 96 FM responses collected under a
> *different* prompt and schema. It is a regression and development
> measurement. **It is not a forecast of final live performance.** The live
> figure is whatever the 10x32 produces; the smoke (§I) is a single sample and
> is reported separately.

---

## A. Final git / config identity

Emitted by `scripts/freeze_method_identity.py` into `METHOD_FREEZE.json`; every
field is computed from the artifact it describes, so the freeze cannot silently
stop describing the tree it names.

```
RELEASE TAG                 fm-tamp-final-experiment-v1   <- identifies the commit
behavioural_identity_sha256 b4abf1a827a4adf669c1c44ce5ba33e703047859107e2dd11de8cf030c7afc89
branch                      vlm-testing-pipeline

prompt_and_schema_sha256    156c661a45a43646b51cdf72a2c68e023e8d4c79e4fcc35f6a0d69074b6fff0b
capability_registry_sha256  98055ba6579345151cd1b7a9ccea32fc2ad47dd0f51f51cf40e3eb9cdb01aef7
runtime_ontology_sha256     4fee971c3f494023adadc94fa350d3d9e27d81dbe41c001c2b8fc1d20bf49d41
schema_version              3

workshop_config_sha256      9b847075051a2d147dfcd628034eb59c16a1311564eed08ef77058b76d12a889
workshop_geometry_sha256    3e41034f4765325856d85a05e0d0f161dfad4e89100c03868e03c992662524e9
semantic_grounding_sha256   7e5a57e631b21374f1ad8f6fd00cac9e6ec6987302801b7a7a5e466bd69dc957
determinism_module_sha256   a8667ce9e27097799859e1d104ef27369de92e4102b96fe09b68f9820578545a

evaluator_module_sha256     9668af76c7a9ee5b31afce919fc90f5fad34799e9d01181eaebcbb7cc28fd80e
completion_claimed          ['ACTION_SEQUENCE_READY']
infeasibility_concluded     ['EXHAUSTED_NO_VALID_GROUNDING', 'INFEASIBLE', 'NO_VALID_COMPLETE_ASSIGNMENT', 'PLANNING_PROVEN_INFEASIBLE']

benchmark grid              32 variants {'kitchen': 12, 'living_room': 10, 'workshop': 10}
```

**Repository identity is the annotated tag, not a field in the artifact.** A
tracked file cannot contain the hash of the commit that contains it: writing it
dirties the tree, committing produces a new SHA, and the recorded one is
immediately stale. An earlier version of this report recorded a SHA that was two
commits behind the branch with `clean: false`. The artifact carries **no commit SHA at all** -- recording the preceding commit
merely moves the staleness by one, so regenerating after the commit that
contains the file yields a different value and the artifact never matches its
own tree. `fm-tamp-final-experiment-v1` names the released commit.

Reproducible child environment, passed to every spawned trial by both the
replay driver and the live runner (these cannot be set in process -- the
interpreter reads `PYTHONHASHSEED` at startup and the numeric libraries read the
thread counts when they first load):

```
{
  "MKL_NUM_THREADS": "2",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "2",
  "PYTHONHASHSEED": "0",
  "TOKENIZERS_PARALLELISM": "false"
}
```

---

## B. Metrics from the archived replay

Three independent full replays (A, B, C) at the final code, `--workers 2`,
96/96 rows each, **zero** `HARNESS_FAILURE`, identical metrics.

| Metric | Value |
| :--- | :--- |
| Trials | 96 (60 feasible, 36 infeasible) |
| **Feasible success** | **34 / 60 (56.7%)** |
| — of which exact GT action sequence | **21 / 34** (§B.1) |
| — of which same steps, any order | **32 / 34** (§B.1) |
| — Kitchen | 11 / 18 |
| — Living Room | 10 / 18 |
| — Workshop | 13 / 24 |
| feasible_outcome_correct | 34 / 60 |
| **infeasible_outcome_correct** | **16 / 36** |
| **overall_outcome_correct** | **50 / 96 (52.1%)** |
| **False completions** | **0 / 96** |
| Complete grounding | 34 |
| Planner failures after complete grounding | **0** |
| Mean GT goal coverage (feasible) | 0.6625 |
| A\* invocations per trial | ≤ 1 |
| Semantic FM calls during replay | **0** |
| GT-leakage audit | **FINDINGS: 0** |
| Combined test surface | 2245 passed, 49 failed, 12 errors (all pre-existing) |
| Pipeline suite at final commit | **1225 passed, 0 failed** |

| Domain | Variant | GT feasible | Success /3 | Mean GT coverage | Complete grounding /3 | Outcome correct /3 | False completions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| kitchen | K1 | yes | 2/3 | 0.833 | 2/3 | 2/3 | 0 |
| kitchen | K2 | yes | 2/3 | 0.917 | 2/3 | 2/3 | 0 |
| kitchen | K3 | yes | 1/3 | 0.667 | 1/3 | 1/3 | 0 |
| kitchen | K4 | yes | 1/3 | 0.500 | 1/3 | 1/3 | 0 |
| kitchen | K5 | yes | 2/3 | 0.667 | 2/3 | 2/3 | 0 |
| kitchen | K6 | yes | 3/3 | 1.000 | 3/3 | 3/3 | 0 |
| kitchen | K7 | no | 0/3 | 0.500 | 0/3 | 0/3 | 0 |
| kitchen | K8 | no | 0/3 | 0.000 | 0/3 | 0/3 | 0 |
| kitchen | K9 | no | 0/3 | 0.000 | 0/3 | 0/3 | 0 |
| kitchen | K10 | no | 0/3 | 0.167 | 0/3 | 0/3 | 0 |
| kitchen | K11 | no | 0/3 | 0.167 | 0/3 | 0/3 | 0 |
| kitchen | K12 | no | 0/3 | 0.333 | 0/3 | 0/3 | 0 |
| living_room | L1 | yes | 2/3 | 0.778 | 2/3 | 2/3 | 0 |
| living_room | L2 | yes | 2/3 | 0.667 | 2/3 | 2/3 | 0 |
| living_room | L3 | yes | 0/3 | 0.222 | 0/3 | 0/3 | 0 |
| living_room | L4 | yes | 3/3 | 1.000 | 3/3 | 3/3 | 0 |
| living_room | L5 | yes | 2/3 | 0.667 | 2/3 | 2/3 | 0 |
| living_room | L6 | yes | 1/3 | 0.333 | 1/3 | 1/3 | 0 |
| living_room | L7 | no | 0/3 | 0.444 | 0/3 | 0/3 | 0 |
| living_room | L8 | no | 0/3 | 0.111 | 0/3 | 1/3 | 0 |
| living_room | L9 | no | 0/3 | 0.000 | 0/3 | 2/3 | 0 |
| living_room | L10 | no | 0/3 | 0.000 | 0/3 | 2/3 | 0 |
| workshop | W1 | yes | 2/3 | 0.778 | 2/3 | 2/3 | 0 |
| workshop | W2 | yes | 2/3 | 0.778 | 2/3 | 2/3 | 0 |
| workshop | W3 | yes | 0/3 | 0.000 | 0/3 | 0/3 | 0 |
| workshop | W4 | yes | 0/3 | 0.333 | 0/3 | 0/3 | 0 |
| workshop | W5 | yes | 0/3 | 0.111 | 0/3 | 0/3 | 0 |
| workshop | W6 | yes | 3/3 | 1.000 | 3/3 | 3/3 | 0 |
| workshop | W7 | yes | 3/3 | 1.000 | 3/3 | 3/3 | 0 |
| workshop | W8 | yes | 3/3 | 1.000 | 3/3 | 3/3 | 0 |
| workshop | W9 | no | 0/3 | 0.222 | 0/3 | 1/3 | 0 |
| workshop | W10 | no | 0/3 | 0.222 | 0/3 | 1/3 | 0 |


### Arithmetic

The decomposition is asserted mechanically, not derived by hand:

```
overall_outcome_correct == feasible_outcome_correct + infeasible_outcome_correct
feasible_success        == feasible_outcome_correct
```

`summarize()` raises `AssertionError` if either fails. A previous report stated
34 feasible successes, 7 infeasibility conclusions and 42 overall, which does not
add up; §E.1 explains the defect that produced it.

---

## B.1 Plan fidelity: task success is not plan identity

`34/60` counts trials whose plan is complete, independently validated
(`symbolic_validation: VALID`, `goal_status: GOAL_SATISFIED`) and confirmed by
ground truth at coverage 1.0. **None is a partial plan.** But satisfying the
goal is weaker than reproducing the reference action sequence, and the two are
reported separately here because only the second speaks to execution fidelity.

Structural comparison against `EXPECTED_GT_ACTIONS/` (operators must match in
order; non-object arguments must agree; object arguments must be used
consistently):

| | count |
| :--- | ---: |
| Successes | 34 |
| **Exact structural match to the GT action sequence** | **21** |
| Reach every GT goal by a different sequence | 13 |

By domain: **Living Room 10/10, Workshop 9/11, Kitchen 2/13.**

### B.1.1 The operator the compiled problem never emitted

The domain's action vocabulary distinguishes setting a utensil beside its bowl
from putting an object down on a surface: the ground-truth executor, the oracle
world state and the expected-action catalogue all name it
`PLACE_SERVING_UTENSIL`. The compiled symbolic problem emitted a generic
`PLACE`, so every Kitchen plan differed from the reference on those steps --
right in effect, wrong in name.

Running K5 through the **oracle** pipeline (`--mode gt`, no VLM) reproduced the
divergence exactly, which is what established that this was not a grounding or
FM-quality problem at all.

Preconditions and effects are unchanged and only the label differs, so planning
is identical: Kitchen task metrics are byte-identical before and after.

| ordering-free fidelity | before | after |
| :--- | ---: | ---: |
| exact GT action sequence | 21 / 34 | 21 / 34 |
| same steps, different order | 0 / 34 | **11 / 34** |
| **differs** | **13** | **2** |

Two of the three problems were in the comparison rather than the pipeline, and
are recorded here because either alone would have hidden the real defect:

* the serving destination is `serving_area` to the executor, oracle state and
  catalogue, and `dining_table` to the pipeline and the goal evaluator, which
  checks `at(cup, dining_table)`. One place, two names, disagreeing between the
  benchmark's own artifacts.
* the unordered check matched greedily, so a one-argument `PICK` could bind a
  body to the wrong instance and dead-end on a later `POUR`.

Neither flatters the pipeline: re-running the pre-fix replay through the
corrected comparison still gives 0/11 on Kitchen.

### B.1.1b The two that still differ

`W6/01` and `W6/02`: a valid driver existed in both `TOOL_CABINET` and
`LEFT_DRAWER`; grounding bound one and the reference the other. A legitimate
binding choice, not a wrong action.

### B.1.2 What was deliberately not done

A tie-break that reproduces the reference ordering would encode the expected
answer, which is the benchmark tuning the scientific constraints forbid, and it
would make the exact-match number meaningless.

The legitimate route to a higher exact-match rate is the **cost model**, stated
generically: penalise setting a held container down and picking it up again, so
"pour everything while holding it, then place" wins on cost rather than on an
arbitrary tie-break. That is a real domain preference expressible without
reference to the benchmark. It is **not** part of this freeze -- it changes
planning for every trial and needs the full regression gate -- and is recorded
here as the next candidate change.

---

## C. Determinism

**Outcome-level reproducibility is achieved. Pixel-level reproducibility is
not, and cannot be in this environment.** Five defects were found; the fifth is
the one that connected the renderer to the results. Detail in §C.1-C.5, with the
measured 96-row proof in §C.4.

### C.1 What was wrong, and what each fix bought

Three independent sources were found. Each was necessary; none was sufficient
alone, and the first two were both reported as "fixed" before the next was
discovered.

**1. GPU kernel autotuning.** No seed, `cudnn.benchmark` left on, no
deterministic-algorithm constraint, at detection thresholds as low as `0.001`.
Fixed in `mujoco_scenes/determinism.py`, requested **strictly** --
`torch.use_deterministic_algorithms(True)` without `warn_only`, because that mode
downgrades an unsupported operation to a warning and continues, which would mean
reporting determinism that was not achieved. On this build (torch 2.9.1+cu128)
strict mode is accepted and a full workshop trial runs through it with no
operation lacking a deterministic implementation.

**2. `PYTHONHASHSEED` unset.** Set-iteration order varied per process, so a choice
among equally ranked grounding candidates went different ways. Proved causal by
*varying* the seed rather than waiting for chance:

| `PYTHONHASHSEED` | `kitchen/K7/trial_01` |
| :--- | :--- |
| 1 | `coffee_container` unseated, 8-action plan |
| 2, 3 | `coffee_source` unseated, 20-action plan |

Those are exactly the two outcomes two concurrent replays had produced. It cannot
be set in process -- the interpreter reads it at startup -- so it is passed to
every spawned trial and recorded in the frozen identity.

**3. The seeded detector path was not the path taken.** `semantic_grounding.py`
constructs YOLO-World twice, and `MUJOCO_SEMANTIC_PROCESS_ISOLATION` defaults to
`"0"`, so the in-process constructor is what nearly every run uses. Only the
isolated worker had been seeded. A test now walks every function that builds a
detector and requires the seeding call to come first in that same function, so a
fifth construction site cannot be added without one.

**4. Confirmed evidence was discarded by a weaker observation.** See §C.3 -- this
is the one that actually moved outcomes.

**5. The two runners pinned different environments.** The replay driver pinned
`PYTHONHASHSEED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM`
and `MUJOCO_GL`; the live runner pinned none of them, so the experiment would
have run under a configuration no offline measurement was ever made under. Both
now take `REPRODUCIBLE_CHILD_ENV` from one dict, and the identity records every
entry.

### C.2 The residual, and why it is not closed

After all three fixes `kitchen/K7/trial_01` still flips. The cause was isolated
by elimination, with positive evidence at each step:

| stage | across runs |
| :--- | :--- |
| FM contract, requirement graph | identical (read from disk) |
| Geometry / tracking (`maximum_cross_section_m`) | identical to 17 significant digits |
| Detector on a **fixed** image, 3 runs | identical box / confidence / class signatures |
| Detections from the **rendered** scene | **differ** |

Concretely, `object_0007` resolves as `coffee canister` (SUPPORTED, 3 supporting
views) in one run and `UNKNOWN` (`CONFLICTING_MULTI_VIEW_LABELS`, 2 supporting
views, hypotheses `[coffee canister, cup]`) in another.

The detector is bitwise deterministic given fixed pixels. The scene state is
identical, since geometry matches exactly. Therefore the rendered frames differ:
**MuJoCo EGL GPU rendering is the residual source**, and a marginal detection
flips with it. `osmesa` CPU rendering is unavailable in this environment
(`AttributeError: 'NoneType' object has no attribute 'glGetError'`), so it cannot
be eliminated here.

### C.3 What actually stabilised it

The renderer jitter is real and unfixed, but it turned out not to be sufficient
on its own to move a trial's outcome. A fourth defect was doing that.

Precedence between an authoritative cached observation and a later, weaker
re-observation was written as `a.get(k) or b.get(k) or c.get(k)`. For a
container-valued field that is wrong: an observation validated with
`reason_codes == []` is reporting that it found **no problems**, and an empty
list is falsy, so the chain fell through to a later observation's complaints. A
belief that had been checked and confirmed could therefore be discarded by one
that had not -- and `SEMANTIC_LABEL_UNKNOWN` on the weak observation is a
lack-of-evidence code, which made the confirmed hypotheses vanish entirely.

`_first_declared` now takes the value from the first source that actually
declares the key. Applied to `reason_codes` and `alternatives` only: `status`
and `canonical_label` are scalars where `None` genuinely means absent, so
truthiness is correct there and was left alone.

With that fixed, a marginal confidence shift from the renderer no longer
propagates into a different outcome, because the retained belief survives it.

### C.4 Measured outcome-level stability

Three independent full replays at the final code, compared on `pipeline_status`,
`success`, `complete_grounding`, `gt_goal_coverage`, `outcome_correct`,
`false_completion`, `grounding_status`, `grounding_missing`, `plan_length`,
`astar_invocations`, `symbolic_goal_status` and `strict_v3_valid`:

```
rows per replay        96 / 96 / 96
HARNESS_FAILURE stubs   0 /  0 /  0
row-level differences   2   (95 of 96 rows bit-identical)
```

Both differences are on `kitchen/K7/trial_01`:

| field | A | B | C |
| :--- | :--- | :--- | :--- |
| `grounding_missing` | `['coffee_source']` | `['coffee_container']` | `['coffee_container']` |
| `plan_length` | 20 | 8 | 8 |

**No scored field differs.** `success`, `outcome_correct`, `false_completion`,
`complete_grounding`, `gt_goal_coverage`, `astar_invocations` and
`strict_v3_valid` are identical in all three, and all three replays report
exactly the same aggregate metrics. The trial reaches the same verdict by a
different internal route.

**The strict gate of zero differences is therefore not met.** The honest
statement is: 95 of 96 rows are bit-identical, one trial is path-bistable, and
no reported metric is affected. An earlier version of this report claimed
"96/96 identical, determinism achieved" on the strength of two replays agreeing;
that was a sampling error on a roughly even flip, and the third replay is what
exposed it.

### C.5 Consequence for the experiment

Renderer output is not bitwise reproducible and cannot be made so in this
environment. Three renders of one identical scene, same state and camera, give
three different images (mean pixel 166.008038194 / 166.007935113 /
166.008051215).

What matters for the experiment is whether that reaches the *outcome*, and with
evidence retention fixed it no longer does on the trial where it used to. The
honest statement is therefore narrow and empirical, not a guarantee: pixel-level
reproducibility is absent, outcome-level reproducibility is measured across
repeated full replays, and the measured figure is in §C.4. Any residual
instability is end-to-end -- FM sampling *and* perception -- not FM-only, and the
paper should say so.

---

## D. Failure attribution

Every remaining feasible failure, grouped by **earliest** causal stage. No
garbage-bin category.

All 26 failing feasible trials, by earliest causal stage. Every category-(A)
attribution was read directly from the archived raw contract, not inferred from
a downstream symptom.

| Stage | n | Trials | Detail |
| :--- | ---: | :--- | :--- |
| **FM generation** — truncated at the token limit | 3 | L2/02, L3/01, L3/02 | `finish_reason=length`, whole budget spent on reasoning, zero content emitted (§D.4) |
| **FM generation** — undeclared participant | 1 | W5/02 | `functional_relations[0]` references `target_assembly`, never declared in roles |
| **FM semantics** — no seating requirement expressed | 3 | L5/03, L6/01, L6/03 | placement expressed; no seat, chair, sofa or person anywhere in the contract |
| **FM semantics** — over-demanded instance counts | 2 | W2/03, K1/03 | two fasteners where the scene holds one; one DISTINCT spoon role of four merging a reusable stirrer with distinct utensils |
| **FM semantics** — wrong binding policy | 1 | K2/01 | `eating_utensil` declared SHARED, i.e. one utensil common to both diners |
| **FM semantics** — per-seat requirement collapsed | 1 | L1/02 | `seating_area` and `surface` both count 1 SHARED for two seats |
| **FM semantics** — seat declared as a carried object | 1 | L3/03 | `chair` declared `entity_kind: OBJECT`, the carried class (§D.2) |
| **Robot capability** — not owned | 1 | K3/03 | stove `FIXED_TARGET` with a heating relation, and handing a bowl to a person |
| **Symbolic goal compilation** | 1 | W1/02 | `repaired` goal retained after its `SCREW` operator was filtered (§D.1) |
| **Grounding** | 1 | W4/01 | `repair_target` unseated, `FUNCTIONAL_ASSIGNMENT_FAILURE` |
| **Geometry** — fastener measured from a 32-point cloud | 3 | W3/01–03 | 5.66 cm against a 3.98 cm measurement of the identical asset (§D.3) |
| **Perception** — no qualifying candidate | 1 | W5/01 | `fastener` plausibility `true=0, plausible=0, unknown=0` |
| **Perception** — label conflicted or absent | 6 | K4/02, K5/01, K4/03, W4/02, W4/03, W5/03 | two-way label conflict or no label at all |
| **Conservative non-declaration** | 1 | K3/01 | GT coverage 1.0, run reports `PARTIAL_ACTION_SEQUENCE_READY` and does not claim completion |
| **Total** | **26** | | |

By domain: Kitchen 7, Living Room 8, Workshop 11. Every trial is attributed; no
garbage-bin category.

Note on K3/01: independent evaluation confirms all four goals, but the run
declined to announce completion, so it is scored a failure. That is the
conservative direction, and it is the trial that exposed the feasible-correctness
defect in §E.1.

### D.1 The three previously unclassified Workshop trials

**W1/02 — symbolic goal compilation.** Not FM, not perception, not planner
search. `domains/workshop.py:189-194` emits the `SCREW` operator and the goal
`("repaired", target)` together; `:295` then filters the operator out when
`operation_ok` is false and **leaves the goal in place**, so the goal set contains
an atom no instantiated action can establish. A\* correctly returns a partial
plan. Evidence:

| trial | fastener binding | actions emitted | `repaired` achievable |
| :--- | :--- | :--- | :--- |
| W1/01 | DISTINCT | PICK, PLACE, PICK, **SCREW**, PLACE | yes → success |
| W1/03 | DISTINCT | PICK, PLACE, PICK, **SCREW**, PLACE | yes → success |
| W1/02 | SHARED | PICK, PLACE, PICK, PLACE, PLACE | **no** → partial |

The duplicate `PLACE` in W1/02 is the `if driver and not operation_ok` fallback,
which confirms the gate fired.

**The goal is deliberately not dropped to match.** Without `("repaired", target)`
the remaining goals are satisfiable, so removing it would convert an honest
partial plan into a reported completion of an unrepaired joint -- a false
completion. The current behaviour is safe; only its *reporting* was poor, and
that is what the diagnostic added in this pass fixes.

**W4/01 — grounding.** `repair_target` (FIXED_TARGET, raw `marked_fastening_site`)
is not seated: `PARTIAL_VERIFIED_GROUNDING`, `FUNCTIONAL_ASSIGNMENT_FAILURE`,
`missing_requirements = ['repair_target']`.

**W5/01 — perception.** `fastener` role plausibility is `true=0, plausible=0,
unknown=0`: no detected object qualifies as a fastener at all, so grounding is
`INFEASIBLE` before any assignment is attempted.

### D.2 L3/03 — closed as unresolved, deliberately

§9 permits one principled attempt or an explicit close. The rule, stated without
reference to the benchmark: *a role declared `entity_kind: OBJECT` whose candidate
categories name only non-carryable furniture may satisfy a place-like role
requirement.* The asymmetry is real -- a succeeding draw declares its tables as
`OBJECT` and is tolerated, while a failing one declares a seat as `OBJECT` and is
not.

It was **not** attempted. Three earlier entity-kind iterations in this project
each regressed working trials and were reverted, and validating a fourth needs a
targeted replay plus a full 96 replay plus the Living Room sentinels -- a whole
measurement cycle. Two reproducibility defects found in this pass matter more to
the result than one additional trial, and §22 asks for a defensible frozen system
rather than the largest success count.

### D.3 W3 — closed, no change

§10 marks this optional and says not to delay the freeze for it. The fastener is
measured from a 32-point cloud where the identical asset measures 106 points
elsewhere, inflating its principal extent from 3.98 cm to 5.66 cm past the joint
depth plus allowance. The obvious gate -- require the cloud to support the
percentile it claims, i.e. `n >= 100/lower_percentile = 100` -- puts five
currently-passing variants 6 points from being reclassified, and any looser
threshold is chosen to catch this variant and nothing else. That is the
benchmark tuning §2 forbids.

### D.5 The prompt change nothing could have caught

The live smoke produced a false completion on `kitchen/K2`:
`ACTION_SEQUENCE_READY` at GT coverage 0.5. The run satisfied every requirement
its contract stated; the contract never asked for the soup to be put in the
bowls.

Under the OPERATIONS wording the 96 archived responses were collected with, the
model emits `Place soup_ingredient into soup_bowl`. Sentences added afterwards
-- name each operation as a single concrete physical motion, split a
move-then-work step into two -- change that to `Place soup bowl on table` with no
filling step at all.

Those sentences landed **after** the collection, as a secondary change in a
commit about relations, and **no measurement since could have detected them**:
every one has been the frozen replay, which feeds archived responses through the
pipeline and never reads the prompt. The change shipped with no evidence, and
the first live run after it is the evidence.

Reverted, and the same variant then produces a 26-action plan that satisfies
ground truth. Only the OPERATIONS sentences are reverted: the ROLES paragraph
also differs from the collection, but deliberately -- `required_count` counts
occasions rather than physical instances and the compiler's binding-policy
handling is built on it. `test_prompt_drift.py` pins both halves, so neither a
reintroduction nor a blanket revert can happen quietly.

### D.4 Token truncation — measured, and not what it looked like

All five unparseable archived calls have `finish_reason = length`,
`completion_tokens = 24000`, `reasoning_tokens = 24000`, `content chars = 0`:
the thinking phase consumed the entire budget before emitting any JSON.

Re-issuing the same five inputs fresh recovers **5/5 at 28000 and 5/5 at 24000**,
consuming 14699/6132/12041/7868/6596 and 12211/8574/9732/8990/5632 tokens
respectively -- every one below the original ceiling.

**The ceiling was never the binding constraint.** Truncation is a stochastic
runaway, not a per-variant token requirement, and raising the ceiling does not
reduce its rate. The ceiling is nevertheless raised to the maximum the context
allows, because the observed convergent tail (14699) already exceeds the archived
maximum (12222), so headroom is justified -- but it is insurance, not a fix.

---

## E. Scientific-integrity audit

### E.1 Four disagreeing metric definitions, now one

`mujoco_scenes/evaluation_outcome.py` is the single authority. Each refuted
definition is pinned as a test so it cannot return.

| # | where | what it did | why it was wrong |
| :--- | :--- | :--- | :--- |
| 1 | frozen-replay scorer | infeasible correct `= not gt_full_task_satisfied` | true by construction -- an impossible task cannot be satisfied, so it credited all 36 infeasible trials and measured nothing |
| 2 | held-out matrix evaluator | credited `NO_MEANINGFUL_CANDIDATE_PLAN`, omitted `INFEASIBLE` | a partial plan scored right and a real conclusion scored wrong; held-out and main numbers were never comparable |
| 3 | live + held-out evaluators | false completion `= not feasible and full_task_satisfied` | requires ground truth to certify an impossible task as done, so it could never fire and missed the adversarial case entirely |
| 4 | feasible correctness | scored from GT satisfaction alone | a run stopping at `PARTIAL_ACTION_SEQUENCE_READY` whose artifacts happened to satisfy every goal scored correct without delivering a plan; this is the 34 + 7 = 42 mismatch |

The definitions now in force:

```
completion_claimed(status)     := status == "ACTION_SEQUENCE_READY"
false_completion               := completion_claimed and not gt_full_task_satisfied
outcome_correct  (feasible)    := completion_claimed and gt_full_task_satisfied
outcome_correct  (infeasible)  := not completion_claimed
                                  and not false_completion
                                  and status in INFEASIBILITY_CONCLUDED_STATUSES
```

`false_completion` is deliberately **not** conditioned on feasibility: announcing
a finished plan for an unfinished task is the same defect either way, and on an
infeasible variant it is the adversarial case by construction.

Five status names carried by the old whitelists (`NO_VALID_GROUNDING`,
`TASK_REJECTED_UNSUPPORTED`, `NO_SEARCH_REGIONS_DECLARED`, and others) are emitted
by no code path. Those that are genuine conclusions are retained for future paths;
`NO_MEANINGFUL_CANDIDATE_PLAN`, which describes a partial result, is not.

### E.2 Runtime / evaluation separation

Enforced statically by `scripts/audit_no_gt_leakage.py`, which now also fails on
an online module importing an evaluation-only module
(`evaluation_outcome`, `evaluation_metrics`, `gf_reference_evaluation`,
`gf_reference_evaluator`, `gt_spec_provider`, `reference_graphs`).

One allowance exists, named rather than hidden: `spec_provider.py` may import
`gt_spec_provider`, because the benchmark has a ground-truth/oracle arm as a
deliberate baseline and the dispatcher must be able to construct it. The import
is lazy and guarded by `if mode == "gt"`, so the VLM path never executes it. Tests
assert the allowance set contains exactly that one entry and that the import stays
lazy.

`raw_semantic_evaluation.py` is exempt by name: it scores the FM's raw output
after the fact and is reached only from `evaluation_metrics.py` and offline
scripts.

The scorer itself was initially placed inside `functional_tamp_pipeline`, where
the audit caught it immediately and correctly -- it takes `gt_feasible`, and no
module in that package may reference ground truth. It now lives outside.

**A rule that cannot fail is not a rule.** A test plants an online module that
imports the scorer and asserts the audit reports it, so `FINDINGS: 0` means
something.

```
$ python scripts/audit_no_gt_leakage.py
FINDINGS: 0
```

### E.3 Prohibitions

| prohibition | status |
| :--- | :--- |
| variant-conditioned runtime branches | none; audit enforces |
| expected-answer maps | none |
| scene-inventory count relaxation | none -- W2/03 fails rather than relaxing a required count of 2 to the 1 the scene holds |
| UNKNOWN treated as TRUE | none -- K4/02 and K5/01 fail rather than selecting the task-convenient reading of a conflicted label |
| false completions | see §B |
| silent deletion of required constraints | none -- W1/02's unreachable goal is deliberately retained rather than dropped (§D.1) |

---

## F. Tests

The **complete** relevant surface, not a subdirectory. An earlier report called
`functional_tamp_pipeline/tests` alone the full suite; it is not.

```bash
python -m pytest mujoco_scenes/functional_tamp_pipeline/tests mujoco_scenes/tests -q
python scripts/audit_no_gt_leakage.py      # must print FINDINGS: 0
```

```
2239 passed, 49 failed, 12 errors, 3 skipped
FINDINGS: 0
```

49 failed and 12 errors are pre-existing; §F.1 triages every group. Failures
went 52 -> 49 across this pass, exactly the three stale Workshop assertions
fixed, and passes went 2203 -> 2221, exactly the tests added. **No regressions.**

No warning in any class indicating a nondeterministic operation, an algorithm
fallback, NaN, invalid geometry, overflow, or precision loss.

### F.1 Pre-existing failures, triaged

These fail at the pushed baseline `f83aa662` as well. `git diff --name-only
f83aa662..HEAD` showed **no production module changed** by the earlier closure
work -- only reports, the evaluation-only scorer, tests and scripts -- so none was
introduced by it.

| group | n | cause |
| :--- | ---: | :--- |
| `test_kitchen_phase_c_execution` | 12 err | missing historical run artifact `runs/integrated_no_pot_clearance_seed19_20260807/` -- fixture, not code |
| `test_phase3_6a7*_contract` | 10 | validation rejects earlier (unauthorized role) than the assertion expects |
| `test_phase3_6b0_reference_evaluator` | 7 | reference-evaluator expectations predate the count/binding work |
| `test_s1_integrated_kitchen` | 3 | asserts the superseded canonical kitchen instruction string |
| `test_workshop.py` | 2 | asserts exactly 10 variants; the tree has 15 (10 benchmark + 5 held-out). Held-out only, outside the 32-variant experiment |

Three Workshop perception assertions **were** fixed in this pass, because they
guard invariants this experiment depends on:

- the drive-slot window asserted 6-15 mm, which is screw-head diameter, not groove
  width. Commit `5cc21285` had deliberately corrected the config to 0-3.1 mm after
  finding that every Phillips head whose cloud segmented slightly elongated was
  classified `SLOT_LIKE`, at which point the verifier correctly refused a
  cross-drive screw for a cross-drive driver. The test encoded the refuted
  calibration.
- two semantic-grounding assertions expected `FAIL` where the policy now returns
  `UNKNOWN`. A label a requirement does not accept is an unresolved candidate, not
  a proven incompatibility.

---

## G. Final FM sampler

Frozen. Every field is recorded in the identity, and the runner aborts if any of
it differs from what an existing output root was started with.

```
{
  "TAMP_FM_ENABLE_THINKING": "true",
  "TAMP_FM_MAX_TOKENS": "28000",
  "TAMP_FM_PRESENCE_PENALTY": "1.0",
  "TAMP_FM_REPETITION_PENALTY": "1.0",
  "TAMP_FM_SCHEMA_VERSION": "3",
  "TAMP_FM_TEMPERATURE": "0.6",
  "TAMP_FM_TOP_K": "20",
  "TAMP_FM_TOP_P": "0.95",
  "TAMP_FM_VIEWS": "3"
}
```

Thinking is **on** and temperature is **0.6**: the repetitions exist to measure
sampling variance, and a greedy sampler would make ten repetitions ten copies of
one draw.

The *sampler* matches the archived collection; `max_tokens` is deliberately
higher (§D.4). **The semantic contract does not match**, and that is what
governs comparability: the prompt and schema differ from the collection
(§I.1), so the archived 34/60 measures the current downstream pipeline on
archived FM responses. It is a regression and development metric, **not a
forecast of final live performance.**

---

## H. Final experiment command

```bash
# tunnel to the vLLM endpoint
ssh -i ~/keyfile -N -L 8000:127.0.0.1:8000 long-horizon@gvlab2.iiit.ac.in &

PYTHONPATH=. python scripts/run_live_repeat_experiment.py \
  --output-root benchmark_reports/live_experiment_10x32 \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen35-9b \
  --run-type full \
  --repeats 10

PYTHONPATH=. python scripts/aggregate_live_repeats.py \
  --root benchmark_reports/live_experiment_10x32
```

Preflight refuses to start on a dirty tree, on a model the endpoint does not
serve, or into a root whose recorded identity differs in any field. Each
repetition writes an immutable `attempt_NN/`; a failed one is preserved and
reported, never silently retried. The aggregator scores only the attempt the
repetition's manifest names as authoritative, and prints the missing cells of the
32-variant grid separately from semantic failures.


---

## I. Live smoke at the exact frozen configuration

Twelve variants covering every domain, the W2-W5 fastener cases, W7 search, the
W8 sentinel and infeasible W10. One repetition, fresh FM output, the frozen
sampler and token ceiling. An execution test, not a performance measurement.

| | first smoke | after prompt revert | after this pass |
| :--- | ---: | ---: | ---: |
| outcome correct | 0 / 12 | 2 / 12 | **4 / 12** |
| GT-satisfied plans | 2 | 4 | 5 |
| **false completions** | **1** | **0** | **0** |

`W10`, the infeasible sentinel, now concludes `EXHAUSTED_NO_VALID_GROUNDING`
live -- end-to-end confirmation of the change in §K. `W5` reaches the same
conclusion. Variation between the last two columns on individual feasible trials
is FM sampling at temperature 0.6; per §14 the smoke is not tuned on.

### I.1 Two live behaviours worth stating

**The symbolic goal is stricter than ground truth.** `K2` and `W8` produce plans
that satisfy every GT goal while the pipeline reports
`PARTIAL_ACTION_SEQUENCE_READY`. Five plans satisfy GT; four are claimed. The
system under-reports its own success and never over-reports it.

**Live is not predicted by the archived replay.** See the note at the top.

---

## J. Performance changes accepted in this pass

**One change, of four investigated.**

### J.1 Accepted -- an exhausted search over a complete contract is a conclusion

*Root cause.* Infeasibility was never validly concluded on any archived
infeasible trial. 17 of 36 had a complete executable contract and a grounding
search that ran to exhaustion without a valid complete assignment, and all
reported `PARTIAL_ACTION_SEQUENCE_READY` -- the same status as a run that merely
stopped early.

*Generic rule.* A run that stated the task fully, enumerated every candidate it
could observe and rejected all of them has proven, over its own contract and
observations, that no assignment exists. All three conditions are required; the
branch sits below the full-plan branch so it cannot displace a success, and it
claims no completion so it cannot create a false completion. One helper, called
by all three domains -- the first implementation went into `run.py` only and
silently did nothing for kitchen and living room.

*Tests.* 8 in `test_exhaustion_proof.py`, including ordering in all three domain
sources and that a feasible variant never becomes correct by concluding
exhaustion.

*Measured.*

| | before | after |
| :--- | ---: | ---: |
| feasible success | 34 / 60 | **34 / 60** |
| infeasible outcome correct | 0 / 36 | **16 / 36** |
| overall outcome correct | 34 / 96 | **50 / 96** |
| false completions | 0 | **0** |
| complete groundings | 34 | 34 |
| planner failures after complete grounding | 0 | 0 |
| mean GT coverage | 0.6625 | 0.6625 |

*Regressions.* None.

### J.2 Investigated and rejected

**Conservative under-reporting.** Only K3/01 offline, and its symbolic problem
has **no unreachable predicate** -- every goal atom has a producer. Legitimately
stricter than GT (§6 Case A), not a spurious compiler goal.

**W1/02 operation/goal consistency.** The raw FM contract declares the fastener
`binding_policy: SHARED` itself. Per §7 an FM semantic failure, not a compiler
artifact.

**Perception conflict retention.** A tree-wide AST scan for container-valued
precedence chains returns **zero** remaining occurrences; the earlier
`_first_declared` fix covered them all.

**W3 geometry and L3/03 entity-kind.** Left closed per §10.

---

## K. GO / NO-GO

```
release tag   fm-tamp-final-experiment-v1
branch        vlm-testing-pipeline
resolve with  git rev-parse fm-tamp-final-experiment-v1
```

The commit hash is deliberately not written here. A document inside the release
cannot name the commit that contains it -- the same self-reference that made
`METHOD_FREEZE.json` perpetually stale (§A). The tag resolves it from outside.


| Gate | Result |
| :--- | :--- |
| 96/96 records, 0 harness failures | pass (x3 replays) |
| 0 semantic FM calls in replay | pass |
| 0 planner failures after complete grounding | pass |
| **0 false completions** | pass (archived and live) |
| 0 GT-goal mismatch successes | pass |
| 0 leakage findings | pass |
| No new test failures | pass |
| Three replays row-identical | **95/96** -- one path-bistable trial, no metric affected |

Four things the reader must be told, none of which blocks the run:

1. **Expect a lower live rate than 34/60.** That figure is archived responses
   through the current pipeline, under a prompt the live run will not use.
2. **One trial of 96 is path-bistable** (§C.4); no reported metric moves with it.
   Residual instability is end-to-end -- FM sampling *and* perception.
3. **The system under-reports success** (§I.1), never over-reports it.
5. **Task success is not plan identity.** 21 of the 34 successes reproduce the
   reference action sequence exactly and 32 of 34 produce the same actions in
   some order (§B.1); subgoals are independent, so ordering is free. Only 2
   differ, both a legitimate binding choice. If execution fidelity is the
   objective, report 32/34 alongside 34/60.
4. **Infeasibility is now concluded 16/36**, up from 0/36, and only where a
   complete contract and an exhausted search justify it.

FINAL 10x32: GO
