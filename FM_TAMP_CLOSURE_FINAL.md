# FM-Guided Functional TAMP — Final Closure Report

Frozen-replay evidence at `HEAD`, measured with **zero FM calls** over the archived
3×32 V3 response distribution
(`benchmark_reports/v3_qwen_distribution_3x32_20260910T053937`, 96 raw contracts,
model `qwen35-9b`). Every number below is read from the evaluator's own per-trial
records; none is hand-entered.

---

## 1. Headline results

| Metric | Value |
| :--- | :--- |
| Feasible-trial full-task success | **34 / 60 (56.7%)** |
| — Kitchen | 11 / 18 (61.1%) |
| — Living Room | 10 / 18 (55.6%) |
| — Workshop | 13 / 24 (54.2%) |
| Outcome correct (all 96 trials) | **42 / 96 (43.8%)** |
| — feasible: task finished | 34 / 60 |
| — infeasible: infeasibility *concluded* | **7 / 36 (19.4%)** |
| Infeasible trials that avoided claiming completion | 36 / 36 (100%) |
| **False completions** | **0 / 96** |
| Complete grounding (feasible) | 34 / 60 |
| Planner failures after complete grounding | **0** |
| A\* invocations per trial | **≤ 1** (values observed: {0, 1}) |
| Semantic FM calls during replay | **0** (values observed: {0}) |
| Strict V3-valid raw contracts | 90 / 96 |
| Executable contract complete | 60 / 96 |
| Mean GT goal coverage (feasible) | 0.6625 |
| Longest plan | 26 actions |
| GT-leakage audit findings | **0** |

Two structural invariants hold without exception: **complete grounding implies a
valid plan** (34 complete groundings, 34 successes, 0 planner failures), and **no
infeasible variant is ever reported complete** (0 false completions in 96 trials).
The system's failures are all failures to *reach* a grounded goal, never failures
to *notice* that it has not.

**The outcome-correct figure in this report is 43.8%, not the 74.0% that earlier
reports carried.** The earlier number was produced by a scoring tautology that was
found and removed in this pass; see §4.4. Nothing about the system's behaviour
changed — only the honesty of the metric. Feasible-trial success, goal coverage
and the false-completion count are unaffected.

One further nuance, stated because it cuts against the reported number: **K3/01**
has GT goal coverage 1.0 and its plan independently satisfies the full task, but
the pipeline reports `PARTIAL_ACTION_SEQUENCE_READY` and declines to declare
completion. It is counted as a failure. So 34 trials are declared successes while
35 produce plans that independently satisfy the task — the system is conservative
here, not wrong.

---

## 2. Per-variant results

| Domain | Variant | GT feasible | Success /3 | Mean GT coverage | Complete grounding /3 | Outcome correct /3 | False completions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| kitchen | K1 | yes | 2/3 | 0.833 | 2/3 | 2/3 | 0 |
| kitchen | K2 | yes | 2/3 | 0.917 | 2/3 | 2/3 | 0 |
| kitchen | K3 | yes | 1/3 | 0.667 | 1/3 | 2/3 | 0 |
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

- kitchen: 11/18
- living_room: 10/18
- workshop: 13/24
- outcome correct overall: 42/96

---

## 3. Where the remaining 26 feasible failures actually die

Eighteen of the 26 were traced to a specific mechanism with instrumented
evidence. Eight were not, and are reported as open rather than dressed up.

| Cause | Trials | Category |
| :--- | :---: | :--- |
| Raw contract cut off at the token limit (`finish_reason='length'`) | 3 | **(A)**, budget |
| FM expressed placement but no seating requirement | 3 | **(A)** |
| FM referenced a participant it never declared (W5/02) | 1 | **(A)** |
| FM over-demand: 2 fasteners where the scene holds 1 (W2/03) | 1 | **(A)** |
| Fastener measured from a 32-point cloud (W3 ×3) | 3 | **(D)** |
| Object label conflicted or absent (K4/02, K5/01, K4/03, W4/02, W4/03, W5/03) | 6 | **(D)** |
| Plan satisfies the task but completion is not declared (K3/01) | 1 | conservative |
| Draw-dependent compile/ground loss, not isolated to one mechanism | 8 | open |
| **Total** | **26** | |

By domain: Kitchen 7, Living Room 8, Workshop 11.

The eight "open" trials are the honest residual. They split two ways:

- **K1/03, K2/01, K3/03, L1/02, W1/02** fail on a draw whose sibling trials of the
  *same variant* succeed. Same scene, same perception, same code — so the variance
  is in the contract the FM produced. L1/02 is the clearest case: it fails to seat
  `SEATING_POSITION` on the identical scene where L1/01 grounds COMPLETE.
- **L3/03, W4/01, W5/01** belong to variants that score 0/3, so no sibling proves
  the variant is reachable at all. These are the least understood trials in the
  benchmark.

None was reduced to a shared mechanism in this pass, and claiming otherwise would
be guessing.

### 3.1 W3 ×3 — the fastener is measured wrong, and the verifier is right

This is the most precisely characterised remaining failure, and it is worth
stating in full because it looks like a threshold problem and is not.

`COMPATIBLE_WITH_TARGET(fastener, repair_target)` returns FALSE in all three W3
trials, which prunes every joint assignment (`NO_GLOBAL_ASSIGNMENT`).

The workshop scenes all place the *same* screw asset,
`workshop_medium_phillips_screw` (a single constant in the GT planner). Its
measured principal extent, however, is not the same:

| Variant | Region | Points in fused cloud | `total_length_m` | Verdict |
| :--- | :--- | ---: | ---: | :--- |
| W1, W2, W9 | LEFT_DRAWER | 106 | 0.0398 | TRUE |
| W6 | — | — | 0.0443 | TRUE |
| W7, W8 | — | — | 0.0381 | TRUE |
| **W3** | **TOOL_CABINET** | **32** | **0.0566** | **FALSE** |

The joint admits `depth + excess = 0.0379 + 0.013 = 0.0509 m`. The consistent
observations (3.8–4.4 cm) pass with ~1 cm to spare; W3's 5.66 cm reading fails by
5.7 mm. W3's minor extent also collapses (0.0026 vs 0.0067) while its maximum
cross-section inflates (0.0182 vs 0.0128) — the signature of a sparse
grazing-angle sliver, not of a longer screw.

`total_length_m` is *already* a robust measurement: the 1st–99th percentile span
along the principal axis (`geometric_grounding.py:192`). At 32 points, 1% of the
cloud is less than one point, so the percentile trims nothing and the "robust"
extent degenerates to the raw min–max.

**Why no patch was shipped.** The principled gate — require the cloud to be dense
enough for the percentile it claims, i.e. `n ≥ 100/lower_percentile = 100` — is
derived rather than hand-picked, but the *passing* observations sit at 106 points.
A threshold at 100 would put five currently-succeeding variants 6 points away from
being re-classified UNKNOWN, risking −6 successes for +0. Any looser threshold
(e.g. 40) is chosen to catch W3 and nothing else, which is variant-targeting by
proxy and forbidden. Reporting UNKNOWN instead of FALSE would be more honest but
gains nothing: UNKNOWN is never TRUE, so the trial still fails.

W3 is therefore **category (D)**: recovering it requires *denser observation of
the tool cabinet* — a capture/viewpoint change — not a semantic, threshold, or
detector change.

### 3.2 K4/02 and K5/01 — a genuine two-way label conflict

Both fail needing two DISTINCT `coffee_container`s. The scene offers one confirmed
`cup` plus one object the detector splits evenly:

```
object_0001  status UNKNOWN  reason CONFLICTING_MULTI_VIEW_LABELS
             hypotheses ["cup", "coffee canister"]
             cup:            2 views, mean conf 0.645, score 4.68
             coffee canister: 2 views, mean conf 0.636, score 4.06
```

Grounding's rule is `S_sem(o,r) = TRUE` iff *every* hypothesis is accepted by the
role. `cup` is accepted by `coffee_container`; `coffee canister` is not; the
verdict is UNKNOWN and the candidate is unusable.

Admitting it would require selecting the task-convenient reading of an ambiguous
observation. That is exactly the "treat UNKNOWN as TRUE" prohibition, and it is
the mechanism by which an infeasible variant could be made to look feasible. It
was **not** implemented. **Category (D).**

### 3.3 Living Room ×8 — traced draw by draw

This was the least-understood block. Reading the archived raw contracts against
the compiler's own gate resolves all eight.

The Living Room gate requires *two* expressed things for a personal placement to
count: a fit relation (`FITS_SET_ON(PERSONAL_CUP_SAUCER_REGION, CUP_SAUCER_SET)`)
**and** a seat-proximity relation (`NEAR_SEAT` involving `SEATING_POSITION` or
`SEATING_PAIR`) — because the task is to place a setting *beside a seat*, not
merely on a surface.

| Trial | What the FM actually emitted | Category |
| :--- | :--- | :--- |
| L2/02 | raw JSON invalid — `finish_reason='length'`, **truncated at the token limit** | (A) |
| L3/01 | raw JSON invalid — **truncated at the token limit** | (A) |
| L3/02 | raw JSON invalid — **truncated at the token limit** | (A) |
| L5/03 | placement expressed (`spatial_support`), **no seating term anywhere in the contract** | (A) |
| L6/01 | placement expressed (`is located on` ×2), **no seating term anywhere in the contract** | (A) |
| L6/03 | seating terms present, but no relation pairing a setting with a seat | (A) |
| L1/02 | seating expressed (`nearby`, `accessible_to`) — fails at grounding, `SEATING_POSITION` unseated on the **same scene where L1/01 grounds COMPLETE** | draw-dependent |
| L3/03 | seating expressed (`on`, `near`, `accessible_to`) — contract incomplete, coverage 0.667 | draw-dependent |

Concretely, for L5/03 the FM declared exactly four roles — cup/mug/glass,
plate/bowl, remote, table — and three relations, none of which mentions a seat,
chair, sofa, or person. Its placement demand is real and is read correctly; the
*seat* half of the requirement was never expressed. Supplying it would create a
task relation the FM did not express, which is the cardinal prohibition. The same
holds for L6/01.

**Three of the eight are pure token-limit truncations**, not semantic failures.
They are the strongest argument for a larger `max_tokens` in a future collection —
see §7 for why that budget was *not* changed here.

### 3.3b The FM's semantic generation is not the bottleneck

Auditing all 96 archived raw responses directly:

| Raw-contract outcome | Count |
| :--- | ---: |
| Parsed and strict-V3 valid | 90 / 96 |
| **Cut off at the token limit** (`finish_reason='length'`) | **5 / 96** |
| Genuine semantic error | **1 / 96** |

Every one of the five invalid-JSON trials — K12/03, L2/02, L3/01, L3/02, L9/02 —
has `finish_reason='length'`. They present as `'schema_version' is a required
property` only because the JSON was truncated before reaching that key. Two of the
five (K12, L9) are infeasible variants and were still correctly rejected, so they
cost nothing; the other three are lost feasible successes.

The single genuine semantic error in the whole distribution is **W5/02**:
`UNDECLARED_PARTICIPANT: functional_relations[0] references ['target_assembly']` —
the FM named a participant in a required relation but omitted it from the roles
list. Materialising a relation-referenced role is arguably within the
"canonicalize an FM-expressed semantic" allowance, since the relation *is* the
expression; it was not implemented because it changes structural repair for all
96 contracts to gain one trial, and a hallucinated role name would then become a
spurious required role. Recorded as a boundary, not shipped.

So the FM's semantic generation failure rate on this benchmark is **1%**, and its
truncation rate is **5%**. The bottleneck is downstream of generation.

### 3.4 W2/03 — FM over-demand

The screw is now detected, tracked (1.28 cm) and accepted. The contract demands
**two** fasteners where the scene contains one. Satisfying it would require
relaxing an instruction-derived count against scene inventory — explicitly
forbidden. **Category (A).**

---

## 4. What was fixed in this pass

Five defects. Four are in the experiment harness rather than the pipeline; two of
those would have invalidated the live experiment outright, and one was silently
inflating a headline number.

### 4.1 A repetition could be scored from a mixture of attempts

`aggregate_live_repeats.py` read *every* `attempt_*` directory of a repetition and
let the newest win per `(domain, variant)`. A variant absent from the newest
attempt silently inherited an older attempt's outcome, so a repetition could be
scored from a composite run that never existed. The runner already names its
`authoritative_attempt` in each repetition's manifest; only that attempt is scored
now, and a variant missing from it is a hole in the grid. A named attempt that is
absent scores nothing rather than something else.

### 4.2 The identity freeze recorded less than it decided

`frozen_identity.json` now carries the endpoint, the **served model's revision**
(a hash of the endpoint's own record, so the same model name over different
weights is refused rather than pooled), every sampler value, the variant set, and
the run type. `--run-type` is required, so a smoke run and the reported experiment
can no longer be pooled by sharing an output root. The repeat count is recorded as
a *history* of what was requested, because an experiment extended after seeing its
numbers is a different claim from one planned.

### 4.3 The live sampler was not the sampler the numbers were measured on

Found only because §4.2 started recording the sampler. The runner sent:

| | live runner (before) | archived V3 distribution |
| :--- | :--- | :--- |
| thinking | off | **on** |
| temperature | 0.0 | **0.6** |
| top_p | 1.0 | **0.95** |
| top_k | −1 | **20** |
| presence_penalty | 0.0 | **1.0** |

Every offline number for this pipeline — including the 34/60 above — was measured
on the archived distribution. A live run under the greedy sampler would have been
a different experiment wearing these numbers: any drop would have been
unattributable between the pipeline and the sampler. Worse, **ten repetitions of a
greedy sampler are ten copies of one draw**, which is not a variance measurement,
so the entire 10× design would have produced no confidence intervals worth
reporting.

The sampler is now pinned to the archived manifest's own `model_config`, with a
test that fails if the two ever diverge again.

### 4.4 A headline metric was a tautology

The most consequential finding of this pass, and it lowers the reported numbers.

Two scorers disagreed about `outcome_correct` on an infeasible variant:

- the **live evaluator** required the pipeline to *reach an infeasibility
  conclusion* — a status in `{INFEASIBLE, EXHAUSTED_NO_VALID_GROUNDING,
  NO_VALID_COMPLETE_ASSIGNMENT, PLANNING_PROVEN_INFEASIBLE}`;
- the **frozen-replay scorer** asked only `not gt_full_task_satisfied`.

On an infeasible variant the task cannot be satisfied, so the replay's condition
is true **by construction**. It credited all 36 infeasible trials automatically
and measured nothing at all. That tautology is where the previously reported
"36/36 infeasible correctly rejected (100%)" and the 74.0% overall figure came
from.

The discrepancy surfaced only because the live smoke (§5) scored K7 and L7 as
incorrect while the replay scored the same variants 3/3 — two scorers, same
behaviour, opposite verdicts.

Under the real rule, only **7 of 36** infeasible trials reach an infeasibility
conclusion. The other 29 stop at `PARTIAL_ACTION_SEQUENCE_READY` (17),
`NO_MEANINGFUL_CANDIDATE_PLAN` (10) or `VLM_SPEC_FAILED` (2): they correctly
decline to claim completion, but they do not conclude that the task is
impossible. Those are genuinely different achievements, and they are now reported
separately — the safety property as the false-completion count (0/96, unchanged),
the competence property as outcome-correct (7/36).

There turned out to be a **third** definition, in the held-out matrix evaluator,
broken in the opposite direction: it credited `NO_MEANINGFUL_CANDIDATE_PLAN` — a
partial plan — while omitting `INFEASIBLE` entirely, so an actual infeasibility
conclusion scored as *wrong* there and a partial plan scored as *right*. Held-out
and main-benchmark numbers were therefore never comparable.

Enumerating what the pipeline actually emits also shows that five of the statuses
named across those whitelists — `NO_VALID_GROUNDING`,
`NO_VALID_COMPLETE_ASSIGNMENT`, `PLANNING_PROVEN_INFEASIBLE`,
`TASK_REJECTED_UNSUPPORTED`, `NO_SEARCH_REGIONS_DECLARED` — are **never produced
by any code path**. In practice the rule reduces to `{INFEASIBLE,
EXHAUSTED_NO_VALID_GROUNDING}`, and all 7 credited trials are `INFEASIBLE`. The
dead names are retained in the canonical set so a future path that does emit them
is scored correctly, but they were carrying no weight.

All three scorers now call one function,
`outcome_classifier.outcome_is_correct`, and tests assert that none keeps a
private copy of the status list and that a partial plan is never credited as a
conclusion.

**This is a −29-trial correction to a headline number with no change to the
system.** It is reported because a 74% that cannot be reproduced by the live
harness is worse than a defensible 43.8%.

### 4.5 Semantic alternatives were deleted before they could be used

(Committed earlier this pass, `f83aa662`.) The association stage applied a
size-based label prior against the *raw proposal cloud*, which measured 1–3 cm
fasteners at 13–27 cm, and deleted every screw hypothesis "as physically
impossible" on a measurement wrong by a factor of three. An earlier stage may rank
hypotheses; it may not delete them. The same constraint is still applied by the
tracker against the *fused* cloud, where it is correct. Recovered W2/01 and W2/02.

---

## 5. Live-path validation

The full live path was exercised end to end against the vLLM endpoint (`qwen35-9b`,
revision `baf3755e…`) over an SSH tunnel — preflight → identity freeze → live FM
calls → per-trial records → aggregation — on a deliberately small variant set
covering all three domains, both feasibility classes, and the infeasible
sentinel. `--run-type smoke`, 6 variants, 1 repetition, 773 s.

| Variant | GT feasible | Live outcome |
| :--- | :---: | :--- |
| W1 | yes | **success**, goal coverage 1.0 |
| K1 | yes | fail — `UNDECLARED_PARTICIPANT: operation_pairings[5] references ['table']` |
| L1 | yes | fail — `NO_GLOBAL_REGION_ASSIGNMENT` |
| K7 | no | correctly did not complete; conclusion not reached |
| L7 | no | correctly did not complete; conclusion not reached |
| W10 | no | correctly did not complete; conclusion not reached |

**False completions: 0.** That is the invariant that matters most, and it holds
live as well as offline.

This is n = 1 per variant on a 9B model at temperature 0.6, so it is a harness
validation and not a performance measurement — 1/3 feasible success here is
entirely consistent with the offline 34/60 at this sample size (Wilson interval
[0.06, 0.79]).

Three things the smoke proved by firing rather than by passing:

1. **Identity drift is blocked.** Re-running into the pre-fix output root was
   refused, naming all five sampler fields and the git SHA that had changed.
2. **The integrity guard works.** The repetition is marked `finished: false` with
   `errors: ["code or working tree changed during the repetition"]` — because the
   repository was being edited while it ran. It correctly refused to certify a run
   whose code moved underneath it. An unfinished repetition still contributes its
   trials to aggregation and is named, so the denominator stays honest.
3. **The two scorers disagreed**, which is how the tautology in §4.4 was found.

---

## 6. Reproduction

Zero-FM-call frozen replay of the whole 3×32 matrix:

```bash
PYTHONPATH=. python scripts/replay_frozen_distribution.py \
  --distribution benchmark_reports/v3_qwen_distribution_3x32_20260910T053937 \
  --out benchmark_reports/<new_dir>
```

Test suites and the leakage audit:

```bash
python -m pytest mujoco_scenes/functional_tamp_pipeline/tests -q
python scripts/audit_no_gt_leakage.py     # must print FINDINGS: 0
```

The live experiment (**not run**; see §7):

```bash
ssh -i ~/keyfile -p <port> -N -L 8000:127.0.0.1:8000 long-horizon@0.tcp.in.ngrok.io &
PYTHONPATH=. python scripts/run_live_repeat_experiment.py \
  --output-root benchmark_reports/live_experiment_10x32 \
  --base-url http://127.0.0.1:8000/v1 --model qwen35-9b \
  --run-type full --repeats 10
PYTHONPATH=. python scripts/aggregate_live_repeats.py \
  --root benchmark_reports/live_experiment_10x32
```

Preflight refuses to start on a dirty tree, on a model the endpoint does not
serve, or into a root whose `frozen_identity.json` differs in any recorded field.

### 6.1 A reproducibility hazard worth knowing about

Replaying with `--workers 6` while a test suite and a live run competed for the
GPU produced **SIGSEGV (rc=-11) in 11 of 96 trials**, all workshop — MuJoCo
rendering contention, not a pipeline defect. The harness handled it correctly:
each crashed trial is written as `pipeline_status: HARNESS_FAILURE` with
`feasible: null`, so it is never scored as a semantic failure. But an aggregate
read without checking for those rows would silently report a 50/46
feasible/infeasible split instead of 60/36 and a depressed success count.

**Always check for `HARNESS_FAILURE` rows before reading a replay aggregate.**
Re-running the affected trials at `--workers 2` reproduces them cleanly; the
driver caches completed rows, so deleting only the stub rows and re-running
redoes only those. The live runner executes trials sequentially and is not
exposed to this.

### 6.2 The "frozen" replay is not bit-deterministic

A more serious finding, and it qualifies every number in this report.

Replaying the *same* archived FM response through the *same* code twice does not
always give the same result. Diffing two full replays, exactly one trial flips:

| | run A (6 workers, under load) | run B (2 workers) |
| :--- | :--- | :--- |
| `workshop/W5/trial_03` status | `PARTIAL_ACTION_SEQUENCE_READY` | `ACTION_SEQUENCE_READY` |
| grounding failure | `OBJECT_DISCOVERY_FAILURE` | none |
| GT goal coverage | 0.333 | **1.000** |
| success | no | **yes** |

Nothing semantic differs — the contract is byte-identical, since it is read from
disk. What differs is **perception**: under contention the trial under-detects
and loses an object; with the machine quiet it detects everything and the trial
succeeds. W5/03 was one of the 11 trials that had segfaulted and been re-run at
lower concurrency, which is how the flip was noticed at all.

This has three consequences, and they are stated plainly because they cut against
the headline:

1. **Feasible success is 34/60 or 35/60 depending on machine load.** The
   difference is one marginal-detection trial, not a code change.
2. The composite directory that mixed the two load conditions is **not** a valid
   measurement, for exactly the reason given in §4.1 about mixed attempts. It was
   discarded rather than reported.
3. The trials most exposed are the ones already identified as detection-limited
   (§3.1, §3.2) — marginal detections are marginal in both directions.

The authoritative number in §1 is therefore from a **single clean replay at
`--workers 2` with nothing else running**, not from a composite and not from a
loaded run.

**What is established, and what is hypothesis.** Establishing this cleanly
matters, so the two are kept apart.

*Established.* The workshop domain — the domain the flip occurred in — runs
YOLO-World on the **GPU** (`workshop_phase1_yoloworld_l_five_view_close.yaml`:
`device: 0`, `inference_size: 1280`) with acceptance thresholds as low as `0.001`,
so a large number of detections sit near their acceptance boundary. And the
repository sets **no determinism controls anywhere**:

```
manual_seed | cudnn.deterministic | use_deterministic_algorithms
np.random.seed | CUBLAS_WORKSPACE_CONFIG        -> no matches
```

*Hypothesis, not isolated.* The most likely mechanism is cuDNN kernel
autotuning: it selects algorithms against currently available GPU memory, so
under contention it can choose differently and return slightly different
confidences, which is enough to flip a detection sitting at a 0.001 threshold.
This is consistent with every observation — the flip is in the GPU domain, on a
marginal detection, and correlates with concurrency — but it was **not** isolated
by controlled experiment. Two replays at identical concurrency were not compared,
so intrinsic run-to-run nondeterminism and load-induced nondeterminism are not
yet distinguished. Either way the remedy is the same, and either way the number
is not reproducible as it stands.

The remedy is the usual one (`torch.manual_seed`, `cudnn.deterministic = True`,
`cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`). It was **not** applied in this pass for one
concrete reason: it changes kernel selection for *every* detection, so it would
require re-measuring the whole matrix from scratch, and editing perception code
while the authoritative replay was running would have corrupted that replay --
the same hazard the live runner's "code changed during the repetition" guard
exists to catch.

This is the highest-value next change in the repository. It does not read ground
truth, does not condition on variants, and converts a benchmark whose headline
moves by a trial between runs into a reproducible one. The cheap first experiment
is two replays at identical concurrency: if they still differ, the
nondeterminism is intrinsic rather than load-induced.

---

## 7. Status of the full experiment

**The 10 × 32 run has not been started.** The harness is validated end to end on a
live endpoint (§5) and the identity freeze now blocks the three ways this
experiment could have been silently corrupted: a different model, a different
sampler, and a composite scoring of retried attempts.

Two decisions remain genuinely open and are the user's to make, not defects.

**1. The sampler.** The 10 repetitions will sample at temperature 0.6 with thinking
on, which is the configuration all offline numbers were measured on. That is the
right choice for comparability and for measuring variance, and it is also why the
expected live success rate is a *distribution* around 56.7% rather than a point.

**2. The token budget.** Four of the six raw-contract failures — L2/02, L3/01,
L3/02 and (in the kitchen/workshop) their counterparts — are `finish_reason
= 'length'`: the model was cut off mid-JSON at 24 000 tokens with thinking on. No
semantics were lost; they were truncated. Raising `TAMP_FM_MAX_TOKENS` would very
likely recover some of these, and it is not a form of cheating — it changes no
semantics and reads no ground truth.

It was **not** changed, for one reason: 24 000 is the budget the archived
distribution used, so raising it would make the live numbers incomparable with
every offline number in this report, exactly the failure §4.3 exists to prevent.
Changing it is a legitimate choice, but it means the live run measures a
*different* configuration and the offline 34/60 stops being its predictor. That
trade is the user's call.

---

## 8. Scientific constraints — compliance

Every constraint was enforced mechanically, not by inspection:

- **No variant-conditioned or trial-conditioned logic.** No branch anywhere on
  `K1…K12`, `L1…L10`, `W1…W10`, or equivalent identifiers. Verified by the
  leakage audit (0 findings).
- **No GT access at runtime** — not role assignment, object identity, feasibility,
  counts, or expected answers.
- **No scene inventory used to relax a requirement.** W2/03 fails rather than
  relaxing a required count of 2 to the 1 the scene holds (§3.4).
- **UNKNOWN is never promoted to TRUE.** K4/02 and K5/01 fail rather than
  selecting the task-convenient reading of a conflicted label (§3.2).
- **No false completion.** 0 across all 96 trials and 0 in the live smoke.
- **No metric that cannot fail.** The infeasible-variant outcome rule was a
  tautology and was removed even though doing so cost 29 trials of reported
  performance (§4.4).
- **No silent deletion of instruction-required constraints.**
- **Monotonicity respected.** Several candidate patches this pass were measured,
  found to trade successes (−1 or worse for +2), and reverted rather than shipped;
  their refutations are pinned as tests so they cannot be re-attempted silently.
- **History preserved.** No rewrite, no destructive reset, no force push.

---

## 9. Honest assessment of what is left

Ordered by what a further pass could plausibly recover.

| Opportunity | Trials | Assessment |
| :--- | :---: | :--- |
| Denser capture of the tool cabinet | 3 (W3) | **Real and tractable**, but a capture-side change, outside the semantic pipeline. Diagnosis is complete and numeric (§3.1). |
| Larger `max_tokens` in a future collection | 3 (L2/02, L3/01, L3/02) | **Real and tractable.** These are pure truncations at the 24 000-token budget with thinking on — no semantics were lost, only cut off. Not changed here because it breaks comparability with the archived distribution (§7). |
| Living Room seating semantics | 3 (L5/03, L6/01, L6/03) | Closed: the FM did not express a seat. Supplying it is forbidden. |
| Living Room draw-dependence | 2 (L1/02, L3/03) | L1/02 grounds COMPLETE in another draw on the identical scene, so the variance is in the contract, not perception. |
| Kitchen residual | 4 | K1/03 object discovery, K3/01 unsupported-process provenance (GT coverage already 1.0), K3/03, K4/03 unlabelled vessel. |
| Fastener labelling in W4/W5 | 3 | Detector-limited: a 2.67 cm object labelled `screwdriver`, and objects with no label at all. Per instruction, detectors were not tuned. |
| FM generation failures | 6 | Irreducible at one call per trial. |

Separately from recovering successes, the largest remaining *scientific* gap is
the infeasibility-conclusion rate: 7/36. The system reliably refuses to claim
completion (0 false completions) but usually stops at a partial plan rather than
concluding the task is impossible. Closing that gap is a statement about what the
planner reports, not about what it grounds, and it is the most promising direction
for a further pass now that the metric measures something real.

The single most valuable *experimental* addition is the live 10×32 with confidence
intervals — the aggregator computes Wilson intervals and the harness is now
trustworthy enough to produce them.

---

## 10. Commit history for this pass

| Commit | Change |
| :--- | :--- |
| `f83aa662` | an earlier stage may rank hypotheses, not delete them (W2 recovery) |
| `2cbae631` | one attempt is the result, and the whole run is on the record |
| `9937b82f` | sample live the way the offline numbers were actually drawn |
| (this pass) | one outcome rule for both scorers; the infeasible case was a tautology |
