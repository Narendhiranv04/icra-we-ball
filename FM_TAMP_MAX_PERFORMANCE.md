> **Superseded by `FM_TAMP_CLOSURE_FINAL.md`.** Every `outcome correct`
> figure in this file was produced by a scoring rule that, on an infeasible
> variant, asked only whether the task went unsatisfied -- true by
> construction, so it credited all 36 infeasible trials automatically. The
> rule was replaced; the honest figure is 42/96, not 69-71/96. Feasible-trial
> success, goal coverage and the false-completion count in this file are
> unaffected.

# Maximum-performance closure — final report

| | |
|---|---|
| branch | `vlm-testing-pipeline` |
| start of this pass | `829c5c5c` |
| final | `b8578d31` |
| pipeline suite | **1164 passed, 2 skipped, 0 failed** |
| harness tests | **6 passed** |
| GT-leakage audit | **0 findings**, 53 online modules |
| FM calls this pass | **0** |

## A. Evolution of performance

| stage | overall | kitchen | living | workshop |
|---|---|---|---|---|
| original baseline | 23/60 | 3/18 | 10/18 | 10/24 |
| cardinality/reuse closure | 31/60 | 11/18 | 10/18 | 10/24 |
| **this pass (W7 semantics)** | **32/60** | 11/18 | 10/18 | **11/24** |

Invariants held throughout: **false completions 0**, **planner failures after
complete grounding 0**, all 32 successes satisfy the full GT goal set and pass
independent symbolic validation, max **1** A* per trial, **0** FM requests on
replay, **0** harness failures, mean GT goal coverage on feasible trials 0.635.
**No sound success was regressed at any point.**

## B. Newly recovered

| trial | old failure | actual root cause | change | new outcome |
|---|---|---|---|---|
| `W7/trial_02` | contract not executable; `TOO_FEW_DISTINCT_PARTICIPANTS` | two separate misreads, below | declared-role-name stripping + robot-pose classification | **success**, GT coverage 1.0, exact 5-step GT sequence |

**Root cause 1.** An operation phrase may name a participant descriptively that
it does not list. Only the operation's *own* participants were stripped before
the physical-action test, so "open storage containers to find the compatible
**fastening** component" kept that word — the component's name — and matched as
an act coordinated with the search. An acquisition directive the search stage
already performs was compiled as a fastening. Fixed by stripping every declared
role's name.

**Root cause 2.** "The robot arm moves to a safe resting position on the
workbench" names the executor and a place and nothing else, so no task object
ends up anywhere. Read as a task operation it had one participant left once the
executor was removed and could state no relation. Now classified as the robot's
own pose — confined to exactly that shape, so "moves the component to the
workbench" keeps its operation and fails closed for under-specifying it.

## C. W2–W5 forensic table

Traced on instrumented replays, per §50.

| trial | raw screw evidence? | alternatives retained? | correct physical track? | final hypotheses | grounding | success |
|---|---|---|---|---|---|---|
| W2/01 | **yes**, ACCEPTED in all 5 views | **no** (`None`) | **yes** — 1.3 cm and 2.8 cm tracks, one with a measured 9 mm `CROSS_LIKE` head | `{screwdriver}` only | `fastener` unbound | no |
| W2/02, W2/03 | yes | no | yes | `{screwdriver}` | `fastener` unbound | no |
| W4/02, W4/03 | yes | no | yes | `{screwdriver}` | `fastener` unbound | no |
| W5/03 | yes | no | yes | `{screwdriver}` | `fastener` unbound | no |
| W4/01 | — | — | — | — | `repair_target` unbound | no |

**The decisive measurement.** Instrumenting `_compute_consensus_semantic_belief`
on a live W2/01 run gives four belief computations, one per track:

```
extent=0.1665  labels=['screwdriver'] -> screwdriver   prior {screw:0.0, screwdriver:2.0}
extent=0.0398  labels=['screwdriver'] -> screwdriver   prior {screw:2.0, screwdriver:0.0}
extent=0.2285  labels=['screwdriver'] -> screwdriver   prior {screw:0.0, screwdriver:2.0}
extent=0.2802  labels=['screwdriver'] -> screwdriver   prior {screw:0.0, screwdriver:2.0}
```

The size prior **works** — at 4 cm it correctly computes `screwdriver: 0.0` —
and changes nothing, because the track's observation set contains only
`screwdriver`. Every observation on every track comes from **one inference
source** (`proposal_crop`), exactly one per camera, with
`semantic_alternatives = None`. The `screw`-labelled full-frame and stage-crop
detections are recorded in diagnostics and never become track observations.

**Classification: `DETECTION_SEMANTIC_ROUTING`.** Not detection, not
association, not track geometry, not fusion weighting, not the ontology. This
rules out every remedy in §4–§11 of the brief: you cannot re-weight, re-associate
or geometrically rank evidence that is absent from the track. Recovery requires
changing which detections contribute semantics to a track, or retaining
alternatives through fusion — a perception change with its own before/after.

## D. Regressions discovered during development (all reverted)

| attempt | intent | measured effect | disposition |
|---|---|---|---|
| per-view median extent | rescue the fastener | workshop 10 → **6**; drivers vanished (occluded views under-measure) | reverted, pinned by tests |
| verb-forms-only coordinated action | fix W7/02 | recovered nothing, **lost W1/03** | reverted |
| operation-local functional forms + exhaustive slot matching | W3 ×3, W1/02 | gained W1/02, **lost W8/01** | reverted |

The third is the most interesting: even creating a *separate* operation-local
role — rather than re-typing the participant — still cost W8/01, because seating
by family changes which capability wins first and hides the projection the
contract needed. That is the open problem for W3.

## E. Remaining feasible failures (28)

| cause | count | trials |
|---|---|---|
| `COMPILER/FM_OMISSION` | 13 | W3 ×3, W1/02, W5/01, K3/01, K3/03, K4/02, K5/01, L3/03, L5/03, L6/01, L6/03 |
| `PERCEPTION` (fastener routing) | 6 | W2 ×3, W4/02, W4/03, W5/03 |
| `FM_GENERATION` / wire | 4 | L2/02, L3/01, L3/02, W5/02 |
| `GROUNDING` | 3 | K1/03, K2/01, L1/02, W4/01 |
| `PERCEPTION` (unlabelled vessel) | 1 | K4/03 |

### Recoverable with current evidence

- **W3 ×3 and W1/02** — the FM expresses fastening in all of them; the blocker
  is a tool-return operation needing an operation-local form of `workbench`.
  Attempted and reverted; needs the form chosen *after* capability selection
  rather than during it. **Risk: moderate** (W8 is the sentinel).
- **K3/01** — GT coverage 1.0 already. Blocked by `Pour SoupSupply into
  SoupContainer`; the runtime has no soup-material source role. Needs the
  unsupported-process provenance work. **Risk: moderate.**

### Irreducible with current evidence

- The **6 fastener trials** — proven above; the screw hypothesis is not in the
  track at all.
- **4 generation/wire failures** — the FM response is invalid before the
  compiler sees it.
- **K4/03** — its supply vessel comes back with no label at all, so no
  acceptance vocabulary could admit it.

## F. Scientific integrity

| check | result |
|---|---|
| variant- or trial-conditioned runtime logic | **none** |
| GT role assignment / inventory / feasibility in runtime | **none** |
| count reduction based on scene availability | **none** |
| object lookup tables | **none** |
| UNKNOWN treated as TRUE | **no** |
| false completion | **0, unchanged** |

## G. Harness validation

| requirement | status |
|---|---|
| aggregator reads `evaluation_records.json` | **fixed** — and a test asserts a file the evaluator never writes is *not* read |
| field names `gt_feasible` / `full_task_satisfied` / `full_task_goal_coverage` | **fixed**, declared once so they cannot drift |
| full 32-variant grid materialised | **fixed** — intended/completed/missing plus every hole, reported separately from conditional performance |
| failed attempts immutable | **fixed** — `repeat_NN/attempt_MM/`, nothing writes into an existing attempt, manifest names the authoritative one |
| exact model required | **fixed** — `--model` required, `ids[0]` fallback removed, absent model aborts |
| sampler explicitly frozen | **fixed** — one `SAMPLER` dict is both passed and recorded |
| smoke and final roots separate | **yes** — distinct `--output-root` per run identity |

## H. The smoke command

Covers every stressed mechanism per §46 — recovered cardinality (K1),
unsupported-process (K3), a Kitchen failure (K2), a stable Living success (L4),
a Living failure (L3), fastener routing (W2, W4, W5), operation forms (W3),
search parsing (W7), the regression sentinel (W8), and an infeasible case (W10):

```bash
cd /home/naren/RA_iiith
PYTHONPATH=. python scripts/run_live_repeat_experiment.py \
  --output-root benchmark_reports/live_smoke_$(date +%Y%m%dT%H%M%S) \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen35-9b \
  --repeats 1 \
  --variants K1,K2,K3,L3,L4,W2,W3,W4,W5,W7,W8,W10
```

then, with the same aggregator the full experiment will use:

```bash
PYTHONPATH=. python scripts/aggregate_live_repeats.py \
  --root benchmark_reports/live_smoke_<timestamp>
```

**The full 10×32 was not started**, per instruction.
