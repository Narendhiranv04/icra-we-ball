> **Superseded by `FM_TAMP_CLOSURE_FINAL.md`.** Every `outcome correct`
> figure in this file was produced by a scoring rule that, on an infeasible
> variant, asked only whether the task went unsatisfied -- true by
> construction, so it credited all 36 infeasible trials automatically. The
> rule was replaced; the honest figure is 42/96, not 69-71/96. Feasible-trial
> success, goal coverage and the false-completion count in this file are
> unaffected.

# W2–W5 forensics, and what this pass established

Branch `vlm-testing-pipeline`, verified at `9eeb36c2` (clean, matching origin).
Frozen baseline reproduced with the actual evaluator: **31/60 feasible success,
31 complete groundings, 69/96 outcome correct, 0 false completions**.

This pass did **not** raise the score. It produced two things instead: a
complete causal chain for the Workshop fastener failures, ending in evidence
that closes the question; and two attempted fixes that were measured, found to
cost successes, and reverted. Both are recorded here because the negative
results are the useful output.

---

## A. The W2 fastener chain, traced end to end

Instrumented on a real replay of `W2/trial_01` (not from summaries).

### 1. Detection — the screws ARE found

`LEFT_DRAWER` alone yields **83 detection records**. Screws are detected and
`ACCEPTED` in **every one of the five views**:

| view | detection | label | conf | depth pts | status |
|---|---|---|---|---|---|
| LEFT | `LEFT_physical_001` | **screw** | 0.025 | 1740 | ACCEPTED |
| CLOSE | `CLOSE_physical_001` | **screw** | 0.046 | 2154 | ACCEPTED |
| FRONT | `FRONT_physical_002` | **screw** | 0.011 | 766 | ACCEPTED |
| RIGHT | `RIGHT_physical_006` | **screw** | 0.002 | 546 | ACCEPTED |
| TOP | `TOP_physical_001` | **screw** | 0.002 | 4992 | ACCEPTED |

Within each camera the screw and the screwdriver are **separate** physical
groups (`LEFT_physical_001` vs `LEFT_physical_002/003`). Detection is not the
failure, and neither is per-view separation.

### 2. Track geometry — the tracks are correct

Final `G_O` holds four objects, and their measured geometry is right:

| object | max cross-section | what it is |
|---|---|---|
| `object_0001` | 0.133 m | a driver |
| `object_0002` | **0.0128 m** | **a screw** |
| `object_0003` | **0.0283 m** | **a screw** |
| `object_0004` | 0.106 m | a driver |

`object_0002` carries a fully measured Phillips head: `head_diameter_m
0.00905`, `interface_geometry: CROSS_LIKE`, `footprint_length_m 0.0398`,
`transverse_width_m 0.00255`. The geometry pipeline has measured a screw.

**Association is not the failure either.** The tracks are not collapsed; a 1.3 cm
object and a 13 cm object are correctly distinct tracks. The transitive-collapse
hypothesis (§5 of the brief) is **not** what is happening here.

### 3. Where it actually dies — the observations attached to each track

Instrumenting `_compute_consensus_semantic_belief` on a live run gives **four**
belief computations, one per track:

```
extent=0.1665  labels=['screwdriver'] -> screwdriver   prior mult {screw:0.0, screwdriver:2.0}
extent=0.0398  labels=['screwdriver'] -> screwdriver   prior mult {screw:2.0, screwdriver:0.0}
extent=0.2285  labels=['screwdriver'] -> screwdriver   prior mult {screw:0.0, screwdriver:2.0}
extent=0.2802  labels=['screwdriver'] -> screwdriver   prior mult {screw:0.0, screwdriver:2.0}
```

Read the second line carefully. The size prior **works**: at 4 cm it correctly
computes `screwdriver: 0.0` and `screw: 2.0`. It changes nothing, because the
track's observation set contains **only** `screwdriver`. There is no screw
hypothesis for the prior to promote.

Dumping every observation on every track:

```
extent=0.0398  n_obs=4
   LEFT   canon=screwdriver raw=Phillips screwdriver conf=0.0080 src=proposal_crop  alts=None
   RIGHT  canon=screwdriver raw=Phillips screwdriver conf=0.0029 src=proposal_crop  alts=None
   TOP    canon=screwdriver raw=Phillips screwdriver conf=0.0020 src=proposal_crop  alts=None
   CLOSE  canon=screwdriver raw=Phillips screwdriver conf=0.0031 src=proposal_crop  alts=None
```

Every observation on every track comes from **one inference source**,
`proposal_crop`, exactly one per camera — and that pass labels all four objects
a screwdriver, including the 4 cm one with a 9 mm cross-like head. The
`screw`-labelled detections from the full-frame and stage-crop passes are
recorded in the diagnostics and **never become track observations**.
`semantic_alternatives` is `None` on all of them, so no runner-up hypothesis is
retained either.

### 4. Conclusion

The classification is **`TRACK_LABEL_AGGREGATION_FAILURE` upstream of
aggregation** — more precisely, an *observation-source* failure:

> A track's semantic evidence is drawn from a single inference pass whose labels
> are systematically wrong for small fasteners, and the detector's alternative
> hypotheses are discarded.

This closes the question the brief asked. It is **not** detection, **not**
association, **not** track geometry, **not** the size prior, and **not** the
ontology. It means:

- no association-cost change (§4–§7 of the brief) can recover it — the tracks
  are already correct;
- no label-aggregation change (§8) can recover it — there is nothing to
  aggregate, only one label exists;
- the size prior cannot recover it — it already fires correctly and has no
  alternative to promote.

The only recovery is to change **which detections contribute semantics to a
track**, or to retain `semantic_alternatives` through fusion so the prior has a
runner-up to promote. Both are perception-architecture changes with their own
before/after, and I did not start one at the end of a semantic pass — the last
two attempts to move faster than the evidence here each cost a working variant.

**Affected: W2 ×3, W4 ×2, W5/03 — six feasible trials.**

---

## B. Two fixes attempted, measured, and reverted

### B.1 W7/02 — search directive misread as a coordinated physical action

`W7/trial_02` has **GT goal coverage 1.0** and a non-executable contract. The
blocking operation is:

> "The robot arm opens and inspects storage containers to find the compatible
> fastening component."

This is a search directive the runtime already performs. It is misread because
`_COORDINATED_PHYSICAL_ACTION` sees a coordinator ("and") followed within 40
characters by `fasten\w*` — matching **"fastening" inside the component's own
descriptive name**. That component is not a participant of *this* operation, so
`operation_action_phrase` does not strip it. The phrase is therefore compiled as
physical, loses its only other participant when the actor is removed, and fails
`TOO_FEW_DISTINCT_PARTICIPANTS_TO_STATE_A_RELATION`.

The diagnosis is certainly right, and the fix direction — restrict the verb
alternation to verb forms, excluding the attributive gerund, exactly as the
relation reader already does — behaves correctly on the discriminating cases:

| phrase | wanted | got |
|---|---|---|
| "opens and inspects storage containers to find the compatible fastening component" | non-physical | non-physical |
| "Search the drawers **and then fasten** the screw into the joint" | physical | physical |
| "Open the cabinet **and place** the tool on the bench" | physical | physical |
| "Inspect the storage to locate the **mounting bracket**" | non-physical | non-physical |

**But a sweep over all 91 compilable contracts showed it did not recover W7/02
and it lost `W1/trial_03`, which is a current success.** Reverted. The real fix
is to strip *every declared role's* name before the coordinated-action test, not
just the operation's own participants — a signature change through several call
sites that I would not land unmeasured.

### B.2 The per-view median extent (earlier pass, recorded again)

Already reverted and pinned by tests in `test_track_extent_robustness.py`. The
evidence above explains *why* it could never have worked: the extent was never
the problem.

---

## C. Remaining feasible failures (29), earliest cause

| count | earliest cause | trials |
|---|---|---|
| 6 | **perception: single-source track semantics** (§A) | W2 ×3, W4/02, W4/03, W5/03 |
| 3 | compiler: second operation blocks contract | W3 ×3 |
| 4 | FM wire invalid / generation | L2/02, L3/01, L3/02, W5/02 |
| 5 | contract not executable, living room | L3/03, L5/03, L6/01, L6/03, L1/02 |
| 2 | compiler: operation form selection | W1/02, W7/02 |
| 1 | grounding: `repair_target` unbound | W4/01 |
| 8 | kitchen: mixed grounding / contract | K1/03, K2/01, K3/01, K3/03, K4/02, K4/03, K5/01, + |

`W7/02` and `K3/01` both have **GT goal coverage 1.0** with non-executable
contracts, which makes them the highest-value remaining semantic targets.
`K3/01`'s blocker is `Pour SoupSupply into SoupContainer` — the runtime has no
soup-material source role, the documented category-(C) gap.

---

## D. Scientific integrity

| check | result |
|---|---|
| variant- or trial-conditioned runtime logic | **none** |
| GT inventory / role assignment leakage | **none** — audit reports 0 findings, 53 modules |
| count reduction based on scene availability | **none** |
| UNKNOWN treated as TRUE | **no** |
| false completion | **0, unchanged** |
| successes lost | **0** — both attempted fixes reverted before landing |

The working tree is byte-for-byte `9eeb36c2`.
