> **Superseded by `FM_TAMP_CLOSURE_FINAL.md`.** Every `outcome correct`
> figure in this file was produced by a scoring rule that, on an infeasible
> variant, asked only whether the task went unsatisfied -- true by
> construction, so it credited all 36 infeasible trials automatically. The
> rule was replaced; the honest figure is 42/96, not 69-71/96. Feasible-trial
> success, goal coverage and the false-completion count in this file are
> unaffected.

# FM-guided TAMP pipeline — state, metrics, and where the failures are

Branch `vlm-testing-pipeline`. Head `f8a6a3d7`. Full pipeline test suite:
**1121 passed, 3 skipped, 0 failed**. GT-leakage audit: **0 findings**.

Every metric below comes from replaying the **same 96 archived FM responses**
through the deterministic pipeline: zero FM calls, zero harness failures, at
most one A* invocation per trial. Artifacts:

| what | where |
|---|---|
| closure report and audits | `benchmark_reports/object_role_closure_20260911T124448/` |
| after (code at `f8a6a3d7`) | `benchmark_reports/frozen_replay_objectrole_final_20260911T120800/` |
| before (code at `c35d062e`) | `benchmark_reports/frozen_replay_pass_baseline_20260911T122527/` |
| archived FM responses | `benchmark_reports/v3_qwen_distribution_3x32_20260910T053937/` |

---

## 1. What this pass did

The living room existed to test **shared versus distinct objects being
placed**, and it scored 0 of 6. The role mapping handled regions and dropped
objects: a participant the model declared `entity_kind: OBJECT` was
canonicalized onto a *region* whenever the categories it listed happened to read
like a surface.

`refreshment_setting` — "holds refreshments for a person", offered as "table,
surface, tray" — came back as `PERSONAL_CUP_SAUCER_REGION`, the side table it
belongs on. The payload the task is about never entered the graph, and every
operation over it went with it. Worse downstream: the detector vocabulary is
built from the compiled graph, so in one variant the cups, saucers **and seats**
were never looked for — the observed scene graph held three regions and a
remote, nothing else.

Five defects were found and fixed, and one of my own fixes was measured to be
harmful and withdrawn.

| commit | what |
|---|---|
| `a98db360` | the declared `entity_kind` was never read; now it is |
| `3eaf3143` | a capability slot could substitute one participant for another, and an operation's phrase reached other operations |
| `4391975f` | `operation_count` was never reconciled with `required_count` |
| `c8cbc6bc` | "arrange" matched no capability once the participant's name was stripped |
| `8d6f2cba` | **withdrew** the broader half of `a98db360` — see §6 |
| `f8a6a3d7` | closure evidence |

---

## 2. Headline metrics — attributable to this pass

96 frozen trials, baseline taken at `c35d062e` so this pass is not credited
with the perception fix (`5cc21285`) that landed before it.

| stage | before | after |
|---|---|---|
| strict V3 valid | 90 | 90 |
| canonical graph emitted | 90 | 90 |
| executable contract complete | 57 | **58** |
| grounding reached | 90 | 90 |
| complete grounding | 20 | **22** |
| A* reached | 60 | **61** |
| success | 20 | **22** |
| outcome correct | 58 | **60** |
| false completion | 0 | **0** |
| regressions | — | **none** |

**Read against the previously published artifact** the same table says success
10 → 22 and outcome correct 47 → 60. Most of that is **not** this pass: commit
`5cc21285` (drive-slot bound and size prior) is what lifts the workshop from 0
to 10. This pass's own contribution is two living-room trials plus the mechanism
that makes object roles compile at all.

### Per domain

| domain | feasible success | outcome correct | before (feasible) |
|---|---|---|---|
| kitchen | 3/18 (16.7%) | 21/36 (58.3%) | 3/18 |
| living room | **9/18 (50.0%)** | **22/30 (73.3%)** | 7/18 |
| workshop | 10/24 (41.7%) | 17/30 (56.7%) | 10/24 |
| **all** | **22/60 (36.7%)** | **60/96 (62.5%)** | 20/60 |

On the living room's six feasible variants at `trial_01`, success goes from
**2 of 6 to 4 of 6** (L1, L2, L4, L5), each at GT goal coverage 1.0.

---

## 3. Variant-wise metrics

`exec` = executable contract, `grounded` = complete grounding, `before` =
success at `c35d062e`. Three trials per variant.

### Kitchen

| variant | scene | feasible | exec | grounded | success | outcome correct | mean GT cov | before | dominant blocker |
|---|---|---|---|---|---|---|---|---|---|
| K1 | F0_ALL_VISIBLE | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.17 | 0/3 | grounding: coffee_source,soup_eating_utensil,water_source (1); grounding: soup_container,soup_eating_utensil (1); grounding: coffee_source,soup_eating_utensil (1) |
| K2 | F1_HIDDEN_COFFEE_VESSEL | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.42 | 0/3 | grounding: soup_eating_utensil (2); grounding: coffee_source (1) |
| K3 | F2_HIDDEN_SOUP_BOWL | yes | 1/3 | 0/3 | **0/3** | 0/3 | 0.00 | 0/3 | contract not executable (2); grounding: coffee_source,soup_eating_utensil (1) |
| K4 | F3_HIDDEN_VESSELS_MIXED | yes | 2/3 | 0/3 | **0/3** | 0/3 | 0.17 | 0/3 | grounding: coffee_container,soup_container,soup_eating_utensil (1); contract not executable (1); grounding: coffee_source,soup_eating_utensil,water_source (1) |
| K5 | F4_TOOLS_IN_DRAWERS | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.67 | 2/3 | contract not executable (1) |
| K6 | F5_FULL_DISTRIBUTED_SEARCH | yes | 3/3 | 1/3 | **1/3** | 1/3 | 0.67 | 1/3 | grounding: soup_eating_utensil (1); grounding: coffee_source (1) |
| K7 | I0_MISSING_COFFEE_VESSEL | no | 3/3 | 0/3 | **0/3** | 3/3 | 0.17 | 0/3 | grounding: coffee_source,soup_eating_utensil,water_source (1); grounding: coffee_source (1); grounding: coffee_container,coffee_source,soup_eating_utensil,water_source (1) |
| K8 | I1_MISSING_SOUP_BOWL | no | 0/3 | 0/3 | **0/3** | 3/3 | 0.00 | 0/3 | contract not executable (3) |
| K9 | I2_MISSING_COFFEE_SPOON | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.33 | 0/3 | grounding: coffee_stirrer (2); contract not executable (1) |
| K10 | I3_MISSING_SOUP_UTENSIL | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.17 | 0/3 | contract not executable (2); grounding: coffee_source,soup_eating_utensil (1) |
| K11 | I4_MISSING_KETTLE | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.00 | 0/3 | contract not executable (2); grounding: coffee_container,soup_container,soup_eating_utensil,water_source (1) |
| K12 | I5_MISSING_COFFEE_JAR | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.17 | 0/3 | grounding: coffee_source (1); grounding: coffee_source,soup_eating_utensil (1); FM wire invalid (1) |

### Living Room

| variant | scene | feasible | exec | grounded | success | outcome correct | mean GT cov | before | dominant blocker |
|---|---|---|---|---|---|---|---|---|---|
| L1 | F0_ALL_OBJECTS_IN_STAGING | yes | 3/3 | 2/3 | **2/3** | 2/3 | 0.78 | 1/3 | grounding: SEATING_POSITION (1) |
| L2 | F1_LEFT_SAUCER_PREPLACED | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.67 | 1/3 | FM wire invalid (1) |
| L3 | F2_LEFT_SAUCER_ON_SHARED | yes | 0/3 | 0/3 | **0/3** | 0/3 | 0.22 | 0/3 | FM wire invalid (2); contract not executable (1) |
| L4 | F3_LEFT_CUP_ON_SHARED | yes | 2/3 | 2/3 | **2/3** | 3/3 | 1.00 | 2/3 | contract not executable (1) |
| L5 | F4_SAUCER_PREPLACED_CUP_ON_SHARED | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.67 | 2/3 | contract not executable (1) |
| L6 | F5_LEFT_PAIR_ON_SHARED | yes | 1/3 | 1/3 | **1/3** | 1/3 | 0.33 | 1/3 | contract not executable (2) |
| L7 | I0_NO_SHARED_TABLE | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.44 | 0/3 | contract not executable (1); grounding: SEATING_POSITION,SHARED_REMOTE_REGION (1); grounding: SHARED_REMOTE_REGION (1) |
| L8 | I1_NO_LEFT_PERSONAL_TABLE | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.11 | 0/3 | contract not executable (2); grounding: PERSONAL_CUP_SAUCER_REGION (1) |
| L9 | I2_NO_PERSONAL_TABLES | no | 0/3 | 0/3 | **0/3** | 3/3 | 0.00 | 0/3 | contract not executable (2); FM wire invalid (1) |
| L10 | I3_NO_TABLES | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.00 | 0/3 | contract not executable (2); grounding: CUP_SAUCER_SET,PERSONAL_CUP_SAUCER_REGION,SEATING_POSITION,SHARED_REMOTE_REGION (1) |

### Workshop

| variant | scene | feasible | exec | grounded | success | outcome correct | mean GT cov | before | dominant blocker |
|---|---|---|---|---|---|---|---|---|---|
| W1 | F0_MANUAL_FIRST_ONE_REGION | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.78 | 2/3 | contract not executable (1) |
| W2 | F1_POWER_FIRST_ONE_REGION | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.22 | 0/3 | grounding: fastener (3) |
| W3 | F2_MANUAL_FIRST_TWO_REGIONS | yes | 0/3 | 0/3 | **0/3** | 0/3 | 0.00 | 0/3 | contract not executable (3) |
| W4 | F3_POWER_FIRST_TWO_REGIONS | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.33 | 0/3 | grounding: fastener (2); grounding: repair_target (1) |
| W5 | F4_MANUAL_FIRST_THREE_REGIONS | yes | 1/3 | 0/3 | **0/3** | 0/3 | 0.11 | 0/3 | grounding: fastener (1); contract not executable (1); FM wire invalid (1) |
| W6 | F5_POWER_FIRST_THREE_REGIONS | yes | 3/3 | 3/3 | **3/3** | 3/3 | 1.00 | 3/3 | — |
| W7 | F6_MANUAL_ONLY | yes | 2/3 | 2/3 | **2/3** | 3/3 | 1.00 | 2/3 | contract not executable (1) |
| W8 | F7_POWER_ONLY | yes | 3/3 | 3/3 | **3/3** | 3/3 | 1.00 | 3/3 | — |
| W9 | I0_NO_DRIVER | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.22 | 0/3 | grounding: repair_target (2); contract not executable (1) |
| W10 | I1_NO_SCREW | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.22 | 0/3 | grounding: fastener (2); contract not executable (1) |

---

## 4. Where the failures happen

Blocking stage, counted once per trial:

| stage | all 96 | feasible only (60) |
|---|---|---|
| FM wire response invalid | 6 | 4 |
| contract not executable | 32 | 15 |
| grounding: object discovery | 26 | 14 |
| grounding: functional assignment | 10 | 5 |
| **planning** | **0** | **0** |
| success | 22 | 22 |

**Planning is not a bottleneck anywhere.** Every trial that reaches a complete
grounding produces a validated plan. All 26 infeasible-variant trials that stop
before a plan are counted as correct rejections.

Roles unbound on feasible trials that had an executable contract:

| count | domain | role |
|---|---|---|
| 9 | kitchen | `soup_eating_utensil` |
| 7 | workshop | `fastener` |
| 6 | kitchen | `coffee_source` |
| 2 | kitchen | `water_source` |
| 2 | kitchen | `soup_container` |
| 1 | kitchen | `coffee_container` |
| 1 | living room | `SEATING_POSITION` |
| 1 | workshop | `repair_target` |

---

## 5. Plans against the GT action catalogue

Of the 22 trials reporting a complete plan, **all 22** satisfy the full GT goal
set and **all 22** pass independent symbolic validation. 17 are also the exact
GT action sequence — same operators in the same order, object arguments used
consistently:

| domain | exact / complete |
|---|---|
| living room | **9 / 9** |
| workshop | 8 / 10 |
| kitchen | 0 / 3 |

The five that differ are not failures. Two workshop plans return the tool to a
different storage region than the catalogue names; three kitchen plans carry a
vessel before pouring where the catalogue pours first. Each reaches every GT
goal by another legitimate route.

---

## 6. A rule I added, measured, and withdrew

The entity-kind reading was first written to strike any family the runtime
realizes solely with roles of the opposite class. It looked safe and was not.

A workshop contract declared its **workbench surface** an `OBJECT`. Its families
were `SUPPORT`, from the role's own name and job description, and `INSTRUMENT`,
from an incidental cue. Striking `SUPPORT` left the incidental cue standing, so
**the bench became the screwdriver**, collided with the real tool, and W4 lost a
contract it previously had. The frozen replay's regression check is what caught
it; nothing in the test suite did.

The surviving rule is the one the case actually needed: a *category list*
naming only families of the other class cannot outrank the job description. It
carries every living-room contract and leaves the workbench a workbench.

Two further narrowings came from tests that encoded real distinctions:

- `FIXED_TARGET` is in neither the carried nor the area class, because the model
  uses it for both — the assembly a screw goes through is a part one could pick
  up *and* a site the runtime fixes to the bench. Treating it as stationary
  stopped a workpiece declared `OBJECT` from being the repair target.
- Both new readings apply only to participants whose family the contract's own
  words settled. A participant it characterizes in no way — a person, "recipient
  of the meal", categories "human, guest" — is ambiguous over the whole domain
  for the correct reason. Narrowing that one left a set small enough to seat, and
  `serve soup(soup_bowl, diner)` compiled into putting the bowl into the person.

---

## 7. Improvements that can still be made, ranked

### 7.1 Workshop: the screw loses the per-object label vote — 7 trials, actionable

The highest-value remaining fix, and the only residual that is a genuine
**engineering** defect rather than an FM or benchmark limit.

W2, W4 and W5 have executable contracts and fail only because `fastener` binds
to nothing. The evidence in
`trial_01/workshop/W2/vlm/detection_diagnostics.json`:

- the detector vocabulary **does** contain `screw`, `Phillips screw`,
  `Phillips head screw`, and the `fastener` role accepts all three;
- detections labelled `screw` **are** produced and reach `ACCEPTED`, including
  many in `LEFT_DRAWER`, where the GT screw lives;
- yet the observed scene graph contains **four objects, all labelled
  `screwdriver`** (in `trial_02`, all four labelled `power_driver`) and **no
  screw at all**.

So the per-object label fusion in `mujoco_scenes/workshop_phase1/tracking.py`
collapses every tracked object onto a driver label. The size prior that should
separate them is configured and loaded
(`workshop_phase1_yoloworld_l_five_view_close.yaml`:
`small_object_max_dimension_m: 0.08`, screw ×2.0 and screwdriver ×0.0 below it,
the reverse above). For all four objects to come out as drivers, each must be
measuring **larger than 8 cm** — so screw detections are being merged with a
neighbouring driver before classification, and the merged extent defeats the
prior.

Direction: apply the size prior per detection before duplicate fusion, or make
the duplicate-merge policy refuse to merge detections whose labels belong to
roles the runtime keeps apart. Worth up to 7 trials. Not attempted here because
it is a perception change that could disturb the 10 workshop successes, and it
needs its own before/after measurement.

### 7.2 Four trials lose the FM response to strict wire validation

L2/02, L3/01, L3/02, W5/02. Each is an FM output defect. More wire
normalization could recover some, but every repair risks inventing structure,
so each would need to be justified individually.

### 7.3 Nothing to gain from the kitchen without cheating

See §8. The kitchen's dominant cause is the model asking for more objects than
the scene contains; honouring that is the fail-closed behaviour.

---

## 8. What remains, by irreducible category

### (E) Documented irreducible ambiguity — the dominant kitchen cause

**19 of 35 kitchen trials (54%) ask for more of something than the scene
contains**, so no grounding can satisfy them:

- **18 over-demand spoons.** The scene holds exactly three. The GT uses one
  stirrer twice and two distinct soup utensils. Those trials declare the
  stirrer `required_count: 2, DISTINCT`, needing four.
- **10 declare a single-instance source as two distinct instances** — two
  coffee jars or two kettles, where the scene has one of each.
- 9 do both.

The instruction says "Serve each soup bowl with **its own** suitable eating
utensil" and says nothing about own-ness for the stirrer, so it marks
distinctness explicitly where it wants it. The wire prompt already states the
distinction plainly ("one jar poured from twice is a count of two as well …
REUSABLE when one can serve every time"). The model gets `required_count` right
and `binding_policy` wrong.

Lowering DISTINCT to REUSABLE would raise the success rate and would be
discarding a requirement the model stated.

**Detection is not the kitchen's problem.** K1 detects all nine objects
(2 cups, 2 bowls, 3 spoons, kettle, coffee jar); K4 detects all nine including
the cup found by searching `C2` and the bowl in `B1` — exactly the GT
inventory. Both then fail on counts.

### (A) Semantics the FM did not express

- **L6** declares no seating and no accessibility at all, only
  "refreshment_setting is located on surface". Supplying the seat anchor was
  tried in an earlier pass and reverted (`470f7c58`): that a placement be
  reachable from both seats is task semantics, not geometry the runtime owns.
- **L3** declares `surface` as "a flat area to hold refreshment items", count 2
  DISTINCT, then reuses that same role for the remote. The shared central region
  was never expressed; supplying a third would create a participant.
- Kitchen contracts naming a **person** as a physical endpoint are classified as
  abstract directives rather than compiled.

### (C) A missing or incompatible runtime role

- No **soup-material source** role, so `Pour SoupSupply into SoupContainer`
  cannot be represented. Preserved as a representability failure; the material
  is never relabelled as the bowl.
- No **intermediate cooking vessel** role, so `pour(cooking_vessel, coffee_cup)`
  and `Pour Coffee(pot, mug)` cannot be seated.

### (D) Perception

One cluster outstanding: §7.1.

---

## 9. Soundness invariants held

- at most **1** A* invocation per trial
- **0** semantic FM requests on a frozen replay, by construction
- **0** harness failures across 96 trials
- **0** false completions, and **0** completions disagreeing with the GT goal set
- front-half compile hash identical across `PYTHONHASHSEED` 0, 1, 7, 12345
- GT-leakage audit: **0** findings over the scanned modules
- test suite: **1121 passed, 3 skipped, 0 failed**
- frozen archive unmodified: 91 `raw_v3.json` on disk, 91 tracked, 0 changed

---

## 10. Commit history for this pass
- `f8a6a3d7` eval(vlm): closure evidence for object roles alongside region roles
- `8d6f2cba` fix(vlm): trust the declared entity kind only against a category list
- `c8cbc6bc` fix(vlm): read arranging as the placement it is
- `4391975f` fix(vlm): reconcile how often an operation runs with what it consumes
- `3eaf3143` fix(vlm): a capability slot may settle where, not what
- `a98db360` fix(vlm): read the entity kind the model declares for each participant
- `c35d062e` fix(vlm): give the live contract normalizer the domain it needs
- `7194fa5e` fix(vlm): declare an undeclared reference to a place the runtime owns
- `ae129ef4` fix(vlm): pin the live schema version, and repair two more kinds of wire noise
- `fb1ce802` test(eval): score produced plans against the GT action catalogue in all three domains
- `32be6ae8` chore(vlm): tolerate an absolute archive path in the closure builder
- `5cc21285` fix(perception): restore the drive-slot bound and apply the size prior everywhere
- `2d563c84` eval(vlm): final semantic closure artifact and frozen 3x32 evidence
- `a94e7c91` chore(vlm): refreeze the method manifest and add the closure instruments
- `c28eb6ea` fix(perception): a stage may not claim the contents of the region next door
- `5cb904a3` fix(workshop): two names for one fixed place are one anchor, not two participants
- `0f5fdf99` test(vlm): audit the boundary between the online path and the reference
- `6d9be9aa` fix(eval): tell a response that never arrived from a task stated wrongly
- `a8ccbed1` fix(vlm): let operations inform role typing, and stop excusing unread relations
- `2dd7bbca` fix(vlm): read relation meaning by stem instead of by remembered phrase
- `1faab876` fix(workshop): tell the fastening site apart from the thing fastened into it
- `2ae30a46` fix(vlm): jointly propagate relation and role semantics

22 commits, no history rewriting, no force push. `ae4a7e90` is the head this
closure pass started from.
