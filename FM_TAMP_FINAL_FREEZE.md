# FM-guided TAMP — final deterministic freeze

## A. Repository

| | |
|---|---|
| branch | `vlm-testing-pipeline` |
| starting SHA (this pass) | `16206178` |
| final SHA | `340f6d73` |
| `git status` | clean of source changes |
| `git diff --check` | clean |
| pipeline test suite | **1143 passed, 3 skipped, 0 failed** |
| GT-leakage audit | **0 findings** over 53 online modules |

Development data: the same 96 archived V3 responses in
`benchmark_reports/v3_qwen_distribution_3x32_20260910T053937/`.
**Zero FM calls were made in this pass.** Closure artifact:
`final_freeze_20260911T155118/`.

---

## B. Frozen before/after — 96 trials, 0 FM calls, 0 harness failures

Before = `16206178`, after = this freeze, identical inputs.

| stage | before | after |
|---|---|---|
| strict V3 valid | 90 | 90 |
| canonical graph emitted | 90 | 90 |
| executable contract complete | 58 | **59** |
| grounding reached | 90 | 90 |
| complete grounding | 22 | **23** |
| A* reached | 61 | 63 |
| **success** | 22 | **23** |
| outcome correct | 60 | 60 |
| online false completion | 0 | **0** |
| GT-goal mismatch completion | 0 | **0** |
| regressions | — | **none** |

Mean GT goal coverage on feasible trials: **0.510 → 0.535**, and
**kitchen 0.347 → 0.431**.

| domain | feasible success | outcome correct |
|---|---|---|
| kitchen | 3/18 | 21/36 |
| living room | **10/18** (was 9) | 22/30 |
| workshop | 10/24 | 17/30 |
| **all** | **23/60 (38.3%)** | 60/96 (62.5%) |

All 23 successes satisfy the **full GT goal set** and pass independent symbolic
validation. 18 of 23 are the **exact GT action sequence**; the 5 that differ
reach every GT goal by another legitimate route. Max **1** A* per trial,
**0** semantic FM requests, **0** harness failures.

---

## C. Exact changes

### C.1 `fix(vlm): resolve binding policy from the whole contract` — `2fde11fa`

**Bug.** `required_count` (how many times the task needs a participant) was
read correctly; `binding_policy` (how those occasions become physical things)
was taken from one field. The model writes DISTINCT for anything used more than
once, which is what repeated use looks like rather than what separate identity
means. **18 of 35 archived kitchen contracts asked for four spoons on a scene
holding three**; **10 asked for two coffee jars or two kettles** where one of
each exists.

**Implementation.** `resolve_binding_policy` corrects DISTINCT → REUSABLE only
when *all* hold: the model wrote DISTINCT for >1 occasion; the runtime
positively declares that role reusable across applications; and **nothing in the
contract asks for separate instances** — not the model's words about that
participant, not the phrase of any relation or operation naming it, not a second
declaration of the same canonical role, not a one-for-one pairing with a
participant that is itself several. `required_count` is never lowered.
Correspondingly, grounding now offers a reusable role its smallest instance
count first: a stirrer allowed two spoons consumed two of the scene's three, and
the two soup utensils the task genuinely wants severally were then reported
undiscovered.

**Why legal.** Joint interpretation of the model's own fields plus the runtime's
declaration of its own physics. The resolution runs while compiling, which takes
no observed scene at all — two tests assert that structurally, by signature.

**Applied (frozen):** 36 overrides — kitchen `coffee_stirrer` 17,
`coffee_source` 10, `water_source` 7; living room `REMOTE` 1; workshop
`repair_target` 1.

**Kept DISTINCT**, correctly: every `soup_eating_utensil` — the runtime calls
it one-per-application *and* the instruction says "its own".

### C.2 `fix(living): read a relation against the form its own operation uses` — `55fd3616`

**Bug.** One generically-worded participant is already realized per operation, so
a single "surface" role standing for both the personal and the shared support
gets a role per form. The relations were left behind: read before the operations
are seated, they still name the raw role, which keeps only one form. So "the
entertainment control is supported by the surface" was checked against the
*personal* support and reported unrepresentable, while the control's own
placement had already compiled against the shared one.

**Implementation.** A relation is re-pointed only when exactly one form's
operation also involves the relation's other participants. Several fits, or
none → the relation stands as written and fails closed.

**Frozen delta.** L4/trial_02 → GT goal coverage 1.0, ten-step plan; living-room
feasible success 9/18 → 10/18.

### C.3 `fix(vlm): a quality the instruction never asked for is the model's proposal` — `d167cbaa`

**Bug.** A required property the runtime has no predicate for held every
candidate for its role unproven, permanently. One contract described the soup
bowl as "heat resistant" for a task that never mentions heat; both bowls sat in
plain view, matched the role's categories, and were reported undiscovered
because no physical evidence can settle a quality the runtime cannot measure.

**Implementation.** Such a property is recorded as
`MODEL_PROPOSED_UNVERIFIABLE_PROPERTY` and kept in the accounting. A quality the
**instruction** names still blocks — the fail-closed reading. 4 archived kitchen
trials carried one.

### C.4 `test(workshop): record why a per-view extent cannot rescue the fastener` — `340f6d73`

See §E. Net source change to the workshop: **none**.

---

## D. Kitchen binding / cardinality audit

Of 35 compiled kitchen trials, **19 (54%) asked for more of something than the
scene contains** before this pass: 18 over-demanded spoons, 10 declared a
single-instance source as two distinct, 9 both.

| | |
|---|---|
| stirrer declared DISTINCT and resolved REUSABLE | 17 |
| source (`coffee_source`/`water_source`) resolved REUSABLE | 17 |
| `soup_eating_utensil` — kept DISTINCT (correct) | all |
| trials binding strictly more roles than before | **10** |

K3/trial_01 went from `INSUFFICIENT_SCENE_OBJECTS_FOR_ROLES` to **no unbound
roles at all**. Kitchen feasible success is still 3/18 because the remaining
kitchen blockers are elsewhere (§F), but mean GT goal coverage rose 0.347 →
0.431 and the cardinality semantics are now globally consistent.

---

## E. Workshop perception: measured, and reverted

The screw/driver defect is real and located. The size prior reads the **bounding
extent of the fused cloud** — the union of every view's mask — so one mask
straying onto a neighbour makes the whole track measure large, and a drawer
holding a screw and a driver comes back holding two drivers and no screw.

The obvious repair — measure per view, take the median, so one straying mask
cannot decide — was **implemented and replayed over all 30 workshop trials**:

| workshop | exec | grounded | success | outcome correct |
|---|---|---|---|---|
| before | 21/30 | 10/30 | **10/30** | 17/30 |
| per-view median | 13/30 | 6/30 | **6/30** | 11/30 |

It is wrong, and the replay says why: **drivers began disappearing**
(W9/trial_02 lost both `fastener` and `driver`). A driver lying in a drawer is
partly occluded, so a close inspection view frames only the shank; that view
measures under the 8 cm threshold on its own, a median over views then calls a
driver small, and the prior zeroes the only labels a driver has. The union is
precisely what rescues an elongated object.

**Conclusion: the extent is not separable from the association problem.** Fixing
this needs association-level work — refusing a merge whose observations imply
incompatible physical size families, or carrying per-detection size evidence
through fusion — not a better estimator over the same merged cloud. That is a
perception redesign with its own before/after, deliberately not started at the
end of a semantic pass.

The change is **reverted**; the workshop is byte-for-byte as it was, and the
replay above confirms 10/30 restored. What landed is the evidence: tests pinning
the constraint that defeats the per-view estimator. **W10 remains correctly
infeasible throughout — no screw is ever found there.**

---

## F. Remaining failures, classified

Of 60 feasible trials: 23 succeed. The 37 that do not:

| cause | trials | category |
|---|---|---|
| kitchen: unverifiable/UNKNOWN candidate evidence after binding (see below) | ~14 | D / G |
| workshop `fastener` lost in track fusion | 7 | **D — perception, §E** |
| FM omitted the destination or the seating semantics entirely (L6, L3, kitchen person-endpoints) | 6 | A |
| FM response failed strict wire validation | 4 | A |
| no runtime role for a soup-material source or an intermediate cooking vessel | 4 | C |
| workshop W3: FM never expressed a seatable fastening operation | 3 | A |
| `repair_target` unbound (W9/03, W4/01) | 2 | G |

**Planning failures: 0.** Every complete grounding yields a validated plan.

---

## G. Variant-wise final state

`start` = success at `16206178`. Three trials per variant.

### Kitchen

| variant | scene | feasible | exec | grounded | success | outcome | GT cov | start | first blocker |
|---|---|---|---|---|---|---|---|---|---|
| K1 | F0_ALL_VISIBLE | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.33 | 0/3 | unbound: coffee_container (1); unbound: soup_eating_utensil (1); unbound: coffee_source,soup_eating_utensil (1) |
| K2 | F1_HIDDEN_COFFEE_VESSEL | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.42 | 0/3 | unbound: soup_eating_utensil (2); unbound: coffee_source (1) |
| K3 | F2_HIDDEN_SOUP_BOWL | yes | 1/3 | 0/3 | **0/3** | 0/3 | 0.00 | 0/3 | contract not executable (2); unbound: coffee_source,soup_eating_utensil (1) |
| K4 | F3_HIDDEN_VESSELS_MIXED | yes | 2/3 | 0/3 | **0/3** | 0/3 | 0.50 | 0/3 | unbound: soup_eating_utensil (1); contract not executable (1); unbound: coffee_container,coffee_source (1) |
| K5 | F4_TOOLS_IN_DRAWERS | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.67 | 2/3 | contract not executable (1) |
| K6 | F5_FULL_DISTRIBUTED_SEARCH | yes | 3/3 | 1/3 | **1/3** | 1/3 | 0.67 | 1/3 | unbound: soup_eating_utensil (1); unbound: coffee_source (1) |
| K7 | I0_MISSING_COFFEE_VESSEL | no | 3/3 | 0/3 | **0/3** | 3/3 | 0.50 | 0/3 | unbound: coffee_container,coffee_source (1); unbound: coffee_source (1); unbound: coffee_container (1) |
| K8 | I1_MISSING_SOUP_BOWL | no | 0/3 | 0/3 | **0/3** | 3/3 | 0.00 | 0/3 | contract not executable (3) |
| K9 | I2_MISSING_COFFEE_SPOON | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.33 | 0/3 | unbound: coffee_stirrer (2); contract not executable (1) |
| K10 | I3_MISSING_SOUP_UTENSIL | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.17 | 0/3 | contract not executable (2); unbound: coffee_source,soup_eating_utensil (1) |
| K11 | I4_MISSING_KETTLE | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.17 | 0/3 | contract not executable (2); unbound: water_source (1) |
| K12 | I5_MISSING_COFFEE_JAR | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.17 | 0/3 | unbound: coffee_source (1); unbound: coffee_source,soup_eating_utensil (1); FM wire invalid (1) |

### Living Room

| variant | scene | feasible | exec | grounded | success | outcome | GT cov | start | first blocker |
|---|---|---|---|---|---|---|---|---|---|
| L1 | F0_ALL_OBJECTS_IN_STAGING | yes | 3/3 | 2/3 | **2/3** | 2/3 | 0.78 | 2/3 | unbound: SEATING_POSITION (1) |
| L2 | F1_LEFT_SAUCER_PREPLACED | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.67 | 2/3 | FM wire invalid (1) |
| L3 | F2_LEFT_SAUCER_ON_SHARED | yes | 0/3 | 0/3 | **0/3** | 0/3 | 0.22 | 0/3 | FM wire invalid (2); contract not executable (1) |
| L4 | F3_LEFT_CUP_ON_SHARED | yes | 3/3 | 3/3 | **3/3** | 3/3 | 1.00 | 2/3 | — |
| L5 | F4_SAUCER_PREPLACED_CUP_ON_SHARED | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.67 | 2/3 | contract not executable (1) |
| L6 | F5_LEFT_PAIR_ON_SHARED | yes | 1/3 | 1/3 | **1/3** | 1/3 | 0.33 | 1/3 | contract not executable (2) |
| L7 | I0_NO_SHARED_TABLE | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.44 | 0/3 | contract not executable (1); unbound: SEATING_POSITION,SHARED_REMOTE_REGION (1); unbound: SHARED_REMOTE_REGION (1) |
| L8 | I1_NO_LEFT_PERSONAL_TABLE | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.11 | 0/3 | contract not executable (2); unbound: PERSONAL_CUP_SAUCER_REGION (1) |
| L9 | I2_NO_PERSONAL_TABLES | no | 0/3 | 0/3 | **0/3** | 3/3 | 0.00 | 0/3 | contract not executable (2); FM wire invalid (1) |
| L10 | I3_NO_TABLES | no | 1/3 | 0/3 | **0/3** | 3/3 | 0.00 | 0/3 | contract not executable (2); unbound: CUP_SAUCER_SET,PERSONAL_CUP_SAUCER_REGION,SEATING_POSITION,SHARED_REMOTE_REGION (1) |

### Workshop

| variant | scene | feasible | exec | grounded | success | outcome | GT cov | start | first blocker |
|---|---|---|---|---|---|---|---|---|---|
| W1 | F0_MANUAL_FIRST_ONE_REGION | yes | 2/3 | 2/3 | **2/3** | 2/3 | 0.78 | 2/3 | contract not executable (1) |
| W2 | F1_POWER_FIRST_ONE_REGION | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.22 | 0/3 | unbound: fastener (3) |
| W3 | F2_MANUAL_FIRST_TWO_REGIONS | yes | 0/3 | 0/3 | **0/3** | 0/3 | 0.00 | 0/3 | contract not executable (3) |
| W4 | F3_POWER_FIRST_TWO_REGIONS | yes | 3/3 | 0/3 | **0/3** | 0/3 | 0.33 | 0/3 | unbound: fastener (2); unbound: repair_target (1) |
| W5 | F4_MANUAL_FIRST_THREE_REGIONS | yes | 1/3 | 0/3 | **0/3** | 0/3 | 0.11 | 0/3 | unbound: fastener (1); contract not executable (1); FM wire invalid (1) |
| W6 | F5_POWER_FIRST_THREE_REGIONS | yes | 3/3 | 3/3 | **3/3** | 3/3 | 1.00 | 3/3 | — |
| W7 | F6_MANUAL_ONLY | yes | 2/3 | 2/3 | **2/3** | 3/3 | 1.00 | 2/3 | contract not executable (1) |
| W8 | F7_POWER_ONLY | yes | 3/3 | 3/3 | **3/3** | 3/3 | 1.00 | 3/3 | — |
| W9 | I0_NO_DRIVER | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.22 | 0/3 | unbound: repair_target (2); contract not executable (1) |
| W10 | I1_NO_SCREW | no | 2/3 | 0/3 | **0/3** | 3/3 | 0.22 | 0/3 | unbound: fastener (2); contract not executable (1) |

---

## H. Reproducibility

Front-half compile hash over every archived contract, identical across four
seeds (`REPRODUCIBILITY_AUDIT.txt`):

```
PYTHONHASHSEED 0 / 1 / 7 / 12345
4c6cd9c4333a9864c3c491aef14adba587a987988393c413c3229ce1529c48a5
```

Determinism test suite passes. The frozen archive is unmodified: 91
`raw_v3.json` on disk, 91 tracked, 0 changed.

---

## I. No-cheating audit

| check | result |
|---|---|
| GT/reference accesses in online code | **0** (`audit_no_gt_leakage.py`, 53 modules) |
| variant-specific branches (K/L/W identifiers) | **0** |
| expected-answer maps | **0** |
| scene-inventory-driven semantic repairs | **0** — asserted structurally by signature |
| `required_count` lowered to fit a scene | **0** — never lowered, only the mapping to objects is re-read |
| UNKNOWN counted as satisfied | **0** — complete grounding still requires TRUE |

---

## J. Freeze manifest

| item | value |
|---|---|
| git SHA | `340f6d73e2a3ac37083016aabe395cadcefe1425` |
| prompt | V3, **unchanged this pass** — `SYSTEM_PROMPT_V3` |
| schema | `TAMP_FM_SCHEMA_VERSION=3` |
| method manifest | `mujoco_scenes/functional_tamp_pipeline/method_freeze.json` |
| max tokens | 24000 |
| sampler | temperature 0.0, `enable_thinking` false |
| views | 3 RGB |
| perception config | `workshop_phase1_yoloworld_l_five_view_close.yaml` (unchanged) |

**The prompt was deliberately not changed.** The whole pass was measured against
frozen responses produced by the current prompt; changing it now would mean the
evidence in this report no longer describes the system that would run. The
generic DISTINCT/REUSABLE clarification the mandate contemplated is **already
handled in the backend** by C.1, which reads the contract rather than asking the
model to be more careful — and unlike a prompt change it required no new FM
calls to validate. If a prompt change is wanted before the live experiment, it
should be a separate step with its own SHA and its own smoke.

---

## K. Ready for the 10-trial experiment

Not run here, as instructed. Per-trial invariants the backend now enforces:
exactly 1 semantic FM call, 0 semantic retries, at most 1 A*, `high_level_replans=0`.
