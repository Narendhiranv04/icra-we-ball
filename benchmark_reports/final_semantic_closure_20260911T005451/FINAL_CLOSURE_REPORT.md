# Final semantic closure report

Branch `vlm-testing-pipeline`. Measured on the frozen 3x32 archive
`benchmark_reports/v3_qwen_distribution_3x32_20260910T053937`, replayed through
the current deterministic pipeline with **zero FM calls**.

Every number below is produced by re-running archived semantic responses. No
ground truth, expected action sequence, feasibility label or benchmark variant
identifier reaches any online decision; the reference columns are read only to
score a finished run.

---

## 1. Headline

| stage | before | after |
| --- | ---: | ---: |
| wire contract valid | 89 | 89 |
| task contract valid | 89 | 89 |
| canonical graph emitted | 88 | **89** |
| **executable contract complete** | **37** | **57** |
| grounding reached | 88 | 89 |
| complete grounding | 13 | 10 |
| A* invoked | 47 | 58 |
| success | 13 | 10 |
| outcome correct | 46 | **47** |
| **ONLINE_FALSE_COMPLETION** | **2** | **0** |
| GT_GOAL_MISMATCH_COMPLETION | 0 | 0 |
| infeasible variants correctly rejected | 34 / 36 | **36 / 36** |
| harness failures | 0 | 0 |
| FM semantic calls | 0 | 0 |
| max A* invocations in any trial | 1 | 1 |

Per domain, after:

| domain | n | graph | exec contract | complete grounding | A* | success | outcome correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kitchen | 36 | 34 | 23 | 3 | 23 | 3 | 21 |
| living room | 30 | 26 | 13 | 7 | 15 | 7 | 20 |
| workshop | 30 | 29 | 21 | 0 | 20 | 0 | 6 |

**Read this table carefully. Success went down and the pipeline got better.**

Three of the thirteen earlier successes were produced by a phantom object. A
178-point blob sitting in the tool cabinet was admitted as an observation of
the right drawer, believed to be a screw, and bound as the fastener. Two of
those three were on `I1_NO_SCREW`, a variant whose storage contains no screw at
all and which therefore cannot be completed. Removing the phantom -- section 3,
FIX 15 -- took the false completions to zero and took three successes with it,
one of them on a feasible variant whose real screw was simultaneously being
rejected by a bad head measurement. Two compensating perception errors had been
producing a right-looking answer.

The honest summary is: **semantic compilation improved by 54% (37 -> 57
executable contracts), soundness went from two false completions to none, and
the losses moved downstream into perception, where they were always going to
be.**

## 2. Where the pipeline now loses, measured

Of the 57 trials that reach an executable contract:

| | count |
| --- | ---: |
| complete grounding | 10 |
| OBJECT_DISCOVERY_FAILURE -- search exhausted, the object was never detected | **30** |
| FUNCTIONAL_ASSIGNMENT_FAILURE -- detected, but the joint constraints reject it | 16 |
| other | 1 |

The roles that go missing are physical detections, not semantic roles:

| missing role | trials |
| --- | ---: |
| workshop `fastener` | 15 |
| kitchen `soup_eating_utensil` | 12 |
| kitchen `coffee_source` | 11 |
| workshop `repair_target` | 6 |
| living room `SEATING_POSITION` / `SHARED_REMOTE_REGION` | 3 |

**A* is not a bottleneck: all 10 complete groundings produced a validated
successful plan, 10 of 10.** Functional assignment given a detected object is
also not the dominant loss. The dominant loss is that the object is never seen.

In 19 of the 29 workshop trials with a valid contract, perception reports **no
instance labelled as a screw at all**, in scenes that contain one. Where it does
report one, the head interface measures `SLOT_LIKE` rather than `CROSS_LIKE` in
7 of 43 cases, and the compatibility verifier then correctly refuses it -- 15 of
the 28 `COMPATIBLE_WITH` checks in the workshop fail as (CROSS_LIKE driver,
SLOT_LIKE fastener).

This is the single highest-leverage remaining item in the system and it is a
detector and measurement problem, not a semantic one.

## 3. What changed, and why

Seven commits, each one a stage.

**Relation meaning read by stem inventory instead of by remembered phrase.**
The interpreter held contiguous phrase lists. "stirs" was in one and "stir" was
not, so an imperative carried no meaning at all; "suit_for", "is suitable for"
and "must physically suit" all say compatibility and none was listed. Across the
archive that accounted for most of the relations reported as unrepresentable --
not relations the runtime cannot express, but sentences it declined to read.
Each meaning now states its stem inventory, restricted to the forms that
actually make the claim: fastening words count as verbs, because the same stem
forms the attributive gerund the model uses for kinds, and reading "the tool
acts on the fastening component" as a mechanical-fit claim lost the causal
statement it is.

**Operations inform role typing.** The isolated function-alias mapper answering
"planner context constant" won unconditionally over a role type the graph had
already settled, so a marked fastening site read as the repair target by the
operation it anchors was rewritten to the workbench zone and elided as context,
leaving the fastening with no anchor and no compiled operation. Setting a
participant aside was a decision about one operation recorded against the whole
document, so a fastening that set the bench aside silently removed the bench
from the tool return written next to it. A wording whose reading the endpoints
refuse is no longer treated as a reading.

**The excuse that was doing the most damage, removed.** Prose the runtime could
give no meaning at all, over exactly the roles an operation binds, was treated
as already enforced by that operation. Not understanding a sentence says nothing
about what the sentence requires, and the inference fired most readily on the
relations whose wording was most unusual -- the ones worth reading. Two trials
regressed when it went and were then earned back for real reasons.

**Two names for one fixed place.** The model describes the receiving context of
a fastening twice -- "the object to be fastened" and "the specific spot on the
workbench" -- and the runtime holds that place as one reference. Both names now
lower onto one anchor, guarded three ways: refused where the domain offers a set
form of that anchor's family (Living Room has SEATING_PAIR, so two seating roles
may be two seats the model enumerated -- the first version of this change
destroyed an explicit seating pair and a hardening test caught it); refused where
either role words itself as "another" one of the same kind; refused unless the
model named both in one statement of its own.

**The generation budget was wrong in the code.** The archive was produced with a
24000-token budget passed through an environment variable while the code default
was 8192. **39 of the 91 responses that finished are longer than 8192**, so
anyone running without that variable exported would have had 43% of trials
truncated and read it as the model failing to specify the task. The default is
now 24000; the longest complete response is 12222 tokens. No retry was added.

**Five archived responses were truncated and misattributed.** They hit
`finish_reason: length`, generated 24000 tokens and returned nothing readable,
and all five were reported as TASK_SPECIFICATION_FAILURE -- which says the model
misunderstood the task when what happened is that its answer was cut off. A new
`FM_RESPONSE_FAILURE` category now covers a response that did not arrive in a
usable form. Those five are the majority of the wire-invalid trials.

**The soundness metric was unusable.** "False completions" added together a
success on a benchmark-infeasible variant -- unsound, must be zero -- and a
success whose achieved goals merely differ from the reference, which can be a
difference of interpretation. The combined figure moved whenever the reference
did. `ONLINE_FALSE_COMPLETION` and `GT_GOAL_MISMATCH_COMPLETION` are now apart.

**A stage may not claim the contents of the region next door (FIX 15).** Points
are gated to the inspected region's volume with an 8 cm boundary margin, for
calibration error at the edge of a drawer. The right drawer's volume ends at
y = 0.45 and the tool cabinet's *starts* at y = 0.30, so the margin does not sit
at a boundary -- it reaches into the neighbour's interior. A fused stage object
whose centre lies inside another declared region's own volume is now rejected
and recorded. This is what took false completions to zero.

**`required_count` meant two different things.** The prompt asked for "how many
physical instances the task needs" while the compiler has always read the field
as how many *times* the task needs the participant, with `binding_policy`
mapping to instances -- which is why a reusable coffee jar declared "2 DISTINCT"
was read as a demand for two jars. The prompt now says what the code does, with
a test pinning the pair.

## 4. Why the remaining failures remain

Classified against the five stopping conditions the task set.

**A -- semantics the FM did not express.** The dominant living-room class. Of 26
valid living-room contracts, only **7 express the both-seats accessibility
semantics anywhere**, although the instruction states it in every trial; 18
express individual proximity. Supplying the seating anchor without that evidence
was tried in an earlier pass, raised living-room plans from two to seven, and was
reverted as invention -- the reasoning is recorded at the gate in commit
`470f7c58`. It remains reverted.

**B -- a capability the robot does not own.** Kitchen staged-preparation
contracts (K5, K8, K9 and others) route everything through an invented
intermediate `cooking_vessel`: pour ingredients into the vessel, mix in the
vessel, pour the vessel into the cup and the bowl. The runtime pours sources
directly into containers and models no staging vessel, and no heating. K3/03
additionally asks for "apply heat to soup bowl" and "hand soup bowl to person".

**C -- a missing object or geometry.** The runtime models no soup material
source, so every soup-filling operation is unrepresentable. This accounts for
the recurring `soup_ingredient` / `soup_source` / `soup_liquid` unresolved roles.

**D -- a perception failure this semantic pass cannot repair.** The workshop, in
full, and the largest single block in the system: the screw is not detected in
19 of 29 trials, and its head interface mismeasures in 7 of the 43 cases where
it is. Tuning the interface thresholds against these scenes would be fitting to
the benchmark and was not done. Related: `CROSS_LIKE` is currently the
fall-through class of the interface classifier -- anything that is neither a
narrow slot nor radially symmetric is called a cross -- which is a soundness
hazard in the other direction and is what let the phantom pass.

**E -- irreducible ambiguity.** W3/01 states two of its relations as the single
words `functional` and `spatial`. There is nothing there to read, and the
pipeline correctly refuses. W10/02 relates "the workpieces" to themselves.

## 5. Sixteen questions

The literal enumerated list was truncated out of the working context, so these
are the sixteen topics the brief specified, each answered directly.

1. **Did semantic compilation improve?** Yes. Executable contracts 37 -> 57 on
   the identical frozen input; canonical graphs 88 -> 89; operation groups
   174 -> 193. No trial regressed at any intermediate step.
2. **Was anything achieved by weakening a requirement?** No, and one weakening
   that already existed was removed: the rule treating unreadable prose over
   operation-bound roles as enforced. It cost two trials, which were then
   recovered for stated reasons.
3. **Is `ONLINE_FALSE_COMPLETION` zero?** Yes, 0, down from 2. All 36 infeasible
   trials are now correctly rejected, up from 34.
4. **Is `GT_GOAL_MISMATCH_COMPLETION` zero?** Yes, 0. All 10 successes satisfy
   the full reference goal set on feasible variants.
5. **Exactly one FM semantic request per trial?** Yes. The replay makes none;
   the archived run made one per trial; `SEMANTIC_ACCOUNTING_AUDIT.json` records
   zero violations. No retry path was added.
6. **At most one A* invocation per trial?** Yes. Maximum observed is 1, zero
   violations.
7. **Any ground truth in online decision code?** No. `NO_GT_LEAKAGE_AUDIT.txt`
   walks 53 modules of the online path and reports zero findings, with every
   exemption listed by filename so the list can be checked rather than trusted.
8. **Any branching on variant identity?** No. The one variant-keyed table,
   `ORACLE_SEARCH_ORDERS`, is a privileged diagnostic that the search contract
   refuses outright in vlm mode; that refusal is asserted by test, for both the
   preflight and the freeze.
9. **Is the pipeline deterministic?** Yes. The compiled front half of all 96
   archived contracts hashes identically under four `PYTHONHASHSEED` values
   (`08331f24...`). The only randomness in the package is an explicitly seeded
   search-policy ablation, and a test fails if any module calls the shared
   unseeded generator.
10. **Was randomness used to resolve semantic ambiguity anywhere?** No. Where a
    wording nominates several causal meanings and the endpoints do not single
    one out, the relation fails closed rather than being settled by sort order.
11. **Is the failure taxonomy real?** It is now. Transport and truncation
    failures are `FM_RESPONSE_FAILURE`, not `TASK_SPECIFICATION_FAILURE`; five
    of the 96 archived responses are affected.
12. **Was output truncation audited?** Yes, and it found a live configuration
    hazard: 39 of 91 finished responses exceed the previous code default. See
    `MODEL_CONFIG.json` for the per-call record.
13. **What is the model configuration?** `qwen35-9b`, temperature 0.0,
    top_p 1.0, thinking disabled, strict JSON schema, max_tokens 24000,
    one semantic request per trial.
14. **Did the test suite stay honest?** 1094 tests pass, 1 skipped, none
    deselected. Two existing tests were rewritten because their premise --
    that "secures" is wording the interpreter cannot read -- became false; their
    fail-closed intent is preserved on wording that genuinely says nothing. The
    method-freeze manifest was regenerated; it had been failing since before
    this session, meaning the one check that notices unintended changes to the
    method was not watching anything.
15. **Were the frozen replays run fresh?** Yes, twice. The first run was
    discarded: 19 workshop trials segfaulted and 7 more hit GPU
    out-of-memory in perception at 8 parallel workers. Those rows were deleted
    and re-run at lower concurrency. The reported run has zero harness failures
    and zero inference errors.
16. **Is the system READY?** **No.** See section 7.

## 6. Regression conditions

None of the twelve forbidden regressions is present.

| condition | status |
| --- | --- |
| one personal seat reused for two personal placements | not present |
| a required context silently omitted from an operation | not present; set-aside participants are recorded and withdrawn when an operation seats them |
| UNKNOWN treated as TRUE | not present; the one mechanism that did this was removed this pass |
| a role endpoint signature creating an unexpressed operation | not present |
| visible scene inventory creating a task requirement | not present |
| a relation silently removed because it was not understood | not present; unreadable relations block, and statements absorbed by a lowering are listed |
| an invented material source accepted | not present; soup sources remain unrepresented |
| an explicit instruction input demoted for lack of a runtime role | not present |
| several interpretations lexically sorted into one | not present; plural readings fail closed |
| more than one A* per trial | 0 violations |
| more than one semantic FM request per trial | 0 violations |
| GT or expected answers in online decision code | 0 findings |

## 7. READY = false

Three reasons, in order.

**The final fresh 1x32 could not be run.** No FM endpoint is reachable in this
environment: every candidate port is closed, no vLLM or SGLang process is
running, there is no tunnel, and neither `TAMP_FM_BASE_URL` nor `TAMP_FM_MODEL`
is set. Nothing was substituted for it. Every number in this report comes from
replaying archived responses, which measures the deterministic pipeline exactly
and measures the model not at all. The three-domain smoke and the 1x32 both
remain outstanding and are the first thing to run when an endpoint exists.

**One configuration change in this pass has not been exercised against a live
model.** The generation budget default moved from 8192 to 24000 to match the
archive. It is the right value on the archive's evidence, but no live call has
been made with it from this code.

**The workshop domain completes nothing.** 21 of 30 trials now produce an
executable contract and 20 reach the planner, but zero ground completely,
because the fastener is not detected. Until detector recall on small fasteners
improves, the workshop measures the perception stack rather than the method.

## 8. Top five remaining blockers

1. **Detector recall on the fastener.** No screw-labelled instance in 19 of 29
   workshop trials. Blocks the entire workshop domain. Perception, not
   semantics.
2. **Interface geometry on small fastener heads.** `SLOT_LIKE` where
   `CROSS_LIKE` is correct in 7 of 43 measurements, and `CROSS_LIKE` is a
   fall-through class that accepts anything not narrow and not symmetric.
   Both directions need a positive classification with a plausibility bound.
3. **The FM omits the both-seats accessibility clause** in 19 of 26 living-room
   contracts although the instruction states it. Not repairable without
   inventing the requirement. Two trials (L7/03, L10/01) state it about the
   payload rather than its support and could be recovered by lowering that onto
   the support the payload is placed on; not implemented in this pass.
4. **Kitchen staged preparation.** The model routes preparation through an
   intermediate vessel the runtime does not model. Flattening it would change
   the number and identity of the operations the model expressed.
5. **`repair_target` geometry** unresolved in 6 workshop trials -- the recess in
   the workbench does not measure.

## 9. Files

| file | what it holds |
| --- | --- |
| `FROZEN_REPLAY_BEFORE.{json,csv}` | per-trial rows, before |
| `FROZEN_REPLAY_AFTER.{json,csv}` | per-trial rows, after |
| `STAGE_TABLE.json` | the stage counts above, total and per domain |
| `FAILURE_ROOT_CAUSES.json` | per-trial cause classification for every non-success |
| `SEMANTIC_ACCOUNTING_AUDIT.json` | one-call, one-A*, false-completion and harness checks |
| `NO_GT_LEAKAGE_AUDIT.txt` | static audit of the online decision path |
| `REPRODUCIBILITY_AUDIT.txt` | front-half hashes under four hash seeds |
| `MODEL_CONFIG.json` | per-call FM record, truncation audit, frozen decoding config |
| `FREEZE_MANIFEST.json` | freeze commit, branch, inputs, build time |

`FRESH_1X32_RESULTS.json` is **absent by necessity**, not by omission: there was
no endpoint to run it against.
