> **Superseded by `FM_TAMP_CLOSURE_FINAL.md`.** Every `outcome correct`
> figure in this file was produced by a scoring rule that, on an infeasible
> variant, asked only whether the task went unsatisfied -- true by
> construction, so it credited all 36 infeasible trials automatically. The
> rule was replaced; the honest figure is 42/96, not 69-71/96. Feasible-trial
> success, goal coverage and the false-completion count in this file are
> unaffected.

# Performance-closure pass — report

| | |
|---|---|
| branch | `vlm-testing-pipeline` |
| starting SHA | `6e0f6bba` |
| final SHA | `22729350` |
| pipeline suite | **1149 passed, 2 skipped, 0 failed** |
| GT-leakage audit | **0 findings** over 53 online modules |
| FM calls made in this pass | **0** |

## C. Frozen replay, before → after (96 archived responses, identical inputs)

| metric | before | after |
|---|---|---|
| **feasible-task success** | **23 / 60** | **31 / 60** |
| kitchen feasible success | 3 / 18 | **11 / 18** |
| living room feasible success | 10 / 18 | 10 / 18 |
| workshop feasible success | 10 / 24 | 10 / 24 |
| executable contracts | 59 | 59 |
| **complete grounding** | **23** | **31** |
| successful full plans | 23 | **31** |
| outcome correct | 60 / 96 | **69 / 96** |
| online false completion | 0 | **0** |
| GT-goal mismatch completion | 0 | **0** |
| planner failures after complete grounding | 0 | **0** |
| A* reached | 63 | 62 |
| max A* per trial | 1 | 1 |
| mean GT goal coverage (feasible) | 0.510 | **0.635** |

All 31 successes satisfy the **full GT goal set** and pass independent symbolic
validation. **Zero success regressions.** A* reached drops by one because two
partially-grounded trials now stop at grounding instead of running a candidate
A*; neither was ever a success.

## A. Code changes

### A.1 `semantic_compiler.py` — operation reuse follows the resolved role

*Old:* an operation's `usage_policy` was decided from the raw contract, before
the role's binding policy was resolved against the whole graph. *New:* where the
resolution relaxed a source to reusable, the consuming operation follows it.

A stirrer the graph says one of will do sat inside a stirring operation still
demanding one per application. The grounder believed the operation, took two of
the scene's three spoons, and the two soup utensils the task *does* want
severally were reported undiscovered. **This single defect accounts for most of
the +8.** Guarded to fire only where the runtime itself declares that role
reusable across applications — the same physical fact the resolution used — and
never where the group was asked to keep its sources distinct.

### A.2 `fm_schema_v3.py` — aggregate counts partition by functional form

*Old:* a generic participant projected into several forms kept its aggregate
count on whichever form stayed behind, and that form was chosen alphabetically.
*New:* each form takes the count of the operations that induced it, and the form
kept on the raw role is the one the participant already resolved to.

One "spoon" role counted four times, for two stirrings and two servings, left
the form behind it demanding four of itself. Keeping the resolved form matters
because every relation naming the participant is still read against it —
choosing alphabetically moved a workshop "workbench" off the site fastened into
and the relation saying the component must fit that site lost its endpoint.

### A.3 `semantic_compiler.py` — binding resolution made real and symmetric

*Old:* the "two declarations of one canonical role" evidence compared
`canonical_role` between raw FM roles, which never carry that field, so it was
dead code; and resolution only ever relaxed DISTINCT. *New:* the mapping comes
from the role hypotheses, and resolution also **tightens** REUSABLE/SHARED to
DISTINCT where the contract asks for separate instances and the runtime says one
instance cannot serve several applications — the direction that costs score and
buys soundness, since the permissive word would otherwise satisfy a two-utensil
requirement with one object.

### A.4 `operation_slot_completion.py` — prefer a reading that invents nothing

*Old:* "the fastening tool is placed on the workbench" — two named participants
— was read as a *fastening*: a fastener was synthesized for the target slot and
the invented participant then had no runtime role. *New:* where exactly one
nominated capability seats every participant the model named, no placeholder is
created.

Deliberately limited to participants whose own resolved type fits the slot.
Reading one in a *different* form was implemented and **reverted inside this
pass**: it decides a participant's meaning earlier than the projection does, and
re-reading a workshop "workbench" as the surface tools are left on cost W8,
which had been succeeding on all three trials.

### A.5 `semantic_compiler.py` — returns to a place the planner owns

An operation putting something back on a fixed place the planner already owns is
audited as a domain transition rather than reported unrepresentable — and only
when the operation's own resolved slot assignment names that constant.

### A.6 `runtime_functional_semantic_ontology.yaml` — the supply vessel's names

Four kitchen trials bound every role but the coffee supply; in one the supply was
detected and labelled `coffee canister`, and the acceptance vocabulary listed a
tin and a jar but not a canister. Canister and caddy are the same closed
dry-goods vessel the role already accepts. Serving vessels — cup, mug, bowl —
stay absent, and a test confirms they still normalize to themselves.

## B. Tests added (`test_closure_pass_semantics.py`, 25 total)

| test | protects against |
|---|---|
| `test_operation_reuse_follows_the_resolved_role_policy` | the A.1 defect, on the real archived contract |
| `test_a_distinct_source_keeps_its_operation_dedicated` | relaxing an operation over a non-reusable role |
| `test_aggregate_participant_count_is_partitioned_across_forms` | the four-times spoon keeping its aggregate |
| `test_two_declarations_of_one_canonical_role_are_seen_through_the_mapping` | the dead `canonical_role` comparison |
| `test_a_permissive_policy_is_tightened_when_the_contract_asks_severally` | SHARED silently satisfying a several requirement |
| `test_a_permissive_policy_over_a_reusable_role_is_left_alone` | over-tightening |
| `test_binding_policy_resolution_cannot_see_the_scene` | scene inventory reaching cardinality, **by signature** |
| `test_resolution_is_identical_whatever_the_scene_would_hold` | non-determinism in resolution |
| `test_separate_identity_wording_defeats_the_override` (×4 wordings) | relaxing "its own" / "separate" / "individual" |
| `test_wording_on_an_unrelated_operation_is_not_evidence` | cross-operation evidence leakage |
| `test_a_quality_the_instruction_never_asks_for_is_the_models_proposal` | FM-invented properties blocking valid bindings |
| `test_track_extent_robustness.py` (5) | the per-view extent estimator that destroys driver labels |

## D. Trial-level changes

Eight newly succeeding, all at **GT goal coverage 1.0**, all kitchen:

| trial | before | after | why |
|---|---|---|---|
| K1/01, K1/02 | grounding: soup utensils unbound | **success** (24-step plan) | stirring no longer consumes two spoons (A.1) |
| K2/02, K2/03 | grounding: utensil unbound | **success** (26-step) | A.1 |
| K3/02 | grounding: `coffee_source` unbound | **success** | A.1 + A.6 |
| K4/01 | grounding: containers unbound | **success** (26-step) | A.1 |
| K6/02 | grounding: utensil unbound | **success** (26-step) | A.1 |
| K6/03 | grounding: `coffee_source` unbound | **success** | A.6 (canister accepted) |

Two changed without changing success: K4/02 and K10/02 no longer run a candidate
A* on a partial grounding. **No success regressions.**

One regression was found *and reverted inside the pass*: W8 (3/3 succeeding)
lost executability to an over-eager participant re-reading in A.4. Caught by a
static sweep over all 96 contracts, traced by file-level bisection, reverted,
and re-verified.

## E. Remaining feasible failures (29 of 60)

| count | classification | recoverable? |
|---|---|---|
| 14 | **compiler / FM omission** — W3 ×3, L3/03, L5/03, L6 ×2, W1/02, W7/02, K3 ×2, K4/02, K5/01, L6/03 | partly: W3 and W7/02 are the strongest remaining candidates; W1/02 needs the per-operation form choice that A.4's reverted half would have given |
| 6 | **perception** — `fastener` lost in track fusion (W2 ×3, W4 ×2, W5/03) | needs association-level redesign; the per-view estimator was proven wrong in the previous pass and its refutation is pinned by tests |
| 4 | **generation / wire** — L2/02, L3/01, L3/02, W5/02 | FM output defects; L3 cases include a generation truncation |
| 4 | **grounding** — K2/01, K3/01, K4/03, W4/01 | K2/01's supply vessel is detected with *no label at all*, so no acceptance vocabulary can admit it; W4/01 is `repair_target` |
| 1 | **perception** — unlabelled vessel | as above |

Honest assessment of the 14: I inspected each raw response. W3's three trials do
contain fastening participants and the compiler does produce
`DRIVE_FASTENER_INTO_TARGET`; the blocker is a second operation, so the mandate
is right that "FM never expressed fastening" was the wrong label. I did not get
to a generic fix for it in this pass.

## F. Soundness audit

| check | result |
|---|---|
| GT/inventory leakage in online code | **0 findings**, 53 modules |
| variant-specific branches (K/L/W) | **0** |
| expected-answer maps | **0** |
| scene inventory influencing cardinality | **0** — asserted by signature, not by sampling |
| `required_count` lowered to fit a scene | **never** — only the mapping to objects is re-read |
| UNKNOWN counted as satisfied | **no** — complete grounding still requires TRUE |
| false completion | **0 → 0** |
| GT-goal mismatch completion | **0 → 0** |

## H. Recommendation

**1. Is the deterministic backend close enough to freeze?** For the semantic
half, yes. The cardinality-propagation family is closed, and its refutations are
pinned by tests. What remains is concentrated in perception (6 trials) and in
contract-level cases needing per-operation form selection (W3, W1/02, W7/02).

**2. New frozen feasible-success count: 31 / 60** (from 23), with kitchen
tripling from 3/18 to 11/18 and zero false completions.

**3. High-confidence recoverable failures still open?** Yes, three:
W3 ×3 (a second operation blocks an otherwise complete fastening contract),
W1/02 and W7/02. Each needs per-operation functional form selection done
*after* operations are seated — the mechanism exists; wiring it into capability
selection without the W8 regression is the next piece of work. I would expect a
few more trials from it, not a step change.

**4. Ready for a small live smoke?** Yes — with the caveat in §G of
`FM_TAMP_FINAL_FREEZE.md`: the archive was generated with a different sampler
and prompt, so frozen replay is a backend regression benchmark and not a
prediction of fresh-response behaviour. The smoke is exactly the right next step
to characterize that.

**5. Ready for the full 10×32?** Not until the smoke has been read. The runner
and aggregator are in place and the backend is materially stronger, but
committing 320 calls before seeing how the current prompt and sampler behave
would risk spending the budget on a configuration nobody has looked at.
