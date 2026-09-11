# Object roles as well as region roles: closure evidence

Every number here comes from replaying the **same 96 archived FM responses**
through two versions of the deterministic pipeline. Zero FM calls, zero harness
failures, at most one A* invocation per trial. `FROZEN_REPLAY_BEFORE.json` is
the code at `c35d062e`; `FROZEN_REPLAY_AFTER.json` is the code at this pass's
head. Nothing in the online path reads ground truth, and
`NO_GT_LEAKAGE_AUDIT.txt` reports 0 findings over the modules it scans.

## What this pass was about

The living room existed in the benchmark to test **shared versus distinct
objects being placed**, and it was scoring 0 of 6. The mapping from FM roles to
runtime roles handled regions and dropped objects: a participant the model
declared `entity_kind: OBJECT` was canonicalized onto a *region* whenever the
categories it listed happened to read like a surface.

So `refreshment_setting` -- "holds refreshments for a person", offered as
"table, surface, tray" -- came back as `PERSONAL_CUP_SAUCER_REGION`, the side
table it belongs on. The payload the task is actually about never entered the
graph, and every operation over it went with it. Downstream, the detector
vocabulary is built from the compiled graph's categories, so in one variant the
cups, saucers and *seats* were never even looked for: the observed scene graph
held three regions and a remote, and nothing else.

## Movement attributable to this pass

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
| regressions | -- | **none** |

Per domain, feasible success and outcome-correct over 96 trials:

| domain | feasible success | outcome correct |
|---|---|---|
| kitchen | 3/18 -> 3/18 | 21/36 -> 21/36 |
| living room | 7/18 -> **9/18** | 20/30 -> **22/30** |
| workshop | 10/24 -> 10/24 | 17/30 -> 17/30 |

On the living room's ten variants at `trial_01`, feasible success goes from
**2 of 6 to 4 of 6** (L1, L2, L4, L5), each at GT goal coverage 1.0, with
8 of 10 outcomes correct and no false completion.

### Attribution, stated plainly

Read against the previously published artifact
(`final_semantic_closure_20260911T005451`) the same table reads success 10 ->
22 and outcome correct 47 -> 60. **Most of that movement is not this pass.**
Four commits landed between that artifact and this pass's baseline, and one of
them -- `5cc21285`, restoring the drive-slot bound and applying the size prior
everywhere -- is what lifts the workshop from 0 to 10. The baseline replayed
here was taken at `c35d062e` precisely so this pass is not credited with it.
This pass's own contribution is the two living-room trials named above, and the
mechanism that makes object roles compile at all.

## The five defects

1. **The declared entity kind was never read.** The wire contract asks, for
   every participant, whether the thing is carried or is an area. Role typing
   worked from the function text alone, and that text describes a thing by where
   it ends up at least as often as by what it is.

2. **A capability slot could substitute one participant for another.**
   `CUP_SAUCER_SET`, pinned by everything around it, was seated as `REMOTE` on
   the strength of both being things one carries. Two capabilities could then
   seat the placement, and slot completion refused the tie -- so the operation
   the task is mostly about was dropped as unrepresentable.

3. **An operation's phrase reached other operations.** Two placements share the
   seating they are arranged around, so "place the control where it is
   accessible to both people" licensed a two-seat anchor for the *drinkware*
   placement as well, manufacturing the same tie.

4. **No cross-participant constraint.** A participant the contract
   characterizes by function but not by category stayed ambiguous across the
   domain's payloads even when another participant was already pinned to the
   alternative.

5. **`operation_count` and `required_count` were never reconciled.** One
   contract declared two DISTINCT refreshment settings *and* wrote "Place two
   refreshment settings near the seating area" with a count of one. Both of its
   own statements say two; honouring the count field alone provided one occasion
   to place two things and silently dropped half the requirement.

A sixth change, `arrang` as a placement stem, exists because the participant's
own name is stripped from the operation phrase before capability matching, which
left the bare verb as the only signal in "arrange_refreshment_setting".

## One rule tried, measured, and withdrawn

The entity-kind reading was first written to strike any family the runtime
realizes solely with roles of the opposite class. It looked safe and was not. A
workshop contract declared its **workbench surface** an `OBJECT`; SUPPORT came
from the role's name and job description, INSTRUMENT from an incidental cue.
Striking SUPPORT left the incidental cue standing, **the bench became the
screwdriver**, it collided with the real tool, and W4 lost a contract it
previously had. The frozen replay's regression check is what caught it.

The surviving rule is narrower and is the one the case actually needed: a
*category list* naming only families of the other class cannot outrank the job
description. It carries every living-room contract and leaves the workbench a
workbench. The broader rule is not in the tree; see `8d6f2cba`.

Two further narrowings came from tests that encoded real distinctions:
`FIXED_TARGET` is in neither the carried nor the area class, because the model
uses it for both and treating it as stationary stopped a workpiece declared
`OBJECT` from being the repair target; and both new readings apply only to
participants whose family the contract's own words settled, because a
participant it characterizes in no way -- a person, "recipient of the meal",
categories "human, guest" -- is ambiguous over the whole domain for the correct
reason. Narrowing that one left a set small enough to seat, and "serve the soup
to the diner" compiled into putting the bowl into the person.

## Plans against the GT action catalogue

Of the 22 trials reporting a complete plan, **all 22** satisfy the full GT goal
set and **all 22** pass independent symbolic validation. 17 of the 22 are also
the **exact GT action sequence** (structural match: same operators in the same
order, object arguments used consistently):

| domain | exact / complete |
|---|---|
| living room | **9 / 9** |
| workshop | 8 / 10 |
| kitchen | 0 / 3 |

The five that differ are not failures: each reaches every GT goal by another
legitimate route. Two workshop plans return the tool to a different storage
region than the catalogue names; three kitchen plans carry a vessel before
pouring where the catalogue pours first.

## What remains, by category

### (A) Semantics the FM did not express

- **L6** declares no seating and no accessibility at all -- only
  "refreshment_setting is located on surface". Supplying the seat anchor was
  tried in an earlier pass and reverted (`470f7c58`): the requirement that a
  placement be reachable from both seats is the task semantics, not geometry the
  runtime happens to own. That decision stands.
- **L3** declares `surface` as "a flat area to hold refreshment items", count 2
  DISTINCT, and then reuses that same role for the remote. The shared central
  region was never expressed, and supplying a third region would create a
  participant.
- Kitchen contracts naming a **person** as a physical endpoint
  (`serve_soup(soup_bowl, recipient_person)`) are classified as abstract
  directives rather than compiled. Inventing a physical endpoint there means
  moving a bowl into a person.

### (E) Documented irreducible ambiguity -- the dominant kitchen blocker

**18 of 35 kitchen trials ask for more spoons than the scene contains.** The
instruction says "Serve each soup bowl with **its own** suitable eating utensil"
and says nothing about own-ness for the stirrer; the GT uses one stirrer twice
and two distinct soup utensils, and the scene holds exactly three spoons. Those
18 trials declare the stirrer `required_count: 2, DISTINCT`, so they need four.
That is unsatisfiable however the grounding is arranged, and the contract is
rejected.

Lowering DISTINCT to REUSABLE would raise the success rate and would be
discarding a requirement the model stated. The instruction marks distinctness
explicitly where it wants it; the model does not exploit the contrast.

### (C) A missing or incompatible runtime role

- No **soup-material source** role exists, so `Pour SoupSupply into
  SoupContainer` cannot be represented. `semantic_typing` preserves this as a
  representability failure and never relabels the material as the bowl.
- No **intermediate cooking vessel** role exists, so `pour(cooking_vessel,
  coffee_cup)` and `Pour Coffee(pot, mug)` cannot be seated.

### (D) Perception

None outstanding in the two kitchen variants examined. K1 detects all nine
objects (2 cups, 2 bowls, 3 spoons, kettle, coffee jar) and K4 detects all nine
including the cup found in `C2` and the bowl in `B1` -- exactly the GT
inventory. Both then fail for the spoon reason above, not for a detection
reason.

## Soundness invariants held

- at most **1** A* invocation per trial
- **0** semantic FM requests on a frozen replay, by construction
- **0** harness failures across 96 trials
- **0** false completions, and **0** completions that disagree with the GT goal set
- front-half compilation hash identical across `PYTHONHASHSEED` 0, 1, 7, 12345
- GT-leakage audit: **0** findings
- full pipeline test suite: **1121 passed, 3 skipped, 0 failed**
- the frozen archive is unmodified: 91 `raw_v3.json` on disk, 91 tracked, 0 changed
