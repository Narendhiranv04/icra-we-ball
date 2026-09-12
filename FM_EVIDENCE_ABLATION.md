# FM Evidence Ablation — design

The GT ablation asks *which evidence channels does grounding need, given a
perfect functional graph?*  This asks the same question of the graph the FM
actually produces, over the same seven masks, so the two tables can be read side
by side.

## What runs

```
archived FM response  ->  G_F  ->  mask  ->  ground_graph  ->  score
```

| | GT ablation | FM ablation |
|---|---|---|
| G_F source | `GTSpecProvider` (oracle) | archived FM response, via the shipped `VLMSpecProvider` |
| Observed graph | `build_oracle_graph` | the pipeline's own `observed_scene_graph.json`, from one unmodified run |
| Masking site | an `evidence_components` argument threaded through `ground_graph` | the specification, before grounding |
| Variance | 10 synthetic seeds over tie-breaks | archived trials (real FM sampling variance) |
| FM calls | none | none |

## Three design decisions, and why

**1. Nothing in the pipeline was modified.**  The mask clears the evidence
fields of G_F and the *unmodified* production grounder runs on the result.
This is not only less invasive than threading a parameter through
`ground_graph`; it is the stricter experiment, because the system under test is
the shipped one byte for byte and the ablation lives entirely in its input.
The two formulations coincide because the grounder skips a check exactly when
the field carrying it is empty — pinned by
`test_fm_evidence_ablation.py::test_masking_matches_component_gating`.

**2. Masking happens at the grounding boundary, not at `run_pipeline`.**  The
obvious design — feed a masked G_F back through the replay path — cannot work
and should not.  That path first calls the task-interface validator, whose job
is to reject a malformed FM contract, and an ablated graph is malformed by
exactly that standard (*"operation group has empty required_relations"*).
Routing ablations through it measures the validator, not the grounder: every
mask that removed a channel failed identically, for a reason having nothing to
do with evidence.  Verified before the design was changed — 5 of 7 conditions
died in the validator.

**3. Repeats replace seeds.**  The GT ablation needed synthetic seeded
tie-breaking because an oracle graph is deterministic, so an underconstrained
mask would otherwise be scored on whichever ordering happened to win.  The FM
pipeline has real sampling variance: the same variant yields a different G_F on
every trial.  Averaging over archived trials measures the actual distribution of
the method rather than a synthetic one.

## Two places the FM setting differs from the oracle setting

**`UNKNOWN` is a real third value.**  Oracle evidence is complete, so the GT
ablation could demand every role slot be `TRUE`.  Real perception returns
`UNKNOWN`, and demanding `TRUE` marks the *unablated* pipeline as failing.  Role
success is therefore "no slot was refuted", with confirmed/refuted/unknown
counts all reported separately.  `UNKNOWN` is never promoted to `TRUE`: a slot
the evidence contradicts still fails.

**`physical_preconditions` overrides `required_relations`.**  The FM's operation
groups often carry the former.  Clearing only `required_relations` would leave
binary geometry fully in force for those groups — an ablation that silently did
nothing while reporting that it had.  Both are cleared, and a test pins it.

## An early finding

Removing binary evidence can make grounding *fail* rather than succeed more
easily, which is the opposite of the oracle setting.  On `kitchen/K1` the FM
picks a `coffee_container` whose category the detector reports `UNKNOWN`; with
binary relations available the grounder resolves it, and without them it
declines to ground at all (`missing_roles: ['coffee_container']`).  That is the
conservative behaviour working: with less evidence the system refuses to
conclude rather than inventing an assignment.  The oracle graph has no
`UNKNOWN`s, so the GT ablation never exercised this path.

## Running it

```bash
python scripts/run_fm_evidence_ablation.py \
  --trials trial_01,trial_02,trial_03 \
  --output-root runs/fm_evidence_ablation/<stamp>
```

Makes no FM call, so it is reproducible, costs no tokens, and cannot disturb or
compete with a live run.

## Files

| Path | Role |
|---|---|
| `mujoco_scenes/fm_evidence_ablation.py` | the seven masks and the masking rules |
| `scripts/run_fm_evidence_ablation.py` | two-phase driver, scoring, summary |
| `mujoco_scenes/tests/test_fm_evidence_ablation.py` | 14 tests |

No file in `functional_tamp_pipeline/` is touched by any of this.
