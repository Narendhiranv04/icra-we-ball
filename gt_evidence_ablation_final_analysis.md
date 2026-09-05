# GT Evidence Ablation — Final Seeded Evaluation Summary and Analysis

## 1. Executive summary

The final seeded ablation run is substantially more meaningful than the earlier single deterministic run.

The experiment now evaluates all seven evidence conditions over the existing 32 benchmark variants and repeats each condition with 10 controlled seeds, giving:

- **3 domains**
- **32 existing variants**
- **7 evidence masks**
- **10 seeds**
- **2,240 total evaluations**
- **200 feasible trials per condition**
- **120 infeasible trials per condition**

The key improvement is that underconstrained ablations no longer receive credit simply because one deterministic ordering happened to select a good assignment. Seeded tie-breaking varies only among candidates or pairings that are equally admissible under the currently enabled evidence.

The final results are internally coherent and make substantially more scientific sense than the previous table.

The most important findings are:

1. **Full semantic + unary + binary evidence is fully stable:** `full` achieves **100% strict GT task success and 100% infeasible rejection across all 10 seeds**.
2. **Semantic evidence is crucial for role identity in Kitchen.** Without semantics, physically coherent but functionally wrong mug/bowl assignments are common.
3. **Binary geometry is crucial for target-specific pairing.** In Kitchen, semantics can identify the correct object classes but still choose physically incompatible spoon→container pairings. In Living Room, semantics can identify the correct support regions but cannot reliably pair personal supports with the correct seat/context.
4. **Unary evidence by itself is weak.** `unary_only` achieves only about **11% aggregate strict task success**.
5. **However, unary evidence is redundant once semantic + binary evidence are already available on the current fixed benchmark objects.** `no_unary` remains **100%**, identical to `full`.
6. **Workshop is not strongly discriminative for the semantic/binary ablations under the current fixed variants.** Semantic-only, binary-only, and every two-component condition remain 100%.
7. The earlier extremely high single-run results were partly caused by deterministic selection luck. The 10-seed experiment exposes this clearly, especially in Kitchen and unary-only Workshop.

---

## 2. Final implementation state

Branch:

```text
baseline_execution
```

Base ablation commit:

```text
55e86c50da665e8ac6a11e11cd074c0f24e3f15b
Evaluate GT evidence ablations through grounded task planning
```

Final seeded-ablation commit:

```text
630d0da56761342aae5f69e3fe6b18da862aaab8
Evaluate GT ablations across seeded grounded plans
```

Only the following files were changed:

```text
mujoco_scenes/functional_tamp_pipeline/grounding.py
mujoco_scenes/run_gt_evidence_ablation.py
mujoco_scenes/tests/test_gt_evidence_ablation_conditions.py
```

No unrelated files were committed.

Production grounding remains deterministic. Seeded selection is enabled only explicitly for the ablation using:

```text
selection_policy="seeded_ablation"
selection_seed=<seed>
```

The seed affects only ordering among assignments/pairings that are already admissible under the active evidence mask.

The priority is stable and reproducible through SHA-256 over:

```text
seed
+ domain/variant namespace
+ role/group scope
+ candidate or pair identity
```

This avoids dependence on Python's randomized `hash()` or incidental iteration order.

---

## 3. Evidence masks

| Condition | Semantic | Unary | Binary |
|---|---:|---:|---:|
| `semantic_only` | ✓ | ✗ | ✗ |
| `unary_only` | ✗ | ✓ | ✗ |
| `binary_only` | ✗ | ✗ | ✓ |
| `no_binary` | ✓ | ✓ | ✗ |
| `no_unary` | ✓ | ✗ | ✓ |
| `no_semantic` | ✗ | ✓ | ✓ |
| `full` | ✓ | ✓ | ✓ |

The narrow Kitchen source exception remains:

- kettle = fixed water-source identity
- coffee jar = fixed coffee-source identity

All other Kitchen objects obey the active evidence mask.

---

## 4. Metric interpretation

The final evaluation correctly distinguishes between several different notions of success.

### Diagnostic-only metrics

These should **not** be treated as headline task success:

- `grounding_complete`
- `plan_generated`
- `replay_valid`

They only answer whether graph slots could be filled, whether a symbolic plan could be produced from that grounding, and whether that plan is internally valid relative to the same symbolic problem.

A wrong grounding can still generate and replay a perfectly valid symbolic plan.

### Role-level correctness

The selected object/region is evaluated against full GT role evidence.

### Pair/binding-level correctness

The exact selected pair/context is checked using full GT binary relations.

Kitchen:
- `INSERTABLE_IN`
- `REACHES_BOTTOM`

Living Room:
- `FITS_SET_ON`
- `NEAR_SEAT`
- `FITS_ON`
- `ACCESSIBLE_FROM_BOTH_SEATS`

Workshop:
- `COMPATIBLE_WITH`
- `REACHES_TARGET`
- `COMPATIBLE_WITH_TARGET`

### Plan-level correctness

The actual ablation-selected grounding is passed to the planner.

The plan is then scored using:

- GT task-plan validity
- exact plan match to paired `full`
- match to any full reference plan
- normalized LCS similarity
- strict GT task success
- strict full-method match

This is much stronger than reporting whether A* merely found a plan.

---

## 5. Aggregate results

Percentages are mean ± population standard deviation across ten seeds.

| Condition | Role GT | Role match | Pair GT | Pair match | GT-valid plan | Exact plan | Any-full plan | Plan LCS | Strict GT | Strict full | Infeasible rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `semantic_only` | 100±0 | 89±4.36 | 74±8.60 | 55.5±7.89 | 74±8.60 | 61.5±7.76 | 87±5.57 | 0.8863±0.0189 | 74±8.60 | 72±7.48 | 100±0 |
| `unary_only` | 14±7.00 | 11±5.83 | 25.5±6.50 | 7.5±5.59 | 11±5.83 | 11.5±6.73 | 18.5±5.50 | 0.5326±0.0414 | 11±5.83 | 10±5.00 | 83.33±0 |
| `binary_only` | 77±3.32 | 77±3.32 | 100±0 | 77±3.32 | 77±3.32 | 77±3.32 | 77±3.32 | 0.8872±0.0189 | 77±3.32 | 77±3.32 | 100±0 |
| `no_binary` | 100±0 | 89±4.36 | 74±8.60 | 55.5±7.89 | 74±8.60 | 61.5±7.76 | 87±5.57 | 0.8863±0.0189 | 74±8.60 | 72±7.48 | 100±0 |
| `no_unary` | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 1.0±0 | 100±0 | 100±0 | 100±0 |
| `no_semantic` | 77±3.32 | 77±3.32 | 100±0 | 77±3.32 | 77±3.32 | 77±3.32 | 77±3.32 | 0.8872±0.0189 | 77±3.32 | 77±3.32 | 100±0 |
| `full` | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 1.0±0 | 100±0 | 100±0 | 100±0 |

### Immediate interpretation

The aggregate results reveal three very clear patterns:

- `semantic_only == no_binary`
- `binary_only == no_semantic`
- `no_unary == full`

This means unary evidence does not change the final solution distribution on the current fixed benchmark once semantic and/or binary evidence already provide the relevant discrimination.

That should be interpreted as **benchmark-specific redundancy**, not as evidence that unary geometry is universally unnecessary.

---

## 6. Seed variability

Strict GT task success by seed:

```text
semantic_only: 80, 75, 85, 60, 70, 80, 75, 85, 60, 70
unary_only:    10, 10,  5, 15, 15, 10,  5, 25,  5, 10
binary_only:   80, 80, 70, 75, 75, 80, 75, 80, 80, 75
no_binary:     80, 75, 85, 60, 70, 80, 75, 85, 60, 70
no_unary:     100,100,100,100,100,100,100,100,100,100
no_semantic:   80, 80, 70, 75, 75, 80, 75, 80, 80, 75
full:         100,100,100,100,100,100,100,100,100,100
```

This validates the seeded evaluation.

Reduced-evidence conditions are genuinely underconstrained, while `full` is correct across every seed.

---

## 7. What changed relative to the old single-run result

| Condition/domain | Old | New seeded result |
|---|---:|---:|
| Kitchen semantic-only | 100% | **63.33%** |
| Kitchen no-binary | 100% | **63.33%** |
| Workshop unary-only | 87.5% | **15%** |
| Aggregate unary-only | 35% | **11%** |
| Living Room semantic-only | 0% | **50%** |

The earlier deterministic result was therefore both overly optimistic in some cases and overly pessimistic in Living Room semantic-only.

The 10-seed evaluation is much more defensible.

---

# 8. Domain-wise analysis

## 8.1 Kitchen

| Condition | Role GT | Pair GT | GT-valid plan | Strict GT | Strict full | LCS |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_only` | 100 | 63.33 | 63.33 | 63.33 | 56.67 | 0.8208 |
| `unary_only` | 23.33 | 61.67 | 13.33 | 13.33 | 10 | 0.5614 |
| `binary_only` | 23.33 | 100 | 23.33 | 23.33 | 23.33 | 0.6241 |
| `no_binary` | 100 | 63.33 | 63.33 | 63.33 | 56.67 | 0.8208 |
| `no_unary` | 100 | 100 | 100 | 100 | 100 | 1.0 |
| `no_semantic` | 23.33 | 100 | 23.33 | 23.33 | 23.33 | 0.6241 |
| `full` | 100 | 100 | 100 | 100 | 100 | 1.0 |

Kitchen is the cleanest domain for showing semantic/binary complementarity.

### Semantic evidence solves role identity

`semantic_only` gets 100% Role GT but only 63.33% Pair GT.

A representative failure selected:

```text
s1i_oversized_spoon -> ab3_narrow_deep_cup
```

The tool was semantically valid as a spoon but failed `INSERTABLE_IN`.

Thus semantics can identify the object class without identifying the correct physical pairing.

### Unary-only cannot distinguish functional roles

Both coffee vessels and soup bowls satisfy `OPEN_CAVITY`.

Representative failure:

```text
coffee_container = ab3_shallow_bowl
soup_container   = ab3_medium_deep_mug
```

Both satisfy the unary geometry but their semantic roles are reversed.

### Binary-only solves physical pairing but not functional identity

`binary_only` gets 100% Pair GT but only 23.33% Role GT.

This is exactly the intended distinction between physical compatibility and functional role semantics.

### Unary adds no incremental gain over semantic + binary

`no_unary = full = 100%`.

For the current fixed Kitchen object set, all final semantic+binary selections already satisfy the relevant unary constraints.

---

## 8.2 Living Room

| Condition | Role GT | Pair GT | GT-valid plan | Strict GT | Strict full | LCS |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_only` | 100 | 50 | 50 | 50 | 50 | 0.8000 |
| `unary_only` | 3.33 | 3.33 | 3.33 | 3.33 | 3.33 | 0.6639 |
| `binary_only` | 100 | 100 | 100 | 100 | 100 | 1.0 |
| `no_binary` | 100 | 50 | 50 | 50 | 50 | 0.8000 |
| `no_unary` | 100 | 100 | 100 | 100 | 100 | 1.0 |
| `no_semantic` | 100 | 100 | 100 | 100 | 100 | 1.0 |
| `full` | 100 | 100 | 100 | 100 | 100 | 1.0 |

The GT functional graph always retains:

- two distinct personal supports
- one shared support
- no cross-role support reuse

These are structural constraints, not ablated observation evidence.

### Semantic-only gets the coarse support-region set right

Role GT is 100%.

The current support set and structural distinct/shared constraints are enough to recover the correct physical region set.

### Semantic-only fails at target/context pairing

Pair GT is only 50%.

A representative failure assigns the correct two personal regions but pairs them to the opposite `SEATING_POSITION`.

The failing relation is `NEAR_SEAT`.

### Binary-only solves the current Living Room task

Binary-only reaches 100% using:

```text
Personal:
FITS_SET_ON
NEAR_SEAT

Shared:
FITS_ON
ACCESSIBLE_FROM_BOTH_SEATS
```

This result is logically consistent with the current benchmark.

---

## 8.3 Workshop

| Condition | Strict GT | Infeasible rejection |
|---|---:|---:|
| `semantic_only` | 100 | 100 |
| `unary_only` | 15 | 0 |
| `binary_only` | 100 | 100 |
| `no_binary` | 100 | 100 |
| `no_unary` | 100 | 100 |
| `no_semantic` | 100 | 100 |
| `full` | 100 | 100 |

Workshop is the least discriminative domain for semantic-vs-binary complementarity.

The fixed variants contain compatible manual/power drivers, the compatible screw, and a hammer distractor.

Semantic-only and binary-only are therefore each sufficient.

Unary-only is highly underconstrained.

Representative wrong assignment:

```text
driver   = wooden hammer
fastener = long Phillips driver
```

This explains its 15% strict task success and 0% Workshop infeasible rejection.

---

# 9. Infeasible-case interpretation

Aggregate infeasible rejection:

```text
semantic_only = 100%
binary_only   = 100%
no_binary     = 100%
no_unary      = 100%
no_semantic   = 100%
full          = 100%

unary_only    = 83.33%
```

The unary-only reduction is consistent with its severe role ambiguity, especially in Workshop.

---

# 10. Full-method stability

`full` achieves:

```text
100% GT role correctness
100% GT pair correctness
100% GT-valid task plans
100% strict GT task success
100% infeasible rejection
```

for every seed.

Full is not byte-identical across seeds because multiple GT-valid symmetric solutions exist.

Kitchen:
- equivalent valid spoon allocations
- unordered vessel assignments

Living Room:
- symmetric personal binding orderings

Workshop:
- manual or power driver can both be valid

Thus correctness is stable even when the exact symbolic witness is not unique.

---

# 11. Does the final experiment make sense?

## Yes.

The results now align well with the information represented by each evidence family.

### Kitchen

```text
semantic -> functional identity
binary   -> pairwise physical compatibility
unary    -> coarse object-local geometry
```

Observed:
- semantic-only: roles correct, pairings often wrong
- unary-only: roles badly ambiguous
- binary-only: physical pairings correct, semantic roles often wrong
- semantic + binary: fully correct

### Living Room

Observed:
- semantic information is enough for the coarse support-region set
- binary/context geometry is needed for target-specific seat/support pairing
- binary relations are sufficient overall on the current variants

### Workshop

Observed:
- semantic alone is sufficient
- binary alone is sufficient
- unary alone is insufficient

This is coherent with the current fixed object set.

---

# 12. Important caveats

## Unary is not shown to be incrementally necessary

The strongest caveat is:

```text
semantic_only == no_binary
binary_only   == no_semantic
no_unary      == full
```

Therefore avoid saying:

> all three evidence components are individually necessary.

A better statement is:

> Unary geometry contributes object-local filtering, but the current benchmark does not contain a case where it adds discriminative power beyond semantic and binary evidence jointly.

## Full-match is not the same as GT correctness

For example:

```text
semantic_only:
Strict GT   = 74%
Strict full = 72%
```

A GT-valid plan can differ from the finite set of full-method reference plans observed across 10 seeds.

Therefore:
- **Strict GT** should be the primary correctness metric.
- full-plan matching and LCS should be secondary method-comparison metrics.

## Workshop should be interpreted separately

The 100% Workshop rows are a property of the current fixed variants, not evidence that semantic/binary complementarity is universally unnecessary.

## Seeds measure ambiguity, not perception noise

The ten seeds vary only among alternatives that are equally admissible under the ablated evidence.

They measure **underdetermination from missing evidence**, not stochastic detector/VLM noise.

---

# 13. Strongest paper-level takeaway

> Across 32 benchmark variants and 10 controlled grounding tie-break seeds, full semantic, unary, and binary evidence achieved 100% task-level correctness. Ablating semantic evidence preserved physical pair compatibility but reduced correct functional-role grounding, most visibly in the Kitchen, where geometrically valid bowls and mugs could be assigned to the wrong task roles. Conversely, removing binary relations preserved semantic role identification but caused incorrect tool-target or region-context bindings, including incompatible spoon-container assignments and incorrect personal-seat associations. Unary evidence alone was highly underconstrained. However, removing unary evidence from the semantic+binary condition caused no degradation on the fixed benchmark object set, indicating that unary constraints provide local filtering but are not independently discriminative once semantic identity and pairwise compatibility are available in these scenes. Seeded evaluation further showed that several strong single-run ablation scores were artifacts of deterministic tie-breaking rather than evidence sufficiency.

---

# 14. Recommended main paper table

| Evidence | Role GT | Pair GT | GT-valid plan | Strict task success | Infeasible rejection |
|---|---:|---:|---:|---:|---:|
| Semantic only | 100 | 74 | 74 | 74 | 100 |
| Unary only | 14 | 25.5 | 11 | 11 | 83.33 |
| Binary only | 77 | 100 | 77 | 77 | 100 |
| Semantic + Unary | 100 | 74 | 74 | 74 | 100 |
| Semantic + Binary | 100 | 100 | 100 | 100 | 100 |
| Unary + Binary | 77 | 100 | 77 | 77 | 100 |
| Full | 100 | 100 | 100 | 100 | 100 |

Plan-match/LCS and per-seed variation can go in the appendix or supplementary material.

The primary story should be about:
1. correct role grounding,
2. correct pair/context binding,
3. correct grounded task plan,

not simply whether a planner produced some sequence.

---

# 15. Result artifacts

Final run directory:

```text
runs/gt_evidence_ablation/seeded10_fullplan_compare_20260905T_final_v2/
```

Main outputs:

```text
summary.csv
summary.json
results.json
failure_breakdown.json
seed_summary.json
plan_comparison.json
```

Per-case layout:

```text
<domain>/<variant>/<condition>/seed_<0-9>.json
```

Each case contains the selected/full role assignments, bindings, GT validations, both plans, replay/GT-plan validity, plan comparison, LCS, strict metrics, and first-failure diagnostics.

---

# 16. Final assessment

The final seeded experiment is **internally consistent and scientifically usable**.

The clearest supported story is:

```text
SEMANTIC
    -> functional role identity

BINARY GEOMETRY
    -> pair/context compatibility

SEMANTIC + BINARY
    -> correct grounded task plan
```

Unary geometry remains a valid local constraint family, but the current fixed benchmark does not show that it is incrementally required once semantic and binary evidence are jointly available.

Most importantly, the results are no longer determined by one arbitrary deterministic grounding choice. The 10-seed evaluation exposes the actual ambiguity of reduced-evidence conditions while showing that the full method remains correct across all controlled tie choices.
