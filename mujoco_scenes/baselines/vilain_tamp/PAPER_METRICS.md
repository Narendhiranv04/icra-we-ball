# ViLaIn-TAMP Paper-Ready Evaluation Metrics Specification

This document formally specifies the evaluation metrics, action-sequence funnel, statistical conventions, causal classifications, and ground-truth isolation guarantees for the ViLaIn-TAMP-Qwen baseline across benchmark domains (`kitchen`, `living_room`, `workshop`).

---

## 1. Core Experimental Principles

1. **The Primary Output is the Action Sequence**:
   The primary output of ViLaIn-TAMP for any task is the **action sequence** it generates. Evaluation is sequence-centric: we measure whether ViLaIn-TAMP generates valid symbolic plans, binds physical entities, refines continuous kinematic trajectories, and executes actions that induce goal state progress. ViLaIn-TAMP is never evaluated against human/teleoperated demonstration sequences or proposed-method action sequences; it is evaluated against the physical benchmark world and its canonical goal requirements.

2. **Strict Post-Terminal Isolation**:
   Ground-truth requirements, feasibility labels ($G_F$), and hidden physical assertions are strictly post-terminal. They are NEVER accessed, leaked, or referenced during baseline perception, VLM prompt construction, PDDL compilation, Fast Downward search, entity resolution, or geometric refinement.

3. **Explicit Denominators and Population Scoping**:
   Every rate metric must state its exact numerator and denominator. No conditional rate may be reported without its sample size:
   - **All Non-Infrastructure Runs ($N_{\text{total}}$)**: Default population for unconditional metrics (Outcome Correct, Plan Generation, Resource Counts).
   - **Ground-Truth Feasible Runs ($N_{\text{feasible}}$)**: Mandatory population for goal-attainment metrics (Feasible-Task Success, Goal Coverage, Sequence-Induced Goal Gain, Physical Plan Found). Infeasible variants have no valid solution trajectory; including them in goal coverage artificially deflates or conflates baseline performance.
   - **Declared Completion Runs ($N_{\text{declared}}$)**: Denominator for False Completion Rate.

4. **Zero-Step Plan Prohibition**:
   Zero-step plans (`len == 0`, empty action sequence) occur when Qwen's generated `:init` state logically entails its generated `:goal`. While Fast Downward trivially exits with success in 0 steps, zero physical actions are executed.
   - Zero-step plans **NEVER** count toward `Physical Plan Found` ($len > 0$ required).
   - Zero-step plans **NEVER** count toward `Non-empty PLAN@EXEC`.
   - If the baseline declares completion on a zero-step plan while physical benchmark requirements remain unmet, it is counted as a **False Completion**.

---

## 2. Manuscript Main-Table Metrics

The paper main table reports the following seven primary metrics plus the definitive goal-gain metric:

| Metric Name | Formal Definition | Numerator | Denominator | Better | Target Population |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Outcome Correct (%)** | Proportion of runs where binary decision matches ground truth feasibility and execution outcome | $\sum [\text{feasible} \land \text{success}] + \sum [\text{infeasible} \land \text{rejected}]$ | All non-infrastructure runs ($N_{\text{total}}$) | Higher | All Runs ($N=32$ smoke, $N=320$ full) |
| **Feasible-Task Success (%)** | Proportion of ground-truth feasible runs achieving all canonical physical requirements | Runs with `actual_task_success == True` | Ground-truth feasible runs ($N_{\text{feasible}}$) | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **Goal Coverage (%)** | Fraction of canonical task requirements satisfied in the terminal physical simulation state | $\sum \text{requirements passed}$ | $\sum \text{requirements total}$ | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **$\Delta$ Goal Coverage (%)** (Sequence-Induced Goal Gain) | Net goal progress induced by executing the generated action sequence: $\text{Coverage}_{\text{terminal}} - \text{Coverage}_{\text{initial}}$ | $\text{Terminal passed} - \text{Initial passed}$ | $\sum \text{requirements total}$ | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **False Completion (%)** | Proportion of declared task completions where physical requirements were not satisfied | Runs declaring completion with `actual_task_success == False` | Runs declaring completion ($N_{\text{declared}}$) | Lower | Declared-Complete Runs |
| **Physical Plan Found (%)** | Proportion of feasible runs producing a valid, non-empty, geometrically refined physical plan | Feasible runs with non-empty plan passing refinement | Ground-truth feasible runs ($N_{\text{feasible}}$) | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **Raw VLM Requests** | Total foundation model (VLM) API requests per run | Sum of object, init, goal, and CP model calls | Completed runs ($N_{\text{total}}$) | Context | All Runs |
| **High-Level Replans** | Number of corrective planning (CP) iterations invoked following planner/refiner failures | Total CP iterations requested | Completed runs ($N_{\text{total}}$) | Context | All Runs |

### Aggregation Conventions
- **Micro Goal Coverage**: Pooled across all requirements in the population: $\frac{\sum_{i} \text{passed}_i}{\sum_{i} \text{total}_i}$.
- **Macro Goal Coverage**: Unweighted mean of variant coverage scores: $\frac{1}{|V|} \sum_{v \in V} \frac{\text{passed}_v}{\text{total}_v}$.
- **Uncertainty Intervals**: 95% Wilson score confidence intervals for all binomial proportions; mean $\pm$ standard deviation for continuous distributions.

---

## 3. Initial Goal Coverage and Sequence-Induced Goal Gain

In manipulation benchmarks, complex scenes often satisfy some task requirements in the initial state prior to any robot motion (e.g., doors already closed, containers already positioned, default alignments).

To prevent attributing passive initial state properties to baseline competence, we evaluate:
1. **Initial Goal Coverage Snapshot**:
   Evaluated post-terminally against the initial MuJoCo simulation state before any action is executed.
   $$\text{Coverage}_{\text{initial}} = \frac{\sum_{r \in \text{Feasible}} \text{RequirementsPassed}_{\text{initial}}(r)}{\sum_{r \in \text{Feasible}} \text{RequirementsTotal}(r)}$$
2. **Terminal Goal Coverage**:
   Evaluated against the final physical simulation state after baseline termination.
   $$\text{Coverage}_{\text{terminal}} = \frac{\sum_{r \in \text{Feasible}} \text{RequirementsPassed}_{\text{terminal}}(r)}{\sum_{r \in \text{Feasible}} \text{RequirementsTotal}(r)}$$
3. **Sequence-Induced Goal Gain ($\Delta$ Goal Coverage)**:
   $$\Delta \text{Goal Coverage} = \text{Coverage}_{\text{terminal}} - \text{Coverage}_{\text{initial}}$$

$\Delta$ Goal Coverage is the definitive metric for whether generated action sequences actually achieve physical progress. A baseline that executes zero actions or fails during refinement will have $\Delta \text{Goal Coverage} \equiv 0.0\%$, even if static scene layout yields $\text{Coverage}_{\text{terminal}} > 0\%$.

---

## 4. Full 7-Stage Action Sequence Funnel

The action sequence lifecycle is tracked across 7 sequential stages:

$$\text{PLAN@FD} \longrightarrow \text{NONEMPTY PLAN@FD} \longrightarrow \text{PLAN@VAL} \longrightarrow \text{PLAN@IDENTITY} \longrightarrow \text{PLAN@REFINE} \longrightarrow \text{PLAN@EXEC} \longrightarrow \text{TASK@FINAL}$$

| Funnel Stage | Formal Definition | Inclusion Criteria |
| :--- | :--- | :--- |
| **1. PLAN@FD** | Fast Downward produced a parseable action sequence from Qwen's PDDL problem | Action sequence found ($len \ge 0$) |
| **2. NONEMPTY PLAN@FD** | Fast Downward produced a plan with at least one physical action | $len > 0$ |
| **3. PLAN@VAL** | The action sequence was verified by VAL against domain semantics | VAL validator returns exit code 0 |
| **4. PLAN@IDENTITY** | Symbolic objects bound to physical MuJoCo bodies without unresolvable ambiguity | Entity resolution succeeds |
| **5. PLAN@REFINE** | Continuous joint trajectories found passing IK and collision checking | Cloned MuJoCo refinement succeeds |
| **6. PLAN@EXEC** | Non-empty trajectory dispatched and executed in physical simulation | Physical execution attempted & completed |
| **7. TASK@FINAL** | Final physical scene satisfies all canonical benchmark requirements | `actual_task_success == True` |

Two rates are reported for every stage:
- **Unconditional Conversion Rate**: Fraction of the entire run population ($N_{\text{total}}$) reaching the stage.
- **Conditional Stage Transition Rate**: Transition probability from the immediate previous stage: $\frac{\text{Count}(\text{Stage}_k)}{\text{Count}(\text{Stage}_{k-1})}$.

---

## 5. Feasibility Classification Semantics

Feasibility classification evaluates the baseline's ability to correctly discriminate feasible from infeasible task variants.

### Binary Classification Matrix
Positive class ($P$) = **Infeasible Variant**; Negative class ($N$) = **Feasible Variant**.

- **True Positive (TP)**: Ground-truth infeasible variant correctly declared infeasible or cleanly rejected.
- **False Positive (FP)**: Ground-truth feasible variant incorrectly declared infeasible (false infeasibility declaration).
- **True Negative (TN)**: Ground-truth feasible variant where a valid plan was found and executed.
- **False Negative (FN)**: Ground-truth infeasible variant where the baseline incorrectly declared completion.

### Diagnostic Classification Metrics
- $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- $\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$
- $\text{Specificity (Feasible Recall)} = \frac{TN}{TN + FP}$
- $\text{Infeasible Recall (Sensitivity)} = \frac{TP}{TP + FN}$
- $\text{Decision Coverage} = \frac{TP + TN + FP + FN}{N_{\text{total}}}$

---

## 6. Neutral Diagnostic Causal Labels

Earlier development iterations employed overclaimed causal labels such as `GENUINE_GEOMETRIC_FAILURE` or speculative scene descriptions. These have been replaced with neutral, verifiable diagnostic categories based directly on observable software components:

### 6.1 Identity Resolution Failures
- **`MULTIPLE_PHYSICAL_CANDIDATES`**: Perception/candidate universe returned $> 1$ physical bodies matching the semantic type within the spatial region, and the resolver could not disambiguate.
- **`NO_PHYSICAL_CANDIDATE`**: Candidate universe returned 0 physical bodies matching the required type and spatial envelope.
- **`SINGLE_PHYSICAL_CANDIDATE_BUT_UNRESOLVED`**: Exactly one candidate body existed, but attribute or affordance verification failed.

### 6.2 Refinement Failures
- **`REFINEMENT_IK_REJECTION`**: Inverse kinematics solver failed to find a collision-free joint configuration for the waypoint (replaces `GENUINE_GEOMETRIC_FAILURE`).
- **`REFINEMENT_COLLISION_REJECTION`**: Collision checking detected body-body contact along the interpolated trajectory.
- **`REFINEMENT_SKILL_ENVELOPE_REJECTION`**: Target pose fell outside the reachability envelope of the parameterized skill.
- **`REFINEMENT_OTHER`**: Unclassified geometric refinement rejection.

### Rationale
A rejection by a specific numerical IK solver or specific heuristic bounding box does not mathematically prove that no collision-free trajectory exists in the continuous configuration space. Using neutral rejection labels maintains scientific rigor and prevents overclaiming refiner incompleteness as intrinsic task geometry.

---

## 7. Canonical Requirement Counts (Single Source of Truth)

All benchmark domains evaluate a fixed canonical set of requirements defined in `mujoco_scenes/baselines/vilain_tamp/evaluation/base.py`:
- **Kitchen**: 8 canonical requirements (`coffee_pot_on_countertop`, `mug_1_on_countertop`, `mug_2_on_countertop`, `spoon_1_in_mug_1`, `spoon_2_in_mug_2`, `milk_carton_on_countertop`, `sugar_dispenser_on_countertop`, `table_service_area_clear`).
- **Living Room**: 6 canonical requirements (`book_on_shelf`, `remote_on_coffee_table`, `cushion_1_arranged`, `cushion_2_arranged`, `basket_on_side_table`, `floor_unobstructed`).
- **Workshop**: 6 canonical requirements (`bracket_fastened_to_workbench`, `faceplate_mounted_on_bracket`, `screws_tightened_in_bracket`, `dowels_seated_in_bracket`, `caliper_returned_to_tool_rack`, `allen_key_returned_to_tool_rack`).

No evaluator or harness component may hardcode alternative requirement totals.
