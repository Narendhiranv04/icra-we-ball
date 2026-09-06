# ViLaIn-TAMP Paper-Ready Evaluation Metrics Specification

This document formally specifies the evaluation metrics, action-sequence funnel, statistical conventions, causal classifications, and ground-truth isolation guarantees for the ViLaIn-TAMP-Qwen baseline across benchmark domains (`kitchen`, `living_room`, `workshop`).

---

## 1. Core Experimental Principles

1. **The Primary Output is the Action Sequence**:
   The primary output of ViLaIn-TAMP for any task is the **action sequence** it generates. Evaluation is sequence-centric: we measure whether ViLaIn-TAMP generates valid symbolic plans, binds physical entities, refines continuous kinematic trajectories, and executes actions that induce goal state progress. ViLaIn-TAMP is never evaluated against human/teleoperated demonstration sequences or proposed-method action sequences; it is evaluated against the physical benchmark world and its canonical goal requirements.

2. **Goal Coverage Definition (Single Source of Truth)**:
   > [!IMPORTANT]
   > **GOAL COVERAGE DOES NOT MEAN**: "fraction of generic benchmark checks that happen to be true."
   > **GOAL COVERAGE MEANS**: "fraction of canonical GT terminal manipulation subgoals satisfied in the physical state."

   Passive structural prerequisites (e.g., containers existing, default alignments, distinct roles, "no object held") can be trivially true before the robot acts and must never inflate Goal Coverage.

3. **Strict Post-Terminal Isolation**:
   Ground-truth requirements, feasibility labels ($G_F$), and hidden physical assertions are strictly post-terminal. They are NEVER accessed, leaked, or referenced during baseline perception, VLM prompt construction, PDDL compilation, Fast Downward search, entity resolution, or geometric refinement.

4. **Decoupling Full Task Success and Goal Coverage**:
   - **Full Task Success**: All canonical benchmark constraints and safety/validity checks satisfied simultaneously.
   - **Goal Coverage**: Fraction of actual canonical terminal manipulation subgoals achieved in the physical simulation state.

5. **Explicit Denominators and Population Scoping**:
   Every rate metric must state its exact numerator and denominator:
   - **All Non-Infrastructure Runs ($N_{\text{total}}$)**: Default population for unconditional metrics (Outcome Correct, Plan Generation, Resource Counts).
   - **Ground-Truth Feasible Runs ($N_{\text{feasible}}$)**: Mandatory population for goal-attainment metrics (Feasible-Task Success, Goal Coverage, Sequence-Induced Goal Gain, Physical Plan Found). Infeasible variants have no valid solution trajectory; including them in goal coverage artificially deflates or conflates baseline performance.
   - **Declared Completion Runs ($N_{\text{declared}}$)**: Denominator for False Completion Rate.

6. **Zero-Step Plan Prohibition**:
   Zero-step plans (`len == 0`, empty action sequence) occur when Qwen's generated `:init` state logically entails its generated `:goal`. While Fast Downward trivially exits with success in 0 steps, zero physical actions are executed.
   - Zero-step plans **NEVER** count toward `Physical Plan Found` ($len > 0$ required).
   - Zero-step plans **NEVER** count toward `NONEMPTY PLAN@EXEC`.
   - If the baseline declares completion on a zero-step plan while physical benchmark requirements remain unmet, it is counted as a **False Completion**.

---

## 2. Canonical GT Terminal Manipulation Subgoals

Defined centrally in `mujoco_scenes/baselines/vilain_tamp/evaluation/subgoals.py`. Every subgoal follows the canonical atomic triple representation `(subject, predicate, target)`.

### 2.1 Living Room Domain (5 Subgoals per Feasible Variant)
- `(cup_left, ON, personal_table_left)`: Left cup placed on left personal table.
- `(saucer_left, ON, personal_table_left)`: Left saucer placed on left personal table.
- `(cup_right, ON, personal_table_right)`: Right cup placed on right personal table.
- `(saucer_right, ON, personal_table_right)`: Right saucer placed on right personal table.
- `(remote, ON, coffee_table_shared)`: Remote control placed on shared coffee table.

*Excluded Passive/Structural Checks*: `required_payloads_present`, `required_supports_present`, `no_payload_held`.

### 2.2 Kitchen Domain (12 Subgoals per Feasible Variant)
Composed of atomic manipulation outcomes for 2 coffee vessels and 2 soup vessels:
- **Coffee Vessel 1 (4 subgoals)**:
  - `(coffee_vessel_1, ON, serving_support)`: Physically served on serving support.
  - `(coffee_vessel_1, HAS_CONTENT, water)`: Water delivered to vessel.
  - `(coffee_vessel_1, HAS_CONTENT, coffee)`: Coffee powder delivered to vessel.
  - `(coffee_vessel_1, STIRRED, true)`: Contents stirred with suitable stirrer.
- **Coffee Vessel 2 (4 subgoals)**:
  - `(coffee_vessel_2, ON, serving_support)`: Physically served on serving support.
  - `(coffee_vessel_2, HAS_CONTENT, water)`: Water delivered to vessel.
  - `(coffee_vessel_2, HAS_CONTENT, coffee)`: Coffee powder delivered to vessel.
  - `(coffee_vessel_2, STIRRED, true)`: Contents stirred with suitable stirrer.
- **Soup Vessel 1 (2 subgoals)**:
  - `(soup_vessel_1, ON, serving_support)`: Physically served on serving support.
  - `(soup_vessel_1, CONTAINS, soup_utensil_1)`: Stably contains suitable soup utensil.
- **Soup Vessel 2 (2 subgoals)**:
  - `(soup_vessel_2, ON, serving_support)`: Physically served on serving support.
  - `(soup_vessel_2, CONTAINS, soup_utensil_2)`: Stably contains suitable soup utensil.

*Excluded Passive/Structural Checks*: `two_coffee_vessels_exist`, `two_soup_vessels_exist`, `vessel_groups_disjoint`, `no_object_held`.

### 2.3 Workshop Domain (3 Subgoals per Feasible Variant)
- `(compatible_screw, INSERTED_IN, repair_joint)`: Compatible screw inserted into target joint with valid geometry.
- `(repair_joint, FASTENED, true)`: Joint physically repaired / screw fully driven.
- `(selected_driver, ON, main_workbench)`: Selected driver returned safely to main workbench.

*Excluded Passive/Structural Checks*: `compatible_driver_check`, `compatible_fastener_check`, `procedural_first_driver_order`, `no_object_held`.

### Domain Canonical Subgoal Totals Across Feasible Matrix (20 Feasible Variants)
- **Kitchen**: $6 \text{ variants} \times 12 = 72 \text{ subgoals}$
- **Living Room**: $6 \text{ variants} \times 5 = 30 \text{ subgoals}$
- **Workshop**: $8 \text{ variants} \times 3 = 24 \text{ subgoals}$
- **Total Feasible Matrix**: $72 + 30 + 24 = 126 \text{ canonical manipulation subgoals}$.

---

## 3. Manuscript Main-Table Metrics

| Metric Name | Formal Definition | Numerator | Denominator | Better | Target Population |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Outcome Correct (%)** | Proportion of runs where binary decision matches ground truth feasibility and execution outcome | $\sum [\text{feasible} \land \text{success}] + \sum [\text{infeasible} \land \text{rejected}]$ | All non-infrastructure runs ($N_{\text{total}}$) | Higher | All Runs ($N=32$ 1-repeat, $N=160$ 5-repeat official `initial_observation_only`) |
| **Feasible-Task Success (%)** | Proportion of ground-truth feasible runs achieving all canonical benchmark constraints | Runs with `actual_task_success == True` | Ground-truth feasible runs ($N_{\text{feasible}}$) | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **Goal Coverage (%)** | Fraction of canonical GT terminal manipulation subgoals satisfied in the physical state | $\sum \text{subgoals passed}$ | $\sum \text{subgoals total}$ | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **Initial Goal Coverage (%)** | Fraction of canonical GT terminal manipulation subgoals satisfied at initial physical state | $\sum \text{initial subgoals passed}$ | $\sum \text{initial subgoals total}$ | Context | Feasible Subset ($N_{\text{feasible}}=20$) |
| **$\Delta$ Goal Coverage (%)** | Net goal progress induced by executing the generated action sequence: $\text{Coverage}_{\text{terminal}} - \text{Coverage}_{\text{initial}}$ | $\text{Coverage}_{\text{terminal}} - \text{Coverage}_{\text{initial}}$ | Percentage points | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **False Completion (%)** | Proportion of declared task completions where physical requirements were not satisfied | Runs declaring completion with `actual_task_success == False` | Runs declaring completion ($N_{\text{declared}}$) | Lower | Declared-Complete Runs |
| **Physical Plan Found (%)** | Proportion of feasible runs producing a valid, non-empty, geometrically refined physical plan | Feasible runs with non-empty plan passing refinement | Ground-truth feasible runs ($N_{\text{feasible}}$) | Higher | Feasible Subset ($N_{\text{feasible}}=20$) |
| **Raw VLM Requests** | Total foundation model (VLM) API requests per run | Sum of object, init, goal, and CP model calls | Completed runs ($N_{\text{total}}$) | Context | All Runs |
| **High-Level Replans** | Number of corrective planning (CP) iterations invoked following planner/refiner failures | Total CP iterations requested | Completed runs ($N_{\text{total}}$) | Context | All Runs |

### Aggregation Conventions
- **Micro Goal Coverage**: Pooled across all subgoals in the population: $\frac{\sum_{i} \text{subgoals\_passed}_i}{\sum_{i} \text{subgoals\_total}_i}$.
- **Macro Goal Coverage**: Unweighted mean of variant coverage scores: $\frac{1}{|V|} \sum_{v \in V} \frac{\text{subgoals\_passed}_v}{\text{subgoals\_total}_v}$.
- **Uncertainty Intervals**: 95% Wilson score confidence intervals for all binomial proportions; mean $\pm$ standard deviation for continuous distributions.

---

## 4. Strict Same-Attempt Action Sequence Funnel

To maintain rigorous causal tracking, general diagnostic counts are decoupled from the strictly nested physical sequence funnel:

### 4.1 General Diagnostic Plan Counts
- **`ANY PLAN@FD`**: Fast Downward produced at least one parseable action sequence (including zero-step plans).
- **`ANY PLAN@VAL`**: VAL validated at least one action sequence (including zero-step plans).

### 4.2 Strict Nested Non-Empty Action Sequence Funnel
Every downstream stage requires that **one single attempt** satisfies all prior stages:

$$\text{NONEMPTY PLAN@FD} \longrightarrow \text{NONEMPTY PLAN@VAL} \longrightarrow \text{NONEMPTY PLAN@IDENTITY} \longrightarrow \text{NONEMPTY PLAN@REFINE} \longrightarrow \text{NONEMPTY PLAN@EXEC} \longrightarrow \text{TASK@FINAL}$$

| Funnel Stage | Formal Definition | Same-Attempt Criterion |
| :--- | :--- | :--- |
| **1. NONEMPTY PLAN@FD** | Fast Downward produced a non-empty plan ($len > 0$) | $\exists a: a.\text{nonempty}$ |
| **2. NONEMPTY PLAN@VAL** | Non-empty plan validated by VAL domain semantics | $\exists a: a.\text{nonempty} \land a.\text{val\_valid}$ |
| **3. NONEMPTY PLAN@IDENTITY** | Non-empty VAL-valid plan resolved to physical entities | $\exists a: a.\text{nonempty} \land a.\text{val\_valid} \land a.\text{identity\_success}$ |
| **4. NONEMPTY PLAN@REFINE** | Non-empty VAL-valid resolved plan passed kinematic refinement | $\exists a: a.\text{nonempty} \land a.\text{val\_valid} \land a.\text{identity\_success} \land a.\text{refinement\_success}$ |
| **5. NONEMPTY PLAN@EXEC** | Non-empty refined plan dispatched and completed execution | Selected executed attempt satisfies non-empty refinement chain |
| **6. TASK@FINAL** | Physical scene satisfies all canonical benchmark constraints | Non-empty executed run achieves `actual_task_success == True` |

Because every stage is strictly nested on the same attempt, **all conditional conversion rates are mathematically bounded $\le 100\%$**.

### 4.3 Selected Sequence Provenance
Every run artifact directory records:
- `selected_attempt_index`
- `selected_plan_sha256`
- `selected_plan_length`
- `selected_action_sequence`

The paper audit asserts that `final_action_plan.json` strictly matches the attempt outcome and refinement records for the selected execution attempt.

---

## 5. Neutral Diagnostic Causal Labels

### 5.1 Corrective Planning (CP) Diagnostic Labels
Rather than generic "invalid PDDL" strings, CP termination is classified using observable symbolic diagnostics:
- **`INVALID_CORRECTIVE_SELECTION`**: Selected fact set violates domain compilation invariants.
- **`MALFORMED_CORRECTIVE_RESPONSE`**: VLM response could not be parsed into valid fact selections.
- **`UNKNOWN_FACT_ID`**: VLM referenced fact IDs not present in the candidate fact universe.
- **`INCONSISTENT_CORRECTIVE_FACT_SET`**: Contradictory facts selected simultaneously.
- **`UNREACHABLE_CORRECTED_GOAL`**: Fast Downward proved the corrected problem unsolvable.
- **`REPEATED_CORRECTION`**: VLM proposed an identical fact revision to an earlier failed attempt.

### 5.2 Identity Resolution Failures
- **`MULTIPLE_PHYSICAL_CANDIDATES`**: Perception returned $> 1$ physical bodies matching semantic type within spatial region.
- **`NO_PHYSICAL_CANDIDATE`**: Perception returned 0 candidate bodies matching type/envelope.
- **`SINGLE_PHYSICAL_CANDIDATE_BUT_UNRESOLVED`**: Exactly one candidate body existed, but attribute/affordance check failed.

### 5.3 Refinement Failures
- **`REFINEMENT_IK_REJECTION`**: Numerical inverse kinematics solver failed to find a collision-free joint configuration.
- **`REFINEMENT_COLLISION_REJECTION`**: Collision checker detected body contact along continuous trajectory.
- **`REFINEMENT_SKILL_ENVELOPE_REJECTION`**: Target pose fell outside reachability envelope of the parameterized skill.
- **`REFINEMENT_OTHER`**: Unclassified geometric refinement rejection.
