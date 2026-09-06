# ViLaIn-TAMP Paper-Ready Evaluation Metrics Specification

This document defines the formal paper-ready evaluation metrics, numerators, denominators, and aggregation semantics for the ViLaIn-TAMP-Qwen baseline on the 320-run benchmark matrix.

---

## 1. Core Principles

1. **Non-Infrastructure Population**:
   All primary rates are computed over non-infrastructure runs ($N = 320$ scheduled runs; 0 infrastructure failures).
2. **Explicit Evaluation Denominators**:
   Metrics representing conditional evaluation (such as generated-goal satisfaction or execution success among attempted plans) MUST state their explicit denominators and coverage rates alongside unconditional figures. Never report "0%" without its exact denominator.
3. **Action-Sequence Decoupling**:
   Symbolic action sequence generation is evaluated independently of downstream geometric refinement or MuJoCo execution. A run where Fast Downward produced a valid plan that later failed geometric collision checking is credited with generating a valid action sequence.
4. **Feasibility Decision Separation**:
   The historical field `infeasibility_accuracy` incorrectly averaged True Positives over all runs ($TP / N$). Feasibility is evaluated via standard binary classification metrics on covered decisions, separating raw termination semantics from causally clean decisions.
5. **Separation of Baseline Planning and Benchmark Evaluation**:
   Benchmark truth ($G_F$, ground truth feasibility, hidden requirements) is strictly post-terminal and is never accessed during baseline perception, interpretation, planning, or refinement.

---

## 2. Primary Metrics Table

| Metric Name | Formal Definition | Numerator | Denominator | Better | Population |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Task Success Rate** | Proportion of runs achieving the hidden benchmark goal in physical simulation | Runs with `actual_task_success == True` | Non-infrastructure runs | Higher | All completed runs (and feasible subset) |
| **Action Sequence Generation Rate** | Proportion of runs where Fast Downward generated at least one parseable symbolic action sequence | Runs with $\ge 1$ parseable FD action sequence | Non-infrastructure runs | Higher | All completed runs |
| **VAL-Valid Plan Rate** | Proportion of runs where VAL validated at least one symbolic plan | Runs with $\ge 1$ VAL-valid action sequence | Non-infrastructure runs | Higher | All completed runs |
| **Execution-Ready Plan Rate** | Proportion of runs producing a plan that succeeded in both entity resolution and geometric refinement | Runs with successful identity binding and cloned refinement | Non-infrastructure runs | Higher | All completed runs |
| **Physical Execution Success Rate (Unconditional)** | Proportion of runs where physical simulation executed the complete plan without failure | Runs with `execution_success == True` | Non-infrastructure runs | Higher | All completed runs |
| **Physical Execution Success Rate (Conditional)** | Proportion of executed runs that completed successfully | Runs with `execution_success == True` | Runs where physical execution was attempted | Higher | Execution-attempted runs |
| **Generated Goal Satisfaction Rate** | Proportion of evaluated generated goals physically satisfied at the terminal state | Runs with `generated_goal_satisfied == True` | Runs where generated goal evaluation was performed | Higher | Runs with generated goal evaluation |
| **Generated Goal Evaluation Coverage** | Proportion of runs where generated goal evaluation was executed | Runs where generated goal evaluation was performed | Non-infrastructure runs | Higher | All completed runs |
| **Benchmark Requirement Coverage** | Average fraction of hidden benchmark requirements satisfied in the physical state | Sum of passed hidden requirement checks | Sum of total hidden requirement checks | Higher | Runs with hidden benchmark evaluation |
| **Generated Goal Atom Coverage** | Average fraction of generated PDDL goal atoms physically satisfied in the terminal state | Sum of passed generated goal atoms | Sum of total generated goal atoms | Higher | Runs with generated goal evaluation |
| **Feasibility Decision Accuracy (Raw Rule)** | Binary accuracy of the baseline's termination status in predicting ground-truth infeasibility | True Infeasible (TP) + True Feasible (TN) | Evaluated runs with feasibility predictions | Higher | Runs with feasibility evaluation |
| **Feasibility Decision Coverage** | Fraction of runs where a definitive feasibility prediction was evaluated | Runs with valid feasibility prediction | Non-infrastructure runs | Higher | All completed runs |
| **Average FM Calls** | Mean count of Foundation Model invocations per run | Total FM model calls | Completed runs | Context | All completed runs |
| **Average CP Calls** | Mean count of Corrective Planning iterations per run | Total CP corrections requested | Completed runs | Context | All completed runs |
| **Average Plan Length** | Mean number of actions in the generated symbolic action sequence | Total symbolic action steps in plans | Runs with symbolic plans found | Context | Runs with symbolic plans |

---

## 3. Action Sequence Funnel Metrics

The planning-to-execution pipeline is evaluated as a sequential funnel:

$$\text{PLAN@FD} \longrightarrow \text{PLAN@VAL} \longrightarrow \text{PLAN@REFINE} \longrightarrow \text{PLAN@EXEC} \longrightarrow \text{TASK@FINAL}$$

1. **PLAN@FD**:
   Fast Downward produced at least one parseable action sequence from the PDDL formulation generated by Qwen.
2. **PLAN@VAL**:
   The action sequence was verified by the PDDL validator VAL against the domain and problem specification.
3. **PLAN@REFINE**:
   At least one complete action sequence successfully bound physical entities and passed collision-free inverse kinematics (IK) in the cloned MuJoCo refinement scene.
4. **PLAN@EXEC**:
   The refined execution trajectory was dispatched to the MuJoCo simulation and completed without controller or dynamic postcondition failure.
5. **TASK@FINAL**:
   The final physical scene satisfied all ground-truth requirements of the benchmark variant.

Both **unconditional conversion** (% of all non-infrastructure runs) and **conditional stage conversion** (transition probability between consecutive stages) are reported.

---

## 4. Feasibility Classification Metrics

### 4.1 Raw Termination Rule (B2A)
Under the historical runtime rule, any run terminating with status `PREDICTED_INFEASIBLE` (or planning status in `{EXHAUSTED, REPEATED_REVISION, INVALID_CORRECTION}`) is considered a prediction of **Task Infeasible**. Runs that successfully refined and executed a plan are considered **Task Feasible**.

Treating **Task Infeasible** as the Positive class ($P$) and **Task Feasible** as the Negative class ($N$):
- **True Positive (TP)**: Ground-truth infeasible variant, predicted infeasible.
- **False Positive (FP)**: Ground-truth feasible variant, predicted infeasible (false infeasible).
- **True Negative (TN)**: Ground-truth feasible variant, predicted feasible.
- **False Negative (FN)**: Ground-truth infeasible variant, predicted feasible (false feasible).

Standard formulas:
- $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- $\text{Precision} = \frac{TP}{TP + FP}$
- $\text{Recall (Infeasible Recall)} = \frac{TP}{TP + FN}$
- $\text{Specificity (Feasible Recall)} = \frac{TN}{TN + FP}$
- $\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2}$
- $F_1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$
- $\text{False Infeasible Rate} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$
- $\text{False Feasible Rate} = \frac{FN}{FN + TP} = 1 - \text{Recall}$
- $\text{Decision Coverage} = \frac{TP + TN + FP + FN}{N_{\text{total}}}$

### 4.2 Causally-Clean Decision Analysis (B2B)
To avoid conflating model extraction failures, entity resolution failures, and geometric refiner failures with deliberate infeasibility declarations, runs are partitioned into post-hoc causal categories:
1. `PLAN_FOUND`: A valid symbolic plan was produced and refined.
2. `SYMBOLIC_NO_PLAN_AFTER_BOUNDED_CP`: Fast Downward proved unsolvability across all allowed CP attempts for valid PDDL.
3. `UNRESOLVED_IDENTITY_FAILURE`: Terminated due to unobserved entities or ambiguity without finding a resolvable model.
4. `UNRESOLVED_REFINEMENT_FAILURE`: Terminated due to geometric collision or unreachable IK.
5. `UNRESOLVED_INVALID_CORRECTION`: Terminated due to Qwen producing invalid PDDL / semantics during CP.
6. `UNRESOLVED_FM_FAILURE`: Terminated due to upstream FM transport or timeout.
7. `OTHER_UNRESOLVED`: Any other unclassified failure.

Feasibility metrics are evaluated on the covered subset of definitive decisions, with `decision_coverage` explicitly reported.

---

## 5. Statistical Uncertainty and Aggregations

1. **Proportions and Rates**:
   Reported as count / denominator, percentage, and 95% Wilson score confidence interval:
   $$w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
   where $z = 1.96$ for a 95% confidence level.
2. **Continuous Distributions**:
   Reported as $\text{mean} \pm \text{std}$ and $\text{median } [\text{IQR}]$.
3. **Macro vs. Micro Aggregation**:
   - **Micro**: Aggregate over all individual runs ($N = 320$).
   - **Macro**: Aggregate across variant means ($N = 32$ variants).
