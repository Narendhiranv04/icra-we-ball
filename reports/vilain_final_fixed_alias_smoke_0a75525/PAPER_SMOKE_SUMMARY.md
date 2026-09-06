# ViLaIn-TAMP-Qwen Authoritative Qualification Smoke & Paper Audit

This directory contains the authoritative paper-ready evaluation artifacts, publication LaTeX tables, CSV audits, and JSON metrics for the **64-run qualification smoke matrix** (32 variants × 2 observation protocols × 1 repeat) of the ViLaIn-TAMP-Qwen baseline.

---

## 1. Provenance and Code Freeze
- **Behavior-Frozen Runtime Commit**: `0a75525470f878d1f4b6bd7ba535c9de4f860b1f` (`"Fix ViLaIn fixed-fixture geometry projection"`)
- **Previous Reporting Commit**: `8730c32d3778589775a88aa7a220ee5b54923a0c` (`"Finalize ViLaIn primary smoke reporting"`)
- **Current Cleanup Scope**: Reporting and documentation caveats cleanup only. **Zero runtime code, prompt, or PDDL modifications.**
- **Analysis Tooling**: `mujoco_scenes/run_vilain_tamp_paper_analysis.py` + offline audit replay
- **Model Server**: vLLM serving `qwen35-9b` (Qwen/Qwen3.5-9B)

---

## 2. Experimental Protocol Architecture

### A. Qualification Matrix Scope (Combined $N=64$)
The baseline qualification run executed 32 variants across two distinct observation protocols:
1. `initial_observation_only`: $32\text{ runs}$ (Primary manuscript baseline)
2. `fixed_full_inspection`: $32\text{ runs}$ (Supplementary development/qualification trace only)

### B. Primary Manuscript Condition ($N=32$)
`initial_observation_only` is the **PRIMARY** manuscript condition. The agent receives only initial camera observations and cannot rely on hard-coded pre-manipulation drawer openings.
- Total Runs: $32$
- Ground-Truth Feasible: $20$
- Ground-Truth Infeasible: $12$
- Canonical Feasible Manipulation Subgoals: $126$ ($72\text{ Kitchen} + 30\text{ Living Room} + 24\text{ Workshop}$)
- **Status**: The primary `initial_observation_only` condition is completely unaffected by any supplementary stage-ID bookkeeping limitation and remains the authoritative benchmark condition.

### C. Supplementary Condition ($N=32$, Diagnostic Qualification Only)
The supplementary `fixed_full_inspection` protocol is retained only as a development/qualification trace. It contains a known stage-ID association limitation in the adapter: candidate entities are registered under `"000_initial"`, while detections in opened storage fixtures carry opened stage IDs like `"002_right_drawer"` or `"003_tool_cabinet"`. The production identity resolver requires a non-empty intersection between candidate visible stages and observation stages, causing it to reject valid physical matches.

> [!IMPORTANT]
> The two protocols are **NOT pooled** in primary manuscript results. Primary results reflect `initial_observation_only` ($N=32$) exclusively. `fixed_full_inspection` is retained for provenance and qualification tracing, but is not used to make scientific conclusions about ViLaIn.

---

## 3. Authoritative Primary 32×1 Manuscript Results

| Metric | Overall ($N=32$) | Kitchen ($N=12$) | Living Room ($N=10$) | Workshop ($N=10$) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Non-Infrastructure Runs** | 32 | 12 | 10 | 10 |
| **GT-Feasible Runs** | 20 | 6 | 6 | 8 |
| **Outcome Correct** | 3.1% (1/32) | 8.3% (1/12) | 0.0% (0/10) | 0.0% (0/10) |
| **Feasible-Task Success** | 0.0% (0/20) | 0.0% (0/6) | 0.0% (0/6) | 0.0% (0/8) |
| **Initial Goal Coverage** | 1.6% (2/126) | 0.0% (0/72) | 6.7% (2/30) | 0.0% (0/24) |
| **Terminal Goal Coverage** | 1.6% (2/126) | 0.0% (0/72) | 6.7% (2/30) | 0.0% (0/24) |
| **$\Delta$ Goal Coverage** | 0.0% (0/126) | 0.0% (0/72) | 0.0% (0/30) | 0.0% (0/24) |
| **False Completion** | 100.0% (1/1) | N/A | 100.0% (1/1) | N/A |
| **Physical Plan Found** | 0.0% (0/20) | 0.0% (0/6) | 0.0% (0/6) | 0.0% (0/8) |
| **ANY PLAN @ FD** | 50.0% (16/32) | 66.7% (8/12) | 30.0% (3/10) | 50.0% (5/10) |
| **NONEMPTY PLAN @ FD** | 46.9% (15/32) | 66.7% (8/12) | 20.0% (2/10) | 50.0% (5/10) |
| **NONEMPTY PLAN @ VAL** | 46.9% (15/32) | 66.7% (8/12) | 20.0% (2/10) | 50.0% (5/10) |
| **NONEMPTY PLAN @ IDENTITY** | 12.5% (4/32) | 25.0% (3/12) | 0.0% (0/10) | 10.0% (1/10) |
| **NONEMPTY PLAN @ REFINE** | 0.0% (0/32) | 0.0% (0/12) | 0.0% (0/10) | 0.0% (0/10) |
| **NONEMPTY PLAN @ EXEC** | 0.0% (0/32) | 0.0% (0/12) | 0.0% (0/10) | 0.0% (0/10) |
| **Raw VLM Requests** | 6.91 $\pm$ 0.91 | 7.17 $\pm$ 0.69 | 6.70 $\pm$ 1.10 | 6.80 $\pm$ 0.87 |
| **High-Level Replans** | 2.91 $\pm$ 0.52 | 3.00 $\pm$ 0.00 | 2.70 $\pm$ 0.90 | 3.00 $\pm$ 0.00 |

### Primary Stage Funnel Progression
```
[Observation]                 32 / 32 (100.0%, conversion: 100.0%)
[Object Estimation]           32 / 32 (100.0%, conversion: 100.0%)
[Valid PDDL]                  32 / 32 (100.0%, conversion: 100.0%)
[ANY PLAN @ FD]               16 / 32 ( 50.0%, conversion:  50.0%)
[ANY PLAN @ VAL]              16 / 32 ( 50.0%, conversion: 100.0%)
[NONEMPTY PLAN @ FD]          15 / 32 ( 46.9%, conversion:  93.8%)
[NONEMPTY PLAN @ VAL]         15 / 32 ( 46.9%, conversion: 100.0%)
[NONEMPTY PLAN @ IDENTITY]     4 / 32 ( 12.5%, conversion:  26.7%)
[NONEMPTY PLAN @ REFINE]       0 / 32 (  0.0%, conversion:   0.0%)
[NONEMPTY PLAN @ EXEC]         0 / 32 (  0.0%, conversion:   0.0%)
[TASK @ FINAL]                 0 / 32 (  0.0%, conversion:   0.0%)
```

### False Completion Characterization
Living Room `F1_LEFT_SAUCER_PREPLACED` produced a zero-action plan even though only the preplaced left-saucer subgoal was satisfied (Initial Goal Coverage 1/5, Terminal Goal Coverage 1/5, $\Delta$ Goal Coverage 0/5), causing a false completion under hidden task evaluation.

---

## 4. Supplementary Protocol Comparison (Diagnostic Qualification Only)

> [!WARNING]
> Diagnostic qualification only — `fixed_full_inspection` contains a known stage-association adapter limitation and is not used for manuscript baseline performance.

| Metric | Initial Observation Only (Primary) | Fixed Full Inspection (Supplementary) |
| :--- | :---: | :---: |
| **Total Non-Infrastructure Runs** | 32 | 32 |
| **Feasible Variant Runs** | 20 | 20 |
| **Outcome Correct** | 3.1% (1/32) | 3.1% (1/32) |
| **Feasible-Task Success** | 0.0% (0/20) | 0.0% (0/20) |
| **Initial Goal Coverage** | 1.6% (2/126) | 1.6% (2/126) |
| **Terminal Goal Coverage** | 1.6% (2/126) | 1.6% (2/126) |
| **$\Delta$ Goal Coverage** | 0.0% (0/126) | 0.0% (0/126) |
| **False Completion** | 100.0% (1/1) | N/A (0/0) |
| **Physical Plan Found** | 0.0% (0/20) | 0.0% (0/20) |
| **ANY PLAN @ FD** | 50.0% (16/32) | 31.2% (10/32) |
| **NONEMPTY PLAN @ FD** | 46.9% (15/32) | 28.1% (9/32) |
| **NONEMPTY PLAN @ VAL** | 46.9% (15/32) | 28.1% (9/32) |
| **NONEMPTY PLAN @ IDENTITY** | 12.5% (4/32) | 0.0% (0/32) |
| **NONEMPTY PLAN @ REFINE** | 0.0% (0/32) | 0.0% (0/32) |
| **Raw VLM Requests** | 6.91 $\pm$ 0.91 | 4.22 $\pm$ 3.23 |
| **High-Level Replans** | 2.91 $\pm$ 0.52 | 1.78 $\pm$ 1.48 |

### Protocol Caveats & Diagnostics
- **FM Termination**: 12 `fixed_full_inspection` runs terminated during FM processing/formalization.
- **Stage-ID Adapter Limitation**: The remaining runs failed at identity resolution because candidate physical entities were only registered under `"000_initial"`, while detections in inspected storage carried stage tags such as `"002_right_drawer"` or `"003_tool_cabinet"`.
- **Manuscript Isolation**: Because of this adapter-level bookkeeping limitation, `fixed_full_inspection` is not treated as a clean baseline comparison and is excluded from primary manuscript claims.

---

## 5. Dominant Baseline Failure Modes & Audits

### A. Identity Resolution Replay Audit
Identity resolution was a major downstream bottleneck.

**Run / Funnel Level**:
15 non-empty VAL-valid primary plans were generated by Fast Downward; 4 reached `NONEMPTY PLAN @ IDENTITY` and 11 did not.

**Attempt-Level Primary Identity Audit**:
Replaying `BaselineIdentityResolver` offline over persisted contract fields across all 64 qualification runs produced 47 attempt failures total (22 in the primary protocol, 25 in the supplementary protocol).

In the primary protocol ($N=22$ attempt failures):
- Failed identity attempts in the primary protocol were dominated by visual localization outside the 0.75 m association threshold (`LOCALIZATION_OUTSIDE_THRESHOLD`, $N=15$), with additional failures due to absent entities in infeasible variants (`ABSENT_OBJECT_IN_INFEASIBLE_VARIANT`, $N=6$) and one geometric ambiguity (`GEOMETRIC_AMBIGUITY`, $N=1$).
- **Primary Unresolved Adapter Defects**: **0**.

In the supplementary protocol ($N=25$ attempt failures):
- 20 attempt failures were caused by the known stage-ID association limitation (`NO_STAGE_COMPATIBLE_ENTITY`: candidate entities registered with initial stage only, while detections in opened fixtures carried opened stage IDs).
- 5 attempt failures were due to `LOCALIZATION_OUTSIDE_THRESHOLD`.
- **Supplementary Status**: Retained as diagnostic qualification trace only; contains known stage-ID association limitation.

### B. Geometric Refinement Rejection Audit
Only two runs across the qualification smoke attempted refinement, both rejected by genuine physical/geometric constraints:
1. **Kitchen `F1_HIDDEN_COFFEE_VESSEL`**: Generated target rejected by the shared IK backend (`s1i_compact_coffee_jar`). Recorded position error was $0.1319\text{ m}$ (tolerance $0.03\text{ m}$); neutral ProfiledIK re-evaluation confirmed a $0.2885\text{ m}$ position error. Taxonomy: `GENERATED_TARGET_REJECTED_BY_SHARED_IK_BACKEND`.
2. **Workshop `F0_MANUAL_FIRST_ONE_REGION`**: Generated trajectory rejected by MuJoCo collision checking (`workshop_medium_phillips_screw`). Continuous collision checking detected negative signed distance (-0.7 cm) between gripper fingertip and workbench table surface. Taxonomy: `GENERATED_TRAJECTORY_REJECTED_BY_MUJOCO_COLLISION_CHECK`.

These are verified rejections by kinematic reachability and continuous collision checks, confirming that fixed-fixture name-resolution bugs are resolved. Neither run represents an unresolved adapter defect, nor do they imply universal task impossibility.

### C. Feasibility Decision Diagnostic (Supplementary Only)
The raw termination-rule diagnostic has 100% infeasible recall but a 95% false-infeasible rate on feasible tasks (Accuracy 40.6% [95% CI: 25.5%, 57.7%], Balanced Accuracy 52.5%). Main manuscript outcome metric remains Outcome Correct = 1/32 (3.1%).

### D. Known Execution Limitation
Across this qualification smoke, zero non-empty generated sequences reached physical robot execution. The physical execution module was fully instrumented, active, and verified by unit tests; however, all candidate plans were rejected upstream by genuine kinematic reachability and continuous collision constraints.

---

## 6. Directory File Index

### Primary Manuscript Files
- **`paper_final_main_table.tex`**: Primary manuscript LaTeX table ($N=32$).
- **`paper_final_goal_attribution_table.tex`**: Initial vs terminal goal coverage LaTeX table ($N_{\text{feasible}}=20$).
- **`paper_final_primary_stage_funnel.csv`** & **`paper_final_primary_stage_funnel.tex`**: Primary stage funnel.
- **`paper_final_primary_failure_breakdown.csv`**: Primary causal failure category distribution.
- **`paper_final_primary_by_domain.csv`**: Domain metrics breakdown table for primary protocol.
- **`paper_final_primary_metrics.json`**: Machine-readable primary metrics.
- **`paper_final_primary_run_metrics.csv`**: Exact 32-run metric records for primary protocol.

### Supplementary & Protocol Comparison Files
- **`paper_final_protocol_comparison_table.tex`**: Side-by-side protocol comparison table (diagnostic qualification).
- **`paper_final_protocol_failure_comparison.csv`**: Failure distribution comparison by protocol.
- **`paper_final_feasibility_table.tex`**: Supplementary feasibility analysis confusion matrix table.
- **`paper_qualification_combined_64_table.tex`**: Preserved pooled 64-run table.
- **`paper_qualification_combined_stage_funnel.csv`**: Preserved pooled 64-run funnel.

### Diagnostic Audits & Logs
- **`paper_final_identity_audit.csv`**: 47-row attempt-level offline replay identity audit.
- **`paper_final_refinement_parity_audit.csv`**: Geometric refinement audit with shared-backend verification.
- **`paper_final_action_sequence_index.csv`**: Index of Fast Downward action sequences.
- **`paper_final_action_sequences.jsonl`**: JSONL stream of all synthesized action plans.
- **`paper_final_audit.json`**: Complete audit provenance record.
- **`PAPER_SMOKE_SUMMARY.md`**: Formal publication smoke summary.

---

## 7. Command for 160-Run Primary Manuscript Benchmark

```bash
cd /home/naren/ViLaIn-TAMP

PYTHONPATH=. \
.venv-vilain-tamp/bin/python \
-m mujoco_scenes.baselines.vilain_tamp.benchmark_harness \
  --output-root /home/naren/ViLaIn-TAMP-results/vilain-final-160-matrix-8730c32 \
  --protocol initial_observation_only \
  --repeats 5
```
*Note: This command runs the 160-run primary benchmark (32 variants × 1 protocol × 5 repeats). Do NOT execute it during qualification smoke cleanup.*
