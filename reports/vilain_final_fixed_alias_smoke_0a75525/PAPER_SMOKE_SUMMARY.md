# ViLaIn-TAMP-Qwen Authoritative Qualification Smoke & Paper Audit

This directory contains the authoritative paper-ready evaluation artifacts, publication LaTeX tables, CSV audits, and JSON metrics for the **64-run qualification smoke matrix** (32 variants × 2 observation protocols × 1 repeat) of the ViLaIn-TAMP-Qwen baseline.

---

## 1. Provenance and Code Freeze
- **Behavior-Frozen Commit**: `0a75525470f878d1f4b6bd7ba535c9de4f860b1f` (`"Fix ViLaIn fixed-fixture geometry projection"`)
- **Report & Evaluation Artifacts Commit**: `155907f743ec586a11a83c794422d3653a32a6a2` (`"Add qualified 32x1 qualification smoke evaluation artifacts and audit"`)
- **Analysis Tooling**: `mujoco_scenes/run_vilain_tamp_paper_analysis.py` + offline audit replay
- **Runtime Modification**: **None**. Analysis and reporting artifacts only.
- **Model Server**: vLLM serving `qwen35-9b` (Qwen/Qwen3.5-9B)

---

## 2. Experimental Protocol Architecture

### A. Qualification Matrix Scope (Combined $N=64$)
The baseline qualification run executed 32 variants across two distinct observation protocols:
1. `initial_observation_only`: $32\text{ runs}$
2. `fixed_full_inspection`: $32\text{ runs}$

### B. Primary Manuscript Condition ($N=32$)
`initial_observation_only` is the **PRIMARY** manuscript condition. The agent receives only initial camera observations and cannot rely on hard-coded pre-manipulation drawer openings.
- Total Runs: $32$
- Ground-Truth Feasible: $20$
- Ground-Truth Infeasible: $12$
- Canonical Feasible Manipulation Subgoals: $126$ ($72\text{ Kitchen} + 30\text{ Living Room} + 24\text{ Workshop}$)

### C. Supplementary Condition ($N=32$)
`fixed_full_inspection` is supplementary only. It provides pre-opened storage inspection before planning.

> [!IMPORTANT]
> The two protocols are **NOT pooled** in primary manuscript results. Primary results reflect `initial_observation_only` ($N=32$) exclusively.

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

---

## 4. Supplementary Protocol Comparison

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

---

## 5. Dominant Baseline Failure Modes & Audits

### A. Identity Resolution Replay Audit
Replayed the production `BaselineIdentityResolver` offline over persisted contract fields across all 64 runs ($47\text{ attempt-level failures}$ total):
- **`LOCALIZATION_OUTSIDE_THRESHOLD`** ($20\text{ total}: 15\text{ Primary}, 5\text{ Supplementary}$): Bbox-depth localization produced a 3D centroid offset exceeding the $0.75\text{ m}$ threshold from ground-truth bodies.
- **`NO_STAGE_COMPATIBLE_ENTITY`** ($20\text{ total}: 0\text{ Primary}, 20\text{ Supplementary}$): In fixed inspection, detections in opened stages did not intersect with scene candidate initial visibility stages.
- **`ABSENT_OBJECT_IN_INFEASIBLE_VARIANT`** ($6\text{ total}: 6\text{ Primary}, 0\text{ Supplementary}$): Ground-truth infeasible variants where required tools/fasteners are physically absent from the scene.
- **`GEOMETRIC_AMBIGUITY`** ($1\text{ total}: 1\text{ Primary}, 0\text{ Supplementary}$): Two candidates within the $0.03\text{ m}$ ambiguity margin.
- **Implementation Defects**: **0**.

### B. Geometric Refinement Rejection Audit
Only two runs across the qualification smoke attempted refinement, both rejected by genuine physical/geometric constraints:
1. **Kitchen `F1_HIDDEN_COFFEE_VESSEL`**: Rejected at `IK` stage (`s1i_compact_coffee_jar`). Shared-backend re-evaluation with ProfiledIK confirmed minimum position residual of $0.2885\text{ m}$ exceeds the $0.03\text{ m}$ reachability tolerance from the robot's fixed base. Taxonomy: `GENERATED_TARGET_REJECTED_BY_SHARED_IK_BACKEND`.
2. **Workshop `F0_MANUAL_FIRST_ONE_REGION`**: Rejected at `COLLISION` stage (`workshop_medium_phillips_screw`). Continuous collision checking detected negative signed distance (-0.7 cm) between gripper finger tip and workbench top. Taxonomy: `GENERATED_TRAJECTORY_REJECTED_BY_MUJOCO_COLLISION_CHECK`.
- **Fixed-Fixture Alias Defects**: **0** (eliminated across all 64 runs).
- **Implementation Defects**: **0**.

### C. Known Limitation
No non-empty generated sequence reached physical execution in this qualification smoke because none passed complete refinement. Execution was fully enabled, instrumented, and verified by unit tests, but blocked upstream by physical collision/reachability constraints.

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
- **`paper_final_protocol_comparison_table.tex`**: Side-by-side protocol comparison table.
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
  --output-root /home/naren/ViLaIn-TAMP-results/vilain-final-160-matrix-155907f \
  --protocol initial_observation_only \
  --repeats 5
```
*Note: This command runs the 160-run primary benchmark (32 variants × 1 protocol × 5 repeats). Do NOT execute it during qualification smoke cleanup.*
