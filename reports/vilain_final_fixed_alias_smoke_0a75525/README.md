# ViLaIn-TAMP-Qwen Final Qualified Smoke Audit & Evaluation Artifacts

This directory contains the authoritative paper-ready evaluation artifacts, LaTeX tables, CSV audits, and JSON metrics for the **64-run qualification smoke matrix** (32 variants × 2 observation protocols × 1 repeat) of the ViLaIn-TAMP-Qwen baseline.

---

## 1. Provenance
- **Behavior-Frozen Commit**: `0a75525470f878d1f4b6bd7ba535c9de4f860b1f` (short SHA: `0a75525`)
- **Fix Description**: Resolved fixed structural symbolic/controller IDs to MuJoCo physical bodies (`resolve_geometry_entity_name`) for collision checking, IK, AABB, and articulation.
- **Model Server**: vLLM serving `qwen35-9b` (Qwen/Qwen3.5-9B) at `http://127.0.0.1:18000/v1`
- **Analysis Tooling**: `mujoco_scenes/run_vilain_tamp_paper_analysis.py`
- **Evaluation Protocols**: `initial_observation_only` (32 runs) and `fixed_full_inspection` (32 runs)
- **Seeds**: Authoritative seed generation (`240000` .. `240063`)

---

## 2. Key Qualification Results

- **Total Non-Infrastructure Runs**: 64 / 64 (0 infrastructure crashes)
- **Feasible Variant Runs**: 40 / 64
- **Outcome Correct Rate**: 3.1% (2/64)
- **Feasible-Task Success Rate**: 0.0% (0/40)
- **Goal Coverage (Micro)**: 1.6% (4/252)
- **False Completion Rate**: 100.0% (1/1)
- **Physical Plan Found**: 0.0% (0/40)
- **Infeasible Recall / Sensitivity**: 100.0% (18/18)
- **Feasible Recall / Specificity**: 3.0% (1/33)
- **Decision Accuracy**: 37.3% [25.3%, 51.0%]
- **Missing Scene Entity Defects**: **0** (eliminated across all 64 runs)

---

## 3. Directory Contents

### LaTeX Tables
- **`paper_final_main_table.tex`** / **`paper_main_table.tex`**: Publication-ready LaTeX table reporting Outcome Correct, Feasible-Task Success, Goal Coverage, False Completion, Physical Plan Found, VLM Requests, and Replans across domains.
- **`paper_final_goal_attribution_table.tex`** / **`paper_goal_attribution_table.tex`**: Initial vs terminal goal coverage and sequence-induced progress ($\Delta$).
- **`paper_final_feasibility_table.tex`** / **`paper_feasibility_table.tex`**: Feasibility confusion matrix (TP, FP, TN, FN) and classification metrics.
- **`paper_final_failure_table.tex`** / **`paper_failure_table.tex`**: Failure mode distribution across stages.

### Aggregate CSV & JSON Tables
- **`paper_final_overall.json`** / **`paper_overall.json`**: Global summary statistics and distributions.
- **`paper_final_by_domain.csv`** / **`paper_by_domain.csv`**: Metrics grouped by domain (Kitchen, Living Room, Workshop).
- **`paper_final_by_protocol.csv`** / **`paper_by_protocol.csv`**: Metrics grouped by observation protocol.
- **`paper_final_by_domain_protocol.csv`** / **`paper_by_domain_protocol.csv`**: Cross-tabulation of domain $\times$ protocol.
- **`paper_final_by_variant.csv`** / **`paper_by_variant.csv`**: Performance per variant across all 32 variants.
- **`paper_final_stage_funnel.csv`** / **`paper_stage_funnel.csv`**: Funnel conversion rates from Observation to Task Success.
- **`paper_final_feasibility_metrics.csv`** / **`paper_feasibility_metrics.csv`**: Feasibility detection metrics.
- **`paper_final_feasibility_confusion.json`** / **`paper_feasibility_confusion.json`**: Feasibility confusion matrix.
- **`paper_final_failure_breakdown.csv`** / **`paper_failure_breakdown.csv`**: Causal failure category counts and percentages.
- **`paper_final_metric_definitions.json`** / **`paper_metric_definitions.json`**: Exact metric formulas and denominators.

### Per-Run Datasets & Diagnostic Audits
- **`paper_final_run_metrics.csv`** / **`paper_final_run_metrics.jsonl`**: Complete 64-run tabular and JSONL metrics.
- **`paper_final_action_sequence_index.csv`** / **`paper_final_action_sequences.jsonl`**: Synthesized Fast Downward action sequences and validation statuses.
- **`paper_final_identity_audit.csv`**: Granular 34-record audit of every entity resolution attempt with 3D distances and taxonomy codes.
- **`paper_final_refinement_parity_audit.csv`**: Parity audit of refinement failures against independent MuJoCo physics and ProfiledIK.
- **`paper_final_identity_failures.csv`**: Summary of identity failure terminal runs.
- **`paper_final_refinement_failures.csv`**: Summary of refinement failure terminal runs.
- **`paper_final_benchmark_failure_audit.csv`**: Audit of runs reaching terminal execution.
- **`paper_final_generated_goal_audit.csv`**: Qwen generated goal satisfaction audit.
- **`paper_final_predicted_infeasible_cause.csv`**: Root-cause analysis of infeasibility classifications.
- **`paper_final_representative_sequences.json`**: Full representative action sequences for Kitchen, Living Room, and Workshop.
