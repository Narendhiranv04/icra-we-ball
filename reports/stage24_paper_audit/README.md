# ViLaIn-TAMP-Qwen Stage-24 Final Paper Audit & Evaluation Artifacts

This directory contains the complete, paper-ready audit and evaluation dataset for the **320-run Stage-24 evaluation matrix** of the ViLaIn-TAMP baseline with Qwen2.5-VL-72B-Instruct.

## Provenance
- **Evaluation Matrix Commit**: `4532495142b7a365d8bdba43f1776ab798818a28`
- **Analysis Tooling**: `mujoco_scenes/run_vilain_tamp_paper_analysis.py` (commit `2b0d313faa5a6a6438af5de7be60b80a7a49c508`)
- **Metric Definitions & Documentation**: [`mujoco_scenes/baselines/vilain_tamp/PAPER_METRICS.md`](../../mujoco_scenes/baselines/vilain_tamp/PAPER_METRICS.md)
- **Matrix Scope**: 320 total runs across 4 domains (Kitchen, Living Room, Lab, Workshop), 16 variants, 2 protocols (`all_candidate_regions`, `target_region_only`), 20 random seeds per variant.

---

## Contents

### LaTeX Tables
- **`paper_main_table.tex`**: Publication-ready LaTeX table broken down by domain and protocol, reporting:
  - Plan Found Rate ($N=320$)
  - VAL Acceptance Rate ($N=141$)
  - Identity Resolution Rate ($N=141$)
  - Refinement Success Rate ($N=68$)
  - Terminal Benchmark Success Rate ($N=320$)
  - Binary Feasibility Accuracy ($N=298$, excluding 22 timeout/syntax parser errors) with 95% Wilson score confidence intervals
- **`paper_failure_table.tex`**: Comprehensive taxonomy of failure modes across the pipeline:
  - Planning: Unsolvable state / FD failure
  - Grounding: Missing objects / unmatched geometric entities
  - Refinement: Collision / unreachable IK
  - Benchmark: Terminal goal predicate unsatisfied
  - System: Timeout / syntax error
- **`paper_feasibility_table.tex`**: Detailed breakdown of feasibility detection (TP, FP, TN, FN, Sensitivity, Specificity, Accuracy) by domain and protocol.

### Aggregate CSV & JSON Tables
- **`paper_overall.json`**: Global scalar summary of all pipeline stages, continuous metrics (action sequence lengths, solve times, refinement times, memory, token usage), Wilson score confidence intervals, and feasibility classification metrics.
- **`paper_stage_funnel.csv`**: Pipeline stage progression from Raw Runs $\to$ FD Plans $\to$ VAL Validated $\to$ Identity Grounded $\to$ Refinement Succeeded $\to$ Terminal Executed.
- **`paper_by_domain.csv`**: Metrics grouped by domain (Kitchen, Living Room, Lab, Workshop).
- **`paper_by_protocol.csv`**: Metrics grouped by candidate region protocol (`all_candidate_regions` vs `target_region_only`).
- **`paper_by_domain_protocol.csv`**: Cross-tabulation of domain $\times$ protocol.
- **`paper_by_variant.csv`**: Per-variant breakdown across all 16 evaluation variants (K1..K4, L1..L4, B1..B4, W1..W4).
- **`paper_feasibility_metrics.csv`**: Complete feasibility metrics table.
- **`paper_feasibility_confusion.json`**: Confusion matrix for feasibility classification.
- **`paper_failure_breakdown.csv`**: Failure category distribution across all 320 runs.
- **`paper_metric_definitions.json`**: Machine-readable specification of numerators, denominators, and calculation formulas.

### Per-Run Datasets & Diagnostic Audits
- **`paper_run_metrics.csv` / `paper_run_metrics.jsonl`**: Complete 320-run tabular and structured log with per-run statuses, durations, action counts, memory, and failure categories.
- **`paper_action_sequence_index.csv` / `paper_action_sequences.jsonl`**: Exact action sequences produced by Fast Downward, VAL validation status, grounded action sequences, and execution outcome for all 141 planned runs.
- **`paper_benchmark_failure_audit.csv`**: Detailed audit of the 18 runs reaching terminal execution (explaining the 0% benchmark success due to hallucinated `:init` states yielding 0-step plans).
- **`paper_identity_failures.csv`**: Inventory of all 73 identity resolution failures where Fast Downward planned with symbols absent from MuJoCo geometric object registries.
- **`paper_refinement_failures.csv`**: Inventory of all 50 geometric refinement failures (colliding trajectories and out-of-reach IK targets).
- **`paper_generated_goal_audit.csv`**: Per-run evaluation of Qwen's generated goal condition against terminal scene state.
- **`predicted_infeasible_cause.csv`**: Root-cause analysis of all runs classified as infeasible.
