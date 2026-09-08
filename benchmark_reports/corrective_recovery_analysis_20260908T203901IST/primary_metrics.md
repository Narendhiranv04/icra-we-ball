# Primary Metric Benchmark Comparison

Comparison across evaluation regimes:
1. **Historical Strict Baseline (`af2dde`):** Collapsed to 0.0% goal coverage due to conflation of offline completeness with online executability and unmapped capability preconditions.
2. **Corrected Method (Development 32x1):** Live V2 pipeline on canonical 32 development cases with frozen config (`qwen35-9b`, `enable_thinking=false`).
3. **Corrected Method (Held-Out 15x1):** Live V2 pipeline on 15 generalization cases under identical frozen config.

| Benchmark Evaluation | Variants (Feas/Infeas) | Outcome Correct ↑ | Feasible Success ↑ | Feasibility Recovery ↑ | Goal Coverage ↑ | False Completion ↓ | VLM Requests ↓ | Replans ↓ |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Strict Baseline (`af2dde`)** | 32 (20 / 12) | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **1.00** | **0.00** |
| **Corrected Method (Dev 32x1)** | 32 (20 / 12) | **0.0%** | **0.0%** | **0.0%** | **2.3%** | **0.0%** | **1.00** | **0.00** |
| **Corrected Method (Held-Out 15x1)** | 15 (9 / 6) | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **1.00** | **0.00** |
