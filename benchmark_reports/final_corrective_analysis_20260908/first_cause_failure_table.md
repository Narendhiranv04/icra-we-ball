# First-Cause Failure Analysis (Feasible Tasks)

| First-Cause Category | Development Matrix (N=20) | Held-Out Matrix (N=9) | Description |
| :--- | ---: | ---: | :--- |
| `TASK_SPECIFICATION_FAILURE` | 20 (100.0%) | 9 (100.0%) | Raw VLM omitted required semantic roles/relations without GT prompt injection |

# Detailed Pipeline Cause Breakdown (All Variants)

| Detailed Cause Code | Development Matrix (N=32) | Held-Out Matrix (N=15) |
| :--- | ---: | ---: |
| `FM_SEMANTIC_OMISSION` | 18 | 9 |
| `FM_STRUCTURAL_ERROR` | 2 | 0 |
| `NONE` | 12 | 6 |
