# Domain Diagnostic Comparison

## Development Matrix (32x1)
| Metric | Kitchen | Living | Workshop | Overall |
|---|---:|---:|---:|---:|
| Raw VLM role precision | 59.0% | 51.3% | 61.5% | 57.4% |
| Raw VLM role recall | 75.0% | 40.0% | 80.0% | 65.6% |
| Raw VLM role F1 | 65.8% | 44.7% | 68.8% | 60.2% |
| Interpreter-matched raw relation F1 | 0.0% | 0.0% | 2.5% | 0.8% |
| Raw VLM group F1 | 17.4% | 0.0% | N/A | 9.5% |
| Raw complete spec rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Sanitization success | 91.7% | 90.0% | 100.0% | 93.8% |
| Full canonicalization | 0.0% | 0.0% | 0.0% | 0.0% |
| Partial canonicalization | 75.0% | 80.0% | 80.0% | 78.1% |
| Any canonicalization success | 75.0% | 80.0% | 80.0% | 78.1% |
| Executable contract complete rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Search eligible | 0.0% | 0.0% | 0.0% | 0.0% |
| Search recovery success | N/A | N/A | N/A | N/A |
| Any verified grounding / eligible | N/A | N/A | N/A | N/A |
| Complete candidate grounding / eligible | N/A | N/A | N/A | N/A |
| Grounded expressed role coverage | N/A | N/A | N/A | N/A |
| A* invoked | 16.7% | 0.0% | 0.0% | 6.2% |
| Non-empty plan generated | 16.7% | 0.0% | 0.0% | 6.2% |
| Candidate planning success / generated | 100.0% | N/A | N/A | 100.0% |
| Partial-plan rate | 8.3% | 0.0% | 0.0% | 3.1% |
| Candidate goal coverage | 71.4% | N/A | N/A | 71.4% |
| Full-task success | 0.0% | 0.0% | 0.0% | 0.0% |
| Mean regions inspected | 0.00 | 0.00 | 0.00 | 0.00 |

## Held-Out Generalization Matrix (15x1)
# Section 40: Pipeline Diagnostic Table (Held-Out Generalization Matrix)

| Metric | Kitchen | Living Room | Workshop | Overall |
| :--- | ---: | ---: | ---: | ---: |
| Raw VLM role recall | 90.0% | 43.3% | 66.7% | 66.7% |
| Raw VLM role F1 | 78.4% | 52.0% | 57.1% | 62.5% |
| Interpreter-matched raw relation F1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Raw complete spec rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Executable contract complete rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Canonicalization success | 100.0% | 100.0% | 100.0% | 100.0% |
| Any verified grounding | 0.0% | 80.0% | 0.0% | 26.7% |
| Complete candidate grounding | 0.0% | 80.0% | 0.0% | 26.7% |
| Grounded role coverage | 0.0% | 0.0% | 0.0% | 0.0% |
| Non-empty plan generated | 0.0% | 0.0% | 0.0% | 0.0% |
| Candidate plan valid / generated | N/A | N/A | N/A | N/A |
| Partial-plan rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Candidate goal coverage | 0.0% | 0.0% | 0.0% | 0.0% |
| Full-task success | 0.0% | 0.0% | 0.0% | 0.0% |
| Mean regions inspected | 0.00 | 0.00 | 0.00 | 0.00 |
