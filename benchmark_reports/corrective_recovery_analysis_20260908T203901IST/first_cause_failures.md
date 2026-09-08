# First-Cause Failure Attribution Analysis

This analysis enforces strict first-cause attribution separating foundation model task specification omissions from compiler representation defects, object discovery failures, grounding failures, and planning failures.

## 1. First-Cause Distribution on Feasible Tasks

| First-Cause Category | Development (20 Feasible) | Held-Out (9 Feasible) | Description |
| :--- | :---: | :---: | :--- |
| **`FM_SEMANTIC_OMISSION`** | 0 (0.0%) | 0 (0.0%) | Raw VLM output truly omitted necessary semantic task participants or operations |
| **`GRAPH_COMPILATION_FAILURE`** | 20 (100.0%) | 9 (100.0%) | Raw semantics were expressed by VLM but failed compiler relation/role mapping |
| **`OBJECT_DISCOVERY_FAILURE`** | 0 (0.0%) | 0 (0.0%) | Contract compiled but required physical entities could not be discovered via search |
| **`FUNCTIONAL_ASSIGNMENT_FAILURE`** | 0 (0.0%) | 0 (0.0%) | Candidates discovered but failed physical verification / joint role binding |
| **`PLANNING_FAILURE`** | 0 (0.0%) | 0 (0.0%) | Valid grounding obtained but symbolic A* search failed to reach goal |

## 2. Detailed Pipeline Diagnostics (All Variants)

### Development Matrix (32 Variants)
| Detailed Cause | Count | Interpretation |
| :--- | ---: | :--- |
| `CANONICALIZATION_AMBIGUITY` | 17 | Compiler representation ambiguity or relation signature mismatch |
| `CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC` | 3 | Compiler representation ambiguity or relation signature mismatch |
| `NONE` | 12 | Correctly identified infeasible task |

### Held-Out Generalization Matrix (15 Variants)
| Detailed Cause | Count | Interpretation |
| :--- | ---: | :--- |
| `CANONICALIZATION_AMBIGUITY` | 8 | Compiler representation ambiguity or relation signature mismatch |
| `CANONICALIZATION_UNRESOLVED_REQUIRED_SEMANTIC` | 1 | Compiler representation ambiguity or relation signature mismatch |
| `NONE` | 6 | Correctly identified infeasible task |
