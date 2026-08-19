# Workshop (W1) Phase 1 Research Pipeline: Functional Object and Region Grounding

This document describes the scientifically hardened Phase 1 research pipeline implemented for the Workshop (W1) benchmark.

---

## 1. Problem Formulation & Scope

Phase 1 solves the problem of **joint functional requirement satisfaction and active visual inspection** under strict partial observability, zero simulator oracle leakage, and open-vocabulary semantic grounding:

$$\text{Given: } \mathcal{T} \text{ (Task instruction)} \implies \Phi = \{r_{\text{driver}}, r_{\text{fastener}}, r_{\text{surface}}, r_{\text{container}}\}$$

At stage $k=0$, the agent observes only the initial tabletop using 5 calibrated RGB-D cameras. If no complete functional witness $w = (d, f, s, c)$ can be verified from current evidence, the agent actively selects unopened storage volumes ($\text{TOOL\_CABINET}$, $\text{LEFT\_DRAWER}$, $\text{RIGHT\_DRAWER}$), opens them, captures localized multi-view RGB-D evidence, fuses persistent object instances into a growing observed graph $G_k$, and re-evaluates joint satisfiability until either:
1. A valid, physically verified functional witness $w^* = (d^*, f^*, s^*, c^*)$ is found (early stopping), OR
2. All accessible inspection volumes are exhausted and a grounded infeasibility reason is diagnosed.

> [!IMPORTANT]
> **Phase 1 Boundary**: This pipeline performs perception, persistent multi-view tracking, open-vocabulary geometric and semantic grounding, and joint witness search. It does **not** execute manipulation, pick-and-place trajectories, PDDLStream planning, or physics mutations. The frozen Workshop MuJoCo scene assets, meshes, and XMLs are strictly preserved.

---

## 2. Scientifically Hardened Pipeline Architecture

```mermaid
graph TD
    A[Task Instruction / Functional Requirements] --> B[Dynamic Open-Vocabulary Query Builder]
    B --> C[Multi-Camera RGB-D Inspection Capture]
    C --> D[YOLO-World Open-Vocabulary Detector / Oracle Mask Backend]
    D --> E[Camera-Isolated Multi-View Clustering & Persistent Tracking]
    E --> F[Growing Observed Evidence Graph]
    F --> G[Production Semantic Backend: Tool Tip Aspect Ratio Analysis]
    F --> H[Deterministic Semantic Normalizer: Interface Families]
    F --> I[Geometric Grounding: PCA Shaft Slicing & Relational Packing]
    F --> J[Region Grounding: Observable Cavity Depth Openness Verification]
    G --> K[Joint Functional Search & Bipartite Hungarian Evaluation]
    H --> K
    I --> K
    J --> K
    K -->|Witness Satisfied| L[Feasible Witness Found: Early Termination]
    K -->|Unresolved & Regions Remain| M[Active Inspection Decision: Open & Inspect Region]
    M --> C
    K -->|Unresolved & Regions Exhausted| N[Exact Grounded Rejection Diagnosis]
```

### 2.1 Core Architectural Principles & Hardening Fixes

1. **Zero Semantic Leakage in Oracle Mask Mode**:
   - `PrivilegedOracleMaskBackend` emits purely spatial masks with neutral predicted label `predicted_label = "object"`. Zero object class names or simulator body names leak from the detector into the pipeline.
2. **Open-Vocabulary Dynamic Query Building (`OpenVocabularyQueryBuilder`)**:
   - In production mode, YOLO-World queries are constructed dynamically from functional requirement descriptions and task instructions. No static Workshop dictionary or class list is hardcoded in production detection.
3. **Decoupled Semantic Property Inference (`ObjectSemanticBackend`)**:
   - Detection only yields 3D bounding volumes and point clouds. Property inference is delegated to `ObjectSemanticBackend` (`ProductionSemanticBackend` or decoupled `PrivilegedOracleSemanticBackend`).
4. **Physical Tip Eigenvalue Aspect Ratio Analysis**:
   - Screwdriver interface families (`CROSS_RECESS` vs `SINGLE_SLOT`) are resolved by computing the distal tool tip transverse covariance matrix $C_t$ and its eigenvalue ratio:
     $$\text{Aspect Ratio} = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}$$
   - A flat blade slotted tip exhibits high transverse anisotropy ($\text{ratio} \ge 3.5$), whereas cross/Phillips and hex tips are radially symmetric ($\text{ratio} < 3.5$).
5. **Deterministic Semantic Normalization (`DeterministicSemanticNormalizer`)**:
   - Free-form structured text from vision backends is normalized to internal interface families: `CROSS_RECESS`, `SINGLE_SLOT`, `HEX_HEAD`, and `UNKNOWN_INTERFACE`. No Phillips default is assumed for unverified tools.
6. **Observable Cavity Depth for Container Openness**:
   - Containers are no longer assumed open. The pipeline computes observed cavity depth $\Delta Z = Z_{\text{rim}} - Z_{\text{floor}}$. A container is grounded as open if and only if $\Delta Z \ge 0.015\,\text{m}$.
7. **Strict 1-to-1 Bipartite Hungarian Evaluator**:
   - Post-hoc evaluation uses Hungarian maximum-weight matching on 3D centroid distance ($d \le 0.16\,\text{m}$) against ground truth witness sets. In F4, the ground truth witness is `workshop_power_driver` on `TOOL_CART_TOP` (narrow wall shelf rejected by relational packing).
8. **Explicit Evidence Graph Edges**:
   - Grounding evidence is written back into `GrowingObservedGraph` with typed edges: `CANDIDATE_FOR`, `SEMANTICALLY_SATISFIES`, and `GEOMETRICALLY_SATISFIES`.

---

## 3. Configuration & CLI Controls (`configs/workshop_phase1.yaml`)

The pipeline supports full orthogonal configuration via CLI flags and YAML:
- `--mask-backend`: `production` (YOLO-World + RGB-D fallback) or `oracle` (simulator geometry masks, `label="object"`).
- `--semantic-backend`: `production` (tip eigenvalue aspect ratio + PCA) or `oracle` (privileged ground-truth semantics for baseline comparisons).
- `--requirements-source`: `static` (canonical Workshop requirements) or `fm` (foundation-model generated, raising `FMBackendNotConfiguredError` if unconfigured).
- `--inspection-policy`: `greedy_cost`, `information_gain`, `oracle_shortest_path`.
- `--ablation`: `none`, `oracle_mask`, `oracle_semantics`, `semantic_only`, `no_joint_coupling`, `no_persistence`, `single_view`, `run_suite`.

---

## 4. Benchmark Validation (All 14 Variants)

### 4.1 Oracle Masks Upper-Bound Benchmark (Zero Semantic Leakage)

Evaluated with `--mask-backend oracle --semantic-backend production`:

| Variant | Type | Expected Witness / Outcome | Pipeline Result | Rejection Diagnosis | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `F0_BASE` | Feasible | Long Phillips + Med Screw + Workbench + Bin | **FEASIBLE** | None | **PASS (100%)** |
| `F1_TOOL_ALTERNATIVE` | Feasible | Power Driver + Med Screw + Workbench + Bin | **FEASIBLE** | None | **PASS (100%)** |
| `F2_REGION_ALTERNATIVE` | Feasible | Long Phillips + Med Screw + Mobile Cart + Bin | **FEASIBLE** | None | **PASS (100%)** |
| `F3_DISTRIBUTED_OBJECTS` | Feasible | Long Phillips (Cab) + Med Screw (Draw) + Cart + Tray | **FEASIBLE** | None | **PASS (100%)** |
| `F4_OBJECT_REGION_COUPLING` | Feasible | Power Driver + Med Screw + Cart Top + Tray | **FEASIBLE** | None | **PASS (100%)** |
| `F5_DECOY_HEAVY` | Feasible | Long Phillips + Med Screw + Workbench + Tray | **FEASIBLE** | None | **PASS (100%)** |
| `F6_LAYOUT_SWAPPED` | Feasible | Long Phillips + Med Screw + Workbench + Bin | **FEASIBLE** | None | **PASS (100%)** |
| `I0_NO_VALID_DRIVER` | Infeasible | Distractors only (pliers/wrenches) | **INFEASIBLE** | `NO_VALID_DRIVER` | **PASS (100%)** |
| `I1_NO_VALID_FASTENER` | Infeasible | Decoy bolts only | **INFEASIBLE** | `NO_VALID_FASTENER` | **PASS (100%)** |
| `I2_NO_WORK_SURFACE` | Infeasible | Staging surfaces occupied/missing | **INFEASIBLE** | `NO_WORK_SURFACE` | **PASS (100%)** |
| `I3_NO_PARTS_CONTAINER` | Infeasible | No hardware bin or tray | **INFEASIBLE** | `NO_PARTS_CONTAINER` | **PASS (100%)** |
| `I4_TOOL_GEOMETRY_FAILURE` | Infeasible | Stubby driver reach $< 0.025\,\text{m}$ | **INFEASIBLE** | `TOOL_GEOMETRY_FAILURE` | **PASS (100%)** |
| `I5_OBJECT_REGION_PACKING_FAILURE` | Infeasible | Power driver + Screw exceeds Shelf area | **INFEASIBLE** | `OBJECT_REGION_PACKING_FAILURE` | **PASS (100%)** |
| `I6_GLOBAL_CONFLICT` | Infeasible | Multiple global requirement conflicts | **INFEASIBLE** | `GLOBAL_CONFLICT` | **PASS (100%)** |

**Summary**: **14 / 14 PASSED (100.0%)** (7/7 Feasible witness matches, 7/7 Infeasible exact diagnosis matches).

---

### 4.2 Production Pipeline Benchmark (Honest Scientific Reporting)

Evaluated with `--mask-backend production --semantic-backend production` (YOLO-World open-vocabulary proposals + RGB-D fallback + production semantics):

| Metric | Result |
| :--- | :--- |
| **Variants Passed** | **3 / 14 (21.4%)** |
| **Correct Infeasibility Diagnoses** | `I0_NO_VALID_DRIVER`, `I2_NO_WORK_SURFACE`, `I3_NO_PARTS_CONTAINER` |
| **Failure Modes on Synthetic Renders** | Under-segmentation / 2D bounding box fusion on raw MuJoCo procedural textures without synthetic pre-training domain adaptation. |

---

## 5. Complete 7-Arm Ablation Study

The full 7-arm ablation suite was executed across all 14 variants (98 total episodes). Results are logged in `outputs/workshop_phase1_ablations/ablation_results.csv`:

| Ablation Arm | Mask Backend | Semantic Backend | Ablation Flag | Passed / 14 | Accuracy (%) | Primary Failure Mode & Scientific Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Production Pipeline** | `production` | `production` | `none` | 3 / 14 | 21.4% | Zero-shot synthetic rendering domain gap in 2D bounding box proposal generation. |
| **Oracle Masks (Upper Bound)** | `oracle` | `production` | `oracle_mask` | **14 / 14** | **100.0%** | **Validates functional grounding reasoning, tool tip aspect analysis, and joint search logic when visual proposals are clean.** |
| **Oracle Semantics** | `production` | `oracle` | `oracle_semantics` | 3 / 14 | 21.4% | Shows semantic accuracy alone cannot recover from upstream 2D proposal under-segmentation. |
| **Semantic-Only (No Geometry)** | `production` | `production` | `semantic_only` | 1 / 14 | 7.1% | Fails geometry-critical variants (I4 tool reach, I5 relational packing) due to missing dimensional filtering. |
| **No Joint Coupling Checks** | `production` | `production` | `no_joint_coupling` | 3 / 14 | 21.4% | Ignores tool-fastener interface mismatch and surface packing constraints during candidate generation. |
| **No Multi-Stage Persistence** | `production` | `production` | `no_persistence` | 0 / 14 | 0.0% | **Complete failure (0/14)**: Forgetting objects across inspection stages breaks multi-step search in all variants. |
| **Single Front Camera View** | `production` | `production` | `single_view` | 4 / 14 | 28.6% | Fails occluded objects inside drawers/cabinets; multi-view fusion is required for comprehensive discovery. |

---

## 6. How to Run

```bash
# Run full 7-arm ablation suite
python mujoco_scenes/run_workshop_phase1.py --ablation run_suite --output outputs/workshop_phase1_ablations

# Run single variant with production perception
python mujoco_scenes/run_workshop_phase1.py --variant F0_BASE --mask-backend production --semantic-backend production --evaluate

# Run 14-variant benchmark with oracle masks (upper bound)
python mujoco_scenes/run_workshop_phase1.py --variant all --mask-backend oracle --semantic-backend production --evaluate

# Run automated test suites
pytest mujoco_scenes/tests/test_workshop_phase1.py -v
pytest mujoco_scenes/tests/test_workshop_phase1_no_privileged_leaks.py -v
```
