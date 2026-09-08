# Corrective Recovery Execution Ledger — Functional-TAMP VLM Interface

> **Repository:** `Narendhiranv04/icra-we-ball`
> **Development branch:** `vlm-testing-pipeline`
> **Authoritative Plan:** `CORRECTIVE_RECOVERY_PLAN_ANTIGRAVITY_FLASH38.md`
> **Baseline HEAD:** `af2dde7341adff911c8a6c2d6b7870797ba87ed3`
> **Target Model:** `Qwen/Qwen3.5-9B` (alias `qwen35-9b`) on remote RTX 5090 via ngrok SSH tunnel

---

## 1. Stage Progress Ledger

| Stage | Status | Commit | Files changed | Tests | Raw replays | Live calls | Result | Remaining |
|---|---|---|---|---|---|---:|---|---|
| 0: Forensic Baseline & Safety Snapshot | PASSED | *pending* | `docs/corrective_recovery.md`, `scripts/forensics_baseline.py`, `scripts/generate_stage0_forensic_table.py` | 32-record baseline metric recompute | K1, K2, K3, L1, L6, W1, W2, W8 | 0 | Gate 0 passed: immutable forensic baseline verified; exact causal failure points established for all 8 representative cases. | Proceeding to Stage 1 |
| 1: Separate Online Executability from Offline Completeness | NOT STARTED | | | | | 0 | | |
| 2: Complete Explicit Operation -> Capability Bridge | NOT STARTED | | | | | 0 | | |
| 3: Robust Role Canonicalization (Qwen Paraphrases) | NOT STARTED | | | | | 0 | | |
| 4: Refactor Relations (Task Semantics vs Physical Verifiers) | NOT STARTED | | | | | 0 | | |
| 5: Fix Search Eligibility and Causal Recovery | NOT STARTED | | | | | 0 | | |
| 6: Planner Action Provenance & Goal Compilation | NOT STARTED | | | | | 0 | | |
| 7: Fix Raw Evaluation & First-Cause Attribution | NOT STARTED | | | | | 0 | | |
| 8: Structured Output & Qwen Inference Reliability | NOT STARTED | | | | | 0 | | |
| 9: Real-Trace Regression Suite | NOT STARTED | | | | | 0 | | |
| 10: Small End-to-End Live Probe | NOT STARTED | | | | | 8 | | |
| 11: Full Test Suite & Method Freeze | NOT STARTED | | | | | 0 | | |
| 12: Final Development 32x1 Matrix | NOT STARTED | | | | | 32 | | |
| 13: Held-Out / Generalization Matrix | NOT STARTED | | | | | 15 | | |
| 14: Final Comparative Analysis & Paper Reports | NOT STARTED | | | | | 0 | | |

---

## 2. Stage 0 — Forensic Baseline Snapshot (`af2dde73`)

### 2.1 Remote Infrastructure Verification

- **Host:** `cstar`
- **GPU:** NVIDIA GeForce RTX 5090 (32,607 MiB VRAM, Driver 580.95.05, CUDA 13.0)
- **vLLM Process:** PID 246883 (`vllm serve Qwen/Qwen3.5-9B --served-model-name qwen35-9b --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 ...`)
- **Remote Port:** `127.0.0.1:8000` (strictly non-public)
- **Persistent Local Tunnel:** `127.0.0.1:18000 -> remote 127.0.0.1:8000` via SSH master socket `~/.cache/tamp-vlm/qwen.sock`
- **Ngrok Process:** PID 4082688 (tunneling `0.tcp.in.ngrok.io:23808 -> remote :22`)
- **API Model Verification:** `curl -sS http://127.0.0.1:18000/v1/models` returns `qwen35-9b` (root: `Qwen/Qwen3.5-9B`, max_model_len: 32768)

### 2.2 Recomputed Baseline Primary Metrics

Directly recomputed from `benchmark_reports/final_corrected_32x1_20260908T192500IST_rerun/evaluation_records.json` using `evaluation_metrics.py`:

| Primary Metric | Value |
|---|---:|
| Outcome Correct | 0.0% (0/32) |
| Feasible Success | 0.0% (0/20) |
| Feasibility Recovery | 0.0% (0/13) |
| Goal Coverage | 0.0% |
| False Completion | 0.0% (0/12) |
| VLM Requests per variant | 1.00 |
| High-level Replans | 0.00 |

### 2.3 Representative Variant Forensic Table

| Variant | Domain | Feasible (GT) | Raw Semantics Expressed (Roles / Ops / Rels) | Production Mappings (Roles / Ops / Rels) | Contract Complete | Search State | Grounding State | A* Invoked? | First Cause (Reported) | Forensic Root Cause |
|---|---|---|---|---|---|---|---|---|---|---|
| **K1** | kitchen | True | Roles: 0, Ops: 0, Rels: 0 | Roles: OK; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: [] | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | Inference token runout / tab loop caused JSON truncation (`finish_reason: length`). Structural parsing failure upstream of compiler. |
| **K2** | kitchen | True | Roles: 6, Ops: 4, Rels: 4 | Roles: 2 unresolved/collided; Ops: 2 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 4 roles | Yes | `TASK_SPECIFICATION_FAILURE` | FM expressed complete kitchen task (sources, cups, bowls, stirrers, spoons, transfer ops, stir op, pair op). Compiler failed on `soup_serving_vessel` (`UNRESOLVED_SEMANTIC`) and `stirring_utensil` (`AMBIGUOUS_ROLE_MAPPING collision with coffee_serving_vessel`). Missing transfer capability marked operations `UNSUPPORTED_OPERATOR`. Search blocked by `contract_incomplete` gate. |
| **K3** | kitchen | True | Roles: 9, Ops: 7, Rels: 5 | Roles: 4 unresolved/collided; Ops: 2 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed valid roles/relations/ops. Compiler suffered role collision/unresolved mapping. Missing material-transfer capability. Search starved by contract gate even though physical objects were unobserved. |
| **L1** | living_room | True | Roles: 5, Ops: 2, Rels: 4 | Roles: 2 unresolved/collided; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: [] | Grounded: 2 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed refreshment and media roles and placement operations. Compiler rejected `refreshment_component` (`UNRESOLVED_SEMANTIC`) and `entertainment_control` (`UNRESOLVED_SEMANTIC`). Relations treated as `SOFT_SEMANTIC_ONLY`. Search never considered. |
| **L6** | living_room | True | Roles: 4, Ops: 2, Rels: 2 | Roles: 1 unresolved/collided; Ops: 1 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: [] | Grounded: 3 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed valid refreshment item & remote roles. Compiler rejected `refreshment_item` (`UNRESOLVED_SEMANTIC`) and rejected `move_remote_to_accessible_surface` (`UNSUPPORTED_OPERATOR`). Planner never engaged. |
| **W1** | workshop | True | Roles: 4, Ops: 3, Rels: 4 | Roles: 2 unresolved/collided; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed fastening tool, component, marked location, and surface. Compiler rejected `fastening_component` (`UNRESOLVED_SEMANTIC`) and `fastening_tool` (`UNRESOLVED_SEMANTIC`). Workshop V2 preconditions skipped. Search starved by contract gate. |
| **W2** | workshop | True | Roles: 5, Ops: 2, Rels: 4 | Roles: 2 unresolved/collided; Ops: Mapped/instantiated | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed valid fastening tool, component, fastener, and location. Compiler rejected `fastening_tool` (`UNRESOLVED_SEMANTIC`) and collided `marked_location` with `fastener`. Search starved. |
| **W8** | workshop | True | Roles: 4, Ops: 2, Rels: 3 | Roles: OK; Ops: 1 unsupported | Exec: False, Runtime: False | Eligible: False, Trace: `['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` | Grounded: 0 roles | No | `TASK_SPECIFICATION_FAILURE` | FM expressed `fastening_tool`, `fastener`, `target_assembly`, `workbench_surface`. Roles mapped, but `perform_fastening` marked `UNSUPPORTED_OPERATOR` because V2 capability preconditions were skipped / operator not bridged. Search starved. |

### 2.4 Verification of the 7 Forensic Hypotheses

1. **Workshop V2 physical-precondition skip:** Verified. In `semantic_compiler.py`, capability physical preconditions for Workshop were bypassed in V2, causing fastening operations to fail or drop preconditions.
2. **Hardcoded online domain completeness:** Verified. Online contract validation checks against hidden task requirements rather than validating whether what the FM expressed is self-consistent and executable.
3. **Kitchen missing material-transfer capability:** Verified. FM expressions like "Transfer coffee substance into serving vessel" and "Transfer water into serving vessel" fail with `UNSUPPORTED_OPERATOR` because the capability registry lacks generic material transfer / pouring.
4. **Role mapping failures on natural Qwen paraphrases:** Verified. Natural phrases like "Holds soup ready for consumption" (`soup_serving_vessel`), "An implement used to manipulate or install the fastening component" (`fastening_tool`), and "Device for controlling media playback" (`entertainment_control`) were marked `UNRESOLVED_SEMANTIC`. Spurious collisions (e.g. `stirring_utensil` colliding with `coffee_serving_vessel`) blocked valid roles.
5. **Search globally starved by contract gate:** Verified. Every single recovery variant (K2, K3, W1, W2, W8) had `search_state_trace: ['CONTRACT_INCOMPLETE_NOT_SEARCHABLE']` because contract completeness was gated on hidden domain completeness.
6. **Planner action generation from role identity:** Verified. In K2, A* was invoked despite missing operations and planned purely off role presence, violating operation provenance.
7. **Raw semantic evaluator undercoverage & overattribution:** Verified. Evaluator reported 0.0 F1 for relations and operations on K2/W1/W8 despite explicit, highly coherent FM semantics, and attributed 100% of failures to `TASK_SPECIFICATION_FAILURE` (FM omission) rather than compiler/interface rejection.

**Gate 0 Status: PASSED.**
