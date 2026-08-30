# Phase 3 Freeze and Phase 4 Handoff Status

> [!NOTE]
> **PASS 3.6A.6 CANDIDATE — FORMAL LIVING GRAPH COMPLETENESS, FAILURE TAXONOMY CLOSURE, AND FINAL AUDITABILITY FIX**
> Phase 3 Status: **NOT FROZEN** (`PHASE 3 FROZEN: NO`, `READY FOR PHASE 4: NO`).
> Pass 3.6A.6 completes Living Room graph canonicalization with operation groups and task-explicit payload/context nodes, eliminates role-guessing fallbacks, establishes strict failure category propagation, and fixes auditability for live evaluation.

---

## 1. Executive Summary

Phase 3 establishes the canonical functional-grounding Task and Motion Planning (TAMP) pipeline architecture. Under Pass 3.6A.1 through Pass 3.6A.6, the interface boundary between the live Vision-Language Model (VLM) and downstream execution has been realigned to adhere strictly to scientific zero-leakage principles with lossless, deterministic canonicalization.

The scientific pipeline is established as:
```
TASK + INITIAL MULTI-VIEW RGB
        ↓
       VLM (Qwen 3.5 9B / Foundation Model)
        ↓
complete natural-language functional specification
        ↓
lossless deterministic canonicalization (phase3_6a5_v1)
        ↓
canonical G_F
        ↓
══════════════════════════════════════════════════════════════════════════
               VLM HAS NO ROLE BELOW THIS POINT
══════════════════════════════════════════════════════════════════════════
        ↓
G_O (sequential inspection) → grounding (phi*) → symbolic compiler → A*
```

---

## 2. Pass 3.6A.6 Contract and Canonicalization Summary

1. **Complete Natural-Language Schema & Generic Prompt**:
   - Clean static prompt: format instruction only, zero concrete semantic examples, zero benchmark nouns.
   - Generic functional-asset vs task-payload/context distinction.
   - Standardized structured failure taxonomy: `MALFORMED_VLM_SPECIFICATION`, `UNMAPPED_FUNCTIONAL_CONCEPT`, `UNSUPPORTED_CHECKER_CAPABILITY`, `AMBIGUOUS_CANONICALIZATION`, `TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE`.
   - Interaction groups support generic `context_role` and `context_relations`.
   - `failure_category` and `failure_reason` recorded in pipeline results and run manifests.

2. **Domain Lossless Canonicalization & Open-Vocabulary Handling**:
   - **Kitchen**: Canonical roles and operation groups with strict required top-level schema validation. Multi-view RGB resolution standardized to 1280×960.
   - **Living Room**: Lossless canonicalization into `FunctionalRequirementGraph` including task payload `OBJECT` roles (`CUP_SAUCER_SET`, `REMOTE`), contextual `FIXED_TARGET` roles (`SEATING_POSITION`, `SEATING_PAIR`), and `OperationGroup` bipartite matching specifications.
   - **Workshop**: Lossless normalized roles and relations. Zero heuristic role-guessing fallbacks. Pure $G_F \to G_O$ semantic category sourcing with run-local prompt tokenization.

3. **Strict Validation & Traceability**:
   - `VLM_CANONICALIZATION_VERSION = "phase3_6a5_v1"` attached to all $G_F$ metadata.
   - Complete transformation trace, raw decompositions, and normalization audits recorded in all run artifacts.

---

## 3. Test Suite Verification

- Comprehensive unit and contract test suites cover all domains (Kitchen, Living Room, Workshop), schema validation, bipartite bipartite matching, open-vocabulary token routing, and error taxonomy mappings.
- Zero test regressions across pipeline modules.

---

## 4. Phase-4 Invariant Boundaries

When Phase 3 is eventually frozen, Phase 4 robot execution will respect the following invariant boundaries:
1. **Zero Modifications to Upstream Science**: Phase 4 MUST NOT alter VLM prompt templates, detector vocabularies, candidate mapping logic, or graph grounding algorithms.
2. **Evaluator Provenance Integrity**: All runs in Phase 4 must preserve manifest SHA-256 provenance (`specification_sha256`, `search_order_source_effective`, `search_seed_effective`).
3. **Independent Isolation**: Physical execution traces and videos are written to per-run output subdirectories without mutating Phase 3 artifacts.
