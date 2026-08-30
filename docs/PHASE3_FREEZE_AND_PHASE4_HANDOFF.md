# Phase 3 Freeze and Phase 4 Handoff Status

> [!NOTE]
> **PASS 3.6A.7.1 CANDIDATE — FINAL PRODUCTION-COMPATIBILITY AND COMPLETENESS HOTFIX**
> Phase 3 Status: **NOT FROZEN** (`PASS 3.6A.7.1 CODE CLOSURE: YES`, `VLM INTERFACE IMPLEMENTATION FROZEN: YES`, `READY FOR PASS 3.6B: YES`, `PHASE 3 FROZEN: NO`, `READY FOR PHASE 4: NO`).
> Pass 3.6A.7.1 closes the remaining deterministic production/interface compatibility mismatches: Living Room $G_F \leftrightarrow G_O$ task-anchor category alignment, domain-scoped Living unary checker capability constraints (`PLANAR_SUPPORT` only), strict generic interaction-group schema validation (`required_relations` mandatory), mode-safe canonical task-interface completeness validation, strict entity-kind compatibility checks, and distinct provenance tracking layers (`raw_vlm_response`, `validated_vlm_specification`, `canonicalization_trace`).

---

## 1. Executive Summary

Phase 3 establishes the canonical functional-grounding Task and Motion Planning (TAMP) pipeline architecture. Under Pass 3.6A.1 through Pass 3.6A.7.1, the interface boundary between the live Vision-Language Model (VLM) and downstream execution has been fully closed and frozen to adhere strictly to scientific zero-leakage principles with lossless, deterministic canonicalization.

The scientific pipeline is established as:
```
TASK + INITIAL MULTI-VIEW RGB
        ↓
       VLM (Qwen 3.5 9B / Foundation Model)
        ↓
raw VLM output (raw_vlm_response)
        ↓
strict generic schema validation (validated_vlm_specification)
        ↓
lossless deterministic canonicalization (phase3_6a7_v1, canonicalization_trace)
        ↓
mode-safe task-interface completeness validation
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

## 2. Pass 3.6A.7.1 Contract and Canonicalization Summary

1. **Complete Natural-Language Schema & Generic Prompt**:
   - Clean static prompt: format instruction only, zero concrete semantic examples, zero benchmark nouns.
   - Generic functional-asset vs task-payload/context distinction.
   - Standardized structured failure taxonomy: `MALFORMED_VLM_SPECIFICATION`, `UNMAPPED_FUNCTIONAL_CONCEPT`, `UNSUPPORTED_CHECKER_CAPABILITY`, `AMBIGUOUS_CANONICALIZATION`, `TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE`.
   - Interaction groups support generic `context_role` and `context_relations` with mandatory `required_relations` (`minItems=1`).
   - `failure_category` and `failure_reason` recorded in pipeline results and run manifests.

2. **Domain Lossless Canonicalization & Open-Vocabulary Handling**:
   - **Kitchen**: Canonical roles and operation groups with strict required top-level schema validation. Multi-view RGB resolution standardized to 1280×960.
   - **Living Room**: Lossless canonicalization into `FunctionalRequirementGraph` including task payload `OBJECT` roles (`CUP_SAUCER_SET`, `REMOTE`), contextual `FIXED_TARGET` roles (`SEATING_POSITION`, `SEATING_PAIR`), and `OperationGroup` bipartite matching specifications. Production $G_O$ task-anchor categories aligned with VLM $G_F$.
   - **Workshop**: Lossless normalized roles and relations. Pure $G_F \to G_O$ semantic category routing (`REACHES_TARGET`, `COMPATIBLE_WITH_TARGET`, `COMPATIBLE_WITH`) with run-local prompt tokenization.

3. **Strict Validation & Traceability**:
   - `VLM_CANONICALIZATION_VERSION = "phase3_6a7_v1"` attached to all $G_F$ metadata.
   - Distinct `unary_property_aliases` and `binary_relation_aliases` tables.
   - Domain-scoped unary checker capabilities enforced fail-closed (`PLANAR_SUPPORT` only in Living Room).
   - Mode-safe canonical task-interface completeness validator executed before planning/grounding.
   - Genuine separation of `raw_vlm_response`, `validated_vlm_specification`, and `canonicalization_trace`.

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
