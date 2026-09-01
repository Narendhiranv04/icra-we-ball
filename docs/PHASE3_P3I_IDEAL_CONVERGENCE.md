# Phase 3 P3-I & P3-I.1 Ideal Convergence & Semantic Authority Record

> **Pass**: P3-I.1 — Final Phase-3 Semantic Authority & Ideal-Convergence Evidence Closure
> **Branch**: `naren/pipeline_check`
> **Date**: 2026-09-01
> **Status**: COMPLETE & FROZEN

---

## 1. Executive Summary

Pass P3-I.1 establishes one unified system-owned semantic acceptance authority (`role_semantic_ontology.py`), eliminates detector-vocabulary vs grounding-acceptance authority leaks, restores task-scoped YOLO-World vocabulary construction, preserves reviewed ontology aliases against runtime prompt collisions, and establishes complete 3-run deterministic convergence across Kitchen (K1), Living Room (L1), and Workshop (W1).

All deterministic Phase-3 architecture is now **FROZEN**.

---

## 2. Shared Semantic Acceptance Authority

- **Ontology Version**: `phase3_p3i_1_semantic_ontology_v1`
- **System Authority**: `mujoco_scenes/functional_tamp_pipeline/role_semantic_ontology.py` defines `SYSTEM_ROLE_SEMANTIC_CATEGORIES` for all domains.
- **Provider Convergence**:
  - `GTSpecProvider` and `VLMSpecProvider` both query the shared system ontology.
  - VLM-supplied candidate categories strictly populate detector dynamic vocabularies and do NOT mutate or expand grounding acceptance categories.
  - Provable metadata provenance flags: `role_semantic_ontology_version`, `semantic_acceptance_source = "SYSTEM_ROLE_SEMANTIC_ONTOLOGY"`, `candidate_categories_used_for_role_identity = False`, `candidate_categories_used_for_grounding_acceptance = False`, `candidate_categories_used_for_detector_vocabulary = True`.

---

## 3. Ideal Fixture SHA-256 Provenance

| Domain | Fixture Path | SHA-256 Checksum |
|---|---|---|
| Kitchen | `mujoco_scenes/functional_tamp_pipeline/fixtures/kitchen_K1.json` | `076fa9379e517e26b07acb0905f4a52ffbe9829d3c5bccf10badb965aa260cd2` |
| Living Room | `mujoco_scenes/functional_tamp_pipeline/fixtures/living_room_L1.json` | `a72d86cf1e054f6d7a9533be6e04b5ad6eeb5400190d6f52abb75447d29c30a4` |
| Workshop | `mujoco_scenes/functional_tamp_pipeline/fixtures/workshop_W1.json` | `42e8c9215ec7f5f946c050ef65c978c4edb9c8efcf524c0fc6095d6fb01eba72` |

---

## 4. Full Pipeline Ideal Convergence Evidence (F1–F14)

### 4.1 Convergence Summary Table

| Domain | Variant | Runs | Terminal Status | $G_F$ Hash Match (GT vs VLM) | Role Identity Recall | Exact Role Recall | Relation Recall | Plan Audit Valid | Replay Valid (0 FM Calls) |
|---|---|---|---|---|---|---|---|---|---|
| Kitchen | K1 | 3/3 PASS | `ACTION_SEQUENCE_READY` | Control Validated | 1.000 | 0.833 | 1.000 | TRUE | TRUE |
| Living Room | L1 | 3/3 PASS | `ACTION_SEQUENCE_READY` | Exact Match (`True`) | 1.000 | 1.000 | 1.000 | TRUE | TRUE |
| Workshop | W1 | 3/3 PASS | `ACTION_SEQUENCE_READY` | Exact Match (`True`) | 1.000 | 1.000 | 1.000 | TRUE | TRUE |

### 4.2 Deterministic Evidence Hashes

| Metric / Artifact | Kitchen (K1) | Living Room (L1) | Workshop (W1) |
|---|---|---|---|
| **VLM $G_F$ Hash** | `005939710dc90fbcb08d0695a1259a0085044f62577648c9ecb93b777023a513` | `c9a99a222b0a0afd738321240fb89f79a5f39a53c8c491a5b5d6455aec583262` | `adec449b43019f80466c53bed72fc19a88006aabd8c8209951cc805816506e54` |
| **GT $G_F$ Hash** | `61943da8955b0401bcf06037f53c3297aaf899be0cc6013b8e141a11de67fc0a` | `c9a99a222b0a0afd738321240fb89f79a5f39a53c8c491a5b5d6455aec583262` | `adec449b43019f80466c53bed72fc19a88006aabd8c8209951cc805816506e54` |
| **Search Contract Hash** | `d110d2efad4ca269f98101540d1902513df033f56051058468476d810d76f865` | `092a8ab7aaf930d400544fc46eac2ee0e005e7e010cc0c86eda195fd613211bd` | `e85b9357f45438e9d73370e6f62110747cea3c2fcd9979eb07667db13ca7029d` |
| **$G_O$ Hash** | `9cd90c2480ce1c8bab2db54f9038800e390067bf82774b19839ba0b0dee83a7f` | `db04ba9ead53b6596ada4714ef4c43a99b8ab64edfbce81c5b97ce46696fb60d` | `a664ff1134c82d1adcb835f5bc15107ccc48dd11e26823f2c84fdf7bd5cd21d8` |
| **Assignment $\\phi^*$ Hash** | `57e60530123427976c48c5de55791bd37da730ad77e8776eff91037d2294d84a` | `6ee23f00ec43aa723cccf86272fcb6910a7b0c38f6a105b8acb7626a2c1a2fd7` | `3cac91aac1e88e0f5f5b239a9e4d6dadc24c1aa6d49f79e5a7718020a833b9ed` |
| **Plan Hash** | `570ee9bf88d14aec5266a2599674f69c81f68e66045732766098c6bb7ab04e44` | `358c7f285910d8001950f7f85c4161406b48d3096d8c819a0344a82d587bdc4d` | `1a5db75ee18e2e079bcbe01a4fe2c531d0ed0f09e77418b937bc471383e27961` |
| **Plan Steps Count** | 24 | 10 | 5 |

---

## 5. Dedicated Authority Unit Test Suite

8/8 unit tests in `mujoco_scenes/functional_tamp_pipeline/tests/test_p3i_1_semantic_authority.py`:
- `test_a1_shared_semantic_authority_across_domains`: PASS
- `test_b1_candidate_categories_do_not_mutate_grounding_acceptance_workshop`: PASS
- `test_b1_candidate_categories_do_not_mutate_grounding_acceptance_kitchen`: PASS
- `test_b2_candidate_categories_order_invariance`: PASS
- `test_c4_kitchen_task_scoped_detector_vocabulary`: PASS
- `test_c3_unresolved_fm_detector_terms_policy`: PASS
- `test_d_workshop_ontology_alias_preservation`: PASS
- `test_e_provenance_and_metadata_verification`: PASS

---

## 6. Architecture Freeze Sign-Off

The deterministic Phase-3 architecture is declared complete and frozen.
Next step: Live VLM evaluation (Pass P3-J).
