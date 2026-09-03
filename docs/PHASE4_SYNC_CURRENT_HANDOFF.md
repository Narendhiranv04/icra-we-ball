# Phase-4 current-handoff sync record

Date: 2026-09-01

Scope: Phase 4 only. No Phase-3 semantics, grounding, canonicalization, or
controller calibration was changed.

## Source revision and fresh artifacts

The pass started at `b0db1c05`. Phase 3 advanced concurrently twice, so the
final artifact and smoke gate was regenerated from current remote revision
`9e43fc10` into:

`runs/phase4_sync_phase3_9e43fc10_20260901`

Key SHA-256 values:

| Variant | Artifact | SHA-256 |
|---|---|---|
| K1 | `run_manifest.json` | `3cab6e27d4ca2ddd68740ed62869e39f15253a2a06f41a9cc0f2c4956a25a4e4` |
| K1 | `graph_grounding_result.json` | `54cd5bdc4ad733b9b2d74f63cc33d696e40b213a68d53cc43d232fb4ae1268e3` |
| K1 | `functional_specification.json` | `183f37b493e572eb294ab0211c2ef7285c708154cd2772be445cd4915d0c4c2b` |
| L1 | `run_manifest.json` | `07a8a09633a55b7c14fa0ca0d2a474d46059727a585f6fc3ece44b4a4afa6ab9` |
| L1 | `graph_grounding_result.json` | `37a79472f42b11e435e24ce5378640509f504488d01790a25a930947bbdee7bc` |
| L1 | `action_sequence/plan.json` | `bf716ff8ea6d85aecc7cb8cbb250e2520ba87e895e8aad0cad07861f91d49018` |
| W1 | `run_manifest.json` | `59812864d77df3493607583238c43ea93b3f6c2f7076400d7fa55f6e2d29e4e9` |
| W1 | `graph_grounding_result.json` | `b4c25c7a65d87a3c31a70a836b30d11845b79e0bc04581d3a98df6192b405845` |
| W1 | `action_plan.json` | `603bccb18381691f6e22fa4ee6f21f904ff0d70386a29f7b62b9413c0f754b66` |
| W6 | `run_manifest.json` | `f2314b638c355681553a6af408c2699f7cbf6c9bab94ec52b39810b5034d2940` |
| W6 | `graph_grounding_result.json` | `d2b4f2ffe9e6d6b2159fb06281906f0c3fa8e8bd2b920bc5ff0dba0b15e6ec9b` |
| W6 | `action_plan.json` | `ffbfa3f4c81b6c627c85135874c5a6e4741b3a9f9f408905bb2e3b65d255a85e` |

Phase-4 smoke artifacts are under:

`runs/phase4_sync_execution_9e43fc10_20260901`

## Current strict smoke results

- K1: Phase 3 ended `INCOMPLETE` after exhausting D1, D2, C2, B1, and C1;
  no phi or final plan exists. Phase 4 reports
  `CURRENT_UPSTREAM_PHASE3_BLOCKED` and performs no simulator action.
- L1: current handoff was accepted and physical execution reached action 10.
  Nine of ten actions completed. Final PLACE failed strict postcondition
  settling (`linear_speed_m_s=0.03837`, threshold `0.02`). No assistance or
  direct-state fallback was reported.
- W1: canonical `object_0002` and `object_0003` resolved to the medium screw
  and power driver. Both persisted inspection regions physically opened and
  three of five task actions completed. Execution stopped at SCREW because
  the legacy implementation writes task state and remains disabled.
- W6: current Phase 3 is `ACTION_SEQUENCE_READY`. Canonical `object_0001`
  resolved to the cabinet screw and `object_0002` resolved by source/centroid
  identity to the cabinet long manual driver despite its observed semantic
  label. Physical cabinet OPEN passed; the first PICK failed at the existing
  calibrated preclose pose (`0.4292 m` miss). No controller tuning was made.

## Verification

`86 passed in 69.36s` across the Phase-4, Kitchen execution, Living Room
mobile execution, and Workshop execution focused test files.

The next Phase-4 pass is physical controller work: Kitchen once Phase 3 again
produces a K1 plan, Living Room placement settling/clearance, Workshop cabinet
PICK calibration, and finally a real force/joint fastening model. The latter
was intentionally not implemented in this sync pass.
