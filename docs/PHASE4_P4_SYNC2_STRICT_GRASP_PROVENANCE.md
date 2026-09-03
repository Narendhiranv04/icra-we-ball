# Phase 4 P4-SYNC.2: strict grasp and provenance closure

Source: `7469b4e73d19af463d93dab724ae04cea6ba639d` (`git_dirty=false`). This is
the P4-SYNC.2 candidate rebased on Phase-3 P3-E.2. No Phase-3 source was changed.

Workshop inspection OPEN now closes the physical fingers and requires 25
consecutive bilateral contacts with the exact region handle geometry before
enabling its current-pose equality. It measures translation and angular snap
immediately and fails as `STRICT_PHYSICAL_HANDLE_GRASP_UNAVAILABLE` before
intentional articulation when the gate is unavailable. The strict Phase-4
adapter enables this mode; historical assisted experiments remain isolated.

Workshop simulator candidates now use the same world geometric AABB centre as
the Phase-3 evaluator, with body position only when no finite box/mesh centre
exists. The handoff hashes its manifest-declared `result.json`. Kitchen and
Living Room require an explicit replay artifact; only Workshop may use embedded
final-plan validation.

## Exact smoke protocol

Environment prefix:

```text
MUJOCO_MENAGERIE_PATH=/home/naren/third_party/mujoco_menagerie MUJOCO_GL=egl
```

Commands:

```text
/home/naren/miniconda3/bin/python -m mujoco_scenes.functional_tamp_pipeline.run --domain living_room --variant L1 --mode gt --search-order auto --dry-run --output-root /tmp/p4-sync2-phase3-final
/home/naren/miniconda3/bin/python -m mujoco_scenes.run_phase4_execution --domain living_room --variant L1 --mode gt --phase3-run /tmp/p4-sync2-phase3-final/living_room/L1/gt --output-root /tmp/p4-sync2-execution-final
/home/naren/miniconda3/bin/python -m mujoco_scenes.functional_tamp_pipeline.run --domain workshop --variant W1 --mode gt --search-order auto --dry-run --output-root /tmp/p4-sync2-phase3-final
```

L1 produced `ACTION_SEQUENCE_READY` and strict execution passed all 10/10
actions with no strict telemetry violation or direct task-state fallback. W1
remained upstream `INCOMPLETE` after the oracle order LEFT_DRAWER,
RIGHT_DRAWER, TOOL_CABINET, so Phase 4 correctly did not execute it.

The compact hashes and machine-readable outcomes are in
`docs/PHASE4_P4_SYNC2_PROVENANCE.json`. Large run artifacts remain under
`/tmp` and are intentionally not committed.
