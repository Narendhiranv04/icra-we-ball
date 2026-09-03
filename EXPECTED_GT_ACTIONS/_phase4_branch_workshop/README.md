# Phase-4 branch Workshop expected-GT actions (reference copy, NOT active)

Provenance: `origin/phase4/execution-integration-replay-contract`
(`863611c`), copied 2026-09-02 without modification.

This directory exists only so the two Workshop action vocabularies can be
compared. **No loader reads it.** The active Workshop references remain in
`EXPECTED_GT_ACTIONS/workshop/`.

The two vocabularies are not interchangeable:

| | Active (`EXPECTED_GT_ACTIONS/workshop/`) | This copy |
|---|---|---|
| W1 action count | 28 | 6 |
| Operators | `MOVE_TO`, `OPEN_STORAGE`, `INSPECT_STORAGE`, `CLOSE_STORAGE`, `PICK`, `PLACE_ON_SURFACE`, `INSERT_FASTENER`, `DRIVE_FASTENER`, `VERIFY_REPAIR` | `OPEN`, `PICK`, `PLACE`, `SCREW` |
| Granularity | Per-navigation and per-inspection steps are explicit | Region open and fastening are single steps |

Kitchen and Living Room expected-GT actions are byte-identical between the two
branches; only Workshop diverges.

Switching the active vocabulary invalidates every Workshop planning comparison
already collected under the 28-action references, so those trials would have to
be rerun. Nothing has been switched.
