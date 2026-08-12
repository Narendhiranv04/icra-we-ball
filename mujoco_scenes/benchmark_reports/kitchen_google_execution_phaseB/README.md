# Kitchen Google Robot Execution — Phase B Scientific Freeze

## Scope

This report closes only physical `PICK → CARRY/MOVE → PLACE` for the
integrated kitchen. `POUR` and `STIR` remain explicit
`UNSUPPORTED_PHASE_C_OPERATOR` operations; no Phase-C symbolic effects are
applied by this executor.

## Architecture and evidence boundary

Frozen generic plan IDs are resolved one-to-one from observed semantic label,
observed source region and observed centroid. MuJoCo body names enter only at
the execution binding boundary. Execution then applies workspace/access
refinement, source-aware grasp and extraction, live bilateral-contact gating,
held-state checking, payload-aware base motion and physical placement
verification.

`validation_summary.json` is derived from every required evidence gate. PICK
success alone cannot close Phase B. `scientific_guard_report.json` gives a
method and evidence path for every guard, and `physical_run_manifest.json`
checksums the compact authoritative telemetry while large raw/video outputs
remain under `runs/`.

## Coverage

The calibrated matrix covers 3/3 fresh-reset trials for tabletop vessel, bowl,
utensil, kettle and coffee source, and for D1, D2, C1, B1, C2 vessel and C2
utensil retrieval. PLACE evidence covers serving support, object-relative
tool-to-bowl placement, and kettle/coffee-source return. Real carried MOVE
evidence covers all five physical families and verifies held transforms before
and after transport.

The C2 spoon presentation fixture may preserve its authored pose during opening
and approach, but is released before physical pre-close/finger contact. The
three authoritative C2 trials record the fixture inactive during bilateral
contact. Weld activation still requires sustained live contact by both finger
pads with the intended target.

## Calibration disclosure

C2 internal presentation/layout was calibrated for executable,
non-overlapping manipulation while preserving object identities, source
region, functional roles, Phase-1 witness semantics and Phase-2 plan semantics.
See `f1_physical_layout_equivalence.json`. The storage centroid gate and the
single Google base/shoulder mechanical-overlap allowance are numerically
audited in their named JSON files; no cabinet, shelf, forearm, payload or
finger collision exemption was added.

## Reproduction

Run `./reproduction_commands.sh quick` to audit captured evidence and run the
focused checks. Run `./reproduction_commands.sh full` to regenerate the
remaining carried-move/composition evidence before the full suite and report.
The exact command split is intentional: regenerating a report is not presented
as regenerating the physical experiments.
