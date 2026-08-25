# Workshop Phase‑1 grounding contract

## Status

This document describes the redesigned 10-variant fixed-object Workshop benchmark. The retired 14-variant driver/fastener/surface/container contract is not compatible with this scene matrix and must not be used for current results.

## Task-derived roles

Canonical instruction:

> Find the compatible screw and the first compatible driver encountered while inspecting the storage regions. Insert the screw tip-down into the fixed workbench repair hole and drive it fully.

The requirement provider emits two roles:

| Function | Accepted categories | Geometry/relationship checks |
|---|---|---|
| `CAN_DRIVE_SCREW` | `screwdriver`, `power_driver` | `REACHES_TARGET`, then pairwise `COMPATIBLE_WITH` |
| `CAN_FASTEN` | `screw` | `COMPATIBLE_WITH_TARGET` |

The ranked open-vocabulary detector prompts are exactly `screwdriver`, `power drill`, `screw`, and `wooden hammer`. The hammer is deliberately detected as a distractor but is accepted by no functional role. Workbench, tool cart, tray, bin, bolt, wrench, pliers, and shelf prompts are not part of this contract.

`MAIN_WORKBENCH_ZONE` and its repair hole are fixed task geometry. They are not alternative VLM-selected region roles. Consequently, Phase‑1 feasibility depends only on a compatible observed `(driver, screw)` pair.

## Inspection and selection policy

Storage is inspected in the fixed order `LEFT_DRAWER`, `RIGHT_DRAWER`, `TOOL_CABINET`. Object tracks and evidence persist across stages. Search stops as soon as both required roles have a compatible pair.

If both drivers are eventually present, selection is based on first observation in the inspection order, not model confidence or arbitrary simulator ordering. This produces one-, two-, and three-region feasible episodes from object position alone.

After exhaustive inspection:

- no accepted driver produces `NO_COMPATIBLE_DRIVER`;
- no accepted screw produces `NO_COMPATIBLE_SCREW`;
- unresolved visual/geometric evidence produces `INSUFFICIENT_EVIDENCE`, not a fabricated infeasibility claim.

## Production/oracle boundary

Production code receives RGB-D observations, detector labels, masks, persistent generic track IDs, measured object geometry, and the task-derived role contract. It does not receive the variant ID, storage inventory, simulator body names, intended label, or oracle object dimensions.

Privileged scene metadata is used only to construct variants, validate compiled physical feasibility, associate predictions during post-hoc evaluation, and generate GT execution assignments. The normalized execution handoff accepts generic driver and screw track IDs plus an external entity resolver; the workbench target defaults to its fixed canonical ID.

## Current closure

The manual requirement provider, semantic grounding, geometric compatibility search, first-observed selection, planner handoff, and physical GT execution use the same two-role schema. The complete GT suite passes 10/10 variants and 266/266 actions. Live VLM/FM requirement generation and new-model prediction artifacts are the remaining model-facing integration work; retired frozen predictions are intentionally not replayed.

See [WORKSHOP_VARIANT_VISUAL_CATALOGUE.md](WORKSHOP_VARIANT_VISUAL_CATALOGUE.md) for all scene variants and [WORKSHOP_PIPELINE_READINESS.md](WORKSHOP_PIPELINE_READINESS.md) for execution details.

The planning-free live Qwen requirements integration and its exact server,
tunnel, validation, prompt-tuning, and grounding-smoke steps are documented in
[WORKSHOP_VLM_REQUIREMENTS_INTEGRATION.md](WORKSHOP_VLM_REQUIREMENTS_INTEGRATION.md).
