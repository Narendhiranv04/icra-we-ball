# Phase-4 execution certification audit

Date: 2026-08-31

Scope: GT Phase-3 handoffs only. No Phase-3 VLM/canonicalization code was changed.

## Meaning of `--assisted-suite`

| Domain/operator | Actual assisted path | Classification | Robot motion | Direct payload pose write | Postcondition basis |
|---|---|---|---|---|---|
| Kitchen `PICK` | Skips the normal pick, writes the free body to a gripper-frame carry pose, enables its weld | TELEPORT / POSE ASSIGNMENT | No task-directed pick controller | Yes | Live weld and held-state inspection |
| Kitchen `PLACE` | Resolves a placement target, disables the weld, writes free-joint position/quaternion, settles MuJoCo | TELEPORT / POSE ASSIGNMENT | No place controller | Yes | Only hand-empty plus primitive success; no independent relation check |
| Kitchen `POUR` | Commands a scripted wrist sinusoid | ASSISTED_CONTROLLER | Wrist actuator only | No | Held-state plus a flag derived from that held-state; no fluid/task effect is modeled |
| Kitchen `STIR` | Commands a scripted wrist sinusoid | ASSISTED_CONTROLLER | Wrist actuator only | No | Held-state plus a flag derived from that held-state; no mixture/task effect is modeled |
| Living Room `PICK` | Normal mobile stance planner and calibrated pick controller | REAL_CONTROLLER | Yes | No | Live held-object/weld/gripper inspection |
| Living Room `PLACE` | Normal mobile stance planner and calibrated place controller | REAL_CONTROLLER | Yes | No | Independent live support contact, footprint, height, penetration, overlap, release and (strictly) settling/orientation checks |
| Workshop `PICK` | Strict controller first; on failure, moves payload to gripper and enables weld | ASSISTED_CONTROLLER, falling back to TELEPORT / POSE ASSIGNMENT | Yes on strict attempt | On fallback | Live weld plus dispatcher held state; symbolic state also updated |
| Workshop `PLACE` | Strict controller first; on failure, installs a fixture or writes a surface pose | ASSISTED_CONTROLLER, falling back to DIRECT_SIM_STATE_CHANGE / TELEPORT | Yes on strict attempt | On fallback | Per-action check relies on symbolic location and hand state; terminal validation reads live simulator state |
| Workshop `SCREW` | Strict controller first; on failure, writes the fastener seated pose, enables fixture and repair state | ASSISTED_CONTROLLER, falling back to TELEPORT / POSE ASSIGNMENT | Yes on strict attempt | On fallback | Per-action check is symbolic; terminal validation independently reads live fastener pose, fixture and simulator repair state |

No final Phase-3 plan in the audited feasible set contains `OPEN` or `CLOSE`.
Workshop inspected-region state is restored before plan execution with
`WorkshopScene.open_container`: an environment articulation actuator is
commanded and MuJoCo is stepped; the joint threshold is checked. This is a
DIRECT_SIM_STATE_CHANGE, not robot-handle execution. Kitchen's dormant
assisted `OPEN`/`CLOSE` paths similarly call scene articulation directly.

## Certification levels demonstrated

The level is assigned to the complete demonstrated episode, not to its best
individual primitive.

| Variants | Highest demonstrated level | Reason |
|---|---|---|
| K1-K6 | E0 | Handoff and dispatch pass, but assisted `PLACE` lacks an independent spatial postcheck and `POUR`/`STIR` have no modeled semantic effect. PICK alone reaches E1. A separate unassisted K1 first PICK demonstrated E2 for that primitive only. |
| L1-L6 | E2 | All plan actions use the mobile/arm/gripper controllers. Final relations are read independently from live MuJoCo state. L3 used the documented structural (rather than strict settling/orientation) acceptance for one live placement, still based on simulator state. |
| W1-W5, W7-W8 | E1 | Plan actions in W1/W2 were controller-driven, but episode setup restores searched containers through direct simulator articulation. W3/W4/W5/W7/W8 additionally used direct assisted action fallback. Terminal repair validation is live simulator-based. |
| W6 | Not E0 / blocked handoff | Persisted concrete handle and source contradict the manifest-selected scene. Phase 4 rejects it before execution. |

## Postcondition audit

- Kitchen `PICK`: independent enough for E1; reads active weld and held geometry.
- Kitchen `PLACE`: insufficient; success can follow the primitive's `True`
  result plus internal hand-empty state without checking live support/region.
- Kitchen `POUR` and `STIR`: insufficient; `pour_motion_verified` and
  `stir_motion_verified` are derived from held-state in assisted mode. No
  target effect exists in MuJoCo. Internal `symbolic_effects_applied` can be
  true without an independently represented task effect.
- Kitchen terminal verification always returns success after counting actions;
  it is not a terminal task-state validator.
- Living Room `PICK` and `PLACE`: live state is checked independently. The
  final validator rechecks every required relation, including initially
  satisfied relations.
- Workshop `PICK`: live weld is checked, but symbolic state participates.
- Workshop `PLACE` and `SCREW`: per-action checks can pass from updated
  symbolic state; however, full-run terminal validation independently checks
  driver support, fastener tip pose/orientation, installed fixture, empty hand
  and simulator repair state.

## Entity-resolution audit

- No adapter changes phi-star, substitutes a planner argument, adds task-level
  actions, or invokes an FM.
- Workshop uses persisted concrete handles and now rejects source/scene
  disagreement loudly.
- Kitchen and Living Room are not purely explicit handle lookups. Both perform
  one-to-one semantic/source/centroid association at the simulator boundary.
- Kitchen additionally reads frozen functional-role assignments to supply
  fallback semantic labels when observed semantics are absent. It does not
  select a new role filler, but this is an exception to a strict
  role-independent entity adapter.
- Living Room uses the persisted payload semantic role as a compatibility gate
  before nearest-centroid matching. It fails on ambiguity/distance rather than
  silently substituting a plan argument.
- Assisted fallbacks are operator/domain-level, not keyed to K/L/W variant IDs.

## W6 causal diagnosis

1. Requested paper variant: `W6`.
2. Mapping: `W6 -> F5_POWER_FIRST_THREE_REGIONS` in
   `final_paper_variant_labels.py`; this mapping is correct.
3. Phase-3 instantiates `WorkshopScene(..., F5_POWER_FIRST_THREE_REGIONS)`.
4. Current scene truth places `workshop_power_driver` in `LEFT_DRAWER`; the
   cabinet contains the medium screw and long manual driver.
5. Saved G_O `object_0002` is sourced from `TOOL_CABINET` but is semantically
   labeled `power_driver`. Its measured 0.220 m slender geometry corresponds
   to the cabinet's long manual driver.
6. `WorkshopDomainAdapter._physical_handle` converts that semantic label into
   concrete handle `workshop_power_driver`; grounding preserves the track's
   source as `TOOL_CABINET`.
7. The final plan therefore requests `PICK(workshop_power_driver,
   TOOL_CABINET)`.
8. Phase 4 correctly instantiates the manifest's F5 scene.
9. Its source audit finds the actual power driver in `LEFT_DRAWER` and rejects
   the handoff.

First semantic divergence: the Phase-3 detector/tracker belief recorded the
cabinet's manual driver track as `power_driver`. First concrete simulator
handle/location contradiction: `WorkshopDomainAdapter._physical_handle` in
`functional_tamp_pipeline/domains/workshop.py` converted that mislabeled track
to `workshop_power_driver` while retaining `TOOL_CABINET`.

Root-cause classification: **GROUNDING_ARTIFACT_BUG**, triggered by a Phase-3
perception semantic misclassification and made executable by the semantic-label
to concrete-handle conversion. It is not a paper-label mapping, manifest,
Phase-4 instantiation, or stale-artifact bug.

A clean isolated GT rerun was performed under
`runs/phase4_certification_w6_regen`. It used the correct F5 scene and search
order, inspected all three regions, and terminated `INFEASIBLE` because the
current perception run failed to retain a fastener. Therefore there is no new
valid action sequence to execute, and W6 was not patched or manually edited.
