# Kitchen Google-Robot Phase C completion evidence

Implementation commit: `5ff90a1aadee3e1ae3ad36f238e97ecf12272c6f`

Phase C is complete for the requested scope: POUR and STIR targets are limited
to countertop objects or box `B1`; cupboard vessels are excluded from both
operators and participate only through OPEN/PICK/PLACE. The frozen 26-action
input therefore yields a 23-action execution contract containing exactly four
POUR and two STIR actions.

Physical evidence passed POUR pair coverage (4/4), both source-family hardest-
target repeatability gates (3/3 each), POUR sequential same-held-object runs,
STIR pair coverage (2/2), STIR repeatability (3/3), and STIR sequential same-
held-object execution. The integrated run committed all six Phase-C ledger
events. No hidden regrasp, object substitution, direct payload qpos write, or
fluid-dynamics claim is made.

The broader frozen-plan run reached 14/23 actions and then stopped at the first
post-Phase-C `SERVE_COFFEE` grasp. This is a Phase-B serving-tail limitation,
not a missing POUR/STIR event, and is preserved explicitly in
`validation_summary.json` rather than represented as full-plan closure.

Use `reproduction_commands.sh` for focused tests and evidence commands.
