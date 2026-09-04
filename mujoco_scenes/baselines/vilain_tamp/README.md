# ViLaIn-TAMP baseline

This package is a **parallel ViLaIn-TAMP baseline**. It owns its observations,
PDDL problems, symbolic plans, refinement records, and execution projections.
It does not run through `G_F`, `G_O`, `ground_graph()`, or `phi*`, and it does
not consume their assignments, witnesses, search results, or action plans.

The baseline does not fabricate `Phase3Handoff` or Phase-4 handoff artifacts.
Safe generic infrastructure—raw benchmark observations, fixed scene mechanics,
generic geometry and motion facilities, and low-level controllers—may
eventually be shared through baseline-owned adapters. Proposed-method semantic
information, task-aware search, grounding results, roles, operation bindings,
functional witnesses, and planning audits may never be shared with this path.

The eventual geometric refinement is a MuJoCo cloned-scene sequence preflight,
an adaptation of ViLaIn-TAMP's MoveIt Task Constructor refinement. It is not an
exact MoveIt Task Constructor reproduction. Stage 1 provides contracts,
configuration, artifact utilities, and boundary tests only; it performs no
planning, model calls, refinement, execution, or simulation.
