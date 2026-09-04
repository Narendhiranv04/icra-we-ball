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

Geometric refinement is a MuJoCo cloned-scene sequence preflight,
an adaptation of ViLaIn-TAMP's MoveIt Task Constructor refinement. It is not an
exact MoveIt Task Constructor reproduction.

## Runner boundary

`mujoco_scenes.run_vilain_tamp_baseline` is the baseline-native CLI. It defaults
to planning-only and requires the explicit `--execute` flag before a runtime
adapter may start scored physical execution. The runner accepts no earlier
pipeline run directory or handoff. It records and locks repository, config,
domain, observation, problem, plan, refinement, and projection provenance before
execution; an inconsistent commit, branch, tracked tree, or artifact hash blocks
execution.

Runtime components are dependency-injected so importing the CLI and displaying
help cannot initialize MuJoCo, external planners, or foundation models. Recorded
offline model fixtures are supplied to the same component factory using
`--offline-model-fixtures`; they remain independent of proposed-method calls and
artifacts.

Example configuration-only validation, which performs no external calls:

```bash
python -m mujoco_scenes.run_vilain_tamp_baseline \
  --domain kitchen \
  --variant F0_ALL_VISIBLE \
  --planning-only \
  --dry-run
```

Scored execution is always opt-in:

```bash
python -m mujoco_scenes.run_vilain_tamp_baseline \
  --domain workshop \
  --variant F0_MANUAL_FIRST_ONE_REGION \
  --observation-mode fixed_full_inspection \
  --fast-downward /absolute/path/to/fast-downward.py \
  --val /absolute/path/to/Validate \
  --execute
```
