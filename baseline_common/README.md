# Shared baseline boundary

This package contains only infrastructure that must remain identical across
comparison methods:

- visible-state observation and validated action records;
- scene action contracts used by the planning-only clients;
- image encoding and frozen-model transport/profile loading; and
- the final adapter to `MuJoCoSkillDispatcher`.

It contains no prompt, policy, subgoal logic, action-sequence logic,
replanning policy, effect ledger, goal interpretation, or scene runner. Those
belong in `llm3_baseline/` or `vlm_tamp_baseline/` and may evolve independently.

Do not add imports from either baseline package here. Boundary tests in
`vlm_tamp_baseline/tests/test_folder_boundaries.py` enforce this rule.

The end-to-end planning benchmark and 1/3/5-camera ablation procedure is in
`docs/BASELINE_BENCHMARK_RUNBOOK.md`. Use
`baseline_common.run_plan_gt_batch` to collect trials and
`baseline_common.summarize_plan_gt_batch` to generate the compact paper table.

The current live baseline adapter is intentionally kitchen-only and physically
supports `INSPECT`, `PICK`, `PLACE`, `POUR`, and `STIR`. Living-room and
workshop entries in the action catalogue are planning contracts, not claims of
implemented end-to-end baseline execution.
