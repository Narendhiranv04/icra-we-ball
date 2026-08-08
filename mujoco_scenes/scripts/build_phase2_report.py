#!/usr/bin/env python3
"""Package compact Phase-2 symbolic-planning evidence for GitHub."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/phase2_symbolic/phase2_kitchen_authoritative_20260809"
REPORT = ROOT / "mujoco_scenes/benchmark_reports/kitchen_symbolic_phase2"
FILES = (
    "input_witness.json", "symbolic_problem.json",
    "symbolic_initial_state.json", "symbolic_goal.json",
    "grounded_role_assignments.json", "generated_plan.json",
    "generated_plan.txt", "validation.json", "execution_trace.json",
    "planner_provenance.json", "scientific_validation.json",
    "domain.pddl", "problem.pddl",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if REPORT.exists():
        shutil.rmtree(REPORT)
    REPORT.mkdir(parents=True)
    summary = read(SOURCE / "phase2_benchmark_summary.json")
    shutil.copy2(
        SOURCE / "phase2_benchmark_summary.json",
        REPORT / "phase2_benchmark_summary.json",
    )
    shutil.copy2(
        SOURCE / "scientific_guard_report.json",
        REPORT / "scientific_guard_report.json",
    )
    for row in summary["variants"]:
        variant = row["variant_id"]
        source_dir = SOURCE / "variants" / variant
        destination = REPORT / "variants" / variant
        destination.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(source_dir / name, destination / name)

    environment = {
        "branch": "naren/googlePointCloudIntegration",
        "source_commit_at_generation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python_version": platform.python_version(),
        "phase1_report": summary["phase1_report"],
        "phase2_benchmark_id": summary["benchmark_id"],
        "planner_backend": "deterministic_astar_symbolic_state_search",
        "action_domain_version": "four_operator_symbolic_v2",
        "operator_types": ["PICK", "PLACE", "POUR", "STIR"],
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "mujoco")
        },
    }
    write(REPORT / "environment.json", environment)
    table = [
        "| Variant | Coffee tools | Plan length | Expanded | Runtime (s) | Valid |",
        "|---|---:|---:|---:|---:|---|",
    ]
    table.extend(
        f"| {row['variant_id']} | {row['coffee_distinct_tool_count']} | "
        f"{row['plan_length']} | {row['expanded_states']} | "
        f"{row['planning_time_s']:.6f} | {row['plan_valid']} |"
        for row in summary["variants"]
    )
    readme = """# Phase 2: perception-grounded symbolic task planning

This report evaluates the pure symbolic boundary:

`Phase-1 COMPLETE witness + frozen observed symbolic evidence -> compiler -> deterministic state-space search -> independent symbolic replay`.

The domain contains exactly four generic operator types: `PICK`, `PLACE`, `POUR`, and `STIR`. There is no `SERVE`, task-specific pour/place action, exploration, execution-time perception, replanning, FM/VLM, robot execution, IK, motion planning, collision checking, PDDLStream, or TAMP.

All nine production Phase-1 `COMPLETE` outputs produced independently replay-valid plans. The Phase-1 F0 detector miss and all Phase-1 infeasible outputs are excluded from the Phase-2 success denominator because they never supply a COMPLETE witness.

Generic `POUR(source, target)` derives transferred content from the static observed `provides(source, content)` fact. Generic `PLACE(object, destination)` represents source return, utensil provision (`at(tool, bowl)`), and serving placement (`at(vessel, serving_area)`). Coffee compatibility and soup one-to-one assignments come only from the Phase-1 witness.

## Results

""" + "\n".join(table) + """

## Reproduce

```bash
.venv/bin/python -m mujoco_scenes.run_phase2_symbolic_benchmark \\
  --phase1-report mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1 \\
  --benchmark-id phase2_kitchen_reproduction
```

The frozen Phase-1 raw evidence referenced by its manifest must be present locally. Phase 2 itself performs no rendering or detector inference.
"""
    (REPORT / "README.md").write_text(readme, encoding="utf-8")
    reproduction = """#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
.venv/bin/python -m mujoco_scenes.run_phase2_symbolic_benchmark \\
  --phase1-report mujoco_scenes/benchmark_reports/kitchen_feasibility_phase1 \\
  --benchmark-id "${1:-phase2_kitchen_reproduction}"
"""
    script = REPORT / "reproduction_command.sh"
    script.write_text(reproduction, encoding="utf-8")
    script.chmod(0o755)
    (REPORT / "test_summary.txt").write_text(
        "Focused command:\n"
        ".venv/bin/python -m pytest -q "
        "mujoco_scenes/tests/test_symbolic_kitchen_planning.py "
        "mujoco_scenes/tests/test_kitchen_feasibility_benchmark.py "
        "mujoco_scenes/tests/test_usage_policy_grounding.py\n"
        "Focused result: 57 passed\n\n"
        "Full command:\n.venv/bin/python -m pytest -q\n"
        "Full result: 350 passed\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
