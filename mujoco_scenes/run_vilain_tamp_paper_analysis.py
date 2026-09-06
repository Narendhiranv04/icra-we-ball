#!/usr/bin/env python3
"""CLI entrypoint for ViLaIn-TAMP paper-readiness audit and aggregation.

Operates strictly read-only on the frozen baseline results root.
Does not perform any model inference, MuJoCo execution, or planning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mujoco_scenes.baselines.vilain_tamp.paper_metrics import run_full_paper_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper-readiness audit and generate publication tables for ViLaIn-TAMP."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/naren/ViLaIn-TAMP-results/stage24-4532495"),
        help="Path to frozen baseline experimental results root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/naren/ViLaIn-TAMP-results/stage24-4532495-paper-audit"),
        help="Output directory for paper metrics, CSVs, JSONs, and LaTeX tables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Auditing ViLaIn-TAMP results from: {args.results_root}")
    print(f"Outputting paper artifacts to:   {args.output_root}")
    
    overall = run_full_paper_analysis(args.results_root, args.output_root)
    
    print("\n=== ViLaIn-TAMP Paper Audit Complete ===")
    print(f"Total Non-Infrastructure Runs:     {overall['total_runs']}")
    print(f"Action Sequence Generation Rate:    {overall['action_sequence_generation_rate']*100:.2f}% ({overall['action_sequence_generation_count']}/{overall['total_runs']})")
    print(f"Non-Empty Plan Rate:                {overall['nonempty_action_sequence_rate']*100:.2f}% ({overall['nonempty_action_sequence_count']}/{overall['total_runs']})")
    print(f"VAL-Valid Plan Rate:                {overall['val_valid_plan_rate']*100:.2f}% ({overall['val_valid_plan_count']}/{overall['total_runs']})")
    print(f"Execution-Ready Plan Rate:          {overall['execution_ready_plan_rate']*100:.2f}% ({overall['execution_ready_plan_count']}/{overall['total_runs']})")
    print(f"Physical Execution Rate (Uncond):   {overall['physical_execution_rate_unconditional']*100:.2f}% ({overall['execution_success_count']}/{overall['total_runs']})")
    print(f"Benchmark Requirement Coverage:     {overall['benchmark_requirement_coverage_micro']*100:.2f}%")
    print(f"Generated Goal Satisfaction Rate:   {overall['generated_goal_satisfaction_rate']*100:.2f}% (N={overall['generated_goal_evaluated_count']})")
    print(f"Feasibility Accuracy (Raw Rule):    {overall['feasibility_confusion']['accuracy']*100:.2f}% (Coverage: {overall['feasibility_confusion']['decision_coverage']*100:.2f}%)")
    print(f"\nArtifacts successfully written to:  {args.output_root}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
