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
    parser.add_argument(
        "--file-prefix",
        type=str,
        default="paper_",
        help="Prefix for output artifact filenames (e.g. 'paper_' or 'smoke_').",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Auditing ViLaIn-TAMP results from: {args.results_root}")
    print(f"Outputting paper artifacts to:   {args.output_root}")
    print(f"Artifact prefix:                 {args.file_prefix}")
    
    overall = run_full_paper_analysis(args.results_root, args.output_root, file_prefix=args.file_prefix)
    
    print("\n=== ViLaIn-TAMP Main Manuscript Metrics ===")
    print(f"Total Non-Infrastructure Runs:     {overall['total_runs']}")
    print(f"Feasible Variant Runs:             {overall['feasible_runs']}")
    print(f"Outcome Correct Rate:               {overall['outcome_correct_rate']*100:.2f}% ({overall['outcome_correct_count']}/{overall['total_runs']})")
    print(f"Feasible-Task Success Rate:         {overall['feasible_task_success_rate']*100:.2f}% ({overall['feasible_task_success_count']}/{overall['feasible_runs']})")
    print(f"Goal Coverage (Micro):              {overall['goal_coverage_micro']*100:.2f}% ({overall['goal_requirements_passed_feasible']}/{overall['goal_requirements_total_feasible']})")
    fc_rate = f"{overall['false_completion_rate']*100:.2f}%" if overall['false_completion_rate'] is not None else "N/A"
    print(f"False Completion Rate:              {fc_rate} ({overall['false_completion_count']}/{overall['declared_completion_count']})")
    print(f"Physical Plan Found Rate:           {overall['physical_plan_found_rate']*100:.2f}% ({overall['physical_plan_found_count']}/{overall['feasible_runs']})")
    raw_vlm = overall['raw_vlm_requests']
    print(f"Raw VLM Requests:                   {raw_vlm.get('mean', 0.0):.2f} ± {raw_vlm.get('std', 0.0):.2f}")
    replans = overall['high_level_replans']
    print(f"High-Level Replans:                 {replans.get('mean', 0.0):.2f} ± {replans.get('std', 0.0):.2f}")
    print(f"\nArtifacts successfully written to:  {args.output_root}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
