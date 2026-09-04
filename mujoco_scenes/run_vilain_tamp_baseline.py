"""Command-line entry point for the isolated ViLaIn-TAMP baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

from .baselines.vilain_tamp.config import (
    BaselineConfig,
    Domain,
    ExternalToolPaths,
    ModelCondition,
    ObservationMode,
)
from .baselines.vilain_tamp.runner import (
    BaselineRunner,
    RunOptions,
    RunnerComponents,
)


PACKAGE_ROOT = Path(__file__).resolve().parent / "baselines" / "vilain_tamp"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    ModelCondition.PAPER_FAITHFUL: PACKAGE_ROOT / "configs" / "paper_faithful.yaml",
    ModelCondition.MODEL_MATCHED: PACKAGE_ROOT / "configs" / "model_matched.yaml",
}
ComponentFactory = Callable[[BaselineConfig, RunOptions], RunnerComponents]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated ViLaIn-TAMP baseline. Planning-only is the "
            "default; physical execution requires --execute."
        )
    )
    parser.add_argument(
        "--domain",
        choices=tuple(item.value for item in Domain),
        required=True,
        help="Benchmark domain.",
    )
    parser.add_argument("--variant", required=True, help="Internal or paper variant ID.")
    parser.add_argument(
        "--observation-mode",
        choices=tuple(item.value for item in ObservationMode),
        default=None,
        help="Override the configured observation condition.",
    )
    parser.add_argument(
        "--model-condition",
        choices=tuple(item.value for item in ModelCondition),
        default=ModelCondition.PAPER_FAITHFUL.value,
        help="Select the paper-faithful or optional model-matched condition.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit baseline YAML; otherwise selected by model condition.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Exact run directory; defaults below the configured output root.",
    )
    parser.add_argument(
        "--offline-model-fixtures",
        type=Path,
        help="Absolute directory containing recorded model responses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved run without model, planner, or simulation calls.",
    )
    parser.add_argument(
        "--cp-limit",
        type=int,
        help="Maximum corrective problem revisions (0-3).",
    )
    parser.add_argument(
        "--fast-downward",
        type=Path,
        help="Absolute Fast Downward executable path.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        help="Absolute VAL executable path.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--planning-only",
        action="store_true",
        help="Stop after a validated, refined execution projection (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly permit scored physical execution and terminal evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Recorded random seed.")
    return parser


def resolve_run(
    args: argparse.Namespace,
) -> tuple[BaselineConfig, Path, RunOptions]:
    model_condition = ModelCondition(args.model_condition)
    config_path = (args.config or DEFAULT_CONFIGS[model_condition]).resolve()
    config = BaselineConfig.from_yaml(config_path)
    if config.model_condition is not model_condition:
        raise ValueError(
            "explicit config model condition differs from --model-condition"
        )
    domain = Domain(args.domain)
    observation_mode = (
        ObservationMode(args.observation_mode)
        if args.observation_mode
        else config.observation_mode
    )
    cp_limit = config.max_cp_corrections if args.cp_limit is None else args.cp_limit
    tools = ExternalToolPaths(
        fast_downward=(
            args.fast_downward.resolve()
            if args.fast_downward is not None
            else config.external_tools.fast_downward
        ),
        val=(
            args.val.resolve() if args.val is not None else config.external_tools.val
        ),
        fast_downward_version=config.external_tools.fast_downward_version,
        val_version=config.external_tools.val_version,
    )
    output_root = (
        args.output_directory.resolve()
        if args.output_directory is not None
        else (
            REPOSITORY_ROOT
            / config.output_root
            / domain.value
            / args.variant
            / observation_mode.value
            / model_condition.value
        ).resolve()
    )
    fixture_root = (
        args.offline_model_fixtures.resolve()
        if args.offline_model_fixtures is not None
        else None
    )
    if fixture_root is not None and not fixture_root.is_dir():
        raise ValueError(
            f"offline model fixture directory is missing: {fixture_root}"
        )
    config = replace(
        config,
        domain=domain,
        observation_mode=observation_mode,
        model_condition=model_condition,
        max_cp_corrections=cp_limit,
        output_root=output_root,
        external_tools=tools,
    )
    options = RunOptions(
        domain=domain,
        variant=args.variant,
        observation_mode=observation_mode,
        model_condition=model_condition,
        output_directory=output_root,
        cp_limit=cp_limit,
        execute=bool(args.execute),
        offline_fixture_root=fixture_root,
        fast_downward_path=tools.fast_downward,
        val_path=tools.val,
        random_seed=args.seed,
    )
    return config, config_path, options


def main(
    argv: Sequence[str] | None = None,
    *,
    component_factory: ComponentFactory | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, config_path, options = resolve_run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(options.to_dict(), indent=2, sort_keys=True))
        return 0
    if component_factory is None:
        parser.error(
            "runtime adapters are required; invoke main with a baseline-owned "
            "component factory (offline fixtures are completed in Stage 15)"
        )
    assert component_factory is not None
    components = component_factory(config, options)
    runner = BaselineRunner(
        config=config,
        config_path=config_path,
        repository_root=REPOSITORY_ROOT,
    )
    result = runner.run(options, components)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.run_status in {"SUCCESS", "PLANNING_COMPLETE", "INFEASIBLE_CORRECT"} else 1


if __name__ == "__main__":
    sys.exit(main())
