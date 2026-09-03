from argparse import Namespace
from pathlib import Path

from baseline_common.run_plan_gt_batch import _command, build_parser


def test_workshop_defaults_to_all_ten_variants():
    args = build_parser().parse_args(["--environment", "workshop", "--output-root", "runs/out"])
    assert args.environment == "workshop"
    assert args.max_model_calls == 1
    assert args.protocol == "native"


def test_workshop_batch_command_targets_new_runner():
    args = Namespace(base_url="http://127.0.0.1:18000/v1", model="qwen35-9b", max_tokens=8192, max_model_calls=1, goal=None, protocol="single_call")
    command = _command("vlm_tamp", "workshop", "W1", 0, 5, Path("runs/out"), args)
    assert command[2] == "vlm_tamp_baseline.run_workshop"
    assert "--max-model-calls" in command
    assert command[command.index("--protocol") + 1] == "single_call"
