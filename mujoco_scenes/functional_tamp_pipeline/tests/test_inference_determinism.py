"""Detector inference must be reproducible, and must say so honestly.

Three replays of the same archived FM responses through the same code disagreed
on workshop/W5/trial_03, moving its GT goal coverage between 0.333 and 1.0.  Two
of those replays ran at identical concurrency on a quiet machine, so the cause
was not machine load but intrinsic nondeterminism in GPU inference: no seed, no
cuDNN determinism, autotuning free to pick different kernels, and acceptance
thresholds as low as 0.001 for a detection to sit at.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mujoco_scenes.determinism import enable_deterministic_inference

REPO = Path(__file__).resolve().parents[3]


def test_determinism_is_requested_strictly_and_reported():
    """warn_only would downgrade an unsupported op to a warning and carry on.

    Claiming determinism while silently not having it is worse than not having
    it, so strict mode is requested and the report says what was achieved.
    """
    report = enable_deterministic_inference()
    assert report.get("strict") is True
    assert report.get("status") in {"DETERMINISTIC", "PARTIAL", "NO_TORCH"}
    if report.get("status") == "DETERMINISTIC":
        assert report["deterministic_algorithms"] == "strict"
        assert report["cudnn_deterministic"] is True
        assert report["cudnn_benchmark"] is False


def test_the_module_never_uses_warn_only():
    """Checked on the call itself, not on prose that explains why it is avoided."""
    import mujoco_scenes.determinism as det
    tree = ast.parse(Path(det.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = ast.unparse(node.func)
        if "use_deterministic_algorithms" not in target:
            continue
        kwargs = {k.arg for k in node.keywords}
        assert "warn_only" not in kwargs, (
            "permissive determinism would report success it did not achieve")
        assert node.args and ast.unparse(node.args[0]) == "True"


def test_torch_is_actually_configured_when_available():
    report = enable_deterministic_inference()
    if report.get("status") != "DETERMINISTIC":
        pytest.skip(f"torch not fully configurable here: {report.get('status')}")
    import torch
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_it_is_applied_before_every_detector_is_constructed():
    """After the model is built, cuDNN has already been free to autotune."""
    perception = (REPO / "mujoco_scenes" / "workshop_phase1" / "perception.py").read_text()
    tree = ast.parse(perception)
    initialisers = [n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "_initialize_model"]
    assert initialisers, "_initialize_model not found"
    body = ast.unparse(initialisers[0])
    assert "enable_deterministic_inference" in body
    assert body.index("enable_deterministic_inference") < body.index("YOLOWorld"), (
        "determinism must be applied before the model is constructed")

    worker = (REPO / "mujoco_scenes" / "semantic_grounding.py").read_text()
    assert "enable_deterministic_inference" in worker, (
        "the isolated detector worker process is not seeded")


def test_the_opt_out_exists_but_is_not_the_default():
    source = (REPO / "mujoco_scenes" / "determinism.py").read_text()
    assert "TAMP_NONDETERMINISTIC_INFERENCE" in source
    report = enable_deterministic_inference()
    assert report.get("status") != "DISABLED_BY_ENV"


def test_hash_seed_is_pinned_for_every_spawned_trial():
    """Set iteration order must not vary between runs.

    Two replays of the same archived responses disagreed on kitchen/K7/trial_01:
    one left coffee_container unseated with an 8-action plan, the other left
    coffee_source unseated with a 20-action plan.  Varying PYTHONHASHSEED
    reproduces exactly those two results (seed 1 vs seeds 2 and 3), so the cause
    is hash-order dependence in a choice among equally ranked candidates.

    It cannot be fixed in-process: the interpreter reads PYTHONHASHSEED at
    startup, so it has to be in the environment of every spawned trial.
    """
    from mujoco_scenes.determinism import REPRODUCIBLE_CHILD_ENV
    assert REPRODUCIBLE_CHILD_ENV["PYTHONHASHSEED"] == "0"
    # Both runners must take the whole environment from the one dict.  They
    # previously pinned different sets -- the replay driver pinned thread counts
    # and the GL backend, the live runner pinned neither -- so the experiment
    # would have run under a configuration no offline measurement was made under.
    for script in ("scripts/replay_frozen_distribution.py",
                   "scripts/run_live_repeat_experiment.py"):
        source = (REPO / script).read_text()
        assert "REPRODUCIBLE_CHILD_ENV" in source, (
            f"{script} spawns trials without the shared reproducible environment")


def test_the_determinism_report_says_when_the_hash_seed_is_unpinned():
    """Silence about an unpinned hash seed would read as determinism."""
    import os
    import importlib
    import mujoco_scenes.determinism as det
    saved = os.environ.pop("PYTHONHASHSEED", None)
    try:
        importlib.reload(det)
        report = det.enable_deterministic_inference()
        assert "NOT REPRODUCIBLE" in str(report.get("PYTHONHASHSEED"))
    finally:
        if saved is not None:
            os.environ["PYTHONHASHSEED"] = saved
        importlib.reload(det)


def test_the_live_identity_records_the_hash_seed():
    """A run whose hash seed is not on the record cannot be reproduced."""
    import importlib.util
    runner = REPO / "scripts" / "run_live_repeat_experiment.py"
    spec = importlib.util.spec_from_file_location("_live_runner_det", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = module._frozen_identity(base_url="u", variant_set="ALL", run_type="full")
    assert identity.get("pythonhashseed") == "0"


def test_every_detector_construction_site_is_seeded():
    """A seeded path that nobody takes is not determinism.

    The first version of this work seeded only the isolated worker in
    semantic_grounding.py.  MUJOCO_SEMANTIC_PROCESS_ISOLATION defaults to "0",
    so the in-process constructor is the path almost every run actually takes,
    and it was left unseeded while determinism was reported as applied.  This
    walks every function that builds a detector and requires the seeding call to
    come first in that same function.
    """
    sources = [
        REPO / "mujoco_scenes" / "semantic_grounding.py",
        REPO / "mujoco_scenes" / "workshop_phase1" / "perception.py",
    ]
    constructors = {"YOLOWorld", "YOLO"}
    found_any = False
    for path in sources:
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.unparse(func)
            builds = any(f"{name}(" in body for name in constructors)
            if not builds:
                continue
            found_any = True
            assert "enable_deterministic_inference" in body, (
                f"{path.name}:{func.name} constructs a detector without seeding it")
            assert body.index("enable_deterministic_inference") < min(
                body.index(f"{n}(") for n in constructors if f"{n}(" in body), (
                f"{path.name}:{func.name} seeds after constructing the detector; "
                "by then autotuning has already been free to run")
    assert found_any, "no detector construction site found -- the guard is vacuous"
