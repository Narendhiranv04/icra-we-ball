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
