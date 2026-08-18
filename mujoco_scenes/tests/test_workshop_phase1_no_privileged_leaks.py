"""Static and dynamic audit tests ensuring ZERO privileged leaks in Phase 1 pipeline."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
import pytest

from mujoco_scenes.workshop_scene import WorkshopScene
from mujoco_scenes.workshop_phase1.types import MaskBackendType
from mujoco_scenes.workshop_phase1.inspection_controller import WorkshopPhase1InspectionController
from mujoco_scenes.workshop_phase1.serialization import assert_no_backend_names


FORBIDDEN_SIMULATOR_METHODS = [
    "get_observed_instances",
    "privileged_get_visible_backend_instances",
    "privileged_get_storage_contents",
    "privileged_get_ground_truth_solution",
    "privileged_get_variant_metadata",
]

PRODUCTION_MODULES = [
    "mujoco_scenes/workshop_phase1/types.py",
    "mujoco_scenes/workshop_phase1/__init__.py",
    "mujoco_scenes/workshop_phase1/capture.py",
    "mujoco_scenes/workshop_phase1/tracking.py",
    "mujoco_scenes/workshop_phase1/evidence_graph.py",
    "mujoco_scenes/workshop_phase1/semantic_grounding.py",
    "mujoco_scenes/workshop_phase1/geometric_grounding.py",
    "mujoco_scenes/workshop_phase1/region_grounding.py",
    "mujoco_scenes/workshop_phase1/functional_search.py",
    "mujoco_scenes/workshop_phase1/fm_adapter.py",
    "mujoco_scenes/workshop_phase1/requirements.py",
    "mujoco_scenes/workshop_phase1/serialization.py",
    "mujoco_scenes/workshop_phase1/inspection_controller.py",
]


def test_static_code_scan_no_forbidden_methods():
    """Static AST scan: verify production modules do NOT call or reference forbidden oracle APIs."""
    base_dir = Path("/home/naren/RA_iiith")

    for rel_path in PRODUCTION_MODULES:
        file_path = base_dir / rel_path
        assert file_path.exists(), f"File {file_path} does not exist."

        source = file_path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SIMULATOR_METHODS:
            assert forbidden not in source, f"Forbidden oracle reference '{forbidden}' found in production file {rel_path}!"


def test_dynamic_monkeypatch_no_privileged_calls():
    """Dynamic test: monkeypatch all privileged methods on WorkshopScene to raise RuntimeError and verify production execution succeeds."""
    scene = WorkshopScene("none", variant="F0_BASE")

    def forbidden_trap(*args, **kwargs):
        raise RuntimeError("PRIVILEGE VIOLATION: Production code attempted to call a privileged oracle method!")

    scene.get_observed_instances = forbidden_trap
    scene.privileged_get_visible_backend_instances = forbidden_trap
    scene.privileged_get_storage_contents = forbidden_trap
    scene.privileged_get_ground_truth_solution = forbidden_trap
    scene.privileged_get_variant_metadata = forbidden_trap

    with tempfile.TemporaryDirectory() as tmpdir:
        controller = WorkshopPhase1InspectionController(
            mask_backend=MaskBackendType.PRODUCTION,
            output_dir=Path(tmpdir),
        )
        # Execution must run without invoking any of the trapped methods
        result = controller.run_episode(scene)
        assert result.status in ("FEASIBLE", "INFEASIBLE")


def test_output_json_contains_zero_simulator_names():
    """Verify that all production JSON dumps contain generic IDs only and zero simulator backend body strings."""
    scene = WorkshopScene("none", variant="F0_BASE")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        controller = WorkshopPhase1InspectionController(
            mask_backend=MaskBackendType.ORACLE,
            output_dir=out_dir,
        )
        controller.run_episode(scene)

        # Inspect all written production json files
        for json_file in out_dir.glob("*.json"):
            import json
            data = json.loads(json_file.read_text(encoding="utf-8"))
            assert_no_backend_names(data)
