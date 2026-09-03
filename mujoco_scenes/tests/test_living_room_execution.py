"""Unit and integration tests for variant-general living-room execution and recording."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

import mujoco
import numpy as np

from mujoco_scenes.living_room_mobile_execution import (
    allocate_observed_placements,
    resolve_execution_entities,
    run_mobile_execution,
)
from mujoco_scenes.living_room_recorder import (
    L2_FIVE_CAMERAS,
    LivingRoomRecorder,
    LivingRoomRecorderTelemetry,
    create_camera_manifest,
)
from mujoco_scenes.living_room_region_function import EXPECTED_VARIANTS, INTEGRATED_PREFIX
from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene
from mujoco_scenes.run_living_room_execution import (
    DEFAULT_PHASE1_ROOT,
    DEFAULT_PHASE2_ROOT,
    normalize_variant_name,
    resolve_phase1_root,
    resolve_phase2_root,
)


@pytest.fixture
def phase1_root() -> Path:
    return resolve_phase1_root()


@pytest.fixture
def phase2_root() -> Path:
    return resolve_phase2_root()


def test_variant_registry_completeness():
    """Verify all 10 fixed-table variants are present and categorized."""
    assert len(EXPECTED_VARIANTS) == 10
    feasible = [v for v, status in EXPECTED_VARIANTS.items() if status == "COMPLETE"]
    infeasible = [v for v, status in EXPECTED_VARIANTS.items() if status != "COMPLETE"]
    assert len(feasible) == 6
    assert len(infeasible) == 4
    assert "F0_ALL_OBJECTS_IN_STAGING" in feasible
    assert "F5_LEFT_PAIR_ON_SHARED" in feasible
    assert "I0_NO_SHARED_TABLE" in infeasible
    assert "I3_NO_TABLES" in infeasible


def test_variant_name_normalization():
    """Test variant name normalization."""
    assert normalize_variant_name("f0_all_objects_in_staging") == "F0_ALL_OBJECTS_IN_STAGING"
    assert normalize_variant_name("F0") == "F0_ALL_OBJECTS_IN_STAGING"
    assert normalize_variant_name("f3") == "F3_LEFT_CUP_ON_SHARED"
    assert normalize_variant_name("I0_NO_SHARED_TABLE") == "I0_NO_SHARED_TABLE"


def test_infeasible_variants_terminate_cleanly(phase1_root: Path, phase2_root: Path):
    """Verify all infeasible variants terminate cleanly before physical execution."""
    infeasible_variants = [v for v, status in EXPECTED_VARIANTS.items() if status != "COMPLETE"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for variant in infeasible_variants:
            p1_dir = phase1_root / variant
            p2_dir = phase2_root / variant
            out_dir = tmp_path / variant
            result = run_mobile_execution(
                phase1_dir=p1_dir,
                phase2_dir=p2_dir,
                output_dir=out_dir,
                variant=variant,
                execute=True,
            )
            assert result["status"] == "INFEASIBLE_CONFIRMED"
            assert result["intended_outcome"] == "INFEASIBLE"
            assert result["execution_attempted"] is False
            assert result["mode"] == "NONE"
            assert (out_dir / "run_summary.json").is_file()
            assert (out_dir / "provenance_manifest.json").is_file()


def test_feasible_variants_entity_resolution(phase1_root: Path, phase2_root: Path):
    """Verify entity resolution maps five objects and three fixed regions."""
    feasible_variants = [v for v, status in EXPECTED_VARIANTS.items() if status == "COMPLETE"]
    for variant in feasible_variants:
        p1_dir = phase1_root / variant
        p2_dir = phase2_root / variant
        scene_name = f"{INTEGRATED_PREFIX}{variant}"
        scene = L2LivingRoomRegionScene(scene_name, "google")
        payloads = json.loads((p1_dir / "payload_registry.json").read_text())
        regions = json.loads((p1_dir / "region_registry.json").read_text())
        plan = json.loads((p2_dir / "plan.json").read_text())
        assignments = json.loads((p1_dir / "region_assignments.json").read_text())

        res = resolve_execution_entities(scene.model, scene.data, payloads, regions)
        assert len(res["objects"]) == 5
        assert len(res["regions"]) == 3

        placements = allocate_observed_placements(payloads, regions, plan, assignments)
        expected_places = sum(
            action["operator"] == "PLACE" for action in plan["actions"]
        )
        assert len(placements["placements"]) == expected_places
        assert placements["phase1_selected_packing_consumed"] is True


def test_provenance_mismatch_detection(phase1_root: Path, phase2_root: Path):
    """Verify that mismatched Phase 1 and Phase 2 inputs are rejected."""
    p1_dir = phase1_root / "F0_ALL_OBJECTS_IN_STAGING"
    p2_dir = phase2_root / "F1_LEFT_SAUCER_PREPLACED"
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "mismatch"
        with pytest.raises(RuntimeError, match="VARIANT_PROVENANCE_MISMATCH"):
            run_mobile_execution(
                phase1_dir=p1_dir,
                phase2_dir=p2_dir,
                output_dir=out_dir,
                variant="F0_ALL_OBJECTS_IN_STAGING",
                execute=False,
            )


def test_5camera_recorder_and_manifest():
    """Verify 5-camera recorder composition and manifest schema."""
    scene = L2LivingRoomRegionScene("L2_integrated_living_room_region_function_F0_ALL_OBJECTS_IN_STAGING", "google")
    tile_w, tile_h = 320, 180
    recorder = LivingRoomRecorder(
        scene,
        output_path=None,
        tile_width=tile_w,
        tile_height=tile_h,
        fps=10,
        show=False,
        record=False,
    )
    assert len(L2_FIVE_CAMERAS) == 5
    assert recorder.mosaic_width == tile_w * 3
    assert recorder.mosaic_height == tile_h * 2

    frame = recorder.capture_frame(force=True)
    assert frame is not None
    assert frame.shape == (tile_h * 2, tile_w * 3, 3)

    manifest = create_camera_manifest(
        output_path="test_5cam.mp4",
        mosaic_width=recorder.mosaic_width,
        mosaic_height=recorder.mosaic_height,
        tile_width=tile_w,
        tile_height=tile_h,
        fps=10,
        total_frames=1,
        duration_sim_s=0.1,
    )
    assert manifest["execution_mode"] == "INTEGRATED_LIVING_ROOM_PHASE3"
    assert len(manifest["tiles"]) == 6
    assert manifest["tiles"][-1]["name"] == "status_panel"
    recorder.close()
