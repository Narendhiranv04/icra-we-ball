from __future__ import annotations

import json

import pytest

from mujoco_scenes import package_final_paper_gt as packager
from mujoco_scenes.package_final_paper_gt import _living_objects_regions


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_action_comparison_is_exact_and_living_room_ignores_only_moves():
    expected = [
        {"operator": "PICK", "arguments": ["object_0001"]},
        {"operator": "PLACE", "arguments": ["object_0001", "region_0001"]},
    ]
    executed = [
        {"operator": "MOVE", "arguments": [{"x": 1.0}]},
        *expected,
    ]
    assert not packager._compare_actions("kitchen", expected, executed)["exact_match"]
    comparison = packager._compare_actions("living_room", expected, executed)
    assert comparison["exact_match"]
    assert comparison["executed_action_count"] == 3
    assert comparison["compared_executed_action_count"] == 2


def test_living_inventory_reports_variant_specific_initial_regions(tmp_path):
    _write_json(tmp_path / "execution_entity_resolution.json", {
        "objects": [{
            "generic_object_id": "object_0001",
            "semantic_role": "saucer",
            "backend_body": "a2_snack_left",
            "backend_centroid_world_m": [-1.72, 0.90, 0.561],
        }],
        "regions": [{
            "generic_region_id": "region_0001",
            "backend_support_geom": "a2_personal_left_top",
            "backend_centroid_world_m": [-1.73, 1.05, 0.55],
        }],
    })
    _write_json(tmp_path / "refined_mobile_plan.json", {"actions": []})

    text = _living_objects_regions(tmp_path, "F1_LEFT_SAUCER_PREPLACED")

    assert "initial_region=PERSONAL_TABLE_LEFT" in text
    assert "logical_region=PERSONAL_TABLE_LEFT" in text
    assert "for all movable payloads" not in text


def test_incremental_package_appends_and_replaces_one_variant(tmp_path, monkeypatch):
    recorded_root = tmp_path / "recorded"
    unrecorded_root = tmp_path / "unrecorded"
    timings_root = tmp_path / "timings"
    output_root = tmp_path / "final"
    monkeypatch.setattr(packager, "EXPECTED_GT_ROOT", tmp_path / "expected")

    monkeypatch.setattr(packager, "_probe", lambda _: {
        "duration_s": 1.0, "frame_rate": 20.0,
    })
    monkeypatch.setattr(packager, "_builders", lambda _: (
        lambda variant, source: (f"actions {variant}", f"assignments {variant}"),
        lambda source, variant: f"objects {variant}",
    ))

    def prepare(variant):
        for root in (recorded_root, unrecorded_root):
            source = root / "kitchen" / variant
            source.mkdir(parents=True)
            _write_json(source / "summary.json", {
                "success": True,
                "execution_profile": "STRICT_ROBOT_PHYSICAL_PRIMITIVES",
                "assisted_action_count": 0,
                "direct_payload_pose_write_count": 0,
                "direct_object_qpos_write_count": 0,
            })
            _write_json(source / "gt_plan.json", {"actions": []})
            _write_json(source / "gt_assignment.json", {"is_feasible": True})
            if root == recorded_root:
                (source / f"{variant}_5cam.mp4").write_bytes(b"video")
                _write_json(source / "camera_manifest.json", {})
        timing = timings_root / "kitchen"
        timing.mkdir(parents=True, exist_ok=True)
        (timing / f"{variant}_with_recording_seconds.txt").write_text("2.0")
        (timing / f"{variant}_without_recording_seconds.txt").write_text("1.0")

    prepare("F0_ALL_VISIBLE")
    first = packager.package(
        recorded_root, unrecorded_root, timings_root, output_root,
        environments=("kitchen",),
        selected_variants={"kitchen": ["F0_ALL_VISIBLE"]},
        append=True,
    )
    assert first["total_variants"] == 1
    assert first["records"][0]["variant"] == "K1"
    assert first["records"][0]["source_variant"] == "F0_ALL_VISIBLE"
    assert first["all_gt_action_sequences_exact_match"]
    assert (output_root / "kitchen" / "K1" / "expected_gt_actions.txt").exists()
    assert (output_root / "kitchen" / "K1" / "executed_gt_actions.txt").exists()
    comparison = json.loads(
        (output_root / "kitchen" / "K1" / "gt_action_comparison.json").read_text()
    )
    assert comparison["exact_match"]

    prepare("V2")
    second = packager.package(
        recorded_root, unrecorded_root, timings_root, output_root,
        environments=("kitchen",), selected_variants={"kitchen": ["V2"]},
        append=True,
    )
    assert second["total_variants"] == 2
    assert (output_root / "kitchen" / "K1" / "robot_execution_5cam.mp4").exists()
    assert (output_root / "kitchen" / "V2" / "robot_execution_5cam.mp4").exists()

    with pytest.raises(FileExistsError):
        packager.package(
            recorded_root, unrecorded_root, timings_root, output_root,
            environments=("kitchen",),
            selected_variants={"kitchen": ["F0_ALL_VISIBLE"]},
            append=True,
        )

    replaced = packager.package(
        recorded_root, unrecorded_root, timings_root, output_root,
        environments=("kitchen",),
        selected_variants={"kitchen": ["F0_ALL_VISIBLE"]},
        append=True, replace_existing=True,
    )
    assert replaced["total_variants"] == 2
