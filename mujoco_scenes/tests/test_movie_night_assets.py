from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from mujoco_scenes.living_room_region_scene import (
    L2LivingRoomRegionScene,
    L2_SCENES,
    MOVIE_NIGHT_ASSETS,
    ROOT,
    build_l2_region_xml,
)


EXPECTED_GSO = {
    "Cole_Hardware_Mug_Classic_Blue",
    "Room_Essentials_Bowl_Turquiose",
    "Threshold_Tray_Rectangle_Porcelain",
    "BlackBlack_Nintendo_3DSXL",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_movie_night_gso_assets_have_complete_verified_manifest_records():
    manifest = json.loads(
        (ROOT / "assets/objects/meshes/manifest.json").read_text()
    )
    records = {
        record["dataset_id"]: record
        for record in manifest["assets"]
        if record["dataset"] == "Google Scanned Objects"
    }
    assert EXPECTED_GSO <= records.keys()
    for dataset_id in EXPECTED_GSO:
        record = records[dataset_id]
        assert record["source_url"].startswith("https://")
        assert record["prepared_from"].startswith("https://")
        assert record["collision_representation"] == (
            "analytic_proxy_defined_in_scene"
        )
        assert np.all(np.isfinite(record["scale"]))
        assert np.all(np.asarray(record["scale"]) > 0)
        for field, digest_field in (
            ("model", "model_sha256"),
            ("texture", "texture_sha256"),
        ):
            path = ROOT / record[field]
            assert path.is_file()
            assert not Path(record[field]).is_absolute()
            assert _sha256(path) == record[digest_field]


def test_project_authored_movie_night_visuals_are_generated_and_textured():
    expected = {
        "sofa.obj",
        "armchair.obj",
        "coffee_table.obj",
        "end_table.obj",
        "c_table.obj",
        "media_console.obj",
        "bookshelf.obj",
        "remote_control.obj",
        "fabric_warm.png",
        "wood_walnut.png",
        "remote_charcoal.png",
    }
    assert expected <= {path.name for path in MOVIE_NIGHT_ASSETS.iterdir()}


def test_all_movie_night_variants_compile_without_robot():
    for scene_name in L2_SCENES:
        model = mujoco.MjModel.from_xml_string(
            build_l2_region_xml(scene_name, robot="none")
        )
        assert model.ncam == 5
        assert model.ngeom > 30


def test_movie_night_free_payloads_settle_finitely():
    for scene_name in (
        "L2_living_room_region_ablation1_primary",
        "L2_living_room_region_ablation2_primary",
        "L2_living_room_region_ablation3_valid",
    ):
        scene = L2LivingRoomRegionScene(scene_name, robot="none")
        assert np.all(np.isfinite(scene.data.qpos))
        assert np.all(np.isfinite(scene.data.qvel))
        free_dofs = [
            int(scene.model.jnt_dofadr[joint_id])
            for joint_id in range(scene.model.njnt)
            if scene.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        ]
        assert max(
            float(np.max(np.abs(scene.data.qvel[dof:dof + 3])))
            for dof in free_dofs
        ) < 0.08
        assert max(
            float(np.max(np.abs(scene.data.qvel[dof + 3:dof + 6])))
            for dof in free_dofs
        ) < 0.25
