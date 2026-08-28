"""Controlled L2 living-room scenes for functional support-region grounding."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_scenes.scene_loader import (
    FREE_CAMERA,
    GOOGLE_ACTUATORS,
    GOOGLE_HOME_QPOS,
    ROBOT_GOOGLE,
    ROBOT_NONE,
    _google_robot_dir,
    _inject_google_robot,
    _load_google_binary_assets,
    _load_object_binary_assets,
)
from mujoco_scenes.living_room_variants import (
    PREFIX as LIVING_ROOM_VARIANT_PREFIX,
    load_living_room_variants,
)


ROOT = Path(__file__).resolve().parent
L2_BASE = ROOT / "assets" / "living_room_region_base.xml"
L2_ABLATION1_SCENES = (
    "L2_living_room_region_ablation1_primary",
    "L2_living_room_region_ablation1_initial_complete",
    "L2_living_room_region_ablation1_exhaustion",
)
L2_ABLATION2_SCENES = (
    "L2_living_room_region_ablation2_primary",
    "L2_living_room_region_ablation2_drinks_dedicated",
    "L2_living_room_region_ablation2_controls_shared",
    "L2_living_room_region_ablation2_exhaustion",
    "L2_living_room_region_ablation2_permuted",
)
L2_ABLATION3_SCENES = (
    "L2_living_room_region_ablation3_primary",
    "L2_living_room_region_ablation3_matching_trap",
    "L2_living_room_region_ablation3_valid",
    "L2_living_room_region_ablation3_permuted",
)
L2_INTEGRATED_SCENES = tuple(
    LIVING_ROOM_VARIANT_PREFIX + variant_id
    for variant_id in load_living_room_variants()
)
L2_SCENES = (
    L2_ABLATION1_SCENES
    + L2_ABLATION2_SCENES
    + L2_ABLATION3_SCENES
    + L2_INTEGRATED_SCENES
)
L2_ABLATION2_BASE = (
    ROOT / "assets" / "living_room_region_ablation2_base.xml"
)
MOVIE_NIGHT_ASSETS = ROOT / "assets" / "movie_night"
REALISTIC_LIVING_ROOM_ASSETS = ROOT / "assets" / "living_room_realistic"
OBJECT_MESHES = ROOT / "assets" / "objects" / "meshes"
L2_CAMERAS = (
    "l2_camera_left",
    "l2_camera_right",
    "l2_camera_top",
    "l2_camera_front",
    "l2_camera_close",
)
L2_GOAL = (
    "Place the loaded refreshment tray on a stable serving surface within "
    "easy reach of both people."
)
L2_ABLATION2_GOAL = (
    "Prepare movie night for two people. Place one drink-and-snack set beside "
    "each person, and keep the TV remote and handheld game device together "
    "on one shared surface accessible to both."
)
L2_ABLATION3_GOAL = (
    "Place one drink-and-snack set on a suitable surface beside each "
    "person’s seating position."
)
L2_INTEGRATED_GOAL = (
    "Prepare the living room for two people watching television. Place one "
    "cup and one saucer on each person's fixed individual side table, and "
    "place the TV remote on the fixed shared coffee table."
)

# Canonical integrated-room construction coordinates.  They are used only to
# build the rendered simulator and opaque evidence gates; production grounding
# receives RGB-D, masks, calibration, and measured point clouds instead.
INTEGRATED_ROOM_LAYOUT = {
    "chair_left": (-0.98, 1.08),
    "chair_right": (0.98, 1.08),
    "side_table_left": (-1.72, 1.02),
    "side_table_right": (1.72, 1.02),
    "coffee_table": (0.0, 0.62),
    "media_console": (0.0, 2.43),
    "staging_table": (0.0, -1.72),
    "rug": (0.0, 0.58),
    "robot_spawn": (0.0, -2.48, 0.06205),
}

INTEGRATED_FURNITURE_DIMENSIONS = {
    "chair": (0.738310, 0.887906, 0.920567),
    "side_table": (0.550000, 0.450000, 0.550662),
    "accent_table": (0.300676, 0.300676, 0.417657),
    "coffee_table": (1.155008, 0.729635, 0.392428),
    "media_console": (1.341307, 0.264259, 0.515379),
    "staging_table": (1.375700, 0.491250, 0.726816),
}


def _remove_world_bodies(root: ET.Element, names: set[str]) -> None:
    worldbody = root.find("worldbody")
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") in names:
            worldbody.remove(child)


def _translate_body_geoms(
    root: ET.Element, body_name: str, target_xy: tuple[float, float]
) -> None:
    body = root.find(f".//body[@name='{body_name}']")
    geoms = list(body.iter("geom"))
    current = np.fromstring(geoms[0].get("pos", "0 0 0"), sep=" ")
    delta = np.asarray(target_xy, float) - current[:2]
    for geom in geoms:
        position = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        position[:2] += delta
        geom.set("pos", " ".join(f"{value:.5f}" for value in position))


def _apply_movie_night_visuals(root: ET.Element, ablation2: bool) -> None:
    """Replace benchmark collision primitives with textured visual meshes.

    Existing geoms remain as invisible, stable collision/measurement proxies.
    Semantic and geometric inference still receives rendered RGB-D and never
    reads this construction metadata.
    """
    asset = root.find("asset")
    declarations = (
        ("texture", {"name": "movie_fabric", "type": "2d", "file": str(MOVIE_NIGHT_ASSETS / "fabric_warm.png")}),
        ("texture", {"name": "movie_chair_fabric", "type": "2d", "file": str(MOVIE_NIGHT_ASSETS / "fabric_chair.png")}),
        ("texture", {"name": "movie_walnut", "type": "2d", "file": str(MOVIE_NIGHT_ASSETS / "wood_walnut.png")}),
        ("texture", {"name": "movie_oak", "type": "2d", "file": str(MOVIE_NIGHT_ASSETS / "wood_oak.png")}),
        ("texture", {"name": "movie_remote_charcoal", "type": "2d", "file": str(MOVIE_NIGHT_ASSETS / "remote_charcoal.png")}),
        ("material", {"name": "movie_fabric_mat", "texture": "movie_fabric", "roughness": "0.82"}),
        ("material", {"name": "movie_chair_mat", "texture": "movie_chair_fabric", "roughness": "0.78"}),
        ("material", {"name": "movie_walnut_mat", "texture": "movie_walnut", "roughness": "0.55"}),
        ("material", {"name": "movie_oak_mat", "texture": "movie_oak", "roughness": "0.58"}),
        ("material", {"name": "movie_remote_charcoal_mat", "texture": "movie_remote_charcoal", "roughness": "0.68"}),
        ("material", {"name": "movie_saucer_mat", "rgba": "0.92 0.94 0.96 1", "roughness": "0.38"}),
        ("material", {"name": "movie_rug_mat", "rgba": "0.36 0.34 0.30 1", "roughness": "0.94"}),
        ("material", {"name": "movie_lamp_mat", "rgba": "0.63 0.57 0.48 1", "roughness": "0.50"}),
        ("material", {"name": "movie_pot_mat", "rgba": "0.45 0.25 0.15 1", "roughness": "0.72"}),
        ("material", {"name": "movie_leaf_mat", "rgba": "0.18 0.34 0.20 1", "roughness": "0.82"}),
    )
    for tag, attributes in declarations:
        ET.SubElement(asset, tag, attributes)
    for name in (
        "sofa", "armchair", "coffee_table", "end_table", "c_table",
        "media_console", "bookshelf", "remote_control",
    ):
        ET.SubElement(
            asset, "mesh",
            {"name": f"movie_{name}", "file": str(MOVIE_NIGHT_ASSETS / f"{name}.obj")},
        )
    scans = {
        "movie_ycb_mug": ("ycb/mug/ycb_mug.obj", "1 1 1"),
        "movie_ycb_bowl": ("ycb/bowl/ycb_bowl.obj", "1 1 1"),
        "movie_gso_mug": (
            "gso/living_room_mug/gso_living_room_mug.obj", "0.78 0.78 0.78"
        ),
        "movie_gso_tray": (
            "gso/living_room_serving_tray/gso_living_room_serving_tray.obj",
            "1.35 1.35 1.35",
        ),
        "movie_gso_console": (
            "gso/living_room_game_console/gso_living_room_game_console.obj",
            "1.30 1.30 1.30",
        ),
        "movie_gso_bowl": (
            "gso/living_room_snack_bowl/gso_living_room_snack_bowl.obj",
            "1.15 1.15 1.15",
        ),
    }
    textures = {
        "movie_ycb_mug": "ycb/mug/ycb_mug.png",
        "movie_ycb_bowl": "ycb/bowl/ycb_bowl.png",
        "movie_gso_mug": "gso/living_room_mug/gso_living_room_mug.png",
        "movie_gso_tray": "gso/living_room_serving_tray/gso_living_room_serving_tray.png",
        "movie_gso_console": "gso/living_room_game_console/gso_living_room_game_console.png",
        "movie_gso_bowl": "gso/living_room_snack_bowl/gso_living_room_snack_bowl.png",
    }
    for name, (path, scale) in scans.items():
        ET.SubElement(
            asset, "mesh",
            {"name": name, "file": str(OBJECT_MESHES / path), "scale": scale},
        )
        ET.SubElement(
            asset, "texture",
            {
                "name": f"{name}_tex", "type": "2d",
                "file": str(OBJECT_MESHES / textures[name]),
            },
        )
        ET.SubElement(
            asset, "material",
            {"name": f"{name}_mat", "texture": f"{name}_tex", "roughness": "0.5"},
        )

    mapping = (
        {
            "a2_seat_left": ("movie_sofa", "movie_fabric_mat", "-0.72 1.05 0", "0.66 0.78 0.78"),
            "a2_seat_right": ("movie_armchair", "movie_chair_mat", "0.92 1.05 0", "0.92 0.92 0.92"),
            "a2_personal_left": ("movie_end_table", "movie_walnut_mat", "-1.48 0.60 0", "1 1 1"),
            "a2_personal_right": ("movie_end_table", "movie_oak_mat", "1.48 0.60 0", "0.94 1.05 1"),
            "a2_shared_drink_trap": ("movie_c_table", "movie_oak_mat", "0 0.72 0", "1.15 1.15 0.88"),
            "a2_control_table": ("movie_coffee_table", "movie_walnut_mat", "0 -0.05 0.02", "0.72 0.74 1"),
            "a2_media_wall": ("movie_media_console", "movie_walnut_mat", "0 1.48 0", "0.72 0.72 1"),
        }
        if ablation2
        else {
            "l2_sofa": ("movie_sofa", "movie_fabric_mat", "-0.62 0.82 0", "0.76 0.82 0.82"),
            "l2_side_table": ("movie_c_table", "movie_oak_mat", "-1.28 0.20 0", "0.72 0.86 0.88"),
            "l2_coffee_table": ("movie_coffee_table", "movie_walnut_mat", "0.42 0.20 0.04", "0.48 0.62 1"),
            "l2_media_wall": ("movie_media_console", "movie_walnut_mat", "0.85 1.35 0", "0.62 0.62 1"),
        }
    )
    for body_name, (mesh, material, position, scale) in mapping.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        for geom in body.findall("geom"):
            if "tv" not in geom.get("name", ""):
                geom.attrib.pop("material", None)
                geom.set("rgba", "0 0 0 0")
                geom.set("group", "3")
        visual_mesh_name = f"{body_name}_movie_visual_mesh"
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": visual_mesh_name,
                "file": str(
                    MOVIE_NIGHT_ASSETS
                    / f"{mesh.removeprefix('movie_')}.obj"
                ),
                "scale": scale,
            },
        )
        ET.SubElement(
            body, "geom",
            {
                "name": f"{body_name}_textured_visual",
                "type": "mesh", "mesh": visual_mesh_name, "material": material,
                "pos": position,
                "contype": "0", "conaffinity": "0", "mass": "0", "group": "2",
            },
        )

    # Shared apartment dressing.  These fixed visuals provide coherent room
    # context but are not candidate regions or payload instances.
    worldbody = root.find("worldbody")
    for rug_name in ("l2_rug_surface", "a2_rug_surface"):
        rug_geom = root.find(f".//geom[@name='{rug_name}']")
        if rug_geom is not None:
            rug_geom.set("material", "movie_rug_mat")
            rug_geom.attrib.pop("rgba", None)
    for border_name in ("l2_rug_border", "a2_rug_border"):
        border = root.find(f".//geom[@name='{border_name}']")
        if border is not None:
            border.set("rgba", "0.28 0.27 0.25 1")
    for staging_body_name in ("l2_staging_table", "a2_staging"):
        staging_body = root.find(f".//body[@name='{staging_body_name}']")
        if staging_body is not None:
            for geom in staging_body.findall("geom"):
                geom.set("material", "movie_walnut_mat")
                geom.attrib.pop("rgba", None)

    shelf = ET.SubElement(
        worldbody, "body", {"name": "movie_background_bookshelf"}
    )
    ET.SubElement(
        shelf, "geom",
        {
            "name": "movie_background_bookshelf_visual", "type": "mesh",
            "mesh": "movie_bookshelf", "material": "movie_walnut_mat",
            "pos": "-1.70 1.38 0",
            "contype": "0", "conaffinity": "0", "group": "2",
        },
    )
    lamp = ET.SubElement(
        worldbody, "body", {"name": "movie_background_floor_lamp"}
    )
    ET.SubElement(
        lamp, "geom",
        {
            "name": "movie_lamp_stem", "type": "cylinder",
            "pos": "1.85 1.18 0.82", "size": "0.018 0.80",
            "material": "movie_lamp_mat", "contype": "0",
            "conaffinity": "0", "group": "2",
        },
    )
    ET.SubElement(
        lamp, "geom",
        {
            "name": "movie_lamp_shade", "type": "cylinder",
            "pos": "1.85 1.18 1.62", "size": "0.18 0.22",
            "material": "movie_lamp_mat", "contype": "0",
            "conaffinity": "0", "group": "2",
        },
    )
    plant = ET.SubElement(
        worldbody, "body", {"name": "movie_background_plant"}
    )
    ET.SubElement(
        plant, "geom",
        {
            "name": "movie_plant_pot", "type": "cylinder",
            "pos": "-1.86 -0.62 0.16", "size": "0.16 0.16",
            "material": "movie_pot_mat", "contype": "0",
            "conaffinity": "0", "group": "2",
        },
    )
    for index, (x, y, z, size) in enumerate(
        (
            (-1.86, -0.62, 0.48, "0.08 0.08 0.30"),
            (-1.76, -0.62, 0.48, "0.07 0.07 0.26"),
            (-1.96, -0.61, 0.46, "0.07 0.07 0.24"),
        )
    ):
        ET.SubElement(
            plant, "geom",
            {
                "name": f"movie_plant_leaf_{index}", "type": "ellipsoid",
                "pos": f"{x} {y} {z}", "size": size,
                "material": "movie_leaf_mat", "contype": "0",
                "conaffinity": "0", "group": "2",
            },
        )
    if not ablation2:
        chair = ET.SubElement(
            worldbody, "body", {"name": "movie_background_armchair"}
        )
        ET.SubElement(
            chair, "geom",
            {
                "name": "movie_background_armchair_visual", "type": "mesh",
                "mesh": "movie_armchair", "material": "movie_chair_mat",
                "pos": "1.45 0.88 0",
                "contype": "0", "conaffinity": "0", "group": "2",
            },
        )

    if ablation2:
        replacements = {
            "a2_drink_left": ("movie_ycb_mug", "movie_ycb_mug_mat", "0 0 0"),
            "a2_drink_right": ("movie_gso_mug", "movie_gso_mug_mat", "0 0 0"),
            "a2_controller_payload": (
                "movie_gso_console", "movie_gso_console_mat", "0 0 0"
            ),
            "a2_remote_payload": (
                "movie_remote_control", "movie_remote_charcoal_mat", "0 0 0"
            ),
        }
        for name, position, mesh, material in (
            (
                "a2_snack_left", "-0.53 -1.34 0.655",
                "movie_ycb_bowl", "movie_ycb_bowl_mat",
            ),
            (
                "a2_snack_right", "-0.03 -1.34 0.655",
                "movie_gso_bowl", "movie_gso_bowl_mat",
            ),
        ):
            body = ET.SubElement(worldbody, "body", {"name": name, "pos": position})
            ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
            ET.SubElement(
                body, "geom",
                {
                    "name": f"{name}_visual", "type": "mesh", "mesh": mesh,
                    "material": material, "contype": "0", "conaffinity": "0",
                    "mass": "0", "group": "2",
                },
            )
            ET.SubElement(
                body, "geom",
                {
                    "name": f"{name}_collision", "type": "cylinder",
                    "size": "0.065 0.025", "rgba": "0 0 0 0",
                    "mass": "0.22", "group": "3",
                },
            )
    else:
        replacements = {
            "l2_refreshment_tray": (
                "movie_gso_tray", "movie_gso_tray_mat", "0 0 0"
            ),
        }
    for body_name, (mesh, material, position) in replacements.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        for geom in body.findall("geom"):
            geom.set("rgba", "0 0 0 0")
            geom.set("group", "3")
            geom.attrib.pop("material", None)
        ET.SubElement(
            body, "geom",
            {
                "name": f"{body_name}_scanned_visual", "type": "mesh",
                "mesh": mesh, "material": material, "pos": position,
                "contype": "0", "conaffinity": "0", "mass": "0", "group": "2",
            },
        )
        if body_name == "l2_refreshment_tray":
            # A controlled compound payload: contents are visual children of
            # the tray and therefore share its instance ID. FITS_ON continues
            # to use the measured tray footprint from rendered evidence.
            ET.SubElement(
                body, "geom",
                {
                    "name": "loaded_tray_mug_visual", "type": "mesh",
                    "mesh": "movie_ycb_mug", "material": "movie_ycb_mug_mat",
                    "pos": "-0.07 0 0.065", "contype": "0", "conaffinity": "0",
                    "mass": "0", "group": "2",
                },
            )
            ET.SubElement(
                body, "geom",
                {
                    "name": "loaded_tray_bowl_visual", "type": "mesh",
                    "mesh": "movie_ycb_bowl", "material": "movie_ycb_bowl_mat",
                    "pos": "0.07 0 0.055", "contype": "0", "conaffinity": "0",
                    "mass": "0", "group": "2",
                },
            )


_REALISTIC_ASSET_PARTS = {
    "modern_arm_chair_01": 2,
    "side_table_01": 1,
    "CoffeeTable_01": 1,
    "WoodenTable_02": 1,
    "chinese_console_table": 1,
}


def _set_box_geom(
    root: ET.Element,
    name: str,
    position: tuple[float, float, float],
    half_size: tuple[float, float, float],
) -> None:
    geom = root.find(f".//geom[@name='{name}']")
    if geom is None:
        raise RuntimeError(f"Integrated construction geom missing: {name}")
    geom.set("type", "box")
    geom.set("pos", " ".join(f"{value:.6f}" for value in position))
    geom.set("size", " ".join(f"{value:.6f}" for value in half_size))
    geom.set("rgba", "0 0 0 0")
    geom.set("group", "3")
    geom.attrib.pop("material", None)


def _camera_quaternion(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> str:
    position_array = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - position_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return " ".join(f"{value:.8f}" for value in quaternion)


def _add_realistic_visual_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    for asset_id, part_count in _REALISTIC_ASSET_PARTS.items():
        for index in range(1, part_count + 1):
            part_id = f"{asset_id}_part_{index:02d}"
            ET.SubElement(
                asset,
                "texture",
                {
                    "name": f"l2real_{part_id}_texture",
                    "type": "2d",
                    "file": str(REALISTIC_LIVING_ROOM_ASSETS / f"{part_id}.png"),
                },
            )
            ET.SubElement(
                asset,
                "material",
                {
                    "name": f"l2real_{part_id}_material",
                    "texture": f"l2real_{part_id}_texture",
                    "roughness": "0.62",
                    "specular": "0.25",
                },
            )


def _set_realistic_visual(
    root: ET.Element,
    body_name: str,
    asset_id: str,
    position: tuple[float, float, float],
    scale: tuple[float, float, float],
    *,
    quaternion: tuple[float, float, float, float] | None = None,
) -> None:
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise RuntimeError(f"Integrated visual body missing: {body_name}")
    for geom in list(body.findall("geom")):
        if "integrated_realistic_visual" in geom.get("name", ""):
            body.remove(geom)
    for index in range(1, _REALISTIC_ASSET_PARTS[asset_id] + 1):
        part_id = f"{asset_id}_part_{index:02d}"
        mesh_name = f"l2real_{body_name}_{part_id}_mesh"
        mesh = root.find(f".//mesh[@name='{mesh_name}']")
        mesh_attributes = {
            "name": mesh_name,
            "file": str(REALISTIC_LIVING_ROOM_ASSETS / f"{part_id}.obj"),
            "scale": " ".join(f"{value:.6f}" for value in scale),
        }
        if mesh is None:
            ET.SubElement(root.find("asset"), "mesh", mesh_attributes)
        else:
            mesh.attrib.update(mesh_attributes)
        attributes = {
            "name": (
                f"{body_name}_integrated_realistic_visual_part_{index:02d}"
            ),
            "type": "mesh",
            "mesh": mesh_name,
            "material": f"l2real_{part_id}_material",
            "pos": " ".join(f"{value:.6f}" for value in position),
            "contype": "0",
            "conaffinity": "0",
            "mass": "0",
            "group": "2",
        }
        if quaternion is not None:
            attributes["quat"] = " ".join(
                f"{value:.7f}" for value in quaternion
            )
        ET.SubElement(body, "geom", attributes)


def _configure_integrated_base_room(root: ET.Element) -> None:
    """Replace the legacy integrated scene with a sparse apartment layout."""
    _add_realistic_visual_assets(root)
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "l2real_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.72 0.78 0.86",
            "rgb2": "0.94 0.93 0.90",
            "width": "512",
            "height": "3072",
        },
    )
    worldbody = root.find("worldbody")
    # The redesigned benchmark has exactly three placement regions.  The old
    # alternate support and game controller are not part of the scene.
    _remove_world_bodies(root, {"a2_shared_drink_trap", "a2_controller_payload"})
    _remove_world_bodies(
        root,
        {
            "movie_background_bookshelf",
            "movie_background_floor_lamp",
            "movie_background_plant",
        },
    )
    # Remove the generated benchmark furniture visuals while retaining the
    # independently defined analytic collision/measurement proxies.
    for body_name in (
        "a2_seat_left",
        "a2_seat_right",
        "a2_personal_left",
        "a2_personal_right",
        "a2_control_table",
        "a2_media_wall",
    ):
        body = root.find(f".//body[@name='{body_name}']")
        for geom in list(body.findall("geom")):
            if geom.get("name", "").endswith("_textured_visual"):
                body.remove(geom)

    # Warm neutral shell, expanded to preserve open floor and robot clearance.
    floor = root.find(".//geom[@name='a2_floor']")
    floor.set("size", "3.20 2.70 0.05")
    floor.set("rgba", "0.64 0.57 0.48 1")
    floor.attrib.pop("material", None)
    _set_box_geom(root, "a2_north_wall", (0.0, 2.70, 1.45), (3.20, 0.04, 1.45))
    _set_box_geom(root, "a2_west_wall", (-3.20, -0.225, 1.45), (0.04, 2.925, 1.45))
    _set_box_geom(root, "a2_east_wall", (3.20, -0.225, 1.45), (0.04, 2.925, 1.45))
    south_wall = ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "a2_south_wall",
            "type": "box",
            "pos": "0 -3.15 1.45",
            "size": "3.20 0.04 1.45",
            "rgba": "0.88 0.86 0.81 1",
        },
    )
    for wall_name in ("a2_north_wall", "a2_west_wall", "a2_east_wall"):
        wall = root.find(f".//geom[@name='{wall_name}']")
        wall.set("rgba", "0.88 0.86 0.81 1")
        wall.set("group", "0")

    left = INTEGRATED_ROOM_LAYOUT["chair_left"]
    right = INTEGRATED_ROOM_LAYOUT["chair_right"]
    for side, (x, y) in (("left", left), ("right", right)):
        _set_box_geom(root, f"a2_seat_{side}_base", (x, y, 0.24), (0.37, 0.40, 0.24))
        _set_box_geom(root, f"a2_seat_{side}_cushion", (x, y + 0.02, 0.51), (0.32, 0.30, 0.08))
        _set_box_geom(root, f"a2_seat_{side}_back", (x, y - 0.34, 0.70), (0.36, 0.10, 0.36))
        _set_box_geom(root, f"a2_seat_{side}_arm_l", (x - 0.35, y, 0.55), (0.05, 0.34, 0.23))
        _set_box_geom(root, f"a2_seat_{side}_arm_r", (x + 0.35, y, 0.55), (0.05, 0.34, 0.23))
        _set_realistic_visual(
            root,
            f"a2_seat_{side}",
            "modern_arm_chair_01",
            (x, y, 0.0),
            (0.90, 0.90, 0.90),
            quaternion=(0.0, 0.0, 0.0, 1.0),
        )

    table_specs = (
        ("a2_personal_left", "a2_personal_left_top", "side_table_left"),
        ("a2_personal_right", "a2_personal_right_top", "side_table_right"),
    )
    for body_name, top_name, layout_name in table_specs:
        x, y = INTEGRATED_ROOM_LAYOUT[layout_name]
        _set_box_geom(root, top_name, (x, y, 0.535), (0.275, 0.225, 0.016))
        body = root.find(f".//body[@name='{body_name}']")
        for geom in body.findall("geom"):
            if geom.get("name") != top_name:
                geom.set("rgba", "0 0 0 0")
                geom.set("group", "3")
        _set_realistic_visual(
            root, body_name, "side_table_01", (x, y, 0.0), (1.0, 1.0, 1.0)
        )

    x, y = INTEGRATED_ROOM_LAYOUT["coffee_table"]
    _set_box_geom(root, "a2_control_table_top", (x, y, 0.377), (0.578, 0.365, 0.016))
    for geom in root.find(".//body[@name='a2_control_table']").findall("geom"):
        if geom.get("name") != "a2_control_table_top":
            geom.set("rgba", "0 0 0 0")
            geom.set("group", "3")
    _set_realistic_visual(
        root, "a2_control_table", "CoffeeTable_01", (x, y, 0.0), (0.75, 0.75, 0.75)
    )

    x, y = INTEGRATED_ROOM_LAYOUT["rug"]
    rug = root.find(".//geom[@name='a2_rug_surface']")
    rug.set("pos", f"{x:.6f} {y:.6f} 0.012")
    rug.set("size", "1.58 1.18 0.012")
    rug.set("rgba", "0.34 0.31 0.27 1")
    rug.attrib.pop("material", None)
    border = root.find(".//geom[@name='a2_rug_border']")
    border.set("pos", f"{x:.6f} {y:.6f} 0.008")
    border.set("size", "1.62 1.22 0.008")
    border.set("rgba", "0.25 0.23 0.21 1")

    x, y = INTEGRATED_ROOM_LAYOUT["media_console"]
    _set_box_geom(root, "a2_media_console", (x, y, 0.250), (0.671, 0.132, 0.250))
    media = root.find(".//geom[@name='a2_media_console']")
    media.set("contype", "1")
    media.set("conaffinity", "1")
    _set_realistic_visual(
        root, "a2_media_wall", "chinese_console_table", (x, y, 0.0), (0.78, 0.78, 0.78)
    )
    tv = root.find(".//geom[@name='a2_tv']")
    tv.set("pos", "0 2.625 1.42")
    tv.set("size", "0.70 0.025 0.405")
    tv.set("rgba", "0.018 0.022 0.030 1")
    tv.attrib.pop("material", None)

    x, y = INTEGRATED_ROOM_LAYOUT["staging_table"]
    _set_box_geom(root, "a2_staging_top", (x, y, 0.711), (0.688, 0.246, 0.016))
    staging = root.find(".//body[@name='a2_staging']")
    for geom in staging.findall("geom"):
        if geom.get("name") != "a2_staging_top":
            geom.set("rgba", "0 0 0 0")
            geom.set("group", "3")
    _set_realistic_visual(
        root, "a2_staging", "chinese_console_table", (x, y, 0.0), (0.80, 1.45, 1.10)
    )

    # Five payloads: two cup/saucer pairs and one shared TV remote.
    payload_positions = {
        "a2_drink_left": (-0.50, -1.80, 0.810),
        "a2_snack_left": (-0.23, -1.80, 0.737),
        "a2_drink_right": (0.18, -1.80, 0.810),
        "a2_snack_right": (0.45, -1.80, 0.737),
        "a2_remote_payload": (0.0, -1.58, 0.755),
    }
    for body_name, position in payload_positions.items():
        body = root.find(f".//body[@name='{body_name}']")
        body.set("pos", " ".join(f"{value:.6f}" for value in position))
    # The compact GSO mug is visually bowl-like from the required overhead
    # views. Use a second instance of the stronger YCB mug scan; semantics
    # still comes solely from rendered RGB.
    second_mug_visual = root.find(
        ".//geom[@name='a2_drink_right_scanned_visual']"
    )
    second_mug_visual.set("mesh", "movie_ycb_mug")
    second_mug_visual.set("material", "movie_ycb_mug_mat")
    # Replace the old bowl visuals with shallow saucers while retaining the
    # stable free bodies and collision-backed manipulation interfaces.
    for body_name in ("a2_snack_left", "a2_snack_right"):
        body = root.find(f".//body[@name='{body_name}']")
        for geom in list(body.findall("geom")):
            if geom.get("contype", "1") == "0":
                body.remove(geom)
            else:
                geom.set("type", "cylinder")
                geom.set("size", "0.095 0.008")
                geom.set("rgba", "0 0 0 0")
                geom.set("group", "3")
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                geom.set("condim", "4")
                geom.set("friction", "0.9 0.02 0.002")
                geom.set("solref", "0.002 1")
                geom.set("solimp", "0.95 0.99 0.001")
                geom.attrib.pop("mesh", None)
                geom.attrib.pop("material", None)
        ET.SubElement(body, "geom", {
            "name": f"{body_name}_saucer_visual",
            "type": "cylinder", "size": "0.095 0.010",
            "material": "movie_saucer_mat", "contype": "0",
            "conaffinity": "0", "mass": "0", "group": "2",
        })

    camera_specs = {
        "l2_camera_left": ((-3.05, -2.72, 2.25), (0.0, 0.25, 0.65), 74.0),
        "l2_camera_right": ((3.05, -2.72, 2.25), (0.0, 0.25, 0.65), 74.0),
        "l2_camera_top": ((0.0, -0.25, 4.65), (0.0, 0.20, 0.30), 72.0),
        "l2_camera_front": ((0.0, -2.75, 1.45), (0.0, -1.66, 0.72), 56.0),
        "l2_camera_close": ((-1.70, 2.50, 1.75), (-0.80, 0.82, 0.50), 65.0),
    }
    for camera_name, (position, target, fovy) in camera_specs.items():
        camera = root.find(f".//camera[@name='{camera_name}']")
        camera.set("pos", " ".join(map(str, position)))
        camera.set("quat", _camera_quaternion(position, target))
        camera.set("fovy", str(fovy))
        camera.attrib.pop("xyaxes", None)

    ET.SubElement(
        worldbody,
        "site",
        {
            "name": "l2_canonical_robot_spawn",
            "pos": "0 -2.48 0.02",
            "size": "0.02",
            "rgba": "0 0 0 0",
            "group": "4",
        },
    )


def _configure_ablation3_scene(root: ET.Element, scene_name: str) -> None:
    """Create target-coverage layouts using construction metadata only."""
    _remove_world_bodies(
        root,
        {
            "a2_control_table",
            "a2_rug",
            "a2_media_wall",
            "a2_remote_payload",
            "a2_controller_payload",
        },
    )
    if scene_name.endswith("_primary"):
        _remove_world_bodies(root, {"a2_shared_drink_trap"})
        _translate_body_geoms(
            root, "a2_personal_right", (-0.55, 0.22)
        )
    elif scene_name.endswith("_matching_trap"):
        _remove_world_bodies(root, {"a2_shared_drink_trap"})
        _translate_body_geoms(root, "a2_personal_right", (0.0, 0.72))
    elif scene_name.endswith("_valid"):
        _remove_world_bodies(root, {"a2_shared_drink_trap"})
    elif scene_name.endswith("_permuted"):
        _remove_world_bodies(root, {"a2_shared_drink_trap"})
        _translate_body_geoms(root, "a2_personal_left", (0.0, 0.72))
        first_drink = root.find(".//body[@name='a2_drink_left']")
        second_drink = root.find(".//body[@name='a2_drink_right']")
        first_position = first_drink.get("pos")
        first_drink.set("pos", second_drink.get("pos"))
        second_drink.set("pos", first_position)
        worldbody = root.find("worldbody")
        movable = [
            child
            for child in list(worldbody)
            if child.tag == "body" and child.find("freejoint") is not None
        ]
        for child in movable:
            worldbody.remove(child)
        for child in reversed(movable):
            worldbody.append(child)


def _set_support_top_size(
    root: ET.Element,
    geom_name: str,
    half_length: float,
    half_width: float,
) -> None:
    """Resize a construction-time support proxy and matching visual.

    Runtime region measurement never calls this helper or reads these sizes;
    it sees only the rendered RGB-D surface selected by an opaque crop.
    """
    top = root.find(f".//geom[@name='{geom_name}']")
    if top is None:
        raise RuntimeError(f"Integrated support geom missing: {geom_name}")
    old = np.fromstring(top.get("size", ""), sep=" ")
    top.set("size", f"{half_length:.5f} {half_width:.5f} {old[2]:.5f}")
    body = next(
        body for body in root.iter("body") if top in list(body.findall("geom"))
    )
    visual = next(
        (
            geom
            for geom in body.findall("geom")
            if geom.get("name", "").endswith("_textured_visual")
        ),
        None,
    )
    if visual is None:
        return
    mesh = root.find(f".//mesh[@name='{visual.get('mesh')}']")
    if mesh is None:
        return
    current = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    reference = {
        "a2_personal_left_top": (0.25, 0.21),
        "a2_personal_right_top": (0.25, 0.21),
        "a2_shared_drink_top": (0.37, 0.22),
        "a2_control_table_top": (0.50, 0.34),
    }[geom_name]
    current[0] *= half_length / reference[0]
    current[1] *= half_width / reference[1]
    mesh.set("scale", " ".join(f"{value:.5f}" for value in current))


def _integrated_scene_code(scene_name: str) -> str:
    return scene_name.rsplit("_", 2)[-2] + "_" + scene_name.rsplit("_", 1)[-1]


def _configure_integrated_scene(root: ET.Element, scene_name: str) -> None:
    """Apply only object-location or table-presence changes."""
    code = scene_name.removeprefix(LIVING_ROOM_VARIANT_PREFIX)
    spec = load_living_room_variants()[code]
    # The fixed benchmark has exactly three inspectable destination regions.
    # Remove the decorative rug as well: because the shared coffee table sits
    # above it, large full-frame rug detections can otherwise overlap the
    # table's segmentation mask and incorrectly win semantic association.
    _remove_world_bodies(root, {"a2_rug"})
    region_bodies = {
        "PERSONAL_TABLE_LEFT": "a2_personal_left",
        "PERSONAL_TABLE_RIGHT": "a2_personal_right",
        "SHARED_TABLE": "a2_control_table",
    }
    for region_id in spec.get("absent_regions", []):
        _remove_world_bodies(root, {region_bodies[region_id]})

    placements = {
        "PERSONAL_TABLE_LEFT": {
            "a2_drink_left": (-1.80, 1.02, 0.622),
            "a2_snack_left": (-1.72, 0.90, 0.561),
        },
        "PERSONAL_TABLE_RIGHT": {
            "a2_drink_right": (1.62, 1.02, 0.622),
            "a2_snack_right": (1.80, 1.02, 0.561),
        },
        "SHARED_TABLE": {
            "a2_drink_left": (-0.16, 0.62, 0.464),
            "a2_snack_left": (0.20, 0.40, 0.403),
        },
    }
    for body_name, region_id in spec.get("object_locations", {}).items():
        body = root.find(f".//body[@name='{body_name}']")
        body.set("pos", " ".join(map(str, placements[region_id][body_name])))


def _add_integrated_execution_annotations(root: ET.Element) -> None:
    """Add invisible grasp frames and inactive welds for Phase-3 execution.

    These annotations have no geom, material, semantic label, or rendered
    pixels.  They are deliberately added only to the robot-bearing execution
    model, after the frozen perception scene has been configured.
    """
    grasp_frames = {
        "a2_drink_left": ((0.0, 0.0, 0.035), "0.006"),
        "a2_drink_right": ((0.0, 0.0, 0.035), "0.006"),
        "a2_snack_left": ((0.0, 0.0, 0.010), "0.006"),
        "a2_snack_right": ((0.0, 0.0, 0.010), "0.006"),
        "a2_remote_payload": ((0.0, 0.0, 0.020), "0.005"),
    }
    for body_name, (position, size) in grasp_frames.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise RuntimeError(f"Integrated execution body missing: {body_name}")
        ET.SubElement(
            body,
            "site",
            {
                "name": f"phase3_grasp_{body_name}",
                "pos": " ".join(f"{value:.6f}" for value in position),
                "size": size,
                "rgba": "0 0 0 0",
                "group": "3",
            },
        )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for body_name in grasp_frames:
        ET.SubElement(
            equality,
            "weld",
            {
                "name": f"google:pick_weld_{body_name}",
                "body1": "google:link_gripper",
                "body2": body_name,
                "active": "false",
                "solref": "0.003 1",
                "solimp": "0.95 0.99 0.001",
            },
        )


def build_l2_region_xml(
    scene_name: str,
    robot: str = ROBOT_GOOGLE,
) -> str:
    """Compose one controlled L2 variant with Google Robot or no robot."""
    if scene_name not in L2_SCENES:
        raise ValueError(f"Unknown L2 region scene: {scene_name}")
    if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
        raise ValueError("L2 region scenes support robot google or none")
    integrated = scene_name in L2_INTEGRATED_SCENES
    ablation2 = scene_name in L2_ABLATION2_SCENES or integrated
    ablation3 = scene_name in L2_ABLATION3_SCENES
    root = ET.parse(
        L2_ABLATION2_BASE if ablation2 or ablation3 else L2_BASE
    ).getroot()
    _apply_movie_night_visuals(root, ablation2 or ablation3)
    if integrated:
        _configure_integrated_base_room(root)
    if scene_name in L2_ABLATION1_SCENES:
        # Keep the narrow C-table visually separate from the sofa so RGB
        # detector boxes can be associated one-to-one with its instance mask.
        _translate_body_geoms(root, "l2_side_table", (-1.65, 0.02))
    if scene_name in L2_ABLATION1_SCENES and scene_name.endswith("_exhaustion"):
        # Retain recognizable coffee-table context while making its observed
        # support patch robustly too small for the tray. Runtime inference
        # never reads this construction-time value.
        top = next(
            geom
            for geom in root.iter("geom")
            if geom.get("name") == "l2_coffee_table_top"
        )
        top.set("size", "0.155 0.100 0.04")
        # Separate the deliberately undersized table from the sofa arm.  At
        # high RGB-D resolution a same-height strip of the adjacent vertical
        # arm face can otherwise join the stage-local support evidence and
        # inflate the PCA footprint.  This is a scene-design separation only;
        # runtime measurement still receives no intended size or pose.
        exhaustion_center_x = 0.62
        top_position = np.fromstring(top.get("pos", ""), sep=" ")
        top_position[0] = exhaustion_center_x
        top.set(
            "pos", " ".join(f"{value:.5f}" for value in top_position)
        )
        for geom in root.iter("geom"):
            name = geom.get("name", "")
            if name.startswith("l2_coffee_leg_"):
                position = np.fromstring(geom.get("pos", ""), sep=" ")
                position[0] = (
                    exhaustion_center_x
                    + np.sign(position[0] - 0.42) * 0.10
                )
                position[1] = 0.20 + np.sign(position[1] - 0.20) * 0.06
                geom.set("pos", " ".join(f"{value:.5f}" for value in position))
    if ablation2 and scene_name.endswith("_exhaustion"):
        # Keep all candidates visible and plausible, but make the only
        # control-semantic surface too narrow for simultaneous two-object
        # packing. Runtime geometry still measures this from RGB-D.
        top = next(
            geom
            for geom in root.iter("geom")
            if geom.get("name") == "a2_control_table_top"
        )
        top.set("size", "0.16 0.065 0.040")
    if ablation2 and scene_name.endswith("_permuted"):
        # Change the visible layout and free-instance creation order without
        # making the detector solve a different appearance/occlusion problem.
        # The two equivalent drink payloads exchange positions; the controls
        # retain the primary scene's reliably recognized viewpoints. Runtime
        # association must still recover fresh generic IDs from image evidence.
        first_drink = root.find(".//body[@name='a2_drink_left']")
        second_drink = root.find(".//body[@name='a2_drink_right']")
        first_position = first_drink.get("pos")
        first_drink.set("pos", second_drink.get("pos"))
        second_drink.set("pos", first_position)
        worldbody = root.find("worldbody")
        payload_bodies = [
            child
            for child in list(worldbody)
            if child.tag == "body"
            and child.get("name", "").startswith("a2_")
        ]
        for child in payload_bodies:
            worldbody.remove(child)
        for child in reversed(payload_bodies):
            worldbody.append(child)
    if ablation3:
        _configure_ablation3_scene(root, scene_name)
    if integrated:
        _configure_integrated_scene(root, scene_name)
    if robot == ROBOT_GOOGLE:
        _inject_google_robot(root, _google_robot_dir())
        if integrated:
            robot_body = root.find(".//body[@name='google:base_link']")
            x, y, z = INTEGRATED_ROOM_LAYOUT["robot_spawn"]
            robot_body.set("pos", f"{x:.6f} {y:.6f} {z:.6f}")
            robot_body.set("quat", "0.7071068 0 0 0.7071068")
            _add_integrated_execution_annotations(root)
    return ET.tostring(root, encoding="unicode")


class L2LivingRoomRegionScene:
    """Compiled L2 room plus only the interfaces needed by observation."""

    goal = L2_GOAL
    point_cloud_cameras = L2_CAMERAS
    payload_instance_name = "l2_refreshment_tray"

    def __init__(
        self,
        scene_name: str = L2_SCENES[0],
        robot: str = ROBOT_GOOGLE,
    ):
        if scene_name not in L2_SCENES:
            raise ValueError(f"Unknown L2 region scene: {scene_name}")
        if robot not in {ROBOT_GOOGLE, ROBOT_NONE}:
            raise ValueError("L2 region scenes support google or none")
        self.scene_name = scene_name
        self.goal = (
            L2_INTEGRATED_GOAL
            if scene_name in L2_INTEGRATED_SCENES
            else L2_ABLATION3_GOAL
            if scene_name in L2_ABLATION3_SCENES
            else (
                L2_ABLATION2_GOAL
                if scene_name in L2_ABLATION2_SCENES
                else L2_GOAL
            )
        )
        self.robot_name = robot
        self.has_robot = robot == ROBOT_GOOGLE
        print(f"[L2RegionScene] Building scene: {scene_name}")
        print(f"  Goal: {self.goal}")
        assets = _load_object_binary_assets()
        assets.update(
            {
                f"movie_night/{path.name}": path.read_bytes()
                for path in MOVIE_NIGHT_ASSETS.iterdir()
                if path.suffix.lower() in {".obj", ".png"}
            }
        )
        assets.update(
            {
                f"living_room_realistic/{path.name}": path.read_bytes()
                for path in REALISTIC_LIVING_ROOM_ASSETS.iterdir()
                if path.suffix.lower() in {".obj", ".png"}
            }
        )
        assets.update(
            _load_google_binary_assets(_google_robot_dir())
            if self.has_robot
            else {}
        )
        self.model = mujoco.MjModel.from_xml_string(
            build_l2_region_xml(scene_name, robot),
            assets=assets,
        )
        self.data = mujoco.MjData(self.model)
        self._set_robot_home_pose()
        if scene_name in (
            L2_ABLATION2_SCENES
            + L2_ABLATION3_SCENES
            + L2_INTEGRATED_SCENES
        ):
            # Scanned mug/bowl visuals retain compact analytic collision
            # proxies. Cylindrical free payloads can otherwise enter a nearly
            # lossless edge-roll on the staging table. MuJoCo's freejoint XML
            # shorthand has no damping attribute, so apply ordinary payload/
            # table dissipation directly to its six compiled DOFs before the
            # deterministic construction settle. Object poses remain free.
            for joint_id in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                if not (joint_name or "").startswith("a2_"):
                    continue
                if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                dof_address = self.model.jnt_dofadr[joint_id]
                # Translational and angular damping have different units.
                # Applying 2.0 to the tiny rotational inertias of scanned
                # mugs made the explicit dynamics unstable on first support
                # contact. A bounded 0.30 angular value removes residual
                # free-payload edge-roll while remaining far below that
                # unstable setting. Strict execution still reports any final
                # yaw drift; the optional GT assistance profile can accept it
                # only when every structural ON-relation check passes.
                angular_damping = 0.30
                self.model.dof_damping[dof_address:dof_address + 6] = (
                    0.25, 0.25, 0.25,
                    angular_damping, angular_damping, angular_damping,
                )
        mujoco.mj_forward(self.model, self.data)
        for _ in range(600):
            mujoco.mj_step(self.model, self.data)
        print(f"  Robot: {robot}")
        integrated_support_count = sum(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
            for name in (
                "a2_personal_left_top", "a2_personal_right_top",
                "a2_control_table_top",
            )
        )
        print(
            "  Candidate supports: "
            f"{integrated_support_count if scene_name in L2_INTEGRATED_SCENES else 2 if scene_name in L2_ABLATION3_SCENES else 5 if scene_name in L2_ABLATION2_SCENES else 3}"
        )
        print("  Scene ready.\n")

    def _set_robot_home_pose(self) -> None:
        if not self.has_robot:
            return
        for joint_name, value in GOOGLE_HOME_QPOS.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise RuntimeError(f"Google Robot joint missing: {joint_name}")
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        for actuator_name, joint_name, _kp, _lower, _upper in GOOGLE_ACTUATORS:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise RuntimeError(
                    f"Google Robot actuator missing: {actuator_name}"
                )
            self.data.ctrl[actuator_id] = GOOGLE_HOME_QPOS[joint_name]

    def get_visible_object_instances(self) -> list[tuple[str, str]]:
        """Expose only the fixed payload as an object-level observation."""
        if self.scene_name in (
            L2_ABLATION2_SCENES
            + L2_ABLATION3_SCENES
            + L2_INTEGRATED_SCENES
        ):
            # Ablation 2 discovers its four generic payload IDs from visible
            # segmentation instances and RGB semantics in one initial capture.
            # Do not leak simulator body names through the legacy object API.
            return []
        return [(self.payload_instance_name, "refreshment_tray")]

    def render_frame(
        self,
        camera: str = "l2_camera_front",
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        if camera not in L2_CAMERAS:
            raise ValueError(f"Unknown L2 camera: {camera}")
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            mujoco.mj_forward(self.model, self.data)
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()

    def print_scene_summary(self) -> None:
        print("=" * 60)
        print(f"Scene: {self.scene_name}")
        print(f"Goal:  {self.goal}")
        print("-" * 60)
        if self.scene_name in (
            L2_ABLATION2_SCENES
            + L2_ABLATION3_SCENES
            + L2_INTEGRATED_SCENES
        ):
            ablation3 = self.scene_name in L2_ABLATION3_SCENES
            integrated = self.scene_name in L2_INTEGRATED_SCENES
            if integrated:
                region_count = sum(
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
                    for name in (
                        "a2_personal_left_top", "a2_personal_right_top",
                        "a2_control_table_top",
                    )
                )
                payload_count = 5
            else:
                region_count = 2 if ablation3 else 5
                payload_count = 2 if ablation3 else 6
            print(
                f"Candidate regions:  {region_count}, "
                "all visible initially"
            )
            print(
                f"Fixed payloads:      {payload_count}, "
                "discovered from RGB-D evidence"
            )
            print("Seating targets:     2, spatially distinct")
        else:
            print(
                "Candidate regions:  "
                "SOFA_SEAT_PATCH, SMALL_SIDE_TABLE, COFFEE_TABLE"
            )
            print("Fixed payload:       one observed refreshment tray")
        print(f"Robot:               {self.robot_name}")
        print("=" * 60)

    def launch_viewer(self, camera: str = FREE_CAMERA) -> None:
        if camera != FREE_CAMERA and camera not in L2_CAMERAS:
            raise ValueError(f"Unknown L2 camera: {camera}")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            if camera == FREE_CAMERA:
                mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
                )
            while viewer.is_running():
                started = time.time()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                remaining = self.model.opt.timestep - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)
