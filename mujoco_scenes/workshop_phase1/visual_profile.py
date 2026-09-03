"""Optional runtime-only visual profiles for Workshop perception experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml


def _rgba(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain four finite RGBA values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


def _rgb(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite RGB values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


def apply_workshop_visual_profile(scene: Any, profile_path: Path | str) -> dict[str, Any]:
    """Apply a named appearance profile to an already compiled MuJoCo scene.

    This changes rendering properties only. Geometry, collision, poses, variant
    contents, camera calibration, and privileged evaluation state are untouched.
    """
    path = Path(profile_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Visual profile must be a YAML mapping: {path}")

    model = scene.model
    variant_config = config.get("variant_overrides", {}).get(
        getattr(scene, "variant_name", ""), {})
    lighting = config.get("lighting", {})
    headlight = lighting.get("headlight", {})
    for field in ("ambient", "diffuse", "specular"):
        if field in headlight:
            getattr(model.vis.headlight, field)[:] = _rgb(
                headlight[field], name=f"lighting.headlight.{field}")

    disable_shadows = bool(lighting.get("disable_cast_shadows", False))
    light_overrides = lighting.get("lights", {})
    for light_name, values in light_overrides.items():
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
        if light_id < 0:
            raise ValueError(f"Unknown Workshop light in visual profile: {light_name}")
        for field in ("ambient", "diffuse", "specular"):
            if field in values:
                getattr(model, f"light_{field}")[light_id] = _rgb(
                    values[field], name=f"lighting.lights.{light_name}.{field}")
        if "cast_shadow" in values:
            model.light_castshadow[light_id] = int(bool(values["cast_shadow"]))
    if disable_shadows:
        model.light_castshadow[:] = 0

    material_overrides = {
        **config.get("materials", {}),
        **variant_config.get("materials", {}),
    }
    for material_name, values in material_overrides.items():
        material_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, material_name)
        if material_id < 0:
            raise ValueError(
                f"Unknown Workshop material in visual profile: {material_name}")
        if "rgba" in values:
            model.mat_rgba[material_id] = _rgba(
                values["rgba"], name=f"materials.{material_name}.rgba")
        if "specular" in values:
            model.mat_specular[material_id] = float(values["specular"])
        if "shininess" in values:
            model.mat_shininess[material_id] = float(values["shininess"])
        if "reflectance" in values:
            model.mat_reflectance[material_id] = float(values["reflectance"])

    def apply_geom_values(geom_id: int, geom_name: str, values: dict[str, Any]) -> None:
        if "material" in values:
            material_name = str(values["material"])
            material_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, material_name)
            if material_id < 0:
                raise ValueError(
                    f"Unknown Workshop material for geom {geom_name}: {material_name}")
            model.geom_matid[geom_id] = material_id
        if "group" in values:
            group = int(values["group"])
            if group < 0 or group > 5:
                raise ValueError(f"geoms.{geom_name}.group must be in [0, 5]")
            model.geom_group[geom_id] = group
        if "rgba" in values:
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = _rgba(
                values["rgba"], name=f"geoms.{geom_name}.rgba")

    # Per-geom overrides intentionally detach only explicitly named geometry
    # from its material. The L profile does not use this for detector targets,
    # so their original learned textures remain intact.
    geom_overrides = {
        **config.get("geoms", {}),
        **variant_config.get("geoms", {}),
    }
    for geom_name, values in geom_overrides.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            if bool(values.get("optional", False)):
                continue
            raise ValueError(f"Unknown Workshop geom in visual profile: {geom_name}")
        apply_geom_values(geom_id, geom_name, values)

    prefix_overrides = {
        **config.get("geom_prefixes", {}),
        **variant_config.get("geom_prefixes", {}),
    }
    for prefix, values in prefix_overrides.items():
        matched = 0
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and geom_name.startswith(prefix):
                apply_geom_values(geom_id, geom_name, values)
                matched += 1
        if matched == 0:
            if bool(values.get("optional", False)):
                continue
            raise ValueError(f"No Workshop geoms match visual profile prefix: {prefix}")

    mujoco.mj_forward(model, scene.data)
    return config
