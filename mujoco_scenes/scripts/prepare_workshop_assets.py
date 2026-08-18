#!/usr/bin/env python3
"""Prepare CC0 and procedural 3D assets used by the integrated Workshop (W1) benchmark.

Source glTF bundles are downloaded into an ignored cache directory (.cache).
Processed OBJ visual parts and PNG textures are exported into
assets/workshop_realistic/. The script normalizes Poly Haven's Y-up convention
to MuJoCo Z-up, centers meshes, computes physical bounding dimensions, and
writes complete hash and provenance metadata into manifest.json.

The script is idempotent. Pass ``--force`` to reprocess existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import trimesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets" / "workshop_realistic"
DEFAULT_CACHE = ROOT / "assets" / ".cache" / "workshop_realistic"
API = "https://api.polyhaven.com"
LICENSE_URL = "https://polyhaven.com/license"
USER_AGENT = "icra-we-ball-workshop-assets/1.0"


ASSET_DEFS: dict[str, dict[str, Any]] = {
    "screwdrivers_02": {
        "human_readable_name": "Screwdrivers and Fasteners Set",
        "author": "Poly Haven / Martin Klekner",
        "roles": [
            "workshop_long_phillips_driver",
            "workshop_stubby_phillips_driver",
            "workshop_medium_phillips_screw",
            "workshop_short_phillips_screw",
            "workshop_long_phillips_screw",
        ],
    },
    "screwdriver": {
        "human_readable_name": "Flathead Screwdriver",
        "author": "Poly Haven / Martin Klekner",
        "roles": ["workshop_flathead_screwdriver"],
    },
    "Drill_01": {
        "human_readable_name": "Cordless Power Drill Driver",
        "author": "Poly Haven / Mike van der Valk",
        "roles": ["workshop_power_driver"],
    },
    "pliers": {
        "human_readable_name": "Combination Pliers",
        "author": "Poly Haven / Martin Klekner",
        "roles": ["workshop_pliers"],
    },
    "combination_wrench": {
        "human_readable_name": "Combination Wrench",
        "author": "Poly Haven / Martin Klekner",
        "roles": ["workshop_combination_wrench"],
    },
    "ratchet_wrench": {
        "human_readable_name": "Ratchet Wrench",
        "author": "Poly Haven / Martin Klekner",
        "roles": ["workshop_ratchet_wrench"],
    },
    "wooden_hammer_01": {
        "human_readable_name": "Wooden Mallet Hammer",
        "author": "Poly Haven / Martin Klekner",
        "roles": ["workshop_wooden_mallet"],
    },
    "tool_cart": {
        "human_readable_name": "Industrial Rolling Tool Cart",
        "author": "Poly Haven / Mike van der Valk",
        "roles": ["workshop_tool_cart"],
    },
    "plastic_container": {
        "human_readable_name": "Plastic Hardware Storage Bin",
        "author": "Poly Haven / Mike van der Valk",
        "roles": ["workshop_hardware_bin"],
    },
    "metal_toolbox": {
        "human_readable_name": "Metal Tool Box Compartment",
        "author": "Poly Haven / Martin Klekner",
        "roles": ["workshop_metal_toolbox"],
    },
    "wooden_table_02": {
        "human_readable_name": "Workshop Wooden Workbench Table",
        "author": "Poly Haven / Martin Klekner",
        "roles": ["workbench"],
    },
}


def _download_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, path: Path, expected_size: int | None = None) -> None:
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if expected_size is not None and len(data) != expected_size:
        raise RuntimeError(
            f"Unexpected download size for {url}: {len(data)} != {expected_size}"
        )
    path.write_bytes(data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_gltf_bundle(asset_id: str, cache_dir: Path) -> Path:
    info = _download_json(f"{API}/files/{asset_id}")
    gltf_spec = info["gltf"]["1k"]["gltf"]
    target_dir = cache_dir / asset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    gltf_file = target_dir / f"{asset_id}_1k.gltf"
    _download_file(gltf_spec["url"], gltf_file, gltf_spec.get("size"))

    for relative_path, include_spec in gltf_spec.get("include", {}).items():
        include_path = target_dir / relative_path
        include_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(
            include_spec["url"], include_path, include_spec.get("size")
        )
    return gltf_file


def _convert_texture(src_jpg: Path, dst_png: Path) -> None:
    img = Image.open(src_jpg).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    img.save(dst_png, format="PNG", optimize=True)


def _to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate mesh from glTF Y-up to MuJoCo Z-up: (x, y, z) -> (x, -z, y)."""
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    transformed = mesh.copy()
    transformed.apply_transform(rot)
    return transformed


def _center_and_ground(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center XY at origin and ground Z min at 0."""
    bounds = mesh.bounds
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    min_z = bounds[0][2]
    transformed = mesh.copy()
    transformed.apply_translation([-center_xy[0], -center_xy[1], -min_z])
    return transformed


def _write_obj_with_texture(mesh: trimesh.Trimesh, obj_path: Path, texture_filename: str) -> None:
    """Write textured OBJ and associated MTL."""
    mtl_path = obj_path.with_suffix(".mtl")
    mtl_name = mtl_path.name
    material_name = f"material_{obj_path.stem}"

    mtl_content = (
        f"newmtl {material_name}\n"
        f"Ka 1.000 1.000 1.000\n"
        f"Kd 1.000 1.000 1.000\n"
        f"Ks 0.100 0.100 0.100\n"
        f"Ns 10.000\n"
        f"map_Kd {texture_filename}\n"
    )
    mtl_path.write_text(mtl_content, encoding="utf-8")

    with obj_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# Exported by prepare_workshop_assets.py\n")
        stream.write(f"mtllib {mtl_name}\n")
        stream.write(f"usemtl {material_name}\n\n")
        for vertex in mesh.vertices:
            stream.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        uvs = getattr(mesh.visual, "uv", None)
        if uvs is not None and len(uvs) == len(mesh.vertices):
            for uv in uvs:
                stream.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            for face in mesh.faces:
                stream.write(
                    f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n"
                )
        else:
            for face in mesh.faces:
                stream.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def generate_workshop_parts_tray(output_dir: Path) -> dict[str, Any]:
    """Generate clean, realistic workshop utility parts tray without boolean operations."""
    w, d, h = 0.22, 0.14, 0.032
    t = 0.008   # wall thickness
    tf = 0.006  # floor thickness

    floor = trimesh.creation.box(extents=[w, d, tf])
    floor.apply_translation([0, 0, tf / 2.0])

    wall_f = trimesh.creation.box(extents=[w, t, h - tf])
    wall_f.apply_translation([0, -d / 2.0 + t / 2.0, tf + (h - tf) / 2.0])

    wall_b = trimesh.creation.box(extents=[w, t, h - tf])
    wall_b.apply_translation([0, d / 2.0 - t / 2.0, tf + (h - tf) / 2.0])

    wall_l = trimesh.creation.box(extents=[t, d - 2 * t, h - tf])
    wall_l.apply_translation([-w / 2.0 + t / 2.0, 0, tf + (h - tf) / 2.0])

    wall_r = trimesh.creation.box(extents=[t, d - 2 * t, h - tf])
    wall_r.apply_translation([w / 2.0 - t / 2.0, 0, tf + (h - tf) / 2.0])

    tray_mesh = trimesh.util.concatenate([floor, wall_f, wall_b, wall_l, wall_r])
    tray_mesh = _center_and_ground(tray_mesh)

    obj_path = output_dir / "workshop_parts_tray.obj"
    mtl_path = output_dir / "workshop_parts_tray.mtl"
    tex_path = output_dir / "workshop_parts_tray_diff.png"

    img = Image.new("RGB", (256, 256), color=(48, 56, 64))
    draw = ImageDraw.Draw(img)
    draw.rectangle([6, 6, 249, 249], outline=(72, 85, 98), width=6)
    img.save(tex_path, format="PNG", optimize=True)

    mtl_content = (
        "newmtl material_workshop_parts_tray\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.300 0.300 0.300\n"
        "Ns 30.000\n"
        "map_Kd workshop_parts_tray_diff.png\n"
    )
    mtl_path.write_text(mtl_content, encoding="utf-8")

    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# Workshop utility parts tray\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl material_workshop_parts_tray\n\n")
        for v in tray_mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in tray_mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def _record_part(obj_path: Path, tex_path: Path | None, part_id: str) -> dict[str, Any]:
    """Inspect the saved OBJ mesh and record exact canonical metadata."""
    loaded = trimesh.load(obj_path)
    return {
        "part_id": part_id,
        "processed_filename": f"assets/workshop_realistic/{obj_path.name}",
        "processed_sha256": _sha256(obj_path),
        "texture_file": f"assets/workshop_realistic/{tex_path.name}" if tex_path and tex_path.exists() else None,
        "texture_sha256": _sha256(tex_path) if tex_path and tex_path.exists() else None,
        "canonical_dimensions_m": [float(x) for x in loaded.extents],
        "triangle_count": len(loaded.faces),
        "vertex_count": len(loaded.vertices),
    }


def generate_workshop_parts_tray(output_dir: Path) -> dict[str, Any]:
    """Generate deterministic utility parts tray using procedural box concatenation."""
    # Outer dimensions: 0.22 x 0.14 x 0.032m, wall thickness 0.006m, floor thickness 0.006m
    length = 0.22
    width = 0.14
    height = 0.032
    wall_t = 0.006
    floor_t = 0.006

    base = trimesh.creation.box(extents=[length, width, floor_t])
    base.apply_translation([0, 0, floor_t / 2.0])

    wall_h = height - floor_t
    wall_z = floor_t + wall_h / 2.0

    left_w = trimesh.creation.box(extents=[wall_t, width, wall_h])
    left_w.apply_translation([-length / 2.0 + wall_t / 2.0, 0, wall_z])

    right_w = trimesh.creation.box(extents=[wall_t, width, wall_h])
    right_w.apply_translation([length / 2.0 - wall_t / 2.0, 0, wall_z])

    inner_l = length - 2 * wall_t
    front_w = trimesh.creation.box(extents=[inner_l, wall_t, wall_h])
    front_w.apply_translation([0, -width / 2.0 + wall_t / 2.0, wall_z])

    back_w = trimesh.creation.box(extents=[inner_l, wall_t, wall_h])
    back_w.apply_translation([0, width / 2.0 - wall_t / 2.0, wall_z])

    tray_mesh = trimesh.util.concatenate([base, left_w, right_w, front_w, back_w])
    tray_mesh = _center_and_ground(tray_mesh)

    tex_path = output_dir / "workshop_parts_tray_diff.png"
    img = Image.new("RGB", (256, 256), color=(140, 145, 150))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 245, 245], outline=(100, 105, 110), width=4)
    img.save(tex_path, format="PNG")

    obj_path = output_dir / "workshop_parts_tray.obj"
    mtl_path = output_dir / "workshop_parts_tray.mtl"

    mtl_content = (
        "newmtl material_workshop_parts_tray\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.300 0.300 0.300\n"
        "Ns 30.000\n"
        "map_Kd workshop_parts_tray_diff.png\n"
    )
    mtl_path.write_text(mtl_content, encoding="utf-8")

    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# Workshop utility parts tray\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl material_workshop_parts_tray\n\n")
        for v in tray_mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in tray_mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return {
        "asset_id": "workshop_parts_tray",
        "human_readable_name": "Workshop Parts Staging Tray",
        "author": "icra-we-ball project",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source": "project-generated procedural mesh",
        "source_url": "mujoco_scenes/scripts/prepare_workshop_assets.py",
        "acquisition_date": "2026-08-18",
        "roles": ["workshop_parts_tray"],
        "processed_parts": [_record_part(obj_path, tex_path, "workshop_parts_tray")],
    }


def generate_workshop_hex_bolt(output_dir: Path) -> dict[str, Any]:
    """Generate deterministic metric M8 hex-head bolt with procedural geometry."""
    shaft = trimesh.creation.cylinder(radius=0.004, height=0.042, sections=16)
    shaft.apply_translation([0, 0, 0.021])

    head = trimesh.creation.cylinder(radius=0.009, height=0.008, sections=6)
    head.apply_translation([0, 0, 0.042 + 0.004])

    bolt_mesh = trimesh.util.concatenate([shaft, head])
    min_z = bolt_mesh.bounds[0, 2]
    bolt_mesh.apply_translation([0, 0, -min_z])
    center_xy = (bolt_mesh.bounds[0, :2] + bolt_mesh.bounds[1, :2]) / 2.0
    bolt_mesh.apply_translation([-center_xy[0], -center_xy[1], 0])

    obj_path = output_dir / "workshop_hex_bolt.obj"
    tex_path = output_dir / "screwdrivers_02_diff.png"
    mtl_path = output_dir / "workshop_hex_bolt.mtl"

    mtl_content = (
        "newmtl material_workshop_hex_bolt\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.400 0.400 0.400\n"
        "Ns 40.000\n"
        "map_Kd screwdrivers_02_diff.png\n"
    )
    mtl_path.write_text(mtl_content, encoding="utf-8")

    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# Workshop metric M8 hex-head bolt\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl material_workshop_hex_bolt\n\n")
        for v in bolt_mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in bolt_mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return {
        "asset_id": "workshop_hex_bolt",
        "human_readable_name": "Metric M8 Hex-Head Machine Bolt",
        "author": "ICRA Benchmark Suite Procedural Generator",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source": "project-generated procedural mesh",
        "source_url": "mujoco_scenes/scripts/prepare_workshop_assets.py",
        "acquisition_date": "2026-08-18",
        "roles": ["workshop_hex_bolt"],
        "processed_parts": [_record_part(obj_path, tex_path, "workshop_hex_bolt")],
    }


def prepare_assets(
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download and normalize workshop visual assets."""
    output = output_dir or DEFAULT_OUTPUT
    cache = cache_dir or DEFAULT_CACHE
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []

    for asset_id, meta in ASSET_DEFS.items():
        print(f"--> Processing asset: {asset_id} ({meta['human_readable_name']})")
        gltf_file = _fetch_gltf_bundle(asset_id, cache_dir=cache)
        scene = trimesh.load(gltf_file)

        tex_cache_dir = cache / asset_id / "textures"
        diff_jpg = None
        if tex_cache_dir.exists():
            for p in tex_cache_dir.iterdir():
                if "diff" in p.name.lower() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                    diff_jpg = p
                    break

        if hasattr(scene, "geometry"):
            geoms = list(scene.geometry.values())
        else:
            geoms = [scene]

        processed_parts: list[dict[str, Any]] = []

        if asset_id == "screwdrivers_02":
            tex_name = f"{asset_id}_diff.png"
            tex_path = output / tex_name
            if diff_jpg:
                _convert_texture(diff_jpg, tex_path)

            first_geom = geoms[0]

            # Long Phillips driver (reach 0.18m, tip 0.006m, length 0.23m)
            long_driver = _center_and_ground(first_geom)
            long_max_xy = max(long_driver.extents[0], long_driver.extents[1])
            long_len = long_driver.extents[2]
            if long_max_xy > 0 and long_len > 0:
                long_driver.apply_scale([0.026 / long_max_xy, 0.026 / long_max_xy, 0.23 / long_len])
            long_driver = _center_and_ground(long_driver)
            long_obj = output / "workshop_long_phillips_driver.obj"
            _write_obj_with_texture(long_driver, long_obj, tex_name)
            processed_parts.append(_record_part(long_obj, tex_path, "workshop_long_phillips_driver"))

            # Stubby Phillips driver (reach 0.020m, length 0.11m)
            stubby_driver = _center_and_ground(first_geom)
            stubby_max_xy = max(stubby_driver.extents[0], stubby_driver.extents[1])
            stubby_len = stubby_driver.extents[2]
            if stubby_max_xy > 0 and stubby_len > 0:
                stubby_driver.apply_scale([0.030 / stubby_max_xy, 0.030 / stubby_max_xy, 0.11 / stubby_len])
            stubby_driver = _center_and_ground(stubby_driver)
            stubby_obj = output / "workshop_stubby_phillips_driver.obj"
            _write_obj_with_texture(stubby_driver, stubby_obj, tex_name)
            processed_parts.append(_record_part(stubby_obj, tex_path, "workshop_stubby_phillips_driver"))

            # Medium Phillips screw (canonical fastener: length 0.045m, head 0.014m, shaft ~0.0055m)
            second_geom = geoms[1] if len(geoms) > 1 else geoms[0]
            med_screw = _center_and_ground(second_geom)
            med_max_xy = max(med_screw.extents[0], med_screw.extents[1])
            med_len = med_screw.extents[2]
            if med_max_xy > 0 and med_len > 0:
                med_screw.apply_scale([0.014 / med_max_xy, 0.014 / med_max_xy, 0.045 / med_len])
            med_screw = _center_and_ground(med_screw)
            med_obj = output / "workshop_medium_phillips_screw.obj"
            _write_obj_with_texture(med_screw, med_obj, tex_name)
            processed_parts.append(_record_part(med_obj, tex_path, "workshop_medium_phillips_screw"))

            # Short Phillips screw (inadequate reach/engagement: length 0.018m, head 0.014m)
            short_screw = _center_and_ground(second_geom)
            short_max_xy = max(short_screw.extents[0], short_screw.extents[1])
            short_len = short_screw.extents[2]
            if short_max_xy > 0 and short_len > 0:
                short_screw.apply_scale([0.014 / short_max_xy, 0.014 / short_max_xy, 0.018 / short_len])
            short_screw = _center_and_ground(short_screw)
            short_obj = output / "workshop_short_phillips_screw.obj"
            _write_obj_with_texture(short_screw, short_obj, tex_name)
            processed_parts.append(_record_part(short_obj, tex_path, "workshop_short_phillips_screw"))

            # Long / oversized Phillips screw (too long: length 0.085m, head 0.0105m)
            long_screw = _center_and_ground(second_geom)
            long_max_xy = max(long_screw.extents[0], long_screw.extents[1])
            long_len = long_screw.extents[2]
            if long_max_xy > 0 and long_len > 0:
                long_screw.apply_scale([0.0105 / long_max_xy, 0.0105 / long_max_xy, 0.085 / long_len])
            long_screw = _center_and_ground(long_screw)
            long_obj = output / "workshop_long_phillips_screw.obj"
            _write_obj_with_texture(long_screw, long_obj, tex_name)
            processed_parts.append(_record_part(long_obj, tex_path, "workshop_long_phillips_screw"))

        else:
            comb = trimesh.util.concatenate(geoms)

            if asset_id == "combination_wrench":
                rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                comb.apply_transform(rot)
                comb.apply_scale(0.21 / comb.extents[1])
                z_up_mesh = _center_and_ground(comb)
            elif asset_id == "Drill_01":
                z_up_mesh = _center_and_ground(_to_z_up(comb))
                z_up_mesh.apply_scale(0.21 / max(z_up_mesh.extents))
            elif asset_id in {"screwdriver", "pliers", "ratchet_wrench", "wooden_hammer_01"}:
                z_up_mesh = _center_and_ground(_to_z_up(comb))
                if asset_id == "screwdriver":
                    z_up_mesh.apply_scale(0.22 / z_up_mesh.extents[2])
                elif asset_id == "pliers":
                    z_up_mesh.apply_scale(0.19 / z_up_mesh.extents[2])
                elif asset_id == "ratchet_wrench":
                    z_up_mesh.apply_scale(0.24 / z_up_mesh.extents[2])
                elif asset_id == "wooden_hammer_01":
                    z_up_mesh.apply_scale(0.28 / z_up_mesh.extents[2])
            elif asset_id == "tool_cart":
                z_up_mesh = _center_and_ground(_to_z_up(comb))
                z_up_mesh.apply_scale([0.70 / z_up_mesh.extents[0], 0.45 / z_up_mesh.extents[1], 0.82 / z_up_mesh.extents[2]])
            elif asset_id == "plastic_container":
                z_up_mesh = _center_and_ground(_to_z_up(comb))
                z_up_mesh.apply_scale([0.16 / z_up_mesh.extents[0], 0.12 / z_up_mesh.extents[1], 0.08 / z_up_mesh.extents[2]])
            elif asset_id == "metal_toolbox":
                z_up_mesh = _center_and_ground(_to_z_up(comb))
                z_up_mesh.apply_scale([0.38 / z_up_mesh.extents[0], 0.18 / z_up_mesh.extents[1], 0.14 / z_up_mesh.extents[2]])
            elif asset_id == "wooden_table_02":
                z_up_mesh = _center_and_ground(_to_z_up(comb))
                z_up_mesh.apply_scale([1.20 / z_up_mesh.extents[0], 0.65 / z_up_mesh.extents[1], 0.68 / z_up_mesh.extents[2]])
            else:
                z_up_mesh = _center_and_ground(_to_z_up(comb))

            z_up_mesh = _center_and_ground(z_up_mesh)

            tex_name = f"{asset_id}_diff.png"
            tex_path = output / tex_name
            if diff_jpg:
                _convert_texture(diff_jpg, tex_path)

            obj_path = output / f"{asset_id}.obj"
            _write_obj_with_texture(z_up_mesh, obj_path, tex_name)
            processed_parts.append(_record_part(obj_path, tex_path, asset_id))

        manifest_entries.append({
            "asset_id": asset_id,
            "human_readable_name": meta["human_readable_name"],
            "author": meta["author"],
            "license": "CC0-1.0",
            "license_url": LICENSE_URL,
            "source": "Poly Haven",
            "source_url": f"https://polyhaven.com/a/{asset_id}",
            "acquisition_date": "2026-08-18",
            "roles": meta["roles"],
            "processed_parts": processed_parts,
        })

    # Procedural parts tray entry
    tray_entry = generate_workshop_parts_tray(output)
    manifest_entries.append(tray_entry)

    # Procedural metric hex bolt entry
    bolt_entry = generate_workshop_hex_bolt(output)
    manifest_entries.append(bolt_entry)

    manifest = {
        "schema_version": 1,
        "description": "Poly Haven CC0 and project-generated textured 3D assets for Workshop (W1) benchmark.",
        "license_summary": "All assets in this directory are licensed CC0 1.0 Universal Public Domain.",
        "assets": manifest_entries,
    }

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n--> Successfully prepared {len(manifest_entries)} asset families in {output}")
    print(f"--> Wrote provenance manifest to {manifest_path}")
    return manifest


def verify_manifest(manifest_path: Path | None = None) -> bool:
    """Verify offline that committed assets match manifest hashes, counts, and dimensions."""
    m_path = manifest_path or (DEFAULT_OUTPUT / "manifest.json")
    if not m_path.is_file():
        print(f"ERROR: Manifest not found: {m_path}")
        return False
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets", [])
    all_ok = True
    base_dir = ROOT

    for asset in assets:
        for part in asset.get("processed_parts", []):
            obj_path = base_dir / part["processed_filename"]
            if not obj_path.is_file():
                print(f"[FAIL] Missing OBJ: {obj_path}")
                all_ok = False
                continue
            actual_sha = _sha256(obj_path)
            if actual_sha != part["processed_sha256"]:
                print(f"[FAIL] Hash mismatch for {part['part_id']}: {actual_sha} != {part['processed_sha256']}")
                all_ok = False
            mesh = trimesh.load(obj_path)
            if len(mesh.faces) != part["triangle_count"] or len(mesh.vertices) != part["vertex_count"]:
                print(f"[FAIL] Geometry count mismatch for {part['part_id']}")
                all_ok = False
            max_dim_err = float(np.max(np.abs(mesh.extents - np.array(part["canonical_dimensions_m"]))))
            if max_dim_err > 1e-4:
                print(f"[FAIL] Dimension drift for {part['part_id']}: max error = {max_dim_err:.6f}")
                all_ok = False

            tex_file = part.get("texture_file")
            if tex_file:
                tex_path = base_dir / tex_file
                if not tex_path.is_file():
                    print(f"[FAIL] Missing texture: {tex_path}")
                    all_ok = False
                elif part.get("texture_sha256") and _sha256(tex_path) != part["texture_sha256"]:
                    print(f"[FAIL] Texture hash mismatch for {tex_file}")
                    all_ok = False

    if all_ok:
        print(f"[PASS] Manifest offline verification passed ({len(assets)} asset families).")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare realistic workshop assets.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing existing files.")
    parser.add_argument("--verify", action="store_true", help="Verify committed assets against manifest offline.")
    args = parser.parse_args()
    if args.verify:
        ok = verify_manifest()
        raise SystemExit(0 if ok else 1)
    prepare_assets(force=args.force)


if __name__ == "__main__":
    main()
