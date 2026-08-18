#!/usr/bin/env python3
"""Prepare CC0 3D assets used by the integrated Workshop (W1) benchmark.

Source glTF bundles are downloaded into an ignored cache directory (.cache).
Processed OBJ visual parts and PNG textures are exported into
assets/workshop_realistic/.  The script normalizes Poly Haven's Y-up convention
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
from PIL import Image
import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "workshop_realistic"
CACHE = ROOT / "assets" / ".cache" / "workshop_realistic"
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
            "workshop_hex_bolt",
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
    "seeding_tray_01": {
        "human_readable_name": "Workshop Parts Staging Tray",
        "author": "Poly Haven / Mike van der Valk",
        "roles": ["workshop_parts_tray"],
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
        "human_readable_name": "Heavy-Duty Workshop Workbench",
        "author": "Poly Haven / Fran Calvente",
        "roles": ["workshop_workbench_table"],
    },
}


def _download_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download_file(url: str, path: Path, expected_size: int | None = None) -> None:
    if path.is_file() and (expected_size is None or path.stat().st_size == expected_size):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _fetch_gltf_bundle(asset_id: str) -> Path:
    info = _download_json(f"{API}/files/{asset_id}")
    gltf_spec = info["gltf"]["1k"]["gltf"]
    target_dir = CACHE / asset_id
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
    # Resize texture to max 1024x1024 if larger, to keep visual quality high and memory lean
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

    vertices = mesh.vertices
    faces = mesh.faces
    uvs = getattr(mesh.visual, "uv", None)

    with obj_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# Exported by prepare_workshop_assets.py\n")
        stream.write(f"mtllib {mtl_name}\n")
        stream.write(f"usemtl {material_name}\n\n")

        for v in vertices:
            stream.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        has_uv = uvs is not None and len(uvs) == len(vertices)
        if has_uv:
            for uv in uvs:
                stream.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        for face in faces:
            if has_uv:
                stream.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
            else:
                stream.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def prepare_assets(force: bool = False) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []

    for asset_id, meta in ASSET_DEFS.items():
        print(f"--> Processing {asset_id} ({meta['human_readable_name']})...")
        gltf_file = _fetch_gltf_bundle(asset_id)
        asset_dir = gltf_file.parent

        # Find diffuse texture
        diff_textures = list(asset_dir.glob("textures/*diff*.jpg"))
        if not diff_textures:
            diff_textures = list(asset_dir.glob("textures/*.jpg"))
        diff_jpg = diff_textures[0] if diff_textures else None

        loaded = trimesh.load(str(gltf_file))
        geoms = loaded.geometry if isinstance(loaded, trimesh.Scene) else {"default": loaded}

        processed_parts = []

        if asset_id == "screwdrivers_02":
            tex_name = "screwdrivers_02_diff.png"
            tex_path = OUTPUT / tex_name
            if diff_jpg:
                _convert_texture(diff_jpg, tex_path)

            geom_list = list(geoms.values())
            # Main long phillips screwdriver (from geom 0)
            long_driver = _center_and_ground(geom_list[0])
            cur_len = long_driver.extents[2]
            if cur_len > 0:
                long_driver.apply_scale(0.26 / cur_len)
            long_driver = _center_and_ground(long_driver)
            driver_obj = OUTPUT / "workshop_long_phillips_driver.obj"
            _write_obj_with_texture(long_driver, driver_obj, tex_name)
            processed_parts.append({
                "part_id": "workshop_long_phillips_driver",
                "processed_filename": f"assets/workshop_realistic/{driver_obj.name}",
                "processed_sha256": _sha256(driver_obj),
                "texture_file": f"assets/workshop_realistic/{tex_name}",
                "texture_sha256": _sha256(tex_path),
                "canonical_dimensions_m": [float(x) for x in long_driver.extents],
                "triangle_count": len(long_driver.faces),
                "vertex_count": len(long_driver.vertices),
            })

            # Stubby phillips screwdriver (scaled shorter shaft)
            stubby_driver = _center_and_ground(geom_list[0])
            stubby_scale = np.array([1.1, 1.1, 0.13 / cur_len])
            stubby_driver.apply_scale(stubby_scale)
            stubby_driver = _center_and_ground(stubby_driver)
            stubby_obj = OUTPUT / "workshop_stubby_phillips_driver.obj"
            _write_obj_with_texture(stubby_driver, stubby_obj, tex_name)
            processed_parts.append({
                "part_id": "workshop_stubby_phillips_driver",
                "processed_filename": f"assets/workshop_realistic/{stubby_obj.name}",
                "processed_sha256": _sha256(stubby_obj),
                "texture_file": f"assets/workshop_realistic/{tex_name}",
                "texture_sha256": _sha256(tex_path),
                "canonical_dimensions_m": [float(x) for x in stubby_driver.extents],
                "triangle_count": len(stubby_driver.faces),
                "vertex_count": len(stubby_driver.vertices),
            })

            # Medium Phillips screw (canonical fastener: length 0.045m, head 0.016m, shaft 0.007m)
            second_geom = geom_list[1] if len(geom_list) > 1 else geom_list[0]
            med_screw = _center_and_ground(second_geom)
            med_len = med_screw.extents[2]
            if med_len > 0:
                med_screw.apply_scale(0.045 / med_len)
            med_screw = _center_and_ground(med_screw)
            med_obj = OUTPUT / "workshop_medium_phillips_screw.obj"
            _write_obj_with_texture(med_screw, med_obj, tex_name)
            processed_parts.append({
                "part_id": "workshop_medium_phillips_screw",
                "processed_filename": f"assets/workshop_realistic/{med_obj.name}",
                "processed_sha256": _sha256(med_obj),
                "texture_file": f"assets/workshop_realistic/{tex_name}",
                "texture_sha256": _sha256(tex_path),
                "canonical_dimensions_m": [float(x) for x in med_screw.extents],
                "triangle_count": len(med_screw.faces),
                "vertex_count": len(med_screw.vertices),
            })

            # Short Phillips screw (inadequate reach/engagement: length 0.018m)
            short_screw = _center_and_ground(second_geom)
            if med_len > 0:
                short_screw.apply_scale(0.018 / med_len)
            short_screw = _center_and_ground(short_screw)
            short_obj = OUTPUT / "workshop_short_phillips_screw.obj"
            _write_obj_with_texture(short_screw, short_obj, tex_name)
            processed_parts.append({
                "part_id": "workshop_short_phillips_screw",
                "processed_filename": f"assets/workshop_realistic/{short_obj.name}",
                "processed_sha256": _sha256(short_obj),
                "texture_file": f"assets/workshop_realistic/{tex_name}",
                "texture_sha256": _sha256(tex_path),
                "canonical_dimensions_m": [float(x) for x in short_screw.extents],
                "triangle_count": len(short_screw.faces),
                "vertex_count": len(short_screw.vertices),
            })

            # Long / oversized Phillips screw (too long: length 0.085m)
            long_screw = _center_and_ground(second_geom)
            if med_len > 0:
                long_screw.apply_scale(0.085 / med_len)
            long_screw = _center_and_ground(long_screw)
            long_obj = OUTPUT / "workshop_long_phillips_screw.obj"
            _write_obj_with_texture(long_screw, long_obj, tex_name)
            processed_parts.append({
                "part_id": "workshop_long_phillips_screw",
                "processed_filename": f"assets/workshop_realistic/{long_obj.name}",
                "processed_sha256": _sha256(long_obj),
                "texture_file": f"assets/workshop_realistic/{tex_name}",
                "texture_sha256": _sha256(tex_path),
                "canonical_dimensions_m": [float(x) for x in long_screw.extents],
                "triangle_count": len(long_screw.faces),
                "vertex_count": len(long_screw.vertices),
            })

            # Hex bolt decoy (incompatible fastener)
            hex_bolt = _center_and_ground(second_geom)
            if med_len > 0:
                hex_bolt.apply_scale([1.3, 1.3, 0.050 / med_len])
            hex_bolt = _center_and_ground(hex_bolt)
            bolt_obj = OUTPUT / "workshop_hex_bolt.obj"
            _write_obj_with_texture(hex_bolt, bolt_obj, tex_name)
            processed_parts.append({
                "part_id": "workshop_hex_bolt",
                "processed_filename": f"assets/workshop_realistic/{bolt_obj.name}",
                "processed_sha256": _sha256(bolt_obj),
                "texture_file": f"assets/workshop_realistic/{tex_name}",
                "texture_sha256": _sha256(tex_path),
                "canonical_dimensions_m": [float(x) for x in hex_bolt.extents],
                "triangle_count": len(hex_bolt.faces),
                "vertex_count": len(hex_bolt.vertices),
            })

        else:
            combined_mesh = trimesh.util.concatenate(list(geoms.values()))
            z_up_mesh = _center_and_ground(_to_z_up(combined_mesh))

            if asset_id == "screwdriver":
                max_ext = max(z_up_mesh.extents)
                if max_ext > 0:
                    z_up_mesh.apply_scale(0.22 / max_ext)
            elif asset_id == "Drill_01":
                max_ext = max(z_up_mesh.extents)
                if max_ext > 0:
                    z_up_mesh.apply_scale(0.21 / max_ext)
            elif asset_id == "pliers":
                max_ext = max(z_up_mesh.extents)
                if max_ext > 0:
                    z_up_mesh.apply_scale(0.19 / max_ext)
            elif asset_id == "combination_wrench":
                max_ext = max(z_up_mesh.extents)
                if max_ext > 0:
                    z_up_mesh.apply_scale(0.21 / max_ext)
            elif asset_id == "ratchet_wrench":
                max_ext = max(z_up_mesh.extents)
                if max_ext > 0:
                    z_up_mesh.apply_scale(0.24 / max_ext)
            elif asset_id == "wooden_hammer_01":
                max_ext = max(z_up_mesh.extents)
                if max_ext > 0:
                    z_up_mesh.apply_scale(0.28 / max_ext)
            elif asset_id == "tool_cart":
                z_up_mesh.apply_scale([0.70 / z_up_mesh.extents[0], 0.45 / z_up_mesh.extents[1], 0.82 / z_up_mesh.extents[2]])
            elif asset_id == "seeding_tray_01":
                z_up_mesh.apply_scale([0.32 / z_up_mesh.extents[0], 0.22 / z_up_mesh.extents[1], 0.045 / z_up_mesh.extents[2]])
            elif asset_id == "plastic_container":
                z_up_mesh.apply_scale([0.16 / z_up_mesh.extents[0], 0.12 / z_up_mesh.extents[1], 0.08 / z_up_mesh.extents[2]])
            elif asset_id == "metal_toolbox":
                z_up_mesh.apply_scale([0.38 / z_up_mesh.extents[0], 0.18 / z_up_mesh.extents[1], 0.14 / z_up_mesh.extents[2]])
            elif asset_id == "wooden_table_02":
                z_up_mesh.apply_scale([1.20 / z_up_mesh.extents[0], 0.65 / z_up_mesh.extents[1], 0.68 / z_up_mesh.extents[2]])

            z_up_mesh = _center_and_ground(z_up_mesh)

            tex_name = f"{asset_id}_diff.png"
            tex_path = OUTPUT / tex_name
            if diff_jpg:
                _convert_texture(diff_jpg, tex_path)

            obj_path = OUTPUT / f"{asset_id}.obj"
            _write_obj_with_texture(z_up_mesh, obj_path, tex_name)

            processed_parts.append({
                "part_id": asset_id,
                "processed_filename": f"assets/workshop_realistic/{obj_path.name}",
                "processed_sha256": _sha256(obj_path),
                "texture_file": f"assets/workshop_realistic/{tex_name}",
                "texture_sha256": _sha256(tex_path) if tex_path.exists() else None,
                "canonical_dimensions_m": [float(x) for x in z_up_mesh.extents],
                "triangle_count": len(z_up_mesh.faces),
                "vertex_count": len(z_up_mesh.vertices),
            })

        manifest_entries.append({
            "asset_id": asset_id,
            "human_readable_name": meta["human_readable_name"],
            "author": meta["author"],
            "license": "CC0-1.0",
            "license_url": LICENSE_URL,
            "source": "Poly Haven",
            "source_url": f"https://polyhaven.com/a/{asset_id}",
            "download_date": str(date.today()),
            "roles": meta["roles"],
            "processed_parts": processed_parts,
        })

    manifest = {
        "schema_version": 1,
        "description": "Poly Haven CC0 textured 3D assets for Workshop (W1) benchmark.",
        "license_summary": "All assets in this directory are licensed CC0 1.0 Universal Public Domain.",
        "assets": manifest_entries,
    }

    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n--> Successfully prepared {len(manifest_entries)} asset families in {OUTPUT}")
    print(f"--> Wrote provenance manifest to {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare realistic workshop assets.")
    parser.add_argument("--force", action="store_true", help="Force reprocessing existing files.")
    args = parser.parse_args()
    prepare_assets(force=args.force)


if __name__ == "__main__":
    main()
