#!/usr/bin/env python3
"""Prepare the CC0 visual assets used by the integrated L2 benchmark.

Source glTF bundles are downloaded into an ignored cache.  Processed OBJ
parts and PNG albedo textures are the only redistributed files.  The script
normalizes Poly Haven's Y-up convention to MuJoCo Z-up and writes complete
hash/provenance metadata beside the processed assets.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "living_room_realistic"
CACHE = ROOT / "assets" / ".cache" / "living_room_realistic"
API = "https://api.polyhaven.com"
LICENSE_URL = "https://polyhaven.com/license"
USER_AGENT = "icra-we-ball asset preparation/1.0"

ASSETS = {
    "modern_arm_chair_01": {
        "human_readable_name": "Modern Arm Chair 01",
        "author": "Vibrant Nordic",
        "roles": ["lounge_chair_left", "lounge_chair_right"],
        "instances": {
            "chair_left": [0.90, 0.90, 0.90],
            "chair_right": [0.90, 0.90, 0.90],
        },
    },
    "side_table_01": {
        "human_readable_name": "Side Table 01",
        "author": "James Ray Cock",
        "roles": ["personal_side_table_left", "personal_side_table_right"],
        "instances": {
            "side_table_left": [1.0, 1.0, 1.0],
            "side_table_right": [1.0, 1.0, 1.0],
        },
    },
    "CoffeeTable_01": {
        "human_readable_name": "Coffee Table 01",
        "author": "Fernando Quinn",
        "roles": ["central_coffee_table"],
        "instances": {"coffee_table": [0.75, 0.75, 0.75]},
    },
    "WoodenTable_02": {
        "human_readable_name": "Wooden Table 02",
        "author": "Fran Calvente",
        "roles": ["accent_table"],
        "instances": {"accent_table": [1.20, 1.20, 1.20]},
    },
    "chinese_console_table": {
        "human_readable_name": "Chinese Console Table",
        "author": "Kirill Sannikov",
        "roles": ["media_console", "staging_table"],
        "instances": {
            "media_console": [0.78, 0.78, 0.78],
            "staging_table": [0.80, 1.45, 1.10],
        },
    },
}


def _download_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, path: Path, expected_size: int) -> None:
    if path.is_file() and path.stat().st_size == expected_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) != expected_size:
        raise RuntimeError(
            f"Unexpected download size for {url}: {len(data)} != {expected_size}"
        )
    path.write_bytes(data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_sha256(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _normalized_parts(source: Path) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    scene = trimesh.load(source, force="scene")
    parts = list(scene.dump(concatenate=False))
    # Poly Haven model assets use Y-up.  Rotate +90 degrees around X so the
    # former Y axis becomes MuJoCo +Z, then put the asset on z=0 and center XY.
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    for part in parts:
        part.apply_transform(rotation)
    bounds = np.array(
        [
            np.min([part.bounds[0] for part in parts], axis=0),
            np.max([part.bounds[1] for part in parts], axis=0),
        ]
    )
    translation = np.eye(4)
    translation[:3, 3] = (
        -0.5 * (bounds[0, 0] + bounds[1, 0]),
        -0.5 * (bounds[0, 1] + bounds[1, 1]),
        -bounds[0, 2],
    )
    for part in parts:
        part.apply_transform(translation)
    bounds = np.array(
        [
            np.min([part.bounds[0] for part in parts], axis=0),
            np.max([part.bounds[1] for part in parts], axis=0),
        ]
    )
    return parts, bounds[1] - bounds[0]


def _write_part(asset_id: str, index: int, mesh: trimesh.Trimesh) -> dict:
    part_name = f"{asset_id}_part_{index:02d}"
    obj_text, resources = trimesh.exchange.obj.export_obj(
        mesh, include_texture=True, return_texture=True
    )
    # MuJoCo receives the albedo through an explicit MJCF material.  Removing
    # the generated MTL reference prevents external material interpretation
    # from varying between mesh loaders.
    obj_text = "\n".join(
        line
        for line in obj_text.splitlines()
        if not line.startswith(("mtllib ", "usemtl "))
    ).rstrip() + "\n"
    obj_path = OUTPUT / f"{part_name}.obj"
    obj_path.write_text(obj_text, encoding="utf-8")
    texture_resources = [
        (name, data)
        for name, data in resources.items()
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not texture_resources:
        raise RuntimeError(f"No albedo texture exported for {part_name}")
    _source_name, texture_data = texture_resources[0]
    texture_path = OUTPUT / f"{part_name}.png"
    texture_path.write_bytes(texture_data)
    if asset_id == "CoffeeTable_01":
        # The source model is excellent but its turquoise distressed paint is
        # visually dominant in a sparse room.  A deterministic warm-neutral
        # grade preserves all photographed albedo detail while harmonizing the
        # asset with the apartment palette.
        with Image.open(texture_path) as image:
            source_rgb = image.convert("RGB")
            tint = Image.new("RGB", source_rgb.size, (112, 76, 49))
            Image.blend(source_rgb, tint, 0.58).save(
                texture_path, optimize=True
            )
    return {
        "part_id": part_name,
        "processed_filename": obj_path.relative_to(ROOT).as_posix(),
        "processed_sha256": _sha256(obj_path),
        "texture_file": texture_path.relative_to(ROOT).as_posix(),
        "texture_sha256": _sha256(texture_path),
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.faces)),
    }


def prepare_asset(asset_id: str, specification: dict) -> dict:
    info = _download_json(f"{API}/info/{asset_id}")
    files = _download_json(f"{API}/files/{asset_id}")
    source = files["gltf"]["1k"]["gltf"]
    cache_dir = CACHE / asset_id
    mapping = {f"{asset_id}_1k.gltf": source, **source["include"]}
    downloaded = []
    for relative_name, record in mapping.items():
        target = cache_dir / relative_name
        _download(record["url"], target, int(record["size"]))
        downloaded.append(target)
    parts, dimensions = _normalized_parts(cache_dir / f"{asset_id}_1k.gltf")
    processed = [
        _write_part(asset_id, index, part)
        for index, part in enumerate(parts, 1)
    ]
    if info.get("authors") != {specification["author"]: "All"}:
        raise RuntimeError(f"Unexpected author metadata for {asset_id}")
    instances = []
    for instance_id, scale in specification["instances"].items():
        instances.append(
            {
                "instance_id": instance_id,
                "final_scale": scale,
                "final_dimensions_m": (
                    dimensions * np.asarray(scale, dtype=float)
                ).round(6).tolist(),
            }
        )
    return {
        "asset_id": asset_id,
        "human_readable_name": specification["human_readable_name"],
        "source": "Poly Haven",
        "source_url": f"https://polyhaven.com/a/{asset_id}",
        "source_author": specification["author"],
        "license": "CC0-1.0",
        "license_url": LICENSE_URL,
        "download_date": date.today().isoformat(),
        "original_filename": f"{asset_id}_1k.gltf bundle",
        "original_sha256": _bundle_sha256(downloaded, cache_dir),
        "original_units": "metres (Poly Haven metadata converted from mm)",
        "canonical_dimensions_m": dimensions.round(6).tolist(),
        "processed_filename": processed[0]["processed_filename"],
        "processed_sha256": processed[0]["processed_sha256"],
        "processed_parts": processed,
        "texture_files": [part["texture_file"] for part in processed],
        "coordinate_transform": "rotate +90 degrees about X; center XY; min Z=0",
        "processing_steps": [
            "download Poly Haven 1k glTF and required resources",
            "verify declared byte sizes",
            "apply glTF scene-node transforms",
            "convert Y-up to MuJoCo Z-up",
            "center on XY origin and ground at Z=0",
            "export independent textured OBJ visual parts",
            *(
                ["apply deterministic warm-neutral albedo grade"]
                if asset_id == "CoffeeTable_01"
                else []
            ),
            "retain analytic collision proxies in MJCF",
        ],
        "final_role": specification["roles"],
        "scene_instances": instances,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = [
        prepare_asset(asset_id, specification)
        for asset_id, specification in ASSETS.items()
    ]
    manifest = {
        "schema_version": 1,
        "collection": "integrated_living_room_realistic_furniture",
        "license_policy": "retained assets must permit repository redistribution",
        "assets": records,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(records)} CC0 assets in {OUTPUT}")


if __name__ == "__main__":
    main()
