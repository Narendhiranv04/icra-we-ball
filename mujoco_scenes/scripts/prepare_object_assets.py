#!/usr/bin/env python3
"""Download and normalize the kitchen object meshes used by the scenes.

YCB meshes come from the official 16k object archives. Google Scanned Object
meshes are downloaded from the lightweight MuJoCo conversion by Kevin Zakka;
the OBJ/PNG payloads remain CC-BY-4.0 Google Scanned Objects assets.

The script is idempotent. Pass ``--force`` to replace existing prepared files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "assets" / "objects" / "meshes"
YCB_BASE = "https://ycb-benchmarks.s3.amazonaws.com/data/google"
YCB_MIRROR_BASE = (
    "https://huggingface.co/datasets/Kashu7100/eden_ycb/resolve/main/models"
)
GSO_MUJOCO_BASE = (
    "https://raw.githubusercontent.com/kevinzakka/"
    "mujoco_scanned_objects/main/models"
)

YCB_OBJECTS = {
    "coffee_can": "002_master_chef_can",
    "sugar_box": "004_sugar_box",
    "tea_box": "008_pudding_box",
    "bowl": "024_bowl",
    "mug": "025_mug",
    "plate": "029_plate",
    "fork": "030_fork",
    "spoon": "031_spoon",
    "knife": "032_knife",
    "marker": "040_large_marker",
    "cup": "065-a_cups",
}

GSO_OBJECTS = {
    "kettle": "Threshold_Porcelain_Teapot_White",
    "coffee_jar": "Nescafe_Tasters_Choice_Instant_Coffee_Decaf_House_Blend_Light_7_oz",
    "sugar_jar": "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold",
    "gso_canister_distractor": (
        "Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister"
    ),
    "gso_spatula_distractor": "OXO_Cookie_Spatula",
}


USER_AGENT = "icra-we-ball-assets/1"


def _single_request(url: str, byte_range: tuple[int, int] | None = None) -> tuple[int, dict, bytes]:
    headers = {"User-Agent": USER_AGENT}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.status, dict(response.headers), response.read()
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 7:
                time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def _request_bytes(url: str) -> bytes:
    """Download with parallel ranges when the server advertises byte ranges."""
    status, headers, probe = _single_request(url, (0, 0))
    content_range = headers.get("Content-Range", "")
    if status != 206 or "/" not in content_range:
        return probe

    total = int(content_range.rsplit("/", 1)[1])
    if total == 1:
        return probe
    chunk_size = 256 * 1024
    ranges = [
        (start, min(start + chunk_size - 1, total - 1))
        for start in range(0, total, chunk_size)
    ]

    def fetch(item: tuple[int, tuple[int, int]]) -> tuple[int, bytes]:
        index, byte_range = item
        chunk_status, _chunk_headers, data = _single_request(url, byte_range)
        expected = byte_range[1] - byte_range[0] + 1
        if chunk_status != 206 or len(data) != expected:
            raise RuntimeError(
                f"Invalid range response for {url}: {byte_range}, "
                f"status={chunk_status}, bytes={len(data)}, expected={expected}"
            )
        return index, data

    chunks: list[bytes | None] = [None] * len(ranges)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for index, data in executor.map(fetch, enumerate(ranges)):
            chunks[index] = data
    payload = b"".join(chunk for chunk in chunks if chunk is not None)
    if len(payload) != total:
        raise RuntimeError(f"Incomplete ranged download for {url}")
    return payload


def _request_plain(url: str) -> bytes:
    """Download a small CDN-hosted file without probing byte ranges."""
    _status, _headers, payload = _single_request(url)
    return payload


def _clean_obj(data: bytes, material_name: str = "material_0") -> bytes:
    """Centre vertices and remove external MTL references."""
    lines = data.decode("utf-8", errors="replace").splitlines()
    vertices = []
    for line in lines:
        if line.startswith("v "):
            values = line.split()
            vertices.append(tuple(float(value) for value in values[1:4]))
    if not vertices:
        raise RuntimeError("OBJ contains no vertices")
    lower = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    upper = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    centre = [(lower[axis] + upper[axis]) / 2 for axis in range(3)]

    cleaned = [f"usemtl {material_name}"]
    vertex_index = 0
    for line in lines:
        if line.startswith(("mtllib ", "usemtl ")):
            continue
        if line.startswith("v "):
            vertex = vertices[vertex_index]
            line = "v " + " ".join(
                f"{vertex[axis] - centre[axis]:.9g}" for axis in range(3)
            )
            vertex_index += 1
        cleaned.append(line)
    return ("\n".join(cleaned) + "\n").encode("utf-8")


def _remove_faces_above(data: bytes, max_z: float) -> bytes:
    """Create a lidless visual by removing faces above a horizontal cut."""
    lines = data.decode("utf-8", errors="replace").splitlines()
    vertices = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(tuple(float(value) for value in line.split()[1:4]))

    output = []
    for line in lines:
        if not line.startswith("f "):
            output.append(line)
            continue
        indices = []
        for token in line.split()[1:]:
            raw_index = int(token.split("/", 1)[0])
            indices.append(raw_index - 1 if raw_index > 0 else len(vertices) + raw_index)
        mean_z = sum(vertices[index][2] for index in indices) / len(indices)
        if mean_z <= max_z:
            output.append(line)
    return ("\n".join(output) + "\n").encode("utf-8")


def _write_texture(data: bytes, destination: Path) -> None:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", optimize=True)


def _write_payload(destination: Path, name: str, data: bytes, force: bool) -> Path:
    path = destination / name
    if force or not path.exists():
        path.write_bytes(data)
    return path


def _prepare_ycb(semantic_name: str, object_id: str, force: bool) -> dict:
    destination = OUTPUT_ROOT / "ycb" / semantic_name
    model_path = destination / f"ycb_{semantic_name}.obj"
    texture_path = destination / f"ycb_{semantic_name}.png"
    _migrate_legacy_names(destination, model_path, texture_path)
    url = f"{YCB_BASE}/{object_id}_google_16k.tgz"
    if force or not (model_path.exists() and texture_path.exists()):
        print(f"[YCB] {semantic_name} <- {object_id}")
        mirror = f"{YCB_MIRROR_BASE}/{object_id}"
        obj_data = _request_plain(f"{mirror}/textured.obj")
        texture_data = _request_plain(f"{mirror}/texture_map.png")
        destination.mkdir(parents=True, exist_ok=True)
        _write_payload(destination, model_path.name, _clean_obj(obj_data), force=True)
        _write_texture(texture_data, texture_path)
    else:
        model_path.write_bytes(_clean_obj(model_path.read_bytes()))
    return _record(
        "YCB",
        semantic_name,
        object_id,
        url,
        model_path,
        texture_path,
        prepared_from=f"{YCB_MIRROR_BASE}/{object_id}",
    )


def _prepare_gso(semantic_name: str, object_id: str, force: bool) -> dict:
    destination = OUTPUT_ROOT / "gso" / semantic_name
    model_path = destination / f"gso_{semantic_name}.obj"
    texture_path = destination / f"gso_{semantic_name}.png"
    _migrate_legacy_names(destination, model_path, texture_path)
    source = f"{GSO_MUJOCO_BASE}/{object_id}"
    if force or not (model_path.exists() and texture_path.exists()):
        print(f"[GSO] {semantic_name} <- {object_id}")
        destination.mkdir(parents=True, exist_ok=True)
        obj_data = _request_plain(f"{source}/model.obj")
        texture_data = _request_plain(f"{source}/texture.png")
        _write_payload(destination, model_path.name, _clean_obj(obj_data), force=True)
        _write_texture(texture_data, texture_path)
    else:
        model_path.write_bytes(_clean_obj(model_path.read_bytes()))
    runtime_model_path = model_path
    if semantic_name == "coffee_jar":
        runtime_model_path = destination / "gso_coffee_jar_open.obj"
        runtime_model_path.write_bytes(
            _remove_faces_above(model_path.read_bytes(), max_z=0.060)
        )
    return _record(
        "Google Scanned Objects",
        semantic_name,
        object_id,
        f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{object_id}",
        runtime_model_path,
        texture_path,
        prepared_from=source,
    )


def _migrate_legacy_names(destination: Path, model_path: Path, texture_path: Path) -> None:
    """Rename early generic outputs so MuJoCo's basename resolver stays unique."""
    legacy_model = destination / "model.obj"
    legacy_texture = destination / "texture.png"
    if legacy_model.exists() and not model_path.exists():
        legacy_model.rename(model_path)
    if legacy_texture.exists() and not texture_path.exists():
        legacy_texture.rename(texture_path)


def _record(
    dataset: str,
    semantic_name: str,
    object_id: str,
    source_url: str,
    model_path: Path,
    texture_path: Path,
    prepared_from: str | None = None,
) -> dict:
    record = {
        "dataset": dataset,
        "semantic_name": semantic_name,
        "dataset_id": object_id,
        "source_url": source_url,
        "model": str(model_path.relative_to(ROOT)),
        "texture": str(texture_path.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "texture_sha256": hashlib.sha256(texture_path.read_bytes()).hexdigest(),
    }
    if prepared_from:
        record["prepared_from"] = prepared_from
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace prepared assets")
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted((*YCB_OBJECTS, *GSO_OBJECTS)),
        help="prepare only this semantic asset (repeatable)",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for semantic_name, object_id in YCB_OBJECTS.items():
        if args.only and semantic_name not in args.only:
            continue
        records.append(_prepare_ycb(semantic_name, object_id, args.force))
    for semantic_name, object_id in GSO_OBJECTS.items():
        if args.only and semantic_name not in args.only:
            continue
        records.append(_prepare_gso(semantic_name, object_id, args.force))

    manifest = {
        "schema_version": 1,
        "assets": records,
        "notes": [
            "GSO has no catalogued tong scan; tongs use the included custom mesh.",
            "Textures are downsampled to at most 1024x1024 for portable rendering.",
        ],
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    if args.only and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        replaced = set(args.only)
        records = [
            record for record in previous.get("assets", [])
            if record.get("semantic_name") not in replaced
        ] + records
        records.sort(key=lambda record: (record["dataset"], record["semantic_name"]))
        manifest["assets"] = records
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} scanned assets; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
