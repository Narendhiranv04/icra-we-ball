#!/usr/bin/env python3
"""Download and checksum the pretrained semantic-grounding model cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    relative_path: Path
    url: str
    sha256: str


ARTIFACTS = (
    ModelArtifact(
        relative_path=Path("yolov8m-worldv2.pt"),
        url=(
            "https://github.com/ultralytics/assets/releases/download/"
            "v8.4.0/yolov8m-worldv2.pt"
        ),
        sha256=(
            "b614d33aa35b8e61d988041ff6939dfb3ed627af88ccaf643e4cdb822eb41d71"
        ),
    ),
    ModelArtifact(
        relative_path=Path("weights/clip/ViT-B-32.pt"),
        url=(
            "https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
            "ViT-B-32.pt"
        ),
        sha256=(
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
        ),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_artifact(root: Path, artifact: ModelArtifact) -> Path:
    destination = root / artifact.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == artifact.sha256:
        print(f"[cached] {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        print(f"[download] {artifact.url}")
        with urllib.request.urlopen(artifact.url) as source:
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        observed = sha256(temporary)
        if observed != artifact.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {artifact.relative_path}: "
                f"{observed} != {artifact.sha256}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[ready] {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the persistent YOLO-World and CLIP cache used by "
            "joint semantic–geometric grounding."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("semantic_model_cache"),
        help="Cache root (default: ./semantic_model_cache)",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    for artifact in ARTIFACTS:
        prepare_artifact(root, artifact)
    print(f"Semantic model cache ready: {root}")


if __name__ == "__main__":
    main()
