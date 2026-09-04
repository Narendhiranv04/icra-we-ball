from __future__ import annotations

import json

from mujoco_scenes.baselines.vilain_tamp.artifacts import (
    atomic_write_json,
    atomic_write_text,
    build_manifest,
    sha256_file,
    sha256_text,
)


def test_atomic_text_and_json_writes(tmp_path) -> None:
    text_path = atomic_write_text(tmp_path / "nested" / "value.txt", "baseline\n")
    json_path = atomic_write_json(
        tmp_path / "record.json",
        {"status": "OK", "api_key": "must-not-persist"},
    )
    assert text_path.read_text(encoding="utf-8") == "baseline\n"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "api_key": "[REDACTED]",
        "status": "OK",
    }


def test_sha256_helpers_agree(tmp_path) -> None:
    path = atomic_write_text(tmp_path / "artifact.txt", "deterministic")
    assert sha256_file(path) == sha256_text("deterministic")


def test_manifest_is_sorted_hashed_and_secret_safe(tmp_path) -> None:
    second = atomic_write_text(tmp_path / "b.txt", "second")
    first = atomic_write_text(tmp_path / "a.txt", "first")
    manifest = build_manifest(
        (second, first),
        root=tmp_path,
        metadata={"run_id": "run-1", "authorization": "Bearer private"},
    )
    assert [entry["path"] for entry in manifest["artifacts"]] == ["a.txt", "b.txt"]
    assert manifest["artifacts"][0]["sha256"] == sha256_text("first")
    assert manifest["metadata"] == {
        "run_id": "run-1",
        "authorization": "[REDACTED]",
    }
