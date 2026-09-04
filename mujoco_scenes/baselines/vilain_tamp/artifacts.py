"""Atomic, secret-conscious artifact and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = frozenset({
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "secret",
    "access_token",
    "refresh_token",
})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | Path, value: Any) -> Path:
    safe_value = redact_secrets(value)
    rendered = json.dumps(safe_value, indent=2, sort_keys=True, ensure_ascii=False)
    return atomic_write_text(path, rendered + "\n")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if str(key).lower() in _SENSITIVE_KEYS else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact_secrets(item) for item in value]
    return value


def artifact_manifest_entry(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact is not a file: {artifact}")
    display_path = artifact
    if root is not None:
        display_path = artifact.resolve().relative_to(Path(root).resolve())
    return {
        "path": display_path.as_posix(),
        "sha256": sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def build_manifest(
    artifacts: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entries = [artifact_manifest_entry(path, root=root) for path in artifacts]
    entries.sort(key=lambda entry: entry["path"])
    return {
        "artifacts": entries,
        "metadata": redact_secrets(dict(metadata or {})),
    }


def write_manifest(
    path: str | Path,
    artifacts: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    return atomic_write_json(
        path,
        build_manifest(artifacts, root=root, metadata=metadata),
    )
