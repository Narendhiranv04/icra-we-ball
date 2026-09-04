"""Atomic, secret-conscious artifact and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
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


def append_jsonl(path: str | Path, value: Any) -> Path:
    """Append one secret-redacted JSON event using an atomic replacement."""
    destination = Path(path)
    existing = destination.read_text(encoding="utf-8") if destination.exists() else ""
    rendered = json.dumps(
        redact_secrets(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return atomic_write_text(destination, existing + rendered + "\n")


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


def repository_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Capture the exact Git state without changing repository state."""
    root = Path(repository_root).resolve()
    head = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    tracked_changes: list[str] = []
    untracked_paths: list[str] = []
    for line in status.splitlines():
        if line.startswith("?? "):
            untracked_paths.append(line[3:])
        elif line:
            tracked_changes.append(line)
    return {
        "repository_root": str(root),
        "head": head,
        "branch": branch,
        "tracked_changes": tracked_changes,
        "untracked_paths": untracked_paths,
        "dirty": bool(tracked_changes or untracked_paths),
    }


def verify_repository_provenance(
    expected: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    required_branch: str,
    allowed_untracked_paths: Iterable[str] = (),
) -> None:
    """Reject changed commits, branches, tracked files, or unknown local files."""
    if current.get("head") != expected.get("head"):
        raise ValueError("repository HEAD changed after the run started")
    if current.get("branch") != required_branch:
        raise ValueError(
            f"execution requires branch {required_branch!r}; "
            f"found {current.get('branch')!r}"
        )
    tracked = tuple(current.get("tracked_changes", ()))
    if tracked:
        raise ValueError("execution requires no tracked repository changes")
    allowed = set(allowed_untracked_paths)
    unexpected = set(current.get("untracked_paths", ())).difference(allowed)
    if unexpected:
        raise ValueError(
            "execution found unexpected untracked paths: "
            + ", ".join(sorted(unexpected))
        )


def verify_artifact_manifest(
    entries: Iterable[Mapping[str, Any]], *, root: str | Path
) -> None:
    """Verify that locked artifacts still exist with their recorded hashes."""
    artifact_root = Path(root).resolve()
    for entry in entries:
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError("artifact manifest entry has no path")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"artifact manifest entry has no hash: {relative}")
        artifact = (artifact_root / relative).resolve()
        try:
            artifact.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(f"artifact escapes run root: {relative}") from error
        if not artifact.is_file():
            raise ValueError(f"required artifact is missing: {relative}")
        if sha256_file(artifact) != expected_hash:
            raise ValueError(f"required artifact hash changed: {relative}")


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"unable to inspect repository provenance: {detail}")
    return completed.stdout.strip()
