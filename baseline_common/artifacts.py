"""Reproducible output-directory handling for comparison episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepare_run_directory(path: str | Path) -> Path:
    """Create an empty run directory and refuse to mix episode artifacts."""
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output}. Choose a fresh "
            "--output-dir so model calls and execution traces are not mixed."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON artifact so interrupted runs stay readable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
