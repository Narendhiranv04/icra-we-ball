"""Small JSONL event log for reproducible TAMP episodes."""

from __future__ import annotations

import json
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.events: list[dict[str, object]] = []
        self._stream = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("a", encoding="utf-8")

    def append(self, event: str, **values: object) -> None:
        record = {
            "time": time.time(),
            "event": event,
            **values,
        }
        encoded = json.dumps(record, allow_nan=False, separators=(",", ":"))
        self.events.append(json.loads(encoded))
        if self._stream is not None:
            self._stream.write(encoded + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
