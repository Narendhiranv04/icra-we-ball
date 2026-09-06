"""Per-run event counts and response archival, including failed compilation."""
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class RunTelemetry:
    directory: Path
    semantic_vlm_requests: int = 0
    transport_attempts: int = 0
    astar_invocations: int = 0
    inference_config: dict[str, Any] = field(default_factory=dict)

    def write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'request_telemetry.json').write_text(json.dumps({
            'semantic_vlm_requests': self.semantic_vlm_requests,
            'transport_attempts': self.transport_attempts,
            'transport_retries': max(0, self.transport_attempts - self.semantic_vlm_requests),
            'astar_invocations': self.astar_invocations,
            'high_level_replans': max(0, self.astar_invocations - 1),
            'inference_config': self.inference_config,
        }, indent=2) + '\n')


current_run: ContextVar[RunTelemetry | None] = ContextVar('functional_tamp_run', default=None)


def record_semantic_request(payload):
    run = current_run.get()
    if run is not None:
        run.semantic_vlm_requests += 1
        run.inference_config = {k: v for k, v in payload.items() if k not in {'messages'}}
        run.write()
