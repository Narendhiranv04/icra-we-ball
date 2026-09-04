"""Common artifact contract for physically executed baseline episodes."""

from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco

from .artifacts import write_json


# Every method reports a satisfied goal under this one name.  Success rates are
# computed from the `success` flag, but failure-mode breakdowns key on
# terminal_status, and a second spelling of success would split one outcome
# across two rows of that table.
GOAL_COMPLETE_STATUS = "GOAL_COMPLETE"


def _cpu_model() -> str:
    """Best-effort CPU identifier for execution provenance."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def physical_terminal_status(
    planning_status: str,
    goal_satisfied: bool,
    action_history: Sequence[Mapping[str, Any]],
) -> str:
    """Describe how a physical episode ended, not how planning ended.

    A planner status such as OWL-TAMP's ``PLAN`` or retrieval's ``RETRIEVED``
    only says a plan was produced; reported in the shared terminal_status
    column it would sit beside real execution outcomes and read as success.
    """
    if goal_satisfied:
        return GOAL_COMPLETE_STATUS
    if planning_status not in {"PLAN", "RETRIEVED"}:
        # Nothing executable ever reached the robot; that is the terminal fact.
        return planning_status
    for row in action_history:
        if not row.get("success"):
            return str(row.get("failure_code") or "execution_failed").upper()
    return "PLAN_EXHAUSTED_GOAL_NOT_SATISFIED"


def write_execution_result(
    output_dir: str | Path,
    *,
    scene: str,
    method: str,
    protocol: str,
    variant: str,
    camera_count: int,
    seed: int,
    success: bool,
    executed_actions: int,
    model_calls: int,
    raw_vlm_requests: int,
    replans: int,
    planning_latency_s: float,
    elapsed_seconds: float,
    terminal_status: str,
    terminal_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the shared final artifact consumed by execution summaries.

    The caller owns its planner and robot interface.  This module deliberately
    records only cross-method outcome data, so a batch summarizer cannot infer
    physical success from a planning-only ``episode_result.json``.
    """
    if scene not in {"kitchen", "living_room", "workshop"}:
        raise ValueError(f"Unsupported benchmark scene {scene!r}")
    if not method or not protocol or not variant:
        raise ValueError("method, protocol, and variant must be non-empty")
    if camera_count not in {1, 3, 5}:
        raise ValueError("camera_count must be one of 1, 3, or 5")
    if min(executed_actions, model_calls, raw_vlm_requests, replans) < 0:
        raise ValueError("execution counters must be non-negative")
    if planning_latency_s < 0 or elapsed_seconds < 0:
        raise ValueError("execution durations must be non-negative")
    if bool(success) != (terminal_status == GOAL_COMPLETE_STATUS):
        raise ValueError(
            f"success={bool(success)} contradicts terminal_status="
            f"{terminal_status!r}: a satisfied goal is reported as "
            f"{GOAL_COMPLETE_STATUS!r} by every method so failure-mode "
            "breakdowns stay comparable"
        )

    payload = {
        "schema_version": 1,
        "scene": scene,
        "method": method,
        "protocol": protocol,
        "variant": variant,
        "camera_count": camera_count,
        "seed": seed,
        "success": bool(success),
        "executed_actions": executed_actions,
        "model_calls": model_calls,
        "raw_vlm_requests": raw_vlm_requests,
        "replans": replans,
        "planning_latency_s": round(float(planning_latency_s), 6),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "terminal_status": terminal_status,
        "terminal_failure": dict(terminal_failure or {}),
        "physical_execution": True,
        # Physical outcomes depend on the contact solver, so the engine build
        # is part of the result.  Episodes produced by different MuJoCo
        # versions are not interchangeable and must not be pooled in one table.
        "mujoco_version": str(mujoco.__version__),
        # Contact-rich MuJoCo stepping is sensitive to the floating-point
        # behaviour of the host, so the machine is part of an episode's
        # provenance.  Episodes from different hosts are comparable only if
        # this is recorded; a grid split across machines can then be audited
        # rather than silently assumed reproducible.
        "host_cpu": _cpu_model(),
        "host_platform": platform.platform(),
    }
    write_json(Path(output_dir) / "benchmark_execution_result.json", payload)
    return payload
