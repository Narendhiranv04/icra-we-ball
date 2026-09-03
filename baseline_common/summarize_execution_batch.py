"""Summarize physical execution batches without mixing planning-only runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FIELDS = (
    "success", "executed_actions", "model_calls", "raw_vlm_requests",
    "replans", "planning_latency_s", "elapsed_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[Path] = set()
    for root in arguments.roots:
        paths = list(root.resolve().rglob("discovery_replanning_result.json"))
        paths.extend(root.resolve().rglob("benchmark_execution_result.json"))
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            row = json.loads(path.read_text(encoding="utf-8"))
            grouped[(
                str(row.get("scene")),
                str(row.get("method", "discovery_replanning")),
                str(row.get("protocol", "native")),
                int(row.get("camera_count", 5)),
            )].append(row)
    if not grouped:
        raise SystemExit(
            "No discovery_replanning_result.json or benchmark_execution_result.json files found"
        )

    output_rows = []
    for (scene, method, protocol, cameras), rows in sorted(grouped.items()):
        # Physical outcomes depend on the contact solver, so episodes produced
        # by different engine builds are different result classes and must not
        # average into one number.  Older artifacts carry no version at all;
        # those are reported as unknown rather than silently treated as equal.
        engines = sorted({str(row.get("mujoco_version", "unrecorded")) for row in rows})
        if len(engines) > 1:
            raise SystemExit(
                f"Refusing to pool {scene}/{method}/{protocol}/images_{cameras}: "
                f"episodes span MuJoCo builds {', '.join(engines)}. Summarize "
                "each build separately."
            )
        output_rows.append({
            "scene": scene,
            "method": method,
            "protocol": protocol,
            "images": cameras,
            "mujoco_version": engines[0],
            "trials": len(rows),
            "success_percent": round(100 * mean(bool(row["success"]) for row in rows), 2),
            "mean_executed_actions": _mean(rows, "executed_actions"),
            "mean_model_calls": _mean(rows, "model_calls"),
            "mean_raw_vlm_requests": _mean(rows, "raw_vlm_requests"),
            "mean_replans": _mean(rows, "replans"),
            "mean_planning_latency_s": _mean(rows, "planning_latency_s"),
            "mean_elapsed_seconds": _mean(rows, "elapsed_seconds"),
        })
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "execution_summary.json").write_text(
        json.dumps({"schema_version": 1, "rows": output_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "execution_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(output_rows, indent=2))


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(mean(values), 4) if values else None


if __name__ == "__main__":
    main()
