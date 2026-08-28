"""Create the compact paper table from planning-to-GT batch artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _metrics(payload: dict[str, Any]) -> tuple[float, float]:
    comparison = payload.get("gt_comparison")
    if not isinstance(comparison, dict):
        return 0.0, 0.0
    sequence = comparison.get("shared_task_vocabulary", comparison)
    if not isinstance(sequence, dict):
        return 0.0, 0.0
    return (
        float(sequence.get("exact_sequence_match", False)),
        float(sequence.get("ordered_f1", 0.0)),
    )


def main() -> None:
    arguments = build_parser().parse_args()
    grouped: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    seen: set[Path] = set()
    for root in arguments.roots:
        summary = root.resolve() / "batch_summary.json"
        if summary.is_file():
            document = json.loads(summary.read_text(encoding="utf-8"))
            for row in document.get("runs", ()):
                method = "OWL-TAMP" if row["method"] == "owl_tamp" else "VLM-TAMP"
                environment = str(row["environment"])
                camera_count = int(row.get("camera_count", 5))
                grouped[(environment, method, camera_count)].append(
                    _metrics({"gt_comparison": row.get("gt_comparison")})
                )
            continue
        for path in root.resolve().rglob("episode_result.json"):
            if path in seen:
                continue
            seen.add(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            method = str(payload.get("baseline", "unknown"))
            method = "OWL-TAMP" if method.startswith("owl_tamp") else "VLM-TAMP"
            environment = str(payload.get("environment", "kitchen"))
            camera_count = int(payload.get("camera_count", 5))
            grouped[(environment, method, camera_count)].append(_metrics(payload))

    if not grouped:
        raise SystemExit("No episode_result.json files found under the supplied roots")

    rows = []
    for (environment, method, camera_count), values in sorted(grouped.items()):
        rows.append(
            {
                "scene": environment.replace("_", " ").title(),
                "method": method,
                "images": camera_count,
                "completed_trials": len(values),
                "gt_exact_match_percent": round(100 * mean(v[0] for v in values), 2),
                "gt_sequence_lcs_f1": round(mean(v[1] for v in values), 4),
            }
        )

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with (output / "table4.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "table4.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Scene | Method | Images | Trials | GT Exact Match | GT LCS-F1")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for row in rows:
        print(
            f"{row['scene']} | {row['method']} | {row['images']} | "
            f"{row['completed_trials']} | {row['gt_exact_match_percent']:.2f}% | "
            f"{row['gt_sequence_lcs_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
