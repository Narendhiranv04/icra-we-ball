"""Summarize planning-to-ground-truth batch artifacts for the paper table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _percent(values: list[float]) -> float | None:
    return round(100 * mean(values), 2) if values else None


def _living_room_placement_metrics(snapshot: dict[str, Any]) -> tuple[float, float]:
    """Return placement correctness and goal coverage for the living-room goal."""
    region_labels = {
        str(row["id"]): str(row["label"])
        for row in snapshot.get("known_regions", ())
        if isinstance(row, dict) and "id" in row and "label" in row
    }
    placements: Counter[tuple[str, str]] = Counter()
    predicted = 0
    for entity in snapshot.get("visible_entities", ()):
        if not isinstance(entity, dict) or entity.get("kind") != "object":
            continue
        role = re.sub(r"_\d+$", "", str(entity.get("label", "")))
        if role not in {"cup", "saucer", "tv_remote"}:
            continue
        region_id = str(entity.get("facts", {}).get("region_id", ""))
        region = region_labels.get(region_id, region_id)
        if not (region.startswith("personal_table_") or region == "shared_table"):
            continue
        predicted += 1
        placements[(role, region)] += 1

    correct = min(placements[("tv_remote", "shared_table")], 1)
    for region in sorted(
        label for label in region_labels.values() if label.startswith("personal_table_")
    ):
        correct += min(placements[("cup", region)], 1)
        correct += min(placements[("saucer", region)], 1)

    correctness = correct / predicted if predicted else 0.0
    coverage = correct / 5
    return correctness, coverage


def _apply_planned_placements(snapshot: dict[str, Any], payload: dict[str, Any]) -> None:
    """Apply symbolic PLACE effects so planning-only baselines are scored fairly."""
    entities = {
        str(entity.get("id")): entity
        for entity in snapshot.get("visible_entities", ())
        if isinstance(entity, dict)
    }
    result = payload.get("result", {})
    actions: list[tuple[str, str]] = []
    for action in result.get("actions", ()):
        if action.get("operator") == "PLACE" and len(action.get("arguments", ())) >= 2:
            actions.append((str(action["arguments"][0]), str(action["arguments"][1])))
    for record in result.get("action_history", ()):
        action = record.get("action", {})
        arguments = action.get("arguments", {})
        if record.get("success") and action.get("skill") == "PLACE":
            actions.append((str(arguments.get("object_id", "")), str(arguments.get("region_id", ""))))
    for object_id, region_id in actions:
        if object_id in entities:
            entities[object_id].setdefault("facts", {})["region_id"] = region_id


def _episode_metrics(path: Path, payload: dict[str, Any]) -> dict[str, float | None]:
    comparison = payload.get("gt_comparison", {})
    expected = comparison.get("expected_outcome")
    predicted = comparison.get("predicted_outcome")
    metrics: dict[str, float | None] = {
        "outcome_correct": float(expected == predicted),
        "goal_complete": None,
        "placement_correctness": None,
        "goal_coverage": None,
        "infeasibility_detected": None,
    }
    if expected != "FEASIBLE":
        metrics["infeasibility_detected"] = float(predicted == "INFEASIBLE")
        return metrics

    snapshot_path = path.parent / "_private_evaluation" / "latest_observation.json"
    if not snapshot_path.is_file():
        return metrics
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    _apply_planned_placements(snapshot, payload)
    if payload.get("environment") == "living_room":
        correctness, coverage = _living_room_placement_metrics(snapshot)
        metrics["goal_complete"] = float(coverage == 1.0)
        metrics["placement_correctness"] = correctness
        metrics["goal_coverage"] = coverage
    else:
        metrics["goal_complete"] = float(bool(snapshot.get("goal_satisfied", False)))
    return metrics


def main() -> None:
    arguments = build_parser().parse_args()
    grouped: dict[tuple[str, str, int], list[dict[str, float | None]]] = defaultdict(list)
    seen: set[Path] = set()
    for root in arguments.roots:
        for path in root.resolve().rglob("episode_result.json"):
            if path in seen:
                continue
            seen.add(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            method = str(payload.get("baseline", "unknown"))
            method = "OWL-TAMP" if method.startswith("owl_tamp") else "VLM-TAMP"
            environment = str(payload.get("environment", "kitchen"))
            camera_count = int(payload.get("camera_count", 5))
            grouped[(environment, method, camera_count)].append(_episode_metrics(path, payload))

    if not grouped:
        raise SystemExit("No episode_result.json files found under the supplied roots")

    rows = []
    for (environment, method, camera_count), values in sorted(grouped.items()):
        def values_for(key: str) -> list[float]:
            return [value[key] for value in values if value[key] is not None]

        rows.append(
            {
                "scene": environment.replace("_", " ").title(),
                "method": method,
                "images": camera_count,
                "completed_trials": len(values),
                "correct_feasibility_decision_percent": _percent(values_for("outcome_correct")),
                "goal_completion_percent": _percent(values_for("goal_complete")),
                "placement_correctness_percent": _percent(values_for("placement_correctness")),
                "required_placement_coverage_percent": _percent(values_for("goal_coverage")),
                "infeasibility_detection_percent": _percent(values_for("infeasibility_detected")),
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
        json.dumps({"schema_version": 2, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Scene | Method | Images | Trials | Correct decision | Goal completion | "
        "Placement correctness | Goal coverage | Infeasibility detection"
    )
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in rows:
        def display(key: str) -> str:
            value = row[key]
            return "n/a" if value is None else f"{value:.2f}%"

        print(
            f"{row['scene']} | {row['method']} | {row['images']} | "
            f"{row['completed_trials']} | {display('correct_feasibility_decision_percent')} | "
            f"{display('goal_completion_percent')} | {display('placement_correctness_percent')} | "
            f"{display('required_placement_coverage_percent')} | "
            f"{display('infeasibility_detection_percent')}"
        )


if __name__ == "__main__":
    main()
