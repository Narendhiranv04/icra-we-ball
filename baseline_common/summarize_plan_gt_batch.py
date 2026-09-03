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


METHOD_LABELS = {
    "vlm_tamp": "VLM-TAMP",
    "owl_tamp": "OWL-TAMP",
    "retrieval": "Retrieval",
}


def _percent(values: list[float]) -> float | None:
    return round(100 * mean(values), 2) if values else None


def _is_planning_method(value: str) -> bool:
    """True when this row belongs to the planning-to-GT table at all."""
    return any(
        value == prefix or value.startswith(f"{prefix}_")
        for prefix in METHOD_LABELS
    )


def _method_label(value: str) -> str:
    """Map a recorded baseline name onto its reported method column.

    An unrecognised name is an error rather than a silent VLM-TAMP row: the
    two baselines have different request structures and cost accounting.
    """
    for prefix, label in METHOD_LABELS.items():
        if value == prefix or value.startswith(f"{prefix}_"):
            return label
    raise SystemExit(
        f"Unknown baseline method {value!r}; expected one of "
        f"{sorted(METHOD_LABELS)}"
    )


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
    planning_rounds = payload.get("planning_rounds")
    if planning_rounds is None:
        planning_rounds = payload.get("result", {}).get("model_calls", 1)
    raw_requests = payload.get("raw_vlm_requests")
    if raw_requests is None:
        raw_requests = (
            2 * int(planning_rounds)
            if str(payload.get("baseline", "")) == "vlm_tamp"
            else int(planning_rounds)
        )
    metrics: dict[str, float | None] = {
        # A trial with no recorded expected/predicted outcome is unscored
        # rather than counted as a correct feasibility decision.
        "outcome_correct": (
            None if expected is None or predicted is None
            else float(expected == predicted)
        ),
        "goal_complete": None,
        "placement_correctness": None,
        "goal_coverage": None,
        "infeasibility_detected": None,
        "planning_rounds": float(planning_rounds),
        "raw_vlm_requests": float(raw_requests),
        "planning_latency_s": _planning_latency_seconds(path, payload),
    }
    if expected is None:
        return metrics
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
    # Kitchen and Workshop snapshots are captured before symbolic plan replay.
    # Their goal_satisfied value therefore describes the input state, not the
    # predicted final state. Leave these metrics absent until their complete
    # domain action semantics are replayed by a dedicated relation scorer.
    return metrics


def _planning_latency_seconds(path: Path, payload: dict[str, Any]) -> float | None:
    trace = path.parent / "model_trace.json"
    if trace.is_file():
        value = json.loads(trace.read_text(encoding="utf-8")).get("latency_ms")
        return None if value is None else float(value) / 1000.0
    values = []
    for call in sorted((path.parent / "model_calls").glob("*.json")):
        value = json.loads(call.read_text(encoding="utf-8")).get("latency_ms")
        if value is not None:
            values.append(float(value) / 1000.0)
    return sum(values) if values else None


def main() -> None:
    arguments = build_parser().parse_args()
    grouped: dict[tuple[str, str, str, int], list[dict[str, float | None]]] = defaultdict(list)
    failed_trials: Counter[tuple[str, str, str, int]] = Counter()
    seen: set[Path] = set()
    recorded_outputs: set[Path] = set()
    recorded_trials: set[tuple[str, str, str, str, int, int]] = set()
    skipped_episodes: list[str] = []
    skipped_rows: list[str] = []
    for root in arguments.roots:
        for path in root.resolve().rglob("episode_result.json"):
            if path in seen:
                continue
            seen.add(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            # This table reports planning-to-GT trials only.  A physically
            # executed episode belongs to the execution summary, and an
            # episode without a GT comparison is not a planning trial.
            if payload.get("physical_execution") or not payload.get("gt_comparison"):
                skipped_episodes.append(str(path))
                continue
            method = _method_label(str(payload.get("baseline", "unknown")))
            environment = str(payload.get("environment", "kitchen"))
            protocol = str(payload.get("protocol", "native"))
            camera_count = int(payload.get("camera_count", 5))
            grouped[(environment, method, protocol, camera_count)].append(_episode_metrics(path, payload))
            recorded_outputs.add(path.parent.resolve())
            recorded_trials.add((
                protocol,
                "owl_tamp" if method == "OWL-TAMP" else "vlm_tamp",
                environment,
                str(payload.get("variant", "")),
                camera_count,
                int(payload.get("seed", 0)),
            ))

    # A crashed or malformed episode has no episode_result.json.  Do not let
    # it disappear from a batch report and inflate the completed-trial count.
    for root in arguments.roots:
        for summary_path in root.resolve().rglob("batch_summary.json"):
            manifest_path = summary_path.parent / "protocol_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("physical_execution"):
                    continue
            document = json.loads(summary_path.read_text(encoding="utf-8"))
            for trial in document.get("runs", ()):
                if not isinstance(trial, dict):
                    continue
                output_value = trial.get("output_dir")
                if not isinstance(output_value, str) or not output_value:
                    continue
                output = Path(output_value).resolve()
                raw_method = str(trial.get("method", "unknown"))
                if not _is_planning_method(raw_method):
                    # A discovery or other execution batch can share a root with
                    # planning batches.  Its rows belong to a different table,
                    # so skip them instead of aborting this one.
                    skipped_rows.append(f"{summary_path}: method={raw_method}")
                    continue
                method = _method_label(raw_method)
                environment = str(trial.get("environment", "kitchen"))
                protocol = str(trial.get("protocol", "native"))
                camera_count = int(trial.get("camera_count", 5))
                key = (
                    protocol,
                    "owl_tamp" if method == "OWL-TAMP" else "vlm_tamp",
                    environment,
                    str(trial.get("variant", "")),
                    camera_count,
                    int(trial.get("seed", 0)),
                )
                if output in recorded_outputs or key in recorded_trials:
                    continue
                failed_trials[(environment, method, protocol, camera_count)] += 1

    if skipped_episodes:
        print(
            f"[summarize] skipped {len(skipped_episodes)} non-planning "
            "episode artifacts (physical execution or no GT comparison):",
            flush=True,
        )
        for skipped in sorted(skipped_episodes):
            print(f"  {skipped}", flush=True)
    if skipped_rows:
        print(
            f"[summarize] skipped {len(skipped_rows)} batch_summary row(s) from "
            "other experiment classes:",
            flush=True,
        )
        for row in sorted(set(skipped_rows)):
            print(f"  {row}", flush=True)
    if not grouped and not failed_trials:
        raise SystemExit("No planning-to-GT episode_result.json files found under the supplied roots")

    rows = []
    keys = set(grouped) | set(failed_trials)
    for environment, method, protocol, camera_count in sorted(keys):
        values = grouped[(environment, method, protocol, camera_count)]
        def values_for(key: str) -> list[float]:
            return [value[key] for value in values if value[key] is not None]

        rows.append(
            {
                "scene": environment.replace("_", " ").title(),
                "method": method,
                "protocol": protocol,
                "images": camera_count,
                "requested_trials": len(values) + failed_trials[(environment, method, protocol, camera_count)],
                "completed_trials": len(values),
                "failed_trials": failed_trials[(environment, method, protocol, camera_count)],
                "correct_feasibility_decision_percent": _percent(values_for("outcome_correct")),
                "goal_completion_percent": _percent(values_for("goal_complete")),
                "placement_correctness_percent": _percent(values_for("placement_correctness")),
                "required_placement_coverage_percent": _percent(values_for("goal_coverage")),
                "infeasibility_detection_percent": _percent(values_for("infeasibility_detected")),
                "mean_planning_rounds": round(mean(values_for("planning_rounds")), 3) if values_for("planning_rounds") else None,
                "mean_raw_vlm_requests": round(mean(values_for("raw_vlm_requests")), 3) if values_for("raw_vlm_requests") else None,
                "mean_planning_latency_s": round(mean(values_for("planning_latency_s")), 3) if values_for("planning_latency_s") else None,
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
        "Scene | Method | Protocol | Images | Requested | Completed | Failed | Correct decision | Goal completion | "
        "Placement correctness | Goal coverage | Infeasibility detection"
    )
    print("--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in rows:
        def display(key: str) -> str:
            value = row[key]
            return "n/a" if value is None else f"{value:.2f}%"

        print(
            f"{row['scene']} | {row['method']} | {row['protocol']} | {row['images']} | "
            f"{row['requested_trials']} | {row['completed_trials']} | {row['failed_trials']} | "
            f"{display('correct_feasibility_decision_percent')} | "
            f"{display('goal_completion_percent')} | {display('placement_correctness_percent')} | "
            f"{display('required_placement_coverage_percent')} | "
            f"{display('infeasibility_detection_percent')}"
        )


if __name__ == "__main__":
    main()
