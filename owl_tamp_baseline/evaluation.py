"""Private expected-GT comparison; never used to construct a model prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .models import Action


EXPECTED_ROOT = Path(__file__).parents[1] / "EXPECTED_GT_ACTIONS"


def load_expected(environment: str, variant: str) -> Mapping[str, object]:
    path = EXPECTED_ROOT / environment / variant / "expected_gt_actions.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"invalid expected GT document: {path}")
    return document


def _canonical(rows: Iterable[Action | Mapping[str, object]]) -> list[tuple[str, tuple[str, ...]]]:
    result = []
    for row in rows:
        if isinstance(row, Action):
            result.append((row.operator, row.arguments))
        else:
            result.append(
                (
                    str(row["operator"]).upper(),
                    tuple(map(str, row.get("arguments", ()))),
                )
            )
    return result


def compare_actions(
    predicted: Iterable[Action | Mapping[str, object]],
    expected: Iterable[Action | Mapping[str, object]],
) -> dict[str, object]:
    left = _canonical(predicted)
    right = _canonical(expected)
    previous = [0] * (len(right) + 1)
    for predicted_row in left:
        current = [0]
        for index, expected_row in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if predicted_row == expected_row
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision = lcs / len(left) if left else float(not right)
    recall = lcs / len(right) if right else float(not left)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_sequence_match": left == right,
        "predicted_action_count": len(left),
        "expected_action_count": len(right),
        "lcs_action_count": lcs,
        "ordered_precision": round(precision, 6),
        "ordered_recall": round(recall, 6),
        "ordered_f1": round(f1, 6),
    }


def normalize_kitchen_actions(
    rows: Iterable[Action | Mapping[str, object]],
) -> list[dict[str, object]]:
    """Map execution-only Kitchen operators to the shared task vocabulary."""
    normalized = []
    for operator, arguments in _canonical(rows):
        if operator == "CLOSE":
            continue
        if operator == "OPEN":
            operator = "INSPECT"
        elif operator == "PLACE_SERVING_UTENSIL":
            operator = "PLACE"
        normalized.append({"operator": operator, "arguments": list(arguments)})
    return normalized


def compare_kitchen_actions(
    predicted: Iterable[Action | Mapping[str, object]],
    expected: Iterable[Action | Mapping[str, object]],
) -> dict[str, object]:
    predicted_rows = tuple(predicted)
    expected_rows = tuple(expected)
    return {
        "raw_execution_vocabulary": compare_actions(predicted_rows, expected_rows),
        "shared_task_vocabulary": compare_actions(
            normalize_kitchen_actions(predicted_rows),
            normalize_kitchen_actions(expected_rows),
        ),
        "normalization": {
            "OPEN": "INSPECT",
            "CLOSE": "excluded_execution_cleanup",
            "PLACE_SERVING_UTENSIL": "PLACE",
        },
    }
