"""Compare exhaustive and semantic-first binary-relation evaluation runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def _load_stages(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for stage_dir in sorted((run_dir / "stages").iterdir()):
        path = stage_dir / "pair_relation_evaluations.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(
            f"No pair_relation_evaluations.json files below {run_dir}"
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strategy": rows[-1]["pairing_strategy"],
        "stage_count": len(rows),
        "relation_evaluation_count": sum(
            int(row["relation_evaluation_count"]) for row in rows
        ),
        "skipped_relation_pair_count": sum(
            int(row.get("skipped_relation_pair_count", 0)) for row in rows
        ),
        "elapsed_seconds": sum(
            float(row.get("elapsed_seconds", 0.0)) for row in rows
        ),
    }


def _draw(report: dict[str, Any], destination: Path) -> None:
    canvas = Image.new("RGB", (1500, 760), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (40, 28),
        "Binary relation pairing strategy ablation",
        font=_font(38, bold=True),
        fill="#0f172a",
    )
    draw.text(
        (40, 82),
        "Same scene/task; unary geometry remains all-object in both strategies",
        font=_font(20),
        fill="#475569",
    )
    strategies = ("exhaustive", "semantic_first")
    colors = {"exhaustive": "#dc2626", "semantic_first": "#2563eb"}
    max_checks = max(
        report[key]["relation_evaluation_count"] for key in strategies
    )
    max_time = max(report[key]["elapsed_seconds"] for key in strategies)
    for index, key in enumerate(strategies):
        item = report[key]
        y = 180 + index * 245
        draw.text(
            (45, y), item["strategy"], font=_font(25, bold=True), fill="#111827"
        )
        checks_width = int(
            900 * item["relation_evaluation_count"] / max(1, max_checks)
        )
        draw.rectangle((390, y, 390 + checks_width, y + 65), fill=colors[key])
        checks_inside = checks_width >= 430
        draw.text(
            ((405 if checks_inside else 405 + checks_width), y + 17),
            f"{item['relation_evaluation_count']} executed relation checks",
            font=_font(18, bold=True),
            fill="white" if checks_inside else "#111827",
        )
        time_width = int(
            900 * item["elapsed_seconds"] / max(max_time, 1e-12)
        )
        draw.rectangle(
            (390, y + 90, 390 + time_width, y + 155), fill="#64748b"
        )
        time_inside = time_width >= 390
        draw.text(
            ((405 if time_inside else 405 + time_width), y + 107),
            f"{item['elapsed_seconds'] * 1000:.3f} ms binary evaluation",
            font=_font(18, bold=True),
            fill="white" if time_inside else "#111827",
        )
    reduction = report["comparison"]["relation_check_reduction_percent"]
    speedup = report["comparison"]["binary_evaluation_speedup"]
    draw.text(
        (40, 675),
        f"Semantic-first: {reduction:.1f}% fewer relation checks; {speedup:.2f}× binary-evaluation speedup in this run",
        font=_font(24, bold=True),
        fill="#166534",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def generate_pairing_strategy_report(
    exhaustive_run: str | Path,
    semantic_first_run: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    exhaustive_path = Path(exhaustive_run).resolve()
    semantic_path = Path(semantic_first_run).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    exhaustive_rows = _load_stages(exhaustive_path)
    semantic_rows = _load_stages(semantic_path)
    exhaustive = _summary(exhaustive_rows)
    semantic = _summary(semantic_rows)
    if exhaustive["strategy"] != "exhaustive_all_pairs":
        raise ValueError("First run is not exhaustive_all_pairs")
    if semantic["strategy"] != "semantic_role_scoped":
        raise ValueError("Second run is not semantic_role_scoped")
    checks = exhaustive["relation_evaluation_count"]
    semantic_checks = semantic["relation_evaluation_count"]
    exhaustive_time = exhaustive["elapsed_seconds"]
    semantic_time = semantic["elapsed_seconds"]
    report = {
        "schema_version": 1,
        "comparison_basis": (
            "same_scene_task_detector_configuration_separate_deterministic_runs"
        ),
        "unary_geometry_scope": "ALL_OBSERVED_OBJECTS",
        "exhaustive": exhaustive,
        "semantic_first": semantic,
        "stages": [
            {
                "stage": int(exhaustive_row["stage"]),
                "region_id": exhaustive_row["region_id"],
                "observed_object_count": len(
                    exhaustive_row["observed_object_ids"]
                ),
                "possible_ordered_pair_count": exhaustive_row[
                    "ordered_distinct_object_pair_count"
                ],
                "exhaustive_relation_checks": exhaustive_row[
                    "relation_evaluation_count"
                ],
                "semantic_first_relation_checks": semantic_row[
                    "relation_evaluation_count"
                ],
                "semantic_first_pruned_checks": semantic_row.get(
                    "skipped_relation_pair_count", 0
                ),
                "exhaustive_elapsed_seconds": exhaustive_row.get(
                    "elapsed_seconds", 0.0
                ),
                "semantic_first_elapsed_seconds": semantic_row.get(
                    "elapsed_seconds", 0.0
                ),
            }
            for exhaustive_row, semantic_row in zip(
                exhaustive_rows, semantic_rows, strict=True
            )
        ],
        "comparison": {
            "relation_check_reduction_count": checks - semantic_checks,
            "relation_check_reduction_percent": (
                100.0 * (checks - semantic_checks) / max(1, checks)
            ),
            "binary_evaluation_speedup": (
                exhaustive_time / max(semantic_time, 1e-12)
            ),
        },
    }
    json_path = output_path / "pairing_strategy_ablation.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    image_path = output_path / "pairing_strategy_ablation.png"
    _draw(report, image_path)
    rows = "".join(
        "<tr>"
        f"<td>{row['stage']:03d}</td><td>{html.escape(str(row['region_id']))}</td>"
        f"<td>{row['observed_object_count']}</td>"
        f"<td>{row['exhaustive_relation_checks']}</td>"
        f"<td>{row['semantic_first_relation_checks']}</td>"
        f"<td>{row['semantic_first_pruned_checks']}</td>"
        f"<td>{row['exhaustive_elapsed_seconds'] * 1000:.3f}</td>"
        f"<td>{row['semantic_first_elapsed_seconds'] * 1000:.3f}</td>"
        "</tr>"
        for row in report["stages"]
    )
    html_path = output_path / "pairing_strategy_ablation.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Pairing strategy ablation</title>"
        "<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;color:#0f172a}"
        "img{width:100%;border:1px solid #cbd5e1}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #cbd5e1;padding:9px}th{background:#e2e8f0}</style>"
        "<h1>Semantic-first versus exhaustive binary geometry</h1>"
        "<p>Both strategies compute unary geometry and semantic role compatibility for every observed object. Exhaustive evaluates every directional relation pair. Semantic-first evaluates only pairs whose subject and object have reliable semantic support for the roles declared by that relation.</p>"
        "<p>This measures the cached binary relation-evaluation portion, not RGB detection, point-cloud reconstruction, or unary property extraction.</p>"
        "<img src='pairing_strategy_ablation.png'>"
        "<table><thead><tr><th>Stage</th><th>Region</th><th>Objects</th><th>Exhaustive checks</th><th>Semantic-first checks</th><th>Pruned</th><th>Exhaustive ms</th><th>Semantic-first ms</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        encoding="utf-8",
    )
    result = {
        "json": str(json_path),
        "png": str(image_path),
        "html": str(html_path),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exhaustive_run", type=Path)
    parser.add_argument("semantic_first_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    generate_pairing_strategy_report(
        args.exhaustive_run, args.semantic_first_run, args.output_dir
    )


if __name__ == "__main__":
    main()
