"""Run the no-robot task-level kitchen feasibility benchmark.

The predicted path ends at FEASIBLE/INFEASIBLE. It deliberately does not
import symbolic planning, generate PDDL, or execute task actions.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from mujoco_scenes.kitchen_feasibility_oracle import (
    STAGE_LABELS,
    evaluate_oracle_variant,
    load_feasibility_benchmark_config,
)
from mujoco_scenes.scene_loader import (
    COUNTER_SPOTS,
    KitchenScene,
    load_all_configs,
)
from mujoco_scenes.sequential_inspection import run_sequential_inspection


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _group(witness: dict[str, Any], group_id: str) -> dict[str, Any]:
    return next(
        (
            item for item in witness.get("function_group_evaluations", [])
            if item.get("function_group_id") == group_id
        ),
        {},
    )


def _stage_label(stage: int | None) -> str | None:
    if not isinstance(stage, int) or not 0 <= stage < len(STAGE_LABELS):
        return None
    return STAGE_LABELS[stage]


def predicted_feasibility_from_witness(
    variant_id: str,
    goal_instruction: str,
    witness: dict[str, Any],
    *,
    inspection_count: int,
) -> dict[str, Any]:
    """Map observed evidence to the benchmark's two terminal labels.

    This function intentionally has no oracle parameter and can run when no
    oracle file exists. Intermediate uncertainty becomes INFEASIBLE only at
    fixed-order exhaustion.
    """
    complete = witness.get("status") == "COMPLETE"
    coffee = _group(witness, "coffee_stirring")
    soup = _group(witness, "soup_serving")
    coffee_assignments = coffee.get("selected_assignments", [])
    soup_assignments = soup.get("selected_assignments", [])
    coffee_tools = sorted({
        item["utensil_object_id"] for item in coffee_assignments
    })
    completion_stage = _stage_label(witness.get("stage")) if complete else None
    return {
        "variant_id": variant_id,
        "goal_instruction": goal_instruction,
        "inference_basis": (
            "OBSERVED_RGBD_SEMANTIC_GEOMETRIC_FUNCTIONAL_WITNESS"
        ),
        "terminal_outcome": "FEASIBLE" if complete else "INFEASIBLE",
        "completion_stage": completion_stage,
        "completion_stage_index": witness.get("stage") if complete else None,
        "inspection_count": inspection_count,
        "witness_status": witness.get("status"),
        "coffee_assignments": coffee_assignments,
        "coffee_unique_tool_count": len(coffee_tools) if complete else None,
        "coffee_valid_target_tool_edges": coffee.get(
            "valid_target_tool_edges", []
        ),
        "coffee_covered_targets": coffee.get(
            "covered_target_object_ids", []
        ),
        "coffee_uncovered_targets": coffee.get(
            "uncovered_target_object_ids", []
        ),
        "coffee_minimum_distinct_tools": coffee.get(
            "minimum_distinct_tool_count"
        ),
        "soup_assignments": soup_assignments,
        "soup_valid_target_tool_edges": soup.get(
            "valid_target_tool_edges", []
        ),
        "soup_maximum_matching_cardinality": soup.get(
            "maximum_matching_cardinality", 0
        ),
        "soup_unmatched_targets": soup.get(
            "uncovered_target_object_ids", []
        ),
        "soup_distinctness_satisfied": soup.get(
            "distinctness_satisfied", False
        ),
        "reason_if_infeasible": (
            None if complete else (
                witness.get("reason_codes")
                or ["NO_COMPLETE_OBSERVED_WITNESS"]
            )
        ),
    }


def _saved_stage_results(run_dir: Path) -> list[dict[str, Any]]:
    results = []
    for stage_dir in sorted((run_dir / "stages").glob("[0-9][0-9][0-9]_*")):
        witness_path = stage_dir / "witness.json"
        if not witness_path.exists():
            continue
        witness = json.loads(witness_path.read_text(encoding="utf-8"))
        results.append({
            "stage": witness.get("stage"),
            "stage_label": _stage_label(witness.get("stage")),
            "witness_status": witness.get("status"),
            "reason_codes": witness.get("reason_codes", []),
            "coffee": _group(witness, "coffee_stirring"),
            "soup": _group(witness, "soup_serving"),
            "evidence_directory": str(stage_dir.relative_to(run_dir)),
        })
    return results


def run_predicted_variant(
    variant_id: str,
    variant: dict[str, Any],
    benchmark: dict[str, Any],
    output_root: Path,
    *,
    width: int,
    height: int,
    semantic_backend: str,
    semantic_model: str | None,
    semantic_vocabulary: str | None,
    semantic_confidence_threshold: float,
    semantic_min_supporting_views: int,
    save_semantic_overlays: bool,
) -> tuple[dict[str, Any], Path]:
    """Run production prediction without receiving any oracle result."""
    scene = KitchenScene(
        variant["scene_name"], include_robot=False, robot="none"
    )
    run_sequential_inspection(
        scene,
        benchmark["inspection_order"],
        runs_root=output_root,
        run_id=variant_id,
        width=width,
        height=height,
        task_requirements=(
            Path(__file__).resolve().parent
            / "configs" / benchmark["task_requirements"]
        ),
        stop_on_complete=True,
        semantic_backend=semantic_backend,
        semantic_model=semantic_model,
        semantic_vocabulary_path=semantic_vocabulary,
        semantic_confidence_threshold=semantic_confidence_threshold,
        semantic_min_supporting_views=semantic_min_supporting_views,
        grounding_mode="joint",
        save_semantic_overlays=save_semantic_overlays,
    )
    run_dir = output_root / variant_id
    witness = json.loads(
        (run_dir / "latest_witness.json").read_text(encoding="utf-8")
    )
    stage_results = _saved_stage_results(run_dir)
    inspection_count = sum(
        result["stage"] != 0 for result in stage_results
    )
    predicted = predicted_feasibility_from_witness(
        variant_id,
        benchmark["goal_instruction"],
        witness,
        inspection_count=inspection_count,
    )
    _write_json(run_dir / "predicted_feasibility.json", predicted)
    _write_json(run_dir / "stage_results.json", stage_results)
    _write_json(run_dir / "functional_witness.json", witness)
    _write_json(run_dir / "grounded_assignments.json", {
        "coffee": predicted["coffee_assignments"],
        "soup": predicted["soup_assignments"],
    })
    return predicted, run_dir


def _comparison(
    variant_id: str,
    variant: dict[str, Any],
    oracle: dict[str, Any],
    predicted: dict[str, Any],
) -> dict[str, Any]:
    oracle_outcome = oracle["oracle_terminal_outcome"]
    predicted_outcome = predicted["terminal_outcome"]
    correctly_feasible = oracle_outcome == predicted_outcome == "FEASIBLE"
    return {
        "variant_id": variant_id,
        "intended_outcome": variant["intended_outcome"],
        "oracle_outcome": oracle_outcome,
        "predicted_outcome": predicted_outcome,
        "classification_correct": oracle_outcome == predicted_outcome,
        "oracle_earliest_feasible_stage": oracle[
            "oracle_earliest_feasible_stage"
        ],
        "predicted_completion_stage": predicted["completion_stage"],
        "stage_correct": (
            correctly_feasible
            and oracle["oracle_earliest_feasible_stage"]
            == predicted["completion_stage"]
        ),
        "oracle_min_coffee_tools": oracle[
            "oracle_coffee_minimum_unique_tools"
        ],
        "predicted_coffee_unique_tools": predicted[
            "coffee_unique_tool_count"
        ],
        "coffee_reuse_optimal": (
            correctly_feasible
            and oracle["oracle_coffee_minimum_unique_tools"]
            == predicted["coffee_unique_tool_count"]
        ),
        "oracle_soup_matching_size": oracle[
            "oracle_soup_matching_size"
        ],
        "predicted_soup_assignment_count": len(
            predicted["soup_assignments"]
        ),
        "soup_distinct_assignment_valid": (
            predicted_outcome != "FEASIBLE"
            or (
                predicted["soup_distinctness_satisfied"]
                and len(predicted["soup_assignments"]) == 3
            )
        ),
        "result": "PASS" if oracle_outcome == predicted_outcome else "FAIL",
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    feasible = [row for row in rows if row["oracle_outcome"] == "FEASIBLE"]
    infeasible = [row for row in rows if row["oracle_outcome"] == "INFEASIBLE"]
    correct = sum(row["classification_correct"] for row in rows)
    correctly_feasible = [
        row for row in feasible if row["classification_correct"]
    ]
    return {
        "number_of_variants": count,
        "number_oracle_feasible": len(feasible),
        "number_oracle_infeasible": len(infeasible),
        "overall_feasibility_accuracy": correct / count if count else 0.0,
        "feasible_recall": (
            sum(row["classification_correct"] for row in feasible)
            / len(feasible) if feasible else 0.0
        ),
        "infeasible_recall": (
            sum(row["classification_correct"] for row in infeasible)
            / len(infeasible) if infeasible else 0.0
        ),
        "false_feasible_count": sum(
            row["oracle_outcome"] == "INFEASIBLE"
            and row["predicted_outcome"] == "FEASIBLE"
            for row in rows
        ),
        "false_infeasible_count": sum(
            row["oracle_outcome"] == "FEASIBLE"
            and row["predicted_outcome"] == "INFEASIBLE"
            for row in rows
        ),
        "earliest_stage_exact_accuracy": (
            sum(row["stage_correct"] for row in correctly_feasible)
            / len(correctly_feasible) if correctly_feasible else 0.0
        ),
        "coffee_minimum_tool_optimality_accuracy": (
            sum(row["coffee_reuse_optimal"] for row in correctly_feasible)
            / len(correctly_feasible) if correctly_feasible else 0.0
        ),
        "soup_distinct_assignment_validity": (
            sum(row["soup_distinct_assignment_valid"] for row in rows)
            / count if count else 0.0
        ),
    }


def run_benchmark(arguments: argparse.Namespace) -> Path:
    benchmark = load_feasibility_benchmark_config()
    variants = benchmark["variants"]
    selected = list(variants) if arguments.all_core_variants else arguments.variant
    if not selected:
        raise ValueError("Select --all-core-variants or at least one --variant")
    unknown = set(selected) - variants.keys()
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    output_root = Path(arguments.output_root).resolve() / arguments.benchmark_id
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant_id in selected:
        variant = variants[variant_id]
        print(f"\n=== {variant_id}: {variant['scene_name']} ===")
        scene_configs = load_all_configs()
        resolved_scene = scene_configs[variant["scene_name"]]
        oracle = evaluate_oracle_variant(
            variant_id, benchmark_config=benchmark,
            scene_configs=scene_configs,
        )
        predicted, run_dir = run_predicted_variant(
            variant_id, variant, benchmark, output_root,
            width=arguments.width,
            height=arguments.height,
            semantic_backend=arguments.semantic_detector,
            semantic_model=arguments.semantic_model,
            semantic_vocabulary=arguments.semantic_vocabulary,
            semantic_confidence_threshold=(
                arguments.semantic_confidence_threshold
            ),
            semantic_min_supporting_views=(
                arguments.semantic_min_supporting_views
            ),
            save_semantic_overlays=arguments.save_semantic_overlays,
        )
        variant_payload = {
            "variant_id": variant_id,
            "goal_instruction": benchmark["goal_instruction"],
            **deepcopy(variant),
            "resolved_layout": {
                "countertop_objects": deepcopy(
                    resolved_scene.countertop_objects
                ),
                "countertop_positions_world_m": {
                    spot: list(COUNTER_SPOTS[spot])
                    for spot in resolved_scene.countertop_objects
                },
                "container_contents": deepcopy(
                    resolved_scene.container_contents
                ),
                "inspection_order": list(
                    benchmark["inspection_order"]
                ),
            },
        }
        _write_json(run_dir / "variant_config.json", variant_payload)
        _write_json(run_dir / "oracle_feasibility.json", {
            key: value for key, value in oracle.items()
            if key != "stage_feasibility"
        })
        _write_json(
            run_dir / "oracle_stage_feasibility.json",
            oracle["stage_feasibility"],
        )
        comparison = _comparison(
            variant_id, variant, oracle, predicted
        )
        _write_json(run_dir / "comparison.json", comparison)
        rows.append(comparison)
        print(
            f"{variant_id}: oracle={comparison['oracle_outcome']} "
            f"predicted={comparison['predicted_outcome']} "
            f"result={comparison['result']}"
        )
    aggregate = _aggregate(rows)
    summary = {
        "schema_version": 1,
        "benchmark_id": arguments.benchmark_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "goal_instruction": benchmark["goal_instruction"],
        "scope": "TASK_LEVEL_FEASIBILITY_ONLY_NO_ACTION_PLANNING",
        "variants": rows,
        "aggregate_metrics": aggregate,
    }
    _write_json(output_root / "benchmark_summary.json", summary)
    columns = list(rows[0]) if rows else []
    with (output_root / "benchmark_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Kitchen task-level feasibility benchmark",
        "",
        benchmark["goal_instruction"],
        "",
        "This benchmark ends at FEASIBLE/INFEASIBLE. It uses no FM, PDDL, "
        "task-action planning, robot, IK, navigation, or manipulation.",
        "",
        f"Overall accuracy: {aggregate['overall_feasibility_accuracy']:.3f}",
        "",
    ]
    lines.extend(
        f"- {row['variant_id']}: oracle={row['oracle_outcome']}, "
        f"predicted={row['predicted_outcome']}, {row['result']}"
        for row in rows
    )
    (output_root / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run observed-vs-oracle kitchen task feasibility"
    )
    parser.add_argument("--all-core-variants", action="store_true")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--no-robot", action="store_true", required=True)
    parser.add_argument("--output-root", default="runs/feasibility_benchmarks")
    parser.add_argument("--benchmark-id", default="kitchen_feasibility")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--semantic-detector", default="yolo_world")
    parser.add_argument("--semantic-model", default=None)
    parser.add_argument(
        "--semantic-vocabulary",
        default="mujoco_scenes/configs/semantic_vocabulary.yaml",
    )
    parser.add_argument("--semantic-confidence-threshold", type=float, default=0.03)
    parser.add_argument("--semantic-min-supporting-views", type=int, default=2)
    parser.add_argument("--save-semantic-overlays", action="store_true")
    arguments = parser.parse_args()
    output = run_benchmark(arguments)
    print(f"\nBenchmark artifacts: {output}")


if __name__ == "__main__":
    main()
