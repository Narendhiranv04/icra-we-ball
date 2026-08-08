"""Privileged offline oracle for the kitchen feasibility benchmark.

This module never reads RGB, point clouds, detector output, registries,
observed graphs, or predicted witnesses. It labels controlled scenes from
their exact construction roster and analytic object specifications only.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import yaml

from mujoco_scenes.scene_loader import (
    KITCHEN_FEASIBILITY_VARIANTS,
    SceneConfig,
    load_all_configs,
)


ORACLE_BASIS = "PRIVILEGED_ORACLE_EVALUATION_ONLY"
STAGE_LABELS = ("INITIAL", "D1", "D2", "C2", "B1", "C1")


def load_feasibility_benchmark_config(
    path: str | Path = KITCHEN_FEASIBILITY_VARIANTS,
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("variants"):
        raise ValueError("Kitchen feasibility variant config is empty")
    return payload


def _scene_instances(scene: SceneConfig) -> list[dict[str, Any]]:
    """Enumerate privileged physical instances with oracle-only IDs."""
    occurrences: dict[str, int] = {}
    result = []

    def add(kind: str, region: str) -> None:
        occurrences[kind] = occurrences.get(kind, 0) + 1
        result.append({
            "oracle_object_id": (
                f"oracle_{len(result) + 1:04d}"
            ),
            "object_kind": kind,
            "occurrence": occurrences[kind],
            "region": region,
        })

    for kind in scene.countertop_objects.values():
        add(kind, "INITIAL")
    for region in ("D1", "D2", "C2", "B1", "C1"):
        for kind in scene.container_contents.get(region, []):
            add(kind, region)
    return result


def _valid_edge(
    tool: dict[str, Any],
    target: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    oracle_config: dict[str, Any],
) -> dict[str, Any] | None:
    tool_spec = specs.get(tool["object_kind"], {})
    target_spec = specs.get(target["object_kind"], {})
    if tool_spec.get("semantic_class") != "spoon":
        return None
    if float(tool_spec.get("elongation_ratio", 0.0)) < float(
        oracle_config["minimum_elongation_ratio"]
    ):
        return None
    required = (
        "total_length_m", "maximum_cross_section_m"
    )
    target_required = ("opening_width_m", "cavity_depth_m")
    if any(key not in tool_spec for key in required) or any(
        key not in target_spec for key in target_required
    ):
        return None
    clearance = float(oracle_config["clearance_margin_m"])
    grip = float(oracle_config["grip_allowance_m"])
    cross = float(tool_spec["maximum_cross_section_m"])
    length = float(tool_spec["total_length_m"])
    opening = float(target_spec["opening_width_m"])
    depth = float(target_spec["cavity_depth_m"])
    insert_margin = opening - (cross + clearance)
    reach_margin = length - grip - depth
    insertable = insert_margin > 0.0
    reaches = reach_margin >= 0.0
    if not (insertable and reaches):
        return None
    return {
        "tool_id": tool["oracle_object_id"],
        "tool_kind": tool["object_kind"],
        "target_id": target["oracle_object_id"],
        "target_kind": target["object_kind"],
        "INSERTABLE_IN": True,
        "REACHES_BOTTOM": True,
        "tool_cross_section_m": cross,
        "opening_width_m": opening,
        "clearance_margin_m": clearance,
        "insertability_margin_m": insert_margin,
        "tool_length_m": length,
        "grip_allowance_m": grip,
        "cavity_depth_m": depth,
        "reach_margin_m": reach_margin,
        "semantic_rank": int(tool_spec.get("semantic_rank", 10**6)),
    }


def _coffee_cover(
    targets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any] | None:
    options = []
    for chosen_targets in combinations(targets, 3):
        per_target = [
            sorted(
                (
                    edge for edge in edges
                    if edge["target_id"] == target["oracle_object_id"]
                ),
                key=lambda edge: (
                    edge["semantic_rank"], edge["tool_id"]
                ),
            )
            for target in chosen_targets
        ]
        if any(not candidates for candidates in per_target):
            continue
        for selected in product(*per_target):
            tools = tuple(edge["tool_id"] for edge in selected)
            options.append({
                "assignments": [deepcopy(edge) for edge in selected],
                "distinct_tool_ids": sorted(set(tools)),
                "unique_tool_count": len(set(tools)),
            })
    options.sort(key=lambda option: (
        option["unique_tool_count"],
        tuple(
            (edge["semantic_rank"], edge["tool_id"])
            for edge in option["assignments"]
        ),
        tuple(edge["target_id"] for edge in option["assignments"]),
    ))
    return options[0] if options else None


def _soup_matching(
    targets: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    full = []
    maximum = 0
    for chosen_targets in combinations(targets, min(3, len(targets))):
        per_target = [
            sorted(
                (
                    edge for edge in edges
                    if edge["target_id"] == target["oracle_object_id"]
                ),
                key=lambda edge: (
                    edge["semantic_rank"], edge["tool_id"]
                ),
            )
            for target in chosen_targets
        ]
        # Exhaustive partial matching determines true cardinality, not count.
        for choices in product(*[[None, *items] for items in per_target]):
            selected = [edge for edge in choices if edge is not None]
            tool_ids = [edge["tool_id"] for edge in selected]
            if len(tool_ids) != len(set(tool_ids)):
                continue
            maximum = max(maximum, len(selected))
        if len(chosen_targets) != 3 or any(not items for items in per_target):
            continue
        for selected in product(*per_target):
            tools = [edge["tool_id"] for edge in selected]
            if len(tools) == len(set(tools)):
                full.append({
                    "assignments": [deepcopy(edge) for edge in selected],
                    "distinct_tool_ids": sorted(tools),
                })
    full.sort(key=lambda option: (
        tuple(
            (edge["semantic_rank"], edge["tool_id"])
            for edge in option["assignments"]
        ),
        tuple(edge["target_id"] for edge in option["assignments"]),
    ))
    return (full[0] if full else None), maximum


def evaluate_oracle_subset(
    instances: Iterable[dict[str, Any]],
    oracle_config: dict[str, Any],
) -> dict[str, Any]:
    instances = list(instances)
    specs = oracle_config["object_specs"]
    coffee_targets = [
        item for item in instances
        if specs.get(item["object_kind"], {}).get("semantic_class")
        == "coffee_container"
    ]
    soup_targets = [
        item for item in instances
        if specs.get(item["object_kind"], {}).get("semantic_class")
        == "soup_container"
    ]
    tools = [
        item for item in instances
        if specs.get(item["object_kind"], {}).get("semantic_class")
        == "spoon"
    ]
    coffee_edges = [
        edge for tool in tools for target in coffee_targets
        if (edge := _valid_edge(tool, target, specs, oracle_config))
        is not None
    ]
    soup_edges = [
        edge for tool in tools for target in soup_targets
        if (edge := _valid_edge(tool, target, specs, oracle_config))
        is not None
    ]
    coffee = _coffee_cover(coffee_targets, coffee_edges)
    soup, soup_cardinality = _soup_matching(soup_targets, soup_edges)
    outcome = "FEASIBLE" if coffee is not None and soup is not None else "INFEASIBLE"
    if len(coffee_targets) < 3:
        failure = "INSUFFICIENT_COFFEE_CONTAINERS"
    elif len(soup_targets) < 3:
        failure = "INSUFFICIENT_SOUP_CONTAINERS"
    elif any(
        not any(edge["target_id"] == target["oracle_object_id"] for edge in coffee_edges)
        for target in coffee_targets
    ):
        failure = "UNCOVERED_COFFEE_TARGET"
    elif coffee is None:
        failure = "NO_COMPLETE_COFFEE_TOOL_COVER"
    elif len({edge["tool_id"] for edge in soup_edges}) < 3:
        failure = "INSUFFICIENT_DISTINCT_SOUP_TOOLS"
    elif soup is None:
        failure = "NO_COMPLETE_SOUP_MATCHING"
    else:
        failure = None
    return {
        "inference_basis": ORACLE_BASIS,
        "terminal_outcome": outcome,
        "coffee_container_count": len(coffee_targets),
        "soup_container_count": len(soup_targets),
        "oracle_valid_coffee_edges": coffee_edges,
        "oracle_valid_soup_edges": soup_edges,
        "oracle_coffee_assignments": (
            coffee["assignments"] if coffee else []
        ),
        "oracle_coffee_minimum_unique_tools": (
            coffee["unique_tool_count"] if coffee else None
        ),
        "oracle_soup_assignments": (
            soup["assignments"] if soup else []
        ),
        "oracle_soup_matching_size": soup_cardinality,
        "oracle_failure_reason": failure,
    }


def evaluate_oracle_variant(
    variant_id: str,
    *,
    benchmark_config: dict[str, Any] | None = None,
    scene_configs: dict[str, SceneConfig] | None = None,
) -> dict[str, Any]:
    benchmark = benchmark_config or load_feasibility_benchmark_config()
    variant = benchmark["variants"][variant_id]
    scenes = scene_configs or load_all_configs()
    scene = scenes[variant["scene_name"]]
    instances = _scene_instances(scene)
    cumulative_regions = {"INITIAL"}
    stages = []
    earliest = None
    for stage_index, stage_label in enumerate(STAGE_LABELS):
        cumulative_regions.add(stage_label)
        result = evaluate_oracle_subset(
            (
                item for item in instances
                if item["region"] in cumulative_regions
            ),
            benchmark["oracle"],
        )
        stages.append({
            "stage_index": stage_index,
            "stage_label": stage_label,
            **result,
        })
        if earliest is None and result["terminal_outcome"] == "FEASIBLE":
            earliest = stage_label
    final = deepcopy(stages[-1])
    final.update({
        "variant_id": variant_id,
        "scene_name": variant["scene_name"],
        "goal_instruction": benchmark["goal_instruction"],
        "oracle_terminal_outcome": final.pop("terminal_outcome"),
        "oracle_earliest_feasible_stage": earliest,
        "stage_feasibility": stages,
    })
    intended = variant["intended_outcome"]
    if final["oracle_terminal_outcome"] != intended:
        raise ValueError(
            f"{variant_id} intended {intended} but oracle calculated "
            f"{final['oracle_terminal_outcome']}"
        )
    expected_tools = variant.get("intended_min_coffee_tools")
    if (
        expected_tools is not None
        and final["oracle_coffee_minimum_unique_tools"] != expected_tools
    ):
        raise ValueError(
            f"{variant_id} intended {expected_tools} coffee tools but "
            f"oracle calculated "
            f"{final['oracle_coffee_minimum_unique_tools']}"
        )
    return final


def evaluate_all_oracle_variants() -> dict[str, dict[str, Any]]:
    benchmark = load_feasibility_benchmark_config()
    scenes = load_all_configs()
    return {
        variant_id: evaluate_oracle_variant(
            variant_id,
            benchmark_config=benchmark,
            scene_configs=scenes,
        )
        for variant_id in benchmark["variants"]
    }
