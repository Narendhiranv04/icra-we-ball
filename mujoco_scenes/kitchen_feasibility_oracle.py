"""Privileged offline oracle for the kitchen feasibility benchmark.

The oracle never reads RGB, point clouds, detector output, registries,
observed graphs, or predicted witnesses.  It extracts geometry from the
effective instantiated MuJoCo model and uses the production geometry config
for thresholds.  It is therefore privileged, but it has no duplicate numeric
object-dimension table.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import yaml

from mujoco_scenes.scene_loader import (
    KITCHEN_FEASIBILITY_VARIANTS,
    KitchenScene,
    SceneConfig,
    load_all_configs,
)
from mujoco_scenes.exact_scene_geometry import extract_exact_object_geometry
from mujoco_scenes.geometry_properties import load_geometry_config


ORACLE_BASIS = "PRIVILEGED_ORACLE_EVALUATION_ONLY"
STAGE_LABELS = ("INITIAL", "D1", "D2", "C2", "B1", "C1")


def load_feasibility_benchmark_config(
    path: str | Path = KITCHEN_FEASIBILITY_VARIANTS,
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("variants"):
        raise ValueError("Kitchen feasibility variant config is empty")
    return payload


def _semantic_class(object_kind: str) -> str:
    if "bowl" in object_kind or object_kind in {"bowl", "mixing_bowl"}:
        return "soup_container"
    if object_kind in {"cup", "mug", "glass"} or "cup" in object_kind or "mug" in object_kind:
        return "coffee_container"
    if "spoon" in object_kind or object_kind == "spoon":
        return "spoon"
    return "excluded_utensil"


def _scene_instances(
    scene: KitchenScene,
    geometry_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enumerate privileged physical instances with oracle-only IDs."""
    occurrences: dict[str, int] = {}
    result = []

    def add(instance_name: str, kind: str, region: str) -> None:
        occurrences[kind] = occurrences.get(kind, 0) + 1
        semantic_class = _semantic_class(kind)
        geometry = None
        if semantic_class in {"coffee_container", "soup_container", "spoon"}:
            geometry = extract_exact_object_geometry(
                scene, instance_name, kind, geometry_config=geometry_config
            )
        result.append({
            "oracle_object_id": (
                f"oracle_{len(result) + 1:04d}"
            ),
            "object_kind": kind,
            "occurrence": occurrences[kind],
            "region": region,
            "semantic_class": semantic_class,
            "semantic_rank": 1 if semantic_class == "spoon" else 10**6,
            "exact_geometry": geometry.as_dict() if geometry else None,
        })
    for instance_name, kind, region in scene._object_instance_records:
        add(instance_name, kind, "INITIAL" if region is None else region)
    return result


def _valid_edge(
    tool: dict[str, Any],
    target: dict[str, Any],
    geometry_config: dict[str, Any],
) -> dict[str, Any] | None:
    tool_spec = tool["exact_geometry"]
    target_spec = target["exact_geometry"]
    if tool["semantic_class"] != "spoon":
        return None
    if tool_spec.get("elongation_ratio") is None or tool_spec["elongation_ratio"] < float(
        geometry_config["elongated_object"]["minimum_dominant_axis_ratio"]
    ):
        return None
    if not target_spec.get("open_cavity"):
        return None
    clearance = float(geometry_config["pairwise_relations"]["clearance_margin_m"])
    grip = float(geometry_config["pairwise_relations"]["grip_allowance_m"])
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
        "semantic_rank": int(tool["semantic_rank"]),
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
    geometry_config: dict[str, Any],
) -> dict[str, Any]:
    instances = list(instances)
    coffee_targets = [
        item for item in instances
        if item["semantic_class"] == "coffee_container"
    ]
    soup_targets = [
        item for item in instances
        if item["semantic_class"] == "soup_container"
    ]
    tools = [
        item for item in instances
        if item["semantic_class"] == "spoon"
    ]
    coffee_edges = [
        edge for tool in tools for target in coffee_targets
        if (edge := _valid_edge(tool, target, geometry_config))
        is not None
    ]
    soup_edges = [
        edge for tool in tools for target in soup_targets
        if (edge := _valid_edge(tool, target, geometry_config))
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
    elif soup is None:
        distinct_tools = len({edge["tool_id"] for edge in soup_edges})
        failure = (
            "NO_COMPLETE_SOUP_MATCHING"
            if soup_cardinality < len(soup_targets) and distinct_tools >= 3
            else "INSUFFICIENT_DISTINCT_SOUP_TOOLS"
        )
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
    scene: KitchenScene | None = None,
) -> dict[str, Any]:
    benchmark = benchmark_config or load_feasibility_benchmark_config()
    variant = benchmark["variants"][variant_id]
    # ``scene_configs`` is retained for backwards-compatible callers, but the
    # oracle deliberately instantiates the exact scene model used by the run.
    del scene_configs
    scene = scene or KitchenScene(
        variant["scene_name"], include_robot=False, robot="none"
    )
    geometry_config = load_geometry_config()
    instances = _scene_instances(scene, geometry_config)
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
            geometry_config,
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
