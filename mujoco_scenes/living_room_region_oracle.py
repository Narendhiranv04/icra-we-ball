"""Privileged oracle for evaluation of the integrated living-room benchmark.

This module is deliberately not imported by production grounding.  It may use
controlled simulator names and exact instantiated geometry and marks every
artifact as evaluation-only.
"""

from __future__ import annotations

import itertools
from typing import Any

import mujoco
import numpy as np

from mujoco_scenes.living_room_region_function import variant_code


ORACLE_MARKER = "PRIVILEGED_ORACLE_EVALUATION_ONLY"


def _geom_record(scene, name: str) -> tuple[np.ndarray, np.ndarray]:
    geom_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"Oracle geom missing: {name}")
    return scene.data.geom_xpos[geom_id].copy(), scene.model.geom_size[geom_id].copy()


def _body_collision_footprint(scene, body_name: str) -> tuple[float, float]:
    """Derive an exact oracle footprint from instantiated collision geoms."""
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if body_id < 0:
        raise RuntimeError(f"Oracle payload body missing: {body_name}")
    points = []
    for geom_id, owner in enumerate(scene.model.geom_bodyid):
        if int(owner) != body_id:
            continue
        if not (
            int(scene.model.geom_contype[geom_id])
            or int(scene.model.geom_conaffinity[geom_id])
        ):
            continue
        local_center = scene.model.geom_aabb[geom_id, :3]
        half = scene.model.geom_aabb[geom_id, 3:]
        rotation = scene.data.geom_xmat[geom_id].reshape(3, 3)
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            local = local_center + half * np.asarray(signs)
            points.append(scene.data.geom_xpos[geom_id] + rotation @ local)
    if not points:
        raise RuntimeError(f"Oracle payload collision geometry missing: {body_name}")
    values = np.asarray(points)
    extents = np.ptp(values[:, :2], axis=0)
    return float(max(extents)), float(min(extents))


def _fits_pair(
    footprints: list[tuple[float, float]],
    region: tuple[float, float],
    *,
    edge: float,
    between: float,
) -> tuple[bool, float]:
    best = -float("inf")
    for rotations in itertools.product((0, 90), repeat=2):
        oriented = [
            (width, length) if angle == 90 else (length, width)
            for (length, width), angle in zip(footprints, rotations)
        ]
        for along_length in (True, False):
            required_length = (
                oriented[0][0] + oriented[1][0] + between
                if along_length else max(oriented[0][0], oriented[1][0])
            )
            required_width = (
                max(oriented[0][1], oriented[1][1])
                if along_length else oriented[0][1] + oriented[1][1] + between
            )
            best = max(
                best,
                min(
                    region[0] - required_length - 2 * edge,
                    region[1] - required_width - 2 * edge,
                ),
            )
    return best >= 0.0, best


def _variant_categories(code: str) -> dict[str, str]:
    categories = {
        "a2_personal_left_top": "side_table",
        "a2_personal_right_top": "side_table",
        "a2_shared_drink_top": "side_table",
        "a2_control_table_top": "coffee_table",
        "a2_rug_surface": "rug",
    }
    if code == "I0_PERSONAL_SEMANTIC_DEFICIT":
        categories["a2_personal_right_top"] = "coffee_table"
        categories["a2_shared_drink_top"] = "coffee_table"
    return categories


def evaluate_privileged_oracle(scene, task: dict[str, Any]) -> dict[str, Any]:
    """Compute an independent exact-geometry benchmark feasibility label."""
    code = variant_code(scene.scene_name)
    region_names = [
        "a2_personal_left_top",
        "a2_personal_right_top",
        "a2_shared_drink_top",
        "a2_control_table_top",
        "a2_rug_surface",
    ]
    categories = _variant_categories(code)
    regions = {}
    for index, name in enumerate(region_names, 1):
        center, half = _geom_record(scene, name)
        regions[f"oracle_region_{index:04d}"] = {
            "construction_geom": name,
            "category": categories[name],
            "centroid_world_m": center.tolist(),
            "support_length_m": float(2 * max(half[0], half[1])),
            "support_width_m": float(2 * min(half[0], half[1])),
        }
    seats = []
    for body_name in ("a2_seat_left", "a2_seat_right"):
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_ids = [
            geom_id for geom_id, owner in enumerate(scene.model.geom_bodyid)
            if int(owner) == body_id
        ]
        seats.append(np.median(scene.data.geom_xpos[geom_ids], axis=0))
    # Exact footprints are independently derived from the instantiated
    # collision geometry. No hand-authored payload dimension table is used.
    refreshment_footprints = [
        [
            _body_collision_footprint(scene, "a2_drink_left"),
            _body_collision_footprint(scene, "a2_snack_left"),
        ],
        [
            _body_collision_footprint(scene, "a2_drink_right"),
            _body_collision_footprint(scene, "a2_snack_right"),
        ],
    ]
    controls = [
        _body_collision_footprint(scene, "a2_remote_payload"),
        _body_collision_footprint(scene, "a2_controller_payload"),
    ]
    edge = float(
        task["geometric_requirements"]["payload_set_region"]
        ["edge_clearance_margin_m"]
    )
    between = float(
        task["geometric_requirements"]["payload_set_region"]
        ["inter_payload_clearance_m"]
    )
    near_max = float(
        task["geometric_requirements"]["personal_context"]
        ["maximum_centroid_distance_m"]
    )
    access_max = float(
        task["geometric_requirements"]["control_context"]
        ["maximum_distance_to_each_seat_m"]
    )
    personal_accepted = set(
        task["semantic_requirements"]["region_roles"]
        ["personal_refreshment_region"]["accepted_categories"]
    )
    shared_accepted = set(
        task["semantic_requirements"]["region_roles"]
        ["shared_controls_region"]["accepted_categories"]
    )
    personal_rows, shared_rows = [], []
    for slot, (seat, payloads) in enumerate(
        zip(seats, refreshment_footprints), 1
    ):
        for region_id, region in regions.items():
            dimensions = (region["support_length_m"], region["support_width_m"])
            fits, fit_margin = _fits_pair(
                payloads, dimensions, edge=edge, between=between
            )
            distance = float(
                np.linalg.norm(
                    np.asarray(region["centroid_world_m"])[:2] - seat[:2]
                )
            )
            valid = (
                region["category"] in personal_accepted
                and fits and distance <= near_max
            )
            personal_rows.append(
                {
                    "slot_id": f"personal_refreshment_slot_{slot}",
                    "region_id": region_id,
                    "valid": valid,
                    "semantic": region["category"] in personal_accepted,
                    "fits": fits,
                    "fit_margin_m": fit_margin,
                    "distance_m": distance,
                    "near_margin_m": near_max - distance,
                }
            )
    for region_id, region in regions.items():
        dimensions = (region["support_length_m"], region["support_width_m"])
        fits, fit_margin = _fits_pair(
            controls, dimensions, edge=edge, between=between
        )
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(region["centroid_world_m"])[:2] - seat[:2]
                )
            )
            for seat in seats
        ]
        valid = (
            region["category"] in shared_accepted
            and fits and max(distances) <= access_max
        )
        shared_rows.append(
            {
                "slot_id": "shared_controls_slot",
                "region_id": region_id,
                "valid": valid,
                "semantic": region["category"] in shared_accepted,
                "fits": fits,
                "fit_margin_m": fit_margin,
                "distances_to_seats_m": distances,
                "access_margin_m": access_max - max(distances),
            }
        )
    options = {
        slot: [row for row in personal_rows if row["slot_id"] == slot and row["valid"]]
        for slot in ("personal_refreshment_slot_1", "personal_refreshment_slot_2")
    }
    shared_options = [row for row in shared_rows if row["valid"]]
    solutions = []
    for first, second, shared in itertools.product(
        options["personal_refreshment_slot_1"],
        options["personal_refreshment_slot_2"],
        shared_options,
    ):
        ids = [first["region_id"], second["region_id"], shared["region_id"]]
        if len(set(ids)) == 3:
            solutions.append(ids)
    status = "COMPLETE" if solutions else "INFEASIBLE"
    return {
        "schema_version": 1,
        "artifact_classification": ORACLE_MARKER,
        "scene_name": scene.scene_name,
        "variant": code,
        "status": status,
        "regions": regions,
        "personal_compatibility": personal_rows,
        "shared_compatibility": shared_rows,
        "complete_solution_count": len(solutions),
        "example_solution": solutions[0] if solutions else None,
        "payload_geometry_source": (
            "instantiated_collision_geom_world_aabb_evaluation_only"
        ),
        "payload_footprints_m": {
            "refreshment_sets": refreshment_footprints,
            "shared_controls": controls,
        },
        "production_consumed_this_artifact": False,
    }
