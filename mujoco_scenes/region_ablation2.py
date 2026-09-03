"""Single-observation living-room region sharing and count ablation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    camera_intrinsics,
    look_at_camera_rotation,
    validate_camera_view,
    voxel_downsample,
    write_ply,
)
from mujoco_scenes.region_grounding import (
    REGION_MEASUREMENT_PURPOSE,
    PayloadMeasurementEvidence,
    RegionMeasurementEvidence,
    _depth_visual,
    _fuse_region_semantics,
    _segmentation_visual,
    _select_upper_support_plane,
    _semantic_overlap_score,
    _volume_mask_from_world_points,
    evaluate_fits_on,
    extract_payload_properties,
    extract_region_properties,
)
from mujoco_scenes.semantic_grounding import (
    NullSemanticDetector,
    SemanticDetector,
    canonicalize_detection,
    detector_vocabulary,
    load_semantic_config,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG = ROOT / "configs" / "l2_region_ablation2_task.yaml"
DEFAULT_DRINKS_TASK_CONFIG = (
    ROOT / "configs" / "l2_region_ablation2_drinks_task.yaml"
)
DEFAULT_CONTROLS_TASK_CONFIG = (
    ROOT / "configs" / "l2_region_ablation2_controls_task.yaml"
)
DEFAULT_EVALUATION_CONFIG = (
    ROOT / "configs" / "l2_region_ablation2_evaluation.yaml"
)
DEFAULT_RIG_CONFIG = ROOT / "configs" / "l2_region_ablation2_rig.yaml"
DEFAULT_SEMANTIC_VOCABULARY = (
    ROOT / "configs" / "l2_region_ablation2_semantic_vocabulary.yaml"
)
POLICIES = ("always_shared", "always_distinct", "function_aware")
USAGE_POLICIES = (
    "DEDICATED_REGION_PER_TARGET",
    "SHARED_REGION_REQUIRED",
    "SHARED_REGION_ALLOWED",
)


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tri_and(*statuses: str) -> str:
    if "FALSE" in statuses:
        return "FALSE"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "TRUE"


def _value(record: dict[str, Any], key: str) -> Any:
    value = record.get(key)
    return value.get("value") if isinstance(value, dict) else value


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_ablation2_task(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if "extends" in config:
        base = load_ablation2_task(path.parent / config.pop("extends"))
        config = _deep_merge(base, config)
    groups = config.get(
        "active_function_groups", list(config["function_groups"])
    )
    unknown = set(groups) - set(config["function_groups"])
    if unknown:
        raise ValueError(f"Unknown function groups: {sorted(unknown)}")
    for group in config["function_groups"].values():
        if group["usage_policy"] not in USAGE_POLICIES:
            raise ValueError(
                f"Unsupported usage policy: {group['usage_policy']}"
            )
    config["active_function_groups"] = list(groups)
    return config


def _free_instance_geom_groups(model: mujoco.MjModel) -> list[np.ndarray]:
    """Discover free rigid instances from topology, never simulator names."""
    roots = sorted(
        {
            int(model.jnt_bodyid[joint_id])
            for joint_id in range(model.njnt)
            if int(model.jnt_type[joint_id])
            == int(mujoco.mjtJoint.mjJNT_FREE)
        }
    )
    groups = []
    for root in roots:
        ids = []
        for geom_id, body_id in enumerate(model.geom_bodyid):
            cursor = int(body_id)
            while cursor > 0:
                if cursor == root:
                    ids.append(geom_id)
                    break
                cursor = int(model.body_parentid[cursor])
        if ids:
            groups.append(np.asarray(ids, np.int32))
    return groups


def evaluate_fits_set_on(
    payloads: Iterable[dict[str, Any]],
    region: dict[str, Any],
    *,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic non-overlapping two-rectangle packing on one region."""
    method = "measured_two_rectangle_packing_v1"
    payloads = list(payloads)
    if len(payloads) != 2:
        return {
            "relation": "FITS_SET_ON",
            "status": "UNKNOWN",
            "value": None,
            "method": method,
            "reason": "EXACTLY_TWO_PAYLOADS_REQUIRED",
        }
    try:
        rectangles = [
            (
                float(payload["footprint_length_m"]["value"]),
                float(payload["footprint_width_m"]["value"]),
            )
            for payload in payloads
        ]
        region_length = float(region["support_length_m"]["value"])
        region_width = float(region["support_width_m"]["value"])
    except (KeyError, TypeError, ValueError):
        return {
            "relation": "FITS_SET_ON",
            "status": "UNKNOWN",
            "value": None,
            "method": method,
            "reason": "MISSING_MEASUREMENT",
        }
    config = task_config["geometric_requirements"]["payload_set_region"]
    edge = float(config["edge_clearance_margin_m"])
    between = float(config["inter_payload_clearance_m"])
    orientations = [
        int(value) for value in config["allowed_orientations_degrees"]
    ]
    tested = []
    for first_angle, second_angle in itertools.product(
        orientations, orientations
    ):
        oriented = []
        for rectangle, angle in zip(
            rectangles, (first_angle, second_angle)
        ):
            oriented.append(
                (rectangle[1], rectangle[0])
                if angle % 180 == 90
                else rectangle
            )
        for arrangement in config["arrangements"]:
            if arrangement == "ALONG_LENGTH":
                required_length = (
                    oriented[0][0] + oriented[1][0] + between
                )
                required_width = max(oriented[0][1], oriented[1][1])
            elif arrangement == "ALONG_WIDTH":
                required_length = max(oriented[0][0], oriented[1][0])
                required_width = (
                    oriented[0][1] + oriented[1][1] + between
                )
            else:
                raise ValueError(f"Unknown packing arrangement: {arrangement}")
            length_margin = region_length - required_length - 2.0 * edge
            width_margin = region_width - required_width - 2.0 * edge
            tested.append(
                {
                    "arrangement": arrangement,
                    "payload_orientations_degrees": [
                        first_angle,
                        second_angle,
                    ],
                    "oriented_payload_footprints_m": [
                        list(oriented[0]),
                        list(oriented[1]),
                    ],
                    "required_length_m": required_length,
                    "required_width_m": required_width,
                    "length_margin_m": length_margin,
                    "width_margin_m": width_margin,
                    "signed_clearance_margin_m": min(
                        length_margin, width_margin
                    ),
                    "non_overlapping": True,
                    "fits": length_margin >= 0.0 and width_margin >= 0.0,
                }
            )
    selected = max(
        tested,
        key=lambda item: (
            item["signed_clearance_margin_m"],
            -item["payload_orientations_degrees"][0],
            -item["payload_orientations_degrees"][1],
            item["arrangement"],
        ),
    )
    return {
        "relation": "FITS_SET_ON",
        "status": "TRUE" if selected["fits"] else "FALSE",
        "value": bool(selected["fits"]),
        "method": method,
        "payload_footprints_m": [list(value) for value in rectangles],
        "region_usable_length_m": region_length,
        "region_usable_width_m": region_width,
        "edge_clearance_margin_m": edge,
        "inter_payload_clearance_m": between,
        "selected_packing": selected,
        "tested_packings": tested,
        "signed_clearance_margin_m": selected[
            "signed_clearance_margin_m"
        ],
    }


def evaluate_near_seat(
    region: dict[str, Any],
    seat: dict[str, Any],
    *,
    maximum_distance_m: float,
) -> dict[str, Any]:
    method = "observed_region_seat_centroid_distance_v1"
    region_centroid = _value(region, "centroid_world_m")
    seat_centroid = seat.get("centroid_world_m")
    if region_centroid is None or seat_centroid is None:
        return {
            "relation": "NEAR_SEAT",
            "status": "UNKNOWN",
            "value": None,
            "method": method,
            "reason": "MISSING_OBSERVED_CENTROID",
        }
    distance = float(
        np.linalg.norm(
            np.asarray(region_centroid, float)[:2]
            - np.asarray(seat_centroid, float)[:2]
        )
    )
    margin = maximum_distance_m - distance
    return {
        "relation": "NEAR_SEAT",
        "status": "TRUE" if margin >= 0.0 else "FALSE",
        "value": margin >= 0.0,
        "method": method,
        "measured_distance_m": distance,
        "maximum_distance_m": maximum_distance_m,
        "signed_margin_m": margin,
        "region_centroid_world_m": region_centroid,
        "seat_centroid_world_m": seat_centroid,
    }


def evaluate_control_accessibility(
    region: dict[str, Any],
    seats: Iterable[dict[str, Any]],
    *,
    maximum_distance_m: float,
) -> dict[str, Any]:
    relations = [
        evaluate_near_seat(
            region, seat, maximum_distance_m=maximum_distance_m
        )
        for seat in seats
    ]
    status = _tri_and(*(relation["status"] for relation in relations))
    return {
        "relation": "ACCESSIBLE_FROM_VIEWING_AREA",
        "status": status,
        "value": True if status == "TRUE" else False if status == "FALSE" else None,
        "method": "all_observed_seats_accessible_v1",
        "seat_relations": relations,
        "signed_margin_m": (
            min(relation["signed_margin_m"] for relation in relations)
            if relations
            and all("signed_margin_m" in relation for relation in relations)
            else None
        ),
    }


@dataclass
class InitialObservation:
    cameras: dict[str, dict[str, Any]]
    regions: dict[str, dict[str, Any]]
    payloads: dict[str, dict[str, Any]]
    seats: dict[str, dict[str, Any]]
    timings_seconds: dict[str, float]


class InitialEvidenceCapture:
    """One fixed five-view render, then many typed evidence gates."""

    def __init__(
        self,
        scene,
        *,
        rig_config: str | Path,
        task_config: dict[str, Any],
        width: int,
        height: int,
    ):
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.width = width
        self.height = height
        self.task = task_config
        with Path(rig_config).open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source)

    def capture(self, observation_dir: Path) -> InitialObservation:
        started = time.perf_counter()
        observation_dir.mkdir(parents=True, exist_ok=False)
        capture_config = self.config["capture"]
        for _ in range(int(capture_config.get("settle_steps", 0))):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        original: dict[str, dict[str, Any]] = {}
        configured: dict[str, dict[str, Any]] = {}
        for camera_id, model_name in self.config["camera_slots"].items():
            model_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, model_name
            )
            camera = capture_config["cameras"][camera_id]
            position = np.asarray(camera["position_world_m"], float)
            target = np.asarray(camera["look_at_world_m"], float)
            rotation = look_at_camera_rotation(
                position,
                target,
                np.asarray(capture_config["up_world"], float),
            )
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
            original[camera_id] = {
                "model_id": model_id,
                "pos": self.model.cam_pos[model_id].copy(),
                "quat": self.model.cam_quat[model_id].copy(),
                "mat0": self.model.cam_mat0[model_id].copy(),
                "mode": int(self.model.cam_mode[model_id]),
                "target": int(self.model.cam_targetbodyid[model_id]),
                "fovy": float(self.model.cam_fovy[model_id]),
            }
            self.model.cam_pos[model_id] = position
            self.model.cam_quat[model_id] = quaternion
            self.model.cam_mat0[model_id] = rotation.reshape(-1)
            self.model.cam_mode[model_id] = int(
                mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            )
            self.model.cam_targetbodyid[model_id] = -1
            self.model.cam_fovy[model_id] = float(camera["fovy_degrees"])
            configured[camera_id] = {
                "model_id": model_id,
                "model_name": model_name,
                "position": position,
                "target": target,
                "rotation": rotation,
            }
        mujoco.mj_forward(self.model, self.data)
        free_groups = _free_instance_geom_groups(self.model)
        raw_cameras: dict[str, dict[str, Any]] = {}
        renderer = mujoco.Renderer(
            self.model, width=self.width, height=self.height
        )
        render_started = time.perf_counter()
        try:
            for camera_id, pose in configured.items():
                renderer.update_scene(self.data, camera=pose["model_id"])
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                intrinsics = camera_intrinsics(
                    float(self.model.cam_fovy[pose["model_id"]]),
                    self.width,
                    self.height,
                )
                validation = validate_camera_view(
                    camera_position=pose["position"],
                    camera_rotation=pose["rotation"],
                    target_world=pose["target"],
                    intrinsics=intrinsics,
                    width=self.width,
                    height=self.height,
                    depth_m=depth,
                    near_depth_m=float(capture_config["near_depth_m"]),
                    far_depth_m=float(capture_config["far_depth_m"]),
                    maximum_target_angle_degrees=float(
                        self.config["view_validation"][
                            "maximum_target_angle_degrees"
                        ]
                    ),
                    minimum_valid_depth_pixels=int(
                        self.config["view_validation"][
                            "minimum_valid_depth_pixels"
                        ]
                    ),
                )
                valid_depth = (
                    np.isfinite(depth)
                    & (depth > float(capture_config["near_depth_m"]))
                    & (depth <= float(capture_config["far_depth_m"]))
                )
                world, pixels = backproject_masked_depth(
                    depth,
                    valid_depth,
                    intrinsics,
                    pose["position"],
                    pose["rotation"],
                    min_depth=float(capture_config["near_depth_m"]),
                    max_depth=float(capture_config["far_depth_m"]),
                )
                is_geom = segmentation[..., 1] == int(
                    mujoco.mjtObj.mjOBJ_GEOM
                )
                payload_masks = {
                    f"raw_{index:04d}": is_geom
                    & np.isin(segmentation[..., 0], geom_ids)
                    for index, geom_ids in enumerate(free_groups)
                }
                payload_union = (
                    np.any(np.stack(list(payload_masks.values())), axis=0)
                    if payload_masks
                    else np.zeros(depth.shape, dtype=bool)
                )
                region_masks = {}
                region_points = {}
                for selector_id, selector in self.config[
                    "region_selectors"
                ].items():
                    mask, selected = _volume_mask_from_world_points(
                        world,
                        pixels,
                        depth.shape,
                        selector["volume"],
                    )
                    # Measure the fixed support surface, not a movable object
                    # resting above it.  Instance masks come from rendered
                    # segmentation and are used only to remove payload pixels
                    # from the neutral region gate before plane extraction.
                    if self.config["processing"].get(
                        "exclude_payload_points_from_regions", False
                    ):
                        mask &= ~payload_union
                        selected = world[mask[pixels[:, 0], pixels[:, 1]]]
                    region_masks[selector_id] = mask
                    region_points[selector_id] = selected
                seat_masks = {}
                seat_points = {}
                for selector_id, selector in self.config[
                    "seating_selectors"
                ].items():
                    mask, selected = _volume_mask_from_world_points(
                        world,
                        pixels,
                        depth.shape,
                        selector["volume"],
                    )
                    seat_masks[selector_id] = mask
                    seat_points[selector_id] = selected
                payload_points = {}
                payload_pixels = {}
                for raw_id, mask in payload_masks.items():
                    points, selected_pixels = backproject_masked_depth(
                        depth,
                        mask,
                        intrinsics,
                        pose["position"],
                        pose["rotation"],
                        min_depth=float(capture_config["near_depth_m"]),
                        max_depth=float(capture_config["far_depth_m"]),
                    )
                    payload_points[raw_id] = points
                    payload_pixels[raw_id] = selected_pixels
                raw_cameras[camera_id] = {
                    "camera_id": camera_id,
                    "model_camera_name": pose["model_name"],
                    "rgb": rgb,
                    "depth_m": depth,
                    "segmentation": segmentation,
                    "intrinsics": intrinsics,
                    "position_world_m": pose["position"],
                    "rotation_world_from_camera": pose["rotation"],
                    "validation": validation,
                    "world_points": world,
                    "world_pixels": pixels,
                    "region_masks": region_masks,
                    "region_points": region_points,
                    "payload_masks": payload_masks,
                    "payload_points": payload_points,
                    "payload_pixels": payload_pixels,
                    "seat_masks": seat_masks,
                    "seat_points": seat_points,
                }
        finally:
            renderer.close()
            for state in original.values():
                model_id = state["model_id"]
                self.model.cam_pos[model_id] = state["pos"]
                self.model.cam_quat[model_id] = state["quat"]
                self.model.cam_mat0[model_id] = state["mat0"]
                self.model.cam_mode[model_id] = state["mode"]
                self.model.cam_targetbodyid[model_id] = state["target"]
                self.model.cam_fovy[model_id] = state["fovy"]
            mujoco.mj_forward(self.model, self.data)
        render_seconds = time.perf_counter() - render_started
        valid_cameras = [
            camera_id
            for camera_id, camera in raw_cameras.items()
            if camera["validation"].get("usable", False)
        ]
        if len(valid_cameras) < int(
            self.config["view_validation"]["minimum_valid_rig_cameras"]
        ):
            raise RuntimeError(
                "Initial five-view rig has insufficient usable cameras: "
                f"{valid_cameras}"
            )
        payload_id_map = self._generic_payload_ids(raw_cameras)
        region_id_map = self._generic_volume_ids(
            raw_cameras, "region_points", "region"
        )
        seat_id_map = self._generic_volume_ids(
            raw_cameras,
            "seat_points",
            self.config.get("entity_id_prefixes", {}).get(
                "seating_target", "seating"
            ),
        )
        cameras = self._rename_masks(
            raw_cameras, payload_id_map, region_id_map, seat_id_map
        )
        regions = self._build_regions(
            cameras, region_id_map, observation_dir
        )
        payloads = self._build_payloads(
            cameras, payload_id_map, observation_dir
        )
        seats = self._build_seats(cameras, seat_id_map, observation_dir)
        self._save_camera_artifacts(cameras, observation_dir)
        metadata = {
            "stage": 0,
            "observation": "INITIAL_SINGLE_MULTI_VIEW_CAPTURE",
            "capture_resolution": [self.width, self.height],
            "valid_cameras": valid_cameras,
            "camera_poses": {
                camera_id: {
                    "model_camera_name": camera["model_camera_name"],
                    "position_world_m": camera[
                        "position_world_m"
                    ].tolist(),
                    "rotation_world_from_camera": camera[
                        "rotation_world_from_camera"
                    ].tolist(),
                    "intrinsics": camera["intrinsics"].tolist(),
                    "validation": camera["validation"],
                }
                for camera_id, camera in cameras.items()
            },
            "region_selector_count": len(regions),
            "payload_instance_count": len(payloads),
            "seating_target_count": len(seats),
            "region_proposal_provenance": self.config.get(
                "region_proposal_provenance",
                {
                    "region_proposal_source": "CONFIGURED_SPATIAL_GATE",
                    "region_proposal_encodes_function": None,
                    "region_proposal_encodes_semantic_class": None,
                    "region_proposal_encodes_expected_validity": None,
                    "region_dimensions_for_functional_reasoning": (
                        "OBSERVED_RGBD_POINT_CLOUD"
                    ),
                },
            ),
            "region_selectors": {
                selector_id: {
                    "volume": selector["volume"],
                    "purpose": "EVIDENCE_SELECTION_ONLY_NOT_MEASUREMENT",
                }
                for selector_id, selector in self.config[
                    "region_selectors"
                ].items()
            },
            "seating_selectors": {
                selector_id: {
                    "volume": selector["volume"],
                    "purpose": "EVIDENCE_SELECTION_ONLY_NOT_MEASUREMENT",
                }
                for selector_id, selector in self.config[
                    "seating_selectors"
                ].items()
            },
        }
        _atomic_json(observation_dir / "inspection_metadata.json", metadata)
        self._save_overviews(cameras, observation_dir)
        return InitialObservation(
            cameras=cameras,
            regions=regions,
            payloads=payloads,
            seats=seats,
            timings_seconds={
                "capture_and_reconstruction": render_seconds,
                "total": time.perf_counter() - started,
            },
        )

    @staticmethod
    def _generic_payload_ids(
        cameras: dict[str, dict[str, Any]]
    ) -> dict[str, str]:
        raw_ids = next(iter(cameras.values()))["payload_masks"]
        ordered = []
        for raw_id in raw_ids:
            points = [
                camera["payload_points"][raw_id]
                for camera in cameras.values()
                if len(camera["payload_points"][raw_id])
            ]
            centroid = (
                np.median(np.concatenate(points), axis=0)
                if points
                else np.full(3, np.inf)
            )
            ordered.append((float(centroid[0]), float(centroid[1]), raw_id))
        return {
            raw_id: f"object_{index:04d}"
            for index, (_x, _y, raw_id) in enumerate(sorted(ordered), 1)
        }

    @staticmethod
    def _generic_volume_ids(
        cameras: dict[str, dict[str, Any]],
        points_key: str,
        prefix: str,
    ) -> dict[str, str]:
        selector_ids = next(iter(cameras.values()))[points_key]
        ordered = []
        for selector_id in selector_ids:
            points = [
                camera[points_key][selector_id]
                for camera in cameras.values()
                if len(camera[points_key][selector_id])
            ]
            centroid = (
                np.median(np.concatenate(points), axis=0)
                if points
                else np.full(3, np.inf)
            )
            ordered.append(
                (
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                    selector_id,
                )
            )
        return {
            selector_id: f"{prefix}_{index:04d}"
            for index, (*_values, selector_id) in enumerate(sorted(ordered), 1)
        }

    @staticmethod
    def _rename_masks(
        cameras: dict[str, dict[str, Any]],
        payload_map: dict[str, str],
        region_map: dict[str, str],
        seat_map: dict[str, str],
    ) -> dict[str, dict[str, Any]]:
        for camera in cameras.values():
            for key, mapping in (
                ("payload_masks", payload_map),
                ("payload_points", payload_map),
                ("payload_pixels", payload_map),
                ("region_masks", region_map),
                ("region_points", region_map),
                ("seat_masks", seat_map),
                ("seat_points", seat_map),
            ):
                camera[key] = {
                    mapping[raw_id]: value
                    for raw_id, value in camera[key].items()
                }
        return cameras

    def _build_regions(
        self,
        cameras: dict[str, dict[str, Any]],
        id_map: dict[str, str],
        observation_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        processing = self.config["processing"]
        records = {}
        selector_by_id = {generic: raw for raw, generic in id_map.items()}
        for region_id, selector_id in selector_by_id.items():
            points, colors, by_camera = [], [], {}
            for camera_id, camera in cameras.items():
                selected = camera["region_points"][region_id]
                by_camera[camera_id] = selected
                if len(selected):
                    pixels = np.column_stack(
                        np.nonzero(camera["region_masks"][region_id])
                    )
                    points.append(selected)
                    colors.append(
                        camera["rgb"][pixels[:, 0], pixels[:, 1]]
                    )
            raw_points = np.concatenate(points) if points else np.empty((0, 3))
            raw_colors = (
                np.concatenate(colors)
                if colors
                else np.empty((0, 3), np.uint8)
            )
            plane_points, plane_colors, diagnostics = (
                _select_upper_support_plane(
                    raw_points,
                    raw_colors,
                    plane_band_m=float(processing["plane_band_m"]),
                    voxel_size_m=float(processing["voxel_size_m"]),
                )
            )
            contributing = tuple(
                camera_id
                for camera_id, selected in by_camera.items()
                if len(selected)
                and cameras[camera_id]["validation"].get("usable", False)
            )
            quality = {
                "quality_is_valid": (
                    len(plane_points)
                    >= int(processing["minimum_region_points"])
                    and len(contributing)
                    >= int(
                        self.config["view_validation"][
                            "minimum_entity_camera_count"
                        ]
                    )
                ),
                "point_count": len(plane_points),
                "contributing_camera_count": len(contributing),
                "cloud_purpose": REGION_MEASUREMENT_PURPOSE,
                **diagnostics,
            }
            path = f"observation/regions/{region_id}/fused.ply"
            evidence = RegionMeasurementEvidence(
                measurement_points=plane_points,
                measurement_colors=plane_colors,
                points_by_camera=by_camera,
                source_stage=0,
                inspection_label=selector_id,
                measurement_cloud_path=path,
                contributing_camera_ids=contributing,
                measurement_quality=quality,
            )
            directory = observation_dir / "regions" / region_id
            write_ply(directory / "fused.ply", plane_points, plane_colors)
            _atomic_json(directory / "quality.json", quality)
            records[region_id] = {
                "region_id": region_id,
                "selector_id": selector_id,
                "candidate_rank": int(
                    self.config["region_selectors"][selector_id][
                        "candidate_rank"
                    ]
                ),
                "evidence": evidence,
                "quality": quality,
                "evidence_path": path,
            }
        return records

    def _build_payloads(
        self,
        cameras: dict[str, dict[str, Any]],
        id_map: dict[str, str],
        observation_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        processing = self.config["processing"]
        records = {}
        for object_id in sorted(id_map.values()):
            points, colors, by_camera = [], [], {}
            for camera_id, camera in cameras.items():
                selected = camera["payload_points"][object_id]
                by_camera[camera_id] = selected
                if len(selected):
                    pixels = camera["payload_pixels"][object_id]
                    points.append(selected)
                    colors.append(
                        camera["rgb"][pixels[:, 0], pixels[:, 1]]
                    )
            if points:
                raw_points = np.concatenate(points)
                raw_colors = np.concatenate(colors)
                median = np.median(raw_points, axis=0)
                primary = (
                    np.linalg.norm(raw_points - median, axis=1)
                    <= float(
                        processing["payload_primary_component_radius_m"]
                    )
                )
                fused_points, fused_colors = voxel_downsample(
                    raw_points[primary],
                    raw_colors[primary],
                    float(processing["voxel_size_m"]),
                )
            else:
                fused_points, fused_colors = (
                    np.empty((0, 3)),
                    np.empty((0, 3), np.uint8),
                )
            contributing = tuple(
                camera_id
                for camera_id, selected in by_camera.items()
                if len(selected)
                and cameras[camera_id]["validation"].get("usable", False)
            )
            quality = {
                "quality_is_valid": (
                    len(fused_points)
                    >= int(processing["minimum_payload_points"])
                    and len(contributing)
                    >= int(
                        self.config["view_validation"][
                            "minimum_entity_camera_count"
                        ]
                    )
                ),
                "point_count": len(fused_points),
                "contributing_camera_count": len(contributing),
                "cloud_purpose": "PAYLOAD_MEASUREMENT_EVIDENCE",
            }
            path = f"observation/payloads/{object_id}/fused.ply"
            evidence = PayloadMeasurementEvidence(
                measurement_points=fused_points,
                measurement_colors=fused_colors,
                points_by_camera=by_camera,
                source_stage=0,
                measurement_cloud_path=path,
                contributing_camera_ids=contributing,
                measurement_quality=quality,
            )
            directory = observation_dir / "payloads" / object_id
            write_ply(directory / "fused.ply", fused_points, fused_colors)
            _atomic_json(directory / "quality.json", quality)
            records[object_id] = {
                "object_id": object_id,
                "evidence": evidence,
                "quality": quality,
                "evidence_path": path,
            }
        return records

    def _build_seats(
        self,
        cameras: dict[str, dict[str, Any]],
        id_map: dict[str, str],
        observation_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        processing = self.config["processing"]
        records = {}
        selector_by_id = {generic: raw for raw, generic in id_map.items()}
        for seat_id, selector_id in selector_by_id.items():
            points = [
                camera["seat_points"][seat_id]
                for camera in cameras.values()
                if len(camera["seat_points"][seat_id])
            ]
            fused = (
                voxel_downsample(
                    np.concatenate(points),
                    np.full(
                        (sum(len(value) for value in points), 3),
                        (45, 120, 210),
                        np.uint8,
                    ),
                    float(processing["voxel_size_m"]),
                )[0]
                if points
                else np.empty((0, 3))
            )
            contributing = [
                camera_id
                for camera_id, camera in cameras.items()
                if len(camera["seat_points"][seat_id])
            ]
            quality_valid = (
                len(fused) >= int(processing["minimum_seating_points"])
                and len(contributing)
                >= int(
                    self.config["view_validation"][
                        "minimum_entity_camera_count"
                    ]
                )
            )
            centroid = (
                np.median(fused, axis=0).tolist()
                if quality_valid
                else None
            )
            path = f"observation/seats/{seat_id}/observed_points.ply"
            write_ply(
                observation_dir / "seats" / seat_id / "observed_points.ply",
                fused,
                np.full((len(fused), 3), (45, 120, 210), np.uint8),
            )
            records[seat_id] = {
                "seating_target_id": seat_id,
                "selector_id": selector_id,
                "centroid_world_m": centroid,
                "point_count": len(fused),
                "contributing_camera_ids": contributing,
                "quality_is_valid": quality_valid,
                "evidence_path": path,
            }
        return records

    def _save_camera_artifacts(
        self,
        cameras: dict[str, dict[str, Any]],
        observation_dir: Path,
    ) -> None:
        colors = [
            (38, 180, 95),
            (52, 110, 210),
            (224, 110, 38),
            (150, 70, 190),
            (225, 195, 45),
        ]
        for camera_id, camera in cameras.items():
            directory = observation_dir / "cameras" / camera_id
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(camera["rgb"]).save(directory / "rgb.png")
            _depth_visual(camera["depth_m"]).save(directory / "depth.png")
            _segmentation_visual(camera["segmentation"]).save(
                directory / "segmentation.png"
            )
            mask_image = np.zeros(
                (*camera["depth_m"].shape, 3), np.uint8
            )
            for index, mask in enumerate(camera["region_masks"].values()):
                mask_image[mask] = colors[index % len(colors)]
            for mask in camera["payload_masks"].values():
                mask_image[mask] = (245, 95, 35)
            for mask in camera["seat_masks"].values():
                mask_image[mask] = (35, 135, 225)
            Image.fromarray(mask_image).save(
                directory / "evidence_masks.png"
            )

    @staticmethod
    def _save_overviews(
        cameras: dict[str, dict[str, Any]], observation_dir: Path
    ) -> None:
        for filename, key in (
            ("initial_scene_overview.png", "rgb"),
            ("region_masks_overview.png", None),
        ):
            images = []
            for camera_id, camera in cameras.items():
                path = (
                    observation_dir
                    / "cameras"
                    / camera_id
                    / (
                        "rgb.png"
                        if key == "rgb"
                        else "evidence_masks.png"
                    )
                )
                images.append((camera_id, Image.open(path).convert("RGB")))
            width = max(image.width for _name, image in images)
            height = max(image.height for _name, image in images)
            canvas = Image.new("RGB", (2 * width, 3 * (height + 30)), "white")
            draw = ImageDraw.Draw(canvas)
            for index, (name, image) in enumerate(images):
                x = index % 2 * width
                y = index // 2 * (height + 30)
                draw.text((x + 5, y + 5), name, fill="black", font=_font(16, True))
                canvas.paste(image, (x, y + 30))
            canvas.save(observation_dir / filename)


def _entity_allowed_labels(
    entity_type: str, task: dict[str, Any]
) -> set[str]:
    semantic = task["semantic_requirements"]
    if entity_type == "region":
        labels = set()
        for role in semantic["region_roles"].values():
            labels.update(role["accepted_categories"])
            labels.update(role["rejected_categories"])
        return labels
    if entity_type == "payload":
        return {
            label
            for labels in semantic["payload_roles"].values()
            for label in labels
        }
    return set(semantic["seating_categories"])


def run_initial_semantics(
    observation: InitialObservation,
    *,
    detector: SemanticDetector,
    semantic_config: dict[str, Any],
    task_config: dict[str, Any],
    observation_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the RGB detector once per camera and associate every entity."""
    vocabulary = detector_vocabulary(semantic_config)
    entity_types = {
        **{entity_id: "region" for entity_id in observation.regions},
        **{entity_id: "payload" for entity_id in observation.payloads},
        **{entity_id: "seat" for entity_id in observation.seats},
    }
    observations = {entity_id: [] for entity_id in entity_types}
    all_detections = []
    all_associations = []
    inference_times = {}
    overlay_images = []
    colors = {
        "region": (35, 170, 85),
        "payload": (230, 95, 35),
        "seat": (45, 105, 210),
    }
    minimum_score = {"region": 0.05, "payload": 0.07, "seat": 0.035}
    for camera_id, camera in observation.cameras.items():
        if not camera["validation"].get("usable", False):
            continue
        started = time.perf_counter()
        raw = detector.detect(camera["rgb"], vocabulary)
        inference_times[camera_id] = time.perf_counter() - started
        detections = [
            canonicalize_detection(detection, semantic_config)
            for detection in raw
        ]
        masks = {
            **camera["region_masks"],
            **camera["payload_masks"],
            **camera["seat_masks"],
        }
        candidates = []
        for detection_index, detection in enumerate(detections):
            for entity_id, entity_type in entity_types.items():
                if detection.canonical_label not in _entity_allowed_labels(
                    entity_type, task_config
                ):
                    continue
                metrics = _semantic_overlap_score(
                    detection, masks[entity_id]
                )
                if (
                    metrics["intersection_pixels"] < 8
                    or metrics["score"] < minimum_score[entity_type]
                ):
                    continue
                candidates.append(
                    {
                        "detection_index": detection_index,
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                        "score": metrics["score"],
                        "matching_score": (
                            float(detection.confidence) * metrics["score"]
                        ),
                        "metrics": metrics,
                    }
                )
        candidates.sort(
            key=lambda item: (
                -item["matching_score"],
                -item["score"],
                item["detection_index"],
                item["entity_id"],
            )
        )
        used_detections, used_entities = set(), set()
        accepted = []
        for candidate in candidates:
            if (
                candidate["detection_index"] in used_detections
                or candidate["entity_id"] in used_entities
            ):
                continue
            used_detections.add(candidate["detection_index"])
            used_entities.add(candidate["entity_id"])
            detection = detections[candidate["detection_index"]]
            record = {
                "entity_id": candidate["entity_id"],
                "entity_type": candidate["entity_type"],
                "detection": {
                    **detection.to_dict(),
                    "source_camera": camera_id,
                },
                "association_metrics": candidate["metrics"],
                "weighted_score": candidate["matching_score"],
            }
            observations[candidate["entity_id"]].append(record)
            accepted.append(record)
        all_detections.extend(
            {"camera_id": camera_id, **detection.to_dict()}
            for detection in detections
        )
        all_associations.append(
            {
                "camera_id": camera_id,
                "accepted": accepted,
                "unmatched_detection_indices": [
                    index
                    for index in range(len(detections))
                    if index not in used_detections
                ],
                "unmatched_entity_ids": [
                    entity_id
                    for entity_id in sorted(entity_types)
                    if entity_id not in used_entities
                ],
            }
        )
        image = Image.fromarray(camera["rgb"])
        draw = ImageDraw.Draw(image)
        for record in accepted:
            detection = record["detection"]
            color = colors[record["entity_type"]]
            draw.rectangle(detection["bbox_xyxy"], outline=color, width=4)
            x1, y1, _x2, _y2 = detection["bbox_xyxy"]
            draw.text(
                (x1 + 3, max(2, y1 - 20)),
                f"{record['entity_id']} {detection['raw_label']} "
                f"{detection['confidence']:.2f}",
                fill=color,
                font=_font(14, True),
            )
        directory = observation_dir / "semantics" / "cameras" / camera_id
        directory.mkdir(parents=True, exist_ok=True)
        image.save(directory / "association_overlay.png")
        overlay_images.append((camera_id, image))
    requirements = task_config["semantic_requirements"]
    records = {}
    for entity_id, values in observations.items():
        record = _fuse_region_semantics(
            values,
            minimum_views=int(requirements["minimum_supporting_views"]),
            minimum_confidence=float(requirements["minimum_mean_confidence"]),
            minimum_margin=float(
                requirements["minimum_winning_score_margin"]
            ),
        )
        record["entity_id"] = entity_id
        record["entity_type"] = entity_types[entity_id]
        record["observations"] = values
        records[entity_id] = record
    semantics_dir = observation_dir / "semantics"
    _atomic_json(
        semantics_dir / "detections.json", {"detections": all_detections}
    )
    _atomic_json(
        semantics_dir / "associations.json",
        {"cameras": all_associations},
    )
    _atomic_json(semantics_dir / "fused_semantics.json", records)
    if overlay_images:
        width = max(image.width for _name, image in overlay_images)
        height = max(image.height for _name, image in overlay_images)
        canvas = Image.new("RGB", (2 * width, 3 * (height + 30)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (name, image) in enumerate(overlay_images):
            x = index % 2 * width
            y = index // 2 * (height + 30)
            draw.text((x + 5, y + 5), name, fill="black", font=_font(16, True))
            canvas.paste(image, (x, y + 30))
        canvas.save(observation_dir / "semantic_overview.png")
    timing = {
        "per_camera_seconds": inference_times,
        "total_seconds": sum(inference_times.values()),
        "camera_count": len(inference_times),
    }
    _atomic_json(semantics_dir / "timings.json", timing)
    return records, timing


def _semantic_role(
    record: dict[str, Any],
    *,
    accepted: dict[str, Any] | list[str],
    rejected: dict[str, Any] | list[str] = (),
) -> dict[str, Any]:
    if record.get("status") != "SUPPORTED":
        return {
            "status": "UNKNOWN",
            "value": None,
            "reason": "INSUFFICIENT_RGB_SEMANTICS",
        }
    label = record["canonical_label"]
    accepted_labels = set(accepted)
    rejected_labels = set(rejected)
    if label in accepted_labels:
        rank = accepted.get(label) if isinstance(accepted, dict) else None
        return {
            "status": "TRUE",
            "value": True,
            "canonical_label": label,
            "semantic_rank": rank,
        }
    if label in rejected_labels:
        return {
            "status": "FALSE",
            "value": False,
            "canonical_label": label,
            "reason": "EXCLUDED_RGB_CATEGORY",
        }
    return {
        "status": "UNKNOWN",
        "value": None,
        "canonical_label": label,
        "reason": "UNCONFIGURED_RGB_CATEGORY",
    }


class RegionAllocationSolver:
    """General group-level shared/distinct region assignment solver."""

    def __init__(
        self,
        *,
        drink_rows: list[dict[str, Any]],
        control_rows: list[dict[str, Any]],
        control_individual_rows: list[dict[str, Any]],
        task_config: dict[str, Any],
    ):
        self.drink_rows = drink_rows
        self.control_rows = control_rows
        self.control_individual_rows = control_individual_rows
        self.task = task_config

    def solve(self, policy: str) -> dict[str, Any]:
        if policy not in POLICIES:
            raise ValueError(f"Unknown allocation policy: {policy}")
        active = set(self.task["active_function_groups"])
        group_options: dict[str, list[dict[str, Any]]] = {}
        if "personal_drinks" in active:
            group_options["personal_drinks"] = self._drink_options(policy)
        if "shared_controls" in active:
            group_options["shared_controls"] = self._control_options(policy)
        if any(not options for options in group_options.values()):
            return {
                "policy": policy,
                "status": "EXHAUSTED",
                "assignments": [],
                "distinct_region_ids": [],
                "distinct_physical_region_count": None,
                "failed_constraints": [
                    f"NO_VALID_{group.upper()}_ASSIGNMENT"
                    for group, options in group_options.items()
                    if not options
                ],
                "alternative_valid_assignments": [],
            }
        combinations = []
        keys = list(group_options)
        for options in itertools.product(
            *(group_options[key] for key in keys)
        ):
            selected = dict(zip(keys, options))
            region_sets = {
                key: set(option["region_ids"])
                for key, option in selected.items()
            }
            if (
                not self.task["allow_cross_function_region_sharing"]
                and len(region_sets) > 1
                and any(
                    first & second
                    for first, second in itertools.combinations(
                        region_sets.values(), 2
                    )
                )
            ):
                continue
            assignments = [
                assignment
                for option in selected.values()
                for assignment in option["assignments"]
            ]
            distinct = sorted(
                {
                    assignment["region_id"]
                    for assignment in assignments
                }
            )
            rank = sum(
                int(assignment.get("candidate_rank", 999))
                for assignment in assignments
            )
            margin = sum(
                float(assignment.get("signed_margin_m") or 0.0)
                for assignment in assignments
            )
            combinations.append(
                {
                    "policy": policy,
                    "status": "COMPLETE",
                    "assignments": assignments,
                    "sharing_groups": [
                        option.get("sharing_group")
                        for option in selected.values()
                        if option.get("sharing_group")
                    ],
                    "distinctness_groups": [
                        option.get("distinctness_group")
                        for option in selected.values()
                        if option.get("distinctness_group")
                    ],
                    "distinct_region_ids": distinct,
                    "distinct_physical_region_count": len(distinct),
                    "failed_constraints": [],
                    "ranking_key": [rank, -margin, distinct],
                }
            )
        if not combinations:
            return {
                "policy": policy,
                "status": "EXHAUSTED",
                "assignments": [],
                "distinct_region_ids": [],
                "distinct_physical_region_count": None,
                "failed_constraints": [
                    "CROSS_FUNCTION_REGION_SHARING_CONFLICT"
                ],
                "alternative_valid_assignments": [],
            }
        combinations.sort(key=lambda item: item["ranking_key"])
        winner = combinations[0]
        winner["alternative_valid_assignments"] = [
            {
                "assignments": item["assignments"],
                "distinct_region_ids": item["distinct_region_ids"],
                "ranking_key": item["ranking_key"],
            }
            for item in combinations[1:6]
        ]
        return winner

    def _drink_options(self, policy: str) -> list[dict[str, Any]]:
        usage_policy = self._effective_usage_policy(
            policy, "personal_drinks"
        )
        slots = sorted({row["slot_id"] for row in self.drink_rows})
        required_slot_count = int(
            self.task["function_groups"]["personal_drinks"].get(
                "required_target_count", 2
            )
        )
        if len(slots) != required_slot_count:
            return []
        candidates = {
            slot: [
                row
                for row in self.drink_rows
                if row["slot_id"] == slot
                and row["compatibility_status"] == "TRUE"
            ]
            for slot in slots
        }
        options = []
        for chosen in itertools.product(*(candidates[slot] for slot in slots)):
            regions = [row["region_id"] for row in chosen]
            if (
                usage_policy == "SHARED_REGION_REQUIRED"
                and len(set(regions)) != 1
            ):
                continue
            if (
                usage_policy == "DEDICATED_REGION_PER_TARGET"
                and len(set(regions)) != len(regions)
            ):
                continue
            assignments = [
                {
                    "function_group": "personal_drinks",
                    "function_id": "PLACE_PERSONAL_DRINK",
                    "slot_id": row["slot_id"],
                    "payload_id": row["payload_id"],
                    "payload_ids": row.get(
                        "payload_ids", [row["payload_id"]]
                    ),
                    "target_id": row["seating_target_id"],
                    "region_id": row["region_id"],
                    "candidate_rank": row["candidate_rank"],
                    "signed_margin_m": min(
                        row["fit_margin_m"],
                        row["near_seat_margin_m"],
                    ),
                }
                for row in chosen
            ]
            options.append(
                {
                    "assignments": assignments,
                    "region_ids": regions,
                    "sharing_group": (
                        {
                            "group_id": "personal_drinks",
                            "region_id": regions[0],
                            "payload_ids": [
                                payload_id
                                for row in chosen
                                for payload_id in row.get(
                                    "payload_ids", [row["payload_id"]]
                                )
                            ],
                        }
                        if len(set(regions)) == 1
                        else None
                    ),
                    "distinctness_group": (
                        {
                            "group_id": "personal_drinks",
                            "region_ids": regions,
                        }
                        if len(set(regions)) == len(regions)
                        else None
                    ),
                }
            )
        return options

    def _control_options(self, policy: str) -> list[dict[str, Any]]:
        usage_policy = self._effective_usage_policy(
            policy, "shared_controls"
        )
        options = []
        if usage_policy in {
            "SHARED_REGION_REQUIRED",
            "SHARED_REGION_ALLOWED",
        }:
            options.extend(
                {
                    "assignments": [
                        {
                            "function_group": "shared_controls",
                            "function_id": "PLACE_SHARED_CONTROLS",
                            "slot_id": payload_id,
                            "payload_id": payload_id,
                            "target_id": "common_viewing_area",
                            "region_id": row["region_id"],
                            "candidate_rank": row["candidate_rank"],
                            "signed_margin_m": min(
                                row["packing_margin_m"],
                                row["accessibility_margin_m"],
                            ),
                        }
                        for payload_id in row["payload_ids"]
                    ],
                    "region_ids": [row["region_id"], row["region_id"]],
                    "sharing_group": {
                        "group_id": "shared_controls",
                        "region_id": row["region_id"],
                        "payload_ids": row["payload_ids"],
                    },
                    "distinctness_group": None,
                }
                for row in self.control_rows
                if row["compatibility_status"] == "TRUE"
            )
        if usage_policy == "SHARED_REGION_REQUIRED":
            return options
        by_payload: dict[str, list[dict[str, Any]]] = {}
        for row in self.control_individual_rows:
            if row["compatibility_status"] == "TRUE":
                by_payload.setdefault(row["payload_id"], []).append(row)
        if len(by_payload) != 2:
            return options
        payload_ids = sorted(by_payload)
        for first, second in itertools.product(
            by_payload[payload_ids[0]], by_payload[payload_ids[1]]
        ):
            if first["region_id"] == second["region_id"]:
                continue
            chosen = [first, second]
            options.append(
                {
                    "assignments": [
                        {
                            "function_group": "shared_controls",
                            "function_id": "PLACE_SHARED_CONTROLS",
                            "slot_id": row["payload_id"],
                            "payload_id": row["payload_id"],
                            "target_id": "common_viewing_area",
                            "region_id": row["region_id"],
                            "candidate_rank": row["candidate_rank"],
                            "signed_margin_m": min(
                                row["fit_margin_m"],
                                row["accessibility_margin_m"],
                            ),
                        }
                        for row in chosen
                    ],
                    "region_ids": [
                        first["region_id"],
                        second["region_id"],
                    ],
                    "sharing_group": None,
                    "distinctness_group": {
                        "group_id": "shared_controls",
                        "region_ids": [
                            first["region_id"],
                            second["region_id"],
                        ],
                    },
                }
            )
        return options

    def _effective_usage_policy(
        self, diagnostic_policy: str, function_group: str
    ) -> str:
        if diagnostic_policy == "always_shared":
            return "SHARED_REGION_REQUIRED"
        if diagnostic_policy == "always_distinct":
            return "DEDICATED_REGION_PER_TARGET"
        return self.task["function_groups"][function_group]["usage_policy"]


class RegionAblation2Run:
    """Authoritative one-observation allocation run and policy diagnostics."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        scene_name: str,
        task_config: str | Path = DEFAULT_TASK_CONFIG,
        evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
        rig_config: str | Path = DEFAULT_RIG_CONFIG,
        semantic_detector: SemanticDetector | None = None,
        semantic_config: dict[str, Any] | None = None,
        width: int = 1280,
        height: int = 960,
    ):
        self.run_dir = Path(run_dir).resolve()
        if self.run_dir.exists():
            raise RuntimeError(f"Run directory already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        self.scene_name = scene_name
        self.task = load_ablation2_task(task_config)
        with Path(evaluation_config).open(encoding="utf-8") as source:
            self.evaluation = yaml.safe_load(source)
        self.rig_config = Path(rig_config)
        with self.rig_config.open(encoding="utf-8") as source:
            self.rig_definition = yaml.safe_load(source)
        self.detector = semantic_detector or NullSemanticDetector()
        self.semantic_config = semantic_config or load_semantic_config(
            vocabulary_path=DEFAULT_SEMANTIC_VOCABULARY
        )
        self.width = width
        self.height = height
        self.events_path = self.run_dir / "events.jsonl"
        self.observation: InitialObservation | None = None
        self.region_registry: dict[str, Any] = {}
        self.payload_registry: dict[str, Any] = {}
        self.seating_registry: dict[str, Any] = {}
        self.drink_rows: list[dict[str, Any]] = []
        self.control_rows: list[dict[str, Any]] = []
        self.control_individual_rows: list[dict[str, Any]] = []
        self.policy_evaluations: dict[str, Any] = {}
        self._write_configs()

    def _write_configs(self) -> None:
        detector = {
            "name": getattr(self.detector, "name", type(self.detector).__name__),
            "checkpoint": getattr(self.detector, "checkpoint", None),
            "version": getattr(self.detector, "version", None),
            "device": getattr(self.detector, "device", None),
            "inference_size": getattr(self.detector, "inference_size", None),
            "process_isolation": getattr(
                self.detector, "process_isolation", None
            ),
        }
        _atomic_json(
            self.run_dir / "run_config.json",
            {
                "schema_version": 1,
                "scene_name": self.scene_name,
                "task_id": self.task["task_id"],
                "natural_language_goal": self.task["natural_language_goal"],
                "active_function_groups": self.task[
                    "active_function_groups"
                ],
                "policy_modes": list(POLICIES),
                "production_policy": "function_aware",
                "single_initial_observation": True,
                "uses_robot": False,
                "uses_foundation_model": False,
                "uses_tamp": False,
                "uses_placement_execution": False,
                "capture_resolution": [self.width, self.height],
                "detector": detector,
                "created_at": datetime.now().astimezone().isoformat(),
            },
        )
        _atomic_json(self.run_dir / "task_requirements.json", self.task)

    def event(self, event: str, **payload: Any) -> None:
        with self.events_path.open("a", encoding="utf-8") as target:
            target.write(
                json.dumps(
                    {
                        "stage": 0,
                        "observation": "INITIAL",
                        "event": event,
                        **payload,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def run(self, scene) -> "RegionAblation2Run":
        observation_dir = self.run_dir / "observation"
        self.observation = InitialEvidenceCapture(
            scene,
            rig_config=self.rig_config,
            task_config=self.task,
            width=self.width,
            height=self.height,
        ).capture(observation_dir)
        for object_id, record in self.observation.payloads.items():
            self.event(
                "PAYLOAD_OBSERVED",
                payload_id=object_id,
                evidence_path=record["evidence_path"],
                point_count=record["quality"]["point_count"],
            )
        for region_id, record in self.observation.regions.items():
            self.event(
                "REGION_OBSERVED",
                region_id=region_id,
                evidence_path=record["evidence_path"],
                point_count=record["quality"]["point_count"],
            )
        semantics, semantic_timing = run_initial_semantics(
            self.observation,
            detector=self.detector,
            semantic_config=self.semantic_config,
            task_config=self.task,
            observation_dir=observation_dir,
        )
        self._build_registries(semantics)
        self._build_compatibility()
        self.policy_evaluations = self._evaluate_policies()
        self._persist(semantic_timing)
        return self

    def _build_registries(
        self, semantics: dict[str, dict[str, Any]]
    ) -> None:
        assert self.observation is not None
        for region_id, record in self.observation.regions.items():
            properties = extract_region_properties(
                record["evidence"], task_config=self.task
            )
            self.region_registry[region_id] = {
                "identity": {
                    "region_id": region_id,
                    "entity_type": "destination_region",
                    "first_seen_stage": 0,
                    "observation_count": 1,
                },
                "candidate_rank": record["candidate_rank"],
                "geometry": properties,
                "semantics": semantics[region_id],
                "provenance": {
                    "measurement_cloud_path": record["evidence_path"],
                    "measurement_purpose": REGION_MEASUREMENT_PURPOSE,
                    "point_count": record["quality"]["point_count"],
                    "contributing_camera_ids": list(
                        record["evidence"].contributing_camera_ids
                    ),
                    **self.rig_definition.get(
                        "region_proposal_provenance", {}
                    ),
                },
            }
        role_categories = self.task["semantic_requirements"]["payload_roles"]
        for object_id, record in self.observation.payloads.items():
            properties = extract_payload_properties(record["evidence"])
            measurement_points = record["evidence"].measurement_points
            observed_centroid = (
                np.median(measurement_points, axis=0).tolist()
                if len(measurement_points)
                else None
            )
            semantic = semantics[object_id]
            role = None
            if semantic.get("status") == "SUPPORTED":
                label = semantic["canonical_label"]
                role = next(
                    (
                        name
                        for name, labels in role_categories.items()
                        if label in labels
                    ),
                    None,
                )
            self.payload_registry[object_id] = {
                "identity": {
                    "object_id": object_id,
                    "entity_type": "fixed_payload",
                    "first_seen_stage": 0,
                    "observation_count": 1,
                },
                "geometry": properties,
                "observed_centroid_world_m": observed_centroid,
                "semantics": semantic,
                "semantic_payload_role": role,
                "provenance": {
                    "measurement_cloud_path": record["evidence_path"],
                    "measurement_purpose": "PAYLOAD_MEASUREMENT_EVIDENCE",
                    "point_count": record["quality"]["point_count"],
                    "contributing_camera_ids": list(
                        record["evidence"].contributing_camera_ids
                    ),
                },
            }
        for seat_id, record in self.observation.seats.items():
            semantic = semantics[seat_id]
            semantic_status = _semantic_role(
                semantic,
                accepted=self.task["semantic_requirements"][
                    "seating_categories"
                ],
            )
            self.seating_registry[seat_id] = {
                "identity": {
                    "seating_target_id": seat_id,
                    "entity_type": "seating_target",
                    "first_seen_stage": 0,
                },
                "centroid_world_m": record["centroid_world_m"],
                "point_count": record["point_count"],
                "quality_is_valid": record["quality_is_valid"],
                "semantics": semantic,
                "semantic_role": semantic_status,
                "provenance": {
                    "evidence_path": record["evidence_path"],
                    "contributing_camera_ids": record[
                        "contributing_camera_ids"
                    ],
                },
            }

    def _required_payloads(self) -> dict[str, list[str]]:
        result = {
            role: []
            for role in self.task["semantic_requirements"]["payload_roles"]
        }
        for object_id, record in self.payload_registry.items():
            role = record["semantic_payload_role"]
            if role in result:
                result[role].append(object_id)
        for values in result.values():
            values.sort()
        return result

    @staticmethod
    def _planar_status(properties: dict[str, Any]) -> str:
        value = _value(properties, "PLANAR_SUPPORT")
        return "UNKNOWN" if value is None else "TRUE" if value else "FALSE"

    def _build_compatibility(self) -> None:
        payloads = self._required_payloads()
        seats = sorted(self.seating_registry)
        personal_role = self.task["semantic_requirements"]["region_roles"][
            "personal_drink_region"
        ]
        control_role = self.task["semantic_requirements"]["region_roles"][
            "shared_control_region"
        ]
        personal_bundles = list(
            zip(
                payloads.get("drink", [])[:2],
                payloads.get("snack_container", [])[:2],
            )
        )
        if len(personal_bundles) >= 2 and len(seats) >= 2:
            for slot_index, (bundle_ids, seat_id) in enumerate(
                zip(personal_bundles[:2], seats[:2]), 1
            ):
                for region_id, region in self.region_registry.items():
                    semantic = _semantic_role(
                        region["semantics"],
                        accepted=personal_role["accepted_categories"],
                        rejected=personal_role["rejected_categories"],
                    )
                    planar = self._planar_status(region["geometry"])
                    fit = evaluate_fits_set_on(
                        [
                            self.payload_registry[payload_id]["geometry"]
                            for payload_id in bundle_ids
                        ],
                        region["geometry"],
                        task_config=self.task,
                    )
                    near = evaluate_near_seat(
                        region["geometry"],
                        self.seating_registry[seat_id],
                        maximum_distance_m=float(
                            self.task["geometric_requirements"][
                                "personal_context"
                            ]["maximum_centroid_distance_m"]
                        ),
                    )
                    status = _tri_and(
                        semantic["status"],
                        planar,
                        fit["status"],
                        near["status"],
                        "TRUE"
                        if self.seating_registry[seat_id][
                            "semantic_role"
                        ]["status"]
                        == "TRUE"
                        else "UNKNOWN",
                    )
                    row = {
                        "function_group": "personal_drinks",
                        "slot_id": f"refreshment_slot_{slot_index}",
                        "payload_id": bundle_ids[0],
                        "payload_ids": list(bundle_ids),
                        "seating_target_id": seat_id,
                        "region_id": region_id,
                        "candidate_rank": region["candidate_rank"],
                        "region_semantic_label": region["semantics"].get(
                            "canonical_label"
                        ),
                        "semantic_role_status": semantic["status"],
                        "PLANAR_SUPPORT": planar,
                        "FITS_SET_ON": fit["status"],
                        "NEAR_SEAT": near["status"],
                        "fit_margin_m": fit.get("signed_clearance_margin_m"),
                        "near_seat_distance_m": near.get(
                            "measured_distance_m"
                        ),
                        "near_seat_margin_m": near.get("signed_margin_m"),
                        "compatibility_status": status,
                        "fit_evidence": fit,
                        "near_seat_evidence": near,
                        "region_evidence_path": region["provenance"][
                            "measurement_cloud_path"
                        ],
                        "payload_evidence_paths": [
                            self.payload_registry[payload_id]["provenance"][
                                "measurement_cloud_path"
                            ]
                            for payload_id in bundle_ids
                        ],
                        "seat_evidence_path": self.seating_registry[seat_id][
                            "provenance"
                        ]["evidence_path"],
                    }
                    self.drink_rows.append(row)
                    self.event(
                        "PAYLOAD_REGION_FIT_EVALUATED",
                        function_group="personal_drinks",
                        payload_ids=list(bundle_ids),
                        region_id=region_id,
                        status=fit["status"],
                        signed_margin_m=fit.get("signed_clearance_margin_m"),
                    )
                    self.event(
                        "REGION_TARGET_COMPATIBILITY_EVALUATED",
                        function_group="personal_drinks",
                        payload_ids=list(bundle_ids),
                        seating_target_id=seat_id,
                        region_id=region_id,
                        status=status,
                    )
        controls = payloads["tv_remote"][:1] + payloads[
            "game_controller"
        ][:1]
        if len(controls) == 2:
            for region_id, region in self.region_registry.items():
                semantic = _semantic_role(
                    region["semantics"],
                    accepted=control_role["accepted_categories"],
                    rejected=control_role["rejected_categories"],
                )
                planar = self._planar_status(region["geometry"])
                packing = evaluate_fits_set_on(
                    [
                        self.payload_registry[object_id]["geometry"]
                        for object_id in controls
                    ],
                    region["geometry"],
                    task_config=self.task,
                )
                access = evaluate_control_accessibility(
                    region["geometry"],
                    self.seating_registry.values(),
                    maximum_distance_m=float(
                        self.task["geometric_requirements"][
                            "control_context"
                        ]["maximum_distance_to_each_seat_m"]
                    ),
                )
                status = _tri_and(
                    semantic["status"],
                    planar,
                    packing["status"],
                    access["status"],
                )
                self.control_rows.append(
                    {
                        "function_group": "shared_controls",
                        "payload_ids": controls,
                        "region_id": region_id,
                        "candidate_rank": region["candidate_rank"],
                        "region_semantic_label": region["semantics"].get(
                            "canonical_label"
                        ),
                        "semantic_role_status": semantic["status"],
                        "PLANAR_SUPPORT": planar,
                        "FITS_SET_ON": packing["status"],
                        "ACCESSIBLE_FROM_VIEWING_AREA": access["status"],
                        "packing_margin_m": packing.get(
                            "signed_clearance_margin_m"
                        ),
                        "accessibility_margin_m": access.get(
                            "signed_margin_m"
                        ),
                        "selected_packing": packing.get("selected_packing"),
                        "compatibility_status": status,
                        "packing_evidence": packing,
                        "accessibility_evidence": access,
                        "region_evidence_path": region["provenance"][
                            "measurement_cloud_path"
                        ],
                    }
                )
                self.event(
                    "PAYLOAD_SET_PACKING_EVALUATED",
                    function_group="shared_controls",
                    payload_ids=controls,
                    region_id=region_id,
                    status=packing["status"],
                    signed_margin_m=packing.get(
                        "signed_clearance_margin_m"
                    ),
                )
                for payload_id in controls:
                    fit = evaluate_fits_on(
                        self.payload_registry[payload_id]["geometry"],
                        region["geometry"],
                        task_config=self.task,
                    )
                    individual_status = _tri_and(
                        semantic["status"],
                        planar,
                        fit["status"],
                        access["status"],
                    )
                    self.control_individual_rows.append(
                        {
                            "function_group": "shared_controls",
                            "payload_id": payload_id,
                            "region_id": region_id,
                            "candidate_rank": region["candidate_rank"],
                            "fit_margin_m": fit.get("signed_fit_margin_m"),
                            "accessibility_margin_m": access.get(
                                "signed_margin_m"
                            ),
                            "compatibility_status": individual_status,
                        }
                    )

    def _evaluate_policies(self) -> dict[str, Any]:
        solver = RegionAllocationSolver(
            drink_rows=self.drink_rows,
            control_rows=self.control_rows,
            control_individual_rows=self.control_individual_rows,
            task_config=self.task,
        )
        results = {}
        for policy in POLICIES:
            result = solver.solve(policy)
            truth_valid = self._valid_against_function_aware_task(result)
            result["valid_against_function_aware_task"] = truth_valid
            if policy == "always_shared" and result["status"] == "COMPLETE":
                result["classification"] = (
                    "CORRECT"
                    if truth_valid
                    else "FALSE_POSITIVE_INVALID_COMPLETE"
                )
            elif (
                policy == "always_distinct"
                and result["status"] != "COMPLETE"
                and solver.solve("function_aware")["status"] == "COMPLETE"
            ):
                result["classification"] = "FALSE_NEGATIVE"
            elif policy == "function_aware":
                result["classification"] = (
                    "CORRECT"
                    if result["status"] == "COMPLETE"
                    else "CORRECT_REJECTION"
                )
            else:
                result["classification"] = (
                    "CORRECT" if truth_valid else "INVALID"
                )
            results[policy] = result
            if result["status"] == "COMPLETE":
                groups = {
                    assignment["function_group"]
                    for assignment in result["assignments"]
                }
                if "personal_drinks" in groups:
                    self.event(
                        "DEDICATED_REGION_ASSIGNMENT_CREATED",
                        policy=policy,
                    )
                if "shared_controls" in groups:
                    self.event(
                        "SHARED_REGION_ASSIGNMENT_CREATED",
                        policy=policy,
                    )
                if not truth_valid:
                    self.event(
                        "REGION_DISTINCTNESS_VIOLATION",
                        policy=policy,
                    )
            else:
                if (
                    policy == "always_distinct"
                    and solver.solve("function_aware")["status"] == "COMPLETE"
                    and "shared_controls"
                    in self.task["active_function_groups"]
                ):
                    self.event(
                        "SHARED_REGION_REQUIREMENT_VIOLATION",
                        policy=policy,
                        reason=(
                            "DIAGNOSTIC_POLICY_FORCED_SHARED_CONTROL_"
                            "PAYLOADS_APART"
                        ),
                    )
                self.event(
                    "NO_COMPLETE_REGION_ASSIGNMENT",
                    policy=policy,
                    failed_constraints=result["failed_constraints"],
                )
        production = results["function_aware"]
        if production["status"] == "COMPLETE":
            self.event("FUNCTION_AWARE_REGION_WITNESS_COMPLETE")
            self._write_handoff(production)
        else:
            self.event("REGION_ALLOCATION_EXHAUSTED")
        return results

    def _valid_against_function_aware_task(
        self, result: dict[str, Any]
    ) -> bool:
        if result["status"] != "COMPLETE":
            return False
        active = set(self.task["active_function_groups"])
        assignments = result["assignments"]
        if "personal_drinks" in active:
            drink_regions = [
                assignment["region_id"]
                for assignment in assignments
                if assignment["function_group"] == "personal_drinks"
            ]
            if len(drink_regions) != 2 or len(set(drink_regions)) != 2:
                return False
        if "shared_controls" in active:
            control_regions = [
                assignment["region_id"]
                for assignment in assignments
                if assignment["function_group"] == "shared_controls"
            ]
            if len(control_regions) != 2 or len(set(control_regions)) != 1:
                return False
        if (
            not self.task["allow_cross_function_region_sharing"]
            and len(active) > 1
        ):
            by_group = {
                group: {
                    assignment["region_id"]
                    for assignment in assignments
                    if assignment["function_group"] == group
                }
                for group in active
            }
            if any(
                first & second
                for first, second in itertools.combinations(
                    by_group.values(), 2
                )
            ):
                return False
        return True

    def _write_handoff(self, result: dict[str, Any]) -> None:
        drink_assignments = [
            assignment
            for assignment in result["assignments"]
            if assignment["function_group"] == "personal_drinks"
        ]
        control_assignments = [
            assignment
            for assignment in result["assignments"]
            if assignment["function_group"] == "shared_controls"
        ]
        control_region = (
            control_assignments[0]["region_id"]
            if control_assignments
            else None
        )
        control_row = next(
            (
                row
                for row in self.control_rows
                if row["region_id"] == control_region
                and row["compatibility_status"] == "TRUE"
            ),
            None,
        )
        handoff = {
            "schema_version": 1,
            "task_id": self.task["task_id"],
            "natural_language_goal": self.task["natural_language_goal"],
            "payload_persistent_ids": sorted(self.payload_registry),
            "seating_target_region_ids": sorted(self.seating_registry),
            "personal_drink_assignments": drink_assignments,
            "personal_drink_region_ids_distinct": (
                len({item["region_id"] for item in drink_assignments})
                == len(drink_assignments)
                if drink_assignments
                else None
            ),
            "shared_control_assignments": control_assignments,
            "shared_control_region_id": control_region,
            "controls_share_same_region": (
                len({item["region_id"] for item in control_assignments}) == 1
                if control_assignments
                else None
            ),
            "FITS_SET_ON": (
                control_row["packing_evidence"] if control_row else None
            ),
            "semantic_evidence": {
                "payloads": {
                    object_id: record["semantics"]
                    for object_id, record in self.payload_registry.items()
                },
                "regions": {
                    region_id: record["semantics"]
                    for region_id, record in self.region_registry.items()
                },
                "seating_targets": {
                    seat_id: record["semantics"]
                    for seat_id, record in self.seating_registry.items()
                },
            },
            "geometric_evidence": {
                "payloads": {
                    object_id: record["geometry"]
                    for object_id, record in self.payload_registry.items()
                },
                "regions": {
                    region_id: record["geometry"]
                    for region_id, record in self.region_registry.items()
                },
            },
            "target_specific_context_evidence": self.drink_rows,
            "total_distinct_physical_region_count": result[
                "distinct_physical_region_count"
            ],
            "cross_function_region_sharing_allowed": self.task[
                "allow_cross_function_region_sharing"
            ],
            "evidence_root": "observation",
            "verified": True,
            "ready_for_tamp": True,
            "placement_executed": False,
            "tamp_executed": False,
        }
        _atomic_json(
            self.run_dir / "verified_region_allocation_handoff.json",
            handoff,
        )

    def _persist(self, semantic_timing: dict[str, Any]) -> None:
        _atomic_json(
            self.run_dir / "payload_registry.json",
            {"schema_version": 1, "objects": self.payload_registry},
        )
        _atomic_json(
            self.run_dir / "region_registry.json",
            {"schema_version": 1, "regions": self.region_registry},
        )
        _atomic_json(
            self.run_dir / "seating_registry.json",
            {"schema_version": 1, "seating_targets": self.seating_registry},
        )
        drink_matrix = {
            "schema_version": 1,
            "function_group": "personal_drinks",
            "rows": self.drink_rows,
        }
        control_matrix = {
            "schema_version": 1,
            "function_group": "shared_controls",
            "rows": self.control_rows,
            "individual_payload_rows": self.control_individual_rows,
        }
        _atomic_json(
            self.run_dir / "drink_region_compatibility.json", drink_matrix
        )
        _atomic_json(
            self.run_dir / "control_region_compatibility.json",
            control_matrix,
        )
        self._write_csv(
            self.run_dir / "drink_region_compatibility.csv",
            self.drink_rows,
        )
        self._write_csv(
            self.run_dir / "control_region_compatibility.csv",
            self.control_rows,
        )
        _atomic_json(
            self.run_dir / "policy_evaluations.json",
            {"schema_version": 1, "policies": self.policy_evaluations},
        )
        _atomic_json(
            self.run_dir / "region_assignments.json",
            {
                policy: result["assignments"]
                for policy, result in self.policy_evaluations.items()
            },
        )
        _atomic_json(
            self.run_dir / "distinct_region_counts.json",
            {
                policy: result["distinct_physical_region_count"]
                for policy, result in self.policy_evaluations.items()
            },
        )
        evidence_paths = sorted(
            path
            for pattern in (
                "observation/cameras/*/rgb.png",
                "observation/cameras/*/depth.png",
                "observation/cameras/*/segmentation.png",
                "observation/cameras/*/evidence_masks.png",
                "observation/regions/*/fused.ply",
                "observation/payloads/*/fused.ply",
                "observation/seats/*/observed_points.ply",
                "observation/semantics/detections.json",
                "observation/semantics/associations.json",
                "observation/semantics/fused_semantics.json",
            )
            for path in self.run_dir.glob(pattern)
        )
        manifest = [
            {
                "path": path.relative_to(self.run_dir).as_posix(),
                "sha256": _hash_file(path),
            }
            for path in evidence_paths
        ]
        summary = {
            "schema_version": 1,
            "scene_name": self.scene_name,
            "task_id": self.task["task_id"],
            "single_initial_observation": True,
            "rerendered_for_policies": False,
            "semantic_inference_repeated_for_policies": False,
            "same_evidence_manifest": manifest,
            "policies": self.policy_evaluations,
            "semantic_inference_timing": semantic_timing,
        }
        _atomic_json(
            self.run_dir
            / "offline_region_policy_ablation_evaluation.json",
            summary,
        )
        _atomic_json(
            self.run_dir / "region_ablation2_summary.json", summary
        )
        validation = self.validate_expected()
        _atomic_json(
            self.run_dir / "region_ablation2_validation.json", validation
        )
        _atomic_json(
            self.run_dir / "observed_graph.json", self.build_graph()
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        flattened = []
        for row in rows:
            flattened.append(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
        fields = sorted({key for row in flattened for key in row})
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(flattened)
        temporary.replace(path)

    def validate_expected(self) -> dict[str, Any]:
        expected = self.evaluation["scenes"].get(
            self.scene_name, {}
        ).get("expected", {})
        checks = []
        for policy, requirement in expected.items():
            observed = self.policy_evaluations[policy]
            expected_valid = requirement["validity"] == "VALID"
            count_matches = (
                "distinct_region_count" not in requirement
                or observed["distinct_physical_region_count"]
                == requirement["distinct_region_count"]
            )
            classification_matches = (
                requirement["classification"]
                in observed["classification"]
                or observed["classification"]
                in requirement["classification"]
            )
            checks.append(
                {
                    "policy": policy,
                    "expected_status": requirement["status"],
                    "observed_status": observed["status"],
                    "expected_validity": requirement["validity"],
                    "observed_validity": (
                        "VALID"
                        if observed["valid_against_function_aware_task"]
                        or (
                            policy == "function_aware"
                            and observed["status"] == "EXHAUSTED"
                        )
                        else "INVALID"
                    ),
                    "expected_classification": requirement[
                        "classification"
                    ],
                    "observed_classification": observed["classification"],
                    "passed": (
                        observed["status"] == requirement["status"]
                        and (
                            observed["valid_against_function_aware_task"]
                            == expected_valid
                            or (
                                policy == "function_aware"
                                and observed["status"] == "EXHAUSTED"
                                and expected_valid
                            )
                        )
                        and count_matches
                        and classification_matches
                    ),
                }
            )
        return {
            "schema_version": 1,
            "scene_name": self.scene_name,
            "checks": checks,
            "passed": bool(checks) and all(check["passed"] for check in checks),
        }

    def build_graph(self) -> dict[str, Any]:
        nodes = []
        edges = []
        for object_id, record in self.payload_registry.items():
            nodes.append(
                {
                    "id": f"payload:{object_id}",
                    "type": "PAYLOAD",
                    "attributes": deepcopy(record),
                }
            )
            role = record.get("semantic_payload_role")
            if role:
                for group, config in self.task["function_groups"].items():
                    if role in config["payload_semantic_roles"]:
                        edges.append(
                            {
                                "source": f"payload:{object_id}",
                                "target": f"function:{group}",
                                "type": "CANDIDATE_FOR_FUNCTION",
                                "status": "TRUE",
                            }
                        )
        for seat_id, record in self.seating_registry.items():
            nodes.append(
                {
                    "id": f"seat:{seat_id}",
                    "type": "SEATING_TARGET",
                    "attributes": deepcopy(record),
                }
            )
        for region_id, record in self.region_registry.items():
            nodes.append(
                {
                    "id": f"region:{region_id}",
                    "type": "DESTINATION_REGION",
                    "attributes": deepcopy(record),
                }
            )
        for group, config in self.task["function_groups"].items():
            nodes.append(
                {
                    "id": f"function:{group}",
                    "type": "FUNCTION_GROUP",
                    "attributes": deepcopy(config),
                }
            )
        for row in self.drink_rows:
            edges.extend(
                [
                    {
                        "source": f"payload:{row['payload_id']}",
                        "target": f"region:{row['region_id']}",
                        "type": "FITS_SET_ON",
                        "status": row["FITS_SET_ON"],
                    },
                    {
                        "source": f"region:{row['region_id']}",
                        "target": f"seat:{row['seating_target_id']}",
                        "type": "NEAR_SEAT",
                        "status": row["NEAR_SEAT"],
                    },
                    {
                        "source": f"region:{row['region_id']}",
                        "target": f"seat:{row['seating_target_id']}",
                        "type": "COMPATIBLE_WITH_TARGET",
                        "status": row["compatibility_status"],
                    },
                ]
            )
        for row in self.control_rows:
            for payload_id in row["payload_ids"]:
                edges.append(
                    {
                        "source": f"payload:{payload_id}",
                        "target": f"region:{row['region_id']}",
                        "type": "FITS_SET_ON",
                        "status": row["FITS_SET_ON"],
                    }
                )
        production = self.policy_evaluations.get("function_aware", {})
        for assignment in production.get("assignments", []):
            edges.append(
                {
                    "source": f"payload:{assignment['payload_id']}",
                    "target": f"region:{assignment['region_id']}",
                    "type": "ASSIGNED_TO_TARGET",
                    "status": "TRUE",
                    "attributes": {
                        "function_group": assignment["function_group"],
                        "target_id": assignment["target_id"],
                    },
                }
            )
        if production.get("status") == "COMPLETE":
            grouped: dict[str, list[dict[str, Any]]] = {}
            for assignment in production["assignments"]:
                grouped.setdefault(
                    assignment["function_group"], []
                ).append(assignment)
            for group, assignments in grouped.items():
                region_ids = sorted(
                    {item["region_id"] for item in assignments}
                )
                for region_id in region_ids:
                    edges.append(
                        {
                            "source": f"region:{region_id}",
                            "target": f"function:{group}",
                            "type": "SATISFIES_FUNCTION_GROUP",
                            "status": "TRUE",
                        }
                    )
                for first, second in itertools.combinations(
                    assignments, 2
                ):
                    edges.append(
                        {
                            "source": f"payload:{first['payload_id']}",
                            "target": f"payload:{second['payload_id']}",
                            "type": (
                                "SHARES_REGION_WITH"
                                if first["region_id"] == second["region_id"]
                                else "DISTINCT_FROM_REGION"
                            ),
                            "status": "TRUE",
                            "attributes": {
                                "first_region_id": first["region_id"],
                                "second_region_id": second["region_id"],
                            },
                        }
                    )
        return {"schema_version": 1, "nodes": nodes, "edges": edges}
