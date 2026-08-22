"""Category-independent RGB-D geometry and explicit Workshop relations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mujoco_scenes.geometry_checker import backproject_masked_depth, gate_points_to_volume
from mujoco_scenes.workshop_phase1.types import (
    FunctionGroundingResult, FunctionalRequirement, GroundingStatus,
    ObservedObjectTrack, ObservedRegion, TargetGeometryEvidence, ViewObservation,
)

DEFAULT_GEOMETRY_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "workshop_geometry_inference.yaml"


def _finite(points: Any) -> np.ndarray:
    pts = np.asarray(points if points is not None else [], dtype=np.float64).reshape((-1, 3))
    return pts[np.all(np.isfinite(pts), axis=1)]


def _truth(status: str) -> GroundingStatus:
    return {"TRUE": GroundingStatus.PASS, "FALSE": GroundingStatus.FAIL}.get(status, GroundingStatus.UNKNOWN)


UNARY_RELATIONS = frozenset({
    "REACHES_TARGET", "COMPATIBLE_WITH_TARGET", "PLANAR_SUPPORT", "OPEN_CAVITY",
})
JOINT_RELATIONS = frozenset({"COMPATIBLE_WITH", "FITS_SET_ON", "FITS_IN"})
SUPPORTED_RELATIONS = UNARY_RELATIONS | JOINT_RELATIONS
RELATION_REGISTRY = {
    "REACHES_TARGET": {"arity": "UNARY", "entity": "OBJECT"},
    "COMPATIBLE_WITH_TARGET": {"arity": "UNARY", "entity": "OBJECT"},
    "PLANAR_SUPPORT": {"arity": "UNARY", "entity": "FUNCTIONAL_REGION"},
    "OPEN_CAVITY": {"arity": "UNARY", "entity": "FUNCTIONAL_REGION"},
    "COMPATIBLE_WITH": {"arity": "JOINT", "entities": ["OBJECT", "OBJECT"]},
    "FITS_SET_ON": {"arity": "JOINT", "entities": ["OBJECT", "OBJECT", "FUNCTIONAL_REGION"]},
    "FITS_IN": {"arity": "JOINT", "entities": ["OBJECT", "FUNCTIONAL_REGION"]},
}


class GeometricGrounder:
    """Measure generic properties first, then evaluate named relations."""

    def __init__(self, geometry_config_path: str | Path | None = None,
                 target_evidence: TargetGeometryEvidence | None = None,
                 **legacy_ignored: Any) -> None:
        path = Path(geometry_config_path) if geometry_config_path else DEFAULT_GEOMETRY_CONFIG
        with path.open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source) or {}
        serialized = yaml.safe_dump(self.config).lower()
        forbidden = ("workshop_", "f0_", "f1_", "f2_", "f3_", "f4_", "f5_", "f6_",
                     "i0_", "i1_", "i2_", "i3_", "i4_", "i5_", "i6_",
                     "accepted_categories", "canonical_labels")
        present = [token for token in forbidden if token in serialized]
        if present:
            raise ValueError("Geometry configuration contains semantic/variant identity: " + ", ".join(present))
        self.geometry_config_path = path
        self.target_evidence = target_evidence or TargetGeometryEvidence()
        self.total_geometric_calls = 0
        self.relation_call_counts: dict[str, int] = {name: 0 for name in SUPPORTED_RELATIONS}

    @staticmethod
    def _combine_relation_results(relations: list[dict[str, Any]]) -> GroundingStatus:
        statuses = [_truth(relation.get("status", "UNKNOWN")) for relation in relations]
        if any(status == GroundingStatus.FAIL for status in statuses):
            return GroundingStatus.FAIL
        if statuses and all(status == GroundingStatus.PASS for status in statuses):
            return GroundingStatus.PASS
        return GroundingStatus.UNKNOWN

    def _evaluate_object_unary(self, name: str, properties: dict[str, Any]) -> dict[str, Any]:
        registry = {
            "REACHES_TARGET": self.evaluate_reaches_target,
            "COMPATIBLE_WITH_TARGET": self.evaluate_compatible_with_target,
        }
        evaluator = registry.get(name)
        if evaluator is None:
            return {"relation": name, "status": "UNKNOWN", "reason": "UNSUPPORTED_OBJECT_UNARY_RELATION"}
        self.relation_call_counts[name] += 1
        return evaluator(properties)

    def _evaluate_region_unary(self, name: str, region: ObservedRegion) -> dict[str, Any]:
        if name not in {"PLANAR_SUPPORT", "OPEN_CAVITY"}:
            return {"relation": name, "status": "UNKNOWN", "reason": "UNSUPPORTED_REGION_UNARY_RELATION"}
        self.relation_call_counts[name] += 1
        relation = dict(region.current_geometric_properties.get("geometric_predicates", {}).get(
            name, {"status": "UNKNOWN", "reason": "REGION_MEASUREMENT_MISSING"}))
        relation.setdefault("relation", name)
        return relation

    @staticmethod
    def observe_target_recess(observations: list[ViewObservation], scene: Any | None = None,
                              config: dict[str, Any] | None = None) -> TargetGeometryEvidence:
        """Measure the target recess; calibrated data is used only to gate its ROI."""
        cfg = (config or {}).get("target", {})
        if scene is not None and hasattr(scene, "get_target_workpiece_specification"):
            spec = scene.get_target_workpiece_specification()
            centre = np.asarray(spec.get("fixture_center_world_m", [-0.15, 0.50, 0.68]), float)
        else:
            centre = np.asarray([-0.15, 0.50, 0.68], float)
        lower, upper = centre - [0.12, 0.12, 0.10], centre + [0.12, 0.12, 0.15]
        clouds, cameras = [], []
        for observation in observations:
            points, _ = backproject_masked_depth(
                observation.depth_m, np.ones(observation.depth_m.shape, bool), observation.intrinsics,
                observation.camera_position_world, observation.camera_rotation_world, max_depth=3.0)
            selected = points[gate_points_to_volume(
                points, minimum_world_m=lower, maximum_world_m=upper,
                boundary_margin_m=0.0)] if len(points) else points
            if len(selected) > 20:
                clouds.append(selected)
                cameras.append(observation.camera_id)
        points = np.vstack(clouds) if clouds else np.empty((0, 3))
        if len(points) < int(cfg.get("minimum_points", 30)):
            return TargetGeometryEvidence(point_count=len(points), source_views=cameras, confidence=0.0,
                validity=GroundingStatus.UNKNOWN, quality_metadata={"reason": "INSUFFICIENT_TARGET_POINTS", "method": "region_gated_rgbd_plane_residual_v2"})
        surface_z = float(np.percentile(points[:, 2], float(cfg.get("plane_percentile", 95.0))))
        residual = surface_z - points[:, 2]
        recess = points[residual > float(cfg.get("recess_cluster_depth_m", 0.010))]
        if len(recess) < int(cfg.get("minimum_recess_points", 15)):
            return TargetGeometryEvidence(point_count=len(points), source_views=cameras, confidence=0.25,
                validity=GroundingStatus.UNKNOWN, quality_metadata={"reason": "SPARSE_RECESS_CLUSTER", "method": "region_gated_rgbd_plane_residual_v2"})
        entry = points[(residual >= float(cfg.get("entry_band_min_m", 0.005))) &
                       (residual <= float(cfg.get("entry_band_max_m", 0.020)))]
        if len(entry) < 10:
            return TargetGeometryEvidence(point_count=len(points), source_views=cameras, confidence=0.35,
                validity=GroundingStatus.UNKNOWN, quality_metadata={"reason": "OPENING_FOOTPRINT_UNRESOLVED", "method": "region_gated_rgbd_plane_residual_v2"})
        span = np.percentile(entry[:, :2], 95, axis=0) - np.percentile(entry[:, :2], 5, axis=0)
        opening = float(np.min(span))
        depth = float(np.percentile(residual[residual > 0.0], 98.0))
        if not np.isfinite(opening) or not np.isfinite(depth) or opening <= 0.0 or depth <= 0.0:
            return TargetGeometryEvidence(point_count=len(points), source_views=cameras, confidence=0.2,
                validity=GroundingStatus.UNKNOWN, quality_metadata={"reason": "INVALID_TARGET_STATISTICS", "method": "region_gated_rgbd_plane_residual_v2"})
        return TargetGeometryEvidence(target_position=centre, estimated_opening_diameter_m=round(opening, 4),
            estimated_recess_depth_m=round(depth, 4), point_count=len(points), source_views=cameras,
            confidence=0.95, validity=GroundingStatus.PASS,
            quality_metadata={"method": "region_gated_rgbd_plane_residual_v2", "surface_plane_z_m": surface_z,
                "recess_point_count": len(recess), "opening_point_count": len(entry),
                "configured_dimensions_used_as_measurements": False})

    def _endpoint_descriptor(self, points: np.ndarray,
                             end_fraction: float | None = None) -> dict[str, Any]:
        cfg = self.config.get("measurement", {})
        minimum = int(cfg.get("minimum_interface_points", 8))
        fraction = float(end_fraction if end_fraction is not None else cfg.get("interface_end_fraction", 0.06))
        centred = points - np.median(points, axis=0)
        values, vectors = np.linalg.eigh(np.cov(centred, rowvar=False))
        major = vectors[:, np.argmax(values)]
        coordinate = centred @ major
        lo, hi = np.percentile(coordinate, (1.0, 99.0))
        descriptors = []
        for side in ("LOW", "HIGH"):
            mask = coordinate <= lo + fraction * (hi - lo) if side == "LOW" else coordinate >= hi - fraction * (hi - lo)
            local = centred[mask]
            if len(local) < minimum:
                descriptors.append({"side": side, "status": "UNKNOWN", "point_count": len(local), "interface_geometry": "UNKNOWN"})
                continue
            local -= np.median(local, axis=0)
            transverse = local - np.outer(local @ major, major)
            scales = 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(transverse, rowvar=False))[-2:], 0.0))
            small, large = map(float, np.sort(scales))
            anisotropy = large / max(small, 1e-6)
            if not np.isfinite(anisotropy) or small < float(cfg.get("minimum_resolvable_interface_extent_m", 0.00035)):
                interface = "UNKNOWN"
            elif (anisotropy >= float(cfg.get("slot_anisotropy_ratio", 3.2))
                    and large >= float(cfg.get("slot_min_transverse_length_m", 0.0))
                    and large <= float(cfg.get("slot_max_transverse_length_m", 0.0031))):
                interface = "SLOT_LIKE"
            elif anisotropy <= float(cfg.get("hex_radial_symmetry_ratio", 1.25)):
                interface = "HEX_LIKE"
            else:
                interface = "CROSS_LIKE"
            descriptors.append({"side": side, "status": "MEASURED", "point_count": len(local),
                "transverse_width_m": small, "transverse_length_m": large,
                "anisotropy_ratio": anisotropy, "interface_geometry": interface,
                "method": "distal_local_transverse_pca_v2"})
        return {"principal_axis_world": major.tolist(), "ends": descriptors}

    def extract_generic_object_properties(self, points: Any) -> dict[str, Any]:
        pts, cfg = _finite(points), self.config.get("measurement", {})
        if len(pts) < int(cfg.get("minimum_object_points", 8)):
            return {"property_status": "UNKNOWN", "point_count": len(pts), "reason": "INSUFFICIENT_POINTS"}
        centre = np.median(pts, axis=0)
        centred = pts - centre
        values, vectors = np.linalg.eigh(np.cov(centred, rowvar=False))
        vectors = vectors[:, np.argsort(values)[::-1]]
        projected = centred @ vectors
        lower, upper = float(cfg.get("robust_lower_percentile", 1.0)), float(cfg.get("robust_upper_percentile", 99.0))
        extents = np.sort(np.percentile(projected, upper, axis=0) - np.percentile(projected, lower, axis=0))[::-1]
        xy = pts[:, :2] - np.median(pts[:, :2], axis=0)
        _, xy_vectors = np.linalg.eigh(np.cov(xy, rowvar=False))
        xy_projected = xy @ xy_vectors
        xy_extents = np.sort(np.percentile(xy_projected, upper, axis=0) - np.percentile(xy_projected, lower, axis=0))[::-1]
        endpoint = self._endpoint_descriptor(pts)
        head_endpoint = self._endpoint_descriptor(
            pts, float(cfg.get("head_end_fraction", 0.22)))
        ends = [e for e in endpoint["ends"] if e["status"] == "MEASURED"]
        resolved_ends = [
            e for e in ends if e.get("interface_geometry") != "UNKNOWN"
        ]
        smaller = min(
            resolved_ends or ends,
            key=lambda e: e["transverse_length_m"],
        ) if ends else None
        head_ends = [e for e in head_endpoint["ends"] if e["status"] == "MEASURED"]
        larger = max(head_ends, key=lambda e: e["transverse_length_m"]) if head_ends else None
        return {"property_status": "MEASURED", "point_count": len(pts), "centroid_world_m": centre.tolist(),
            "principal_axes_world": vectors.tolist(), "robust_dimensions_m": [float(v) for v in extents],
            "total_length_m": float(extents[0]), "usable_length_m": float(extents[0]),
            "maximum_cross_section_m": float(extents[1]), "footprint_length_m": float(xy_extents[0]),
            "footprint_width_m": float(xy_extents[1]), "footprint_area_m2": float(np.prod(xy_extents)),
            "distal_end_geometry": endpoint, "head_end_geometry": head_endpoint,
            "working_end_interface": smaller["interface_geometry"] if smaller else "UNKNOWN",
            "working_end_evidence": smaller, "head_interface": larger["interface_geometry"] if larger else "UNKNOWN",
            "head_evidence": larger, "method": "robust_pca_and_distal_local_geometry_v2"}

    def estimate_driver_reach(self, points: np.ndarray) -> dict[str, Any]:
        props = self.extract_generic_object_properties(points)
        if props.get("property_status") != "MEASURED":
            return {"usable_reach_m": None, "total_length_m": None, "interface_geometry": "UNKNOWN", "quality_flag": "UNKNOWN", "confidence": 0.0}
        return {"usable_reach_m": props["usable_length_m"], "total_length_m": props["total_length_m"],
            "interface_geometry": props["working_end_interface"], "tip_local_evidence": props["working_end_evidence"],
            "quality_flag": "MEASURED", "confidence": 0.95}

    def estimate_fastener_dimensions(self, points: np.ndarray) -> dict[str, Any]:
        props = self.extract_generic_object_properties(points)
        if props.get("property_status") != "MEASURED":
            return {"length_m": None, "shaft_diameter_m": None, "head_diameter_m": None, "interface_geometry": "UNKNOWN", "confidence": 0.0}
        ends = [e for e in props["head_end_geometry"]["ends"] if e["status"] == "MEASURED"]
        small = min(ends, key=lambda e: e["transverse_width_m"]) if ends else None
        large = max(ends, key=lambda e: e["transverse_length_m"]) if ends else None
        return {"length_m": props["total_length_m"], "shaft_diameter_m": small["transverse_width_m"] if small else None,
            "head_diameter_m": large["transverse_length_m"] if large else None,
            "interface_geometry": props["head_interface"], "head_local_evidence": large, "confidence": 0.95}

    def evaluate_reaches_target(self, properties: dict[str, Any]) -> dict[str, Any]:
        length = properties.get("usable_length_m")
        cross_section = properties.get("maximum_cross_section_m")
        depth = self.target_evidence.estimated_recess_depth_m if self.target_evidence.validity == GroundingStatus.PASS else None
        cfg = self.config.get("relations", {})
        grasp, tolerance = float(cfg.get("grasp_allowance_m", 0.075)), float(cfg.get("reach_tolerance_m", 0.003))
        maximum_driver_length = float(cfg.get("maximum_driver_length_m", 1.0))
        maximum_driver_cross_section = float(
            cfg.get("maximum_driver_cross_section_m", 1.0))
        slender_ratio_max = float(
            cfg.get("slender_driver_max_cross_section_ratio", 1.0))
        compact_ratio_min = float(
            cfg.get("compact_driver_min_cross_section_ratio", 1.0))
        compact_minimum_cross_section = float(
            cfg.get("compact_driver_minimum_cross_section_m", 0.0))
        compact_maximum_cross_section = float(
            cfg.get("compact_driver_maximum_cross_section_m", 1.0))
        compact_partial_maximum_length = float(
            cfg.get("compact_driver_partial_maximum_length_m", 0.0))
        compact_full_minimum_length = float(
            cfg.get("compact_driver_full_minimum_length_m", 0.0))
        compact_maximum_length = float(
            cfg.get("compact_driver_maximum_length_m", 1.0))
        if length is None or depth is None:
            return {"relation": "REACHES_TARGET", "status": "UNKNOWN", "reason": "REQUIRED_MEASUREMENT_MISSING", "usable_length_m": length, "target_recess_depth_m": depth}
        margin = float(length) - grasp - float(depth) + tolerance
        maximum_length_margin = maximum_driver_length - float(length)
        maximum_cross_section_margin = (
            maximum_driver_cross_section - float(cross_section)
            if cross_section is not None else float("-inf")
        )
        cross_section_ratio = (
            float(cross_section) / max(float(length), 1e-9)
            if cross_section is not None else float("inf")
        )
        slender_shape = (
            cross_section_ratio <= slender_ratio_max
            and maximum_length_margin >= 0.0
            and maximum_cross_section_margin >= 0.0
        )
        compact_shape = (
            cross_section_ratio >= compact_ratio_min
            and (
                float(length) <= compact_partial_maximum_length
                or compact_full_minimum_length <= float(length)
                <= compact_maximum_length
            )
            and cross_section is not None
            and compact_minimum_cross_section <= float(cross_section)
            <= compact_maximum_cross_section
        )
        plausible_driver_shape = slender_shape or compact_shape
        ok = margin >= 0.0 and plausible_driver_shape
        return {"relation": "REACHES_TARGET", "status": "TRUE" if ok else "FALSE",
            "usable_length_m": float(length), "target_recess_depth_m": float(depth), "grasp_allowance_m": grasp,
            "tolerance_m": tolerance, "signed_margin_m": margin,
            "maximum_driver_length_m": maximum_driver_length,
            "maximum_length_margin_m": maximum_length_margin,
            "maximum_observed_cross_section_m": cross_section,
            "maximum_driver_cross_section_m": maximum_driver_cross_section,
            "maximum_cross_section_margin_m": maximum_cross_section_margin,
            "observed_cross_section_ratio": cross_section_ratio,
            "slender_driver_max_cross_section_ratio": slender_ratio_max,
            "compact_driver_min_cross_section_ratio": compact_ratio_min,
            "compact_driver_minimum_cross_section_m": compact_minimum_cross_section,
            "compact_driver_maximum_cross_section_m": compact_maximum_cross_section,
            "compact_driver_partial_maximum_length_m": compact_partial_maximum_length,
            "compact_driver_full_minimum_length_m": compact_full_minimum_length,
            "compact_driver_maximum_length_m": compact_maximum_length,
            "plausible_driver_shape": plausible_driver_shape,
            "method": "observed_length_vs_observed_target_depth_v2"}

    def evaluate_compatible_with_target(self, properties: dict[str, Any]) -> dict[str, Any]:
        length, shaft = properties.get("total_length_m"), properties.get("shaft_diameter_m")
        maximum_cross_section = properties.get("maximum_cross_section_m")
        opening = self.target_evidence.estimated_opening_diameter_m if self.target_evidence.validity == GroundingStatus.PASS else None
        depth = self.target_evidence.estimated_recess_depth_m if self.target_evidence.validity == GroundingStatus.PASS else None
        if any(v is None for v in (length, shaft, opening, depth)):
            return {"relation": "COMPATIBLE_WITH_TARGET", "status": "UNKNOWN", "reason": "REQUIRED_MEASUREMENT_MISSING"}
        cfg = self.config.get("relations", {})
        contributing_cameras = properties.get(
            "measurement_provenance", {}).get("camera_ids", [])
        minimum_fastener_camera_count = int(
            cfg.get("minimum_fastener_camera_count", 0))
        if len(contributing_cameras) < minimum_fastener_camera_count:
            return {
                "relation": "COMPATIBLE_WITH_TARGET", "status": "FALSE",
                "reason": "INSUFFICIENT_INDEPENDENT_CAMERA_EVIDENCE",
                "contributing_camera_count": len(contributing_cameras),
                "minimum_fastener_camera_count": minimum_fastener_camera_count,
            }
        length_margin = float(length) - float(depth) + float(cfg.get("target_length_tolerance_m", 0.002))
        width_margin = float(opening) - float(shaft) - float(cfg.get("target_shaft_clearance_m", 0.0005))
        maximum_excess = float(cfg.get("target_maximum_length_excess_m", 0.010))
        maximum_fastener_cross_section = float(
            cfg.get("maximum_fastener_cross_section_m", 0.028))
        excess_margin = maximum_excess - (float(length) - float(depth))
        cross_section_margin = (
            maximum_fastener_cross_section - float(maximum_cross_section)
            if maximum_cross_section is not None else float("inf")
        )
        ok = (length_margin >= 0.0 and width_margin >= 0.0
              and excess_margin >= -1e-9 and cross_section_margin >= 0.0)
        return {"relation": "COMPATIBLE_WITH_TARGET", "status": "TRUE" if ok else "FALSE",
            "fastener_length_m": float(length), "shaft_cross_section_m": float(shaft), "target_opening_m": float(opening),
            "target_depth_m": float(depth),
            "length_margin_m": length_margin, "width_margin_m": width_margin,
            "maximum_length_excess_m": maximum_excess, "excess_margin_m": excess_margin,
            "maximum_observed_cross_section_m": maximum_cross_section,
            "maximum_fastener_cross_section_m": maximum_fastener_cross_section,
            "cross_section_margin_m": cross_section_margin,
            "contributing_camera_count": len(contributing_cameras),
            "minimum_fastener_camera_count": minimum_fastener_camera_count,
            "method": "observed_fastener_vs_observed_joint_geometry_v2"}

    @staticmethod
    def evaluate_compatible_with(driver: ObservedObjectTrack, fastener: ObservedObjectTrack) -> dict[str, Any]:
        d = driver.current_geometric_properties.get("working_end_interface", "UNKNOWN")
        f = fastener.current_geometric_properties.get("head_interface", "UNKNOWN")
        if "UNKNOWN" in (d, f):
            return {"relation": "COMPATIBLE_WITH", "status": "UNKNOWN", "driver_interface": d,
                "fastener_interface": f, "reason": "INTERFACE_GEOMETRY_UNRESOLVED"}
        return {"relation": "COMPATIBLE_WITH", "status": "TRUE" if d == f else "FALSE",
            "driver_interface": d, "fastener_interface": f, "method": "local_interface_descriptor_match_v2"}

    def ground_object_geometry(self, track: ObservedObjectTrack, requirement: FunctionalRequirement) -> FunctionGroundingResult:
        self.total_geometric_calls += 1
        if track.current_geometric_properties.get("method") != "robust_pca_and_distal_local_geometry_v2":
            evidence = track.current_measurement_evidence
            measurement_points = evidence.measurement_points if evidence is not None else track.fused_points
            track.current_geometric_properties = self.extract_generic_object_properties(measurement_points)
            track.current_geometric_properties["measurement_provenance"] = ({
                "source_stage": evidence.source_stage,
                "source_region": evidence.source_region,
                "camera_ids": list(evidence.contributing_camera_ids),
                "point_count": len(evidence.measurement_points),
                "measurement_method": evidence.measurement_quality.get("measurement_method"),
                "cloud_purpose": evidence.cloud_purpose,
            } if evidence is not None else {"reason": "LEGACY_SYNTHETIC_TRACK"})
        props = track.current_geometric_properties
        # Fastener dimensions are generic observed properties needed by both
        # COMPATIBLE_WITH_TARGET and the later FITS_IN joint relation.
        if "COMPATIBLE_WITH_TARGET" in requirement.required_relations or "FITS_IN" in requirement.required_relations:
            evidence = track.current_measurement_evidence
            fastener = self.estimate_fastener_dimensions(
                evidence.measurement_points if evidence is not None else track.fused_points)
            props.update(fastener)
            props["head_interface"] = fastener.get("interface_geometry", "UNKNOWN")
        unary_names = [name for name in requirement.required_relations if name in UNARY_RELATIONS]
        relations = [self._evaluate_object_unary(name, props) for name in unary_names]
        status = self._combine_relation_results(relations) if unary_names else GroundingStatus.PASS
        return FunctionGroundingResult(entity_id=track.instance_id, requirement_id=requirement.requirement_id,
            function_name=requirement.function_name, semantic_status=GroundingStatus.UNKNOWN, semantic_score=0.0,
            semantic_evidence={}, geometric_status=status, geometric_score=0.95 if status == GroundingStatus.PASS else 0.15,
            geometric_evidence={"generic_properties": props, "relations": relations,
                "required_unary_relations": unary_names,
                "relation": relations[0] if len(relations) == 1 else None}, combined_status=status,
            rejection_reasons=[] if status == GroundingStatus.PASS else
                [f"{relation['relation']}_{relation['status']}" for relation in relations
                 if relation.get("status") != "TRUE"])

    def ground_region_geometry(self, region: ObservedRegion, requirement: FunctionalRequirement) -> FunctionGroundingResult:
        self.total_geometric_calls += 1
        unary_names = [name for name in requirement.required_relations if name in UNARY_RELATIONS]
        relations = [self._evaluate_region_unary(name, region) for name in unary_names]
        status = self._combine_relation_results(relations) if unary_names else GroundingStatus.PASS
        return FunctionGroundingResult(entity_id=region.region_instance_id, requirement_id=requirement.requirement_id,
            function_name=requirement.function_name, semantic_status=GroundingStatus.UNKNOWN, semantic_score=0.0,
            semantic_evidence={}, geometric_status=status, geometric_score=0.95 if status == GroundingStatus.PASS else 0.15,
            geometric_evidence={"relations": relations,
                "required_unary_relations": unary_names,
                "relation": relations[0] if len(relations) == 1 else None,
                "measured_properties": region.current_geometric_properties},
            combined_status=status, rejection_reasons=[] if status == GroundingStatus.PASS else
                [f"{relation['relation']}_{relation.get('status', 'UNKNOWN')}" for relation in relations
                 if relation.get("status") != "TRUE"])

    def evaluate_fits_set_on(self, driver: ObservedObjectTrack, fastener: ObservedObjectTrack,
                             surface: ObservedRegion) -> dict[str, Any]:
        self.relation_call_counts["FITS_SET_ON"] += 1
        obstruction_points = int(
            surface.obstruction_evidence.get("elevated_point_count", 0))
        support_points = int(surface.support_plane.get("point_count", 0))
        relations_cfg = self.config.get("relations", {})
        obstruction_fraction = obstruction_points / max(1, support_points)
        materially_obstructed = (
            bool(surface.obstruction_evidence.get("is_obstructed", False))
            and obstruction_points >= int(
                relations_cfg.get("obstruction_minimum_elevated_points", 500))
            and obstruction_fraction >= float(
                relations_cfg.get("obstruction_minimum_elevated_fraction", 0.15))
        )
        if materially_obstructed:
            return {
                "relation": "FITS_SET_ON",
                "status": "FALSE",
                "reason": "SURFACE_OBSTRUCTED",
                "obstruction_evidence": surface.obstruction_evidence,
                "obstruction_fraction": obstruction_fraction,
                "method": "observed_obstruction_gate_v1",
            }
        d, f, s = driver.current_geometric_properties, fastener.current_geometric_properties, surface.current_geometric_properties
        values = (d.get("footprint_length_m"), d.get("footprint_width_m"), f.get("footprint_length_m"),
                  f.get("footprint_width_m"), s.get("support_length_m"), s.get("support_width_m"))
        if any(value is None for value in values):
            return {"relation": "FITS_SET_ON", "status": "UNKNOWN", "reason": "MEASURED_FOOTPRINT_MISSING"}
        dl, dw, fl, fw, sl, sw = map(float, values)
        cfg = relations_cfg
        edge, gap = float(cfg.get("packing_edge_clearance_m", 0.006)), float(cfg.get("packing_inter_object_clearance_m", 0.006))
        tested = []
        for drot in (0, 90):
            dlen, dwid = (dl, dw) if drot == 0 else (dw, dl)
            for frot in (0, 90):
                flen, fwid = (fl, fw) if frot == 0 else (fw, fl)
                for axis in ("LENGTH", "WIDTH"):
                    used_l = dlen + flen + gap if axis == "LENGTH" else max(dlen, flen)
                    used_w = max(dwid, fwid) if axis == "LENGTH" else dwid + fwid + gap
                    lm, wm = sl - used_l - 2.0 * edge, sw - used_w - 2.0 * edge
                    tested.append({"driver_orientation_degrees": drot, "fastener_orientation_degrees": frot,
                        "arrangement_axis": axis, "length_margin_m": lm, "width_margin_m": wm,
                        "signed_margin_m": min(lm, wm), "fits": lm >= 0.0 and wm >= 0.0})
        selected = max(tested, key=lambda item: item["signed_margin_m"])
        return {"relation": "FITS_SET_ON", "status": "TRUE" if selected["fits"] else "FALSE",
            "tested_arrangements": tested, "selected_arrangement": selected, "support_length_m": sl,
            "support_width_m": sw, "edge_clearance_m": edge, "inter_object_clearance_m": gap,
            "method": "oriented_two_rectangle_packing_v2"}

    def check_relational_packing(self, driver_track: ObservedObjectTrack, fastener_track: ObservedObjectTrack,
                                 surface_region: ObservedRegion) -> tuple[bool, dict[str, Any]]:
        result = self.evaluate_fits_set_on(driver_track, fastener_track, surface_region)
        return result["status"] == "TRUE", result

    def evaluate_fits_in(self, fastener: ObservedObjectTrack, container: ObservedRegion) -> dict[str, Any]:
        self.relation_call_counts["FITS_IN"] += 1
        fp, cp = fastener.current_geometric_properties, container.current_geometric_properties
        length, cross = fp.get("total_length_m"), fp.get("shaft_diameter_m", fp.get("maximum_cross_section_m"))
        opening_l, opening_w, depth = cp.get("opening_length_m"), cp.get("opening_width_m"), cp.get("cavity_depth_m")
        if any(v is None for v in (length, cross, opening_l, opening_w, depth)):
            return {"relation": "FITS_IN", "status": "UNKNOWN", "reason": "MEASURED_DIMENSION_MISSING"}
        clearance = float(self.config.get("relations", {}).get("container_clearance_m", 0.003))
        margins = (max(float(opening_l), float(opening_w)) - float(length) - clearance,
                   min(float(opening_l), float(opening_w)) - float(cross) - clearance,
                   float(depth) - float(cross) - clearance)
        return {"relation": "FITS_IN", "status": "TRUE" if min(margins) >= 0.0 else "FALSE",
            "length_margin_m": margins[0], "width_margin_m": margins[1], "depth_margin_m": margins[2],
            "method": "observed_fastener_vs_observed_cavity_v2"}
