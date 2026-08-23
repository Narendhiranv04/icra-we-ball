"""Privileged evaluation layer and metric computation for Workshop Phase 1.

NEVER called during production execution; operates strictly post-episode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None

from mujoco_scenes.workshop_phase1.types import EpisodeResult, ObservedObjectTrack, ObservedRegion


class PrivilegedPhase1Evaluator:
    """Computes post-hoc evaluation metrics by comparing generic production tracks to simulator ground truth."""

    def __init__(self, scene: Any) -> None:
        self.scene = scene

    @staticmethod
    def _canonical_object_category(name: str) -> str | None:
        lowered = name.lower()
        if "power" in lowered and "driver" in lowered:
            return "power_driver"
        if "driver" in lowered or "screwdriver" in lowered:
            return "screwdriver"
        if "screw" in lowered:
            return "screw"
        if "bolt" in lowered:
            return "bolt"
        if "wrench" in lowered:
            return "wrench"
        if "pliers" in lowered:
            return "pliers"
        if "hammer" in lowered or "mallet" in lowered:
            return "hammer"
        return None

    @staticmethod
    def _canonical_region_category(name: str) -> str | None:
        return {
            "MAIN_WORKBENCH_ZONE": "workbench",
            "TOOL_CART_TOP": "tool_cart",
            "NARROW_WALL_SHELF": "shelf",
            "PARTS_TRAY": "parts_tray",
            "HARDWARE_BIN": "hardware_bin",
        }.get(name)

    def evaluate_semantic_diagnostics(
        self,
        detection_records: list[dict[str, Any]],
        gt_objects: list[dict[str, Any]],
        regions: list[ObservedRegion],
        region_to_gt: dict[str, str],
    ) -> dict[str, Any]:
        """Post-hoc matching only; results never flow back into grounding."""
        categories = [
            "screwdriver", "power_driver", "screw", "hammer",
            "workbench", "tool_cart", "shelf", "parts_tray", "hardware_bin",
        ]
        rows = {category: {
            "visible_gt_count": 0, "yolo_detection_count": 0,
            "matched_gt_count": 0, "false_positive_count": 0,
            "duplicate_detections": 0, "association_correct_count": 0,
            "association_evaluated_count": 0,
        } for category in categories}

        gt_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in categories}
        for gt in gt_objects:
            category = self._canonical_object_category(gt["name"])
            if category:
                gt_by_category[category].append(gt)
        active_names = set(getattr(self.scene, "active_surfaces", [])) | set(
            getattr(self.scene, "active_containers", []))
        active_region_categories = {
            category for name in active_names
            if (category := self._canonical_region_category(name)) is not None
        }
        for name in active_names:
            category = self._canonical_region_category(name)
            if category:
                rows[category]["visible_gt_count"] += 1
        for category, objects in gt_by_category.items():
            rows[category]["visible_gt_count"] += len(objects)

        accepted = [record for record in detection_records if record.get("status") == "ACCEPTED"]
        matched_ids_by_category: dict[str, list[str]] = {category: [] for category in categories}
        region_detection_matches: dict[tuple[int, str], tuple[str, str]] = {}
        for region in regions:
            expected = self._canonical_region_category(region_to_gt.get(region.region_instance_id, ""))
            for observation in region.semantic_observations:
                detection_id = observation.get("detection_id")
                if detection_id and expected in active_region_categories:
                    region_detection_matches[(int(observation.get("stage_index", 0)), detection_id)] = (
                        expected, region_to_gt[region.region_instance_id])

        for record in accepted:
            category = record.get("canonical_label")
            if category not in rows:
                continue
            rows[category]["yolo_detection_count"] += 1
            detection_key = record.get("detection_id")
            if category in {"workbench", "tool_cart", "shelf", "parts_tray", "hardware_bin"}:
                match = region_detection_matches.get((int(record.get("stage_index", 0)), detection_key))
                expected = match[0] if match else None
                rows[category]["association_evaluated_count"] += int(expected is not None)
                rows[category]["association_correct_count"] += int(expected == category)
                if expected is None:
                    rows[category]["false_positive_count"] += 1
                elif expected == category:
                    matched_ids_by_category[category].append(match[1])
                continue
            centroid = record.get("centroid_world_m")
            best = None
            if centroid is not None:
                for gt in gt_objects:
                    distance = float(np.linalg.norm(np.asarray(centroid) - gt["pos"]))
                    if distance <= 0.16 and (best is None or distance < best[0]):
                        best = (distance, gt)
            if best is None:
                rows[category]["false_positive_count"] += 1
                continue
            expected = self._canonical_object_category(best[1]["name"])
            rows[category]["association_evaluated_count"] += 1
            rows[category]["association_correct_count"] += int(expected == category)
            if expected == category:
                matched_ids_by_category[category].append(best[1]["name"])
            else:
                rows[category]["false_positive_count"] += 1

        for category, row in rows.items():
            matched = matched_ids_by_category[category]
            row["matched_gt_count"] = len(set(matched))
            row["duplicate_detections"] = max(0, len(matched) - len(set(matched)))
            row["recall"] = round(row["matched_gt_count"] / max(1, row["visible_gt_count"]), 4)
            row["association_accuracy"] = round(
                row["association_correct_count"] / max(1, row["association_evaluated_count"]), 4)
        return {"categories": rows, "privileged_use": "POST_HOC_EVALUATION_ONLY"}

    def evaluate_episode(
        self,
        result: EpisodeResult,
        tracks: list[ObservedObjectTrack],
        regions: list[ObservedRegion],
        output_dir: Path | None = None,
        detection_diagnostics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate generic predictions against privileged oracle metadata."""
        if mujoco is None:
            return {"error": "MuJoCo not available"}

        variant_meta = self.scene.variant_meta
        expected_status = variant_meta.get("intended_outcome", variant_meta.get("expected_feasibility", "FEASIBLE"))
        expected_rejection = variant_meta.get("rejection_reason", variant_meta.get("expected_rejection_reason"))
        variant_name = variant_meta.get("variant_name", getattr(self.scene, "variant_name", ""))

        # 1. Extract ground truth free-body objects and their geometric AABB centers
        gt_objects = []
        for bid in range(self.scene.model.nbody):
            bname = mujoco.mj_id2name(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not bname or self.scene.model.body_dofnum[bid] != 6:
                continue

            b_min = np.array([np.inf, np.inf, np.inf])
            b_max = np.array([-np.inf, -np.inf, -np.inf])
            for gid in range(self.scene.model.ngeom):
                if self.scene.model.geom_bodyid[gid] == bid:
                    gpos = self.scene.data.geom_xpos[gid]
                    gmat = self.scene.data.geom_xmat[gid].reshape(3, 3)
                    gtype = self.scene.model.geom_type[gid]
                    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                        mid = self.scene.model.geom_dataid[gid]
                        v_start = self.scene.model.mesh_vertadr[mid]
                        v_num = self.scene.model.mesh_vertnum[mid]
                        verts = self.scene.model.mesh_vert[v_start:v_start + v_num]
                        wverts = (gmat @ verts.T).T + gpos
                        b_min = np.minimum(b_min, wverts.min(axis=0))
                        b_max = np.maximum(b_max, wverts.max(axis=0))
                    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        gsz = self.scene.model.geom_size[gid]
                        corners = np.array([[sx * gsz[0], sy * gsz[1], sz * gsz[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                        wcorners = (gmat @ corners.T).T + gpos
                        b_min = np.minimum(b_min, wcorners.min(axis=0))
                        b_max = np.maximum(b_max, wcorners.max(axis=0))

            if np.all(np.isfinite(b_min)):
                center = (b_min + b_max) / 2.0
            else:
                center = self.scene.data.xpos[bid].copy()

            gt_objects.append({"name": bname, "body_id": bid, "pos": center})

        # 2. Strict 1-to-1 Bipartite Hungarian matching between predicted tracks and GT objects
        track_to_gt: dict[str, str] = {}
        valid_tracks = [t for t in tracks if t.fused_points is not None and len(t.fused_points) > 0]

        if valid_tracks and gt_objects:
            cost_matrix = np.full((len(valid_tracks), len(gt_objects)), 1e6, dtype=float)
            for i, trk in enumerate(valid_tracks):
                trk_center = trk.fused_points.mean(axis=0)
                for j, gt in enumerate(gt_objects):
                    dist = float(np.linalg.norm(trk_center - gt["pos"]))
                    if dist <= 0.16:
                        cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= 0.16:
                    track_to_gt[valid_tracks[r].instance_id] = gt_objects[c]["name"]

        # 3. Map generic regions to backend region names
        region_to_gt: dict[str, str] = {}
        proposals = self.scene.get_candidate_regions()
        for reg in regions:
            r_id = reg.region_instance_id
            r_min = np.array(reg.proposal_bounds_m["minimum_world_m"])
            r_max = np.array(reg.proposal_bounds_m["maximum_world_m"])
            r_center = (r_min + r_max) / 2.0
            for prop in proposals:
                p_min = np.array(prop["proposal_bounds_m"]["minimum_world_m"])
                p_max = np.array(prop["proposal_bounds_m"]["maximum_world_m"])
                p_center = (p_min + p_max) / 2.0
                if np.linalg.norm(r_center - p_center) < 0.05:
                    backend_reg_name = self.scene.privileged_backend_name_for_region(prop["region_instance_id"])
                    region_to_gt[r_id] = backend_reg_name

        # 4. Assess status & rejection reason
        status_correct = (result.status == expected_status)
        rejection_correct = True
        if expected_status == "INFEASIBLE":
            rejection_correct = (result.rejection_reason == expected_rejection)

        # 5. Assess witness validity against exact variant-valid sets
        witness_correct = False
        witness_details = {}

        if result.status == "FEASIBLE" and result.witness is not None and expected_status == "FEASIBLE":
            pred_d_gt = track_to_gt.get(result.witness.driver_id)
            pred_f_gt = track_to_gt.get(result.witness.fastener_id)
            expected_solution = variant_meta.get("expected_solution", {})
            expected_driver = expected_solution.get("driver")
            expected_fastener = expected_solution.get(
                "fastener", "workshop_medium_phillips_screw")
            d_ok = pred_d_gt == expected_driver
            f_ok = pred_f_gt == expected_fastener
            s_ok = result.witness.work_surface_id == "MAIN_WORKBENCH_ZONE"
            c_ok = result.witness.parts_container_id is None

            witness_correct = bool(d_ok and f_ok and s_ok and c_ok)
            witness_details = {
                "pred_driver_gt": pred_d_gt,
                "pred_fastener_gt": pred_f_gt,
                "fixed_insertion_target": result.witness.work_surface_id,
                "parts_container": result.witness.parts_container_id,
                "driver_ok": d_ok,
                "fastener_ok": f_ok,
                "surface_ok": s_ok,
                "container_ok": c_ok,
            }

        overall_pass = bool(status_correct and (rejection_correct if expected_status == "INFEASIBLE" else witness_correct))

        metrics = {
            "overall_pass": overall_pass,
            "status_correct": status_correct,
            "expected_status": expected_status,
            "predicted_status": result.status,
            "rejection_correct": rejection_correct,
            "expected_rejection": expected_rejection,
            "predicted_rejection": result.rejection_reason,
            "witness_correct": witness_correct,
            "witness_details": witness_details,
            "tracked_objects_count": len(tracks),
            "discovered_gt_objects_count": len(set(track_to_gt.values())),
            "total_gt_objects_in_scene": len(gt_objects),
            "stages_used": result.metrics.get("stages_executed", 1),
            "early_stopped": result.metrics.get("early_stopped", False),
        }

        if detection_diagnostics is not None:
            metrics["semantic_diagnostics"] = self.evaluate_semantic_diagnostics(
                detection_diagnostics, gt_objects, regions, region_to_gt)
            status_counts: dict[str, int] = {}
            for record in detection_diagnostics:
                status = str(record.get("status", "UNKNOWN"))
                status_counts[status] = status_counts.get(status, 0) + 1
            accepted = status_counts.get("ACCEPTED", 0)
            rejected_refinement = sum(
                count for status, count in status_counts.items()
                if status in {"REJECTED_DEPTH_SUPPORT", "REJECTED_REFINEMENT",
                              "REJECTED_NO_COHERENT_COMPONENT"})
            rejected_volume = status_counts.get("REJECTED_STAGE_VOLUME", 0)
            evaluated = accepted + rejected_refinement + rejected_volume
            camera_stages = {
                (int(record.get("stage_index", 0)), str(record.get("camera_id", "")))
                for record in detection_diagnostics if record.get("camera_id")
            }
            metrics["proposal_quality"] = {
                "status_counts": status_counts,
                "accepted_physical_proposals": accepted,
                "evaluated_camera_stages": len(camera_stages),
                "average_accepted_proposals_per_camera_stage": round(
                    accepted / max(1, len(camera_stages)), 4),
                "mask_refinement_rejection_rate": round(
                    rejected_refinement / max(1, evaluated), 4),
                "stage_volume_rejection_rate": round(
                    rejected_volume / max(1, evaluated), 4),
                "suppressed_duplicate_count": status_counts.get("SUPPRESSED_DUPLICATE", 0),
            }

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "privileged_eval_mapping.json", "w", encoding="utf-8") as f:
                json.dump({"track_to_gt": track_to_gt, "region_to_gt": region_to_gt}, f, indent=2)
            with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if detection_diagnostics is not None:
                with open(output_dir / "semantic_diagnostics.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "detector_records": detection_diagnostics,
                        "summary": metrics["semantic_diagnostics"],
                    }, f, indent=2)

        return metrics
