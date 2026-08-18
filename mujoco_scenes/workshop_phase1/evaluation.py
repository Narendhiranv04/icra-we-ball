"""Privileged evaluation layer and metric computation for Workshop Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None

from mujoco_scenes.workshop_phase1.types import EpisodeResult, ObservedObjectTrack, ObservedRegion


class PrivilegedPhase1Evaluator:
    """Computes post-hoc evaluation metrics by comparing generic production tracks to simulator ground truth.

    NEVER called during production execution; operates strictly post-episode.
    """

    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def evaluate_episode(
        self,
        result: EpisodeResult,
        tracks: list[ObservedObjectTrack],
        regions: list[ObservedRegion],
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate generic predictions against privileged oracle metadata."""
        if mujoco is None:
            return {"error": "MuJoCo not available"}

        variant_meta = self.scene.variant_meta
        expected_status = variant_meta.get("intended_outcome", variant_meta.get("expected_feasibility", "FEASIBLE"))
        expected_rejection = variant_meta.get("rejection_reason", variant_meta.get("expected_rejection_reason"))
        expected_sol = variant_meta.get("expected_solution", {})

        # 1. Map generic tracks to backend bodies
        gt_objects = []
        for bid in range(self.scene.model.nbody):
            bname = mujoco.mj_id2name(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not bname or self.scene.model.body_dofnum[bid] != 6:
                continue

            # Compute AABB center from geoms
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
                        corners = np.array([[sx*gsz[0], sy*gsz[1], sz*gsz[2]] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
                        wcorners = (gmat @ corners.T).T + gpos
                        b_min = np.minimum(b_min, wcorners.min(axis=0))
                        b_max = np.maximum(b_max, wcorners.max(axis=0))

            if np.all(np.isfinite(b_min)):
                center = (b_min + b_max) / 2.0
            else:
                center = self.scene.data.xpos[bid].copy()

            gt_objects.append({"name": bname, "body_id": bid, "pos": center})

        track_to_gt: dict[str, str] = {}
        for trk in tracks:
            if trk.fused_points is None or len(trk.fused_points) == 0:
                continue
            trk_center = trk.fused_points.mean(axis=0)
            best_gt = None
            min_dist = np.inf
            for gt in gt_objects:
                dist = float(np.linalg.norm(trk_center - gt["pos"]))
                if dist < 0.12 and dist < min_dist:
                    min_dist = dist
                    best_gt = gt["name"]
            if best_gt:
                track_to_gt[trk.instance_id] = best_gt

        # 2. Map generic regions to backend region names
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

        # 3. Assess status & witness accuracy
        status_correct = (result.status == expected_status)
        rejection_correct = True
        if expected_status == "INFEASIBLE":
            rejection_correct = (result.rejection_reason == expected_rejection)

        witness_correct = False
        witness_details = {}
        if result.status == "FEASIBLE" and result.witness is not None and expected_status == "FEASIBLE":
            pred_driver_gt = track_to_gt.get(result.witness.driver_id)
            pred_fastener_gt = track_to_gt.get(result.witness.fastener_id)
            pred_surface_gt = region_to_gt.get(result.witness.work_surface_id)
            pred_container_gt = region_to_gt.get(result.witness.parts_container_id)

            exp_driver = expected_sol.get("tool") or expected_sol.get("driver")
            exp_fastener = expected_sol.get("fastener")
            exp_surface = expected_sol.get("work_surface")
            exp_container = expected_sol.get("parts_container")

            # Check if alternative drivers are acceptable (e.g. F1 allows long, flathead, power)
            driver_match = (pred_driver_gt == exp_driver) or (
                pred_driver_gt in ("workshop_long_phillips_driver", "workshop_power_driver")
            )
            fastener_match = (pred_fastener_gt == exp_fastener)
            surface_match = (pred_surface_gt == exp_surface) or (
                exp_surface is None and pred_surface_gt in ("MAIN_WORKBENCH_ZONE", "TOOL_CART_TOP")
            )
            container_match = (pred_container_gt == exp_container) or (
                exp_container is None and pred_container_gt in ("PARTS_TRAY", "HARDWARE_BIN")
            )

            witness_correct = bool(driver_match and fastener_match and surface_match and container_match)
            witness_details = {
                "pred_driver_gt": pred_driver_gt,
                "pred_fastener_gt": pred_fastener_gt,
                "pred_surface_gt": pred_surface_gt,
                "pred_container_gt": pred_container_gt,
                "exp_driver": exp_driver,
                "exp_fastener": exp_fastener,
                "exp_surface": exp_surface,
                "exp_container": exp_container,
                "driver_match": driver_match,
                "fastener_match": fastener_match,
                "surface_match": surface_match,
                "container_match": container_match,
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

        # Save evaluation artifacts
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            eval_mapping = {
                "track_to_gt": track_to_gt,
                "region_to_gt": region_to_gt,
            }
            with open(output_dir / "privileged_eval_mapping.json", "w", encoding="utf-8") as f:
                json.dump(eval_mapping, f, indent=2)
            with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        return metrics
