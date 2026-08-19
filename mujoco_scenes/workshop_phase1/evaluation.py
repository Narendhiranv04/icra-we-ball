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
            pred_s_gt = region_to_gt.get(result.witness.work_surface_id)
            pred_c_gt = region_to_gt.get(result.witness.parts_container_id)

            if variant_name == "F4_OBJECT_REGION_COUPLING":
                d_ok = (pred_d_gt == "workshop_power_driver")
                f_ok = (pred_f_gt == "workshop_medium_phillips_screw")
                s_ok = (pred_s_gt == "TOOL_CART_TOP")  # NARROW_WALL_SHELF fails packing
                c_ok = (pred_c_gt in ("PARTS_TRAY", "HARDWARE_BIN"))
            elif variant_name == "F1_TOOL_ALTERNATIVE":
                d_ok = (pred_d_gt in ("workshop_power_driver", "workshop_long_phillips_driver"))
                f_ok = (pred_f_gt == "workshop_medium_phillips_screw")
                s_ok = (pred_s_gt in ("MAIN_WORKBENCH_ZONE", "TOOL_CART_TOP"))
                c_ok = (pred_c_gt in ("HARDWARE_BIN", "PARTS_TRAY"))
            elif variant_name == "F2_REGION_ALTERNATIVE":
                d_ok = (pred_d_gt in ("workshop_long_phillips_driver", "workshop_power_driver"))
                f_ok = (pred_f_gt == "workshop_medium_phillips_screw")
                s_ok = (pred_s_gt == "TOOL_CART_TOP")  # Workbench is obstructed
                c_ok = (pred_c_gt in ("HARDWARE_BIN", "PARTS_TRAY"))
            elif variant_name == "F3_DISTRIBUTED_OBJECTS":
                d_ok = (pred_d_gt in ("workshop_long_phillips_driver", "workshop_power_driver"))
                f_ok = (pred_f_gt == "workshop_medium_phillips_screw")
                s_ok = (pred_s_gt in ("TOOL_CART_TOP", "MAIN_WORKBENCH_ZONE"))
                c_ok = (pred_c_gt in ("PARTS_TRAY", "HARDWARE_BIN"))
            elif variant_name == "F5_DECOY_HEAVY":
                d_ok = (pred_d_gt in ("workshop_long_phillips_driver", "workshop_power_driver"))
                f_ok = (pred_f_gt == "workshop_medium_phillips_screw")
                s_ok = (pred_s_gt in ("MAIN_WORKBENCH_ZONE", "TOOL_CART_TOP"))
                c_ok = (pred_c_gt in ("PARTS_TRAY", "HARDWARE_BIN"))
            elif variant_name == "F6_LAYOUT_SWAPPED":
                d_ok = (pred_d_gt in ("workshop_long_phillips_driver", "workshop_power_driver"))
                f_ok = (pred_f_gt == "workshop_medium_phillips_screw")
                s_ok = (pred_s_gt in ("MAIN_WORKBENCH_ZONE", "TOOL_CART_TOP"))
                c_ok = (pred_c_gt in ("HARDWARE_BIN", "PARTS_TRAY"))
            else:  # F0_BASE
                d_ok = (pred_d_gt in ("workshop_long_phillips_driver", "workshop_power_driver"))
                f_ok = (pred_f_gt == "workshop_medium_phillips_screw")
                s_ok = (pred_s_gt in ("MAIN_WORKBENCH_ZONE", "TOOL_CART_TOP"))
                c_ok = (pred_c_gt in ("HARDWARE_BIN", "PARTS_TRAY"))

            witness_correct = bool(d_ok and f_ok and s_ok and c_ok)
            witness_details = {
                "pred_driver_gt": pred_d_gt,
                "pred_fastener_gt": pred_f_gt,
                "pred_surface_gt": pred_s_gt,
                "pred_container_gt": pred_c_gt,
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

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "privileged_eval_mapping.json", "w", encoding="utf-8") as f:
                json.dump({"track_to_gt": track_to_gt, "region_to_gt": region_to_gt}, f, indent=2)
            with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        return metrics
