"""Neutral, baseline-owned physical fixed-scene evidence provider."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import Domain
from .contracts import FixedSceneEvidence, ViLaInObservation


# Public domain fixtures mapped to candidate simulator bodies/geoms and PDDL types.
# This represents public domain knowledge, NOT benchmark variant truth.
DOMAIN_FIXED_FIXTURES: Mapping[str, Mapping[str, tuple[str, tuple[str, ...]]]] = {
    "kitchen": {
        "countertop": ("surface", ("countertop",)),
        "serving_area": ("surface", ("serving_area",)),
        "d1": ("storage", ("drawer_D1_frame", "drawer_D1_tray")),
        "d2": ("storage", ("drawer_D2_frame", "drawer_D2_tray")),
        "c2": ("storage", ("cabinet_C2", "C2_door")),
        "b1": ("storage", ("box_B1", "B1_lid")),
        "c1": ("storage", ("cabinet_C1", "C1_door")),
    },
    "living_room": {
        "personal_table_left": ("support", ("a2_personal_left",)),
        "personal_table_right": ("support", ("a2_personal_right",)),
        "shared_table": ("support", ("a2_control_table",)),
        "staging": ("location", ("a2_staging", "l2_staging_table")),
    },
    "workshop": {
        "left_drawer": ("storage", ("left_tool_drawer", "left_tool_drawer_frame")),
        "right_drawer": ("storage", ("right_tool_drawer", "right_tool_drawer_frame")),
        "tool_cabinet": ("storage", ("tool_cabinet", "tool_cabinet_door")),
        "main_workbench_zone": (
            "target",
            ("workshop_frame_fixture", "workshop_frame_joint", "workbench"),
        ),
        "main_workbench_zone_surface": ("surface", ("workbench",)),
    },
}


def project_world_aabb_to_camera(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    intrinsics: Sequence[Sequence[float]],
    extrinsics: Sequence[Sequence[float]],
    image_width: int = 640,
    image_height: int = 480,
) -> list[int] | None:
    """Project a 3-D axis-aligned bounding box into camera coordinates [0, 1000]."""
    t_w_c = np.asarray(extrinsics, dtype=np.float64)
    r_w_c = t_w_c[:3, :3]
    p_w_c = t_w_c[:3, 3]
    r_c_w = r_w_c.T
    p_c_w = -r_c_w @ p_w_c
    k_mat = np.asarray(intrinsics, dtype=np.float64)

    corners = np.array(
        [
            [lower[0], lower[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], upper[1], upper[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], lower[2]],
            [upper[0], upper[1], upper[2]],
        ],
        dtype=np.float64,
    )

    pts_cam = (corners @ r_c_w.T) + p_c_w
    if np.all(pts_cam[:, 2] <= 0.05):
        return None

    pts_cam_clamped = pts_cam.copy()
    pts_cam_clamped[pts_cam_clamped[:, 2] < 0.05, 2] = 0.05
    proj = pts_cam_clamped @ k_mat.T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    x1 = int(np.clip(np.min(u), 0, image_width) / image_width * 1000)
    x2 = int(np.clip(np.max(u), 0, image_width) / image_width * 1000)
    y1 = int(np.clip(np.min(v), 0, image_height) / image_height * 1000)
    y2 = int(np.clip(np.max(v), 0, image_height) / image_height * 1000)

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class FixedSceneEvidenceProvider:
    """Extract neutral physical presence and geometry of static fixtures from the live scene."""

    def __init__(self, *, scene: Any, domain: Domain | str) -> None:
        self.scene = scene
        self.domain_key = domain.value if isinstance(domain, Domain) else str(domain).lower()
        if self.domain_key not in DOMAIN_FIXED_FIXTURES:
            raise ValueError(f"unknown domain for fixed-scene evidence: {self.domain_key}")

    def __call__(
        self,
        *,
        observations: Sequence[ViLaInObservation] = (),
        observation_root: str | Path | None = None,
    ) -> tuple[FixedSceneEvidence, ...]:
        import mujoco

        model = getattr(self.scene, "model", None)
        data = getattr(self.scene, "data", None)
        if model is None or data is None:
            return ()

        mujoco.mj_forward(model, data)

        # Load camera calibrations if observations and observation_root are provided
        camera_calibs: list[tuple[str, str, Sequence[Sequence[float]], Sequence[Sequence[float]], int, int]] = []
        if observations and observation_root is not None:
            obs_root = Path(observation_root)
            for obs in observations:
                for frame in obs.camera_frames:
                    calib_path = obs_root / frame.calibration_path
                    if calib_path.is_file():
                        try:
                            calib_data = json.loads(calib_path.read_text(encoding="utf-8"))
                            shape = calib_data.get("image_shape", [480, 640])
                            h, w = int(shape[0]), int(shape[1])
                            camera_calibs.append(
                                (
                                    obs.stage_id,
                                    frame.camera_id,
                                    calib_data["intrinsics"],
                                    calib_data["extrinsics"],
                                    w,
                                    h,
                                )
                            )
                        except Exception:
                            pass

        fixture_table = DOMAIN_FIXED_FIXTURES[self.domain_key]
        evidence_list: list[FixedSceneEvidence] = []

        for symbolic_id, (pddl_type, candidate_bodies) in fixture_table.items():
            found_body = None
            geom_ids = []
            for bname in candidate_bodies:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
                if bid >= 0:
                    if found_body is None:
                        found_body = bname
                    gstart = int(model.body_geomadr[bid])
                    gnum = int(model.body_geomnum[bid])
                    geom_ids.extend(range(gstart, gstart + gnum))

            if not geom_ids or found_body is None:
                evidence_list.append(
                    FixedSceneEvidence(
                        symbolic_id=symbolic_id,
                        pddl_type=pddl_type,
                        entity_name=None,
                        physically_present=False,
                        centroid_m=None,
                        aabb_min_m=None,
                        aabb_max_m=None,
                        camera_projections=(),
                        description=f"Public fixture {symbolic_id!r} is physically absent from current scene",
                    )
                )
                continue

            lower = np.full(3, np.inf)
            upper = np.full(3, -np.inf)
            for gid in geom_ids:
                rot = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
                c = np.asarray(data.geom_xpos[gid]) + rot @ np.asarray(model.geom_aabb[gid, :3])
                h = np.abs(rot) @ np.asarray(model.geom_aabb[gid, 3:])
                lower = np.minimum(lower, c - h)
                upper = np.maximum(upper, c + h)

            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                evidence_list.append(
                    FixedSceneEvidence(
                        symbolic_id=symbolic_id,
                        pddl_type=pddl_type,
                        entity_name=found_body,
                        physically_present=False,
                        centroid_m=None,
                        aabb_min_m=None,
                        aabb_max_m=None,
                        camera_projections=(),
                        description=f"Public fixture {symbolic_id!r} geometry non-finite",
                    )
                )
                continue

            centroid = (lower + upper) / 2.0
            projections: list[dict[str, Any]] = []
            for stage_id, cam_id, k_mat, t_mat, w, h in camera_calibs:
                bbox_1000 = project_world_aabb_to_camera(
                    lower,
                    upper,
                    intrinsics=k_mat,
                    extrinsics=t_mat,
                    image_width=w,
                    image_height=h,
                )
                if bbox_1000 is not None:
                    projections.append(
                        {
                            "stage_id": stage_id,
                            "camera_id": cam_id,
                            "bbox_1000": bbox_1000,
                        }
                    )

            evidence_list.append(
                FixedSceneEvidence(
                    symbolic_id=symbolic_id,
                    pddl_type=pddl_type,
                    entity_name=found_body,
                    physically_present=True,
                    centroid_m=tuple(float(x) for x in centroid),
                    aabb_min_m=tuple(float(x) for x in lower),
                    aabb_max_m=tuple(float(x) for x in upper),
                    camera_projections=tuple(projections),
                    description=f"Public fixture {symbolic_id!r} physically present at {found_body!r}",
                )
            )

        return tuple(evidence_list)
