"""Live physical adapter for discovery-based replanning in the L2 living room."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from vlm_tamp_baseline.living_room_runtime import (
    STAGING_REGION_GEOM,
    STAGING_REGION_ID,
    LivingRoomPlanningRuntime,
)

from .generic_manipulation import CalibratedPickPlaceExecutor
from .living_room_mobile_execution import (
    ALLOWED_INTERACTION_BODIES,
    SETTLE_MAXIMUM_STEPS,
    SETTLE_ORIENTATION_DRIFT_RAD,
    SETTLE_POSITION_DRIFT_M,
    SETTLE_STABLE_WINDOWS,
    SETTLE_WINDOW_STEPS,
    SUPPORT_HEIGHT,
    LivingRoomMobileExecutor,
    _actual_body_yaw,
    _carry_position,
    _configure_execution_base_limits,
    _world_to_joint_base,
    candidate_stances,
    inspect_held_object_state,
    make_pick_specs,
    resume_held_object_from_simulator,
    stance_holds_under_settle,
    validate_manipulation_at_pose,
    verify_physical_on_relation,
)
from .robot_profiles import manipulation_profile
from .tamp.physical_dispatcher import MuJoCoSkillDispatcher


# Minimum free gap left between two payload footprints sharing one support.
PLACEMENT_CLEARANCE_M = 0.025

ROLE_FOOTPRINTS = {
    "cup": (0.11, 0.09),
    "saucer": (0.19, 0.18),
    "tv_remote": (0.20, 0.08),
}


class LivingRoomDiscoveryRuntime(LivingRoomPlanningRuntime):
    """Five-camera Living Room observation plus guarded Google-robot skills."""

    def __init__(
        self,
        variant: str,
        output_dir: str | Path,
        *,
        camera_count: int = 5,
        show_viewer: bool = True,
        viewer_camera: str = "free",
        image_width: int = 960,
        image_height: int = 540,
    ):
        super().__init__(
            variant,
            Path(output_dir),
            camera_count=camera_count,
            robot="google",
            physical_execution=True,
            image_width=image_width,
            image_height=image_height,
        )
        _configure_execution_base_limits(self.scene)
        mujoco.mj_resetData(self.scene.model, self.scene.data)
        self.scene._set_robot_home_pose()
        mujoco.mj_forward(self.scene.model, self.scene.data)
        self._restore_payload_poses()
        self.mobile = LivingRoomMobileExecutor(self.scene.model, self.scene.data)
        self.pick_specs = make_pick_specs(self.payload_registry, self.resolution)
        self.held_object: str | None = None
        self._placement_records: dict[str, dict[str, Any]] = {}
        # Payload yaw induced by each grasp, cancelled at placement time.
        self._grasp_yaw_offsets: dict[str, float] = {}
        self.viewer = None
        self.show_viewer = show_viewer
        self.viewer_camera = viewer_camera
        self.status = "Living Room discovery runtime ready"
        self.dispatcher = MuJoCoSkillDispatcher(self)

    @property
    def object_annotation_aliases(self) -> dict[str, str]:
        return dict(self.object_aliases)

    def _restore_payload_poses(self) -> None:
        for row in self.resolution["objects"]:
            object_id = str(row["generic_object_id"])
            body_name = str(row["backend_body"])
            body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = int(self.scene.model.body_jntadr[body_id])
            qpos = int(self.scene.model.jnt_qposadr[joint_id])
            dof = int(self.scene.model.jnt_dofadr[joint_id])
            pose = self.payload_registry["objects"][object_id]["observed_centroid_world_m"]
            self.scene.data.qpos[qpos : qpos + 3] = pose
            self.scene.data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
            self.scene.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.scene.model, self.scene.data)

    def open(self) -> None:
        if not self.show_viewer:
            return
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.scene.model, self.scene.data)
        if self.viewer_camera == "free":
            mujoco.mjv_defaultFreeCamera(self.scene.model, self.viewer.cam)
        else:
            camera_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_CAMERA, self.viewer_camera
            )
            if camera_id < 0:
                raise ValueError(f"Unknown Living Room viewer camera {self.viewer_camera!r}")
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = camera_id
        self.sync(self.status)

    def close(self) -> None:
        super().close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def sync(self, status: str) -> None:
        self.status = status
        if self.viewer is None or not self.viewer.is_running():
            return
        if hasattr(self.viewer, "set_texts"):
            self.viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Living Room discovery execution",
                    status,
                )
            )
        self.viewer.sync()

    def _step_callback(self) -> None:
        self.sync(self.status)

    def _body(self, object_id: str) -> str:
        try:
            return self.object_backends[object_id]
        except KeyError as error:
            raise ValueError(f"Unknown visible object {object_id!r}") from error

    def _support(self, region_id: str) -> str:
        if region_id == STAGING_REGION_ID:
            return STAGING_REGION_GEOM
        try:
            return self.region_backends[region_id]
        except KeyError as error:
            raise ValueError(f"Unknown visible support region {region_id!r}") from error

    def _select_stance(self, body: str, target: np.ndarray) -> tuple[Any, list[Any] | None]:
        current = self.mobile.current_pose()
        spec = self.pick_specs[body]
        for index, pose in enumerate(candidate_stances(target, current)):
            checker = self.mobile.collision_checker()
            distance = float(np.linalg.norm(target[:2] - np.array((pose.x, pose.y))))
            if not 0.40 <= distance <= 1.10:
                continue
            if not stance_holds_under_settle(checker, pose):
                continue
            feasibility = validate_manipulation_at_pose(
                self.scene.model,
                self.scene.data,
                pose,
                body,
                target,
                spec,
                target_rotation=manipulation_profile("google").top_down_rotation,
            )
            if not feasibility["feasible"]:
                continue
            path = None
            if index:
                try:
                    path = self.mobile.plan(pose)
                except RuntimeError:
                    continue
            return pose, path
        raise RuntimeError(f"No collision-free manipulation stance for {body}")

    def _move_to_stance(self, pose: Any, path: list[Any] | None) -> None:
        if path is None:
            return
        self.status = "Navigating to collision-free manipulation stance"
        self.mobile.execute(path, step_callback=self._step_callback)

    def _run_picker(self, picker: CalibratedPickPlaceExecutor, terminal_mode: str) -> str | None:
        steps = 0
        while picker.mode not in {terminal_mode, "failed"}:
            picker.update()
            mujoco.mj_step(self.scene.model, self.scene.data)
            self._step_callback()
            steps += 1
            if steps > 60_000:
                picker._fail("MANIPULATION_TIMEOUT")
        return picker.failure

    def _picker(self, body: str) -> CalibratedPickPlaceExecutor:
        specs = dict(self.pick_specs)
        return CalibratedPickPlaceExecutor(
            self.scene.model,
            self.scene.data,
            "google",
            pick_specs_override=specs,
            calibrated_objects_override=tuple(specs),
            base_stance=_world_to_joint_base(self.mobile.current_pose()),
            base_approach_forward=0.0,
            arm_command_speed=1.35,
            intermediate_tracking_tolerance=0.065,
            allowed_collision_bodies=ALLOWED_INTERACTION_BODIES,
        )

    def _footprint(self, object_id: str) -> tuple[float, float]:
        return ROLE_FOOTPRINTS[self.object_roles[object_id]]

    def _place_target(self, object_id: str, region_id: str) -> tuple[np.ndarray, float]:
        support = self._support(region_id)
        geom_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, support)
        center = self.scene.data.geom_xpos[geom_id].copy()
        half = self.scene.model.geom_size[geom_id, :2].copy()
        rotation = self.scene.data.geom_xmat[geom_id].reshape(3, 3)
        axis_x, axis_y = rotation[:, 0], rotation[:, 1]
        length, width = self._footprint(object_id)
        maximum_x = float(half[0] - length / 2 - 0.02)
        maximum_y = float(half[1] - width / 2 - 0.02)
        if maximum_x <= 0 or maximum_y <= 0:
            raise RuntimeError(f"Support {region_id} is too small for {object_id}")
        # Candidate offsets are fractions of the usable envelope.  They reach
        # close to the support edge: a personal side table only has ~0.20 m of
        # usable half-extent, so a grid capped well inside it cannot separate
        # two payloads at all.
        # A support may have to hold two payloads (each personal table takes a
        # cup and a saucer).  Centring the first one strands the second: the
        # larger payload's own envelope is then too small to clear it.  Offer
        # modest off-centre slots first and keep the centre as a late fallback,
        # so two payloads can sit on opposite sides of the same support.
        candidates = (
            (-0.62, 0.0), (0.62, 0.0), (0.0, -0.62), (0.0, 0.62),
            (-0.90, 0.0), (0.90, 0.0), (0.0, -0.90), (0.0, 0.90),
            (-0.62, -0.62), (0.62, 0.62), (-0.62, 0.62), (0.62, -0.62),
            (-0.90, -0.90), (0.90, 0.90), (-0.90, 0.90), (0.90, -0.90),
            (0.0, 0.0),
        )
        occupied = [
            other
            for other, location in self.locations.items()
            if other != object_id and location == region_id
        ]
        for x_fraction, y_fraction in candidates:
            point = center + x_fraction * maximum_x * axis_x + y_fraction * maximum_y * axis_y
            point[2] = center[2] + self.scene.model.geom_size[geom_id, 2] + SUPPORT_HEIGHT[self.object_roles[object_id]]
            valid = True
            for other in occupied:
                other_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, self._body(other))
                other_position = self.scene.data.xpos[other_id]
                other_length, other_width = self._footprint(other)
                # Both payloads are placed axis-aligned with the support, so
                # separation is a rectangle test on the support's own axes.  A
                # circumscribed-circle test overestimates a rectangular
                # footprint by its diagonal and rejects placements that fit.
                offset = point[:2] - other_position[:2]
                separation_x = abs(float(np.dot(offset, axis_x[:2])))
                separation_y = abs(float(np.dot(offset, axis_y[:2])))
                required_x = 0.5 * (length + other_length) + PLACEMENT_CLEARANCE_M
                required_y = 0.5 * (width + other_width) + PLACEMENT_CLEARANCE_M
                if separation_x < required_x and separation_y < required_y:
                    valid = False
                    break
            if valid:
                yaw = math.atan2(float(axis_x[1]), float(axis_x[0]))
                return point, yaw
        raise RuntimeError(f"No collision-free placement point on {region_id}")

    def _verify_place(
        self,
        object_id: str,
        region_id: str,
        body: str,
        target: np.ndarray,
        yaw: float,
        picker: CalibratedPickPlaceExecutor,
    ) -> bool:
        support = self._support(region_id)
        geom_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, support)
        rotation = self.scene.data.geom_xmat[geom_id].reshape(3, 3)
        length, width = self._footprint(object_id)
        record = {
            "object_id": object_id,
            "region_id": region_id,
            "desired_body_world_m": target.tolist(),
            "yaw_world_rad": yaw,
            "footprint_length_m": length,
            "footprint_width_m": width,
            "inter_payload_clearance_m": 0.02,
        }
        region = {
            "geometry": {
                "centroid_world_m": {"value": self.scene.data.geom_xpos[geom_id].tolist()},
                "principal_axis_world": {"value": rotation[:, 0].tolist()},
                "support_length_m": {"value": float(2 * self.scene.model.geom_size[geom_id, 0])},
                "support_width_m": {"value": float(2 * self.scene.model.geom_size[geom_id, 1])},
            }
        }
        checks = verify_physical_on_relation(
            self.scene.model,
            self.scene.data,
            object_id,
            body,
            region_id,
            support,
            region,
            record,
            {**self._placement_records, object_id: record},
            self.object_backends,
            released_by_executor=picker.held_object is None,
            assisted_validation=False,
        )
        self.last_place_verification = checks
        self.last_placement_record = record
        return bool(checks["verified"])

    def _settle_payload(self, body: str) -> bool:
        """Wait until the payload's pose stops changing on its support.

        Settling is measured as pose displacement over a window rather than an
        instantaneous velocity sample.  A resting mesh payload chatters in the
        contact solver: the saucer reports an angular speed alternating between
        0.06 and 0.48 rad/s on consecutive steps while its pose provably does
        not move (0.03 mm and 6e-4 rad per 500 steps), so a velocity threshold
        can never be met no matter how long it waits.  Displacement is both what
        "came to rest" means and what ``_verify_place`` goes on to measure, and
        the bounds below are tighter than the velocity thresholds they replace
        (0.5 mm per 0.1 s is 5 mm/s against the former 20 mm/s).
        """
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
        position = self.scene.data.xpos[body_id].copy()
        orientation = self.scene.data.xquat[body_id].copy()
        stable_windows = 0
        for step in range(1, SETTLE_MAXIMUM_STEPS + 1):
            mujoco.mj_step(self.scene.model, self.scene.data)
            self._step_callback()
            if step % SETTLE_WINDOW_STEPS:
                continue
            previous_position, previous_orientation = position, orientation
            position = self.scene.data.xpos[body_id].copy()
            orientation = self.scene.data.xquat[body_id].copy()
            # Quaternion sign is free, so compare the geodesic angle rather
            # than the component difference.
            alignment = min(1.0, abs(float(np.dot(orientation, previous_orientation))))
            stable = (
                float(np.linalg.norm(position - previous_position))
                <= SETTLE_POSITION_DRIFT_M
                and 2.0 * math.acos(alignment) <= SETTLE_ORIENTATION_DRIFT_RAD
            )
            stable_windows = stable_windows + 1 if stable else 0
            if stable_windows >= SETTLE_STABLE_WINDOWS:
                return True
        return False

    def execute_phase2_action(self, action: dict[str, Any]) -> dict[str, Any]:
        name = str(action.get("action", "")).upper()
        arguments = [str(value) for value in action.get("arguments", ())]
        try:
            if name == "PICK" and len(arguments) == 1:
                return self._pick(arguments[0])
            if name == "PLACE" and len(arguments) == 2:
                return self._place(arguments[0], arguments[1])
            return {"success": False, "failure_code": "UNSUPPORTED", "message": str(action)}
        except (RuntimeError, ValueError) as error:
            return {"success": False, "failure_code": "EXECUTION_FAILED", "message": str(error)}

    def _pick(self, object_id: str) -> dict[str, Any]:
        if self.held_object is not None:
            return {"success": False, "failure_code": "GRIPPER_OCCUPIED"}
        body = self._body(object_id)
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
        target = self.scene.data.xpos[body_id].copy()
        self.status = f"Picking {self.object_aliases[object_id]}"
        pose, path = self._select_stance(body, target)
        self._move_to_stance(pose, path)
        spec = self.pick_specs[body]
        self.pick_specs[body] = replace(spec, carry_position=_carry_position(pose, target))
        picker = self._picker(body)
        picker.request_pick(body)
        failure = self._run_picker(picker, "holding")
        if failure:
            return {"success": False, "failure_code": "GRASP", "message": str(failure)}
        held = inspect_held_object_state(self.scene.model, self.scene.data, object_id, body)
        if held.validation_status != "TRUE":
            return {"success": False, "failure_code": "GRASP", "message": ",".join(held.rejection_reasons)}
        self.held_object = object_id
        self.locations[object_id] = None
        # Closing the fingers on a curved payload twists it in the grasp: a cup
        # measured 0 -> 10.8 deg while the gripper was commanded top-down at
        # yaw 0.  That offset rides along to the placement, so a place that
        # commands the support's yaw lands the payload rotated by it.  Record it
        # here and cancel it in _place, so the commanded orientation is the one
        # actually achieved rather than one the verifier has to forgive.
        self._grasp_yaw_offsets[object_id] = _actual_body_yaw(
            self.scene.data,
            mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, body),
        )
        self.revision += 1
        self.invalidate_images()
        return {"success": True, "effects": [f"holding({object_id})"], "held": asdict(held)}

    def _place(self, object_id: str, region_id: str) -> dict[str, Any]:
        if self.held_object != object_id:
            return {"success": False, "failure_code": "NOT_HELD", "message": object_id}
        body = self._body(object_id)
        target, yaw = self._place_target(object_id, region_id)
        self.status = f"Placing {self.object_aliases[object_id]} on {self.region_aliases[region_id]}"
        pose, path = self._select_stance(body, target)
        self._move_to_stance(pose, path)
        picker = self._picker(body)
        resume_held_object_from_simulator(picker, object_id, body)
        # Cancel the twist the grasp introduced so the payload, not the
        # gripper, ends up at the support's yaw.  Only the gripper command is
        # offset: `yaw` stays the support's yaw, which is what the placement
        # record asserts and what _verify_place holds the result to.
        command_yaw = yaw - self._grasp_yaw_offsets.get(object_id, 0.0)
        rotation = np.array(
            (
                (math.cos(command_yaw), -math.sin(command_yaw), 0.0),
                (math.sin(command_yaw), math.cos(command_yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        ) @ picker.profile.top_down_rotation
        picker.request_place_world(target, rotation)
        failure = self._run_picker(picker, "idle")
        if failure:
            return {"success": False, "failure_code": "PLACE", "message": str(failure)}
        if not self._settle_payload(body):
            return {"success": False, "failure_code": "PLACE", "message": "Payload did not settle"}
        if not self._verify_place(object_id, region_id, body, target, yaw, picker):
            return {"success": False, "failure_code": "PLACE", "message": "Physical ON relation was not verified"}
        self.held_object = None
        self.locations[object_id] = region_id
        self._placement_records[object_id] = self.last_placement_record
        self.revision += 1
        self.invalidate_images()
        return {"success": True, "effects": [f"placed({object_id},{region_id})"]}

    def accept_effects(self, _effects: tuple[str, ...]) -> None:
        """Physical skills already update the private observed state."""
