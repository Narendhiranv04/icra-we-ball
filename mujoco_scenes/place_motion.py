"""Buffered region sampling and smooth release trajectories for place actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_scenes.pick_motion import (
    OPEN_WIDTH,
    PICK_SPECS,
    ArmWaypoint,
    PickExecutor,
    VerticalIK,
)


EDGE_BUFFER = 0.025
OBJECT_CLEARANCE = 0.018
GRIPPER_OBSTACLE_CLEARANCE = 0.085
PLACE_HOVER_CLEARANCE = 0.08
SIDE_JAR_HOVER_CLEARANCE = 0.025
PLACE_DROP_CLEARANCE = 0.025
SIDE_LOADED_DROP_CLEARANCE = 0.050
LOADED_TRACKING_COMPENSATION = 2.50
PLACE_SETTLE_TICKS = 400
JAR_SETTLE_ASSIST_TICKS = 340
JAR_SETTLE_FADE_TICKS = 80
PLACE_TRACKING_GRACE_TICKS = 1000
SPOON_RELEASE_LEAN = math.radians(6.0)


@dataclass(frozen=True)
class PlacementRegion:
    """A world-frame rectangular sampling region on one support surface."""

    name: str
    bounds: tuple[float, float, float, float]
    surface_geom: str
    surface_height: float = 0.58
    edge_buffer: float = EDGE_BUFFER


SERVING_REGION = PlacementRegion(
    "serving_table",
    (-0.25, 0.25, -0.71, -0.41),
    "serving_surface",
)

# These are manipulation-facing strips within the full counter footprint.
# They deliberately stop before the box on the right and before the far edge.
TABLE_SUBREGIONS = {
    "home": PlacementRegion(
        "table_sub_1", (-0.36, 0.36, -0.37, -0.14), "counter_surface"
    ),
    "cupboard1": PlacementRegion(
        "table_sub_2", (-0.68, -0.36, -0.34, 0.22), "counter_surface"
    ),
    "right_side": PlacementRegion(
        "table_sub_3", (0.36, 0.68, -0.37, -0.12), "counter_surface"
    ),
}

DRAWER_REGIONS = {
    "drawer_D1": PlacementRegion(
        "drawer_D1", (-0.606, -0.274, -0.682, -0.424), "D1_tray_base", 0.408, 0.008
    ),
    "drawer_D2": PlacementRegion(
        "drawer_D2", (0.274, 0.606, -0.682, -0.424), "D2_tray_base", 0.408, 0.008
    ),
}

HANGING_OBJECT_SETTLED_X = {
    "spoon": (-0.105, 0.105),
    "fork": (-0.100, 0.100),
    "knife": (-0.108, 0.108),
    "stirrer": (-0.070, 0.070),
    # The simple spatula body origin is at its handle end. Its +X working end
    # hangs downward and the deliberate release lean lays it toward world -X.
    "spatula": (-0.190, 0.020),
    "tongs": (-0.100, 0.100),
    "gso_spatula_distractor": (-0.120, 0.120),
}


def resolve_place_region(name: str, physical_location: str) -> PlacementRegion:
    """Resolve public aliases to the current support rectangle."""
    if name == "serving_table":
        if physical_location != "home":
            raise RuntimeError("Serving-table placement requires Move (home) first")
        return SERVING_REGION
    if name in DRAWER_REGIONS:
        if physical_location != "home":
            raise RuntimeError("Drawer placement requires Move (home) first")
        return DRAWER_REGIONS[name]
    if name != "table":
        raise ValueError(
            "Place region must be 'serving_table', 'table', 'drawer_D1', or "
            "'drawer_D2'"
        )
    try:
        return TABLE_SUBREGIONS[physical_location]
    except KeyError as error:
        raise RuntimeError(
            f"No table placement subregion for base pose '{physical_location}'"
        ) from error


def buffered_center_bounds(
    region: PlacementRegion,
    object_offsets: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Shrink a region so every side of an object retains the edge buffer."""
    min_x, max_x, min_y, max_y = region.bounds
    object_min_x, object_max_x, object_min_y, object_max_y = object_offsets
    bounds = (
        min_x + region.edge_buffer - object_min_x,
        max_x - region.edge_buffer - object_max_x,
        min_y + region.edge_buffer - object_min_y,
        max_y - region.edge_buffer - object_max_y,
    )
    if bounds[0] > bounds[1] or bounds[2] > bounds[3]:
        raise RuntimeError(f"Held object does not fit safely inside {region.name}")
    return bounds


class PlaceExecutor:
    """Place the object held by a PickExecutor into one sampled region."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        picker: PickExecutor,
        *,
        seed: int = 29,
    ):
        self.model = model
        self.data = data
        self.picker = picker
        self.rng = np.random.default_rng(seed)
        self.mode = "idle"
        self.status = "Place idle"
        self.failure: str | None = None
        self.has_run = False
        self.region: PlacementRegion | None = None
        self.public_region: str | None = None
        self.sampled_point: np.ndarray | None = None
        self.target_object: str | None = None
        self.release_ticks = 0
        self.retreat_assist_ticks = 0
        self.tracking_grace_ticks = 0
        self.retreat_waypoints: list[ArmWaypoint] = []
        self._candidate_data = mujoco.MjData(model)

    @property
    def busy(self) -> bool:
        return self.mode in {"approach", "releasing", "retreating"}

    def _body_world_bounds(self, body_id: int) -> np.ndarray:
        """Return collision-geometry AABB bounds as min/max XYZ rows."""
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != body_id:
                continue
            if (
                self.model.geom_contype[geom_id] == 0
                and self.model.geom_conaffinity[geom_id] == 0
            ):
                continue
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            local_center = self.model.geom_aabb[geom_id, :3]
            local_half = self.model.geom_aabb[geom_id, 3:]
            center = self.data.geom_xpos[geom_id] + rotation @ local_center
            half = np.abs(rotation) @ local_half
            lower = np.minimum(lower, center - half)
            upper = np.maximum(upper, center + half)
        if not np.all(np.isfinite(lower)):
            raise RuntimeError("Held object has no collision geometry")
        return np.vstack((lower, upper))

    def _nearby_obstacle_bounds(
        self, surface_height: float
    ) -> list[np.ndarray]:
        obstacles = []
        for body_id in range(1, self.model.nbody):
            if body_id == self.picker.target_body_id:
                continue
            if int(self.model.body_jntnum[body_id]) != 1:
                continue
            joint_id = int(self.model.body_jntadr[body_id])
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            try:
                bounds = self._body_world_bounds(body_id)
            except RuntimeError:
                continue
            if bounds[1, 2] < surface_height - 0.02:
                continue
            if bounds[0, 2] > surface_height + 0.35:
                continue
            obstacles.append(bounds)
        return obstacles

    @staticmethod
    def _overlaps_obstacle(
        candidate_xy: np.ndarray,
        object_offsets: np.ndarray,
        obstacles: list[np.ndarray],
    ) -> bool:
        candidate = np.array(
            (
                candidate_xy[0] + object_offsets[0, 0],
                candidate_xy[0] + object_offsets[1, 0],
                candidate_xy[1] + object_offsets[0, 1],
                candidate_xy[1] + object_offsets[1, 1],
            )
        )
        for obstacle in obstacles:
            separated = (
                candidate[1] + OBJECT_CLEARANCE < obstacle[0, 0]
                or candidate[0] - OBJECT_CLEARANCE > obstacle[1, 0]
                or candidate[3] + OBJECT_CLEARANCE < obstacle[0, 1]
                or candidate[2] - OBJECT_CLEARANCE > obstacle[1, 1]
            )
            if not separated:
                return True
        return False

    @staticmethod
    def _gripper_overlaps_obstacle(
        grip_xy: np.ndarray,
        obstacles: list[np.ndarray],
        clearance: float = GRIPPER_OBSTACLE_CLEARANCE,
    ) -> bool:
        """Reserve room for the hand/wrist, not only the held object."""
        for obstacle in obstacles:
            if (
                obstacle[0, 0] - clearance
                <= grip_xy[0]
                <= obstacle[1, 0] + clearance
                and obstacle[0, 1] - clearance
                <= grip_xy[1]
                <= obstacle[1, 1] + clearance
            ):
                return True
        return False

    def _candidate_has_contact(
        self,
        body_position: np.ndarray,
        body_quaternion: np.ndarray,
        surface_geom: str,
    ) -> bool:
        candidate = self._candidate_data
        candidate.qpos[:] = self.data.qpos
        candidate.qvel[:] = 0.0
        candidate.eq_active[:] = self.data.eq_active
        if self.picker.grasp_equality_id >= 0:
            candidate.eq_active[self.picker.grasp_equality_id] = 0
        if self.picker.spoon_pivot_equality_id >= 0:
            candidate.eq_active[self.picker.spoon_pivot_equality_id] = 0
        joint_id = int(self.model.body_jntadr[self.picker.target_body_id])
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        candidate.qpos[qpos_address : qpos_address + 3] = body_position
        candidate.qpos[qpos_address + 3 : qpos_address + 7] = body_quaternion
        mujoco.mj_forward(self.model, candidate)
        for contact in candidate.contact:
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if self.picker.target_body_id not in {body1, body2}:
                continue
            other_geom = contact.geom2 if body1 == self.picker.target_body_id else contact.geom1
            other_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or ""
            if other_name != surface_geom:
                return True
        return False

    def _planned_robot_has_contact(
        self,
        arm_joints: np.ndarray,
        body_position: np.ndarray,
        body_quaternion: np.ndarray,
    ) -> bool:
        """Check the hand/arm at the planned release, not just the object."""
        candidate = self._candidate_data
        candidate.qpos[:] = self.data.qpos
        candidate.qvel[:] = 0.0
        candidate.eq_active[:] = self.data.eq_active
        if self.picker.grasp_equality_id >= 0:
            candidate.eq_active[self.picker.grasp_equality_id] = 0
        if self.picker.spoon_pivot_equality_id >= 0:
            candidate.eq_active[self.picker.spoon_pivot_equality_id] = 0
        candidate.qpos[self.picker.arm_qpos] = arm_joints
        joint_id = int(self.model.body_jntadr[self.picker.target_body_id])
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        candidate.qpos[qpos_address : qpos_address + 3] = body_position
        candidate.qpos[qpos_address + 3 : qpos_address + 7] = body_quaternion
        mujoco.mj_forward(self.model, candidate)
        for contact in candidate.contact:
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body1
            ) or ""
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body2
            ) or ""
            robot1 = name1.startswith("robot0:")
            robot2 = name2.startswith("robot0:")
            if robot1 == robot2:
                continue
            other_body = body2 if robot1 else body1
            # Finger/object contact is the intended live grasp during descent.
            if other_body == self.picker.target_body_id:
                continue
            other_geom = contact.geom2 if robot1 else contact.geom1
            other_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or ""
            # Fetch's base proxy normally rests on the floor.
            robot_geom = contact.geom1 if robot1 else contact.geom2
            robot_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom
            ) or ""
            if other_name == "floor" and robot_name == "robot0:base_collision_proxy":
                continue
            return True
        return False

    def _candidate_points(
        self,
        bounds: tuple[float, float, float, float],
        count: int = 96,
        preferred: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        min_x, max_x, min_y, max_y = bounds
        centre = np.array(((min_x + max_x) / 2, (min_y + max_y) / 2))
        random_points = self.rng.uniform(
            (min_x, min_y), (max_x, max_y), size=(count - 1, 2)
        )
        points = [centre]
        if preferred is not None:
            points.insert(
                0,
                np.clip(
                    np.asarray(preferred, dtype=float),
                    (min_x, min_y),
                    (max_x, max_y),
                ),
            )
        points.extend(point for point in random_points)
        return points

    def _plan_trajectory(
        self,
        release_grip: np.ndarray,
        target_rotation: np.ndarray,
        carry_joints: np.ndarray,
        carry_grip: np.ndarray,
        region: PlacementRegion,
        ik_seed_joints: np.ndarray,
    ) -> tuple[list[ArmWaypoint], list[ArmWaypoint]]:
        ik = VerticalIK(self.model, self.data)
        hover_clearance = (
            SIDE_JAR_HOVER_CLEARANCE
            if region.name in {"table_sub_2", "table_sub_3"}
            and self.target_object in {"coffee_jar", "sugar_jar"}
            else PLACE_HOVER_CLEARANCE
        )
        hover = release_grip + np.array((0.0, 0.0, hover_clearance))
        approach_axis = target_rotation[:, 0]
        horizontal_gripper = abs(float(approach_axis @ (0.0, 0.0, 1.0))) < 0.5
        if horizontal_gripper:
            # The centre-table jar needs a side corridor to remain on its
            # load-bearing IK branch. Side/serving placements can go directly
            # to hover, eliminating the former low-arrival-then-rise motion.
            travel_height = max(float(carry_grip[2]), float(release_grip[2]))
            goals: list[np.ndarray] = []
            if region.name == "table_sub_1":
                side = 0.34
                cross_y = -0.36
                transit_height = max(float(hover[2]), 0.88)
                goals.extend(
                    (
                        np.array((side, carry_grip[1], travel_height)),
                        np.array((side, cross_y, travel_height)),
                        np.array((side, cross_y, transit_height)),
                        np.array((hover[0], hover[1], transit_height)),
                        hover,
                    )
                )
            else:
                goals.append(hover)
            approach = []
            current_joints = ik_seed_joints.copy()
            cursor = carry_grip.copy()
            for goal in goals:
                points = self.picker._cartesian_points(cursor, goal, 0.025)
                segment, current_joints = self.picker._solve_path(
                    ik,
                    points,
                    current_joints,
                    "Moving directly to place hover",
                    target_rotation,
                    angle_tolerance=math.radians(4.5),
                )
                approach.extend(segment)
                cursor = goal
            hover_joints = current_joints
        else:
            points = self.picker._cartesian_points(carry_grip, hover, 0.025)
            approach, hover_joints = self.picker._solve_path(
                ik,
                points,
                ik_seed_joints,
                "Moving directly from carry to place hover",
                target_rotation,
            )
        descent_points = self.picker._cartesian_points(hover, release_grip, 0.010)
        descent, _ = self.picker._solve_path(
            ik,
            descent_points,
            hover_joints,
            "Lowering object for release",
            target_rotation,
        )
        place_waypoints = [
            ArmWaypoint(carry_joints.copy(), "Leaving carry pose"),
            *approach,
            *descent,
        ]
        retreat = [
            ArmWaypoint(w.joints.copy(), "Retreating vertically after release")
            for w in reversed(descent[:-1])
        ]
        retreat.append(
            ArmWaypoint(hover_joints.copy(), "Release clearance restored")
        )
        retreat.extend(
            ArmWaypoint(w.joints.copy(), "Returning to empty carry pose")
            for w in reversed(approach[:-1])
        )
        retreat.append(
            ArmWaypoint(carry_joints.copy(), "Empty gripper in carry pose")
        )
        return place_waypoints, retreat

    def request_place(self, region_name: str, physical_location: str) -> None:
        if self.busy:
            raise RuntimeError("A place action is already running")
        if self.picker.busy:
            raise RuntimeError("Wait for the manipulation action to finish")
        if self.picker.held_object is None or self.picker.mode != "holding":
            raise RuntimeError("Place requires an object in the gripper")

        mujoco.mj_forward(self.model, self.data)
        region = resolve_place_region(region_name, physical_location)
        if region_name in DRAWER_REGIONS:
            drawer_name = region_name.removeprefix("drawer_")
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{drawer_name}_slide_joint",
            )
            if joint_id < 0:
                raise RuntimeError(f"Missing slide joint for {drawer_name}")
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            if (
                self.data.qpos[qpos_address]
                < self.model.jnt_range[joint_id, 1] - 0.015
            ):
                raise RuntimeError(f"{drawer_name} must be fully open before placement")
        self.target_object = self.picker.held_object
        body_id = self.picker.target_body_id
        body_position = self.data.xpos[body_id].copy()
        body_quaternion = self.data.xquat[body_id].copy()
        body_bounds = self._body_world_bounds(body_id)
        offsets = body_bounds - body_position
        spec = PICK_SPECS[self.target_object]
        passive_hang = self.target_object == "spoon" or spec.passive_hang
        # A vertically hanging utensil topples along world X after release.
        # Reserve its full settled length in that direction. Counter/serving
        # placement keeps the older conservative all-direction allowance;
        # the narrow drawer uses the intentional X fall direction so a long
        # utensil still fits safely between its front and back walls.
        if passive_hang:
            settled_min_x, settled_max_x = HANGING_OBJECT_SETTLED_X[
                self.target_object
            ]
            offsets[0, 0] = min(offsets[0, 0], settled_min_x)
            offsets[1, 0] = max(offsets[1, 0], settled_max_x)
            if region_name not in DRAWER_REGIONS:
                half_length = max(abs(settled_min_x), abs(settled_max_x))
                offsets[0, 1] = min(offsets[0, 1], -half_length)
                offsets[1, 1] = max(offsets[1, 1], half_length)
        centre_bounds = buffered_center_bounds(
            region,
            (offsets[0, 0], offsets[1, 0], offsets[0, 1], offsets[1, 1]),
        )
        obstacles = self._nearby_obstacle_bounds(region.surface_height)
        if self.picker.carry_goal_joints is None:
            raise RuntimeError("Held object has no shared pick/place carry pose")
        carry_joints = self.picker.carry_goal_joints.copy()
        ik_seed_joints = self.picker._current_arm()
        carry_grip = self.data.site_xpos[self.picker.grip_site_id].copy()
        target_rotation = self.data.site_xmat[
            self.picker.grip_site_id
        ].reshape(3, 3).copy()
        grip_offset = carry_grip - body_position
        if passive_hang:
            # A perfectly vertical spoon can topple toward the serving
            # table's short Y edge. Lean the whole live grasp slightly around
            # world Y so gravity still performs the placement but the bowl
            # falls along the much wider X dimension.
            cosine = math.cos(SPOON_RELEASE_LEAN)
            sine = math.sin(SPOON_RELEASE_LEAN)
            placement_transform = np.array(
                ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
            )
            target_rotation = placement_transform @ target_rotation
            grip_offset = placement_transform @ grip_offset
            body_rotation = placement_transform @ self.data.xmat[body_id].reshape(
                3, 3
            )
            mujoco.mju_mat2Quat(body_quaternion, body_rotation.ravel())
        lowest_offset = float(offsets[0, 2])

        last_error: Exception | None = None
        chosen_point = None
        chosen_waypoints = None
        drop_clearance = (
            SIDE_LOADED_DROP_CLEARANCE
            if region.name in {"table_sub_2", "table_sub_3"}
            and self.target_object in {"coffee_jar", "sugar_jar"}
            else PLACE_DROP_CLEARANCE
        )
        preferred_point = (
            self.picker.pick_source_position[:2]
            if region_name in DRAWER_REGIONS
            and self.picker.pick_source_position is not None
            else None
        )
        for candidate_index, point in enumerate(
            self._candidate_points(centre_bounds, preferred=preferred_point)
        ):
            restores_vacated_drawer_slot = (
                preferred_point is not None
                and candidate_index == 0
            )
            # The exact source pose is a previously stable, now-vacated slot.
            # Preserve it as the first deterministic drawer sample even when
            # conservative AABBs of adjacent long tools overlap slightly.
            if (
                not restores_vacated_drawer_slot
                and self._overlaps_obstacle(point, offsets, obstacles)
            ):
                if last_error is None:
                    last_error = RuntimeError("sample overlaps another free object")
                continue
            desired_body = np.array(
                (
                    point[0],
                    point[1],
                    region.surface_height + drop_clearance - lowest_offset,
                )
            )
            if self._candidate_has_contact(
                desired_body, body_quaternion, region.surface_geom
            ):
                if last_error is None:
                    last_error = RuntimeError("released object would contact a wall")
                continue
            release_grip = desired_body + grip_offset
            gripper_clearance = (
                0.030 if region_name in DRAWER_REGIONS else GRIPPER_OBSTACLE_CLEARANCE
            )
            if not restores_vacated_drawer_slot and self._gripper_overlaps_obstacle(
                release_grip[:2], obstacles, gripper_clearance
            ):
                if last_error is None:
                    last_error = RuntimeError(
                        "gripper lacks clearance from another object"
                    )
                continue
            try:
                chosen_waypoints = self._plan_trajectory(
                    release_grip,
                    target_rotation,
                    carry_joints,
                    carry_grip,
                    region,
                    ik_seed_joints,
                )
            except RuntimeError as error:
                last_error = error
                continue
            if region.name in {"table_sub_1", *DRAWER_REGIONS} and self._planned_robot_has_contact(
                chosen_waypoints[0][-1].joints,
                desired_body,
                body_quaternion,
            ):
                if last_error is None:
                    last_error = RuntimeError(
                        "planned release pose contacts the furniture"
                    )
                continue
            chosen_point = point
            break

        if chosen_point is None or chosen_waypoints is None:
            detail = f": {last_error}" if last_error is not None else ""
            raise RuntimeError(f"No safe reachable sample in {region.name}{detail}")

        self.region = region
        self.public_region = region_name
        self.sampled_point = chosen_point.copy()
        place_waypoints, self.retreat_waypoints = chosen_waypoints
        self.picker._start_trajectory(place_waypoints)
        self.mode = "approach"
        self.failure = None
        self.has_run = True
        self.release_ticks = 0
        self.tracking_grace_ticks = 0
        self.status = (
            f"Place {self.target_object}: sampled {region.name} at "
            f"({chosen_point[0]:.3f}, {chosen_point[1]:.3f})"
        )

    def _release(self) -> None:
        if self.picker.grasp_equality_id >= 0:
            self.data.eq_active[self.picker.grasp_equality_id] = 0
        if self.picker.spoon_pivot_equality_id >= 0:
            self.data.eq_active[self.picker.spoon_pivot_equality_id] = 0
        self.data.xfrc_applied[self.picker.target_body_id] = 0.0
        self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH

    def _assist_jar_settle(self, assist_tick: int | None = None) -> None:
        """Damp jar tilt during release without constraining its free body."""
        if self.target_object not in {"coffee_jar", "sugar_jar"}:
            return
        body_id = self.picker.target_body_id
        self.data.xfrc_applied[body_id] = 0.0
        tick = self.release_ticks if assist_tick is None else assist_tick
        if tick >= JAR_SETTLE_ASSIST_TICKS:
            return
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        up = np.array((0.0, 0.0, 1.0))
        body_up = self.data.xmat[body_id].reshape(3, 3)[:, 2]
        angular_velocity = velocity[:3]
        tilt_velocity = angular_velocity - up * float(angular_velocity @ up)
        stiffness, damping, limit = (
            (5.0, 0.50, 0.80)
            if self.target_object == "coffee_jar"
            else (4.0, 0.40, 0.60)
        )
        fade = min(
            1.0,
            (JAR_SETTLE_ASSIST_TICKS - tick)
            / JAR_SETTLE_FADE_TICKS,
        )
        torque = fade * (
            stiffness * np.cross(body_up, up) - damping * tilt_velocity
        )
        self.data.xfrc_applied[body_id, 3:] = self.picker._limited(torque, limit)

    def _live_release_is_safe(self) -> bool:
        """Accept load-induced XY compliance only while safely over the region."""
        assert self.region is not None
        mujoco.mj_forward(self.model, self.data)
        body_id = self.picker.target_body_id
        body_position = self.data.xpos[body_id].copy()
        bounds = self._body_world_bounds(body_id)
        min_x, max_x, min_y, max_y = self.region.bounds
        margin = self.region.edge_buffer
        horizontally_inside = (
            bounds[0, 0] >= min_x + margin
            and bounds[1, 0] <= max_x - margin
            and bounds[0, 1] >= min_y + margin
            and bounds[1, 1] <= max_y - margin
        )
        bottom_clearance = float(bounds[0, 2] - self.region.surface_height)
        # Mesh AABBs can extend about a centimetre below their actual contact
        # hull.  A small negative conservative clearance therefore still
        # represents an object resting at the support, not penetrating it.
        if not horizontally_inside or not -0.012 <= bottom_clearance <= 0.060:
            return False
        offsets = bounds - body_position
        if self._overlaps_obstacle(
            body_position[:2], offsets, self._nearby_obstacle_bounds(
                self.region.surface_height
            )
        ):
            return False
        # This becomes the effective physics-aware sample reported to the UI.
        self.sampled_point = body_position[:2].copy()
        return True

    def update(self) -> None:
        if not self.busy:
            return
        if self.mode == "approach":
            finished, waypoint = self.picker._advance_trajectory(
                final_tolerance=0.012
            )
            # Position actuators otherwise sag substantially under a
            # horizontal jar grasp. Feed a bounded fraction of the live
            # tracking error into the position command, preserving the same
            # smooth reference curve while reaching its actual hover/descent.
            command = self.data.ctrl[self.picker.arm_actuators].copy()
            live_joints = self.data.qpos[self.picker.arm_qpos]
            compensated = command + LOADED_TRACKING_COMPENSATION * (
                command - live_joints
            )
            joint_ranges = self.model.jnt_range[self.picker.arm_joint_ids]
            joint_limited = self.model.jnt_limited[
                self.picker.arm_joint_ids
            ].astype(bool)
            self.data.ctrl[self.picker.arm_actuators] = np.clip(
                compensated,
                np.where(joint_limited, joint_ranges[:, 0], -np.inf),
                np.where(joint_limited, joint_ranges[:, 1], np.inf),
            )
            self.status = f"Place {self.target_object}: {waypoint.label}"
            trajectory_ended = (
                self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            )
            if trajectory_ended and not finished:
                # A horizontal jar can pull a side-facing arm several
                # centimetres away from its commanded XY point.  Release at
                # the live equilibrium only if the complete object footprint
                # still satisfies the same edge and obstacle margins.
                if self._live_release_is_safe():
                    finished = True
                    self.status = (
                        f"Place {self.target_object}: using safe loaded "
                        "equilibrium"
                    )
                else:
                    self.tracking_grace_ticks += 1
                    if self.tracking_grace_ticks >= PLACE_TRACKING_GRACE_TICKS:
                        self.mode = "failed"
                        self.failure = (
                            "Arm could not reach a safe release pose; object "
                            "remains held"
                        )
                        self.status = f"Place failed: {self.failure}"
                        return
            if finished:
                self._release()
                self.mode = "releasing"
                self.release_ticks = 0
                self.status = (
                    f"Place {self.target_object}: gripper open, physics settling"
                )
            return

        if self.mode == "releasing":
            self.release_ticks += 1
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            self._assist_jar_settle()
            if self.release_ticks >= PLACE_SETTLE_TICKS:
                self.data.xfrc_applied[self.picker.target_body_id] = 0.0
                self.picker._start_trajectory(self.retreat_waypoints)
                self.mode = "retreating"
                self.retreat_assist_ticks = 0
                self.status = f"Place {self.target_object}: retreating after release"
            return

        if self.mode == "retreating":
            # Keep only the rotational spring active until the open hand has
            # fully cleared the jar; otherwise a fingertip can tip the jar
            # late in the long side-pose return path.
            self._assist_jar_settle(0)
            self.retreat_assist_ticks += 1
            # The return target was captured while the arm carried the
            # object's load.  Once released, the same position-actuator
            # command has a slightly different gravity equilibrium; allow
            # that unloaded carry equilibrium instead of waiting forever on
            # an unreachable loaded joint vector.
            finished, waypoint = self.picker._advance_trajectory(
                final_tolerance=0.080
            )
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            self.status = f"Place {self.target_object}: {waypoint.label}"
            if finished:
                self.data.xfrc_applied[self.picker.target_body_id] = 0.0
                placed_object = self.target_object
                placed_region = self.public_region
                self.picker.held_object = None
                self.picker.target_object = None
                self.picker.pick_source_position = None
                self.picker.target_body_id = -1
                self.picker.target_free_dof = -1
                self.picker.grasp_equality_id = -1
                self.picker.spoon_pivot_equality_id = -1
                self.picker.mode = "idle"
                self.picker.status = "Pick idle: gripper empty"
                self.mode = "complete"
                self.status = (
                    f"Place complete: {placed_object} in {placed_region}; "
                    "empty gripper returned to carry pose"
                )

    def progress(self) -> float:
        if self.mode == "approach":
            ratio = self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
            return 0.55 * ratio
        if self.mode == "releasing":
            return 0.55 + 0.10 * self.release_ticks / PLACE_SETTLE_TICKS
        if self.mode == "retreating":
            ratio = self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
            return 0.65 + 0.35 * ratio
        if self.mode == "complete":
            return 1.0
        return 0.0
