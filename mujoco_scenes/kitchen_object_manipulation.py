"""Physical Google PICK/PLACE primitives for perception-resolved kitchen objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import replace
from enum import Enum
import time
from typing import Any

import mujoco
import numpy as np

from .generic_manipulation import (
    CalibratedPickPlaceExecutor,
    GraspPoseCandidate,
    GOOGLE_PICK_SPECS,
    SimplePickSpec,
)
from .kitchen_execution_entities import ObjectSourceContext
from .kitchen_execution_policy import KitchenWorkspace
from .living_room_mobile_execution import (
    oriented_rectangle_corners,
    oriented_rectangles_clearance,
    rectangle_inside_observed_support,
)
from .robot_profiles import manipulation_profile


PHASE_B_MOUNT_ALLOWANCES = {
    # Menagerie visual shells overlap at the concentric shoulder mount.  This
    # execution-local allowance is bounded to that mechanical pair; all other
    # self-collisions remain strict.
    frozenset(("google:base_link", "google:link_shoulder")): -0.060,
}


class ObjectExecutionFailureCode(str, Enum):
    ENTITY_RESOLUTION_FAILED = "ENTITY_RESOLUTION_FAILED"
    SOURCE_CONTEXT_UNKNOWN = "SOURCE_CONTEXT_UNKNOWN"
    SOURCE_RELATION_INVALID = "SOURCE_RELATION_INVALID"
    WORKSPACE_PRECONDITION_UNSATISFIED = "WORKSPACE_PRECONDITION_UNSATISFIED"
    CONTAINER_ACCESS_REQUIRED = "CONTAINER_ACCESS_REQUIRED"
    PICK_IK_FAILED = "PICK_IK_FAILED"
    PICK_PATH_COLLISION = "PICK_PATH_COLLISION"
    PICK_CONTACT_FAILED = "PICK_CONTACT_FAILED"
    GRASP_FAILED = "GRASP_FAILED"
    DIRECT_GRASP_INFEASIBLE = "DIRECT_GRASP_INFEASIBLE"
    PRESENTATION_IK_FAILED = "PRESENTATION_IK_FAILED"
    PRESENTATION_PATH_COLLISION = "PRESENTATION_PATH_COLLISION"
    PRESENTATION_CONTACT_FAILED = "PRESENTATION_CONTACT_FAILED"
    PRESENTATION_FAILED = "PRESENTATION_FAILED"
    NEIGHBOUR_OBJECT_DISTURBED = "NEIGHBOUR_OBJECT_DISTURBED"
    PRESENTATION_POSTCONDITION_FAILED = "PRESENTATION_POSTCONDITION_FAILED"
    REGRASP_FAILED = "REGRASP_FAILED"
    SOURCE_CLEARANCE_FAILED = "SOURCE_CLEARANCE_FAILED"
    HELD_STATE_INVALID = "HELD_STATE_INVALID"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    PLACEMENT_FAILED = "PLACEMENT_FAILED"
    UNSUPPORTED_PHASE_C_OPERATOR = "UNSUPPORTED_PHASE_C_OPERATOR"


@dataclass(frozen=True)
class HeldObjectState:
    generic_object_id: str
    backend_body: str
    weld_id: int
    weld_active: bool
    gripper_body: str
    relative_position_m: tuple[float, float, float]
    relative_orientation_wxyz: tuple[float, float, float, float]
    finger_joint_positions: tuple[float, float]
    exclusive_payload_weld: bool
    floor_contact: bool
    validation_status: str
    rejection_reasons: tuple[str, ...]


@dataclass
class PhysicalPickResult:
    generic_object_id: str
    backend_body: str
    source_context: dict[str, Any]
    required_workspace: str
    grasp_family: str
    success: bool
    status: str
    failure_code: str | None
    message: str
    physics_steps: int
    duration_s: float
    bilateral_contact: bool
    contact_sides: tuple[int, ...]
    contact_geoms: tuple[str, ...]
    attachment_translation_snap_m: float | None
    attachment_angle_snap_rad: float | None
    navigation_safe_carry_reached: bool
    held_state: dict[str, Any] | None
    direct_object_qpos_write: bool = False
    selected_grasp_candidate_id: str | None = None
    target_contact_geoms: tuple[str, ...] = ()
    presentation: dict[str, Any] | None = None
    direct_grasp_analysis: dict[str, Any] | None = None


@dataclass(frozen=True)
class PlacementTarget:
    generic_object_id: str
    symbolic_destination: str
    destination_kind: str
    target_position_world_m: tuple[float, float, float]
    target_yaw_world_rad: float
    support_backend: str | None
    target_object_id: str | None
    required_workspace: KitchenWorkspace
    edge_margin_m: float
    relation_to_verify: str
    provenance: str


@dataclass(frozen=True)
class ServingPlacementState:
    object_id: str
    backend_body: str
    centre_xy_m: tuple[float, float]
    footprint_xy_m: tuple[float, float]
    yaw_world_rad: float
    support_backend: str


@dataclass
class PhysicalPlaceResult:
    generic_object_id: str
    backend_body: str
    symbolic_destination: str
    placement_target: dict[str, Any]
    success: bool
    status: str
    failure_code: str | None
    message: str
    physics_steps: int
    duration_s: float
    grasp_released: bool
    support_contact: bool
    floor_contact: bool
    stable: bool
    physical_relation_verified: bool
    footprint_inside_support: bool = False
    edge_margin_m: float | None = None
    object_relative_distance_m: float | None = None
    final_body_position_world_m: tuple[float, float, float] | None = None
    object_contact_pairs: tuple[tuple[str, str], ...] = ()
    direct_object_qpos_write: bool = False
    linear_speed_m_s: float | None = None
    angular_speed_rad_s: float | None = None
    footprint_corners_world_m: tuple[tuple[float, float], ...] = ()
    pairwise_payload_checks: tuple[dict[str, Any], ...] = ()
    invalid_object_contacts: tuple[tuple[str, str], ...] = ()
    source_return_xy_error_m: float | None = None
    source_region_membership: bool | None = None
    upright_alignment: float | None = None
    intended_target_uniquely_closest: bool | None = None


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> set[int]:
    first = int(model.body_geomadr[body_id])
    return set(range(first, first + int(model.body_geomnum[body_id])))


def _body_yaw(data: mujoco.MjData, body_id: int) -> float:
    rotation = data.xmat[body_id].reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _axis_angle_rotation(rotation_vector: np.ndarray) -> np.ndarray:
    """Return a deterministic small-angle candidate rotation."""
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotation_vector / angle
    skew = np.array(
        ((0.0, -axis[2], axis[1]),
         (axis[2], 0.0, -axis[0]),
         (-axis[1], axis[0], 0.0))
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


class UtensilGraspCandidateGenerator:
    """Generate bounded handle-frame candidates without object-name tuning."""

    HANDLE_FRACTIONS = (0.55, 0.65, 0.75, 0.85)
    HEIGHT_OFFSETS_M = (
        0.0, 0.003, -0.003, 0.006, -0.006,
        # Storage trays need the finger tips kept above the tray plane.  The
        # pads still close onto the raised handle collision capsule, and the
        # bilateral handle-contact gate rejects candidates that are too high.
        0.009, 0.012, 0.015,
    )

    @classmethod
    def generate(
        cls, scene, body_id: int, source_kind: str = "TABLE"
    ) -> tuple[GraspPoseCandidate, ...]:
        handle_id = next(
            (
                geom_id
                for geom_id in _body_geom_ids(scene.model, body_id)
                if "handle_collision" in (
                    mujoco.mj_id2name(
                        scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    ) or ""
                )
            ),
            None,
        )
        if handle_id is None:
            return ()
        local_rotation = np.empty(9)
        mujoco.mju_quat2Mat(local_rotation, scene.model.geom_quat[handle_id])
        handle_axis_local = local_rotation.reshape(3, 3)[:, 2]
        half_length = float(scene.model.geom_size[handle_id, 1])
        centre = scene.model.geom_pos[handle_id].copy()
        endpoints = (centre - half_length * handle_axis_local,
                     centre + half_length * handle_axis_local)
        head_geoms = [
            geom_id for geom_id in _body_geom_ids(scene.model, body_id)
            if "bowl_collision" in (
                mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or ""
            )
        ]
        head = (
            scene.model.geom_pos[head_geoms[0]]
            if head_geoms else centre - handle_axis_local * half_length
        )
        near, far = sorted(endpoints, key=lambda point: np.linalg.norm(point - head))
        world_axis = scene.data.geom_xmat[handle_id].reshape(3, 3)[:, 2]
        handle_yaw = float(np.arctan2(world_axis[1], world_axis[0]))
        profile = manipulation_profile("google")
        candidates = []
        if source_kind == "DRAWER":
            # A utensil lying low in a drawer cannot always be pinched from
            # above without the finger tips entering the tray.  These two
            # bounded handle-axis approaches retain horizontal jaw closure
            # and are still subject to full collision and bilateral-contact
            # acceptance.
            carry_yaw = handle_yaw + np.deg2rad(330.0)
            carry_tilt = (
                _axis_angle_rotation(np.array((np.deg2rad(15.0), 0.0, 0.0)))
                @ _axis_angle_rotation(np.array((0.0, np.deg2rad(15.0), 0.0)))
            )
            drawer_carry_rotation = (
                _axis_angle_rotation(np.array((0.0, 0.0, carry_yaw)))
                @ carry_tilt
                @ profile.top_down_rotation
            )
            drawer_front_rotation = (
                _axis_angle_rotation(np.array((0.0, 0.0, carry_yaw)))
                @ profile.top_down_rotation
            )
            # Local +Y is the jaw-closing axis.  Rotating by the measured
            # handle yaw keeps that axis perpendicular to the handle.
            level_yaw = handle_yaw
            level_rotation = (
                _axis_angle_rotation(np.array((0.0, 0.0, level_yaw)))
                @ profile.top_down_rotation
            )
            for fraction in (0.55, 0.65, 0.75, 0.85):
                # The UTENSIL spec adds another 20 mm grasp offset.  These
                # local raises therefore produce the 25--35 mm level grip
                # heights established by the contact-presentation sweep.
                for storage_raise in (-0.005, 0.0, 0.005, 0.010):
                    level_position = near + fraction * (far - near)
                    level_position[2] += storage_raise
                    candidates.append(
                        GraspPoseCandidate(
                            candidate_id=(
                                f"drawer_level_perpendicular_"
                                f"{int(storage_raise * 1000)}mm_"
                                f"{int(fraction * 100)}pct"
                            ),
                            grasp_site_local_position_m=tuple(
                                map(float, level_position)
                            ),
                            target_rotation_world=level_rotation,
                            carry_rotation_world=drawer_carry_rotation,
                            approach_clearance_m=0.05,
                            approach_offset_world_m=(0.0, -0.10, 0.0),
                            approach_route_offsets_world_m=(
                                (0.0, -0.35, 0.30),
                                (0.0, -0.10, 0.18),
                            ),
                        )
                    )
            for fraction in (0.55, 0.65, 0.75, 0.85):
                for storage_raise in (0.030, 0.040):
                    front_position = near + fraction * (far - near)
                    front_position[2] += storage_raise
                    candidates.append(
                        GraspPoseCandidate(
                            candidate_id=(
                                f"drawer_front_{int(storage_raise * 1000)}mm_"
                                f"{int(fraction * 100)}pct"
                            ),
                            grasp_site_local_position_m=tuple(map(float, front_position)),
                            target_rotation_world=drawer_front_rotation,
                            approach_clearance_m=0.05,
                            carry_rotation_world=drawer_carry_rotation,
                            approach_offset_world_m=(0.0, -0.10, 0.0),
                            approach_route_offsets_world_m=(
                                (0.0, -0.35, 0.30),
                                (0.0, -0.10, 0.20),
                            ),
                        )
                    )
            side_rotations = [
                ("negative_axis", np.array(
                    ((0.0, 0.0, 1.0),
                     (0.0, 1.0, 0.0),
                     (-1.0, 0.0, 0.0))
                ), (-0.08, 0.0, 0.0)),
                ("positive_axis", np.array(
                    ((0.0, 0.0, -1.0),
                     (0.0, 1.0, 0.0),
                     (1.0, 0.0, 0.0))
                ), (0.08, 0.0, 0.0)),
            ]
            positive_rotation = side_rotations[1][1]
            for axis_name, rotation_vector in (
                ("pitch_plus15", (0.0, np.deg2rad(15.0), 0.0)),
                ("pitch_minus15", (0.0, -np.deg2rad(15.0), 0.0)),
                ("roll_plus15", (np.deg2rad(15.0), 0.0, 0.0)),
                ("roll_minus15", (-np.deg2rad(15.0), 0.0, 0.0)),
            ):
                side_rotations.append(
                    (
                        f"positive_axis_{axis_name}",
                        _axis_angle_rotation(np.asarray(rotation_vector, float))
                        @ positive_rotation,
                        (0.08, 0.0, 0.0),
                    )
                )
            side_rotations.append(
                (
                    "positive_axis_roll_minus15_short",
                    _axis_angle_rotation(
                        np.array((-np.deg2rad(15.0), 0.0, 0.0))
                    ) @ positive_rotation,
                    (0.04, 0.0, 0.0),
                )
            )
            for fraction in (0.65, 0.75):
                local_position = near + fraction * (far - near)
                local_position[2] += 0.010
                for branch, candidate_rotation, approach_offset in side_rotations:
                    candidates.append(
                        GraspPoseCandidate(
                            candidate_id=(
                                f"drawer_{branch}_{int(fraction * 100)}pct"
                            ),
                            grasp_site_local_position_m=tuple(map(float, local_position)),
                            target_rotation_world=candidate_rotation,
                            approach_clearance_m=0.04,
                            carry_rotation_world=drawer_carry_rotation,
                            approach_offset_world_m=approach_offset,
                            approach_route_offsets_world_m=((approach_offset[0], 0.0, 0.20),),
                        )
                    )
        for fraction in cls.HANDLE_FRACTIONS:
            local_position = near + fraction * (far - near)
            # The nominal perpendicular jaw branches are supplemented by
            # bounded +/-30 degree wrist branches.  These are necessary for
            # drawer/cupboard apertures where the same handle-frame grasp is
            # valid but one wrist elbow branch intersects the fixture.
            for branch_degrees in (330.0, 300.0, 0.0, 150.0, 120.0, 180.0):
                yaw = handle_yaw + np.deg2rad(branch_degrees)
                yaw_rotation = _axis_angle_rotation(np.array((0.0, 0.0, yaw)))
                # A shallow diagonal approach keeps the wrist away from the
                # table while leaving the finger pads perpendicular to the
                # thin handle. The actual contact gate remains authoritative.
                tilt = (
                    _axis_angle_rotation(np.array((np.deg2rad(15.0), 0.0, 0.0)))
                    @ _axis_angle_rotation(np.array((0.0, np.deg2rad(15.0), 0.0)))
                )
                rotation = yaw_rotation @ tilt @ profile.top_down_rotation
                for height in cls.HEIGHT_OFFSETS_M:
                    position = local_position.copy()
                    position[2] += height
                    candidates.append(
                        GraspPoseCandidate(
                            candidate_id=(
                                f"handle_{int(fraction * 100)}pct_"
                                f"yaw{int(branch_degrees)}_z{height:+.3f}"
                            ),
                            grasp_site_local_position_m=tuple(map(float, position)),
                            target_rotation_world=rotation,
                            approach_clearance_m=0.060,
                        )
                    )
        return tuple(candidates)


class JarGraspCandidateGenerator:
    """Generate vertical pinch branches for approximately cylindrical jars."""

    YAW_DEGREES = (0, 30, 60, 90, 120, 150, 180)
    LOCAL_OFFSETS_M = (
        (0.0, 0.0, 0.0),
        (0.003, 0.0, 0.0), (-0.003, 0.0, 0.0),
        (0.0, 0.003, 0.0), (0.0, -0.003, 0.0),
        (0.0, 0.0, 0.003), (0.0, 0.0, -0.003),
    )

    @classmethod
    def generate(cls, scene, grasp_site_id: int) -> tuple[GraspPoseCandidate, ...]:
        origin = scene.model.site_pos[grasp_site_id].copy()
        top_down = manipulation_profile("google").top_down_rotation
        # The first branch approaches the cylindrical wall horizontally from
        # the robot-facing side.  Its finger closing axis is horizontal and
        # the required-contact filter admits wall geoms only.
        wall_midpoint = origin.copy()
        wall_midpoint[2] = 0.020
        candidates = [
            GraspPoseCandidate(
                candidate_id="jar_side_wall_from_negative_y",
                grasp_site_local_position_m=tuple(map(float, wall_midpoint)),
                target_rotation_world=np.array(
                    ((0.0, 1.0, 0.0),
                     (0.0, 0.0, 1.0),
                     (1.0, 0.0, 0.0))
                ),
                approach_clearance_m=0.04,
                approach_offset_world_m=(0.0, -0.06, 0.0),
            )
        ]
        for yaw_degrees in cls.YAW_DEGREES:
            yaw = np.deg2rad(float(yaw_degrees))
            rotation = _axis_angle_rotation(np.array((0.0, 0.0, yaw))) @ top_down
            for offset in cls.LOCAL_OFFSETS_M:
                position = origin + np.asarray(offset, float)
                candidates.append(
                    GraspPoseCandidate(
                        candidate_id=(
                            f"jar_yaw{yaw_degrees}_offset_"
                            f"{offset[0]:+.3f}_{offset[1]:+.3f}_{offset[2]:+.3f}"
                        ),
                        grasp_site_local_position_m=tuple(map(float, position)),
                        target_rotation_world=rotation,
                        approach_clearance_m=0.10,
                    )
                )
        return tuple(candidates)


def make_kitchen_pick_specs(
    scene,
    inventory: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, SimplePickSpec]:
    inventory_by_id = {row["generic_object_id"]: row for row in inventory["objects"]}
    specs = {}
    table_top = 0.572
    for row in resolution["accepted"]:
        body = row["physical_backend_body"]
        family = row["grasp_family"]
        observed = inventory_by_id[row["generic_object_id"]]
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
        support_height = max(0.008, float(scene.data.xpos[body_id, 2] - table_top))
        rotation = None
        if family == "UTENSIL":
            angle = np.deg2rad(300.0)
            rotation = np.array(
                ((np.cos(angle), -np.sin(angle), 0.0),
                 (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
            ) @ manipulation_profile("google").top_down_rotation
        if family == "KETTLE":
            angle = np.deg2rad(315.0)
            rotation = np.array(
                ((np.cos(angle), -np.sin(angle), 0.0),
                 (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
            ) @ manipulation_profile("google").top_down_rotation
        utensil_reference = GOOGLE_PICK_SPECS["spoon"] if family == "UTENSIL" else None
        grasp_offset = {
            "UTENSIL": 0.020,
            "KETTLE": 0.008,
            "JAR_SOURCE": 0.011,
        }.get(family, 0.0)
        collision_geom_names = tuple(
            name
            for geom_id in sorted(_body_geom_ids(scene.model, body_id))
            if (name := mujoco.mj_id2name(
                scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ))
            and "collision" in name
        )
        if family == "UTENSIL":
            required_contact_geoms = tuple(
                name for name in collision_geom_names if "handle_collision" in name
            )
        elif family == "KETTLE":
            required_contact_geoms = tuple(
                name for name in collision_geom_names if "handle_collision" in name
            )
        elif family == "JAR_SOURCE":
            required_contact_geoms = tuple(
                name for name in collision_geom_names if "wall_" in name
            )
        else:
            # Vessel/bowl meshes are intentionally collision-active in the
            # benchmark; either their prepared collision shell or their body
            # mesh is a valid contact surface. Target-body gating still
            # prevents contact with a neighbouring object from authorizing a
            # weld.
            required_contact_geoms = ()
        specs[body] = SimplePickSpec(
            label=f"Phase B {family} from {observed['source_context']['source_kind']}",
            grasp_site=f"{body}_grasp",
            support_height=support_height,
            grasp_z_offset=grasp_offset,
            place_supported=True,
            top_down_rotation=rotation,
            home_seed=(utensil_reference.home_seed if utensil_reference else None),
            carry_position=(
                np.array((0.20, -0.35, 0.90))
                if family == "JAR_SOURCE" else None
            ),
            final_tracking_tolerance=0.035 if family == "KETTLE" else 0.018,
            required_contact_geoms=required_contact_geoms,
            ik_angle_tolerance_rad=(
                np.deg2rad(2.5) if family == "KETTLE"
                else None
            ),
            ik_position_tolerance=None,
            approach_clearance_m=0.060 if family == "UTENSIL" else 0.120,
            ik_orientation_weight=0.30,
            grasp_candidates=(
                UtensilGraspCandidateGenerator.generate(
                    scene, body_id, observed["source_context"]["source_kind"]
                )
                if family == "UTENSIL"
                else JarGraspCandidateGenerator.generate(
                    scene,
                    mujoco.mj_name2id(
                        scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{body}_grasp"
                    ),
                ) if family == "JAR_SOURCE" else ()
            ),
            ik_restart_offsets=(
                (
                    (0.0, 0.0, 0.25, -0.25, 0.0, 0.0, 0.0),
                    (0.0, 0.0, -0.25, 0.25, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0, 0.25, -0.25, 0.0),
                    (0.0, 0.0, 0.0, 0.0, -0.25, 0.25, 0.0),
                )
                if family == "JAR_SOURCE"
                or observed["source_context"]["source_kind"] == "DRAWER"
                else ()
            ),
            intermediate_ik_position_tolerance=(
                0.030
                if observed["source_context"]["source_kind"] == "DRAWER"
                else 0.025 if family == "JAR_SOURCE" else None
            ),
            intermediate_ik_angle_tolerance_rad=(
                np.deg2rad(4.0)
                if observed["source_context"]["source_kind"] == "DRAWER"
                else np.deg2rad(3.0) if family == "JAR_SOURCE" else None
            ),
        )
    return specs


class KitchenPlacementResolver:
    """Deterministic dynamic placement allocation over symbolic destinations."""

    def __init__(self, scene, inventory: dict[str, Any], resolution: dict[str, Any]):
        self.scene = scene
        self.inventory = inventory
        self.binding = {row["generic_object_id"]: row for row in resolution["accepted"]}
        self.allocated_serving: dict[str, PlacementTarget] = {}
        self.serving_placements: dict[str, ServingPlacementState] = {}
        self.inventory_by_id = {
            row["generic_object_id"]: row for row in inventory["objects"]
        }
        self.support_height_by_id = {
            row["generic_object_id"]: max(
                0.008, float(row["observed_centroid_world_m"][2]) - 0.58
            )
            for row in inventory["objects"]
        }
        self.serving_slot_by_id: dict[str, tuple[float, float]] = {}
        for function, row_y in (("coffee_vessel", -0.48), ("soup_bowl", -0.64)):
            group = [
                row for row in inventory["objects"]
                if function in set(row["selected_functions"])
            ]
            # Keep each payload's transfer corridor close to its observed X
            # coordinate.  This avoids sweeping bowls across other tabletop
            # objects while preserving deterministic, non-overlapping slots.
            group.sort(key=lambda row: (row["observed_centroid_world_m"][0], row["generic_object_id"]))
            for slot_x, row in zip((-0.16, 0.0, 0.16), group):
                self.serving_slot_by_id[row["generic_object_id"]] = (slot_x, row_y)

    def footprint(self, object_id: str) -> tuple[float, float]:
        dimensions = self.inventory_by_id[object_id].get("observed_dimensions_m", {})
        length = dimensions.get("length")
        width = dimensions.get("width")
        # Missing measured geometry cannot be invented. The conservative
        # execution fallback is used only for collision spacing, not inference.
        if length is None or width is None:
            return (0.10, 0.10)
        return (float(length), float(width))

    @staticmethod
    def footprints_overlap(
        centre_a: tuple[float, float], footprint_a: tuple[float, float],
        centre_b: tuple[float, float], footprint_b: tuple[float, float],
        clearance_m: float = 0.012,
    ) -> bool:
        return (
            abs(centre_a[0] - centre_b[0])
            < (footprint_a[0] + footprint_b[0]) / 2.0 + clearance_m
            and abs(centre_a[1] - centre_b[1])
            < (footprint_a[1] + footprint_b[1]) / 2.0 + clearance_m
        )

    def record_successful_serving_placement(
        self, object_id: str, target: PlacementTarget
    ) -> None:
        self.serving_placements[object_id] = ServingPlacementState(
            object_id=object_id,
            backend_body=self.binding[object_id]["physical_backend_body"],
            centre_xy_m=tuple(target.target_position_world_m[:2]),
            footprint_xy_m=self.footprint(object_id),
            yaw_world_rad=target.target_yaw_world_rad,
            support_backend=target.support_backend or "",
        )

    def resolve(self, object_id: str, destination: str) -> PlacementTarget:
        if object_id not in self.binding:
            raise ValueError("DESTINATION_RESOLUTION_FAILED: unresolved object")
        if destination == "serving_area":
            if object_id not in self.serving_slot_by_id:
                raise ValueError("DESTINATION_RESOLUTION_FAILED: object is not a serving target")
            x, y = self.serving_slot_by_id[object_id]
            footprint = self.footprint(object_id)
            candidates = [
                (x, y), (-0.16, y), (0.0, y), (0.16, y),
                (-0.16, -0.56), (0.0, -0.56), (0.16, -0.56),
            ]
            valid_candidates = []
            for candidate in dict.fromkeys(candidates):
                edge_x = 0.25 - abs(candidate[0]) - footprint[0] / 2.0
                edge_y = 0.15 - abs(candidate[1] + 0.56) - footprint[1] / 2.0
                edge = min(edge_x, edge_y)
                if edge < 0.008:
                    continue
                if any(
                    self.footprints_overlap(
                        candidate, footprint, placed.centre_xy_m,
                        placed.footprint_xy_m,
                    )
                    for placed in self.serving_placements.values()
                ):
                    continue
                valid_candidates.append((edge, candidate))
            if not valid_candidates:
                raise ValueError("DESTINATION_RESOLUTION_FAILED: serving support is full")
            # Preserve the semantic row/column slot when it is physically
            # valid, then maximize clearance, then coordinate tie-break.
            _, (x, y) = min(
                valid_candidates,
                key=lambda item: (
                    (item[1][0] - candidates[0][0]) ** 2
                    + (item[1][1] - candidates[0][1]) ** 2,
                    -item[0],
                    item[1],
                ),
            )
            support_height = self.support_height_by_id[object_id]
            target = PlacementTarget(
                object_id, destination, "SERVING_SUPPORT",
                (x, y, 0.58 + support_height), 0.0, "serving_surface", None,
                KitchenWorkspace.HOME, 0.012, "ON",
                "OBSERVED_FOOTPRINT_PERSISTENT_SERVING_ALLOCATOR_V2",
            )
            self.allocated_serving[object_id] = target
            return target
        if destination == "countertop":
            row = next(row for row in self.inventory["objects"] if row["generic_object_id"] == object_id)
            position = tuple(float(x) for x in row["observed_centroid_world_m"])
            return PlacementTarget(
                object_id, destination, "SOURCE_RETURN", position, 0.0,
                "counter_surface", None, KitchenWorkspace.HOME, 0.03,
                "RETURNED_TO_SOURCE", "ORIGINAL_OBSERVED_SOURCE_POSE_V1",
            )
        if destination in self.binding:
            target_backend = self.binding[destination]["physical_backend_body"]
            target_body = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, target_backend
            )
            current = self.scene.data.xpos[target_body]
            position = (
                # Place the utensil beside, rather than over, the vessel.  A
                # lateral service offset preserves a collision-free vertical
                # gripper corridor while remaining uniquely associated with
                # the intended bowl.
                float(current[0]), float(current[1] - 0.16),
                0.59 + self.support_height_by_id[object_id],
            )
            return PlacementTarget(
                object_id, destination, "OBJECT_RELATIVE_DESTINATION", position,
                0.0, "serving_surface", destination, KitchenWorkspace.HOME,
                0.025, "WITH",
                "CURRENT_RESOLVED_TARGET_POSE_PLUS_SERVICE_OFFSET_V1",
            )
        raise ValueError(f"DESTINATION_RESOLUTION_FAILED: {destination}")


def inspect_held_object_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    generic_object_id: str,
    backend_body: str,
    supported_backend_bodies: set[str],
) -> HeldObjectState:
    profile = manipulation_profile("google")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.gripper_body)
    weld_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{backend_body}"
    )
    reasons = []
    active = weld_id >= 0 and bool(data.eq_active[weld_id])
    if not active:
        reasons.append("GRASP_WELD_INACTIVE")
    active_payload_welds = []
    for body in supported_backend_bodies:
        candidate = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{body}"
        )
        if candidate >= 0 and data.eq_active[candidate]:
            active_payload_welds.append(candidate)
    exclusive = active_payload_welds == [weld_id]
    if not exclusive:
        reasons.append("UNEXPECTED_ACTIVE_PAYLOAD_WELD")
    if weld_id < 0 or int(model.eq_obj1id[weld_id]) != gripper_id or int(model.eq_obj2id[weld_id]) != body_id:
        reasons.append("WELD_BODY_MISMATCH")
    relative = data.xpos[body_id] - data.xpos[gripper_id]
    if float(np.linalg.norm(relative)) > 0.55:
        reasons.append("PAYLOAD_SEPARATED_FROM_GRIPPER")
    payload_geoms = _body_geom_ids(model, body_id)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_contact = any(
        floor_id in {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        and bool(payload_geoms & {int(data.contact[index].geom1), int(data.contact[index].geom2)})
        for index in range(data.ncon)
    )
    if floor_contact:
        reasons.append("PAYLOAD_ON_FLOOR")
    finger_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in profile.finger_joints
    ]
    finger_values = tuple(float(data.qpos[model.jnt_qposadr[j]]) for j in finger_ids)
    relative_rotation = data.xmat[gripper_id].reshape(3, 3).T @ data.xmat[body_id].reshape(3, 3)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, relative_rotation.ravel())
    return HeldObjectState(
        generic_object_id=generic_object_id,
        backend_body=backend_body,
        weld_id=weld_id,
        weld_active=active,
        gripper_body=profile.gripper_body,
        relative_position_m=tuple(map(float, relative)),
        relative_orientation_wxyz=tuple(map(float, quaternion)),
        finger_joint_positions=finger_values,
        exclusive_payload_weld=exclusive,
        floor_contact=floor_contact,
        validation_status="TRUE" if not reasons else "FALSE",
        rejection_reasons=tuple(reasons),
    )


class KitchenObjectManipulationExecutor:
    """Synchronous typed facade over the existing Google pick/place executor."""

    def __init__(self, scene, inventory: dict[str, Any], resolution: dict[str, Any], *, step_callback=None):
        self.scene = scene
        self.inventory = inventory
        self.resolution = resolution
        self.step_callback = step_callback
        self.by_id = {row["generic_object_id"]: row for row in resolution["accepted"]}
        self.inventory_by_id = {row["generic_object_id"]: row for row in inventory["objects"]}
        self.backend_bodies = {row["physical_backend_body"] for row in resolution["accepted"]}
        self.placement_resolver = KitchenPlacementResolver(scene, inventory, resolution)
        specs = make_kitchen_pick_specs(scene, inventory, resolution)
        self.executor = CalibratedPickPlaceExecutor(
            scene.model,
            scene.data,
            "google",
            scene.scene_name,
            pick_specs_override=specs,
            calibrated_objects_override=tuple(specs),
            base_stance=np.zeros(3),
            mounting_allowances=PHASE_B_MOUNT_ALLOWANCES,
        )

    def sync_workspace(self, workspace: KitchenWorkspace) -> None:
        """Make the generic local primitive relative to the live named pose."""
        base = self.executor.data.qpos[self.executor.base_qpos].copy()
        self.executor.base_stance = base
        self.executor.base_manipulation_target = base.copy()

    @staticmethod
    def _target_dict(target: PlacementTarget) -> dict[str, Any]:
        return {**asdict(target), "required_workspace": target.required_workspace.value}

    def _step_until_stable_mode(self, maximum_steps: int = 30000) -> int:
        for step in range(1, maximum_steps + 1):
            self.executor.update()
            mujoco.mj_step(self.scene.model, self.scene.data)
            if self.step_callback:
                self.step_callback()
            if self.executor.mode in {"holding", "idle", "failed"}:
                return step
        raise RuntimeError("MANIPULATION_EXECUTION_TIMEOUT")

    def _drawer_presentation(
        self,
        generic_object_id: str,
        backend: str,
    ) -> dict[str, Any]:
        """Physically slide a low drawer utensil toward the open edge."""
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
        )
        handle_id = next(
            (
                geom_id
                for geom_id in _body_geom_ids(self.scene.model, body_id)
                if "handle_collision" in (
                    mujoco.mj_id2name(
                        self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    ) or ""
                )
            ),
            None,
        )
        if handle_id is None:
            raise RuntimeError("PRESENTATION_FAILED: target has no handle geometry")

        neighbour_start: dict[str, np.ndarray] = {}
        source_container = self.inventory_by_id[generic_object_id][
            "source_context"
        ]["source_container"]
        for object_id, row in self.inventory_by_id.items():
            if object_id == generic_object_id:
                continue
            if row["source_context"].get("source_container") != source_container:
                continue
            binding = self.by_id.get(object_id)
            if binding is None:
                continue
            neighbour_body = mujoco.mj_name2id(
                self.scene.model,
                mujoco.mjtObj.mjOBJ_BODY,
                binding["physical_backend_body"],
            )
            if neighbour_body >= 0:
                neighbour_start[object_id] = self.scene.data.xpos[
                    neighbour_body
                ].copy()

        target_start = self.scene.data.xpos[body_id].copy()
        handle = self.scene.data.geom_xpos[handle_id].copy()
        drawer_candidates = self.executor.pick_specs[backend].grasp_candidates
        handle_axis = self.scene.data.geom_xmat[handle_id].reshape(3, 3)[:, 2]
        handle_yaw = float(np.arctan2(handle_axis[1], handle_axis[0]))
        # Use the mirrored inward-facing diagonal on the two drawer sides.
        # Capsule axes are sign-ambiguous (D2 reports the same physical axis
        # reversed by 180 degrees), so adding a constant to raw axis yaw is
        # not mirror symmetric.
        canonical_handle_yaw = (
            (handle_yaw + np.pi / 2.0) % np.pi
        ) - np.pi / 2.0
        presentation_yaw = (
            canonical_handle_yaw
            + np.sign(handle[0]) * np.deg2rad(60.0)
        )
        rotation = (
            _axis_angle_rotation(np.array((0.0, 0.0, presentation_yaw)))
            @ manipulation_profile("google").top_down_rotation
        )
        carry_rotation = next(
            candidate.carry_rotation_world
            for candidate in self.executor.pick_specs[backend].grasp_candidates
            if candidate.candidate_id.startswith("drawer_front_")
        )
        contact_height = float(handle[2] + 0.030)
        # Offset the gripper toward the robot sagittal centreline.  A fixed
        # +X offset is correct for D1 (negative X) but pushes the symmetric D2
        # target farther out of reach.
        contact_x = float(handle[0] - np.sign(handle[0]) * 0.032)
        carry = self.executor.pick_specs[backend].carry_position
        if carry is None:
            carry = manipulation_profile("google").carry_position
        # D1/D2 open toward negative world Y.  Approach through the open front,
        # press the closed gripper exterior onto the handle, and use friction
        # to drag it outward.  This needs no inaccessible behind-object pose.
        path = (
            tuple(map(float, carry)),
            (contact_x, float(handle[1] - 0.30), contact_height + 0.25),
            (contact_x, float(handle[1]), contact_height + 0.14),
            (contact_x, float(handle[1]), contact_height),
            (contact_x, float(handle[1] - 0.070), contact_height - 0.005),
        )
        # A parallel-jaw gripper has a 180-degree yaw-equivalent carry pose.
        # Select between those two kinematically equivalent orientations using
        # strict IK, rather than attaching object-name-specific wrist angles.
        carry_rotations = (
            ("measured_axis", carry_rotation),
            (
                "measured_axis_pi_equivalent",
                _axis_angle_rotation(np.array((0.0, 0.0, np.pi)))
                @ carry_rotation,
            ),
        )
        contact_rotations = (
            ("measured_axis", rotation),
            (
                "measured_axis_pi_equivalent",
                _axis_angle_rotation(np.array((0.0, 0.0, np.pi))) @ rotation,
            ),
        )
        carry_failures = []
        for contact_branch, candidate_contact_rotation in contact_rotations:
            for carry_branch, candidate_carry_rotation in carry_rotations:
                try:
                    result = self.executor.execute_contact_presentation(
                        backend,
                        path,
                        candidate_contact_rotation,
                        step_callback=self.step_callback,
                        waypoint_timeout_steps=300,
                        carry_rotation_world=candidate_carry_rotation,
                    )
                    result["contact_orientation_branch"] = contact_branch
                    result["contact_rotation_world"] = (
                        candidate_contact_rotation.tolist()
                    )
                    result["carry_orientation_branch"] = carry_branch
                    result["carry_rotation_world"] = (
                        candidate_carry_rotation.tolist()
                    )
                    rotation = candidate_contact_rotation
                    break
                except RuntimeError as error:
                    carry_failures.append(
                        f"{contact_branch}/{carry_branch}: {error}"
                    )
                    if not any(
                        marker in str(error)
                        for marker in (
                            "IK misses",
                            "Unsafe IK segment",
                            "carry path collision",
                        )
                    ):
                        raise
            else:
                continue
            break
        else:
            raise RuntimeError("; ".join(carry_failures))
        target_end = self.scene.data.xpos[body_id].copy()
        neighbour_displacements = {}
        for object_id, start in neighbour_start.items():
            neighbour_body = mujoco.mj_name2id(
                self.scene.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.by_id[object_id]["physical_backend_body"],
            )
            neighbour_displacements[object_id] = float(
                np.linalg.norm(self.scene.data.xpos[neighbour_body] - start)
            )
        maximum_neighbour = max(neighbour_displacements.values(), default=0.0)
        displacement = float(np.linalg.norm(target_end - target_start))
        result.update(
            target_displacement_m=displacement,
            neighbour_displacements_m=neighbour_displacements,
            maximum_neighbour_displacement_m=maximum_neighbour,
            displacement_bounds_m=(0.008, 0.05),
            neighbour_displacement_limit_m=0.02,
            source_container=source_container,
        )
        if not result["success"]:
            raise RuntimeError("PRESENTATION_CONTACT_FAILED: no robot-target contact")
        if not 0.008 <= displacement <= 0.05:
            raise RuntimeError(
                f"PRESENTATION_POSTCONDITION_FAILED: displacement {displacement:.3f} m"
            )
        if maximum_neighbour > 0.02:
            raise RuntimeError(
                "NEIGHBOUR_OBJECT_DISTURBED: "
                f"maximum displacement {maximum_neighbour:.3f} m"
            )
        if result["grasp_weld_active"]:
            raise RuntimeError("PRESENTATION_FAILED: grasp weld became active")
        result["postcondition_valid"] = True
        return result

    def _drawer_tip_presentation(self, backend: str) -> dict[str, Any]:
        """Press the utensil head to raise the opposite handle end."""
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
        )
        geom_ids = _body_geom_ids(self.scene.model, body_id)
        handle_id = next(
            geom_id for geom_id in geom_ids
            if "handle_collision" in (
                mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                ) or ""
            )
        )
        head_id = next(
            geom_id for geom_id in geom_ids
            if "bowl_collision" in (
                mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                ) or ""
            )
        )
        handle_start_z = float(self.scene.data.geom_xpos[handle_id, 2])
        head = self.scene.data.geom_xpos[head_id].copy()
        handle_axis = self.scene.data.geom_xmat[handle_id].reshape(3, 3)[:, 2]
        yaw = float(np.arctan2(handle_axis[1], handle_axis[0])) + np.deg2rad(300.0)
        rotation = (
            _axis_angle_rotation(np.array((0.0, 0.0, yaw)))
            @ manipulation_profile("google").top_down_rotation
        )
        carry_rotation = next(
            candidate.carry_rotation_world
            for candidate in self.executor.pick_specs[backend].grasp_candidates
            if candidate.candidate_id.startswith("drawer_front_")
        )
        carry = self.executor.pick_specs[backend].carry_position
        contact_x = float(head[0] + 0.032)
        contact_z = float(head[2] + 0.045)
        path = (
            tuple(map(float, carry)),
            (contact_x, float(head[1] - 0.25), contact_z + 0.25),
            (contact_x, float(head[1]), contact_z + 0.12),
            (contact_x, float(head[1]), contact_z),
            (contact_x, float(head[1]), contact_z - 0.008),
            (contact_x, float(head[1] - 0.10), contact_z + 0.16),
        )
        result = self.executor.execute_contact_presentation(
            backend,
            path,
            rotation,
            step_callback=self.step_callback,
            waypoint_timeout_steps=300,
            carry_rotation_world=carry_rotation,
        )
        handle_elevation = float(
            self.scene.data.geom_xpos[handle_id, 2] - handle_start_z
        )
        result.update(
            strategy="CONTACT_DRIVEN_DRAWER_TIP",
            handle_elevation_m=handle_elevation,
            handle_elevation_target_m=(0.008, 0.040),
            postcondition_valid=bool(
                result["success"]
                and 0.008 <= handle_elevation <= 0.040
                and not result["grasp_weld_active"]
            ),
        )
        if not result["postcondition_valid"]:
            raise RuntimeError(
                "PRESENTATION_POSTCONDITION_FAILED: handle elevation "
                f"{handle_elevation:.3f} m"
            )
        return result

    def pick(
        self,
        generic_object_id: str,
        current_workspace: KitchenWorkspace,
        physically_open_containers: set[str],
    ) -> PhysicalPickResult:
        started = time.perf_counter()
        if generic_object_id not in self.by_id:
            return PhysicalPickResult(generic_object_id, "", {}, "", "", False,
                "ENTITY_RESOLUTION_FAILED", ObjectExecutionFailureCode.ENTITY_RESOLUTION_FAILED.value, "Generic ID is unresolved",
                0, 0.0, False, (), (), None, None, False, None)
        binding = self.by_id[generic_object_id]
        context_row = self.inventory_by_id[generic_object_id]["source_context"]
        required = KitchenWorkspace(context_row["required_workspace"])
        if current_workspace != required:
            return PhysicalPickResult(generic_object_id, binding["physical_backend_body"], context_row,
                required.value, binding["grasp_family"], False,
                "WORKSPACE_PRECONDITION_UNSATISFIED",
                ObjectExecutionFailureCode.WORKSPACE_PRECONDITION_UNSATISFIED.value,
                f"PICK requires {required.value}",
                0, time.perf_counter() - started, False, (), (), None, None, False, None)
        container = context_row["source_container"]
        if container and container not in physically_open_containers:
            return PhysicalPickResult(generic_object_id, binding["physical_backend_body"], context_row,
                required.value, binding["grasp_family"], False, "CONTAINER_ACCESS_REQUIRED",
                ObjectExecutionFailureCode.CONTAINER_ACCESS_REQUIRED.value,
                f"PICK requires physical OPEN({container})",
                0, time.perf_counter() - started, False, (), (), None, None, False, None)
        backend = binding["physical_backend_body"]
        weld_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{backend}"
        )
        before_pos = self.scene.data.xpos[
            mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend)
        ].copy()
        # HOME covers the full main-table workspace, while a bounded local
        # approach aligns the arm with the selected observed object.  This is
        # not a named workspace transition and is retracted by the primitive.
        self.sync_workspace(current_workspace)
        local = np.zeros(3)
        if current_workspace == KitchenWorkspace.HOME:
            family_forward = {
                "UTENSIL": (
                    0.25
                ),
                "KETTLE": 0.20,
                "JAR_SOURCE": 0.20,
            }.get(binding["grasp_family"], 0.20)
            lateral_limit = (
                0.28
                if context_row["source_kind"] == "DRAWER"
                else 0.23 if binding["grasp_family"] == "JAR_SOURCE" else 0.18
            )
            lateral = float(np.clip(-before_pos[0], -lateral_limit, lateral_limit))
            local = np.array((family_forward, lateral, 0.0))
        self.executor.base_manipulation_target = self.executor.base_stance + local
        # The profile carry point is world-calibrated at the HOME base pose.
        # A bounded local base approach must translate that waypoint with the
        # base, otherwise the arm is incorrectly commanded back through the
        # original world-frame point before descending to the object.
        world_translation = np.array((-local[1], local[0], 0.0))
        if self.executor.pick_specs[backend].carry_position is None:
            self.executor.pick_specs[backend] = replace(
                self.executor.pick_specs[backend],
                carry_position=(
                    manipulation_profile("google").carry_position
                    + world_translation
                ),
            )
        presentation = None
        presentation_grasp_adopted = False
        direct_grasp_analysis = None
        drawer_presentation_required = False
        if (
            context_row["source_kind"] == "DRAWER"
            and binding["grasp_family"] == "UTENSIL"
        ):
            try:
                self.executor.move_to_local_manipulation_base(
                    step_callback=self.step_callback
                )
                # Match the normal pick-base approach's settled planning
                # state before classifying direct feasibility.  Planning on
                # the first in-tolerance base tick can produce a false
                # collision/IK rejection for the opposite drawer.
                for _ in range(120):
                    mujoco.mj_step(self.scene.model, self.scene.data)
                    if self.step_callback:
                        self.step_callback()
                direct_grasp_analysis = (
                    self.executor.direct_pick_plan_feasibility(backend)
                )
                drawer_presentation_required = not bool(
                    direct_grasp_analysis["feasible"]
                )
                if not drawer_presentation_required:
                    local_target = self.executor.base_manipulation_target.copy()
                    self.executor.base_manipulation_target = (
                        self.executor.base_stance.copy()
                    )
                    self.executor.move_to_local_manipulation_base(
                        step_callback=self.step_callback
                    )
                    self.executor.base_manipulation_target = local_target
            except RuntimeError:
                # Let the normal structured execution path report base-motion
                # failures rather than relabeling them as grasp infeasibility.
                drawer_presentation_required = False
        if (
            context_row["source_kind"] == "DRAWER"
            and binding["grasp_family"] == "UTENSIL"
            and drawer_presentation_required
        ):
            local_target = self.executor.base_manipulation_target.copy()
            try:
                self.executor.move_to_local_manipulation_base(
                    step_callback=self.step_callback
                )
                presentation_origin = self.scene.data.xpos[
                    mujoco.mj_name2id(
                        self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
                    )
                ].copy()
                presentation_attempts = []
                cumulative = 0.0
                followup_blocker = None
                for _ in range(1):
                    try:
                        presentation_attempts.append(
                            self._drawer_presentation(generic_object_id, backend)
                        )
                    except RuntimeError as error:
                        if presentation_attempts:
                            followup_blocker = str(error)
                            break
                        raise
                    cumulative = float(
                        np.linalg.norm(
                            self.scene.data.xpos[
                                mujoco.mj_name2id(
                                    self.scene.model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    backend,
                                )
                            ]
                            - presentation_origin
                        )
                    )
                    if cumulative >= 0.035:
                        break
                tip_result = None
                bilateral_ready = bool(
                    presentation_attempts
                    and presentation_attempts[-1].get(
                        "bilateral_contact_steps", 0
                    ) >= 5
                )
                if cumulative < 0.035 and not bilateral_ready:
                    try:
                        tip_result = self._drawer_tip_presentation(backend)
                        presentation_attempts.append(tip_result)
                    except RuntimeError as error:
                        followup_blocker = (
                            f"{followup_blocker}; tip: {error}"
                        )
                presentation = {
                    "strategy": "CONTACT_DRIVEN_DRAWER_MULTI_STROKE",
                    "direct_grasp_analysis": direct_grasp_analysis,
                    "success": (
                        0.008 <= cumulative <= 0.080
                        and (tip_result is None or tip_result["postcondition_valid"])
                    ),
                    "attempts": presentation_attempts,
                    "attempt_count": len(presentation_attempts),
                    "cumulative_target_displacement_m": cumulative,
                    # Match the independently validated per-stroke lower
                    # bound.  Requiring 10 mm here rejected otherwise valid
                    # 8--10 mm presentations purely because of reset noise.
                    "cumulative_displacement_bounds_m": (0.008, 0.080),
                    "followup_stroke_blocker": followup_blocker,
                    "grasp_weld_active_during_presentation": any(
                        attempt["grasp_weld_active"]
                        for attempt in presentation_attempts
                    ),
                    "direct_object_qpos_write": False,
                }
                if not presentation["success"]:
                    raise RuntimeError(
                        "PRESENTATION_POSTCONDITION_FAILED: cumulative "
                        f"displacement {cumulative:.3f} m"
                    )
                if bilateral_ready:
                    handle_id = next(
                        geom_id for geom_id in _body_geom_ids(
                            self.scene.model,
                            mujoco.mj_name2id(
                                self.scene.model,
                                mujoco.mjtObj.mjOBJ_BODY,
                                backend,
                            ),
                        )
                        if "handle_collision" in (
                            mujoco.mj_id2name(
                                self.scene.model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                geom_id,
                            ) or ""
                        )
                    )
                    axis = self.scene.data.geom_xmat[handle_id].reshape(3, 3)[:, 2]
                    yaw = float(np.arctan2(axis[1], axis[0])) + np.deg2rad(300.0)
                    regrasp_rotation = np.asarray(
                        presentation_attempts[-1].get(
                            "contact_rotation_world",
                            _axis_angle_rotation(np.array((0.0, 0.0, yaw)))
                            @ manipulation_profile("google").top_down_rotation,
                        ),
                        dtype=float,
                    )
                    carry_rotation = np.asarray(
                        presentation_attempts[-1].get(
                            "carry_rotation_world",
                            next(
                                candidate.carry_rotation_world
                                for candidate in self.executor.pick_specs[
                                    backend
                                ].grasp_candidates
                                if candidate.candidate_id.startswith(
                                    "drawer_front_"
                                )
                            ),
                        ),
                        dtype=float,
                    )
                    presentation["bilateral_regrasp"] = (
                        self.executor.adopt_presented_bilateral_grasp(
                            backend,
                            regrasp_rotation,
                            carry_rotation,
                            preconfirmed_contact_steps=int(
                                presentation_attempts[-1].get(
                                    "terminal_bilateral_contact_steps", 0
                                )
                            ),
                            step_callback=self.step_callback,
                        )
                    )
                    presentation_grasp_adopted = True
                else:
                    self.executor.base_manipulation_target = (
                        self.executor.base_stance.copy()
                    )
                    self.executor.move_to_local_manipulation_base(
                        step_callback=self.step_callback
                    )
                    self.executor.base_manipulation_target = local_target
            except RuntimeError as error:
                message = str(error)
                if "IK" in message:
                    code = ObjectExecutionFailureCode.PRESENTATION_IK_FAILED
                elif "collision" in message.lower():
                    code = ObjectExecutionFailureCode.PRESENTATION_PATH_COLLISION
                elif "CONTACT" in message:
                    code = ObjectExecutionFailureCode.PRESENTATION_CONTACT_FAILED
                elif "NEIGHBOUR" in message:
                    code = ObjectExecutionFailureCode.NEIGHBOUR_OBJECT_DISTURBED
                elif "POSTCONDITION" in message:
                    code = ObjectExecutionFailureCode.PRESENTATION_POSTCONDITION_FAILED
                else:
                    code = ObjectExecutionFailureCode.PRESENTATION_FAILED
                return PhysicalPickResult(
                    generic_object_id,
                    backend,
                    context_row,
                    required.value,
                    binding["grasp_family"],
                    False,
                    code.value,
                    code.value,
                    message,
                    0,
                    time.perf_counter() - started,
                    False,
                    (),
                    (),
                    None,
                    None,
                    False,
                    None,
                    presentation=presentation,
                    direct_grasp_analysis=direct_grasp_analysis,
                )
        if not presentation_grasp_adopted:
            self.executor.request_pick(backend)
        try:
            steps = self._step_until_stable_mode()
        except RuntimeError as error:
            site_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{backend}_grasp"
            )
            grip_error = None
            if site_id >= 0:
                grip_error = float(np.linalg.norm(
                    self.scene.data.site_xpos[self.executor.grip_site_id]
                    - self.scene.data.site_xpos[site_id]
                ))
            return PhysicalPickResult(
                generic_object_id, backend, context_row, required.value,
                binding["grasp_family"], False,
                "REGRASP_FAILED" if presentation_grasp_adopted else "GRASP_FAILED",
                (
                    ObjectExecutionFailureCode.REGRASP_FAILED.value
                    if presentation_grasp_adopted else ObjectExecutionFailureCode.GRASP_FAILED.value
                ),
                f"{error}; mode={self.executor.mode}; status={self.executor.status}; "
                f"grip_site_error_m={grip_error}",
                30000, time.perf_counter() - started, False, (), (), None,
                None, False, None, presentation=presentation,
                direct_grasp_analysis=direct_grasp_analysis,
            )
        if self.executor.mode != "holding":
            return PhysicalPickResult(generic_object_id, backend, context_row, required.value,
                binding["grasp_family"], False,
                "REGRASP_FAILED" if presentation_grasp_adopted else "GRASP_FAILED",
                (ObjectExecutionFailureCode.REGRASP_FAILED.value
                 if presentation_grasp_adopted else ObjectExecutionFailureCode.GRASP_FAILED.value),
                self.executor.failure or self.executor.status, steps,
                time.perf_counter() - started, False, (), (), None, None, False,
                None, presentation=presentation,
                direct_grasp_analysis=direct_grasp_analysis)
        state = inspect_held_object_state(
            self.scene.model, self.scene.data, generic_object_id, backend, self.backend_bodies
        )
        after_pos = self.scene.data.xpos[
            mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend)
        ].copy()
        success = state.validation_status == "TRUE" and float(after_pos[2] - before_pos[2]) > 0.03
        return PhysicalPickResult(
            generic_object_id, backend, context_row, required.value, binding["grasp_family"],
            success, "PICK_SUCCESS" if success else "HELD_STATE_INVALID",
            None if success else ObjectExecutionFailureCode.HELD_STATE_INVALID.value,
            "" if success else ",".join(state.rejection_reasons),
            steps, time.perf_counter() - started,
            self.executor.confirmed_contact_sides == (0, 1),
            self.executor.confirmed_contact_sides,
            self.executor.confirmed_contact_geoms,
            self.executor.attachment_translation_snap_m,
            self.executor.attachment_angle_snap_rad,
            self.executor.navigation_safe and self.executor.mode == "holding",
            asdict(state),
            selected_grasp_candidate_id=self.executor.selected_grasp_candidate_id,
            target_contact_geoms=self.executor.confirmed_target_contact_geoms,
            presentation=presentation,
            direct_grasp_analysis=direct_grasp_analysis,
        )

    def place(
        self,
        generic_object_id: str,
        symbolic_destination: str,
        current_workspace: KitchenWorkspace,
    ) -> PhysicalPlaceResult:
        started = time.perf_counter()
        binding = self.by_id.get(generic_object_id)
        backend = binding["physical_backend_body"] if binding else ""
        if binding is None or self.executor.held_object != backend:
            return PhysicalPlaceResult(
                generic_object_id, backend, symbolic_destination, {}, False,
                "HELD_STATE_INVALID", ObjectExecutionFailureCode.HELD_STATE_INVALID.value,
                "The requested generic object is not physically held", 0,
                time.perf_counter() - started, False, False, False, False, False,
            )
        target = self.placement_resolver.resolve(generic_object_id, symbolic_destination)
        if current_workspace != target.required_workspace:
            return PhysicalPlaceResult(
                generic_object_id, backend, symbolic_destination, self._target_dict(target), False,
                "WORKSPACE_PRECONDITION_UNSATISFIED",
                ObjectExecutionFailureCode.WORKSPACE_PRECONDITION_UNSATISFIED.value,
                f"PLACE requires {target.required_workspace.value}", 0,
                time.perf_counter() - started, False, False, False, False, False,
            )
        position = np.asarray(target.target_position_world_m, float)
        self.sync_workspace(current_workspace)
        local = np.zeros(3)
        if current_workspace == KitchenWorkspace.HOME:
            local = np.array((0.20, float(np.clip(-position[0], -0.18, 0.18)), 0.0))
        self.executor.base_manipulation_target = self.executor.base_stance + local
        yaw = target.target_yaw_world_rad
        rotation = np.array(
            ((np.cos(yaw), -np.sin(yaw), 0.0),
             (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0))
        ) @ manipulation_profile("google").top_down_rotation
        if (
            target.destination_kind == "SOURCE_RETURN"
            or binding["grasp_family"] == "UTENSIL"
        ):
            grasp_rotation = self.executor.pick_specs[backend].top_down_rotation
            if grasp_rotation is not None:
                rotation = grasp_rotation
        self.executor.request_place_world(position, rotation)
        steps = self._step_until_stable_mode()
        if self.executor.mode != "idle":
            return PhysicalPlaceResult(
                generic_object_id, backend, symbolic_destination, self._target_dict(target), False,
                "PLACEMENT_FAILED", ObjectExecutionFailureCode.PLACEMENT_FAILED.value,
                self.executor.failure or self.executor.status, steps,
                time.perf_counter() - started, False, False, False, False, False,
            )
        for _ in range(400):
            mujoco.mj_step(self.scene.model, self.scene.data)
            steps += 1
        mujoco.mj_forward(self.scene.model, self.scene.data)
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
        )
        object_geoms = _body_geom_ids(self.scene.model, body_id)
        support_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_GEOM,
            target.support_backend or "",
        )
        floor_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        support_contact = floor_contact = False
        contact_pairs = []
        invalid_object_contacts = []
        other_payload_geom_ids = {
            geom_id
            for other_backend in self.backend_bodies - {backend}
            for geom_id in _body_geom_ids(
                self.scene.model,
                mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_BODY, other_backend
                ),
            )
        }
        for index in range(self.scene.data.ncon):
            pair = {
                int(self.scene.data.contact[index].geom1),
                int(self.scene.data.contact[index].geom2),
            }
            if not object_geoms & pair:
                continue
            named_pair = tuple(
                mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom
                ) or f"geom_{geom}"
                for geom in sorted(pair)
            )
            contact_pairs.append(named_pair)
            support_contact |= support_id in pair
            floor_contact |= floor_id in pair
            if other_payload_geom_ids & pair:
                invalid_object_contacts.append(named_pair)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.scene.model, self.scene.data, mujoco.mjtObj.mjOBJ_BODY,
            body_id, velocity, 0,
        )
        # MuJoCo spatial velocity is [angular, linear].
        angular_speed = float(np.linalg.norm(velocity[:3]))
        linear_speed = float(np.linalg.norm(velocity[3:]))
        stable = angular_speed <= 0.10 and linear_speed <= 0.02
        weld_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"google:pick_weld_{backend}",
        )
        released = weld_id >= 0 and not bool(self.scene.data.eq_active[weld_id])
        relative_ok = True
        relative_distance = None
        intended_target_uniquely_closest = None
        if target.target_object_id:
            target_backend = self.by_id[target.target_object_id]["physical_backend_body"]
            target_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, target_backend
            )
            distance = float(np.linalg.norm(
                self.scene.data.xpos[body_id, :2] - self.scene.data.xpos[target_id, :2]
            ))
            relative_distance = distance
            relative_ok = 0.06 <= distance <= 0.18
            bowl_distances = {}
            for candidate_id, candidate in self.inventory_by_id.items():
                if "soup_bowl" not in set(candidate.get("selected_functions", ())):
                    continue
                candidate_binding = self.by_id.get(candidate_id)
                if candidate_binding is None:
                    continue
                candidate_body = mujoco.mj_name2id(
                    self.scene.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    candidate_binding["physical_backend_body"],
                )
                bowl_distances[candidate_id] = float(np.linalg.norm(
                    self.scene.data.xpos[body_id, :2]
                    - self.scene.data.xpos[candidate_body, :2]
                ))
            intended_target_uniquely_closest = bool(bowl_distances) and all(
                distance + 0.005 < other_distance
                for candidate_id, other_distance in bowl_distances.items()
                if candidate_id != target.target_object_id
            )
            relative_ok &= intended_target_uniquely_closest
        final_xy = self.scene.data.xpos[body_id, :2]
        footprint_inside = True
        actual_edge_margin = None
        footprint_corners = np.empty((0, 2))
        pairwise_payload_checks = []
        if target.destination_kind == "SERVING_SUPPORT":
            length, width = self.placement_resolver.footprint(generic_object_id)
            footprint_corners = oriented_rectangle_corners(
                final_xy, length, width, _body_yaw(self.scene.data, body_id)
            )
            support_axis = self.scene.data.geom_xmat[support_id].reshape(3, 3)[:2, 0]
            boundary = rectangle_inside_observed_support(
                footprint_corners,
                self.scene.data.geom_xpos[support_id, :2],
                support_axis,
                float(self.scene.model.geom_size[support_id, 0] * 2.0),
                float(self.scene.model.geom_size[support_id, 1] * 2.0),
            )
            actual_edge_margin = float(boundary["minimum_edge_margin_m"])
            footprint_inside = actual_edge_margin >= target.edge_margin_m
            for placed_id, placed in self.placement_resolver.serving_placements.items():
                other_binding = self.by_id.get(placed_id)
                if other_binding is None:
                    continue
                other_body = mujoco.mj_name2id(
                    self.scene.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    other_binding["physical_backend_body"],
                )
                other_length, other_width = self.placement_resolver.footprint(placed_id)
                other_corners = oriented_rectangle_corners(
                    self.scene.data.xpos[other_body, :2],
                    other_length,
                    other_width,
                    _body_yaw(self.scene.data, other_body),
                )
                check = oriented_rectangles_clearance(
                    footprint_corners, other_corners
                )
                check.update(
                    other_object_id=placed_id,
                    required_clearance_m=0.012,
                    valid_nonoverlap=(
                        not check["overlap"]
                        and check["signed_clearance_m"] >= 0.012
                    ),
                )
                pairwise_payload_checks.append(check)
            footprint_inside &= all(
                check["valid_nonoverlap"] for check in pairwise_payload_checks
            )
        source_return_xy_error = None
        source_region_membership = None
        upright_alignment = None
        source_return_ok = True
        if target.destination_kind == "SOURCE_RETURN":
            observed_source = np.asarray(
                self.inventory_by_id[generic_object_id]["observed_centroid_world_m"],
                float,
            )
            source_return_xy_error = float(np.linalg.norm(
                self.scene.data.xpos[body_id, :2] - observed_source[:2]
            ))
            # The planner-visible source region is observation provenance.
            # For tabletop sources, membership is verified by proximity to
            # that observed source point plus actual contact with the intended
            # countertop support; no hidden scene source mapping is consulted.
            source_region_membership = bool(
                source_return_xy_error <= 0.04 and support_contact
            )
            upright_alignment = float(
                self.scene.data.xmat[body_id].reshape(3, 3)[2, 2]
            )
            source_return_ok = (
                source_return_xy_error <= 0.04
                and source_region_membership
                and upright_alignment >= float(np.cos(np.deg2rad(15.0)))
            )
        verified = (
            released and support_contact and not floor_contact and stable
            and relative_ok and footprint_inside and source_return_ok
            and not invalid_object_contacts
        )
        if verified and target.destination_kind == "SERVING_SUPPORT":
            self.placement_resolver.record_successful_serving_placement(
                generic_object_id, target
            )
        return PhysicalPlaceResult(
            generic_object_id=generic_object_id,
            backend_body=backend,
            symbolic_destination=symbolic_destination,
            placement_target={
                **asdict(target),
                "required_workspace": target.required_workspace.value,
            },
            success=verified,
            status="PLACE_SUCCESS" if verified else "POSTCONDITION_FAILED",
            failure_code=(
                None if verified else ObjectExecutionFailureCode.PLACEMENT_FAILED.value
            ),
            message="" if verified else "Physical placement relation did not validate",
            physics_steps=steps,
            duration_s=time.perf_counter() - started,
            grasp_released=released,
            support_contact=support_contact,
            floor_contact=floor_contact,
            stable=stable,
            physical_relation_verified=verified,
            footprint_inside_support=footprint_inside,
            edge_margin_m=actual_edge_margin,
            object_relative_distance_m=relative_distance,
            final_body_position_world_m=tuple(
                float(x) for x in self.scene.data.xpos[body_id]
            ),
            object_contact_pairs=tuple(contact_pairs),
            linear_speed_m_s=linear_speed,
            angular_speed_rad_s=angular_speed,
            footprint_corners_world_m=tuple(
                tuple(map(float, corner)) for corner in footprint_corners
            ),
            pairwise_payload_checks=tuple(pairwise_payload_checks),
            invalid_object_contacts=tuple(invalid_object_contacts),
            source_return_xy_error_m=source_return_xy_error,
            source_region_membership=source_region_membership,
            upright_alignment=upright_alignment,
            intended_target_uniquely_closest=intended_target_uniquely_closest,
        )
