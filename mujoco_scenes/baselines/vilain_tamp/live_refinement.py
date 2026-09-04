"""Concrete planning-copy geometry backends for ViLaIn sequence refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_json, sha256_bytes
from .contracts import RefinementStage, SymbolicAction
from .refinement import (
    MuJoCoSequenceRefiner,
    PlanningSceneStateAdapter,
    RefinementStageBackend,
    RefinementStageContext,
    RefinementStageOutcome,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_BLACK_BOX_SKILLS = frozenset({"pour", "stir", "drive"})
_DATA_ARRAYS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
    "userdata",
    "plugin_state",
)


class LiveRefinementError(RuntimeError):
    """A concrete geometric stage could not certify its requested property."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        numeric_evidence: Mapping[str, float] | None = None,
        collision_pair: tuple[str, str] | None = None,
        recoverable_by_problem_revision: bool = True,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.numeric_evidence = dict(numeric_evidence or {})
        self.collision_pair = collision_pair
        self.recoverable_by_problem_revision = recoverable_by_problem_revision


@dataclass(frozen=True)
class RefinementThresholds:
    approach_clearance_m: float = 0.10
    placement_clearance_m: float = 0.03
    position_tolerance_m: float = 0.03
    angle_tolerance_rad: float = math.radians(15.0)
    joint_interpolation_step_rad: float = 0.035
    collision_resolution_rad: float = 0.035
    payload_collision_tolerance_m: float = 0.005
    pour_tilt_rad: float = math.radians(60.0)
    pour_min_tilt_rad: float = math.radians(45.0)
    skill_clearance_m: float = 0.015
    stir_axis_tolerance_rad: float = math.radians(20.0)
    stir_min_depth_m: float = 0.015
    stir_radius_fraction: float = 0.75
    drive_axis_tolerance_rad: float = math.radians(15.0)
    drive_min_depth_m: float = 0.005
    drive_max_depth_m: float = 0.08

    def __post_init__(self) -> None:
        values = (
            self.approach_clearance_m,
            self.placement_clearance_m,
            self.position_tolerance_m,
            self.angle_tolerance_rad,
            self.joint_interpolation_step_rad,
            self.collision_resolution_rad,
            self.payload_collision_tolerance_m,
            self.pour_tilt_rad,
            self.pour_min_tilt_rad,
            self.skill_clearance_m,
            self.stir_axis_tolerance_rad,
            self.stir_min_depth_m,
            self.stir_radius_fraction,
            self.drive_axis_tolerance_rad,
            self.drive_min_depth_m,
            self.drive_max_depth_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("refinement thresholds must be finite and positive")


@dataclass(frozen=True)
class EntityGeometry:
    entity_name: str
    body_id: int
    centroid_m: tuple[float, float, float]
    aabb_min_m: tuple[float, float, float]
    aabb_max_m: tuple[float, float, float]
    body_rotation: tuple[tuple[float, float, float], ...]
    dimensions_m: tuple[float, float, float]
    principal_axis_world: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "body_id": self.body_id,
            "centroid_m": list(self.centroid_m),
            "aabb_min_m": list(self.aabb_min_m),
            "aabb_max_m": list(self.aabb_max_m),
            "body_rotation": [list(row) for row in self.body_rotation],
            "dimensions_m": list(self.dimensions_m),
            "principal_axis_world": list(self.principal_axis_world),
        }


@dataclass(frozen=True)
class GeometricCandidate:
    candidate_id: str
    target_position_m: tuple[float, float, float]
    approach_position_m: tuple[float, float, float]
    target_rotation: tuple[tuple[float, float, float], ...]
    source: str
    skill_parameters: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_position_m": list(self.target_position_m),
            "approach_position_m": list(self.approach_position_m),
            "target_rotation": [list(row) for row in self.target_rotation],
            "source": self.source,
            "skill_parameters": dict(self.skill_parameters),
        }


@dataclass(frozen=True)
class PayloadAttachment:
    entity_name: str
    body_id: int
    free_joint_id: int
    qpos_address: int
    gripper_body_id: int
    gripper_relative_position_m: tuple[float, float, float]
    gripper_relative_rotation: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class IKEvaluation:
    candidate_id: str
    reachable: bool
    approach_qpos: tuple[float, ...]
    target_qpos: tuple[float, ...]
    approach_position_error_m: float | None
    target_position_error_m: float | None
    approach_angle_error_rad: float | None
    target_angle_error_rad: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reachable": self.reachable,
            "approach_qpos": list(self.approach_qpos),
            "target_qpos": list(self.target_qpos),
            "approach_position_error_m": self.approach_position_error_m,
            "target_position_error_m": self.target_position_error_m,
            "approach_angle_error_rad": self.approach_angle_error_rad,
            "target_angle_error_rad": self.target_angle_error_rad,
            "error": self.error,
        }


@dataclass
class MuJoCoPlanningScene:
    """A model-and-data-isolated MuJoCo planning copy."""

    model: Any
    data: Any
    profile: Any
    mujoco: Any
    source_qpos_sha256: str
    predicted_state: dict[str, Any] = field(default_factory=dict)
    workspaces: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_root: Path | None = None
    attachment: PayloadAttachment | None = None

    def set_refinement_output_root(self, output_root: Path) -> None:
        self.output_root = Path(output_root)

    def workspace(self, action_instance_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(action_instance_id):
            raise LiveRefinementError(
                "INVALID_ACTION_ID",
                "action instance ID is unsafe for refinement artifacts",
                recoverable_by_problem_revision=False,
            )
        return self.workspaces.setdefault(action_instance_id, {})


class MuJoCoPlanningSceneFactory:
    """Clone live MuJoCo data without mutating or stepping scored execution."""

    def __init__(
        self,
        live_scene: Any,
        *,
        robot_name: str = "google",
        initial_state: Mapping[str, Any] | None = None,
        mujoco_module: Any | None = None,
        profile: Any | None = None,
    ) -> None:
        if not hasattr(live_scene, "model") or not hasattr(live_scene, "data"):
            raise ValueError("live scene must expose model and data")
        if mujoco_module is None:
            import mujoco as mujoco_module
        if profile is None:
            from mujoco_scenes.robot_profiles import manipulation_profile

            profile = manipulation_profile(robot_name)
        self.live_scene = live_scene
        self.mujoco = mujoco_module
        self.profile = profile
        self.initial_state = dict(initial_state or {})

    def __call__(self) -> MuJoCoPlanningScene:
        model = _clone_model(self.mujoco, self.live_scene.model)
        source = self.live_scene.data
        copied = self.mujoco.MjData(model)
        for name in _DATA_ARRAYS:
            source_value = getattr(source, name, None)
            copied_value = getattr(copied, name, None)
            if source_value is None or copied_value is None:
                continue
            copied_value[...] = source_value
        if hasattr(source, "time") and hasattr(copied, "time"):
            copied.time = source.time
        self.mujoco.mj_forward(model, copied)
        qpos = np.asarray(copied.qpos, dtype=np.float64)
        scene = MuJoCoPlanningScene(
            model=model,
            data=copied,
            profile=self.profile,
            mujoco=self.mujoco,
            source_qpos_sha256=sha256_bytes(qpos.tobytes()),
            predicted_state=dict(self.initial_state),
        )
        held = scene.predicted_state.get("held_entity")
        if isinstance(held, str) and held:
            scene.attachment = _create_attachment(scene, held, scene.data)
        return scene


class MuJoCoPlanningStateAdapter(PlanningSceneStateAdapter):
    def snapshot(self, planning_scene: MuJoCoPlanningScene) -> Mapping[str, Any]:
        return dict(planning_scene.predicted_state)

    def apply_predicted_transition(
        self,
        planning_scene: MuJoCoPlanningScene,
        action: SymbolicAction,
        predicted_terminal_state: Mapping[str, Any],
    ) -> None:
        workspace = planning_scene.workspace(action.action_instance_id)
        operator = _operator(action)
        contact_qpos = workspace.get("contact_arm_qpos")
        terminal_qpos = workspace.get("terminal_arm_qpos")
        if operator == "pick-from" and contact_qpos is not None:
            addresses = _arm_qpos_addresses(planning_scene)
            planning_scene.data.qpos[addresses] = np.asarray(contact_qpos, dtype=float)
            _forward(planning_scene, planning_scene.data)
            entity = workspace["entities"][0].entity_name
            planning_scene.attachment = _create_attachment(
                planning_scene, entity, planning_scene.data
            )
        if terminal_qpos is not None:
            addresses = _arm_qpos_addresses(planning_scene)
            planning_scene.data.qpos[addresses] = np.asarray(terminal_qpos, dtype=float)
            planning_scene.data.qvel[:] = 0.0
            _forward(planning_scene, planning_scene.data)
            if planning_scene.attachment is not None:
                _sync_attachment(planning_scene, planning_scene.data)
        if operator in {"place-on", "place-in", "insert"}:
            pose = workspace.get("predicted_object_pose")
            if planning_scene.attachment is None or pose is None:
                raise LiveRefinementError(
                    "INVALID_PREDICTED_RELEASE",
                    f"{operator} requires a held payload and a predicted release pose",
                )
            _set_free_body_pose(
                planning_scene,
                planning_scene.data,
                planning_scene.attachment,
                np.asarray(pose["position_m"], dtype=float),
                np.asarray(pose["rotation"], dtype=float),
            )
            planning_scene.attachment = None
            _forward(planning_scene, planning_scene.data)
        if operator == "open-storage":
            entity = workspace["entities"][0].entity_name
            joint_name, position = _open_articulation(planning_scene, entity)
            workspace["opened_joint_name"] = joint_name
            workspace["opened_joint_position"] = position
        planning_scene.predicted_state = dict(predicted_terminal_state)


class MuJoCoGeometryKernel:
    """Real MuJoCo geometry, generic IK, and collision operations."""

    def __init__(
        self,
        *,
        thresholds: RefinementThresholds | None = None,
        ik_factory: Callable[..., Any] | None = None,
        collision_checker_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.thresholds = thresholds or RefinementThresholds()
        self._ik_factory = ik_factory
        self._collision_checker_factory = collision_checker_factory

    def resolve_entities(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> tuple[EntityGeometry, ...]:
        scene.mujoco.mj_forward(scene.model, scene.data)
        resolved = tuple(
            self._entity_geometry(scene, name)
            for name in context.projection.resolved_entities
        )
        scene.workspace(context.action.action_instance_id)["entities"] = resolved
        return resolved

    def generate_candidates(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> tuple[GeometricCandidate, ...]:
        workspace = scene.workspace(context.action.action_instance_id)
        entities = _required_workspace(workspace, "entities")
        primary = entities[0]
        operator = _operator(context.action)
        target = self._target_position(scene, operator, entities)
        positions = (
            self._named_grasp_positions(scene, primary.entity_name)
            if operator in {"open-storage", "pick-from"}
            else ()
        )
        source = "NAMED_GRASP_SITE" if positions else "BODY_AABB"
        if not positions:
            positions = (target,)
        rotations = self._candidate_rotations(scene, operator, entities)
        candidates = tuple(
            GeometricCandidate(
                candidate_id=f"{context.action.action_instance_id}_g{index:02d}",
                target_position_m=_vector_tuple(position),
                approach_position_m=_vector_tuple(
                    self._approach_position(
                        operator, np.asarray(position, dtype=float), entities
                    )
                ),
                target_rotation=_matrix_tuple(rotation),
                source=source,
                skill_parameters=self._skill_parameters(
                    scene, operator, entities, np.asarray(position), rotation
                ),
            )
            for index, (position, rotation) in enumerate(
                (
                    item
                    for position in positions
                    for item in ((position, rot) for rot in rotations)
                )
            )
        )
        if not candidates:
            raise LiveRefinementError(
                "NO_GRASP_CANDIDATES", "no geometric candidates were generated"
            )
        workspace["candidates"] = candidates
        return candidates

    def solve_ik(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> tuple[IKEvaluation, tuple[IKEvaluation, ...]]:
        workspace = scene.workspace(context.action.action_instance_id)
        candidates = _required_workspace(workspace, "candidates")
        ik_factory = self._ik_factory
        if ik_factory is None:
            from mujoco_scenes.generic_manipulation import ProfiledIK

            ik_factory = ProfiledIK
        solver = ik_factory(scene.model, scene.data, scene.profile)
        seed = np.asarray(scene.data.qpos[_arm_qpos_addresses(scene)], dtype=float)
        evaluations: list[IKEvaluation] = []
        for candidate in candidates:
            try:
                rotation = np.asarray(candidate.target_rotation, dtype=float)
                approach_qpos, approach_error, approach_angle = solver.solve(
                    np.asarray(candidate.approach_position_m), seed, rotation
                )
                target_qpos, target_error, target_angle = solver.solve(
                    np.asarray(candidate.target_position_m), approach_qpos, rotation
                )
                reachable = bool(
                    approach_error <= self.thresholds.position_tolerance_m
                    and target_error <= self.thresholds.position_tolerance_m
                    and approach_angle <= self.thresholds.angle_tolerance_rad
                    and target_angle <= self.thresholds.angle_tolerance_rad
                )
                evaluation = IKEvaluation(
                    candidate_id=candidate.candidate_id,
                    reachable=reachable,
                    approach_qpos=tuple(float(value) for value in approach_qpos),
                    target_qpos=tuple(float(value) for value in target_qpos),
                    approach_position_error_m=float(approach_error),
                    target_position_error_m=float(target_error),
                    approach_angle_error_rad=float(approach_angle),
                    target_angle_error_rad=float(target_angle),
                )
            except Exception as error:
                evaluation = IKEvaluation(
                    candidate.candidate_id,
                    False,
                    (),
                    (),
                    None,
                    None,
                    None,
                    None,
                    f"{type(error).__name__}: {error}",
                )
            evaluations.append(evaluation)
        reachable = [item for item in evaluations if item.reachable]
        if not reachable:
            finite_errors = [
                item.target_position_error_m
                for item in evaluations
                if item.target_position_error_m is not None
                and math.isfinite(item.target_position_error_m)
            ]
            evidence = {"candidate_count": float(len(evaluations))}
            if finite_errors:
                evidence["best_position_error_m"] = min(finite_errors)
            raise LiveRefinementError(
                "NO_IK_SOLUTION",
                "no grasp candidate satisfies the IK tolerances",
                numeric_evidence=evidence,
            )
        chosen = min(
            reachable,
            key=lambda item: (
                item.target_position_error_m,
                item.target_angle_error_rad,
                item.candidate_id,
            ),
        )
        workspace["ik_evaluations"] = tuple(evaluations)
        workspace["chosen_ik"] = chosen
        return chosen, tuple(evaluations)

    def plan_trajectory(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> np.ndarray:
        workspace = scene.workspace(context.action.action_instance_id)
        chosen = _required_workspace(workspace, "chosen_ik")
        start = np.asarray(scene.data.qpos[_arm_qpos_addresses(scene)], dtype=float)
        evaluations = _required_workspace(workspace, "ik_evaluations")
        trajectories: dict[str, np.ndarray] = {}
        contact_indices: dict[str, int] = {}
        for evaluation in evaluations:
            if not evaluation.reachable:
                continue
            approach = np.asarray(evaluation.approach_qpos, dtype=float)
            target = np.asarray(evaluation.target_qpos, dtype=float)
            first = _interpolate(
                start, approach, self.thresholds.joint_interpolation_step_rad
            )
            second = _interpolate(
                approach, target, self.thresholds.joint_interpolation_step_rad
            )[1:]
            contact_index = len(first) + len(second) - 1
            parts = (first, second)
            if _operator(context.action) in _BLACK_BOX_SKILLS or _operator(
                context.action
            ) in {"pick-from", "place-on", "place-in", "insert"}:
                retreat = _interpolate(
                    target, approach, self.thresholds.joint_interpolation_step_rad
                )[1:]
                parts = (first, second, retreat)
            candidate_trajectory = np.concatenate(parts, axis=0)
            if not np.all(np.isfinite(candidate_trajectory)):
                continue
            trajectories[evaluation.candidate_id] = candidate_trajectory
            contact_indices[evaluation.candidate_id] = contact_index
        if chosen.candidate_id not in trajectories:
            raise LiveRefinementError(
                "INVALID_TRAJECTORY", "chosen IK candidate has no finite trajectory"
            )
        trajectory = trajectories[chosen.candidate_id]
        operator = _operator(context.action)
        terminal = (
            np.asarray(chosen.approach_qpos, dtype=float)
            if operator in _BLACK_BOX_SKILLS
            or operator in {"pick-from", "place-on", "place-in", "insert"}
            else np.asarray(chosen.target_qpos, dtype=float)
        )
        workspace["trajectories"] = trajectories
        workspace["contact_indices"] = contact_indices
        workspace["trajectory"] = trajectory
        workspace["contact_arm_qpos"] = np.asarray(chosen.target_qpos, dtype=float)
        workspace["terminal_arm_qpos"] = terminal.copy()
        return trajectory

    def check_collision(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> tuple[bool, str | None, int, str]:
        workspace = scene.workspace(context.action.action_instance_id)
        trajectories = _required_workspace(workspace, "trajectories")
        entities = _required_workspace(workspace, "entities")
        evaluations = _required_workspace(workspace, "ik_evaluations")
        checker_factory = self._collision_checker_factory
        if checker_factory is None:
            from mujoco_scenes.generic_manipulation import (
                RobotConfigurationCollisionChecker,
            )

            checker_factory = RobotConfigurationCollisionChecker
        checker = checker_factory(scene.model, scene.data, scene.profile)
        # Only the manipulated body may be contacted. Destination/support
        # bodies remain collision obstacles for the robot itself.
        allowed = _body_descendants(scene.model, entities[0].body_id)
        ordered = sorted(
            (item for item in evaluations if item.reachable),
            key=lambda item: (
                item.target_position_error_m,
                item.target_angle_error_rad,
                item.candidate_id,
            ),
        )
        checked_segments = 0
        last_reason: str | None = None
        for evaluation in ordered:
            trajectory = trajectories.get(evaluation.candidate_id)
            if trajectory is None:
                continue
            candidate_valid = True
            for index in range(1, len(trajectory)):
                checked_segments += 1
                valid, reason = checker.segment_valid(
                    trajectory[index - 1],
                    trajectory[index],
                    allowed_environment_bodies=allowed,
                    resolution=self.thresholds.collision_resolution_rad,
                )
                if not valid:
                    candidate_valid = False
                    last_reason = reason
                    break
            if candidate_valid and scene.attachment is not None:
                valid, reason, samples, pair = self._payload_path_valid(
                    scene,
                    trajectory,
                    context,
                    _required_workspace(workspace, "contact_indices")[
                        evaluation.candidate_id
                    ],
                )
                checked_segments += samples
                if not valid:
                    candidate_valid = False
                    last_reason = reason
                    workspace["payload_collision_pair"] = pair
            if candidate_valid:
                workspace["chosen_ik"] = evaluation
                workspace["trajectory"] = trajectory
                workspace["contact_arm_qpos"] = np.asarray(
                    evaluation.target_qpos, dtype=float
                )
                workspace["terminal_arm_qpos"] = np.asarray(
                    evaluation.approach_qpos
                    if _operator(context.action) in _BLACK_BOX_SKILLS
                    or _operator(context.action)
                    in {"pick-from", "place-on", "place-in", "insert"}
                    else evaluation.target_qpos,
                    dtype=float,
                )
                workspace["collision_free"] = True
                return True, None, checked_segments, evaluation.candidate_id
        raise LiveRefinementError(
            "PATH_COLLISION",
            last_reason or "collision checker rejected every reachable candidate",
            numeric_evidence={
                "candidates_checked": float(len(ordered)),
                "segments_checked": float(checked_segments),
            },
            collision_pair=workspace.get("payload_collision_pair"),
        )

    def check_skill_envelope(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> Mapping[str, Any]:
        workspace = scene.workspace(context.action.action_instance_id)
        chosen = _required_workspace(workspace, "chosen_ik")
        collision_free = bool(_required_workspace(workspace, "collision_free"))
        operator = _operator(context.action)
        black_box = operator in _BLACK_BOX_SKILLS
        if black_box and context.interaction_mode != "BLACK_BOX_SKILL_ENVELOPE":
            raise LiveRefinementError(
                "INVALID_SKILL_MODE",
                f"{operator} must be checked as a black-box controller envelope",
                recoverable_by_problem_revision=False,
            )
        if black_box and len(context.projection.resolved_entities) < 2:
            raise LiveRefinementError(
                "INCOMPLETE_SKILL_ENVELOPE",
                f"{operator} has fewer than two resolved interaction entities",
            )
        if black_box and (
            scene.attachment is None
            or scene.attachment.entity_name != context.projection.resolved_entities[0]
            or context.predicted_state.get("held_entity")
            != context.projection.resolved_entities[0]
        ):
            raise LiveRefinementError(
                "SKILL_PAYLOAD_NOT_HELD",
                f"{operator} requires its projected source to be attached",
            )
        start_valid = bool(chosen.reachable and collision_free)
        end_valid = bool(
            chosen.target_position_error_m <= self.thresholds.position_tolerance_m
            and chosen.target_angle_error_rad <= self.thresholds.angle_tolerance_rad
        )
        candidate = next(
            item
            for item in _required_workspace(workspace, "candidates")
            if item.candidate_id == chosen.candidate_id
        )
        metrics = dict(candidate.skill_parameters)
        constraints: dict[str, bool] = {}
        if operator == "pour":
            constraints = {
                "relative_pose": metrics["horizontal_offset_m"]
                <= metrics["target_opening_radius_m"] * 0.8,
                "reachable_tilt": metrics["tilt_rad"]
                >= self.thresholds.pour_min_tilt_rad,
                "clearance": metrics["vertical_clearance_m"]
                >= self.thresholds.skill_clearance_m,
                "approach_and_return_collision_free": collision_free,
            }
        elif operator == "stir":
            constraints = {
                "axis_alignment": metrics["axis_alignment_rad"]
                <= self.thresholds.stir_axis_tolerance_rad,
                "insertion_depth": metrics["insertion_depth_m"]
                >= self.thresholds.stir_min_depth_m,
                "radius_envelope": metrics["radial_offset_m"]
                + metrics["utensil_radius_m"]
                <= metrics["vessel_radius_m"] * self.thresholds.stir_radius_fraction,
                "approach_and_return_collision_free": collision_free,
            }
        elif operator == "drive":
            constraints = {
                "axis_alignment": metrics["axis_alignment_rad"]
                <= self.thresholds.drive_axis_tolerance_rad,
                "insertion_depth": self.thresholds.drive_min_depth_m
                <= metrics["insertion_depth_m"]
                <= self.thresholds.drive_max_depth_m,
                "orientation": metrics["orientation_error_rad"]
                <= self.thresholds.angle_tolerance_rad,
                "approach_and_return_collision_free": collision_free,
            }
        geometry_valid = all(constraints.values())
        if not start_valid or not end_valid or not geometry_valid:
            failed = sorted(name for name, valid in constraints.items() if not valid)
            raise LiveRefinementError(
                "SKILL_ENVELOPE_FAILED",
                f"{operator} geometric envelope failed: {', '.join(failed) or 'IK'}",
                numeric_evidence=metrics,
            )
        result = {
            "operator": operator,
            "mode": context.interaction_mode,
            "controller_invoked": False,
            "start_envelope_valid": start_valid,
            "end_envelope_valid": end_valid,
            "position_error_m": chosen.target_position_error_m,
            "angle_error_rad": chosen.target_angle_error_rad,
            "metrics": metrics,
            "constraints": constraints,
        }
        workspace["skill_envelope"] = result
        return result

    def predict_transition(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> Mapping[str, Any]:
        workspace = scene.workspace(context.action.action_instance_id)
        _required_workspace(workspace, "skill_envelope")
        state = dict(context.predicted_state)
        completed = list(state.get("completed_action_instance_ids", ()))
        completed.append(context.action.action_instance_id)
        state["completed_action_instance_ids"] = completed
        state["last_operator"] = _operator(context.action)
        state["last_arguments"] = list(context.action.arguments)
        relations = list(state.get("predicted_relations", ()))
        relations.append(
            {
                "operator": _operator(context.action),
                "arguments": list(context.action.arguments),
            }
        )
        state["predicted_relations"] = relations
        operator = _operator(context.action)
        if operator == "pick-from":
            if scene.attachment is not None:
                raise LiveRefinementError(
                    "PICK_WITH_HELD_PAYLOAD", "pick requires an empty gripper"
                )
            _create_attachment(
                scene, context.projection.resolved_entities[0], scene.data
            )
            state["held_entity"] = context.projection.resolved_entities[0]
        elif operator in {"place-on", "place-in", "insert"}:
            if (
                scene.attachment is None
                or scene.attachment.entity_name
                != context.projection.resolved_entities[0]
            ):
                raise LiveRefinementError(
                    "INVALID_PREDICTED_RELEASE",
                    f"{operator} requires its projected payload to be attached",
                )
            state["held_entity"] = None
            entities = _required_workspace(workspace, "entities")
            primary, target = entities[0], entities[1]
            position = np.asarray(target.centroid_m, dtype=float)
            if operator == "place-on":
                position[2] = (
                    target.aabb_max_m[2]
                    + 0.5 * primary.dimensions_m[2]
                    + self.thresholds.placement_clearance_m
                )
            else:
                position[2] = target.centroid_m[2]
            chosen = _required_workspace(workspace, "chosen_ik")
            candidate = next(
                item
                for item in _required_workspace(workspace, "candidates")
                if item.candidate_id == chosen.candidate_id
            )
            rotation = np.asarray(candidate.target_rotation, dtype=float) @ np.asarray(
                scene.attachment.gripper_relative_rotation, dtype=float
            )
            workspace["predicted_object_pose"] = {
                "position_m": position.tolist(),
                "rotation": rotation.tolist(),
            }
        elif operator == "open-storage":
            _resolve_articulation_joint(scene, context.projection.resolved_entities[0])
            opened = list(state.get("opened_entities", ()))
            if context.projection.resolved_entities[0] not in opened:
                opened.append(context.projection.resolved_entities[0])
            state["opened_entities"] = opened
        return state

    def _entity_geometry(
        self, scene: MuJoCoPlanningScene, entity_name: str
    ) -> EntityGeometry:
        body_id = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, entity_name
        )
        if body_id < 0:
            raise LiveRefinementError(
                "MISSING_SCENE_ENTITY", f"MuJoCo body is missing: {entity_name!r}"
            )
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        body_ids = _body_descendants(scene.model, int(body_id))
        for geom_id in range(scene.model.ngeom):
            if int(scene.model.geom_bodyid[geom_id]) not in body_ids:
                continue
            rotation = np.asarray(scene.data.geom_xmat[geom_id]).reshape(3, 3)
            local_center = np.asarray(scene.model.geom_aabb[geom_id, :3])
            local_half = np.asarray(scene.model.geom_aabb[geom_id, 3:])
            center = np.asarray(scene.data.geom_xpos[geom_id]) + rotation @ local_center
            half = np.abs(rotation) @ local_half
            lower = np.minimum(lower, center - half)
            upper = np.maximum(upper, center + half)
        if not np.all(np.isfinite(lower)):
            raise LiveRefinementError(
                "MISSING_ENTITY_GEOMETRY", f"body has no geometry: {entity_name!r}"
            )
        centroid = (lower + upper) / 2.0
        body_rotation = np.asarray(scene.data.xmat[body_id], dtype=float).reshape(3, 3)
        dimensions = upper - lower
        principal_local = np.zeros(3)
        principal_local[int(np.argmax(dimensions))] = 1.0
        principal_world = body_rotation @ principal_local
        return EntityGeometry(
            entity_name,
            int(body_id),
            _vector_tuple(centroid),
            _vector_tuple(lower),
            _vector_tuple(upper),
            _matrix_tuple(body_rotation),
            _vector_tuple(dimensions),
            _vector_tuple(principal_world),
        )

    def _named_grasp_positions(
        self, scene: MuJoCoPlanningScene, entity_name: str
    ) -> tuple[np.ndarray, ...]:
        normalized = entity_name.lower().replace("-", "_")
        entity_body = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, entity_name
        )
        body_ids = _body_descendants(scene.model, int(entity_body))
        site_body_ids = getattr(scene.model, "site_bodyid", None)
        matches = []
        for site_id in range(scene.model.nsite):
            name = (
                scene.mujoco.mj_id2name(
                    scene.model, scene.mujoco.mjtObj.mjOBJ_SITE, site_id
                )
                or ""
            )
            lowered = name.lower().replace("-", "_")
            belongs_to_entity = (
                site_body_ids is not None and int(site_body_ids[site_id]) in body_ids
            )
            if (belongs_to_entity or normalized in lowered) and (
                "grasp" in lowered or "handle" in lowered
            ):
                matches.append((name, np.asarray(scene.data.site_xpos[site_id]).copy()))
        matches.sort(key=lambda item: item[0])
        return tuple(position for _, position in matches)

    def _target_position(
        self,
        scene: MuJoCoPlanningScene,
        operator: str,
        entities: Sequence[EntityGeometry],
    ) -> np.ndarray:
        primary = entities[0]
        if operator in {"place-on", "place-in", "insert", "pour", "stir", "drive"}:
            if len(entities) < 2:
                raise LiveRefinementError(
                    "MISSING_TARGET_ENTITY", f"{operator} requires a target entity"
                )
            target = entities[1]
            target_point = np.asarray(target.centroid_m, dtype=float)
            if operator == "stir":
                depth = max(
                    self.thresholds.stir_min_depth_m * 1.5,
                    0.35 * target.dimensions_m[2],
                )
                target_point[2] = target.aabb_max_m[2] - depth
                return target_point
            if operator == "drive":
                axis = _normalized(np.asarray(target.principal_axis_world))
                return target_point + axis * min(
                    0.02, 0.4 * float(max(target.dimensions_m))
                )
            target_point[2] = (
                target.aabb_max_m[2]
                + 0.5 * (primary.aabb_max_m[2] - primary.aabb_min_m[2])
                + self.thresholds.placement_clearance_m
            )
            return target_point
        point = np.asarray(primary.centroid_m, dtype=float)
        point[2] = primary.aabb_max_m[2]
        return point

    def _candidate_rotations(
        self,
        scene: MuJoCoPlanningScene,
        operator: str,
        entities: Sequence[EntityGeometry],
    ) -> tuple[np.ndarray, ...]:
        base = np.asarray(scene.profile.top_down_rotation, dtype=float)
        if operator not in _BLACK_BOX_SKILLS:
            return _yaw_variants(base)
        source_axis = np.asarray(entities[0].principal_axis_world, dtype=float)
        target_axis = (
            np.asarray(entities[1].principal_axis_world, dtype=float)
            if operator == "drive"
            else np.asarray(entities[1].body_rotation, dtype=float)[:, 2]
        )
        target_axis = _normalized(target_axis)
        if operator == "pour":
            perpendicular = _orthogonal(target_axis)
            desired_axis = (
                math.cos(self.thresholds.pour_tilt_rad) * target_axis
                + math.sin(self.thresholds.pour_tilt_rad) * perpendicular
            )
        else:
            desired_axis = target_axis
        grip_rotation = _gripper_rotation(scene, scene.data)
        relative_axis = grip_rotation.T @ _normalized(source_axis)
        aligned = _rotation_mapping(relative_axis, desired_axis)
        return tuple(
            _axis_angle_rotation(desired_axis, yaw) @ aligned
            for yaw in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
        )

    def _approach_position(
        self,
        operator: str,
        target: np.ndarray,
        entities: Sequence[EntityGeometry],
    ) -> np.ndarray:
        if operator == "drive" and len(entities) >= 2:
            axis = _normalized(np.asarray(entities[1].principal_axis_world))
            return target - axis * self.thresholds.approach_clearance_m
        return target + np.asarray((0.0, 0.0, self.thresholds.approach_clearance_m))

    def _skill_parameters(
        self,
        scene: MuJoCoPlanningScene,
        operator: str,
        entities: Sequence[EntityGeometry],
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> Mapping[str, float]:
        if operator not in _BLACK_BOX_SKILLS or len(entities) < 2:
            return {}
        source, target = entities[0], entities[1]
        source_axis = _normalized(np.asarray(source.principal_axis_world))
        current_grip = _gripper_rotation(scene, scene.data)
        planned_axis = _normalized(rotation @ (current_grip.T @ source_axis))
        target_center = np.asarray(target.centroid_m)
        target_axis = (
            _normalized(np.asarray(target.principal_axis_world))
            if operator == "drive"
            else _normalized(np.asarray(target.body_rotation)[:, 2])
        )
        target_dimensions = np.asarray(target.dimensions_m)
        source_dimensions = np.asarray(source.dimensions_m)
        if operator == "pour":
            horizontal = position - target_center
            horizontal -= np.dot(horizontal, target_axis) * target_axis
            source_half_height = 0.5 * float(np.max(source_dimensions))
            clearance = float(position[2] - source_half_height - target.aabb_max_m[2])
            return {
                "horizontal_offset_m": float(np.linalg.norm(horizontal)),
                "target_opening_radius_m": 0.5 * float(min(target_dimensions[:2])),
                "tilt_rad": _undirected_angle(planned_axis, target_axis),
                "vertical_clearance_m": clearance,
            }
        if operator == "stir":
            radial = position - target_center
            radial -= np.dot(radial, target_axis) * target_axis
            depth = max(0.0, float(target.aabb_max_m[2] - position[2]))
            return {
                "axis_alignment_rad": _undirected_angle(planned_axis, target_axis),
                "insertion_depth_m": depth,
                "radial_offset_m": float(np.linalg.norm(radial)),
                "utensil_radius_m": 0.5 * float(np.partition(source_dimensions, 1)[1]),
                "vessel_radius_m": 0.5 * float(min(target_dimensions[:2])),
            }
        delta = position - target_center
        insertion = abs(float(np.dot(delta, target_axis)))
        alignment = _undirected_angle(planned_axis, target_axis)
        return {
            "axis_alignment_rad": alignment,
            "orientation_error_rad": alignment,
            "insertion_depth_m": insertion,
        }

    def _payload_path_valid(
        self,
        scene: MuJoCoPlanningScene,
        trajectory: np.ndarray,
        context: RefinementStageContext,
        contact_index: int,
    ) -> tuple[bool, str | None, int, tuple[str, str] | None]:
        attachment = scene.attachment
        if attachment is None:
            return True, None, 0, None
        scratch = scene.mujoco.MjData(scene.model)
        _copy_data_state(scene.data, scratch)
        addresses = _arm_qpos_addresses(scene)
        robot_prefix = scene.profile.gripper_body.split(":", 1)[0] + ":"
        payload_geoms = [
            geom_id
            for geom_id in range(scene.model.ngeom)
            if int(scene.model.geom_bodyid[geom_id])
            in _body_descendants(scene.model, attachment.body_id)
            and _geom_enabled(scene.model, geom_id)
        ]
        payload_body_ids = _body_descendants(scene.model, attachment.body_id)
        allowed_at_contact: set[int] = set()
        for item in scene.workspace(context.action.action_instance_id)["entities"][1:]:
            allowed_at_contact.update(_body_descendants(scene.model, item.body_id))
        operator = _operator(context.action)
        active_trajectory = (
            trajectory[: contact_index + 1]
            if operator in {"place-on", "place-in", "insert"}
            else trajectory
        )
        for sample_index, joints in enumerate(active_trajectory):
            scratch.qpos[addresses] = joints
            _forward(scene, scratch)
            _sync_attachment(scene, scratch, attachment)
            for payload_geom in payload_geoms:
                for other_geom in range(scene.model.ngeom):
                    other_body = int(scene.model.geom_bodyid[other_geom])
                    if other_body in payload_body_ids or not _geom_enabled(
                        scene.model, other_geom
                    ):
                        continue
                    other_name = _body_name(scene, other_body)
                    if other_name.startswith(robot_prefix):
                        continue
                    intentional_destination_contact = (
                        operator in {"place-on", "place-in", "insert"}
                        and other_body in allowed_at_contact
                    )
                    if intentional_destination_contact:
                        continue
                    distance = float(
                        scene.mujoco.mj_geomDistance(
                            scene.model,
                            scratch,
                            payload_geom,
                            other_geom,
                            self.thresholds.payload_collision_tolerance_m,
                            None,
                        )
                    )
                    if distance < -self.thresholds.payload_collision_tolerance_m:
                        payload_name = attachment.entity_name
                        return (
                            False,
                            f"carried payload collision {payload_name} / {other_name}",
                            sample_index + 1,
                            (payload_name, other_name),
                        )
        return True, None, len(active_trajectory), None


class _ConcreteStageBackend(RefinementStageBackend):
    stage: RefinementStage

    def __init__(self, kernel: MuJoCoGeometryKernel) -> None:
        self.kernel = kernel

    def _trace(
        self,
        scene: MuJoCoPlanningScene,
        context: RefinementStageContext,
        payload: Mapping[str, Any],
    ) -> str | None:
        if scene.output_root is None:
            return None
        path = (
            scene.output_root
            / "refinement_stages"
            / context.action.action_instance_id
            / f"{self.stage.value.lower()}.json"
        )
        atomic_write_json(path, payload)
        return str(path)

    def _failure(
        self,
        scene: MuJoCoPlanningScene,
        context: RefinementStageContext,
        error: LiveRefinementError,
    ) -> RefinementStageOutcome:
        trace = self._trace(
            scene,
            context,
            {
                "success": False,
                "reason_code": error.reason_code,
                "summary": str(error),
                "numeric_evidence": error.numeric_evidence,
            },
        )
        return RefinementStageOutcome(
            stage=self.stage,
            success=False,
            resolved_entities=context.projection.resolved_entities,
            reason_code=error.reason_code,
            summary=str(error),
            involved_entities=context.projection.resolved_entities,
            collision_pair=error.collision_pair,
            numeric_evidence=error.numeric_evidence,
            backend_trace_artifact=trace,
            recoverable_by_problem_revision=error.recoverable_by_problem_revision,
        )


class EntityResolutionBackend(_ConcreteStageBackend):
    stage = RefinementStage.ENTITY_RESOLUTION

    def refine_stage(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> RefinementStageOutcome:
        try:
            entities = self.kernel.resolve_entities(scene, context)
        except LiveRefinementError as error:
            return self._failure(scene, context, error)
        trace = self._trace(
            scene,
            context,
            {"success": True, "entities": [item.to_dict() for item in entities]},
        )
        return RefinementStageOutcome(
            self.stage,
            True,
            resolved_entities=tuple(item.entity_name for item in entities),
            numeric_evidence={"entity_count": float(len(entities))},
            backend_trace_artifact=trace,
        )


class GraspCandidateBackend(_ConcreteStageBackend):
    stage = RefinementStage.GRASP_GENERATION

    def refine_stage(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> RefinementStageOutcome:
        try:
            candidates = self.kernel.generate_candidates(scene, context)
        except LiveRefinementError as error:
            return self._failure(scene, context, error)
        trace = self._trace(
            scene,
            context,
            {"success": True, "candidates": [item.to_dict() for item in candidates]},
        )
        return RefinementStageOutcome(
            self.stage,
            True,
            candidate_artifacts=(trace,) if trace else (),
            resolved_entities=context.projection.resolved_entities,
            chosen_grasp=candidates[0].to_dict(),
            chosen_target_pose={"position_m": list(candidates[0].target_position_m)},
            numeric_evidence={"candidate_count": float(len(candidates))},
            backend_trace_artifact=trace,
        )


class IKBackend(_ConcreteStageBackend):
    stage = RefinementStage.IK

    def refine_stage(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> RefinementStageOutcome:
        try:
            chosen, evaluations = self.kernel.solve_ik(scene, context)
        except LiveRefinementError as error:
            return self._failure(scene, context, error)
        trace = self._trace(
            scene,
            context,
            {
                "success": True,
                "chosen": chosen.to_dict(),
                "evaluations": [item.to_dict() for item in evaluations],
            },
        )
        return RefinementStageOutcome(
            self.stage,
            True,
            candidate_artifacts=(trace,) if trace else (),
            resolved_entities=context.projection.resolved_entities,
            reachable=True,
            numeric_evidence={
                "position_error_m": chosen.target_position_error_m,
                "angle_error_rad": chosen.target_angle_error_rad,
            },
            backend_trace_artifact=trace,
        )


class TrajectoryBackend(_ConcreteStageBackend):
    stage = RefinementStage.TRAJECTORY

    def refine_stage(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> RefinementStageOutcome:
        try:
            trajectory = self.kernel.plan_trajectory(scene, context)
        except LiveRefinementError as error:
            return self._failure(scene, context, error)
        trace = self._trace(
            scene, context, {"success": True, "joint_waypoints": trajectory.tolist()}
        )
        return RefinementStageOutcome(
            self.stage,
            True,
            trajectory_artifacts=(trace,) if trace else (),
            resolved_entities=context.projection.resolved_entities,
            numeric_evidence={"waypoint_count": float(len(trajectory))},
            backend_trace_artifact=trace,
        )


class CollisionBackend(_ConcreteStageBackend):
    stage = RefinementStage.COLLISION

    def refine_stage(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> RefinementStageOutcome:
        try:
            collision_free, reason, segments, candidate_id = (
                self.kernel.check_collision(scene, context)
            )
        except LiveRefinementError as error:
            return self._failure(scene, context, error)
        trace = self._trace(
            scene,
            context,
            {
                "success": True,
                "collision_free": collision_free,
                "reason": reason,
                "segments_checked": segments,
                "chosen_candidate_id": candidate_id,
            },
        )
        return RefinementStageOutcome(
            self.stage,
            True,
            resolved_entities=context.projection.resolved_entities,
            collision_free=collision_free,
            numeric_evidence={"segments_checked": float(segments)},
            backend_trace_artifact=trace,
        )


class SkillEnvelopeBackend(_ConcreteStageBackend):
    stage = RefinementStage.SKILL_ENVELOPE

    def refine_stage(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> RefinementStageOutcome:
        try:
            envelope = self.kernel.check_skill_envelope(scene, context)
        except LiveRefinementError as error:
            return self._failure(scene, context, error)
        trace = self._trace(scene, context, {"success": True, "envelope": envelope})
        return RefinementStageOutcome(
            self.stage,
            True,
            resolved_entities=context.projection.resolved_entities,
            reachable=True,
            collision_free=True,
            numeric_evidence={
                "position_error_m": float(envelope["position_error_m"]),
                "angle_error_rad": float(envelope["angle_error_rad"]),
            },
            backend_trace_artifact=trace,
        )


class PredictedStateTransitionBackend(_ConcreteStageBackend):
    stage = RefinementStage.STATE_TRANSITION

    def refine_stage(
        self, scene: MuJoCoPlanningScene, context: RefinementStageContext
    ) -> RefinementStageOutcome:
        try:
            terminal = self.kernel.predict_transition(scene, context)
        except LiveRefinementError as error:
            return self._failure(scene, context, error)
        trace = self._trace(
            scene, context, {"success": True, "predicted_terminal_state": terminal}
        )
        return RefinementStageOutcome(
            self.stage,
            True,
            resolved_entities=context.projection.resolved_entities,
            predicted_terminal_state=terminal,
            backend_trace_artifact=trace,
        )


@dataclass(frozen=True)
class LiveRefinementRuntime:
    refiner: MuJoCoSequenceRefiner
    planning_scene_factory: MuJoCoPlanningSceneFactory


def create_live_refinement_runtime(
    live_scene: Any,
    *,
    robot_name: str = "google",
    initial_state: Mapping[str, Any] | None = None,
    thresholds: RefinementThresholds | None = None,
    mujoco_module: Any | None = None,
    profile: Any | None = None,
    kernel: MuJoCoGeometryKernel | None = None,
) -> LiveRefinementRuntime:
    factory = MuJoCoPlanningSceneFactory(
        live_scene,
        robot_name=robot_name,
        initial_state=initial_state,
        mujoco_module=mujoco_module,
        profile=profile,
    )
    geometry = kernel or MuJoCoGeometryKernel(thresholds=thresholds)
    backends: dict[RefinementStage, RefinementStageBackend] = {
        RefinementStage.ENTITY_RESOLUTION: EntityResolutionBackend(geometry),
        RefinementStage.GRASP_GENERATION: GraspCandidateBackend(geometry),
        RefinementStage.IK: IKBackend(geometry),
        RefinementStage.TRAJECTORY: TrajectoryBackend(geometry),
        RefinementStage.COLLISION: CollisionBackend(geometry),
        RefinementStage.SKILL_ENVELOPE: SkillEnvelopeBackend(geometry),
        RefinementStage.STATE_TRANSITION: PredictedStateTransitionBackend(geometry),
    }
    return LiveRefinementRuntime(
        refiner=MuJoCoSequenceRefiner(
            stage_backends=backends,
            state_adapter=MuJoCoPlanningStateAdapter(),
        ),
        planning_scene_factory=factory,
    )


def _required_workspace(workspace: Mapping[str, Any], key: str) -> Any:
    if key not in workspace:
        raise LiveRefinementError(
            "MISSING_STAGE_INPUT",
            f"refinement stage input {key!r} is unavailable",
            recoverable_by_problem_revision=False,
        )
    return workspace[key]


def _arm_qpos_addresses(scene: MuJoCoPlanningScene) -> np.ndarray:
    joint_ids = np.asarray(
        [
            scene.mujoco.mj_name2id(scene.model, scene.mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in scene.profile.arm_joints
        ],
        dtype=int,
    )
    if np.any(joint_ids < 0):
        raise LiveRefinementError(
            "ROBOT_PROFILE_MISMATCH",
            "planning model does not contain every configured arm joint",
            recoverable_by_problem_revision=False,
        )
    return np.asarray(scene.model.jnt_qposadr[joint_ids], dtype=int)


def _interpolate(start: np.ndarray, goal: np.ndarray, resolution: float) -> np.ndarray:
    if start.shape != goal.shape or start.ndim != 1:
        raise LiveRefinementError(
            "INVALID_TRAJECTORY", "joint vectors have incompatible dimensions"
        )
    steps = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / resolution)))
    return np.linspace(start, goal, steps + 1)


def _yaw_variants(base: np.ndarray) -> tuple[np.ndarray, ...]:
    rotations = []
    for yaw in (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0):
        cosine, sine = math.cos(yaw), math.sin(yaw)
        around_z = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        rotations.append(around_z @ base)
    return tuple(rotations)


def _operator(action: SymbolicAction) -> str:
    return action.operator.lower().replace("_", "-")


def _vector_tuple(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise LiveRefinementError("INVALID_GEOMETRY", "expected a finite 3-D vector")
    return tuple(float(item) for item in array)


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise LiveRefinementError("INVALID_GEOMETRY", "expected a finite 3x3 rotation")
    return tuple(tuple(float(item) for item in row) for row in array)


def _clone_model(mujoco_module: Any, source_model: Any) -> Any:
    """Clone a compiled model through MuJoCo's lossless MJB representation."""
    with tempfile.TemporaryDirectory(prefix="vilain-planning-model-") as directory:
        path = Path(directory) / "planning_scene.mjb"
        mujoco_module.mj_saveModel(source_model, str(path), None)
        cloned = mujoco_module.MjModel.from_binary_path(str(path))
    if cloned is source_model:
        raise LiveRefinementError(
            "MODEL_CLONE_FAILED",
            "MuJoCo returned the live model instead of an independent clone",
            recoverable_by_problem_revision=False,
        )
    return cloned


def _copy_data_state(source: Any, target: Any) -> None:
    for name in _DATA_ARRAYS:
        source_value = getattr(source, name, None)
        target_value = getattr(target, name, None)
        if source_value is not None and target_value is not None:
            target_value[...] = source_value
    if hasattr(source, "time") and hasattr(target, "time"):
        target.time = source.time


def _forward(scene: MuJoCoPlanningScene, data: Any) -> None:
    scene.mujoco.mj_forward(scene.model, data)


def _gripper_body_id(scene: MuJoCoPlanningScene) -> int:
    body_id = scene.mujoco.mj_name2id(
        scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, scene.profile.gripper_body
    )
    if body_id < 0:
        raise LiveRefinementError(
            "ROBOT_PROFILE_MISMATCH",
            f"planning model lacks gripper body {scene.profile.gripper_body!r}",
            recoverable_by_problem_revision=False,
        )
    return int(body_id)


def _gripper_rotation(scene: MuJoCoPlanningScene, data: Any) -> np.ndarray:
    return np.asarray(data.xmat[_gripper_body_id(scene)], dtype=float).reshape(3, 3)


def _create_attachment(
    scene: MuJoCoPlanningScene, entity_name: str, data: Any
) -> PayloadAttachment:
    body_id = scene.mujoco.mj_name2id(
        scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, entity_name
    )
    if body_id < 0:
        raise LiveRefinementError(
            "MISSING_SCENE_ENTITY", f"missing held body {entity_name!r}"
        )
    joint_count = int(scene.model.body_jntnum[body_id])
    if joint_count != 1:
        raise LiveRefinementError(
            "PAYLOAD_NOT_FREE",
            f"held body {entity_name!r} must own exactly one free joint",
        )
    joint_id = int(scene.model.body_jntadr[body_id])
    free_type = int(scene.mujoco.mjtJoint.mjJNT_FREE)
    if int(scene.model.jnt_type[joint_id]) != free_type:
        raise LiveRefinementError(
            "PAYLOAD_NOT_FREE", f"held body {entity_name!r} is not free-moving"
        )
    qpos_address = int(scene.model.jnt_qposadr[joint_id])
    gripper_id = _gripper_body_id(scene)
    gripper_position = np.asarray(data.xpos[gripper_id], dtype=float)
    gripper_rotation = np.asarray(data.xmat[gripper_id], dtype=float).reshape(3, 3)
    body_position = np.asarray(data.xpos[body_id], dtype=float)
    body_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    return PayloadAttachment(
        entity_name=entity_name,
        body_id=int(body_id),
        free_joint_id=joint_id,
        qpos_address=qpos_address,
        gripper_body_id=gripper_id,
        gripper_relative_position_m=_vector_tuple(
            gripper_rotation.T @ (body_position - gripper_position)
        ),
        gripper_relative_rotation=_matrix_tuple(gripper_rotation.T @ body_rotation),
    )


def _sync_attachment(
    scene: MuJoCoPlanningScene,
    data: Any,
    attachment: PayloadAttachment | None = None,
) -> None:
    attachment = attachment or scene.attachment
    if attachment is None:
        return
    grip_position = np.asarray(data.xpos[attachment.gripper_body_id], dtype=float)
    grip_rotation = np.asarray(
        data.xmat[attachment.gripper_body_id], dtype=float
    ).reshape(3, 3)
    relative_position = np.asarray(attachment.gripper_relative_position_m)
    relative_rotation = np.asarray(attachment.gripper_relative_rotation)
    _set_free_body_pose(
        scene,
        data,
        attachment,
        grip_position + grip_rotation @ relative_position,
        grip_rotation @ relative_rotation,
    )
    _forward(scene, data)


def _set_free_body_pose(
    scene: MuJoCoPlanningScene,
    data: Any,
    attachment: PayloadAttachment,
    position: np.ndarray,
    rotation: np.ndarray,
) -> None:
    address = attachment.qpos_address
    data.qpos[address : address + 3] = position
    data.qpos[address + 3 : address + 7] = _matrix_to_quaternion(rotation)


def _open_articulation(
    scene: MuJoCoPlanningScene, entity_name: str
) -> tuple[str, float]:
    joint_id = _resolve_articulation_joint(scene, entity_name)
    address = int(scene.model.jnt_qposadr[joint_id])
    position = float(scene.model.jnt_range[joint_id, 1])
    scene.data.qpos[address] = position
    if hasattr(scene.data, "qvel"):
        dof_address = int(scene.model.jnt_dofadr[joint_id])
        scene.data.qvel[dof_address] = 0.0
    _forward(scene, scene.data)
    name = (
        scene.mujoco.mj_id2name(scene.model, scene.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        or f"joint_{joint_id}"
    )
    return name, position


def _resolve_articulation_joint(scene: MuJoCoPlanningScene, entity_name: str) -> int:
    body_id = scene.mujoco.mj_name2id(
        scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, entity_name
    )
    if body_id < 0:
        raise LiveRefinementError(
            "MISSING_STORAGE_ARTICULATION",
            f"storage body {entity_name!r} is missing",
        )
    movable_types = {
        int(scene.mujoco.mjtJoint.mjJNT_HINGE),
        int(scene.mujoco.mjtJoint.mjJNT_SLIDE),
    }
    candidates: list[int] = []
    parent_ids = getattr(scene.model, "body_parentid", None)
    for candidate_body in range(len(scene.model.body_jntnum)):
        descendant = candidate_body == body_id
        ancestor = candidate_body
        while not descendant and parent_ids is not None and ancestor > 0:
            ancestor = int(parent_ids[ancestor])
            descendant = ancestor == body_id
        if not descendant:
            continue
        first = int(scene.model.body_jntadr[candidate_body])
        count = int(scene.model.body_jntnum[candidate_body])
        candidates.extend(
            joint_id
            for joint_id in range(first, first + count)
            if int(scene.model.jnt_type[joint_id]) in movable_types
        )
    if len(candidates) != 1:
        raise LiveRefinementError(
            "MISSING_STORAGE_ARTICULATION"
            if not candidates
            else "AMBIGUOUS_STORAGE_ARTICULATION",
            f"storage body {entity_name!r} has {len(candidates)} hinge/slide articulations",
        )
    return candidates[0]


def _geom_enabled(model: Any, geom_id: int) -> bool:
    contype = getattr(model, "geom_contype", None)
    affinity = getattr(model, "geom_conaffinity", None)
    if contype is None or affinity is None:
        return True
    return bool(contype[geom_id] or affinity[geom_id])


def _body_name(scene: MuJoCoPlanningScene, body_id: int) -> str:
    return (
        scene.mujoco.mj_id2name(scene.model, scene.mujoco.mjtObj.mjOBJ_BODY, body_id)
        or f"body_{body_id}"
    )


def _body_descendants(model: Any, root_body_id: int) -> frozenset[int]:
    parent_ids = getattr(model, "body_parentid", None)
    if parent_ids is None:
        return frozenset((root_body_id,))
    descendants = {root_body_id}
    for body_id in range(len(parent_ids)):
        ancestor = body_id
        while ancestor > 0 and ancestor not in descendants:
            ancestor = int(parent_ids[ancestor])
        if ancestor in descendants:
            descendants.add(body_id)
    return frozenset(descendants)


def _normalized(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise LiveRefinementError("INVALID_GEOMETRY", "zero-length geometric axis")
    return vector / norm


def _orthogonal(axis: np.ndarray) -> np.ndarray:
    basis = np.asarray((1.0, 0.0, 0.0))
    if abs(float(np.dot(axis, basis))) > 0.9:
        basis = np.asarray((0.0, 1.0, 0.0))
    return _normalized(basis - np.dot(basis, axis) * axis)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalized(axis)
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    return (
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def _rotation_mapping(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source, target = _normalized(source), _normalized(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3)
    if dot < -1.0 + 1e-10:
        return _axis_angle_rotation(_orthogonal(source), math.pi)
    cross = np.cross(source, target)
    return _axis_angle_rotation(cross, math.acos(dot))


def _undirected_angle(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(_normalized(first), _normalized(second))))
    return math.acos(float(np.clip(dot, -1.0, 1.0)))


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_a, next_b = (index + 1) % 3, (index + 2) % 3
        scale = (
            math.sqrt(
                max(
                    0.0,
                    1.0
                    + matrix[index, index]
                    - matrix[next_a, next_a]
                    - matrix[next_b, next_b],
                )
            )
            * 2.0
        )
        quat = np.zeros(4)
        quat[index + 1] = 0.25 * scale
        quat[0] = (matrix[next_b, next_a] - matrix[next_a, next_b]) / scale
        quat[next_a + 1] = (matrix[next_a, index] + matrix[index, next_a]) / scale
        quat[next_b + 1] = (matrix[next_b, index] + matrix[index, next_b]) / scale
    return quat / np.linalg.norm(quat)
