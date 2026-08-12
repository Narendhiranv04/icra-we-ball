import math

import numpy as np

from mujoco_scenes.manipulation_stance import (
    ManipulationStancePlanner,
    PlanarStance,
    StanceEvaluation,
    base_relative_pose_to_world,
    qpos_to_world_stance,
    world_stance_to_qpos,
    yaw_rotation,
)


def test_base_qpos_world_stance_round_trip():
    qpos = np.array((0.42, 1.025, -math.pi / 2))
    stance = qpos_to_world_stance(qpos, home_y=-0.52)
    np.testing.assert_allclose(
        world_stance_to_qpos(stance, home_y=-0.52), qpos
    )


def test_base_relative_carry_transforms_home_left_right_and_yaw():
    relative = np.array((0.0, 0.47, 0.94))
    orientation = np.eye(3)
    home, home_r = base_relative_pose_to_world(
        PlanarStance(0.0, -0.52, 0.0), relative, orientation
    )
    left, left_r = base_relative_pose_to_world(
        PlanarStance(-1.025, -0.10, -math.pi / 2), relative, orientation
    )
    right, right_r = base_relative_pose_to_world(
        PlanarStance(1.025, -0.10, math.pi / 2), relative, orientation
    )
    np.testing.assert_allclose(home, (0.0, -0.05, 0.94), atol=1e-9)
    np.testing.assert_allclose(left, (-0.555, -0.10, 0.94), atol=1e-9)
    np.testing.assert_allclose(right, (0.555, -0.10, 0.94), atol=1e-9)
    np.testing.assert_allclose(left_r, yaw_rotation(-math.pi / 2))
    np.testing.assert_allclose(right_r, yaw_rotation(math.pi / 2))


def test_stance_planner_is_deterministic_and_rejects_before_accepting():
    planner = ManipulationStancePlanner()
    anchor = PlanarStance(1.0, 2.0, 0.25)

    def evaluate(stance, index):
        return StanceEvaluation(
            stance=stance,
            valid=index == 3,
            collision_clearance_m=0.02 if index == 3 else None,
            ik_residual_m=0.001 if index == 3 else None,
            joint_displacement_rad=0.2 if index == 3 else None,
            candidate_index=index,
            reason=None if index == 3 else "COLLISION_OR_UNREACHABLE",
        )

    first, first_audit = planner.select(anchor, evaluate)
    second, second_audit = planner.select(anchor, evaluate)
    assert first == second
    assert first_audit == second_audit
    assert first is not None and first.candidate_index == 3
    assert len(first_audit) == 4
    assert all(not row.valid for row in first_audit[:3])


def test_stance_planner_reports_no_solution_without_fabrication():
    planner = ManipulationStancePlanner()

    def reject(stance, index):
        return StanceEvaluation(
            stance, False, None, None, None, index, "BASE_PATH_COLLISION"
        )

    selected, audit = planner.select(PlanarStance(0.0, 0.0, 0.0), reject)
    assert selected is None
    assert len(audit) == len(planner.candidates(PlanarStance(0.0, 0.0, 0.0)))
    assert {row.reason for row in audit} == {"BASE_PATH_COLLISION"}
