import math

import numpy as np

from mujoco_scenes.generic_manipulation import GraspPoseCandidate
from mujoco_scenes.kitchen_object_manipulation import storage_probe_candidates
from mujoco_scenes.manipulation_stance import (
    BilateralContactPrediction,
    GraspStanceEvaluation,
    ManipulationStancePlanner,
    PlanarStance,
    StanceEvaluation,
    base_relative_pose_to_world,
    qpos_to_world_stance,
    rank_grasp_stance_pairs,
    world_stance_to_qpos,
    yaw_rotation,
)


def _candidate(candidate_id):
    return GraspPoseCandidate(
        candidate_id=candidate_id,
        grasp_site_local_position_m=(0.0, 0.0, 0.0),
        target_rotation_world=np.eye(3),
        approach_clearance_m=0.1,
    )


def test_cupboard_vessel_probe_covers_top_diameter_and_rim_families():
    candidates = tuple(_candidate(item) for item in (
        "cupboard_vessel_top_yaw+30",
        "cupboard_vessel_top_yaw+60",
        "cupboard_vessel_diameter_+0.000_+0.000",
        "cupboard_front_rim_0_jawroll+20_wrist0",
        "cupboard_front_rim_2_jawroll+0_wrist0",
    ))
    selected = storage_probe_candidates(candidates, "CUPBOARD", "VESSEL")
    identifiers = {row.candidate_id for row in selected}
    assert any("vessel_top" in item for item in identifiers)
    assert any("vessel_diameter" in item for item in identifiers)
    assert any("front_rim" in item for item in identifiers)


def test_cupboard_utensil_probe_covers_both_approaches_and_handle_spread():
    candidates = tuple(
        _candidate(f"cupboard_{approach}_{fraction}pct_z+0.010")
        for approach in ("side_horizontal_jaws", "front_vertical_jaws")
        for fraction in (55, 65, 75, 85)
    )
    selected = storage_probe_candidates(candidates, "CUPBOARD", "UTENSIL")
    identifiers = {row.candidate_id for row in selected}
    assert any("side_horizontal" in item for item in identifiers)
    assert any("front_vertical" in item for item in identifiers)
    assert {fraction for fraction in (55, 75, 85)
            if any(f"_{fraction}pct_" in item for item in identifiers)} == {
                55, 75, 85
            }


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


def test_stance_planner_ranks_better_later_stance_over_first_feasible():
    planner = ManipulationStancePlanner()
    anchor = PlanarStance(1.0, 2.0, 0.25)

    def evaluate(stance, index):
        return StanceEvaluation(
            stance=stance,
            valid=index in {3, 5},
            collision_clearance_m=0.02 if index in {3, 5} else None,
            ik_residual_m=0.001 if index in {3, 5} else None,
            joint_displacement_rad=0.2 if index in {3, 5} else None,
            candidate_index=index,
            reason=None if index in {3, 5} else "COLLISION_OR_UNREACHABLE",
            feasible_grasp_families=("TOP_DOWN",) if index == 3 else (
                ("TOP_DOWN", "FRONTAL_DIAMETER") if index == 5 else ()
            ),
            predicted_grasp_score=1.0 if index == 3 else (
                2.0 if index == 5 else 0.0
            ),
        )

    first, first_audit = planner.select(anchor, evaluate)
    second, second_audit = planner.select(anchor, evaluate)
    assert first == second
    assert first_audit == second_audit
    assert first is not None and first.candidate_index == 5
    assert len(first_audit) == len(planner.candidates(anchor))
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


def _pair(index, candidate_id, prediction, *, valid=True):
    return GraspStanceEvaluation(
        stance=PlanarStance(1.0, 2.0, 0.0),
        stance_index=index,
        grasp_candidate_id=candidate_id,
        grasp_family="FRONTAL_DIAMETER",
        carry_valid=valid,
        approach_valid=valid,
        grasp_ik_position_error_m=0.001,
        grasp_ik_angle_error_rad=0.01,
        collision_free=valid,
        minimum_collision_clearance_m=None,
        contact_prediction=prediction,
        base_displacement_m=0.05,
        joint_displacement_rad=0.2,
        valid=valid,
    )


def test_predicted_bilateral_pair_outranks_planning_only_pair_deterministically():
    unilateral = BilateralContactPrediction(
        0.001, 0.030, 0.9, True, 0.08, 0.04, True, False, False
    )
    bilateral = BilateralContactPrediction(
        0.002, 0.002, 0.9, True, 0.08, 0.04, True, True, True
    )
    rows = (_pair(0, "first_feasible", unilateral),
            _pair(1, "later_bilateral", bilateral))
    first = rank_grasp_stance_pairs(rows)
    second = rank_grasp_stance_pairs(rows)
    assert first == second
    assert first[0].grasp_candidate_id == "later_bilateral"


def test_stance_shortlist_is_bounded():
    planner = ManipulationStancePlanner()

    def accept(stance, index):
        return StanceEvaluation(
            stance, True, 0.01, 0.001, 0.2, index,
            feasible_grasp_families=("TOP_DOWN",),
            predicted_grasp_score=float(index),
        )

    shortlist, audit = planner.shortlist(
        PlanarStance(0.0, 0.0, 0.0), accept, maximum=5, candidate_limit=20
    )
    assert len(shortlist) == 5
    assert len(audit) == 20
