import math

import mujoco
import numpy as np

from mujoco_scenes.generic_manipulation import GraspPoseCandidate
from mujoco_scenes.kitchen_object_manipulation import (
    StorageGraspCandidateGenerator,
    UtensilGraspCandidateGenerator,
    bilateral_first_contact_metrics,
    physical_contact_target_geoms,
    storage_probe_candidates,
)
from mujoco_scenes.scene_loader import KitchenScene
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


def test_collision_target_filter_excludes_closer_visual_geometries():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary", robot="google"
    )
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, "s1i_wide_shallow_cup"
    )
    names = {
        mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in physical_contact_target_geoms(scene.model, body_id)
    }
    assert "s1i_wide_shallow_cup_visual" not in names
    assert "s1i_wide_shallow_cup_interior_visual" not in names
    assert "s1i_wide_shallow_cup_wall_0" in names
    assert "s1i_wide_shallow_cup_bottom_collision" in names


def test_contact_centred_vessel_candidates_follow_collision_shell_scale():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary", robot="google"
    )
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, "s1i_wide_shallow_cup"
    )
    first = StorageGraspCandidateGenerator.cupboard(
        scene, body_id, np.eye(3), "VESSEL"
    )
    candidate = next(
        row for row in first if "contact_diameter" in row.candidate_id
    )
    first_spacing = np.linalg.norm(
        np.asarray(candidate.predicted_contact_points_world_m[1])
        - np.asarray(candidate.predicted_contact_points_world_m[0])
    )
    wall_ids = [
        geom_id for geom_id in physical_contact_target_geoms(
            scene.model, body_id
        )
        if "wall" in (
            mujoco.mj_id2name(
                scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or ""
        )
    ]
    scene.model.geom_pos[wall_ids, :2] *= 1.25
    mujoco.mj_forward(scene.model, scene.data)
    second = StorageGraspCandidateGenerator.cupboard(
        scene, body_id, np.eye(3), "VESSEL"
    )
    scaled = next(
        row for row in second
        if row.candidate_id == candidate.candidate_id
    )
    second_spacing = np.linalg.norm(
        np.asarray(scaled.predicted_contact_points_world_m[1])
        - np.asarray(scaled.predicted_contact_points_world_m[0])
    )
    assert second_spacing > first_spacing * 1.20
    assert all("wall" in name for name in candidate.predicted_contact_geom_names)


def test_box_bowl_grasp_is_collision_centred_and_probe_is_mirrored():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary", robot="google"
    )
    site_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_SITE, "ab3_deep_bowl_grasp"
    )
    candidates = StorageGraspCandidateGenerator.box(
        scene, site_id, np.eye(3), "BOWL"
    )
    centred = next(
        row for row in candidates
        if row.candidate_id == "box_bowl_diameter_0_yaw+60_z+0.35"
    )
    assert abs(centred.grasp_site_local_position_m[0]) < 0.01
    assert abs(centred.grasp_site_local_position_m[1]) < 0.01
    assert centred.predicted_contact_geom_names == (
        "ab3_deep_bowl_wall_0", "ab3_deep_bowl_wall_6"
    )
    probe_ids = {
        row.candidate_id
        for row in storage_probe_candidates(candidates, "BOX", "BOWL")
    }
    assert {
        "box_bowl_diameter_0_yaw+60_z+0.35",
        "box_bowl_diameter_0_yaw+30_z+0.35",
        "box_bowl_diameter_0_yaw+0_z+0.35",
        "box_bowl_diameter_0_yaw-30_z+0.35",
        "box_bowl_diameter_0_yaw-60_z+0.35",
    } == probe_ids


def test_first_contact_synchrony_accepts_centred_and_rejects_asymmetric_sweep():
    closures = [0.0, 0.4, 0.8, 1.2]
    names = [("left_wall", "right_wall")] * len(closures)
    centred = bilateral_first_contact_metrics(
        closures,
        [(0.02, 0.02), (0.006, 0.006), (0.0002, 0.0003), (-0.001, -0.001)],
        names,
    )
    asymmetric = bilateral_first_contact_metrics(
        closures,
        [(0.02, 0.02), (0.0002, 0.010), (-0.002, 0.006), (-0.004, 0.0002)],
        names,
    )
    unilateral = bilateral_first_contact_metrics(
        closures,
        [(0.02, 0.02), (0.0002, 0.020), (-0.002, 0.015), (-0.004, 0.010)],
        names,
    )
    assert centred["closure_delta"] == 0.0
    assert math.isclose(asymmetric["closure_delta"], 0.8)
    assert asymmetric["maximum_precontact_penetration_m"] >= 0.002
    assert unilateral["right"] is None


def test_cupboard_utensil_probe_uses_horizontal_over_handle_grasp():
    candidates = tuple(
        _candidate(f"cupboard_{approach}_{fraction}pct_z+0.010")
        for approach in ("horizontal_over_handle", "front_vertical_jaws")
        for fraction in (55, 65, 75, 85)
    )
    selected = storage_probe_candidates(candidates, "CUPBOARD", "UTENSIL")
    identifiers = {row.candidate_id for row in selected}
    assert any("horizontal_over_handle" in item for item in identifiers)
    assert not any("front_vertical" in item for item in identifiers)
    assert identifiers == {
        "cupboard_horizontal_over_handle_55pct_z+0.010"
    }


def test_upright_cupboard_utensil_uses_horizontal_inward_grasp():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_primary", robot="google"
    )
    scene.open_container("C2")
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, "s1i_c2_soup_spoon"
    )
    first = UtensilGraspCandidateGenerator.generate(
        scene, body_id, "CUPBOARD"
    )
    assert first
    assert all(
        "horizontal_inward_upright_handle" in row.candidate_id
        for row in first
    )
    # Local +Y is the horizontal jaw-closing axis; local +Z enters C2
    # through its open front in world +Y.
    assert all(
        np.allclose(row.target_rotation_world[:, 1], (1.0, 0.0, 0.0))
        and np.allclose(row.target_rotation_world[:, 2], (0.0, 1.0, 0.0))
        for row in first
    )
    assert all(
        row.approach_offset_world_m == (0.0, -0.09, 0.0)
        for row in first
    )
    assert all(not row.position_first_approach for row in first)


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
