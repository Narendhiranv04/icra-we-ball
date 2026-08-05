import math
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from mujoco_scenes.living_room_dusting import DUST_ROWS, TVDustExecutor
from mujoco_scenes.living_room_cameras import SOFA_CAMERAS, TOP_CAMERAS
from mujoco_scenes.living_room_drawer import MediaConsoleDrawerExecutor
from mujoco_scenes.living_room_manipulation import (
    CALIBRATED_LIVING_ROOM_OBJECTS,
    LIVING_ROOM_PICK_SPECS,
    PLACE_SITE_BY_OBJECT,
    LivingRoomManipulationExecutor,
)
from mujoco_scenes.living_room_navigation import (
    LivingRoomLayout,
    LivingRoomNavigationExecutor,
)
from mujoco_scenes.living_room_remote import RemoteTVExecutor
from mujoco_scenes.mobile_motion import MuJoCoBaseCollisionChecker
from mujoco_scenes.living_room_scene import (
    PICKABLE_OBJECTS,
    ROBOT_GOOGLE,
    ROBOT_NONE,
    TV_CELL_COUNT,
    LivingRoomScene,
    build_living_room_xml,
)


def _named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise AssertionError(f"Missing {object_type.name}: {name}")
    return object_id


class LivingRoomSceneTests(unittest.TestCase):
    """Headless structural and reset checks for the living-room model."""

    @classmethod
    def setUpClass(cls):
        # The robot-free model is enough for structural checks.  Keep one
        # Google scene for the entire class so Menagerie assets compile once.
        cls.robot_free_model = mujoco.MjModel.from_xml_string(
            build_living_room_xml(ROBOT_NONE)
        )
        cls.scene = LivingRoomScene(ROBOT_GOOGLE)

    def setUp(self):
        # These tests do not require physics settling; resetting directly also
        # makes the expected body/joint poses exact and keeps the suite quick.
        self.scene.reset(settle_steps=0)

    def test_scene_compiles_with_no_robot(self):
        model = self.robot_free_model
        self.assertEqual(model.nu, 2)
        for side in ("left", "right"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"media_console_{side}_drawer_actuator",
                ),
                0,
            )
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "coffee_table"
            ),
            0,
        )
        self.assertEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "google:base_link"
            ),
            -1,
        )

    def test_robot_camera_render_is_rejected_without_robot(self):
        robot_free = SimpleNamespace(has_robot=False)
        with self.assertRaisesRegex(ValueError, "requires Google Robot"):
            LivingRoomScene.render_frame(
                robot_free, camera="top_front_camera"
            )

    def test_scene_compiles_with_google_robot(self):
        model = self.scene.model
        self.assertGreater(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "google:base_link"
            ),
            -1,
        )
        self.assertGreater(model.nu, 0)
        self.assertEqual(model.neq, len(PICKABLE_OBJECTS))
        self.assertEqual(len(SOFA_CAMERAS), 2)
        self.assertEqual(len(TOP_CAMERAS), 5)
        for camera_name in SOFA_CAMERAS + TOP_CAMERAS:
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                ),
                0,
            )

    def test_lost_remote_starts_inside_real_sofa_clearance(self):
        scene = LivingRoomScene(ROBOT_GOOGLE, scenario="lost_remote")
        remote = scene.data.xpos[scene.body_id("remote_control")]
        self.assertLess(float(remote[1]), -1.22)
        self.assertLess(float(remote[2]), 0.05)
        base_id = _named_id(
            scene.model, mujoco.mjtObj.mjOBJ_GEOM, "couch_south_base"
        )
        base_bottom = (
            scene.model.geom_pos[base_id, 2]
            - scene.model.geom_size[base_id, 2]
        )
        self.assertGreater(float(base_bottom), 0.20)

    def test_coffee_table_is_fixed_and_unactuated(self):
        model = self.robot_free_model
        table_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "coffee_table"
        )
        self.assertEqual(int(model.body_jntnum[table_id]), 0)
        np.testing.assert_allclose(model.body_pos[table_id], (0.0, -0.35, 0.0))

    def test_room_has_north_and_west_walls_and_static_table(self):
        model = self.robot_free_model
        geom_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in range(model.ngeom)
        }

        wall_geoms = {
            name
            for name in geom_names
            if name and name.endswith("feature_wall_panel")
        }
        self.assertEqual(
            wall_geoms,
            {
                "media_feature_wall_panel",
                "west_feature_wall_panel",
            },
        )
        for wall_name in wall_geoms:
            wall_id = _named_id(
                model, mujoco.mjtObj.mjOBJ_GEOM, wall_name
            )
            self.assertAlmostEqual(
                float(model.geom_pos[wall_id, 2] - model.geom_size[wall_id, 2]),
                0.0,
            )
        self.assertFalse(any(name and "cushion" in name for name in geom_names))
        self.assertFalse(any(name and "rail" in name for name in geom_names))
        self.assertIn("coffee_table_top", geom_names)
        self.assertNotIn("coffee_table_edge_grip", geom_names)
        self.assertEqual(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                "coffee_table_south_edge_grasp",
            ),
            -1,
        )

    def test_rug_is_larger_and_centered_on_coffee_table(self):
        model = self.robot_free_model
        rug_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "living_room_rug_field"
        )
        table_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "coffee_table"
        )
        table_top_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "coffee_table_top"
        )
        np.testing.assert_allclose(
            model.geom_pos[rug_id, :2], model.body_pos[table_id, :2]
        )
        self.assertGreater(
            float(model.geom_size[rug_id, 0]),
            float(model.geom_size[table_top_id, 0]),
        )
        self.assertGreater(
            float(model.geom_size[rug_id, 1]),
            float(model.geom_size[table_top_id, 1]),
        )

    def test_couch_sections_form_one_exact_seam_l(self):
        model = self.robot_free_model

        def interval(name, axis):
            geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            center = float(model.geom_pos[geom_id, axis])
            half_size = float(model.geom_size[geom_id, axis])
            return center - half_size, center + half_size

        corner_x = interval("couch_corner_base", 0)
        corner_y = interval("couch_corner_base", 1)
        west_y = interval("couch_west_base", 1)
        south_x = interval("couch_south_base", 0)
        self.assertAlmostEqual(corner_y[1], west_y[0])
        self.assertAlmostEqual(corner_x[1], south_x[0])

        corner_seat_x = interval("couch_corner_seat", 0)
        corner_seat_y = interval("couch_corner_seat", 1)
        west_seat_y = interval("couch_west_seat", 1)
        south_seat_x = interval("couch_south_seat", 0)
        self.assertAlmostEqual(corner_seat_y[1], west_seat_y[0])
        self.assertAlmostEqual(corner_seat_x[1], south_seat_x[0])

        west_back_x = interval("couch_west_back", 0)
        south_back_x = interval("couch_south_back", 0)
        self.assertAlmostEqual(west_back_x[1], south_back_x[0])

    def test_media_storage_geometry_and_drawer_interface_exist(self):
        model = self.robot_free_model
        for side in ("left", "right"):
            drawer_joint = _named_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"media_console_{side}_drawer_slide",
            )
            self.assertEqual(
                int(model.jnt_type[drawer_joint]),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            np.testing.assert_allclose(
                model.jnt_range[drawer_joint], (0.0, 0.27)
            )
        for site in (
            "media_shelf_book_place",
            "media_shelf_controller_place",
            "left_drawer_place_controller",
            "drawer_place_controller",
            "game_controller_grasp",
        ):
            _named_id(model, mujoco.mjtObj.mjOBJ_SITE, site)

    def test_drawer_executor_opens_closes_and_settles(self):
        scene = self.scene
        for side in ("left", "right"):
            drawer = MediaConsoleDrawerExecutor(scene, side)
            for action, expected_open in (("open", True), ("close", False)):
                drawer.request(action, "drawer")
                for _ in range(2000):
                    drawer.update()
                    mujoco.mj_step(scene.model, scene.data)
                    if not drawer.busy:
                        break
                self.assertIsNone(drawer.failure)
                self.assertEqual(drawer.mode, "complete")
                self.assertEqual(drawer.is_open, expected_open)

    def test_tv_is_wall_mounted_without_console_clipping_stand(self):
        model = self.robot_free_model
        _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "tv_wall_mount")
        _named_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "media_feature_wall_panel"
        )
        self.assertEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "tv_stand_post"
            ),
            -1,
        )
        self.assertEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "tv_stand_foot"
            ),
            -1,
        )

    def test_dust_alpha_and_coverage_are_restored_by_reset(self):
        model = self.scene.model
        visual_ids = [
            _named_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"dust_cell_visual_{index}",
            )
            for index in range(TV_CELL_COUNT)
        ]
        initial_alphas = model.geom_rgba[visual_ids, 3].copy()
        self.assertTrue(np.all(initial_alphas > 0.0))
        self.assertEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "dust_screen_film"
            ),
            -1,
        )

        self.scene.mark_tv_cell_clean(2)
        self.scene.mark_tv_cell_clean(11)
        self.assertEqual(self.scene.cleaned_cells, {2, 11})
        self.assertAlmostEqual(self.scene.dust_coverage, 2 / TV_CELL_COUNT)
        np.testing.assert_allclose(
            self.scene.dust_cell_opacities, initial_alphas
        )
        for _ in range(1000):
            self.scene.update_visual_effects()
        faded = self.scene.dust_cell_opacities
        self.assertAlmostEqual(float(faded[2]), 0.0, places=4)
        self.assertAlmostEqual(float(faded[11]), 0.0, places=4)
        untouched = [index for index in range(TV_CELL_COUNT) if index not in {2, 11}]
        np.testing.assert_allclose(faded[untouched], initial_alphas[untouched])

        for index in range(TV_CELL_COUNT):
            self.scene.mark_tv_cell_clean(index)
        for _ in range(1500):
            self.scene.update_visual_effects()
        np.testing.assert_allclose(
            self.scene.dust_cell_opacities,
            np.zeros(TV_CELL_COUNT),
            atol=1e-4,
        )

        self.scene.reset(settle_steps=0)

        self.assertEqual(self.scene.cleaned_cells, set())
        self.assertEqual(self.scene.dust_coverage, 0.0)
        np.testing.assert_allclose(
            self.scene.dust_cell_opacities, initial_alphas
        )

    def test_tabletop_pickables_use_a_spaced_staggered_layout(self):
        positions = {
            name: self.scene.data.xpos[self.scene.body_id(name), :2].copy()
            for name in (
                "remote_control",
                "living_room_mug",
                "hardback_book",
                "game_controller",
            )
        }
        unique_y = {round(float(position[1]), 2) for position in positions.values()}
        self.assertGreaterEqual(len(unique_y), 3)
        names = tuple(positions)
        for first_index, first in enumerate(names):
            for second in names[first_index + 1 :]:
                self.assertGreater(
                    float(np.linalg.norm(positions[first] - positions[second])),
                    0.20,
                    msg=f"{first} and {second} are still crowded",
                )

    def test_tv_power_visual_state_is_reset(self):
        scene = self.scene
        material_id = _named_id(
            scene.model, mujoco.mjtObj.mjOBJ_MATERIAL, "tv_screen"
        )
        initial_rgba = scene.model.mat_rgba[material_id].copy()
        initial_emission = float(scene.model.mat_emission[material_id])
        led_id = _named_id(
            scene.model, mujoco.mjtObj.mjOBJ_GEOM, "tv_power_led"
        )
        initial_led = scene.model.geom_rgba[led_id].copy()

        scene.set_tv_power(True)
        self.assertTrue(scene.tv_power_on)
        self.assertGreater(float(scene.model.mat_emission[material_id]), 0.0)
        self.assertGreater(
            float(scene.model.geom_rgba[led_id, 1]),
            float(scene.model.geom_rgba[led_id, 0]),
        )
        scene.reset(settle_steps=0)

        self.assertFalse(scene.tv_power_on)
        np.testing.assert_allclose(scene.model.mat_rgba[material_id], initial_rgba)
        self.assertAlmostEqual(
            float(scene.model.mat_emission[material_id]), initial_emission
        )
        np.testing.assert_allclose(scene.model.geom_rgba[led_id], initial_led)

    def test_reset_restores_all_grasp_equalities(self):
        model = self.scene.model
        data = self.scene.data
        initial_eq_data = model.eq_data.copy()
        initial_eq_solref = model.eq_solref.copy()
        initial_active = model.eq_active0.copy()
        self.assertGreater(model.neq, 0)
        self.assertFalse(np.any(initial_active))

        model.eq_data[:] = 0.375
        model.eq_solref[:] = (0.123, 0.456)
        data.eq_active[:] = 1

        self.scene.reset(settle_steps=0)

        np.testing.assert_allclose(model.eq_data, initial_eq_data)
        np.testing.assert_allclose(model.eq_solref, initial_eq_solref)
        np.testing.assert_array_equal(data.eq_active, initial_active)

    def test_table_side_destinations_use_fixed_table_pose(self):
        scene = self.scene
        layout = LivingRoomLayout()
        table_x, table_y, table_yaw = scene.table_pose
        self.assertAlmostEqual(table_x, 0.0)
        self.assertAlmostEqual(table_y, -0.35)
        self.assertAlmostEqual(table_yaw, 0.0)
        table_yaw = scene.table_pose[2]
        cosine = math.cos(table_yaw)
        sine = math.sin(table_yaw)
        home = layout.destination_pose(scene, "home")
        south = layout.destination_pose(scene, "table_south")
        north = layout.destination_pose(scene, "table_north")
        east = layout.destination_pose(scene, "table_east")
        west = layout.destination_pose(scene, "table_west")

        self.assertAlmostEqual(home.x, table_x + 0.90 * sine)
        self.assertAlmostEqual(home.y, table_y - 0.90 * cosine)
        self.assertAlmostEqual(home.yaw, table_yaw)
        self.assertAlmostEqual(south.x, table_x + 0.68 * sine)
        self.assertAlmostEqual(south.y, table_y - 0.68 * cosine)
        self.assertAlmostEqual(south.yaw, table_yaw)
        self.assertAlmostEqual(north.x, table_x - 0.68 * sine)
        self.assertAlmostEqual(north.y, table_y + 0.68 * cosine)
        self.assertAlmostEqual(
            north.yaw,
            math.atan2(
                math.sin(table_yaw + math.pi),
                math.cos(table_yaw + math.pi),
            ),
        )
        self.assertAlmostEqual(east.x, table_x + 0.82 * cosine)
        self.assertAlmostEqual(east.y, table_y + 0.82 * sine)
        self.assertAlmostEqual(east.yaw, table_yaw + math.pi / 2)
        self.assertAlmostEqual(west.x, table_x - 0.82 * cosine)
        self.assertAlmostEqual(west.y, table_y - 0.82 * sine)
        self.assertAlmostEqual(west.yaw, table_yaw - math.pi / 2)

    def test_fixed_furniture_interaction_stances_are_collision_free(self):
        scene = self.scene
        model = scene.model
        navigation = LivingRoomNavigationExecutor(scene)
        checker = MuJoCoBaseCollisionChecker(
            model, scene.data, navigation.profile
        )
        for destination in (
            "home",
            "table_south",
            "table_north",
            "table_east",
            "table_west",
            "bookshelf",
            "drawer",
            "drawer_left",
            "drawer_right",
        ):
            pose = navigation.layout.destination_pose(scene, destination)
            self.assertTrue(
                checker.is_pose_valid(pose.x, pose.y, pose.yaw),
                f"{destination} collides with the fixed layout",
            )

    def test_coasters_settle_on_top_without_embedding(self):
        model = self.robot_free_model
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)

        table_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "coffee_table"
        )
        for name in ("coaster_left", "coaster_right"):
            coaster_id = _named_id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            distances = [
                float(contact.dist)
                for contact in data.contact
                if {
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                }
                == {table_id, coaster_id}
            ]
            self.assertTrue(distances, f"{name} lost table support")
            self.assertGreaterEqual(min(distances), -0.0005)

    def test_mug_place_target_is_the_live_right_coaster(self):
        model = self.robot_free_model
        data = mujoco.MjData(model)
        self.assertEqual(
            PLACE_SITE_BY_OBJECT["living_room_mug"],
            "coaster_right_mug_place",
        )
        pair_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_PAIR, "mug_coaster_support"
        )
        self.assertGreaterEqual(pair_id, 0)

        coaster_joint = _named_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "coaster_right_free"
        )
        coaster_qpos = int(model.jnt_qposadr[coaster_joint])
        site_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "coaster_right_mug_place"
        )
        mujoco.mj_forward(model, data)
        original = data.site_xpos[site_id].copy()
        data.qpos[coaster_qpos] += 0.08
        data.qpos[coaster_qpos + 1] -= 0.06
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.site_xpos[site_id, :2], original[:2] + (0.08, -0.06)
        )

    def test_manipulation_advertises_all_calibrated_objects(self):
        manipulation = LivingRoomManipulationExecutor(
            self.scene, calibration_mode=True
        )
        advertised = set(manipulation.all_pick_specs)
        calibrated = set(manipulation.calibrated_objects)
        candidates = advertised - calibrated

        self.assertEqual(advertised, set(PICKABLE_OBJECTS))
        self.assertEqual(
            calibrated, set(CALIBRATED_LIVING_ROOM_OBJECTS)
        )
        self.assertEqual(candidates, set())
        self.assertIn("hardback_book", calibrated)
        self.assertIn("game_controller", calibrated)
        self.assertIn("rigid_duster", calibrated)
        self.assertEqual(advertised, set(LIVING_ROOM_PICK_SPECS))
        self.assertTrue(
            all(LIVING_ROOM_PICK_SPECS[name].label for name in advertised)
        )

    def test_storage_objects_have_reversible_symbolic_destinations(self):
        manipulation = LivingRoomManipulationExecutor(self.scene)
        self.assertEqual(
            manipulation.required_pick_location("hardback_book"), "home"
        )
        self.assertEqual(
            manipulation.required_pick_location("game_controller"), "home"
        )

        manipulation.object_locations["hardback_book"] = "bookshelf"
        manipulation.object_locations["game_controller"] = "drawer"
        self.assertEqual(
            manipulation.required_pick_location("hardback_book"), "bookshelf"
        )
        self.assertEqual(
            manipulation.required_pick_location("game_controller"), "drawer"
        )

    def test_remote_executor_rejects_missing_remote_and_wrong_location(self):
        manipulation = LivingRoomManipulationExecutor(self.scene)
        remote = RemoteTVExecutor(self.scene, manipulation)
        with self.assertRaisesRegex(RuntimeError, "Pick up the TV remote"):
            remote.request_toggle("tv")

        manipulation.executor = SimpleNamespace(
            held_object="remote_control",
            close_target=0.1,
        )
        with self.assertRaisesRegex(RuntimeError, "Move to TV"):
            remote.request_toggle("home")

    def test_dust_rows_cover_each_tv_cell_exactly_once(self):
        flattened = tuple(cell for row in DUST_ROWS for cell in row)

        self.assertEqual(len(DUST_ROWS), 3)
        self.assertTrue(all(len(row) == 5 for row in DUST_ROWS))
        self.assertEqual(len(flattened), TV_CELL_COUNT)
        self.assertEqual(len(set(flattened)), TV_CELL_COUNT)
        self.assertEqual(set(flattened), set(range(TV_CELL_COUNT)))

    def test_dust_executor_requires_google_robot(self):
        robot_free_scene = SimpleNamespace(robot_name=ROBOT_NONE)
        with self.assertRaisesRegex(ValueError, "requires Google Robot"):
            TVDustExecutor(robot_free_scene, SimpleNamespace())

    def test_dust_executor_starts_idle_after_scene_reset(self):
        manipulation = LivingRoomManipulationExecutor(self.scene)
        dusting = TVDustExecutor(self.scene, manipulation)

        self.assertEqual(dusting.mode, "idle")
        self.assertFalse(dusting.busy)
        self.assertTrue(dusting.navigation_safe)
        self.assertIsNone(dusting.failure)
        self.assertFalse(
            bool(self.scene.data.eq_active[dusting.grasp_equality_id])
        )

    def test_dust_request_rejects_invalid_action_state_before_planning(self):
        manipulation = LivingRoomManipulationExecutor(self.scene)
        dusting = TVDustExecutor(self.scene, manipulation)

        with self.assertRaisesRegex(RuntimeError, "Pick the rigid duster"):
            dusting.request_dust("tv")
        with self.assertRaisesRegex(RuntimeError, "held-object state is stale"):
            dusting.request_dust("tv", held_object="rigid_duster")

        held = SimpleNamespace(
            held_object="rigid_duster",
            navigation_safe=True,
        )
        dusting = TVDustExecutor(self.scene, held)
        with self.assertRaisesRegex(RuntimeError, "Move to TV"):
            dusting.request_dust("home")
        with self.assertRaisesRegex(RuntimeError, "transport weld is not active"):
            dusting.request_dust("tv")

        self.scene.data.eq_active[dusting.grasp_equality_id] = 1
        held.navigation_safe = False
        with self.assertRaisesRegex(RuntimeError, "compact carry"):
            dusting.request_dust("tv")

        # Scene reset must make the same transport precondition false again.
        self.scene.reset(settle_steps=0)
        self.assertFalse(
            bool(self.scene.data.eq_active[dusting.grasp_equality_id])
        )


if __name__ == "__main__":
    unittest.main()
