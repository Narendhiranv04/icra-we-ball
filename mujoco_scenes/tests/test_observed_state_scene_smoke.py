import json
import tempfile
import unittest
from pathlib import Path

from mujoco_scenes.geometry_checker import GeometryChecker
from mujoco_scenes.observed_state import ObservedStateRun
from mujoco_scenes.scene_loader import CONTAINER_JOINTS, KitchenScene
from mujoco_scenes.sequential_inspection import run_sequential_inspection


class ObservedStateSceneSmokeTests(unittest.TestCase):
    def test_ablation2_repeated_categories_have_distinct_instances(self):
        scene = KitchenScene(
            "S1_ablation2_count_reuse_primary",
            include_robot=False,
        )
        visible = scene.get_visible_object_instances()
        names = [name for name, _kind in visible]
        kinds = [kind for _name, kind in visible]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(kinds.count("cup"), 2)
        self.assertEqual(kinds.count("mixing_bowl"), 2)
        self.assertEqual(kinds.count("spoon"), 1)
        self.assertEqual(kinds.count("fork"), 0)
        self.assertEqual(
            scene.config.container_contents["D2"],
            ["fork"],
        )
        self.assertEqual(
            sorted(name for name, kind in visible if kind == "cup"),
            ["cup", "cup_2"],
        )

    def test_closed_s1_initial_witness_uses_only_visible_evidence(self):
        scene = KitchenScene("S1_coffee_missing_mug", include_robot=False)
        with tempfile.TemporaryDirectory() as directory:
            session = ObservedStateRun.create_for_scene(
                scene,
                runs_root=directory,
                run_id="closed_witness",
                task_requirements="configs/s1_find_open_receptacle.yaml",
            )
            _cloud_run, stage_dir = session.observe_scene(
                scene,
                stage_label="initial",
                width=160,
                height=120,
            )
            self.assertIn(
                session.latest_witness["status"],
                {"COMPLETE", "INCOMPLETE", "INDETERMINATE"},
            )
            self.assertEqual(
                session.latest_witness["inference_basis"], "GEOMETRY_ONLY"
            )
            self.assertNotIn(
                "bowl", str(session.latest_witness).lower()
            )
            if session.latest_witness["status"] == "COMPLETE":
                selected = session.latest_witness["selected_witness"][
                    "open_receptacle"
                ]
                for object_id in selected:
                    record = session.registry["objects"][object_id]
                    self.assertEqual(
                        record["last_property_source_region"], "INITIAL"
                    )
                    self.assertEqual(
                        record["last_property_update_stage"], 0
                    )
                    self.assertIn(
                        "/evidence/", record["measurement_cloud_path"]
                    )
            self.assertTrue((stage_dir / "witness.json").exists())
            self.assertTrue((session.run_dir / "latest_witness.json").exists())

    def test_existing_open_all_and_point_cloud_export_remain_functional(self):
        scene = KitchenScene("S1_coffee_missing_mug", include_robot=False)
        scene.set_all_containers_open_snapshot()
        self.assertEqual(
            scene.state.opened_containers,
            set(CONTAINER_JOINTS),
        )
        self.assertTrue(
            all(scene.state.container_open_state.values())
        )
        with tempfile.TemporaryDirectory() as directory:
            run = GeometryChecker(scene, width=160, height=120).run(directory)
            output = Path(directory)
            self.assertTrue((output / "all_visible_objects.ply").exists())
            self.assertTrue((output / "manifest.json").exists())
        self.assertIn("mug", {cloud.object_kind for cloud in run.clouds.values()})
        self.assertGreater(run.total_points, 0)

    def test_counterexample_geometry_comes_from_actual_five_view_evidence(self):
        scene = KitchenScene(
            "S1_joint_stir_counterexamples", include_robot=False
        )
        with tempfile.TemporaryDirectory() as directory:
            session = run_sequential_inspection(
                scene,
                sequence=("D1", "D2"),
                runs_root=directory,
                run_id="counterexample_geometry",
                width=640,
                height=480,
                task_requirements="configs/stir_contents_joint.yaml",
                grounding_mode="geometry-only",
                stop_on_complete=False,
            )
            stages = sorted((session.run_dir / "stages").iterdir())
            self.assertEqual(len(stages), 3)
            registry_counts = [
                len(json.loads((stage / "properties.json").read_text())["objects"])
                for stage in stages
            ]
            self.assertEqual(registry_counts, [2, 3, 4])

            initial = json.loads(
                (stages[0] / "grounding_mode_comparison.json").read_text()
            )["modes"]["geometry-only"]
            self.assertEqual(
                initial["selected_witness"]["mixing_tool"],
                ["object_0002"],
            )

            def relation_status(stage, tool_id, relation_name):
                comparison = json.loads(
                    (
                        stage / "grounding_mode_comparison.json"
                    ).read_text()
                )["modes"]["geometry-only"]
                assignment = next(
                    item
                    for item in comparison["assignment_evaluations"]
                    if item.get("selected_objects", {}).get(
                        "mixing_container"
                    )
                    == ["object_0001"]
                    and item.get("selected_objects", {}).get(
                        "mixing_tool"
                    )
                    == [tool_id]
                )
                return next(
                    check["status"]
                    for check in assignment["relation_checks"]
                    if check["relation"] == relation_name
                )

            self.assertEqual(
                relation_status(stages[0], "object_0002", "INSERTABLE_IN"),
                "TRUE",
            )
            self.assertEqual(
                relation_status(stages[0], "object_0002", "REACHES_BOTTOM"),
                "TRUE",
            )
            self.assertEqual(
                relation_status(stages[1], "object_0003", "INSERTABLE_IN"),
                "FALSE",
            )
            self.assertEqual(
                relation_status(stages[2], "object_0004", "INSERTABLE_IN"),
                "TRUE",
            )
            self.assertEqual(
                relation_status(stages[2], "object_0004", "REACHES_BOTTOM"),
                "TRUE",
            )
            events = [
                json.loads(line)
                for line in session.events_path.read_text().splitlines()
            ]
            self.assertTrue(
                any(
                    event.get("event")
                    == "CANDIDATE_REJECTED_GEOMETRY"
                    and event.get("stage") == 1
                    and event.get("object_id") == "object_0003"
                    and event.get("failed_relation") == "INSERTABLE_IN"
                    for event in events
                )
            )


if __name__ == "__main__":
    unittest.main()
