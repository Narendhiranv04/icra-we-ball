import json

import yaml

from mujoco_scenes.evaluate_joint_grounding_run import evaluate_saved_run


def _mode(tool_label):
    return {
        "selected_witness": {
            "mixing_container": ["object_0001"],
            "mixing_tool": ["object_0002"],
        },
        "selected_candidate_edges": [
            {
                "role": "mixing_container",
                "object_id": "object_0001",
                "canonical_label": "bowl",
            },
            {
                "role": "mixing_tool",
                "object_id": "object_0002",
                "canonical_label": tool_label,
            },
        ],
        "candidate_evaluations": [],
        "assignment_evaluations": [],
    }


def test_offline_report_separates_expected_ablation_from_ground_truth(
    tmp_path,
):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stages" / "000_initial"
    stage_dir.mkdir(parents=True)
    modes = {
        "joint": _mode("fork"),
        "geometry-only": _mode("pen"),
        "semantic-only": _mode("spoon"),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps({"scene_name": "synthetic"})
    )
    (run_dir / "ablation_summary.json").write_text(
        json.dumps(
            {
                "task_id": "stir_contents",
                "shared_observation_evidence": True,
                "stages": [
                    {
                        "stage": 0,
                        "region_id": "INITIAL",
                        "modes": {
                            name: {"status": "COMPLETE"}
                            for name in modes
                        },
                    }
                ],
            }
        )
    )
    (stage_dir / "grounding_mode_comparison.json").write_text(
        json.dumps({"modes": modes})
    )
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(
        yaml.safe_dump(
            {
                "purpose": "OFFLINE_EVALUATION_ONLY",
                "runtime_import_forbidden": True,
                "scenes": {
                    "synthetic": {
                        "ground_truth": {
                            "status": "COMPLETE",
                            "selected_labels": {
                                "mixing_container": "bowl",
                                "mixing_tool": "fork",
                            },
                            "incorrect_selection_reasons": {
                                "geometry-only": "semantic counterexample",
                                "semantic-only": "geometric counterexample",
                            },
                        },
                        "expected": {
                            name: {
                                "completion_stage": 0,
                                "selected_labels": {
                                    "mixing_container": "bowl",
                                    "mixing_tool": tool,
                                },
                            }
                            for name, tool in (
                                ("joint", "fork"),
                                ("geometry-only", "pen"),
                                ("semantic-only", "spoon"),
                            )
                        },
                    }
                },
            }
        )
    )

    external_output = tmp_path / "reports" / "offline.json"
    report = evaluate_saved_run(
        run_dir,
        evaluation_config=evaluation_path,
        output_path=external_output,
    )

    assert external_output.exists()
    assert not (run_dir / "offline_ablation_evaluation.json").exists()
    assert report["all_expected_results_matched"]
    assert report["modes"]["joint"][
        "matches_evaluation_ground_truth"
    ]
    assert not report["modes"]["geometry-only"][
        "matches_evaluation_ground_truth"
    ]
    assert not report["modes"]["semantic-only"][
        "matches_evaluation_ground_truth"
    ]
