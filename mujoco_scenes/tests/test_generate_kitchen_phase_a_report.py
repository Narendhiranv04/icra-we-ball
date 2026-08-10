import json

from mujoco_scenes.generate_kitchen_phase_a_report import generate


def _step(container, action, workspace):
    return {
        "action": action,
        "target_container": container,
        "actual_workspace": workspace,
        "required_workspace": workspace,
        "success": True,
        "status": "EXECUTION_SUCCESS",
        "handle_contact_evidence": True,
        "handle_attachment_evidence": True,
        "collision_status": "PLAN_VALID_AND_LIVE_GUARD_ACTIVE",
        "direct_container_actuator_used": False,
        "live_qpos_write_used": False,
        "unexpected_articulation_motion": False,
        "final_postcondition": action,
        "attachment_translation_snap_m": 0.0001,
        "attachment_angle_snap_rad": 0.001,
        "ik_max_position_residual_m": 0.0008,
        "ik_max_angle_residual_rad": 0.01,
        "physical_motion_source": "GOOGLE_ROBOT_HANDLE_MANIPULATION",
    }


def test_report_is_derived_from_complete_physical_run(tmp_path):
    workspaces = {
        "D1": "home", "D2": "home", "C1": "left_side",
        "C2": "right_side", "B1": "right_side",
    }
    records = []
    current = "home"
    for container, workspace in workspaces.items():
        for action in ("OPEN", "CLOSE"):
            move = current != workspace
            steps = []
            if move:
                steps.append({
                    "action": "MOVE", "success": True,
                    "source_workspace": current, "target_workspace": workspace,
                })
                current = workspace
            steps.append(_step(container, action, workspace))
            records.append({
                "request": {"action": action, "target": container},
                "refinement": {
                    "auto_move_inserted": move,
                    "starting_workspace": steps[0].get("source_workspace", current),
                    "required_workspace": workspace,
                },
                "steps": steps,
                "success": True,
            })
    run = {
        "scene": "S1_integrated_kitchen_object_function_primary",
        "status": "SUCCESS",
        "records": records,
    }
    source = tmp_path / "run.json"
    source.write_text(json.dumps(run))
    report = generate(source, report_root=tmp_path / "report")
    guards = json.loads((report / "scientific_guard_report.json").read_text())
    assert guards["status"] == "PASS"
    assert all(guards["checks"].values())
    assert (report / "authoritative/B1/open.json").exists()
