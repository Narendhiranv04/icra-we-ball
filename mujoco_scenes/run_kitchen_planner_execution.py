"""Execute an observed-state planner result through Kitchen Phase C.

The input artifacts are produced by perception/entity resolution. This runner
never reads the scene's hidden object inventory or an oracle plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .kitchen_tamp_execution import KitchenGroundedExecution, effects_goal_verifier
from .scene_loader import KitchenScene
from .tamp.grounded_execution import FixedSequence
from .tamp.physical_dispatcher import canonical_action
from .tamp.skills import SkillAction


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def planner_actions(payload: Any) -> tuple[SkillAction, ...]:
    """Accept LLM3, TAMP, or canonical Phase-C JSON action records."""
    if isinstance(payload, dict) and "plan" in payload:
        payload = payload["plan"]
    if isinstance(payload, dict) and "actions" in payload:
        payload = payload["actions"]
    if not isinstance(payload, list) or not payload:
        raise ValueError("Planner output must contain a non-empty actions list")
    result = []
    names = {
        "PICK": ("object_id",),
        "PLACE": ("object_id", "region_id"),
        "POUR": ("source_id", "target_id", "content"),
        "STIR": ("tool_id", "target_id"),
        "INSPECT": ("region_id",),
    }
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Each planner action must be an object")
        name = str(row.get("skill") or row.get("name") or row.get("action") or row.get("operator") or "").upper()
        arguments = row.get("arguments", {})
        if name not in names:
            raise ValueError(f"Unsupported kitchen action: {name or row!r}")
        expected_names = names[name]
        if isinstance(arguments, list):
            minimum_arity = 2 if name == "POUR" else len(expected_names)
            if not minimum_arity <= len(arguments) <= len(expected_names):
                raise ValueError(
                    f"{name} requires {minimum_arity}"
                    + (
                        f"-{len(expected_names)} arguments"
                        if minimum_arity != len(expected_names)
                        else " arguments"
                    )
                )
            arguments = {
                key: value
                for key, value in zip(expected_names, arguments)
                if value is not None
            }
        if not isinstance(arguments, dict):
            raise ValueError(f"{name} arguments must be an object or list")
        required_names = set(expected_names[:2] if name == "POUR" else expected_names)
        allowed_names = set(expected_names)
        if not required_names <= set(arguments) or set(arguments) - allowed_names:
            raise ValueError(
                f"{name} arguments must contain {sorted(required_names)}"
                + (
                    f" with optional {sorted(allowed_names - required_names)}"
                    if allowed_names != required_names
                    else ""
                )
            )
        if any(
            not isinstance(value, str) or not value
            for value in arguments.values()
        ):
            raise ValueError(f"{name} arguments must be non-empty strings")
        result.append(SkillAction(name, arguments))
    return tuple(result)


def execute_plan(
    scene: KitchenScene,
    inventory: dict[str, Any],
    resolution: dict[str, Any],
    registry: dict[str, Any],
    witness: dict[str, Any],
    actions: tuple[SkillAction, ...],
    *,
    goal: str | None = None,
    step_callback=None,
) -> dict[str, Any]:
    terminal_effects = []
    for action in actions:
        request = canonical_action(action)
        name, arguments = request["action"], request["arguments"]
        if name == "PLACE":
            terminal_effects.append(f"placed({arguments[0]},{arguments[1]})")
        elif name == "POUR":
            terminal_effects.append(f"poured({arguments[0]},{arguments[1]})")
        elif name == "STIR":
            terminal_effects.append(f"stirred({arguments[0]},{arguments[1]})")
    execution_ref: dict[str, KitchenGroundedExecution] = {}

    def physical_step() -> None:
        if step_callback is None:
            return
        active = execution_ref.get("execution")
        step_callback(
            active.executive.status if active is not None else "Initializing"
        )

    execution = KitchenGroundedExecution(
        scene,
        inventory,
        resolution,
        registry,
        FixedSequence(actions),
        effects_goal_verifier(terminal_effects),
        step_callback=physical_step if step_callback is not None else None,
        max_replans=0,
    )
    execution_ref["execution"] = execution
    execution.start(
        "external_grounded_plan",
        goal or scene.config.goal,
        witness,
    )
    executive = execution.run()
    trace = list(executive.history)
    return {
        "scene": scene.config.name,
        "goal": goal or scene.config.goal,
        "success": executive.mode == "complete",
        "status": executive.mode.upper(),
        "actions_requested": len(actions),
        "actions_executed": executive.executed_actions,
        "deterministic_replans": executive.replans,
        "trace": trace,
        "verified_witness_consumed": True,
        "foundation_model_called_during_execution": False,
        "oracle_inventory_used": False,
        "oracle_plan_used": False,
    }


def execute_with_viewer(
    scene: KitchenScene,
    inventory: dict[str, Any],
    resolution: dict[str, Any],
    registry: dict[str, Any],
    witness: dict[str, Any],
    actions: tuple[SkillAction, ...],
    *,
    goal: str | None = None,
    show_viewer: bool = True,
    camera: str = "free",
    close_on_complete: bool = False,
    report_status: bool = True,
) -> dict[str, Any]:
    """Execute a frozen plan while synchronizing a passive MuJoCo viewer."""
    viewer = None
    status = "Preparing grounded execution"
    last_reported_status: str | None = None

    def sync_viewer(message: str) -> None:
        nonlocal status, last_reported_status
        status = message
        if report_status and status != last_reported_status:
            print(f"[execution] {status}", flush=True)
            last_reported_status = status
        if viewer is None or not viewer.is_running():
            return
        if hasattr(viewer, "set_texts"):
            import mujoco

            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Grounded kitchen execution",
                    status,
                )
            )
        viewer.sync()

    try:
        if show_viewer:
            import mujoco
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
            if camera == "free":
                mujoco.mjv_defaultFreeCamera(scene.model, viewer.cam)
            else:
                camera_id = mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
                )
                if camera_id < 0:
                    raise ValueError(f"Unknown camera: {camera}")
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            sync_viewer(status)

        result = execute_plan(
            scene,
            inventory,
            resolution,
            registry,
            witness,
            actions,
            goal=goal,
            step_callback=sync_viewer,
        )
        sync_viewer(
            "Execution complete" if result["success"] else "Execution failed"
        )
        if viewer is not None and not close_on_complete:
            while viewer.is_running():
                sync_viewer(status)
                time.sleep(1.0 / 60.0)
        return result
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output")
    goal_group = parser.add_mutually_exclusive_group()
    goal_group.add_argument("--goal")
    goal_group.add_argument(
        "--goal-file",
        help="Plain-text goal file; useful for repeatable experiments",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show physical execution in a live MuJoCo window",
    )
    parser.add_argument(
        "--camera",
        default="free",
        help="Viewer startup camera (default: free)",
    )
    parser.add_argument(
        "--close-on-complete",
        action="store_true",
        help="Close the live viewer when execution terminates",
    )
    args = parser.parse_args()
    actions = planner_actions(_read(args.plan))
    scene = KitchenScene(args.scene, include_robot=True, robot="google")
    goal = args.goal
    if args.goal_file:
        goal = Path(args.goal_file).read_text(encoding="utf-8").strip()
        if not goal:
            parser.error("--goal-file must contain a non-empty goal")

    try:
        result = execute_with_viewer(
            scene,
            _read(args.inventory),
            _read(args.resolution),
            _read(args.registry),
            _read(args.witness),
            actions,
            goal=goal,
            show_viewer=args.viewer,
            camera=args.camera,
            close_on_complete=args.close_on_complete,
        )
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
