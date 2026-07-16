"""Headless repeatable checks for calibrated robot actions."""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from mujoco_scenes.generic_manipulation import CalibratedPickPlaceExecutor
from mujoco_scenes.mobile_motion import (
    MobileMoveExecutor,
    MuJoCoBaseCollisionChecker,
)
from mujoco_scenes.scene_loader import KitchenScene


def _run_until(scene, executor, terminal_modes: set[str], max_steps: int) -> int:
    for step in range(max_steps):
        executor.update()
        mujoco.mj_step(scene.model, scene.data)
        if executor.mode in terminal_modes:
            return step + 1
    raise RuntimeError(
        f"Calibration action timed out after {max_steps} steps: {executor.status}"
    )


def _run_move(scene, destination: str, max_steps: int) -> None:
    executor = MobileMoveExecutor(scene.model, scene.data, scene.robot_name)
    executor.request_move(destination)
    for step in range(max_steps):
        executor.update()
        mujoco.mj_step(scene.model, scene.data)
        if not executor.busy:
            print(f"[PASS] {executor.status}; simulation steps={step + 1}")
            return
    raise RuntimeError(f"Move timed out after {max_steps} steps: {executor.status}")


def _run_monitored_move(scene, executor, destination: str, max_steps: int) -> int:
    checker = MuJoCoBaseCollisionChecker(
        scene.model, scene.data, executor.profile
    )
    executor.request_move(destination)
    for step in range(max_steps):
        executor.update()
        mujoco.mj_step(scene.model, scene.data)
        for contact in scene.data.contact:
            first_robot = checker._geom_is_robot(contact.geom1)
            second_robot = checker._geom_is_robot(contact.geom2)
            if first_robot == second_robot:
                continue
            other_geom = contact.geom2 if first_robot else contact.geom1
            other_name = mujoco.mj_id2name(
                scene.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or ""
            if other_name != "floor":
                raise RuntimeError(
                    f"Live collision while moving to {destination}: "
                    f"{other_name or 'unnamed environment geom'}"
                )
        if not executor.busy:
            return step + 1
    raise RuntimeError(
        f"Monitored move timed out after {max_steps} steps: {executor.status}"
    )


def _validate_spoon_carry(scene, executor) -> None:
    spoon_rotation = scene.data.xmat[executor.target_body_id].reshape(3, 3)
    bowl_axis = -spoon_rotation[:, 0]
    angle = math.acos(
        float(np.clip(bowl_axis @ np.array((0.0, 0.0, -1.0)), -1.0, 1.0))
    )
    if angle > math.radians(5.0):
        raise RuntimeError(
            f"Spoon carry is {math.degrees(angle):.1f} degrees from vertical"
        )
    if executor.spoon_pivot_equality_id < 0:
        raise RuntimeError("Spoon pick never activated its passive pivot")
    if scene.data.eq_active[executor.spoon_pivot_equality_id]:
        raise RuntimeError("Spoon pivot remained active after carry stabilization")
    if not scene.data.eq_active[executor.grasp_equality_id]:
        raise RuntimeError("Spoon transport weld is inactive after stabilization")
    if executor.can_place:
        raise RuntimeError("Uncalibrated spoon placement was exposed")
    print(
        f"[PASS] Google spoon hangs bowl-down within "
        f"{math.degrees(angle):.1f} degrees and is secured for carry"
    )


def _run_pick_place(
    scene,
    object_name: str,
    place: bool,
    held_move: str | None,
    move_after_place: str | None,
    max_steps: int,
) -> None:
    executor = CalibratedPickPlaceExecutor(
        scene.model, scene.data, scene.robot_name, scene.scene_name
    )
    executor.request_pick(object_name)
    pick_steps = _run_until(scene, executor, {"holding", "failed"}, max_steps)
    if executor.mode == "failed":
        raise RuntimeError(executor.status)
    print(f"[PASS] {executor.status}; simulation steps={pick_steps}")
    if object_name == "spoon" and scene.robot_name == "google":
        _validate_spoon_carry(scene, executor)
    if held_move:
        mobile = MobileMoveExecutor(scene.model, scene.data, scene.robot_name)
        outbound_steps = _run_monitored_move(
            scene, mobile, held_move, max_steps
        )
        return_steps = _run_monitored_move(scene, mobile, "home", max_steps)
        print(
            f"[PASS] Held-object round trip: home -> {held_move} -> home; "
            f"simulation steps={outbound_steps + return_steps}"
        )
    if not place:
        return

    executor.request_place("serving_spot")
    place_steps = _run_until(scene, executor, {"idle", "failed"}, max_steps)
    if executor.mode == "failed":
        raise RuntimeError(executor.status)
    for _ in range(1000):
        mujoco.mj_step(scene.model, scene.data)
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
    )
    site_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_SITE, "serving_spot"
    )
    horizontal_error = float(
        np.linalg.norm(scene.data.xpos[body_id, :2] - scene.data.site_xpos[site_id, :2])
    )
    if horizontal_error > 0.05:
        raise RuntimeError(
            f"Placed object is {horizontal_error * 100:.1f} cm from serving-site centre"
        )
    print(
        f"[PASS] {executor.status}; simulation steps={place_steps}; "
        f"centre error={horizontal_error * 100:.1f} cm"
    )
    if move_after_place:
        mobile = MobileMoveExecutor(scene.model, scene.data, scene.robot_name)
        move_steps = _run_monitored_move(
            scene, mobile, move_after_place, max_steps
        )
        print(
            f"[PASS] Post-place compact move to {move_after_place}; "
            f"simulation steps={move_steps}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run headless mobile/manipulation calibration checks"
    )
    parser.add_argument("--scene", default="S1_coffee_missing_mug")
    parser.add_argument("--robot", default="google")
    parser.add_argument(
        "--move",
        choices=("home", "cupboard1", "cupboard2", "box"),
        help="Run one navigation calibration check",
    )
    parser.add_argument(
        "--pick", help="Run a calibrated pick for this countertop object"
    )
    parser.add_argument(
        "--place",
        action="store_true",
        help="After --pick, place the held object at serving_spot",
    )
    parser.add_argument(
        "--move-while-holding",
        choices=("cupboard1", "cupboard2", "box"),
        help="After --pick, monitor a held-object round trip to this destination",
    )
    parser.add_argument(
        "--move-after-place",
        choices=("cupboard1", "cupboard2", "box"),
        help="After --place, monitor one compact empty-gripper move",
    )
    parser.add_argument("--max-steps", type=int, default=30000)
    args = parser.parse_args()
    if args.place and not args.pick:
        parser.error("--place requires --pick")
    if args.move_while_holding and not args.pick:
        parser.error("--move-while-holding requires --pick")
    if args.move_after_place and not args.place:
        parser.error("--move-after-place requires --place")
    if not args.move and not args.pick:
        parser.error("select at least one of --move or --pick")

    scene = KitchenScene(args.scene, robot=args.robot)
    if args.move:
        _run_move(scene, args.move, args.max_steps)
    if args.pick:
        _run_pick_place(
            scene,
            args.pick,
            args.place,
            args.move_while_holding,
            args.move_after_place,
            args.max_steps,
        )


if __name__ == "__main__":
    main()
