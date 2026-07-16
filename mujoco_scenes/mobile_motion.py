"""PDDL-style mobile move execution with collision-checked RRT* paths."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import mujoco
import mujoco.viewer
import numpy as np


BASE_JOINTS = (
    "robot0:base_forward_joint",
    "robot0:base_lateral_joint",
    "robot0:base_yaw_joint",
)
BASE_ACTUATORS = (
    "robot0:base_forward_actuator",
    "robot0:base_lateral_actuator",
    "robot0:base_yaw_actuator",
)


@dataclass(frozen=True)
class BasePose:
    """A named world-frame planar pose for the Fetch base."""

    x: float
    y: float
    yaw: float


HOME_POSE = BasePose(0.0, -1.10, 0.0)
# The table ends at X=+-0.70. These side poses halve the former
# centre-to-table-edge separation from 0.65 m to 0.325 m while retaining
# collision-free clearance for the base and tucked arm.
LEFT_POSE = BasePose(-1.025, -0.10, -math.pi / 2)
RIGHT_POSE = BasePose(1.025, -0.10, math.pi / 2)

# Cupboard 2 and B1 deliberately share one physical manipulation pose.
LOCATION_ALIASES = {
    "home": "home",
    "cupboard1": "cupboard1",
    "cupboard2": "right_side",
    "box": "right_side",
}
PHYSICAL_POSES = {
    "home": HOME_POSE,
    "cupboard1": LEFT_POSE,
    "right_side": RIGHT_POSE,
}

# These anchors enforce the requested motion shape: move laterally beyond the
# table first, advance beside the table second, and rotate only at the end.
ROUTE_ANCHORS = {
    "home": HOME_POSE,
    "left_staging": BasePose(-1.35, HOME_POSE.y, 0.0),
    "left_clearance": BasePose(-1.35, LEFT_POSE.y, 0.0),
    "cupboard1": BasePose(LEFT_POSE.x, LEFT_POSE.y, 0.0),
    "right_staging": BasePose(1.35, HOME_POSE.y, 0.0),
    "right_clearance": BasePose(1.35, RIGHT_POSE.y, 0.0),
    "right_side": BasePose(RIGHT_POSE.x, RIGHT_POSE.y, 0.0),
}
ANCHOR_BRANCHES = {
    "home": ("home",),
    "cupboard1": ("home", "left_staging", "left_clearance", "cupboard1"),
    "right_side": ("home", "right_staging", "right_clearance", "right_side"),
}


def physical_location(name: str) -> str:
    """Resolve a UI/PDDL location name to its physical pose name."""
    try:
        return LOCATION_ALIASES[name]
    except KeyError as error:
        choices = ", ".join(LOCATION_ALIASES)
        raise ValueError(f"Unknown move destination '{name}'. Choose from: {choices}") from error


def anchor_route(source: str, destination: str) -> list[str]:
    """Return the canonical anchor route between two physical locations."""
    if source == destination:
        return [source]
    source_branch = ANCHOR_BRANCHES[source]
    destination_branch = ANCHOR_BRANCHES[destination]
    common = 0
    for left, right in zip(source_branch, destination_branch):
        if left != right:
            break
        common += 1
    up = list(reversed(source_branch[common:]))
    lca = source_branch[common - 1]
    down = list(destination_branch[common:])
    return up + [lca] + down


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


@dataclass
class _Node:
    point: np.ndarray
    parent: int | None
    cost: float


class RRTStarPlanner:
    """Small deterministic 2-D RRT* planner with edge collision checking."""

    def __init__(
        self,
        state_valid: Callable[[float, float], bool],
        bounds: tuple[tuple[float, float], tuple[float, float]],
        *,
        step_size: float = 0.14,
        edge_resolution: float = 0.035,
        neighbor_radius: float = 0.32,
        max_iterations: int = 1800,
        goal_sample_rate: float = 0.18,
        seed: int = 13,
    ):
        self.state_valid = state_valid
        self.bounds = bounds
        self.step_size = step_size
        self.edge_resolution = edge_resolution
        self.neighbor_radius = neighbor_radius
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.rng = np.random.default_rng(seed)

    def _edge_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        length = _distance(start, end)
        count = max(1, int(math.ceil(length / self.edge_resolution)))
        for fraction in np.linspace(0.0, 1.0, count + 1):
            point = start + fraction * (end - start)
            if not self.state_valid(float(point[0]), float(point[1])):
                return False
        return True

    def _interpolate(self, path: Iterable[np.ndarray]) -> list[tuple[float, float]]:
        points = list(path)
        output: list[tuple[float, float]] = []
        for start, end in zip(points, points[1:]):
            length = _distance(start, end)
            count = max(1, int(math.ceil(length / 0.06)))
            for fraction in np.linspace(0.0, 1.0, count, endpoint=False):
                point = start + fraction * (end - start)
                output.append((float(point[0]), float(point[1])))
        output.append((float(points[-1][0]), float(points[-1][1])))
        return output

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        start_point = np.asarray(start, dtype=float)
        goal_point = np.asarray(goal, dtype=float)
        if not self.state_valid(*start_point):
            raise RuntimeError(f"RRT* start is in collision: {tuple(start_point)}")
        if not self.state_valid(*goal_point):
            raise RuntimeError(f"RRT* goal is in collision: {tuple(goal_point)}")

        # A visible straight edge is the Euclidean lower bound and therefore
        # already the globally optimal RRT* result for this anchored segment.
        if self._edge_valid(start_point, goal_point):
            return self._interpolate((start_point, goal_point))

        nodes = [_Node(start_point, None, 0.0)]
        goal_index: int | None = None
        for _ in range(self.max_iterations):
            if self.rng.random() < self.goal_sample_rate:
                sample = goal_point
            else:
                sample = np.array(
                    [
                        self.rng.uniform(*self.bounds[0]),
                        self.rng.uniform(*self.bounds[1]),
                    ]
                )
            distances = [_distance(node.point, sample) for node in nodes]
            nearest_index = int(np.argmin(distances))
            nearest = nodes[nearest_index]
            delta = sample - nearest.point
            length = float(np.linalg.norm(delta))
            new_point = sample if length <= self.step_size else nearest.point + delta / length * self.step_size
            if not self._edge_valid(nearest.point, new_point):
                continue

            near_indices = [
                index for index, node in enumerate(nodes)
                if _distance(node.point, new_point) <= self.neighbor_radius
                and self._edge_valid(node.point, new_point)
            ]
            parent_index = nearest_index
            parent_cost = nearest.cost + _distance(nearest.point, new_point)
            for index in near_indices:
                candidate = nodes[index].cost + _distance(nodes[index].point, new_point)
                if candidate < parent_cost:
                    parent_index, parent_cost = index, candidate
            nodes.append(_Node(new_point, parent_index, parent_cost))
            new_index = len(nodes) - 1

            for index in near_indices:
                rewired_cost = parent_cost + _distance(new_point, nodes[index].point)
                if rewired_cost < nodes[index].cost and self._edge_valid(new_point, nodes[index].point):
                    nodes[index].parent = new_index
                    nodes[index].cost = rewired_cost

            if _distance(new_point, goal_point) <= self.step_size and self._edge_valid(new_point, goal_point):
                cost = parent_cost + _distance(new_point, goal_point)
                if goal_index is None:
                    nodes.append(_Node(goal_point, new_index, cost))
                    goal_index = len(nodes) - 1
                elif cost < nodes[goal_index].cost:
                    nodes[goal_index].parent = new_index
                    nodes[goal_index].cost = cost

        if goal_index is None:
            raise RuntimeError(f"RRT* could not connect {start} to {goal}")
        path = []
        index: int | None = goal_index
        while index is not None:
            path.append(nodes[index].point)
            index = nodes[index].parent
        path.reverse()
        return self._interpolate(path)


class MuJoCoBaseCollisionChecker:
    """Collision checker that evaluates candidate base poses in spare MjData."""

    def __init__(self, model: mujoco.MjModel, reference_data: mujoco.MjData):
        self.model = model
        self.data = mujoco.MjData(model)
        self.reference_qpos = reference_data.qpos.copy()
        self.data.eq_active[:] = reference_data.eq_active
        self.qpos_addresses = tuple(
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in BASE_JOINTS
        )
        self.attached_body_ids: set[int] = set()
        for equality_id in range(model.neq):
            if not reference_data.eq_active[equality_id]:
                continue
            if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_WELD:
                continue
            first_body = int(model.eq_obj1id[equality_id])
            second_body = int(model.eq_obj2id[equality_id])
            first_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, first_body
            ) or ""
            second_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, second_body
            ) or ""
            if first_name.startswith("robot0:") and not second_name.startswith("robot0:"):
                self.attached_body_ids.add(second_body)
            elif second_name.startswith("robot0:") and not first_name.startswith("robot0:"):
                self.attached_body_ids.add(first_body)
        self.reference_base = self.reference_qpos[list(self.qpos_addresses)].copy()
        self.attached_free_qpos: list[int] = []
        for body_id in self.attached_body_ids:
            joint_address = int(model.body_jntadr[body_id])
            if model.body_jntnum[body_id] != 1:
                continue
            if model.jnt_type[joint_address] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            self.attached_free_qpos.append(int(model.jnt_qposadr[joint_address]))
        self.cache: dict[tuple[int, int], bool] = {}

    def __call__(self, x: float, y: float) -> bool:
        key = (round(x * 1000), round(y * 1000))
        if key in self.cache:
            return self.cache[key]
        forward = y - HOME_POSE.y
        lateral = -x
        if not (-1.0 <= forward <= 1.0 and -1.5 <= lateral <= 1.5):
            self.cache[key] = False
            return False
        self.data.qpos[:] = self.reference_qpos
        self.data.qpos[list(self.qpos_addresses)] = (forward, lateral, 0.0)
        if self.attached_free_qpos:
            ref_forward, ref_lateral, ref_yaw = self.reference_base
            reference_xy = np.array((-ref_lateral, HOME_POSE.y + ref_forward))
            candidate_xy = np.array((x, y))
            cosine, sine = math.cos(-ref_yaw), math.sin(-ref_yaw)
            inverse_rotation = np.array(((cosine, -sine), (sine, cosine)))
            delta_quat = np.array(
                (math.cos(-ref_yaw / 2), 0.0, 0.0, math.sin(-ref_yaw / 2))
            )
            for qpos_address in self.attached_free_qpos:
                relative_xy = inverse_rotation @ (
                    self.reference_qpos[qpos_address:qpos_address + 2] - reference_xy
                )
                self.data.qpos[qpos_address:qpos_address + 2] = candidate_xy + relative_xy
                rotated_quat = np.empty(4)
                mujoco.mju_mulQuat(
                    rotated_quat,
                    delta_quat,
                    self.reference_qpos[qpos_address + 3:qpos_address + 7],
                )
                self.data.qpos[qpos_address + 3:qpos_address + 7] = rotated_quat
        mujoco.mj_forward(self.model, self.data)
        valid = True
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            second = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            first_robot = (
                first.startswith("robot0:")
                or self.model.geom_bodyid[contact.geom1] in self.attached_body_ids
            )
            second_robot = (
                second.startswith("robot0:")
                or self.model.geom_bodyid[contact.geom2] in self.attached_body_ids
            )
            if first_robot == second_robot:
                continue
            other = second if first_robot else first
            if other != "floor":
                valid = False
                break
        self.cache[key] = valid
        return valid


def _angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


class MobileMoveExecutor:
    """Plans and incrementally executes one named mobile move action."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.joint_addresses = tuple(
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in BASE_JOINTS
        )
        self.actuator_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in BASE_ACTUATORS
        )
        self.current_physical_location = "home"
        self.current_symbolic_location = "home"
        self.requested_location: str | None = None
        self.targets: list[tuple[float, float, float]] = []
        self.target_index = 0
        self.status = "Idle at home"
        self.started_at = 0.0

    @property
    def busy(self) -> bool:
        return self.target_index < len(self.targets)

    def _current_world_pose(self) -> BasePose:
        forward, lateral, yaw = self.data.qpos[list(self.joint_addresses)]
        return BasePose(float(-lateral), float(HOME_POSE.y + forward), float(yaw))

    @staticmethod
    def _rotation_targets(x: float, y: float, start: float, goal: float) -> list[tuple[float, float, float]]:
        delta = _angle_delta(goal, start)
        count = max(1, int(math.ceil(abs(delta) / math.radians(4))))
        return [
            (x, y, start + delta * fraction)
            for fraction in np.linspace(0.0, 1.0, count + 1)[1:]
        ]

    def request_move(self, destination: str) -> None:
        if self.busy:
            raise RuntimeError("A move action is already running")
        destination_physical = physical_location(destination)
        if destination_physical == self.current_physical_location:
            self.current_symbolic_location = destination
            self.status = f"Already at {destination} (shared physical pose)"
            return

        current = self._current_world_pose()
        route = anchor_route(self.current_physical_location, destination_physical)
        checker = MuJoCoBaseCollisionChecker(self.model, self.data)
        planner = RRTStarPlanner(
            checker,
            bounds=((-1.42, 1.42), (HOME_POSE.y, LEFT_POSE.y)),
        )
        targets: list[tuple[float, float, float]] = []
        if abs(current.yaw) > math.radians(1):
            # Leave a close manipulation pose by backing straight away from
            # the table before rotating. This protects a carried object and
            # the extended carry arm from the cabinet/table during the turn.
            if len(route) > 1 and route[1].endswith("_clearance"):
                retreat = ROUTE_ANCHORS[route[1]]
                targets.append((retreat.x, retreat.y, current.yaw))
                current = BasePose(retreat.x, retreat.y, current.yaw)
                route = route[1:]
            targets.extend(self._rotation_targets(current.x, current.y, current.yaw, 0.0))

        source_anchor = ROUTE_ANCHORS[route[0]]
        if _distance(
            np.array((current.x, current.y)),
            np.array((source_anchor.x, source_anchor.y)),
        ) > 0.002:
            # Remove the small actuator tracking residual before planning the
            # next canonical branch. This correction is at most a few cm.
            targets.append((source_anchor.x, source_anchor.y, 0.0))
        cursor = (source_anchor.x, source_anchor.y)
        for anchor_name in route[1:]:
            anchor = ROUTE_ANCHORS[anchor_name]
            segment = planner.plan(cursor, (anchor.x, anchor.y))
            targets.extend((x, y, 0.0) for x, y in segment[1:])
            cursor = (anchor.x, anchor.y)

        final_pose = PHYSICAL_POSES[destination_physical]
        targets.extend(self._rotation_targets(cursor[0], cursor[1], 0.0, final_pose.yaw))
        self.targets = targets
        self.target_index = 0
        self.requested_location = destination
        self.status = f"Moving to {destination}: RRT* path has {len(targets)} waypoints"
        self.started_at = time.monotonic()

    def update(self) -> None:
        if not self.busy:
            return
        x, y, yaw = self.targets[self.target_index]
        joint_target = np.array((y - HOME_POSE.y, -x, yaw))
        for actuator_id, value in zip(self.actuator_ids, joint_target):
            self.data.ctrl[actuator_id] = value
        current = self.data.qpos[list(self.joint_addresses)]
        position_error = float(np.linalg.norm(current[:2] - joint_target[:2]))
        yaw_error = abs(_angle_delta(float(joint_target[2]), float(current[2])))
        if position_error < 0.018 and yaw_error < math.radians(1.2):
            self.target_index += 1
            if not self.busy:
                assert self.requested_location is not None
                self.current_symbolic_location = self.requested_location
                self.current_physical_location = physical_location(self.requested_location)
                elapsed = time.monotonic() - self.started_at
                self.status = f"Move complete: {self.requested_location} ({elapsed:.1f} s)"

    def progress(self) -> float:
        if not self.targets:
            return 0.0
        return min(1.0, self.target_index / len(self.targets))


def launch_action_viewer(scene, camera: str) -> None:
    """Launch MuJoCo plus a companion hierarchical Actions panel."""
    import tkinter as tk
    from tkinter import ttk

    from mujoco_scenes.pick_motion import PickExecutor, TABLE_PICK_SPECS

    camera_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id
    executor = MobileMoveExecutor(scene.model, scene.data)
    picker = PickExecutor(scene.model, scene.data)

    root = tk.Tk()
    root.title("Kitchen Actions")
    root.geometry("380x680+20+20")
    root.minsize(360, 560)
    root.columnconfigure(0, weight=1)

    status = tk.StringVar(value=executor.status)
    location = tk.StringVar(value="home")
    progress = tk.DoubleVar(value=0.0)
    selected_pick = tk.StringVar(value="Selected: none")
    ui_error: str | None = None

    actions_open = tk.BooleanVar(value=True)
    actions_toggle = ttk.Checkbutton(root, text="Actions", variable=actions_open)
    actions_toggle.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
    actions_body = ttk.Frame(root, padding=(12, 4))
    actions_body.grid(row=1, column=0, sticky="nsew")
    actions_body.columnconfigure(0, weight=1)

    move_open = tk.BooleanVar(value=True)
    move_toggle = ttk.Checkbutton(actions_body, text="Move", variable=move_open)
    move_toggle.grid(row=0, column=0, sticky="ew")
    move_body = ttk.LabelFrame(actions_body, text="Destination", padding=10)
    move_body.grid(row=1, column=0, sticky="ew", pady=(4, 10))
    move_body.columnconfigure(0, weight=1)

    move_buttons: list[ttk.Button] = []

    def request_move(destination: str) -> None:
        nonlocal ui_error
        ui_error = None
        try:
            if picker.busy:
                raise RuntimeError("Wait for the pick action to finish")
            executor.request_move(destination)
            status.set(executor.status)
        except Exception as error:  # surfaced in the panel instead of killing the viewer
            ui_error = f"Move failed: {error}"
            status.set(ui_error)
            print(f"[Actions] {ui_error}")

    for row, (label, destination) in enumerate(
        (("Home", "home"), ("Cupboard 1", "cupboard1"),
         ("Cupboard 2", "cupboard2"), ("Box", "box"))
    ):
        button = ttk.Button(
            move_body,
            text=label,
            command=lambda name=destination: request_move(name),
        )
        button.grid(row=row, column=0, sticky="ew", pady=3)
        move_buttons.append(button)

    pick_open = tk.BooleanVar(value=True)
    pick_toggle = ttk.Checkbutton(actions_body, text="Pick", variable=pick_open)
    pick_toggle.grid(row=2, column=0, sticky="ew", pady=(4, 0))
    pick_body = ttk.LabelFrame(actions_body, text="Table object", padding=10)
    pick_body.grid(row=3, column=0, sticky="ew", pady=(4, 10))
    pick_body.columnconfigure(0, weight=1)
    pick_buttons: list[ttk.Button] = []

    def request_pick(object_name: str) -> None:
        nonlocal ui_error
        ui_error = None
        selected_pick.set(f"Selected: {TABLE_PICK_SPECS[object_name].label}")
        status.set(f"Planning pick for {object_name}...")
        root.update_idletasks()
        try:
            if executor.busy:
                raise RuntimeError("Wait for the move action to finish")
            if executor.current_physical_location != "home":
                raise RuntimeError("Tabletop picks require Move (home) first")
            picker.request_pick(object_name)
            status.set(picker.status)
        except Exception as error:
            ui_error = f"Pick failed: {error}"
            status.set(ui_error)
            print(f"[Actions] {ui_error}")

    for row, (object_name, spec) in enumerate(TABLE_PICK_SPECS.items()):
        present = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        ) >= 0
        button = ttk.Button(
            pick_body,
            text=spec.label,
            command=lambda name=object_name: request_pick(name),
        )
        button.grid(row=row, column=0, sticky="ew", pady=3)
        if not present:
            button.configure(state="disabled")
        pick_buttons.append(button)
    ttk.Label(pick_body, textvariable=selected_pick).grid(
        row=len(TABLE_PICK_SPECS), column=0, sticky="w", pady=(8, 0)
    )

    future = ttk.LabelFrame(actions_body, text="Future actions", padding=8)
    future.grid(row=4, column=0, sticky="ew")
    ttk.Label(future, text="Place  ·  Open  ·  Close  ·  Pour", state="disabled").grid()

    ttk.Separator(root).grid(row=2, column=0, sticky="ew", padx=12, pady=8)
    ttk.Label(root, text="Current symbolic location:").grid(row=3, column=0, sticky="w", padx=12)
    ttk.Label(root, textvariable=location).grid(row=4, column=0, sticky="w", padx=12)
    ttk.Progressbar(root, variable=progress, maximum=1.0).grid(
        row=5, column=0, sticky="ew", padx=12, pady=8
    )
    ttk.Label(root, textvariable=status, wraplength=330).grid(
        row=6, column=0, sticky="ew", padx=12
    )

    def toggle_actions(*_args) -> None:
        if actions_open.get():
            actions_body.grid()
        else:
            actions_body.grid_remove()

    def toggle_move(*_args) -> None:
        if move_open.get():
            move_body.grid()
        else:
            move_body.grid_remove()

    def toggle_pick(*_args) -> None:
        if pick_open.get():
            pick_body.grid()
        else:
            pick_body.grid_remove()

    actions_toggle.configure(command=toggle_actions)
    move_toggle.configure(command=toggle_move)
    pick_toggle.configure(command=toggle_pick)

    closed = False

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        if viewer.is_running():
            viewer.close()
        root.destroy()

    def tick() -> None:
        if closed:
            return
        if not viewer.is_running():
            close()
            return
        for _ in range(5):
            if picker.busy:
                picker.update()
            else:
                executor.update()
            mujoco.mj_step(scene.model, scene.data)
        active_status = ui_error or (
            picker.status
            if picker.busy or picker.held_object is not None or picker.failure
            else executor.status
        )
        if hasattr(viewer, "set_texts"):
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Actions / Move + Pick",
                    active_status,
                )
            )
        viewer.sync()
        location.set(executor.current_symbolic_location)
        status.set(active_status)
        progress.set(picker.progress() if picker.busy else executor.progress())
        move_state = "disabled" if executor.busy or picker.busy else "normal"
        for button in move_buttons:
            button.configure(state=move_state)
        can_pick = (
            not executor.busy
            and not picker.busy
            and picker.held_object is None
            and executor.current_physical_location == "home"
        )
        for button, object_name in zip(pick_buttons, TABLE_PICK_SPECS):
            present = mujoco.mj_name2id(
                scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
            ) >= 0
            button.configure(state="normal" if can_pick and present else "disabled")
        root.after(10, tick)

    root.protocol("WM_DELETE_WINDOW", close)
    tick()
    root.mainloop()
