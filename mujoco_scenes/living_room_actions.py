"""Companion Actions panel for the rigid Google Robot living room."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer

from mujoco_scenes.living_room_manipulation import (
    LivingRoomManipulationExecutor,
)
from mujoco_scenes.living_room_commands import (
    COMMAND_HELP,
    DEFAULT_ACTION_FILE,
    LivingRoomCommand,
    load_living_room_commands,
)
from mujoco_scenes.living_room_drawer import MediaConsoleDrawerExecutor
from mujoco_scenes.living_room_navigation import (
    LIVING_ROOM_DESTINATIONS,
    LivingRoomNavigationExecutor,
)
from mujoco_scenes.living_room_scene import FREE_CAMERA, TV_CELL_COUNT
from mujoco_scenes.living_room_remote import RemoteTVExecutor
from mujoco_scenes.living_room_sofa import SofaInspectionExecutor
from mujoco_scenes.living_room_tamp import (
    STORAGE_TARGETS,
    LivingRoomTampController,
    semantic_storage_location,
)


MOVE_LABELS = {
    "home": "Home / coffee table",
    "table_south": "Coffee table - south side",
    "table_north": "Coffee table - north side",
    "table_east": "Coffee table - east side",
    "table_west": "Coffee table - west side",
    "bookshelf": "Book shelf",
    "drawer": "Media-console drawer",
    "drawer_left": "Media console - left drawer",
    "drawer_right": "Media console - right drawer",
    "couch": "Couch",
    "tv": "TV",
    "duster": "Duster rack",
}

def _value(instance: Any, name: str, default: Any = None) -> Any:
    """Return a property or zero-argument method without assuming which."""
    value = getattr(instance, name, default)
    return value() if callable(value) else value


def _construct_optional_duster(
    scene, manipulation: LivingRoomManipulationExecutor
) -> tuple[Any | None, str | None]:
    """Construct TVDustExecutor when that optional controller is available."""
    try:
        module = importlib.import_module("mujoco_scenes.living_room_dusting")
        executor_type = getattr(module, "TVDustExecutor")
        signature = inspect.signature(executor_type)
        context = {
            "scene": scene,
            "manipulation": manipulation,
            "manipulator": manipulation,
            "manipulation_executor": manipulation,
        }
        positional: list[Any] = []
        keywords: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if parameter.name in context:
                argument = context[parameter.name]
                if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                    positional.append(argument)
                else:
                    keywords[parameter.name] = argument
            elif parameter.default is inspect.Parameter.empty:
                return None, (
                    "TVDustExecutor has an unsupported required constructor "
                    f"argument: {parameter.name}"
                )
        return executor_type(*positional, **keywords), None
    except ModuleNotFoundError as error:
        if error.name == "mujoco_scenes.living_room_dusting":
            return None, "TV dust controller is not installed"
        return None, f"TV dust controller import failed: {error}"
    except Exception as error:  # display optional-controller failures in the UI
        return None, f"TV dust controller unavailable: {error}"


def _request_dust(
    executor: Any,
    *,
    current_location: str,
    held_object: str | None,
) -> None:
    """Call either request_dust() or request() using its declared context."""
    method = getattr(executor, "request_dust", None)
    if method is None:
        method = getattr(executor, "request", None)
    if method is None or not callable(method):
        raise RuntimeError("TVDustExecutor exposes neither request_dust nor request")

    context = {
        "current_location": current_location,
        "location": current_location,
        "held_object": held_object,
    }
    signature = inspect.signature(method)
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in context:
            argument = context[parameter.name]
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                positional.append(argument)
            else:
                keywords[parameter.name] = argument
        elif parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                "Unsupported TV dust request argument: " f"{parameter.name}"
            )
    method(*positional, **keywords)


def launch_living_room_actions(
    scene,
    camera: str = FREE_CAMERA,
    calibration_mode: bool = False,
    sofa_perception: str = "oracle",
    robot_debug_view: bool = False,
) -> None:
    """Launch MuJoCo with the living-room action and calibration panel."""
    if scene.robot_name != "google":
        raise ValueError("Living-room Actions currently require Google Robot")

    import tkinter as tk
    from tkinter import ttk

    navigation = LivingRoomNavigationExecutor(scene)
    manipulation = LivingRoomManipulationExecutor(
        scene, calibration_mode=calibration_mode
    )
    left_drawer = MediaConsoleDrawerExecutor(scene, "left")
    right_drawer = MediaConsoleDrawerExecutor(scene, "right")
    remote = RemoteTVExecutor(scene, manipulation)
    dusting, dust_unavailable_reason = _construct_optional_duster(
        scene, manipulation
    )
    tamp = LivingRoomTampController(
        navigation, manipulation, left_drawer, right_drawer
    )
    sofa = SofaInspectionExecutor(
        scene, perception_mode=sofa_perception
    )

    viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
    if camera == FREE_CAMERA:
        mujoco.mjv_defaultFreeCamera(scene.model, viewer.cam)
    else:
        camera_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        if camera_id < 0:
            viewer.close()
            raise ValueError(f"Unknown living-room camera: {camera}")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

    root = tk.Tk()
    root.title(
        "Living Room Calibration" if calibration_mode else "Living Room Actions"
    )
    root.geometry("500x820+20+20")
    root.minsize(430, 650)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    debug_view = None

    def open_debug_view() -> None:
        nonlocal debug_view
        if debug_view is not None and not debug_view.closed:
            debug_view.focus()
            return
        from mujoco_scenes.living_room_camera_debug import RobotCameraDebugView

        debug_view = RobotCameraDebugView(root, scene)

    status = tk.StringVar(value=navigation.status)
    location = tk.StringVar(value=navigation.current_location)
    held = tk.StringVar(value="none")
    table_pose = tk.StringVar(value="")
    coverage = tk.StringVar(value=f"TV coverage: 0/{TV_CELL_COUNT} (0%)")
    tv_state = tk.StringVar(value="TV power: off")
    progress = tk.DoubleVar(value=0.0)
    selected_pick = tk.StringVar(value="Selected: none")
    action_file = Path(
        os.environ.get("LIVING_ROOM_ACTION_FILE", str(DEFAULT_ACTION_FILE))
    ).expanduser()
    action_file_text = tk.StringVar(value=str(action_file))
    action_output = tk.StringVar(
        value="Edit the file, then reload and run it."
    )
    action_queue: list[LivingRoomCommand] = []
    action_script_running = False
    ui_error: str | None = None
    last_controller: Any = navigation

    canvas = tk.Canvas(root, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=1, column=0, sticky="nsew")
    scrollbar.grid(row=1, column=1, sticky="ns")
    actions_body = ttk.Frame(canvas, padding=(12, 6))
    actions_window = canvas.create_window(
        (0, 0), window=actions_body, anchor="nw"
    )
    actions_body.columnconfigure(0, weight=1)

    def resize_scroll_region(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def resize_actions(event) -> None:
        canvas.itemconfigure(actions_window, width=event.width)

    actions_body.bind("<Configure>", resize_scroll_region)
    canvas.bind("<Configure>", resize_actions)

    actions_open = tk.BooleanVar(value=True)
    actions_toggle = ttk.Checkbutton(root, text="Actions", variable=actions_open)
    actions_toggle.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 2))

    move_body = ttk.LabelFrame(actions_body, text="Move", padding=8)
    move_body.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    move_body.columnconfigure(0, weight=1)
    move_body.columnconfigure(1, weight=1)
    move_buttons: list[ttk.Button] = []

    drawer_body = ttk.LabelFrame(
        actions_body, text="Media-console drawers", padding=8
    )
    drawer_body.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    drawer_body.columnconfigure(0, weight=1)
    drawer_body.columnconfigure(1, weight=1)

    pick_body = ttk.LabelFrame(actions_body, text="Pick rigid object", padding=8)
    pick_body.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    pick_body.columnconfigure(0, weight=1)
    pick_body.columnconfigure(1, weight=1)
    pick_buttons: list[tuple[ttk.Button, str]] = []

    place_body = ttk.LabelFrame(actions_body, text="Place / return", padding=8)
    place_body.grid(row=3, column=0, sticky="ew", pady=(0, 8))
    place_body.columnconfigure(0, weight=1)

    tamp_body = ttk.LabelFrame(
        actions_body, text="Functional task", padding=8
    )
    tamp_body.grid(row=4, column=0, sticky="ew", pady=(0, 8))
    tamp_body.columnconfigure(0, weight=1)

    action_file_body = ttk.LabelFrame(
        actions_body, text="Grounded action file", padding=8
    )
    action_file_body.grid(row=5, column=0, sticky="ew", pady=(0, 8))
    action_file_body.columnconfigure(0, weight=1)
    action_file_body.columnconfigure(1, weight=1)

    sofa_body = ttk.LabelFrame(
        actions_body, text="Under-sofa inspection", padding=8
    )
    sofa_body.grid(row=6, column=0, sticky="ew", pady=(0, 8))
    sofa_body.columnconfigure(0, weight=1)

    remote_body = ttk.LabelFrame(actions_body, text="TV remote", padding=8)
    remote_body.grid(row=7, column=0, sticky="ew", pady=(0, 8))
    remote_body.columnconfigure(0, weight=1)

    dust_body = ttk.LabelFrame(actions_body, text="Dust TV", padding=8)
    dust_body.grid(row=8, column=0, sticky="ew", pady=(0, 8))
    dust_body.columnconfigure(0, weight=1)

    controls_body = ttk.LabelFrame(actions_body, text="Simulation", padding=8)
    controls_body.grid(row=9, column=0, sticky="ew")
    controls_body.columnconfigure(0, weight=1)

    def controllers_busy() -> bool:
        return bool(
            navigation.busy
            or manipulation.busy
            or left_drawer.busy
            or right_drawer.busy
            or remote.busy
            or sofa.busy
            or tamp.busy
            or (dusting is not None and _value(dusting, "busy", False))
        )

    def dust_failed() -> bool:
        return bool(
            dusting is not None and _value(dusting, "failure", None)
        )

    def action_configuration_safe() -> bool:
        dust_safe = not dust_failed()
        if dusting is not None and hasattr(dusting, "navigation_safe"):
            dust_safe = dust_safe and bool(
                _value(dusting, "navigation_safe", False)
            )
        return bool(
            manipulation.navigation_safe
            and left_drawer.navigation_safe
            and right_drawer.navigation_safe
            and remote.navigation_safe
            and sofa.navigation_safe
            and dust_safe
        )

    def present_error(prefix: str, error: Exception) -> None:
        nonlocal ui_error
        ui_error = f"{prefix} failed: {error}"
        status.set(ui_error)
        print(f"[Living Room Actions] {ui_error}")

    def request_move(destination: str) -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            if not action_configuration_safe():
                raise RuntimeError(
                    "The arm is not navigation-safe; reset the failed attempt"
                )
            navigation.request_move(destination)
            last_controller = navigation
            status.set(navigation.status)
        except Exception as error:
            present_error("Move", error)

    for index, destination in enumerate(LIVING_ROOM_DESTINATIONS):
        button = ttk.Button(
            move_body,
            text=MOVE_LABELS.get(destination, destination.replace("_", " ").title()),
            command=lambda name=destination: request_move(name),
        )
        button.grid(
            row=index // 2,
            column=index % 2,
            sticky="ew",
            padx=(0, 3) if index % 2 == 0 else (3, 0),
            pady=3,
        )
        move_buttons.append(button)

    def request_drawer(side: str, action: str) -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            selected_drawer = (
                left_drawer if side == "left" else right_drawer
            )
            selected_drawer.request(action, navigation.current_location)
            last_controller = selected_drawer
            status.set(selected_drawer.status)
        except Exception as error:
            present_error("Drawer", error)

    ttk.Label(drawer_body, text="Left drawer").grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    left_drawer_open_button = ttk.Button(
        drawer_body,
        text="Open left",
        command=lambda: request_drawer("left", "open"),
    )
    left_drawer_open_button.grid(
        row=1, column=0, sticky="ew", padx=(0, 3), pady=3
    )
    left_drawer_close_button = ttk.Button(
        drawer_body,
        text="Close left",
        command=lambda: request_drawer("left", "close"),
    )
    left_drawer_close_button.grid(
        row=1, column=1, sticky="ew", padx=(3, 0), pady=3
    )
    ttk.Label(drawer_body, text="Right drawer (controller storage)").grid(
        row=2, column=0, columnspan=2, sticky="w", pady=(5, 0)
    )
    right_drawer_open_button = ttk.Button(
        drawer_body,
        text="Open right",
        command=lambda: request_drawer("right", "open"),
    )
    right_drawer_open_button.grid(
        row=3, column=0, sticky="ew", padx=(0, 3), pady=3
    )
    right_drawer_close_button = ttk.Button(
        drawer_body,
        text="Close right",
        command=lambda: request_drawer("right", "close"),
    )
    right_drawer_close_button.grid(
        row=3, column=1, sticky="ew", padx=(3, 0), pady=3
    )
    ttk.Label(
        drawer_body,
        text=(
            "Move to the media console first. Store or retrieve the "
            "controller with the right drawer open."
        ),
        state="disabled",
        wraplength=430,
    ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def pick_is_calibrated(object_name: str) -> bool:
        return object_name in manipulation.calibrated_objects

    def request_pick(object_name: str) -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        kind = "calibrated" if pick_is_calibrated(object_name) else "candidate"
        spec = manipulation.all_pick_specs[object_name]
        selected_pick.set(f"Selected: {spec.label} [{kind}]")
        try:
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            if not pick_is_calibrated(object_name) and not calibration_mode:
                raise RuntimeError(
                    "Candidate picks require --calibration-mode"
                )
            manipulation.request_pick(object_name, navigation.current_location)
            last_controller = manipulation
            status.set(manipulation.status)
        except Exception as error:
            present_error("Pick", error)

    for index, (object_name, spec) in enumerate(
        manipulation.all_pick_specs.items()
    ):
        kind = "calibrated" if pick_is_calibrated(object_name) else "candidate"
        button = ttk.Button(
            pick_body,
            text=f"{spec.label}  [{kind}]",
            command=lambda name=object_name: request_pick(name),
        )
        button.grid(
            row=index // 2,
            column=index % 2,
            sticky="ew",
            padx=(0, 3) if index % 2 == 0 else (3, 0),
            pady=3,
        )
        pick_buttons.append((button, object_name))
    ttk.Label(pick_body, textvariable=selected_pick).grid(
        row=(len(pick_buttons) + 1) // 2,
        column=0,
        columnspan=2,
        sticky="w",
        pady=(6, 0),
    )

    def request_place() -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            manipulation.request_place(navigation.current_location)
            last_controller = manipulation
            status.set(manipulation.status)
        except Exception as error:
            present_error("Place / return", error)

    place_button = ttk.Button(
        place_body, text="Place held object", command=request_place
    )
    place_button.grid(row=0, column=0, sticky="ew", pady=3)
    ttk.Label(
        place_body,
        text=(
            "Book alternates between table/shelf; controller between "
            "table/drawer."
        ),
        state="disabled",
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def request_store_controller() -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            if not action_configuration_safe():
                raise RuntimeError(
                    "The robot is not in a safe state for autonomous motion"
                )
            tamp.request_store_controller()
            last_controller = tamp
            status.set(tamp.status)
        except Exception as error:
            present_error("Functional task", error)

    tamp_button = ttk.Button(
        tamp_body,
        text="Store game controller",
        command=request_store_controller,
    )
    tamp_button.grid(row=0, column=0, sticky="ew", pady=3)
    ttk.Label(
        tamp_body,
        text=(
            "The model assesses only visible storage alternatives; "
            "execution verifies the selected result."
        ),
        state="disabled",
        wraplength=430,
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def request_explicit_place(destination: str) -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if destination not in STORAGE_TARGETS:
                raise ValueError(
                    "Storage target must be one of: "
                    + ", ".join(STORAGE_TARGETS)
                )
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            if manipulation.held_object != "game_controller":
                raise RuntimeError(
                    "Explicit storage placement requires the held "
                    "game_controller"
                )
            values = STORAGE_TARGETS[destination]
            side = values.get("drawer_side")
            if side is not None:
                drawer = left_drawer if side == "left" else right_drawer
                if not drawer.is_open:
                    raise RuntimeError(f"Open the {side} drawer first")
            occupants = [
                object_id
                for object_id, object_location
                in manipulation.object_locations.items()
                if object_id != manipulation.held_object
                and semantic_storage_location(object_location)
                == destination
            ]
            if occupants:
                raise RuntimeError(
                    f"{destination} contains {', '.join(occupants)}"
                )
            manipulation.request_place_at(
                navigation.current_location,
                destination,
                str(values["place_site"]),
            )
            last_controller = manipulation
            status.set(manipulation.status)
        except Exception as error:
            present_error("Explicit place", error)

    def state_summary(ground_truth: bool) -> dict[str, object]:
        if ground_truth:
            return {
                "location": navigation.current_location,
                "held_object": manipulation.held_object,
                "object_locations": dict(
                    manipulation.object_locations
                ),
                "drawers_open": {
                    "left": left_drawer.is_open,
                    "right": right_drawer.is_open,
                },
                "under_sofa": {
                    "inspected": scene.under_sofa_inspected,
                    "remote_present": (
                        scene.scenario == "lost_remote"
                        and not scene.lost_remote_extracted
                    ),
                },
            }
        state = tamp.observer()
        return {
            "location": state.robot.location,
            "held_object": state.robot.held_object,
            "visible_objects": [
                object_id
                for object_id, observation in state.objects.items()
                if observation.visible
            ],
            "under_sofa": {
                "inspected": scene.under_sofa_inspected,
                "remote_observed": scene.lost_remote_detected,
            },
            "regions": {
                region_id: {
                    "open": region.open,
                    "occupied_by": (
                        list(region.occupied_by)
                        if region.occupied_by is not None
                        else "unknown"
                    ),
                }
                for region_id, region in state.regions.items()
            },
        }

    def execute_grounded_command(command: LivingRoomCommand) -> None:
        nonlocal ui_error
        ui_error = None
        argument = command.arguments[0] if command.arguments else None
        if command.verb == "move":
            if argument not in LIVING_ROOM_DESTINATIONS:
                raise ValueError(
                    "Destination must be one of: "
                    + ", ".join(LIVING_ROOM_DESTINATIONS)
                )
            request_move(str(argument))
        elif command.verb == "inspect":
            if argument != "sofa":
                raise ValueError("Available inspection region: sofa")
            request_sofa_inspection()
        elif command.verb in {"open", "close"}:
            if argument not in {"left", "right"}:
                raise ValueError("Drawer side must be left or right")
            request_drawer(str(argument), command.verb)
        elif command.verb == "pick":
            if argument not in manipulation.all_pick_specs:
                raise ValueError(
                    "Object must be one of: "
                    + ", ".join(manipulation.all_pick_specs)
                )
            request_pick(str(argument))
        elif command.verb == "place":
            if argument is None:
                request_place()
            else:
                request_explicit_place(argument)
        elif command.verb == "task":
            if argument != "store_game_controller":
                raise ValueError("Available task: store_game_controller")
            request_store_controller()
        elif command.verb == "state":
            mode = argument or "observed"
            if mode not in {"observed", "ground_truth"}:
                raise ValueError(
                    "State mode must be observed or ground_truth"
                )
            encoded = json.dumps(
                state_summary(mode == "ground_truth"),
                separators=(",", ":"),
                sort_keys=True,
            )
            action_output.set(encoded)
            print(f"[Living Room Action File] {encoded}")
            return
        else:
            action_output.set(COMMAND_HELP)
            return

        if ui_error is not None:
            raise RuntimeError(ui_error)
        action_output.set(ui_error or status.get())

    def start_action_script() -> None:
        nonlocal action_script_running, ui_error
        ui_error = None
        try:
            if controllers_busy() or action_script_running:
                raise RuntimeError("Wait for the active action to finish")
            commands = load_living_room_commands(action_file)
            action_queue[:] = commands
            action_script_running = True
            action_output.set(f"Loaded {len(commands)} action(s)")
            status.set(f"Loaded action file: {action_file}")
        except Exception as error:
            action_queue.clear()
            action_script_running = False
            present_error("Action file", error)
            action_output.set(str(error))

    def stop_action_script() -> None:
        nonlocal action_script_running
        action_queue.clear()
        action_script_running = False
        message = "Action queue stopped"
        if controllers_busy():
            message += "; the active action will finish safely"
        action_output.set(message)
        status.set(message)

    ttk.Label(
        action_file_body,
        textvariable=action_file_text,
        wraplength=430,
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    run_action_file_button = ttk.Button(
        action_file_body,
        text="Reload and run",
        command=start_action_script,
    )
    run_action_file_button.grid(
        row=1, column=0, sticky="ew", padx=(0, 3), pady=(6, 0)
    )
    stop_action_file_button = ttk.Button(
        action_file_body,
        text="Stop queue",
        command=stop_action_script,
    )
    stop_action_file_button.grid(
        row=1, column=1, sticky="ew", padx=(3, 0), pady=(6, 0)
    )
    ttk.Label(
        action_file_body,
        text=COMMAND_HELP,
        state="disabled",
        wraplength=430,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
    ttk.Label(
        action_file_body,
        textvariable=action_output,
        wraplength=430,
    ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

    def request_sofa_inspection() -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            sofa.request_inspect(navigation.current_location)
            last_controller = sofa
            status.set(sofa.status)
            tamp.observer()
        except Exception as error:
            present_error("Under-sofa inspection", error)

    sofa_button = ttk.Button(
        sofa_body,
        text="Inspect beneath sofa with foot cameras",
        command=request_sofa_inspection,
    )
    sofa_button.grid(row=0, column=0, sticky="ew", pady=3)
    debug_view_button = ttk.Button(
        sofa_body,
        text="Open live robot-camera view",
        command=open_debug_view,
    )
    debug_view_button.grid(row=1, column=0, sticky="ew", pady=3)
    sofa_state = ttk.Label(
        sofa_body,
        text=(
            f"Perception: {sofa_perception}. Requires Move (Couch)."
        ),
        state="disabled",
        wraplength=430,
    )
    sofa_state.grid(row=2, column=0, sticky="w", pady=(4, 0))

    def request_remote_toggle() -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            remote.request_toggle(navigation.current_location)
            last_controller = remote
            status.set(remote.status)
        except Exception as error:
            present_error("TV remote", error)

    remote_button = ttk.Button(
        remote_body, text="Aim remote and toggle TV", command=request_remote_toggle
    )
    remote_button.grid(row=0, column=0, sticky="ew", pady=3)
    ttk.Label(remote_body, textvariable=tv_state).grid(
        row=1, column=0, sticky="w", pady=(4, 0)
    )

    def request_tv_dust() -> None:
        nonlocal ui_error, last_controller
        ui_error = None
        try:
            if dusting is None:
                raise RuntimeError(dust_unavailable_reason or "TV dust unavailable")
            if controllers_busy():
                raise RuntimeError("Wait for the active action to finish")
            if manipulation.held_object != "rigid_duster":
                raise RuntimeError("Pick the rigid TV duster first")
            if navigation.current_location != "tv":
                raise RuntimeError("Move to TV before dusting")
            _request_dust(
                dusting,
                current_location=navigation.current_location,
                held_object=manipulation.held_object,
            )
            last_controller = dusting
            status.set(str(_value(dusting, "status", "TV dusting started")))
        except Exception as error:
            present_error("TV dust", error)

    dust_button = ttk.Button(
        dust_body, text="Dust the TV screen", command=request_tv_dust
    )
    dust_button.grid(row=0, column=0, sticky="ew", pady=3)
    ttk.Label(dust_body, textvariable=coverage).grid(
        row=1, column=0, sticky="w", pady=(5, 0)
    )
    ttk.Label(
        dust_body,
        text=(
            dust_unavailable_reason
            if dusting is None
            else "Requires the held rigid duster and Move (TV)."
        ),
        state="disabled",
        wraplength=430,
    ).grid(row=2, column=0, sticky="w", pady=(3, 0))

    def reset_simulation() -> None:
        nonlocal navigation, manipulation, left_drawer, right_drawer, remote
        nonlocal tamp, sofa
        nonlocal dusting, dust_unavailable_reason, ui_error, last_controller
        if controllers_busy() or action_script_running:
            ui_error = "Reset blocked: wait for the active action to finish"
            status.set(ui_error)
            return
        scene.reset()
        navigation = LivingRoomNavigationExecutor(scene)
        manipulation = LivingRoomManipulationExecutor(
            scene, calibration_mode=calibration_mode
        )
        left_drawer = MediaConsoleDrawerExecutor(scene, "left")
        right_drawer = MediaConsoleDrawerExecutor(scene, "right")
        remote = RemoteTVExecutor(scene, manipulation)
        dusting, dust_unavailable_reason = _construct_optional_duster(
            scene, manipulation
        )
        tamp.close()
        tamp = LivingRoomTampController(
            navigation, manipulation, left_drawer, right_drawer
        )
        sofa = SofaInspectionExecutor(
            scene, perception_mode=sofa_perception
        )
        ui_error = None
        last_controller = navigation
        selected_pick.set("Selected: none")
        action_queue.clear()
        action_output.set("Edit the file, then reload and run it.")
        location.set("home")
        held.set("none")
        tv_state.set("TV power: off")
        progress.set(0.0)
        status.set("Living-room simulation reset")

    reset_button = ttk.Button(
        controls_body, text="Reset living room", command=reset_simulation
    )
    reset_button.grid(row=0, column=0, sticky="ew", pady=3)
    ttk.Label(
        controls_body,
        text=(
            "Calibration mode: candidate picks are enabled."
            if calibration_mode
            else "Normal mode: candidate picks are visible but disabled."
        ),
        state="disabled",
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    footer = ttk.Frame(root, padding=(12, 7))
    footer.grid(row=2, column=0, columnspan=2, sticky="ew")
    footer.columnconfigure(0, weight=1)
    ttk.Separator(footer).grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ttk.Label(footer, text="Location:").grid(row=1, column=0, sticky="w")
    ttk.Label(footer, textvariable=location).grid(row=2, column=0, sticky="w")
    ttk.Label(footer, text="Held object:").grid(row=3, column=0, sticky="w", pady=(3, 0))
    ttk.Label(footer, textvariable=held).grid(row=4, column=0, sticky="w")
    ttk.Label(footer, textvariable=table_pose).grid(row=5, column=0, sticky="w", pady=(3, 0))
    ttk.Progressbar(footer, variable=progress, maximum=1.0).grid(
        row=6, column=0, sticky="ew", pady=6
    )
    ttk.Label(footer, textvariable=status, wraplength=450).grid(
        row=7, column=0, sticky="ew"
    )

    def toggle_actions() -> None:
        if actions_open.get():
            canvas.grid()
            scrollbar.grid()
        else:
            canvas.grid_remove()
            scrollbar.grid_remove()

    actions_toggle.configure(command=toggle_actions)

    def on_mousewheel(event) -> None:
        if actions_open.get():
            canvas.yview_scroll(int(-event.delta / 120), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)
    canvas.bind_all(
        "<Button-4>",
        lambda _event: canvas.yview_scroll(-1, "units")
        if actions_open.get()
        else None,
    )
    canvas.bind_all(
        "<Button-5>",
        lambda _event: canvas.yview_scroll(1, "units")
        if actions_open.get()
        else None,
    )

    closed = False

    if robot_debug_view:
        open_debug_view()

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")
        tamp.close()
        if debug_view is not None and not debug_view.closed:
            debug_view.close()
        if viewer.is_running():
            viewer.close()
        root.destroy()

    def active_controller() -> Any | None:
        if tamp.busy:
            return tamp
        if sofa.busy:
            return sofa
        if dusting is not None and _value(dusting, "busy", False):
            return dusting
        if remote.busy:
            return remote
        if left_drawer.busy:
            return left_drawer
        if right_drawer.busy:
            return right_drawer
        if manipulation.busy:
            return manipulation
        if navigation.busy:
            return navigation
        return None

    def advance_action_script() -> None:
        nonlocal action_script_running
        if not action_script_running or controllers_busy():
            return
        if not action_queue:
            action_script_running = False
            action_output.set("Action file complete")
            status.set("Action file complete")
            return

        command = action_queue.pop(0)
        command_text = " ".join((command.verb, *command.arguments))
        action_output.set(
            f"Running: {command_text} "
            f"({len(action_queue)} remaining)"
        )
        try:
            execute_grounded_command(command)
        except Exception as error:
            action_queue.clear()
            action_script_running = False
            present_error("Action file", error)
            action_output.set(f"Stopped at '{command_text}': {error}")

    def tick() -> None:
        if closed:
            return
        if not viewer.is_running():
            close()
            return
        for _ in range(5):
            active = active_controller()
            if active is not None:
                active.update()
                if not bool(_value(active, "busy", False)):
                    # Preserve observations made by direct/manual actions too.
                    tamp.observer()
            mujoco.mj_step(scene.model, scene.data)

        advance_action_script()
        active = active_controller()
        displayed_controller = active or last_controller
        active_status = ui_error or str(
            _value(displayed_controller, "status", navigation.status)
        )
        if hasattr(viewer, "set_texts"):
            viewer.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Living Room Actions",
                    active_status,
                )
            )
        viewer.sync()
        if debug_view is not None and not debug_view.closed:
            debug_view.refresh(sofa.last_evidence, sofa.status)

        location.set(navigation.current_location)
        held.set(manipulation.held_object or "none")
        x, y, yaw = scene.table_pose
        table_pose.set(
            f"Fixed table: x={x:.2f} m, y={y:.2f} m, "
            f"yaw={math.degrees(yaw):.1f} deg"
        )
        cleaned = len(scene.cleaned_cells)
        coverage.set(
            f"TV coverage: {cleaned}/{TV_CELL_COUNT} "
            f"({scene.dust_coverage * 100:.0f}%)"
        )
        tv_state.set(f"TV power: {'on' if scene.tv_power_on else 'off'}")
        status.set(active_status)
        progress_value = _value(displayed_controller, "progress", 0.0)
        progress.set(float(progress_value or 0.0))

        idle = not controllers_busy() and not action_script_running
        safe = action_configuration_safe()
        move_state = "normal" if idle and safe else "disabled"
        for button in move_buttons:
            button.configure(state=move_state)

        at_left_drawer = navigation.current_location in {
            "drawer",
            "drawer_left",
        }
        at_right_drawer = navigation.current_location in {
            "drawer",
            "drawer_right",
        }
        left_drawer_open_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and at_left_drawer
                and not left_drawer.is_open
                else "disabled"
            )
        )
        left_drawer_close_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and at_left_drawer
                and left_drawer.is_open
                and manipulation.held_object is None
                else "disabled"
            )
        )
        right_drawer_open_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and at_right_drawer
                and not right_drawer.is_open
                else "disabled"
            )
        )
        right_drawer_close_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and at_right_drawer
                and right_drawer.is_open
                and manipulation.held_object is None
                else "disabled"
            )
        )

        for button, object_name in pick_buttons:
            required_location = manipulation.required_pick_location(object_name)
            allowed_kind = pick_is_calibrated(object_name) or calibration_mode
            storage_accessible = (
                required_location
                not in {"drawer", "drawer_left", "drawer_right"}
                or (
                    required_location in {"drawer", "drawer_right"}
                    and right_drawer.is_open
                )
                or (
                    required_location == "drawer_left"
                    and left_drawer.is_open
                )
            )
            can_pick = (
                idle
                and safe
                and manipulation.held_object is None
                and navigation.current_location == required_location
                and allowed_kind
                and storage_accessible
            )
            button.configure(state="normal" if can_pick else "disabled")

        held_object = manipulation.held_object
        place_location = manipulation.place_destination
        storage_accessible = not (
            place_location == "drawer" and not right_drawer.is_open
        )
        can_place = (
            idle
            and manipulation.can_place
            and held_object is not None
            and navigation.current_location == place_location
            and storage_accessible
        )
        place_button.configure(
            text=manipulation.place_label,
            state="normal" if can_place else "disabled",
        )
        tamp_button.configure(
            state="normal" if idle and safe else "disabled"
        )
        sofa_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and scene.scenario == "lost_remote"
                and navigation.current_location == "couch"
                else "disabled"
            )
        )
        run_action_file_button.configure(
            state=(
                "normal"
                if not controllers_busy() and not action_script_running
                else "disabled"
            )
        )
        stop_action_file_button.configure(
            state="normal" if action_script_running else "disabled"
        )

        remote_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and manipulation.held_object == "remote_control"
                and navigation.current_location == "tv"
                else "disabled"
            )
        )

        can_dust = (
            dusting is not None
            and idle
            and not dust_failed()
            and manipulation.held_object == "rigid_duster"
            and navigation.current_location == "tv"
            and scene.dust_coverage < 1.0
        )
        dust_button.configure(state="normal" if can_dust else "disabled")
        reset_button.configure(state="normal" if idle else "disabled")
        root.after(10, tick)

    root.protocol("WM_DELETE_WINDOW", close)
    tick()
    root.mainloop()


__all__ = ["launch_living_room_actions"]
