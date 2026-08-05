"""Companion Actions panel for the rigid Google Robot living room."""

from __future__ import annotations

import importlib
import inspect
import math
from typing import Any

import mujoco
import mujoco.viewer

from mujoco_scenes.living_room_manipulation import (
    LivingRoomManipulationExecutor,
)
from mujoco_scenes.living_room_drawer import MediaConsoleDrawerExecutor
from mujoco_scenes.living_room_navigation import (
    LIVING_ROOM_DESTINATIONS,
    LivingRoomNavigationExecutor,
)
from mujoco_scenes.living_room_scene import FREE_CAMERA, TV_CELL_COUNT
from mujoco_scenes.living_room_remote import RemoteTVExecutor


MOVE_LABELS = {
    "home": "Home / coffee table",
    "table_south": "Coffee table - south side",
    "table_north": "Coffee table - north side",
    "table_east": "Coffee table - east side",
    "table_west": "Coffee table - west side",
    "bookshelf": "Book shelf",
    "drawer": "Media-console drawer",
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

    status = tk.StringVar(value=navigation.status)
    location = tk.StringVar(value=navigation.current_location)
    held = tk.StringVar(value="none")
    table_pose = tk.StringVar(value="")
    coverage = tk.StringVar(value=f"TV coverage: 0/{TV_CELL_COUNT} (0%)")
    tv_state = tk.StringVar(value="TV power: off")
    progress = tk.DoubleVar(value=0.0)
    selected_pick = tk.StringVar(value="Selected: none")
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

    remote_body = ttk.LabelFrame(actions_body, text="TV remote", padding=8)
    remote_body.grid(row=4, column=0, sticky="ew", pady=(0, 8))
    remote_body.columnconfigure(0, weight=1)

    dust_body = ttk.LabelFrame(actions_body, text="Dust TV", padding=8)
    dust_body.grid(row=5, column=0, sticky="ew", pady=(0, 8))
    dust_body.columnconfigure(0, weight=1)

    controls_body = ttk.LabelFrame(actions_body, text="Simulation", padding=8)
    controls_body.grid(row=6, column=0, sticky="ew")
    controls_body.columnconfigure(0, weight=1)

    def controllers_busy() -> bool:
        return bool(
            navigation.busy
            or manipulation.busy
            or left_drawer.busy
            or right_drawer.busy
            or remote.busy
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
        nonlocal dusting, dust_unavailable_reason, ui_error, last_controller
        if controllers_busy():
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
        ui_error = None
        last_controller = navigation
        selected_pick.set("Selected: none")
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

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")
        if viewer.is_running():
            viewer.close()
        root.destroy()

    def active_controller() -> Any | None:
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
            mujoco.mj_step(scene.model, scene.data)

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

        idle = not controllers_busy()
        safe = action_configuration_safe()
        move_state = "normal" if idle and safe else "disabled"
        for button in move_buttons:
            button.configure(state=move_state)

        at_drawer = navigation.current_location == "drawer"
        left_drawer_open_button.configure(
            state=(
                "normal"
                if idle and safe and at_drawer and not left_drawer.is_open
                else "disabled"
            )
        )
        left_drawer_close_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and at_drawer
                and left_drawer.is_open
                and manipulation.held_object is None
                else "disabled"
            )
        )
        right_drawer_open_button.configure(
            state=(
                "normal"
                if idle and safe and at_drawer and not right_drawer.is_open
                else "disabled"
            )
        )
        right_drawer_close_button.configure(
            state=(
                "normal"
                if idle
                and safe
                and at_drawer
                and right_drawer.is_open
                and manipulation.held_object is None
                else "disabled"
            )
        )

        for button, object_name in pick_buttons:
            required_location = manipulation.required_pick_location(object_name)
            allowed_kind = pick_is_calibrated(object_name) or calibration_mode
            storage_accessible = not (
                required_location == "drawer" and not right_drawer.is_open
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
