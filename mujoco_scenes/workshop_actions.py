"""Interactive MuJoCo viewer and action panel for the Workshop benchmark."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Callable

import mujoco
import mujoco.viewer

from .workshop_ground_truth_execution import WorkshopExecutionDispatcher
from .workshop_ground_truth_planner import (
    COMPATIBLE_DRIVERS,
    COMPATIBLE_SCREW,
    generate_gt_plan,
    load_variant_specs,
    solve_gt_assignment,
)
from .workshop_ground_truth_state import initial_workshop_state
from .workshop_scene import WORKSHOP_CAMERAS, WORKSHOP_REGIONS


OBJECT_LABELS = {
    "workshop_long_phillips_driver": "Manual Phillips driver",
    "workshop_power_driver": "Power driver",
    "workshop_medium_phillips_screw": "Phillips screw",
    "workshop_wooden_hammer": "Hammer distractor",
}


def _action(operator: str, *arguments: str) -> dict[str, Any]:
    return {"operator": operator, "arguments": list(arguments)}


def launch_workshop_action_viewer(scene, camera: str = "free") -> None:
    """Launch the Workshop viewer with selectable, preconditioned actions."""
    import tkinter as tk
    from tkinter import ttk

    if scene.robot_name != "google":
        raise RuntimeError("The Workshop Actions panel requires --robot google")
    if camera != "free" and camera not in WORKSHOP_CAMERAS:
        raise ValueError(f"Unknown Workshop camera: {camera}")

    spec = load_variant_specs()[scene.variant_name]
    gt_assignment = solve_gt_assignment(scene.variant_name)
    assignment = gt_assignment
    world = initial_workshop_state(spec["storage_contents"])
    gt_plan = generate_gt_plan(gt_assignment)
    gt_index = 0
    action_count = 0
    busy = False
    closed = False
    last_sync = 0.0
    initial_eq_data = scene.model.eq_data.copy()

    viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
    if camera == "free":
        mujoco.mjv_defaultFreeCamera(scene.model, viewer.cam)
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )

    root = tk.Tk()
    root.title(f"Workshop Actions — {scene.variant_name}")
    root.geometry("620x940+20+20")
    root.minsize(540, 620)

    canvas = tk.Canvas(root, highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    body = ttk.Frame(canvas, padding=12)
    body.bind(
        "<Configure>",
        lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=body, anchor="nw", width=580)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    body.columnconfigure(0, weight=1)

    status = tk.StringVar(value="Ready — choose an action")
    location = tk.StringVar()
    held = tk.StringVar()
    storage = tk.StringVar()
    inspected = tk.StringVar()
    selection = tk.StringVar()
    gt_progress = tk.StringVar()

    ttk.Label(body, text="Workshop Actions", font=("TkDefaultFont", 15, "bold")).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(body, text=scene.variant_name).grid(row=1, column=0, sticky="w")
    ttk.Label(body, textvariable=status, wraplength=550).grid(
        row=2, column=0, sticky="ew", pady=(5, 8)
    )
    state_box = ttk.LabelFrame(body, text="Live symbolic state", padding=8)
    state_box.grid(row=3, column=0, sticky="ew", pady=4)
    for row, variable in enumerate((location, held, storage, inspected, selection, gt_progress)):
        ttk.Label(state_box, textvariable=variable, wraplength=530).grid(
            row=row, column=0, sticky="w"
        )

    log = tk.Text(body, height=8, wrap="word", state="disabled")
    log.grid(row=11, column=0, sticky="ew", pady=(8, 4))
    action_buttons: list[tuple[ttk.Button, Callable[[], dict[str, Any]]]] = []

    def append_log(message: str) -> None:
        log.configure(state="normal")
        log.insert("end", message.rstrip() + "\n")
        log.see("end")
        log.configure(state="disabled")

    def refresh_state() -> None:
        location.set("Navigation: implicit in the selected generic action")
        held.set(f"Holding: {world.held_object or 'nothing'}")
        storage.set(
            "Storage: " + ", ".join(
                f"{name}={'open' if world.storage_open[name] else 'closed'}"
                for name in WORKSHOP_REGIONS
            )
        )
        inspected.set(
            "Inspected: " + (", ".join(sorted(world.inspected_storage)) or "none")
        )
        selected_name = assignment.driver or "none"
        selection.set(
            f"Drive selection: {OBJECT_LABELS.get(selected_name, selected_name)}"
            + (" (manual override)" if assignment != gt_assignment else " (GT)")
        )
        gt_progress.set(f"GT sequence: {gt_index}/{len(gt_plan)}")
        for button, factory in action_buttons:
            try:
                candidate = factory()
                check_assignment = assignment
                if candidate["operator"] == "SCREW" and world.held_object in COMPATIBLE_DRIVERS:
                    check_assignment = replace(
                        assignment,
                        driver=world.held_object,
                        fastener=COMPATIBLE_SCREW,
                        is_feasible=True,
                        intended_outcome="FEASIBLE",
                    )
                valid, _ = world.check(candidate, check_assignment)
            except Exception:
                valid = False
            button.configure(state="normal" if valid and not busy else "disabled")

    def frame_callback(force: bool) -> None:
        nonlocal last_sync
        now = time.monotonic()
        if force or now - last_sync >= 1.0 / 30.0:
            if viewer.is_running():
                viewer.sync()
            root.update_idletasks()
            last_sync = now

    dispatcher = WorkshopExecutionDispatcher(
        scene, assignment, frame_callback=frame_callback
    )

    def execute(candidate: dict[str, Any], *, from_gt: bool = False) -> bool:
        nonlocal assignment, dispatcher, busy, action_count, gt_index
        if busy:
            return False
        if candidate["operator"] == "SCREW" and world.held_object in COMPATIBLE_DRIVERS:
            # Interactive exploration may deliberately choose the other valid
            # driver. The benchmark GT assignment itself remains unchanged.
            assignment = replace(
                assignment,
                driver=world.held_object,
                fastener=COMPATIBLE_SCREW,
                is_feasible=True,
                intended_outcome="FEASIBLE",
                rejection_reason=None,
                assignment_source="INTERACTIVE_DRIVER_OVERRIDE",
            )
            dispatcher.assignment = assignment
        valid, reason = world.check(candidate, assignment)
        if not valid:
            status.set(f"Blocked: {reason}")
            append_log(f"BLOCKED {candidate['operator']} {candidate['arguments']}: {reason}")
            refresh_state()
            return False
        busy = True
        refresh_state()
        status.set(f"Executing {candidate['operator']} {candidate['arguments']} …")
        root.update_idletasks()
        try:
            result = dispatcher.execute(candidate, world)
            if not result.get("success"):
                raise RuntimeError(result.get("detail") or result.get("status") or "action failed")
            world.apply(candidate)
            action_count += 1
            if from_gt:
                gt_index += 1
            status.set(f"Completed {candidate['operator']} {candidate['arguments']}")
            append_log(f"{action_count:03d} OK  {candidate['operator']} {candidate['arguments']}")
            return True
        except Exception as error:
            status.set(f"Failed: {error}")
            append_log(f"FAIL {candidate['operator']} {candidate['arguments']}: {error}")
            return False
        finally:
            busy = False
            refresh_state()
            frame_callback(True)

    def add_button(parent, text: str, factory: Callable[[], dict[str, Any]], row: int, column: int = 0) -> None:
        button = ttk.Button(parent, text=text, command=lambda: execute(factory()))
        button.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        action_buttons.append((button, factory))

    storage_box = ttk.LabelFrame(body, text="Open once to inspect", padding=8)
    storage_box.grid(row=4, column=0, sticky="ew", pady=4)
    storage_box.columnconfigure((0, 1), weight=1)
    for row, region in enumerate(WORKSHOP_REGIONS):
        ttk.Label(storage_box, text=region.replace("_", " ").title()).grid(row=row, column=0, sticky="w")
        add_button(storage_box, "Open", lambda value=region: _action("OPEN", value), row, 1)

    pick_box = ttk.LabelFrame(body, text="Pick object from its current location", padding=8)
    pick_box.grid(row=5, column=0, sticky="ew", pady=4)
    pick_box.columnconfigure((0, 1), weight=1)
    for index, (object_name, label) in enumerate(OBJECT_LABELS.items()):
        factory = lambda value=object_name: _action(
            "PICK", value, world.object_locations.get(value, "ABSENT")
        )
        add_button(pick_box, label, factory, index // 2, index % 2)

    task_box = ttk.LabelFrame(body, text="Place / fasten", padding=8)
    task_box.grid(row=6, column=0, sticky="ew", pady=4)
    task_box.columnconfigure((0, 1), weight=1)
    add_button(
        task_box,
        "Place held object on workbench",
        lambda: _action("PLACE", world.held_object or "NONE", "MAIN_WORKBENCH_ZONE"),
        0,
        0,
    )
    add_button(
        task_box,
        "Place held screw at frame joint",
        lambda: _action("PLACE", world.held_object or "NONE", "workshop_frame_joint"),
        0,
        1,
    )
    add_button(
        task_box,
        "Screw inserted fastener",
        lambda: _action(
            "SCREW",
            world.held_object or "NONE",
            COMPATIBLE_SCREW,
            "workshop_frame_joint",
        ),
        1,
        0,
    )
    sequence_box = ttk.LabelFrame(body, text="Ground-truth sequence", padding=8)
    sequence_box.grid(row=7, column=0, sticky="ew", pady=4)
    sequence_box.columnconfigure((0, 1), weight=1)

    def next_gt_action() -> None:
        if gt_index >= len(gt_plan):
            status.set("GT sequence is complete")
            return
        execute(gt_plan[gt_index], from_gt=True)

    ttk.Button(sequence_box, text="Execute next GT action", command=next_gt_action).grid(
        row=0, column=0, sticky="ew", padx=3, pady=3
    )

    def run_remaining_gt() -> None:
        while gt_index < len(gt_plan) and viewer.is_running():
            if not execute(gt_plan[gt_index], from_gt=True):
                break

    ttk.Button(sequence_box, text="Run remaining GT sequence", command=run_remaining_gt).grid(
        row=0, column=1, sticky="ew", padx=3, pady=3
    )

    controls = ttk.Frame(body)
    controls.grid(row=8, column=0, sticky="ew", pady=4)
    controls.columnconfigure((0, 1), weight=1)

    def reset() -> None:
        nonlocal assignment, dispatcher, world, gt_index, action_count, busy
        if busy:
            return
        scene.model.eq_data[:] = initial_eq_data
        # Use WorkshopScene's reset contract so both kitchen-equivalent
        # drawers are explicitly restored closed before the action panel
        # presents the first OPEN action.
        scene.reset()
        assignment = gt_assignment
        world = initial_workshop_state(spec["storage_contents"])
        dispatcher = WorkshopExecutionDispatcher(
            scene, assignment, frame_callback=frame_callback
        )
        gt_index = 0
        action_count = 0
        status.set("Simulation reset")
        append_log("--- RESET ---")
        refresh_state()
        frame_callback(True)

    ttk.Button(controls, text="Reset variant", command=reset).grid(
        row=0, column=0, sticky="ew", padx=3
    )

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        if viewer.is_running():
            viewer.close()
        root.destroy()

    ttk.Button(controls, text="Close", command=close).grid(
        row=0, column=1, sticky="ew", padx=3
    )

    def tick() -> None:
        if closed:
            return
        if not viewer.is_running():
            close()
            return
        if not busy:
            for _ in range(3):
                mujoco.mj_step(scene.model, scene.data)
            viewer.sync()
            refresh_state()
        root.after(20, tick)

    root.protocol("WM_DELETE_WINDOW", close)
    refresh_state()
    tick()
    root.mainloop()


__all__ = ["launch_workshop_action_viewer"]
