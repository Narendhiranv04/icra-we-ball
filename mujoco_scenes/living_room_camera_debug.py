"""Live Google Robot camera viewer for living-room perception debugging."""

from __future__ import annotations

import time
from typing import Sequence

import mujoco
import numpy as np
from PIL import Image, ImageTk

from mujoco_scenes.living_room_cameras import ROBOT_DEBUG_CAMERAS
from mujoco_scenes.living_room_sofa import SofaCameraEvidence


PREVIEW_SIZE = (240, 180)
REFRESH_SECONDS = 0.25


def mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB image with detected pixels highlighted in green."""
    overlay = rgb.astype(np.float32).copy()
    overlay[mask] = (
        0.55 * overlay[mask] + 0.45 * np.array((50.0, 220.0, 90.0))
    )
    return overlay.astype(np.uint8)


class RobotCameraDebugView:
    """A throttled Tk window showing the two low and five top camera feeds."""

    def __init__(self, parent, scene) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._scene = scene
        self._renderer = mujoco.Renderer(
            scene.model,
            height=PREVIEW_SIZE[1],
            width=PREVIEW_SIZE[0],
        )
        self._camera_ids = {
            name: mujoco.mj_name2id(
                scene.model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            for name in ROBOT_DEBUG_CAMERAS
        }
        if any(camera_id < 0 for camera_id in self._camera_ids.values()):
            self._renderer.close()
            raise RuntimeError("The Google Robot debug camera rig is unavailable")

        self._window = tk.Toplevel(parent)
        self._window.title("Google Robot Camera Debug View")
        self._window.geometry("1000x470+420+20")
        self._window.resizable(False, False)
        self._window.protocol("WM_DELETE_WINDOW", self.close)
        for column in range(4):
            self._window.columnconfigure(column, weight=1)

        self._show_overlay = tk.BooleanVar(value=False)
        self._status = tk.StringVar(value="Live RGB - no inspection captured")
        self._labels = {}
        self._photos = {}
        for index, camera_name in enumerate(ROBOT_DEBUG_CAMERAS):
            panel_row, column = divmod(index, 4)
            label_row = panel_row * 2
            ttk.Label(self._window, text=camera_name).grid(
                row=label_row, column=column, pady=(8, 3)
            )
            label = ttk.Label(self._window)
            label.grid(row=label_row + 1, column=column, padx=6)
            self._labels[camera_name] = label

        ttk.Checkbutton(
            self._window,
            text="Show latest inspection masks",
            variable=self._show_overlay,
        ).grid(row=4, column=0, sticky="w", padx=8, pady=(7, 2))
        ttk.Label(self._window, textvariable=self._status).grid(
            row=4, column=1, columnspan=3, sticky="e", padx=8, pady=(7, 2)
        )
        self._last_refresh = 0.0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def focus(self) -> None:
        if not self._closed:
            self._window.deiconify()
            self._window.lift()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._renderer.close()
        self._window.destroy()

    def refresh(
        self,
        evidence: Sequence[SofaCameraEvidence],
        inspection_status: str,
    ) -> None:
        """Refresh the live feeds or display the latest captured overlays."""
        if self._closed:
            return
        now = time.monotonic()
        if now - self._last_refresh < REFRESH_SECONDS:
            return
        self._last_refresh = now

        evidence_by_camera = {item.camera_id: item for item in evidence}
        show_overlay = self._show_overlay.get() and bool(evidence_by_camera)
        mujoco.mj_forward(self._scene.model, self._scene.data)
        frames = {}
        for name, camera_id in self._camera_ids.items():
            self._renderer.update_scene(self._scene.data, camera=camera_id)
            frames[name] = self._renderer.render().copy()
        if show_overlay:
            frames.update({
                name: mask_overlay(item.rgb, item.mask)
                for name, item in evidence_by_camera.items()
            })
            self._status.set(
                f"Latest ground masks + live top RGB - {inspection_status}"
            )
        else:
            self._status.set("Live RGB")

        for name in ROBOT_DEBUG_CAMERAS:
            frame = frames.get(name)
            if frame is None:
                continue
            image = Image.fromarray(frame.astype(np.uint8)).resize(PREVIEW_SIZE)
            photo = ImageTk.PhotoImage(image=image, master=self._window)
            self._labels[name].configure(image=photo)
            self._photos[name] = photo


__all__ = ["RobotCameraDebugView", "mask_overlay"]
