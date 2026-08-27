"""Five-view Workshop execution mosaic recorder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .live_mosaic_viewer import LiveMosaicViewer
from .workshop_scene import WORKSHOP_CAMERAS


class WorkshopRecorder:
    def __init__(
        self, scene, output: Path | None, *, width: int = 640,
        height: int = 360, fps: int = 20, show: bool = False,
    ):
        self.scene = scene
        self.output = output
        self.width, self.height, self.fps = width, height, fps
        self.renderer = mujoco.Renderer(scene.model, height=height, width=width)
        self.writer = None
        self.show = bool(show)
        self.window_name = "Workshop GT execution — five views"
        self.live_viewer = (
            LiveMosaicViewer(width * 3, height * 2, fps, self.window_name)
            if self.show else None
        )
        self.frames = 0
        self.frame_interval_sim = 1.0 / float(fps)
        self.last_capture_sim_time = -1.0
        self.telemetry: dict[str, Any] = {"variant": scene.variant_name, "operator": "INITIAL", "index": 0, "total": 0, "status": "RUNNING"}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width * 3, height * 2))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer {output}")

    def _panel(self) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), (18, 22, 30))
        draw = ImageDraw.Draw(image)
        lines = [
            "WORKSHOP GT EXECUTION",
            str(self.telemetry.get("variant", "")),
            f"{self.telemetry.get('index', 0)}/{self.telemetry.get('total', 0)}  {self.telemetry.get('operator', '')}",
            str(self.telemetry.get("arguments", ""))[:44],
            str(self.telemetry.get("status", "RUNNING")),
            "ROBOT-ACTUATED GT",
            "Contact-gated grasp + measured task state",
        ]
        y = 14
        for index, line in enumerate(lines):
            draw.text((12, y), line, fill=(88, 166, 255) if index == 0 else (235, 240, 246))
            y += 23
        return np.asarray(image)

    def _detail_camera(self) -> str:
        operator = str(self.telemetry.get("operator", ""))
        arguments = [str(value) for value in self.telemetry.get("arguments", [])]
        joined = " ".join(arguments)
        if "LEFT_DRAWER" in joined:
            return "workshop_camera_left_drawer_detail"
        if "RIGHT_DRAWER" in joined:
            return "workshop_camera_right_drawer_detail"
        if "TOOL_CABINET" in joined:
            return "workshop_camera_cabinet_detail"
        if operator in {"PLACE", "SCREW"} or "workshop_frame_joint" in joined:
            return "workshop_camera_repair_detail"
        return "workshop_camera_close"

    def capture(self, force: bool = True) -> None:
        if self.writer is None and not self.show:
            return
        current_sim_time = float(self.scene.data.time)
        if not force and self.last_capture_sim_time >= 0.0:
            if current_sim_time - self.last_capture_sim_time < self.frame_interval_sim - 1e-5:
                return
        mujoco.mj_forward(self.scene.model, self.scene.data)
        tiles = []
        for camera in WORKSHOP_CAMERAS:
            rendered_camera = self._detail_camera() if camera == "workshop_camera_close" else camera
            self.renderer.update_scene(self.scene.data, camera=rendered_camera)
            tile = self.renderer.render().copy()
            label = "DETAIL" if camera == "workshop_camera_close" else camera.replace("workshop_camera_", "").upper()
            cv2.putText(tile, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        tiles.append(self._panel())
        mosaic = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
        mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        if self.writer is not None:
            self.writer.write(mosaic_bgr)
        if self.live_viewer is not None:
            self.live_viewer.show(mosaic)
        self.frames += 1
        self.last_capture_sim_time = current_sim_time

    def close(self, hold_frames: int = 10) -> None:
        if self.writer:
            for _ in range(hold_frames):
                self.capture(True)
            self.writer.release()
        if self.live_viewer is not None:
            self.live_viewer.close()
            self.live_viewer = None
        self.renderer.close()
