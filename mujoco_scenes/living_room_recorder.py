"""Synchronized five-camera renderer, mosaic frame compositor, live viewer, and MP4 recorder for L2 Living Room.

This module renders all five existing living-room project cameras at synchronized simulation
timestamps and composes them into a 3x2 mosaic with an interactive status panel on the 6th tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .live_mosaic_viewer import LiveMosaicViewer


L2_FIVE_CAMERAS = (
    "l2_camera_left",
    "l2_camera_right",
    "l2_camera_top",
    "l2_camera_front",
    "l2_camera_close",
)


@dataclass
class LivingRoomRecorderTelemetry:
    """Live execution metrics displayed on the status panel."""

    variant_id: str = ""
    scene_name: str = ""
    intended_outcome: str = "FEASIBLE"
    current_action_index: int = 0
    total_actions: int = 0
    current_operator: str = "INITIALIZING"
    current_arguments: list[str] = field(default_factory=list)
    current_reason: str = ""
    held_object: str | None = None
    high_level_phase: str = "STARTUP"
    execution_status: str = "RUNNING"
    infeasible_reason: str | None = None


class LivingRoomRecorder:
    """Renders 5 living-room project cameras into 1 mosaic frame with live display and streaming MP4 writing."""

    def __init__(
        self,
        scene,
        output_path: Path | str | None = None,
        *,
        tile_width: int = 640,
        tile_height: int = 360,
        fps: int = 20,
        show: bool = False,
        record: bool = False,
        no_overlay: bool = False,
    ):
        self.scene = scene
        self.output_path = Path(output_path) if output_path else None
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.mosaic_width = self.tile_width * 3
        self.mosaic_height = self.tile_height * 2
        self.fps = int(fps)
        self.frame_interval_sim = 1.0 / max(1, self.fps)
        self.show = bool(show)
        self.record = bool(record)
        self.no_overlay = bool(no_overlay)

        self.telemetry = LivingRoomRecorderTelemetry()
        self.last_capture_sim_time = -1.0
        self.total_frames_captured = 0
        self.wall_start_time = time.perf_counter()
        self.aborted_by_user = False
        self.live_viewer = (
            LiveMosaicViewer(
                self.mosaic_width,
                self.mosaic_height,
                self.fps,
                "Living Room Mobile Execution (5 Cameras)",
            )
            if self.show else None
        )

        # Initialize MuJoCo renderer for camera tile size
        self.renderer = mujoco.Renderer(
            scene.model, height=self.tile_height, width=self.tile_width
        )

        # Setup video writer if recording
        self.video_writer = None
        self.imageio_writer = None
        if self.record and self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_video_writer()

    def _init_video_writer(self) -> None:
        """Initialize streaming video writer with fallbacks."""
        out_str = str(self.output_path)
        try:
            import imageio.v2 as imageio
            self.imageio_writer = imageio.get_writer(
                out_str,
                fps=self.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
            )
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                out_str, fourcc, float(self.fps), (self.mosaic_width, self.mosaic_height)
            )
            if self.video_writer is not None and not self.video_writer.isOpened():
                self.video_writer = None

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        text: str,
        fill: tuple[int, int, int] = (255, 255, 255),
        size: int = 16,
        bold: bool = False,
    ) -> None:
        """Draw text using DejaVu font or default fallback."""
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        draw.text(xy, text, fill=fill, font=font)

    def _render_status_panel(self) -> np.ndarray:
        """Generate the status information panel for Tile 6."""
        img = Image.new("RGB", (self.tile_width, self.tile_height), color=(18, 22, 30))
        draw = ImageDraw.Draw(img)

        # Header background
        draw.rectangle([(0, 0), (self.tile_width, 42)], fill=(28, 35, 48))
        self._draw_text(draw, (14, 12), "LIVING_ROOM_EXECUTION", fill=(88, 166, 255), size=17, bold=True)

        # Status badge
        status_text = self.telemetry.execution_status
        status_color = (
            (46, 160, 67) if "SUCCESS" in status_text or "COMPLETE" in status_text or "CONFIRMED" in status_text
            else (218, 54, 51) if "FAIL" in status_text or "ABORT" in status_text or "INFEASIBLE" in status_text or "MISMATCH" in status_text
            else (56, 139, 253)
        )
        draw.rounded_rectangle([(self.tile_width - 145, 8), (self.tile_width - 14, 34)], radius=4, fill=status_color)
        self._draw_text(draw, (self.tile_width - 138, 13), status_text[:16], fill=(255, 255, 255), size=12, bold=True)

        # Metrics rows
        y = 52
        dy = 23

        # Variant & Intended Outcome
        self._draw_text(draw, (14, y), "Variant:", fill=(139, 148, 158), size=13, bold=True)
        self._draw_text(draw, (90, y), f"{self.telemetry.variant_id}", fill=(240, 246, 252), size=13, bold=True)
        y += dy

        self._draw_text(draw, (14, y), "Intended:", fill=(139, 148, 158), size=13)
        outcome_color = (46, 160, 67) if self.telemetry.intended_outcome == "FEASIBLE" else (210, 153, 34)
        self._draw_text(draw, (90, y), f"{self.telemetry.intended_outcome}", fill=outcome_color, size=13, bold=True)
        y += dy

        # Phase
        self._draw_text(draw, (14, y), "Phase:", fill=(139, 148, 158), size=13)
        self._draw_text(draw, (90, y), f"{self.telemetry.high_level_phase}", fill=(210, 168, 255), size=13, bold=True)
        y += dy

        # Current Action
        total = self.telemetry.total_actions or 1
        idx = self.telemetry.current_action_index
        args_str = ", ".join(self.telemetry.current_arguments)
        action_str = f"{self.telemetry.current_operator}({args_str})" if args_str else self.telemetry.current_operator
        self._draw_text(draw, (14, y), f"Action [{idx}/{total}]:", fill=(139, 148, 158), size=13)
        y += dy
        self._draw_text(draw, (24, y), action_str[:42], fill=(255, 215, 0), size=13, bold=True)
        y += dy

        # Held Object
        held_str = str(self.telemetry.held_object) if self.telemetry.held_object else "None (Hand Empty)"
        self._draw_text(draw, (14, y), "Held Object:", fill=(139, 148, 158), size=13)
        self._draw_text(draw, (110, y), held_str[:30], fill=(126, 231, 135) if self.telemetry.held_object else (139, 148, 158), size=13)
        y += dy

        # Infeasible Reason if applicable
        if self.telemetry.infeasible_reason:
            self._draw_text(draw, (14, y), "Infeasible Reason:", fill=(248, 81, 73), size=12, bold=True)
            y += 18
            self._draw_text(draw, (24, y), self.telemetry.infeasible_reason[:42], fill=(248, 81, 73), size=12)
            y += 20

        # Progress bar
        bar_x1, bar_x2 = 14, self.tile_width - 14
        bar_y = self.tile_height - 48
        bar_w = bar_x2 - bar_x1
        progress = min(1.0, max(0.0, float(idx) / float(total)))
        draw.rounded_rectangle([(bar_x1, bar_y), (bar_x2, bar_y + 7)], radius=3, fill=(48, 54, 61))
        if progress > 0.0:
            draw.rounded_rectangle([(bar_x1, bar_y), (bar_x1 + int(bar_w * progress), bar_y + 7)], radius=3, fill=(46, 160, 67))

        # Time footer
        sim_time = float(self.scene.data.time)
        wall_time = time.perf_counter() - self.wall_start_time
        time_str = f"Sim: {sim_time:6.2f}s  |  Wall: {wall_time:5.1f}s  |  {self.fps} fps"
        self._draw_text(draw, (14, self.tile_height - 28), time_str, fill=(139, 148, 158), size=12)

        return np.asarray(img, dtype=np.uint8)

    def _render_camera(self, camera_name: str) -> np.ndarray:
        """Render a single project camera view."""
        cam_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            frame = np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
        else:
            self.renderer.update_scene(self.scene.data, camera=cam_id)
            frame = self.renderer.render().copy()

        if not self.no_overlay:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            label = f"  {camera_name}  "
            draw.rounded_rectangle([(10, 10), (10 + len(label) * 8 + 10, 30)], radius=4, fill=(0, 0, 0, 180))
            self._draw_text(draw, (14, 12), label, fill=(240, 246, 252), size=12, bold=True)
            frame = np.asarray(img, dtype=np.uint8)

        return frame

    def capture_frame(self, force: bool = False) -> np.ndarray | None:
        """Capture and compose all 5 cameras + status panel into one mosaic frame."""
        current_sim_time = float(self.scene.data.time)
        if not force and self.last_capture_sim_time >= 0.0:
            if (current_sim_time - self.last_capture_sim_time) < (self.frame_interval_sim - 1e-5):
                return None

        mujoco.mj_forward(self.scene.model, self.scene.data)

        # Render 5 cameras
        cam_frames = [self._render_camera(cam) for cam in L2_FIVE_CAMERAS]
        status_panel = self._render_status_panel()

        # Compose 3x2 grid:
        # Row 1: l2_camera_left, l2_camera_right, l2_camera_top
        # Row 2: l2_camera_front, l2_camera_close, status_panel
        top_row = np.hstack([cam_frames[0], cam_frames[1], cam_frames[2]])
        bottom_row = np.hstack([cam_frames[3], cam_frames[4], status_panel])
        mosaic_rgb = np.vstack([top_row, bottom_row])

        self.last_capture_sim_time = current_sim_time
        self.total_frames_captured += 1

        # Write to video
        if self.record:
            if self.imageio_writer is not None:
                self.imageio_writer.append_data(mosaic_rgb)
            elif self.video_writer is not None:
                mosaic_bgr = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR)
                self.video_writer.write(mosaic_bgr)

        # Show live GUI if requested
        if self.live_viewer is not None:
            self.live_viewer.show(mosaic_rgb)

        return mosaic_rgb

    def step_callback(self, *args, **kwargs) -> None:
        """Simulation step callback hook called during physical steps."""
        self.capture_frame(force=False)

    def hold_final_frame(self, duration_s: float = 1.5) -> None:
        """Hold the final mosaic frame in video recording and GUI."""
        steps = int(round(duration_s * self.fps))
        for _ in range(max(1, steps)):
            self.capture_frame(force=True)

    def close(self) -> None:
        """Finalize video recording, destroy GUI windows, and close renderer."""
        if self.imageio_writer is not None:
            try:
                self.imageio_writer.close()
            except Exception:
                pass
            self.imageio_writer = None

        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        if self.live_viewer is not None:
            self.live_viewer.close()
            self.live_viewer = None

        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None


def create_camera_manifest(
    output_path: Path | str,
    mosaic_width: int,
    mosaic_height: int,
    tile_width: int,
    tile_height: int,
    fps: int,
    total_frames: int,
    duration_sim_s: float,
) -> dict[str, Any]:
    """Generate manifest documenting the 5-camera mosaic layout."""
    manifest = {
        "execution_mode": "INTEGRATED_LIVING_ROOM_PHASE3",
        "video_path": str(output_path),
        "fps": fps,
        "total_frames": total_frames,
        "simulation_duration_s": duration_sim_s,
        "mosaic_dimensions": {"width": mosaic_width, "height": mosaic_height},
        "tile_dimensions": {"width": tile_width, "height": tile_height},
        "layout_grid": {"columns": 3, "rows": 2},
        "tiles": [
            {"row": 0, "col": 0, "name": L2_FIVE_CAMERAS[0], "role": "PROJECT_CAMERA_LEFT"},
            {"row": 0, "col": 1, "name": L2_FIVE_CAMERAS[1], "role": "PROJECT_CAMERA_RIGHT"},
            {"row": 0, "col": 2, "name": L2_FIVE_CAMERAS[2], "role": "PROJECT_CAMERA_TOP"},
            {"row": 1, "col": 0, "name": L2_FIVE_CAMERAS[3], "role": "PROJECT_CAMERA_FRONT"},
            {"row": 1, "col": 1, "name": L2_FIVE_CAMERAS[4], "role": "PROJECT_CAMERA_CLOSE"},
            {"row": 1, "col": 2, "name": "status_panel", "role": "INTERACTIVE_HUD_METRICS"},
        ],
    }
    return manifest
