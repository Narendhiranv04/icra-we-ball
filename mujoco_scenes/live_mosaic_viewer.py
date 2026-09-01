"""Small ffplay-backed live viewer for RGB mosaics.

The project environment uses an OpenCV build without GUI support, so
``cv2.imshow`` cannot be used.  ffplay is already required alongside ffprobe
for the final-paper evidence workflow and accepts raw RGB frames over stdin.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np


class LiveMosaicViewerClosed(RuntimeError):
    """Raised when the user closes the external live-view window."""


class LiveMosaicViewer:
    """Display fixed-size RGB24 frames in a low-latency ffplay window."""

    def __init__(self, width: int, height: int, fps: int, title: str):
        executable = shutil.which("ffplay")
        if executable is None:
            raise RuntimeError("--show requires ffplay, but ffplay was not found in PATH")
        self.process = subprocess.Popen(
            [
                executable,
                "-loglevel", "error",
                "-nostats",
                "-autoexit",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{int(width)}x{int(height)}",
                "-framerate", str(max(1, int(fps))),
                "-window_title", title,
                "-i", "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )

    def show(self, frame_rgb: np.ndarray) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            raise LiveMosaicViewerClosed(
                "The ffplay live-view window exited. Ensure this command is run "
                "from the desktop session with DISPLAY available."
            )
        frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise LiveMosaicViewerClosed(
                "The ffplay live-view window closed"
            ) from error

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
