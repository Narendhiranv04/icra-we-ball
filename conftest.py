"""Repository-wide pytest bootstrap for headless MuJoCo rendering."""

from __future__ import annotations

import os


# MuJoCo's default GLFW renderer aborts the whole process when no desktop
# display is available. Tests are headless unless the caller explicitly asks
# for another backend.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
