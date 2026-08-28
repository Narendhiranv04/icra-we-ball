"""Resolve the exact PDDLStream revision used by the VLM-TAMP baseline."""

from __future__ import annotations

import os
import collections
import collections.abc
from pathlib import Path
import subprocess
import sys
import types


PDDLSTREAM_URL = "https://github.com/caelan/pddlstream.git"
PDDLSTREAM_COMMIT = "b38137e47fd4a4116a3e36bc4be691cbe5da6cb0"


class PDDLStreamDependencyError(RuntimeError):
    pass


def dependency_root() -> Path:
    override = os.environ.get("PDDLSTREAM_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".paper_deps" / "pddlstream"


def _revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def activate_pddlstream(*, require_pinned_revision: bool = True) -> Path:
    root = dependency_root()
    package = root / "pddlstream"
    if not package.is_dir():
        raise PDDLStreamDependencyError(
            f"PDDLStream is not installed at {root}. Run "
            "`bash vlm_tamp_baseline/setup_pddlstream.sh`."
        )
    revision = _revision(root)
    if require_pinned_revision and revision != PDDLSTREAM_COMMIT:
        raise PDDLStreamDependencyError(
            f"Expected PDDLStream {PDDLSTREAM_COMMIT}, found {revision or 'unknown'} "
            f"at {root}. Re-run the setup script or set PDDLSTREAM_HOME to the "
            "pinned checkout."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    # The paper checkout predates Python 3.10's collections re-export removal.
    # Patch the two legacy names at the adapter boundary without modifying the
    # pinned third-party source, preserving an auditable upstream revision.
    for name in ("Iterator", "Sequence", "Sized"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))
    # This historical fork imports two print helpers from its original
    # PyBullet host repository. They are logging-only and are not part of the
    # planner. Supplying them here avoids installing a second simulator while
    # leaving the pinned planning source untouched.
    if "pybullet_tools.logging" not in sys.modules:
        package = types.ModuleType("pybullet_tools")
        package.__path__ = []
        logging_module = types.ModuleType("pybullet_tools.logging")
        logging_module.myprint = print
        logging_module.dump_json = lambda *_args, **_kwargs: None
        bullet_module = types.ModuleType("pybullet_tools.bullet_utils")
        bullet_module.print_action_plan = (
            lambda plan, stream_plan=None, **_kwargs: str(plan)
        )
        sys.modules.setdefault("pybullet_tools", package)
        sys.modules["pybullet_tools.logging"] = logging_module
        sys.modules["pybullet_tools.bullet_utils"] = bullet_module
        planning_package = types.ModuleType("pybullet_planning")
        planning_package.__path__ = []
        planning_tools = types.ModuleType("pybullet_planning.pybullet_tools")
        planning_tools.__path__ = []
        sys.modules.setdefault("pybullet_planning", planning_package)
        sys.modules.setdefault("pybullet_planning.pybullet_tools", planning_tools)
        sys.modules["pybullet_planning.pybullet_tools.logging"] = logging_module
    # The pinned fork enables process-global visualization and learned stream
    # statistics by default. Those facilities write into the caller's current
    # directory and its JSON logger assumes the original PyBullet host. The
    # baseline records its own complete per-run trace, so disable both here to
    # keep planning deterministic and artifact directories isolated.
    from pddlstream.algorithms.visualization import set_visualizations_false
    from pddlstream.language import statistics

    set_visualizations_false()
    statistics.LOAD_STATISTICS = False
    statistics.SAVE_STATISTICS = False
    return root
