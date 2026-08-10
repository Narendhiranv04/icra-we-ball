"""Command-line runner for Kitchen Google Robot Execution Phase A."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import mujoco

from .kitchen_google_execution import KitchenGoogleExecutionDispatcher
from .scene_loader import KitchenScene


SEQUENCES = {
    "d1_cycle": (("OPEN", "D1"), ("CLOSE", "D1")),
    "d2_cycle": (("OPEN", "D2"), ("CLOSE", "D2")),
    "c1_cycle": (("OPEN", "C1"), ("CLOSE", "C1")),
    "c2_cycle": (("OPEN", "C2"), ("CLOSE", "C2")),
    "b1_cycle": (("OPEN", "B1"), ("CLOSE", "B1")),
    "drawer_validation": (("OPEN", "D1"), ("CLOSE", "D1"),
                          ("OPEN", "D2"), ("CLOSE", "D2")),
    "container_validation": tuple(
        pair for container in ("D1", "D2", "C1", "C2", "B1")
        for pair in (("OPEN", container), ("CLOSE", container))
    ),
    "workspace_validation": (("OPEN", "C1"), ("CLOSE", "C1"),
                             ("OPEN", "C2"), ("CLOSE", "C2")),
}


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", default="S1_integrated_kitchen_object_function_primary"
    )
    parser.add_argument("--action", choices=("open", "close"))
    parser.add_argument("--container", choices=("D1", "D2", "C1", "C2", "B1"))
    parser.add_argument("--sequence", choices=tuple(SEQUENCES))
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--viewer", action="store_true",
        help="Open and continuously synchronize a MuJoCo viewer during execution",
    )
    parser.add_argument("--output-dir", default="runs/kitchen_google_execution_phaseA")
    args = parser.parse_args()
    if bool(args.sequence) == bool(args.action and args.container):
        parser.error("choose either --sequence, or both --action and --container")
    if args.cycles < 1:
        parser.error("--cycles must be positive")

    actions = SEQUENCES[args.sequence] if args.sequence else (
        (args.action.upper(), args.container),
    )
    scene = KitchenScene(args.scene, robot="google")
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
        mujoco.mjv_defaultFreeCamera(scene.model, viewer.cam)

    def sync_viewer():
        if viewer is not None and viewer.is_running():
            viewer.sync()

    dispatcher = KitchenGoogleExecutionDispatcher(
        scene, step_callback=sync_viewer if viewer is not None else None
    )
    records = []
    for cycle in range(args.cycles):
        for action, container in actions:
            print(f"[Phase A] cycle={cycle + 1} {action}({container})", flush=True)
            record = dispatcher.request(action, container, execute=args.execute)
            record["cycle"] = cycle + 1
            records.append(record)
            print(
                f"  status={record['status']} success={record['success']}",
                flush=True,
            )
            if not record["success"]:
                break
        if records and not records[-1]["success"]:
            break
    output = {
        "phase": "KITCHEN_GOOGLE_EXECUTION_PHASE_A",
        "scene": args.scene,
        "execute": args.execute,
        "requested_cycles": args.cycles,
        "status": "SUCCESS" if records and all(r["success"] for r in records) else "FAILED",
        "records": records,
    }
    output_dir = Path(args.output_dir)
    _atomic_json(output_dir / "execution_results.json", output)
    if viewer is not None:
        viewer.sync()
    print(f"Saved: {output_dir / 'execution_results.json'}", flush=True)
    exit_code = 0 if output["status"] == "SUCCESS" else 1
    if viewer is not None:
        # GLFW on Wayland/XWayland can crash while clearing MuJoCo's EGL
        # context during interpreter teardown.  All execution artifacts have
        # already been atomically written, so bypass only the faulty native
        # teardown path.  Headless runs retain ordinary Python cleanup.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
