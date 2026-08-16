"""Manage the standalone Docker inference workspace."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

try:
    from .server import BACKENDS, build_command, load_profiles, printable_command, resolve_profile
except ImportError:  # Direct execution from this standalone workspace.
    from server import BACKENDS, build_command, load_profiles, printable_command, resolve_profile


ROOT = Path(__file__).resolve().parent


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected NAME=value")
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip("\"'")
    return values


def environment(profile: str | None = None, backend: str | None = None) -> dict[str, str]:
    result = _load_env(ROOT / ".env")
    result.update(os.environ)
    if profile:
        result["INFERENCE_MODEL"] = profile
    if backend:
        result["INFERENCE_BACKEND"] = backend
    return result


def compose_command(*arguments: str) -> list[str]:
    return ["docker", "compose", "--project-directory", str(ROOT), *arguments]


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def list_models() -> None:
    print(f"{'PROFILE':28} {'BACKEND':8} {'PRECISION':12} {'CONTEXT':>7}  MODEL")
    for name, profile in load_profiles().items():
        if not profile.get("available", True):
            print(f"{name:28} {'-':8} {'unavailable':12} {'-':>7}  {profile['notes']}")
            continue
        print(
            f"{name:28} {profile['default_backend']:8} "
            f"{profile['precision']:12} {profile['max_model_len']:7}  {profile['model_id']}"
        )


def doctor() -> None:
    failures: list[str] = []
    checks = (
        ("docker", ["docker", "--version"]),
        ("Docker Compose", ["docker", "compose", "version"]),
        (
            "NVIDIA driver",
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
        ),
    )
    for label, command in checks:
        if shutil.which(command[0]) is None:
            print(f"FAIL {label}: {command[0]} is not installed")
            failures.append(label)
            continue
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        detail = (completed.stdout or completed.stderr).strip().splitlines()
        if completed.returncode:
            print(f"FAIL {label}: {detail[0] if detail else 'command failed'}")
            failures.append(label)
        else:
            print(f"OK   {label}: {detail[0] if detail else 'available'}")

    configured = environment()
    if not (ROOT / ".env").exists():
        print("FAIL configuration: copy .env.example to .env")
        failures.append("configuration")
    elif not configured.get("INFERENCE_API_KEY", "").strip() or configured.get(
        "INFERENCE_API_KEY"
    ) == "replace-with-a-long-random-key":
        print("FAIL configuration: set a private INFERENCE_API_KEY in .env")
        failures.append("configuration")
    else:
        print("OK   configuration: .env and API key are present")

    if failures:
        raise SystemExit("Doctor found missing requirements: " + ", ".join(failures))


def start(profile_name: str, backend_override: str | None, detach: bool) -> None:
    env = environment(profile_name, backend_override)
    _, profile, backend = resolve_profile(env)
    print(
        f"Starting {profile['model_id']} with {backend} and the functional API"
    )
    arguments = [
        "--profile",
        backend,
        "--profile",
        "planner",
        "up",
        "--remove-orphans",
    ]
    if detach:
        arguments.append("--detach")
    arguments.extend((backend, "planner"))
    run(compose_command(*arguments), env)


def stop() -> None:
    run(
        compose_command(
            "--profile",
            "vllm",
            "--profile",
            "sglang",
            "--profile",
            "planner",
            "down",
        ),
        environment(),
    )


def logs(backend: str, follow: bool) -> None:
    arguments = ["--profile", backend, "--profile", "planner", "logs"]
    if follow:
        arguments.append("--follow")
    arguments.extend((backend, "planner"))
    run(compose_command(*arguments), environment(backend=backend))


def show_command(profile_name: str, backend: str | None) -> None:
    env = environment(profile_name, backend)
    print(printable_command(build_command(env)))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list supported model profiles")
    commands.add_parser("doctor", help="check Docker, NVIDIA, and configuration")

    up = commands.add_parser("up", help="start one model server")
    up.add_argument("profile")
    up.add_argument("--backend", choices=BACKENDS)
    up.add_argument("--detach", "-d", action="store_true")

    commands.add_parser("down", help="stop and remove the inference containers")

    log_parser = commands.add_parser("logs", help="show server logs")
    log_parser.add_argument("--backend", choices=BACKENDS, default="vllm")
    log_parser.add_argument("--follow", "-f", action="store_true")

    command_parser = commands.add_parser(
        "command", help="print the generated backend command"
    )
    command_parser.add_argument("profile")
    command_parser.add_argument("--backend", choices=BACKENDS)
    return result


def main() -> None:
    arguments = parser().parse_args()
    try:
        if arguments.command == "list":
            list_models()
        elif arguments.command == "doctor":
            doctor()
        elif arguments.command == "up":
            start(arguments.profile, arguments.backend, arguments.detach)
        elif arguments.command == "down":
            stop()
        elif arguments.command == "logs":
            logs(arguments.backend, arguments.follow)
        elif arguments.command == "command":
            show_command(arguments.profile, arguments.backend)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}") from error


if __name__ == "__main__":
    main()
