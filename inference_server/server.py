"""Build and launch one profiled OpenAI-compatible inference server."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL_REGISTRY = ROOT / "models.json"
BACKENDS = ("vllm", "sglang")


def _is_loopback(host: str) -> bool:
    return host.strip().strip("[]").lower() in {"127.0.0.1", "::1", "localhost"}


def load_profiles(path: str | Path = MODEL_REGISTRY) -> dict[str, dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported model registry schema")
    profiles = document.get("models")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Model registry must contain profiles")
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name or not isinstance(profile, dict):
            raise ValueError("Invalid model profile entry")
        if profile.get("available", True) is not True:
            if not isinstance(profile.get("notes"), str):
                raise ValueError(f"Unavailable profile {name!r} needs notes")
            continue
        required_strings = ("model_id", "served_name", "default_backend", "precision")
        if any(
            not isinstance(profile.get(field), str) or not profile[field].strip()
            for field in required_strings
        ):
            raise ValueError(f"Profile {name!r} has invalid required fields")
        backend = profile["default_backend"]
        if backend not in BACKENDS or not isinstance(profile.get(backend), dict):
            raise ValueError(f"Profile {name!r} has an invalid default backend")
        for field in ("max_model_len", "max_concurrency"):
            value = profile.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Profile {name!r} has invalid {field}")
        max_images = profile.get("max_images", 0)
        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 0:
            raise ValueError(f"Profile {name!r} has invalid max_images")
        planner = profile.get("planner")
        if not isinstance(planner, dict):
            raise ValueError(f"Profile {name!r} needs planner configuration")
    return profiles


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _fraction(value: str, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def _port(value: str, name: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return parsed


def _setting(environ: Mapping[str, str], name: str, default: object) -> str:
    """Return a non-empty environment override or the profile default."""
    return environ.get(name, "").strip() or str(default)


def resolve_profile(
    environ: Mapping[str, str],
    profiles: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any], str]:
    available_profiles = load_profiles() if profiles is None else profiles
    profile_name = environ.get("INFERENCE_MODEL", "").strip()
    if not profile_name:
        raise ValueError("INFERENCE_MODEL is required")
    if profile_name not in available_profiles:
        choices = ", ".join(sorted(available_profiles))
        raise ValueError(f"Unknown model profile {profile_name!r}; choose from {choices}")
    profile = available_profiles[profile_name]
    if profile.get("available", True) is not True:
        raise ValueError(f"{profile_name} is not runnable: {profile.get('notes', '')}")
    backend = environ.get(
        "INFERENCE_BACKEND", profile.get("default_backend", "vllm")
    ).strip()
    if backend not in BACKENDS:
        raise ValueError(f"INFERENCE_BACKEND must be one of {', '.join(BACKENDS)}")
    if backend not in profile:
        raise ValueError(f"{profile_name} has no {backend} configuration")
    return profile_name, profile, backend


def build_command(
    environ: Mapping[str, str],
    extra_args: Sequence[str] = (),
    *,
    profiles: Mapping[str, dict[str, Any]] | None = None,
) -> list[str]:
    _profile_name, profile, backend = resolve_profile(environ, profiles)
    model_id = environ.get("INFERENCE_MODEL_ID", "").strip() or profile["model_id"]
    served_name = environ.get("INFERENCE_SERVED_NAME", "").strip() or profile[
        "served_name"
    ]
    host = _setting(environ, "INFERENCE_HOST", "0.0.0.0")
    api_key = environ.get("INFERENCE_API_KEY", "").strip()
    if not api_key and not _is_loopback(host):
        raise ValueError("INFERENCE_API_KEY is required for a non-loopback host")
    port = _port(
        _setting(environ, "INFERENCE_CONTAINER_PORT", "8000"),
        "INFERENCE_CONTAINER_PORT",
    )
    tensor_parallel = _positive_int(
        _setting(environ, "INFERENCE_TENSOR_PARALLEL_SIZE", 1),
        "INFERENCE_TENSOR_PARALLEL_SIZE",
    )
    max_model_len = _positive_int(
        _setting(environ, "INFERENCE_MAX_MODEL_LEN", profile["max_model_len"]),
        "INFERENCE_MAX_MODEL_LEN",
    )
    max_concurrency = _positive_int(
        _setting(environ, "INFERENCE_MAX_CONCURRENCY", profile["max_concurrency"]),
        "INFERENCE_MAX_CONCURRENCY",
    )
    memory_fraction = _fraction(
        _setting(environ, "INFERENCE_GPU_MEMORY_UTILIZATION", "0.90"),
        "INFERENCE_GPU_MEMORY_UTILIZATION",
    )
    backend_profile = profile[backend]

    if backend == "vllm":
        command = [
            "vllm",
            "serve",
            model_id,
            "--served-model-name",
            served_name,
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tensor_parallel),
            "--max-model-len",
            str(max_model_len),
            "--max-num-seqs",
            str(max_concurrency),
            "--gpu-memory-utilization",
            str(memory_fraction),
            "--dtype",
            "bfloat16",
            "--enable-prefix-caching",
            "--generation-config",
            "vllm",
        ]
        if api_key:
            command.extend(("--api-key", api_key))
        max_images = int(profile.get("max_images", 0))
        if max_images:
            command.extend(
                (
                    "--limit-mm-per-prompt",
                    json.dumps({"image": max_images}, separators=(",", ":")),
                )
            )
    else:
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_id,
            "--served-model-name",
            served_name,
            "--host",
            host,
            "--port",
            str(port),
            "--tp-size",
            str(tensor_parallel),
            "--context-length",
            str(max_model_len),
            "--max-running-requests",
            str(max_concurrency),
            "--mem-fraction-static",
            str(memory_fraction),
            "--dtype",
            "bfloat16",
        ]
        if api_key:
            command.extend(("--api-key", api_key))

    if profile.get("trust_remote_code"):
        command.append("--trust-remote-code")
    quantization = backend_profile.get("quantization")
    if quantization:
        command.extend(("--quantization", quantization))
    command.extend(str(value) for value in backend_profile.get("args", ()))
    command.extend(extra_args)
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--show-profile", action="store_true")
    return parser


def printable_command(command: Sequence[str]) -> str:
    safe_command = list(command)
    if "--api-key" in safe_command:
        safe_command[safe_command.index("--api-key") + 1] = "<redacted>"
    return shlex.join(safe_command)


def main() -> None:
    arguments, extra_args = build_parser().parse_known_args()
    try:
        profile_name, profile, backend = resolve_profile(os.environ)
        command = build_command(os.environ, extra_args)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}") from error
    if arguments.show_profile:
        print(json.dumps({"name": profile_name, "backend": backend, **profile}, indent=2))
        return
    if arguments.print_command:
        print(printable_command(command))
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
