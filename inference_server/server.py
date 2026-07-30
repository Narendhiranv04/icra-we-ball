"""Launch the repository's vLLM ranking server."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence


def build_command(
    environ: Mapping[str, str], extra_args: Sequence[str] = ()
) -> list[str]:
    model = environ.get("VLLM_MODEL", "").strip()
    api_key = environ.get("VLLM_API_KEY", "").strip()
    if not model:
        raise ValueError("VLLM_MODEL is required")
    if not api_key:
        raise ValueError("VLLM_API_KEY is required")

    tensor_parallel_size = int(
        environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")
    )
    if tensor_parallel_size < 1:
        raise ValueError("VLLM_TENSOR_PARALLEL_SIZE must be positive")

    command = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        environ.get("VLLM_SERVED_MODEL_NAME", "tamp-ranker"),
        "--host",
        environ.get("VLLM_HOST", "0.0.0.0"),
        "--port",
        environ.get("VLLM_PORT", "8000"),
        "--api-key",
        api_key,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
    ]

    optional_values = {
        "--max-model-len": environ.get("VLLM_MAX_MODEL_LEN"),
        "--gpu-memory-utilization": environ.get(
            "VLLM_GPU_MEMORY_UTILIZATION"
        ),
    }
    for flag, value in optional_values.items():
        if value:
            command.extend((flag, value))
    command.extend(extra_args)
    return command


def main() -> None:
    try:
        command = build_command(os.environ, sys.argv[1:])
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}") from error
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
