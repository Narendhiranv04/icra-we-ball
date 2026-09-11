"""Make detector inference reproducible run to run.

Three replays of the *same* archived FM responses through the *same* code did
not agree.  `workshop/W5/trial_03` lost an object in two of them and grounded
completely in the third, moving its GT goal coverage between 0.333 and 1.0.
Nothing semantic differed -- the contract is read from disk.  Two of those
replays ran at identical concurrency on a quiet machine and still disagreed, so
machine load is not the explanation: the variation is intrinsic to inference.

The detector runs on the GPU at inference size 1280 with acceptance thresholds
as low as 0.001, so a large number of detections sit at their boundary, and
nothing in this repository pinned a seed, disabled cuDNN autotuning, or asked
for deterministic kernels.  A benchmark whose headline moves between identical
runs cannot be reported.

This is a reproducibility control, not a tuning knob: it changes no threshold,
reads no ground truth, and conditions on no variant.  It does change which
kernels run, so the whole matrix is re-measured after enabling it.

Determinism is requested *strictly*.  `torch.use_deterministic_algorithms` has a
`warn_only` mode that downgrades an unsupported operation to a warning and
carries on non-deterministically; using it would mean claiming determinism while
silently not having it.  Strict mode is requested instead, and if an operation
has no deterministic implementation the resulting error names it, which is the
information needed to decide what to do about that specific operation.
"""
from __future__ import annotations

import os

DEFAULT_SEED = 0
_STATE: dict[str, object] | None = None


def determinism_report() -> dict[str, object]:
    """What was actually applied, or why not.  Never assumes success."""
    return dict(_STATE or {"status": "NOT_APPLIED"})


def enable_deterministic_inference(seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Pin every reachable source of run-to-run variation.  Idempotent.

    Returns a record of what was applied rather than a bare success flag, so a
    run can store what it actually achieved instead of asserting determinism it
    did not get.
    """
    global _STATE
    if _STATE is not None:
        return dict(_STATE)

    state: dict[str, object] = {"requested_seed": seed, "strict": True}
    if os.environ.get("TAMP_NONDETERMINISTIC_INFERENCE", "").strip() in ("1", "true", "yes"):
        _STATE = {"status": "DISABLED_BY_ENV",
                  "reason": "TAMP_NONDETERMINISTIC_INFERENCE is set"}
        return dict(_STATE)

    # Must be set before the first CUBLAS workspace is created, which is why
    # this module is imported before the detector model is constructed rather
    # than merely before a prediction call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    state["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]

    # PYTHONHASHSEED cannot be set from here: the interpreter reads it at
    # startup, so by the time this runs the hash seed is already fixed.  It is
    # recorded rather than assumed, because without it set-iteration order -- and
    # so a choice among equally ranked candidates -- varies between processes.
    state["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "<unset: NOT REPRODUCIBLE>")

    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
        state["numpy"] = "seeded"
    except Exception as exc:
        state["numpy"] = f"unavailable: {exc!r}"

    try:
        import torch
    except Exception as exc:
        _STATE = {**state, "status": "NO_TORCH", "torch": f"unavailable: {exc!r}"}
        return dict(_STATE)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        state["cuda"] = "seeded"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state["cudnn_deterministic"] = True
    state["cudnn_benchmark"] = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state["tf32"] = "disabled"
    except Exception:
        pass

    # Strict, never warn_only.  If this raises here the build cannot honour it
    # at all; if an individual operation lacks a deterministic implementation
    # the error surfaces at inference and names that operation.
    try:
        torch.use_deterministic_algorithms(True)
        state["deterministic_algorithms"] = "strict"
        state["status"] = "DETERMINISTIC"
    except Exception as exc:
        state["deterministic_algorithms"] = f"refused: {exc!r}"
        state["status"] = "PARTIAL"

    state["torch_version"] = torch.__version__
    _STATE = state
    return dict(_STATE)
