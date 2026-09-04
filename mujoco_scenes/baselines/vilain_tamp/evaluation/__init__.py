"""Terminal-only hidden benchmark evaluation for ViLaIn-TAMP."""

from .base import (
    EvaluationContractError,
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    evaluate_hidden_benchmark,
)

__all__ = [
    "EvaluationContractError",
    "HiddenBenchmarkContext",
    "TerminalStateSnapshot",
    "evaluate_hidden_benchmark",
]
