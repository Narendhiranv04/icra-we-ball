"""Canonical error definitions for functional TAMP pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all functional TAMP pipeline errors."""


class VLMSpecificationError(PipelineError, ValueError):
    """Raised when VLM functional specification acquisition, validation, or canonicalization fails."""


class ReplaySpecificationError(PipelineError):
    """Raised when replaying a specification JSON file fails due to missing file, malformed format, etc."""


class GroundingError(PipelineError):
    """Raised when functional grounding encounters an unexpected structural failure."""


class PlanningCompilationError(PipelineError):
    """Raised when compiling the grounded observed state into a symbolic problem fails."""
