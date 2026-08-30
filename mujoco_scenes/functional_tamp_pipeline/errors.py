"""Canonical error definitions for functional TAMP pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all functional TAMP pipeline errors."""


class VLMSpecificationError(PipelineError, ValueError):
    """Raised when VLM functional specification acquisition, validation, or canonicalization fails."""

    def __init__(self, message: str, category: str = "UNMAPPED_FUNCTIONAL_CONCEPT"):
        super().__init__(message)
        self.category = category


class MalformedVLMSpecificationError(VLMSpecificationError):
    """Raised when VLM response violates structure, has undeclared endpoints, or contains contradictions."""

    def __init__(self, message: str):
        super().__init__(message, category="MALFORMED_VLM_SPECIFICATION")


class UnmappedFunctionalConceptError(VLMSpecificationError):
    """Raised when a VLM natural-language phrase cannot be mapped to any reviewed functional grammar."""

    def __init__(self, message: str):
        super().__init__(message, category="UNMAPPED_FUNCTIONAL_CONCEPT")


class UnsupportedCheckerCapabilityError(VLMSpecificationError):
    """Raised when a requirement is physically meaningful but no checker exists in the framework."""

    def __init__(self, message: str):
        super().__init__(message, category="UNSUPPORTED_CHECKER_CAPABILITY")


class AmbiguousCanonicalizationError(VLMSpecificationError):
    """Raised when a VLM phrase matches multiple distinct canonical concepts."""

    def __init__(self, message: str):
        super().__init__(message, category="AMBIGUOUS_CANONICALIZATION")


class TransportOrStructuredOutputError(VLMSpecificationError):
    """Raised when transport, connection, timeout, or JSON decoding fails."""

    def __init__(self, message: str):
        super().__init__(message, category="TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE")


class ReplaySpecificationError(PipelineError):
    """Raised when replaying a specification JSON file fails due to missing file, malformed format, etc."""


class GroundingError(PipelineError):
    """Raised when functional grounding encounters an unexpected structural failure."""


class PlanningCompilationError(PipelineError):
    """Raised when compiling the grounded observed state into a symbolic problem fails."""
