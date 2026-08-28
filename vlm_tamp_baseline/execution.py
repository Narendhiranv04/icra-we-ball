"""VLM-TAMP-specific name for the neutral MuJoCo action adapter."""

from baseline_common.execution import MuJoCoActionExecutor, observation_from_state


class VLMTAMPMuJoCoExecutor(MuJoCoActionExecutor):
    """Run refined VLM-TAMP skills through the shared physical dispatcher."""


__all__ = ["VLMTAMPMuJoCoExecutor", "observation_from_state"]
