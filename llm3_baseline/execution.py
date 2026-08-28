"""LLM3-specific name for the neutral MuJoCo action adapter."""

from baseline_common.execution import MuJoCoActionExecutor, observation_from_state
from baseline_common.models import Action, ActionResult


class LLM3MuJoCoExecutor(MuJoCoActionExecutor):
    """Execute LLM3's sampled parameters through collision-checked skills."""

    def execute(self, action: Action) -> ActionResult:
        physical = getattr(self.dispatcher, "physical", None)
        setter = getattr(physical, "set_external_action_parameters", None)
        clearer = getattr(physical, "clear_external_action_parameters", None)
        if callable(setter):
            try:
                setter(dict(action.parameters))
            except Exception as error:
                return ActionResult.failed(
                    "internal_error",
                    "Could not apply sampled action parameters: "
                    f"{type(error).__name__}: {error}",
                    recoverable=False,
                )
        result: ActionResult | None = None
        execution_error: Exception | None = None
        cleanup_error: Exception | None = None
        try:
            result = super().execute(action)
        except Exception as error:
            execution_error = error
        finally:
            if callable(clearer):
                try:
                    clearer()
                except Exception as error:
                    cleanup_error = error
        if cleanup_error is not None:
            return ActionResult.failed(
                "internal_error",
                "Could not clear sampled action parameters: "
                f"{type(cleanup_error).__name__}: {cleanup_error}",
                recoverable=False,
            )
        if execution_error is not None:
            return ActionResult.failed(
                "internal_error",
                f"Action adapter raised {type(execution_error).__name__}: "
                f"{execution_error}",
                recoverable=False,
            )
        if result is None:
            return ActionResult.failed(
                "internal_error",
                "Action adapter returned no result",
                recoverable=False,
            )
        return result


__all__ = ["LLM3MuJoCoExecutor", "observation_from_state"]
