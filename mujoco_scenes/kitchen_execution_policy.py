"""Execution-only access policy for Google Robot kitchen actions.

This module intentionally does not modify the frozen symbolic task domain.
It refines physical actions with the workspace needed to execute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KitchenWorkspace(str, Enum):
    HOME = "home"
    LEFT_SIDE = "left_side"
    RIGHT_SIDE = "right_side"


class KitchenExecutionAction(str, Enum):
    MOVE = "MOVE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"


WORKSPACE_DESTINATIONS = {
    KitchenWorkspace.HOME: "home",
    KitchenWorkspace.LEFT_SIDE: "cupboard1",
    KitchenWorkspace.RIGHT_SIDE: "cupboard2",
}

CONTAINER_WORKSPACES = {
    "D1": KitchenWorkspace.HOME,
    "D2": KitchenWorkspace.HOME,
    "C1": KitchenWorkspace.LEFT_SIDE,
    "C2": KitchenWorkspace.RIGHT_SIDE,
    "B1": KitchenWorkspace.RIGHT_SIDE,
}


def required_workspace(
    action: KitchenExecutionAction | str, target: str
) -> KitchenWorkspace:
    action = (
        action
        if isinstance(action, KitchenExecutionAction)
        else KitchenExecutionAction(str(action).upper())
    )
    if action not in {KitchenExecutionAction.OPEN, KitchenExecutionAction.CLOSE}:
        raise ValueError(f"Workspace policy is undefined for {action.value}")
    try:
        return CONTAINER_WORKSPACES[target]
    except KeyError as error:
        raise ValueError(f"Unknown kitchen container: {target}") from error


@dataclass(frozen=True)
class ExecutionRefinement:
    requested_action: KitchenExecutionAction
    target: str
    required_workspace: KitchenWorkspace
    starting_workspace: KitchenWorkspace
    auto_move_inserted: bool
    refined_actions: tuple[tuple[str, str], ...]


class KitchenExecutionPolicy:
    """Pure deterministic policy used by the physical dispatcher."""

    def refine(
        self,
        action: KitchenExecutionAction | str,
        target: str,
        current_workspace: KitchenWorkspace | str,
    ) -> ExecutionRefinement:
        action = (
            action
            if isinstance(action, KitchenExecutionAction)
            else KitchenExecutionAction(str(action).upper())
        )
        current = (
            current_workspace
            if isinstance(current_workspace, KitchenWorkspace)
            else KitchenWorkspace(current_workspace)
        )
        required = required_workspace(action, target)
        move = current != required
        steps: list[tuple[str, str]] = []
        if move:
            steps.append((KitchenExecutionAction.MOVE.value, required.value))
        steps.append((action.value, target))
        return ExecutionRefinement(
            requested_action=action,
            target=target,
            required_workspace=required,
            starting_workspace=current,
            auto_move_inserted=move,
            refined_actions=tuple(steps),
        )
