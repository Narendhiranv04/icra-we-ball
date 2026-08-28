"""Production functional-search to Kitchen Phase-C execution boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from .kitchen_phase_c_execution import KitchenPhaseCExecutionDispatcher
from .tamp.grounded_execution import (
    GoalVerifier,
    GroundedPlanExecutive,
    GroundedTask,
    Sequencer,
)
from .tamp.observation_adapter import observed_registry_state
from .tamp.physical_dispatcher import MuJoCoSkillDispatcher


class KitchenExecutionObserver:
    """Fresh bounded state backed by observed inventory and live robot state."""

    def __init__(self, phase_b: KitchenPhaseBExecutionDispatcher):
        self.phase_b = phase_b
        self.revision = 0
        self.generic_for_backend = {
            str(row["physical_backend_body"]): str(row["generic_object_id"])
            for row in phase_b.resolution["accepted"]
        }
        self.generic_ids = frozenset(self.generic_for_backend.values())

    def _generic_held_object(self) -> str | None:
        held = self.phase_b.manipulation.executor.held_object
        if held is None:
            return None
        held = str(held)
        if held in self.generic_ids:
            return held
        generic = self.generic_for_backend.get(held)
        if generic is None:
            raise RuntimeError(
                "Observed held object has no generic execution binding; "
                "refusing to expose a simulator body name"
            )
        return generic

    def __call__(self):
        self.revision += 1
        held = self._generic_held_object()
        return observed_registry_state(
            self.phase_b.inventory,
            self.phase_b.scene.get_region_observation_states(),
            robot_location=self.phase_b.current_workspace.value,
            held_object=held,
            revision=self.revision,
            live_locations=getattr(self.phase_b, "live_object_locations", None),
        )


class KitchenGroundedExecution:
    """Compose the production sequencer with observed Phase-C execution."""

    def __init__(
        self,
        scene,
        inventory: dict[str, Any],
        resolution: dict[str, Any],
        registry: dict[str, Any],
        sequencer: Sequencer,
        goal_verifier: GoalVerifier,
        *,
        step_callback=None,
        max_replans: int = 3,
    ):
        self.phase_b = KitchenPhaseBExecutionDispatcher(
            scene, inventory, resolution, step_callback=step_callback
        )
        self.phase_c = KitchenPhaseCExecutionDispatcher(
            self.phase_b, registry, []
        )
        self.observer = KitchenExecutionObserver(self.phase_b)
        self.dispatcher = MuJoCoSkillDispatcher(self.phase_c)
        self.executive = GroundedPlanExecutive(
            self.observer,
            sequencer,
            self.dispatcher,
            goal_verifier,
            max_replans=max_replans,
        )

    def start(
        self,
        task_id: str,
        goal: str,
        verified_witness: Mapping[str, Any],
    ) -> None:
        status = str(verified_witness.get("status", "")).upper()
        if status not in {"TRUE", "COMPLETE", "FEASIBLE"}:
            raise ValueError(
                "Execution requires a complete semantic/geometric witness"
            )
        self.executive.start(
            GroundedTask(task_id, goal, dict(verified_witness))
        )

    def run(self, *, maximum_updates: int = 1000) -> GroundedPlanExecutive:
        for _ in range(maximum_updates):
            if not self.executive.busy:
                return self.executive
            self.executive.update()
        raise RuntimeError("Grounded execution update budget exhausted")


def effects_goal_verifier(
    required_effects: Sequence[str],
) -> GoalVerifier:
    """Verify final goals from physically accepted skill postconditions."""
    required = frozenset(required_effects)

    def verify(_task, _state, history) -> bool:
        observed = {
            str(effect)
            for row in history
            if row.get("success") is True
            for effect in row.get("effects", ())
        }
        return bool(required) and required <= observed

    return verify
