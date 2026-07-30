"""Visible candidate generation for functional assessment."""

from __future__ import annotations

from mujoco_scenes.foundation_model import Candidate
from mujoco_scenes.tamp.functions import FunctionSpec
from mujoco_scenes.tamp.state import ObservedState


def _public_facts(facts) -> dict[str, object]:
    return {
        key: value
        for key, value in facts.items()
        if not key.startswith("_")
    }


def visible_candidates(
    state: ObservedState, function: FunctionSpec
) -> tuple[Candidate, ...]:
    if function.candidate_kind == "object":
        return tuple(
            Candidate(
                observation.object_id,
                observation.category,
                {
                    "location": observation.location,
                    **_public_facts(observation.facts),
                },
            )
            for observation in state.objects.values()
            if observation.visible
        )

    return tuple(
        Candidate(
            observation.region_id,
            observation.category,
            {
                "inspected": observation.inspected,
                "open": observation.open,
                "occupied_by": (
                    list(observation.occupied_by)
                    if observation.occupied_by is not None
                    else "unknown"
                ),
                **_public_facts(observation.facts),
            },
        )
        for observation in state.regions.values()
        if observation.visible
    )
