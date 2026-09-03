"""Common deterministic A* sequencing boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mujoco_scenes.symbolic_planning_core import (
    SearchResult, SymbolicProblem, deterministic_astar, independent_replay,
)


class PlanningCompiler(Protocol):
    def compile_problem(self, assignment: dict[str, str], context: dict[str, Any]) -> SymbolicProblem: ...


@dataclass(frozen=True)
class PlannedSequence:
    actions: tuple[dict[str, Any], ...]
    search: SearchResult
    validation: dict[str, Any]


def plan_with_common_astar(
    compiler: PlanningCompiler,
    assignment: dict[str, str],
    context: dict[str, Any],
) -> PlannedSequence:
    problem = compiler.compile_problem(assignment, context)
    search = deterministic_astar(problem)
    validation = independent_replay(problem, search.plan)
    if validation["status"] != "VALID":
        raise RuntimeError(f"A* plan failed independent replay: {validation}")
    actions = tuple({
        "action_index": index,
        "action_instance_id": f"fact_{index:03d}_{action.name.lower()}",
        "operator": action.name.upper(),
        "arguments": list(action.arguments),
    } for index, action in enumerate(search.plan, start=1))
    return PlannedSequence(actions=actions, search=search, validation=validation)
