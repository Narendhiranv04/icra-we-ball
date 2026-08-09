"""Small scene-independent deterministic classical planning core.

The core deliberately knows nothing about MuJoCo, perception, kitchens, living
rooms, or benchmark variants.  Compilers provide grounded atoms and actions;
the planner only searches that symbolic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import time
from typing import Iterable


Atom = tuple[str, ...]


@dataclass(frozen=True, order=True)
class SymbolicAction:
    name: str
    arguments: tuple[str, ...]
    positive_preconditions: frozenset[Atom]
    negative_preconditions: frozenset[Atom]
    add_effects: frozenset[Atom]
    delete_effects: frozenset[Atom]
    cost: int = 1

    def render(self) -> str:
        return f"{self.name.upper()}({', '.join(self.arguments)})"


@dataclass(frozen=True)
class SymbolicProblem:
    initial_atoms: frozenset[Atom]
    goal_atoms: frozenset[Atom]
    actions: tuple[SymbolicAction, ...]

    def __post_init__(self) -> None:
        if not self.goal_atoms:
            raise ValueError("A symbolic problem must contain at least one goal")
        if any(action.cost <= 0 for action in self.actions):
            raise ValueError("Action costs must be positive")


@dataclass(frozen=True)
class SearchResult:
    plan: tuple[SymbolicAction, ...]
    statistics: dict[str, int | float | str]


class NoSymbolicPlan(RuntimeError):
    """Raised when deterministic search exhausts its frontier."""


def is_applicable(state: frozenset[Atom], action: SymbolicAction) -> bool:
    return (
        action.positive_preconditions <= state
        and not action.negative_preconditions.intersection(state)
    )


def apply_action(
    state: frozenset[Atom], action: SymbolicAction
) -> frozenset[Atom]:
    """Planner transition function (not used by independent replay)."""
    if not is_applicable(state, action):
        raise ValueError(f"Action is not applicable: {action.render()}")
    return (state - action.delete_effects) | action.add_effects


def applicable_actions(
    problem: SymbolicProblem, state: frozenset[Atom]
) -> tuple[SymbolicAction, ...]:
    return tuple(action for action in problem.actions if is_applicable(state, action))


def _heuristic(problem: SymbolicProblem, state: frozenset[Atom]) -> int:
    return len(problem.goal_atoms - state)


def deterministic_astar(problem: SymbolicProblem) -> SearchResult:
    """Return a unit/positive-cost plan with deterministic expansion order."""
    started = time.perf_counter()
    initial = problem.initial_atoms
    frontier: list[tuple[int, int, int, frozenset[Atom]]] = []
    serial = 0
    heappush(frontier, (_heuristic(problem, initial), 0, serial, initial))
    parents: dict[
        frozenset[Atom], tuple[frozenset[Atom], SymbolicAction] | None
    ] = {initial: None}
    best_cost = {initial: 0}
    expanded = 0
    generated = 0
    frontier_peak = 1
    while frontier:
        _score, cost, _serial, state = heappop(frontier)
        if cost != best_cost.get(state):
            continue
        expanded += 1
        if problem.goal_atoms <= state:
            plan: list[SymbolicAction] = []
            cursor = state
            while parents[cursor] is not None:
                previous, action = parents[cursor]
                plan.append(action)
                cursor = previous
            plan.reverse()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return SearchResult(
                tuple(plan),
                {
                    "algorithm": "deterministic_astar_symbolic_state_search",
                    "expanded_states": expanded,
                    "generated_states": generated,
                    "visited_states": len(best_cost),
                    "frontier_peak": frontier_peak,
                    "plan_cost": sum(action.cost for action in plan),
                    "plan_length": len(plan),
                    "search_time_ms": elapsed_ms,
                },
            )
        for action in applicable_actions(problem, state):
            successor = apply_action(state, action)
            generated += 1
            next_cost = cost + action.cost
            if next_cost >= best_cost.get(successor, 10**18):
                continue
            best_cost[successor] = next_cost
            parents[successor] = (state, action)
            serial += 1
            heappush(
                frontier,
                (
                    next_cost + _heuristic(problem, successor),
                    next_cost,
                    serial,
                    successor,
                ),
            )
        frontier_peak = max(frontier_peak, len(frontier))
    raise NoSymbolicPlan("Deterministic symbolic search found no valid plan")


def independent_replay(
    problem: SymbolicProblem, plan: Iterable[SymbolicAction]
) -> dict:
    """Replay a plan without calling ``apply_action`` or planner successors."""
    state = set(problem.initial_atoms)
    steps = []
    for index, action in enumerate(plan):
        known_action = action in problem.actions
        missing = sorted(action.positive_preconditions - state)
        forbidden = sorted(action.negative_preconditions.intersection(state))
        failure = None
        if not known_action:
            failure = {"unknown_or_malformed_ground_action": action.render()}
        elif missing or forbidden:
            failure = {
                "missing_positive_preconditions": [list(atom) for atom in missing],
                "violated_negative_preconditions": [list(atom) for atom in forbidden],
            }
        steps.append(
            {
                "step": index,
                "operator": action.name.upper(),
                "arguments": list(action.arguments),
                "status": "VALID" if failure is None else "INVALID",
                "failure": failure,
            }
        )
        if failure is not None:
            return {
                "status": "INVALID",
                "goal_status": "NOT_EVALUATED",
                "failed_step": index,
                "steps": steps,
            }
        state.difference_update(action.delete_effects)
        state.update(action.add_effects)
    missing_goals = sorted(problem.goal_atoms - state)
    return {
        "status": "VALID" if not missing_goals else "INVALID",
        "goal_status": "GOAL_SATISFIED" if not missing_goals else "GOAL_NOT_SATISFIED",
        "missing_goals": [list(atom) for atom in missing_goals],
        "final_atoms": [list(atom) for atom in sorted(state)],
        "steps": steps,
        "validator": "independent_symbolic_replay_v1",
        "uses_planner_transition": False,
    }
