"""Executed(i) symbolic search followed by bounded continuous sampling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Protocol, Sequence

from baseline_common.models import Observation

from .models import Action, Constraint, PlanningResult, PlanSketch


PAPER_MAX_SAMPLES_PER_ACTION = 500
PAPER_MAX_SKELETONS = 5


class SampleOracle(Protocol):
    def __call__(
        self,
        action: Action,
        constraints: Sequence[Constraint],
        sample_index: int,
    ) -> bool: ...


@dataclass(frozen=True)
class _State:
    holding: str | None
    locations: tuple[tuple[str, str], ...]
    opened: frozenset[str]
    facts: frozenset[str]
    executed: int

    @property
    def location_map(self) -> dict[str, str]:
        return dict(self.locations)


def _initial_state(observation: Observation) -> _State:
    holding = observation.robot.get("holding")
    holding = str(holding) if holding else None
    locations = []
    for entity in observation.entities:
        location = entity.facts.get("region_id") or entity.facts.get("location")
        if location and location != "held":
            locations.append((entity.entity_id, str(location)))
    opened = frozenset(
        region.region_id for region in observation.regions if region.state == "open"
    )
    facts = {f"at({object_id},{region_id})" for object_id, region_id in locations}
    facts.update(f"open({region_id})" for region_id in opened)
    if holding:
        facts.add(f"holding({holding})")
    return _State(holding, tuple(sorted(locations)), opened, frozenset(facts), 0)


def _transition(state: _State, action: Action, sketch: Sequence[Action]) -> _State | None:
    locations = state.location_map
    facts = set(state.facts)
    holding = state.holding
    name = action.operator
    args = action.arguments
    if name == "PICK":
        if holding is not None or args[0] not in locations:
            return None
        holding = args[0]
        locations.pop(args[0], None)
        facts = {fact for fact in facts if not fact.startswith(f"at({args[0]},")}
        facts.add(f"holding({args[0]})")
    elif name == "PLACE":
        if holding != args[0]:
            return None
        holding = None
        locations[args[0]] = args[1]
        facts.discard(f"holding({args[0]})")
        facts.add(f"at({args[0]},{args[1]})")
    elif name in {"POUR", "STIR"}:
        if holding != args[0] or args[1] not in locations:
            return None
        predicate = "poured" if name == "POUR" else "stirred"
        facts.add(f"{predicate}({args[0]},{args[1]})")
    elif name == "PLACE_SERVING_UTENSIL":
        if holding != args[0] or args[1] not in locations:
            return None
        holding = None
        locations[args[0]] = args[1]
        facts.discard(f"holding({args[0]})")
        facts.add(f"at({args[0]},{args[1]})")
        facts.add(f"served_with({args[1]},{args[0]})")
    elif name == "OPEN":
        if args[0] in state.opened:
            return None
    else:
        return None
    opened = state.opened | ({args[0]} if name == "OPEN" else set())
    if name == "OPEN":
        facts.add(f"open({args[0]})")
    executed = state.executed
    if executed < len(sketch) and action == sketch[executed]:
        executed += 1
    return _State(
        holding,
        tuple(sorted(locations.items())),
        frozenset(opened),
        frozenset(facts),
        executed,
    )


_LITERAL = re.compile(r"([a-z_]+)\(([^()]*)\)", re.IGNORECASE)
_PREDICATE_ARITY = {
    "at": 2,
    "holding": 1,
    "open": 1,
    "poured": 2,
    "stirred": 2,
    "served_with": 2,
}


def _goal_facts(literals: Sequence[str], observation: Observation) -> frozenset[str]:
    known = observation.object_ids | observation.region_ids
    result = set()
    for literal in literals:
        match = _LITERAL.fullmatch(literal.replace(" ", ""))
        if match is None:
            raise ValueError(f"unsupported goal literal syntax {literal!r}")
        predicate = match.group(1).lower()
        arguments = tuple(part for part in match.group(2).split(",") if part)
        if predicate not in _PREDICATE_ARITY or len(arguments) != _PREDICATE_ARITY[predicate]:
            raise ValueError(f"unsupported goal literal {literal!r}")
        if any(argument not in known for argument in arguments):
            raise ValueError(f"goal literal references an unobserved ID: {literal!r}")
        result.add(f"{predicate}({','.join(arguments)})")
    return frozenset(result)


def constrained_breadth_first_search(
    observation: Observation,
    grounded_actions: Sequence[Action],
    sketch: Sequence[Action],
    goal_literals: Sequence[str] = (),
    *,
    max_expansions: int = 100_000,
) -> tuple[Action, ...] | None:
    """Find the shortest plan containing the VLM sketch as a subsequence."""
    initial = _initial_state(observation)
    goals = _goal_facts(goal_literals, observation)
    queue = deque([(initial, ())])
    visited = {initial}
    expansions = 0
    while queue and expansions < max_expansions:
        state, path = queue.popleft()
        expansions += 1
        if state.executed == len(sketch) and goals <= state.facts:
            return path
        for action in grounded_actions:
            successor = _transition(state, action, sketch)
            if successor is None or successor in visited:
                continue
            visited.add(successor)
            queue.append((successor, path + (action,)))
    return None


def search_then_sample(
    observation: Observation,
    grounded_actions: Sequence[Action],
    sketch: PlanSketch,
    constraints: Sequence[Constraint],
    oracle: SampleOracle,
    *,
    max_samples_per_action: int = PAPER_MAX_SAMPLES_PER_ACTION,
    max_skeletons: int = PAPER_MAX_SKELETONS,
) -> PlanningResult:
    """Run OWL-TAMP's symbolic-search then continuous-sampling protocol.

    This benchmark has deterministic discrete operators, so a single shortest
    skeleton is produced by the Executed(i) search.  The public five-skeleton
    budget is retained for provenance and future domains with skeleton choices.
    """
    if sketch.status == "NO_PLAN":
        return PlanningResult("NO_PLAN", sketch, (), (), 0, 0, "model returned NO_PLAN")
    try:
        skeleton = constrained_breadth_first_search(
            observation, grounded_actions, sketch.actions, sketch.goal_literals
        )
    except ValueError as error:
        return PlanningResult(
            "INVALID_GOAL_LITERALS", sketch, (), tuple(constraints), 0, 0, str(error)
        )
    if skeleton is None:
        return PlanningResult(
            "NO_SYMBOLIC_PLAN", sketch, (), tuple(constraints), 1, 0,
            "no plan satisfies the Executed(i) subsequence constraints",
        )
    samples = 0
    by_action = {index: [] for index in range(len(sketch.actions))}
    for constraint in constraints:
        by_action.setdefault(constraint.action_index, []).append(constraint)
    sketch_cursor = 0
    for action in skeleton:
        sketch_index = -1
        if (
            sketch_cursor < len(sketch.actions)
            and action == sketch.actions[sketch_cursor]
        ):
            sketch_index = sketch_cursor
            sketch_cursor += 1
        accepted = False
        for trial in range(max_samples_per_action):
            samples += 1
            if oracle(action, by_action.get(sketch_index, ()), trial):
                accepted = True
                break
        if not accepted:
            return PlanningResult(
                "NO_CONTINUOUS_PLAN", sketch, skeleton, tuple(constraints),
                min(1, max_skeletons), samples,
                f"continuous sampling exhausted for {action.operator}",
            )
    return PlanningResult(
        "PLAN", sketch, skeleton, tuple(constraints), min(1, max_skeletons), samples
    )
