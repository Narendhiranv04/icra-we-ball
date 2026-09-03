"""PDDLStream symbolic refinement for the planning-only Workshop benchmark."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from baseline_common.models import Action, Observation

from .models import RefinementFailure, Subgoal
from .pddlstream_dependency import activate_pddlstream
from .pddlstream_refiner import PDDLStreamProtocol
from .refiner import RefinementResult, held_object, subgoal_satisfied


class WorkshopPDDLStreamRefiner:
    """Refine one Workshop subgoal through the actual PDDLStream solver.

    This planning-only domain intentionally has no continuous streams: it
    validates the symbolic skeleton only.  The common closed-cell observation
    contract is preserved; hidden storage contents never enter the VLM prompt.
    """

    def __init__(self, runtime: Any, *, protocol: PDDLStreamProtocol = PDDLStreamProtocol()):
        self.runtime = runtime
        self.protocol = protocol
        directory = Path(__file__).with_name("pddl")
        self.domain_pddl = (directory / "workshop_domain.pddl").read_text(encoding="utf-8")
        self.stream_pddl = (directory / "workshop_stream.pddl").read_text(encoding="utf-8")
        self._activated = False
        self.last_trace: dict[str, Any] = {}

    def refine(self, subgoal: Subgoal, observation: Observation) -> RefinementResult:
        if subgoal_satisfied(subgoal, observation):
            return RefinementResult()
        if not self._activated:
            activate_pddlstream()
            self._activated = True
        started = time.perf_counter()
        try:
            actions, trace = self._solve(subgoal, observation)
        except (RuntimeError, ValueError, KeyError) as error:
            self.last_trace = {"backend": "pddlstream", "error": f"{type(error).__name__}: {error}"}
            return RefinementResult(failure=RefinementFailure("tamp_refinement_failed", str(error), subgoal))
        self.last_trace = {"backend": "pddlstream", "protocol": self._protocol_dict(), "elapsed_seconds": time.perf_counter() - started, **trace}
        if actions is None:
            return RefinementResult(failure=RefinementFailure("tamp_refinement_failed", "PDDLStream found no symbolic Workshop refinement.", subgoal))
        return RefinementResult(tuple(actions))

    def _solve(self, subgoal: Subgoal, observation: Observation) -> tuple[list[Action] | None, dict[str, Any]]:
        from pddlstream.algorithms.meta import solve
        from pddlstream.language.constants import PDDLProblem

        init = self._initial_facts(observation)
        problem = PDDLProblem(self.domain_pddl, {}, self.stream_pddl, {}, init, self._goal(subgoal))
        planner_output = StringIO()
        with redirect_stdout(planner_output), redirect_stderr(planner_output):
            raw = solve(
                problem,
                algorithm=self.protocol.algorithm,
                planner=self.protocol.planner,
                max_time=self.protocol.timeout_seconds,
                max_complexity=self.protocol.max_complexity,
                max_skeletons=self.protocol.max_skeletons,
                unit_costs=True,
                log_failures=False,
                verbose=False,
            )
        plan, cost = (None, None) if raw is None else (raw[0], float(raw[1]))
        return (
            None if plan is None else self._to_actions(plan),
            {
                "status": "INFEASIBLE" if plan is None else "SOLVED",
                "cost": cost,
                "pddl_plan": [] if plan is None else [
                    {"operator": str(operator), "arguments": [str(value) for value in values]}
                    for operator, values in plan
                ],
                "planner_log": planner_output.getvalue(),
                "continuous_streams": "NONE_PLANNING_ONLY",
            },
        )

    def _initial_facts(self, observation: Observation) -> list[tuple[Any, ...]]:
        facts: list[tuple[Any, ...]] = [("=", ("total-cost",), 0), ("HandEmpty",)]
        held = held_object(observation)
        if held:
            facts.remove(("HandEmpty",))
            facts.append(("Holding", held))
        for region in observation.regions:
            facts.extend([("Region", region.region_id), ("Destination", region.region_id)])
            if region.region_id in self.runtime.storage_region_ids:
                facts.append(("Storage", region.region_id))
            if region.state == "open":
                facts.extend([("Open", region.region_id), ("Accessible", region.region_id)])
            if region.inspected:
                facts.append(("Inspected", region.region_id))
        for entity in observation.entities:
            if entity.entity_id == self.runtime.target_object_id:
                facts.extend([("Target", entity.entity_id), ("Destination", entity.entity_id), ("Accessible", entity.entity_id)])
            else:
                facts.append(("Movable", entity.entity_id))
                if self.runtime.is_fastener(entity.entity_id):
                    facts.append(("CompatibleFastener", entity.entity_id))
                if self.runtime.is_driver(entity.entity_id):
                    facts.append(("CompatibleDriver", entity.entity_id))
            location = entity.facts.get("region_id") or entity.facts.get("location")
            if location and location != "held":
                facts.append(("At", entity.entity_id, str(location)))
            for fastener in entity.facts.get("inserted_fasteners", ()):
                facts.append(("Inserted", str(fastener), entity.entity_id))
            fastened_with = entity.facts.get("fastened_with")
            if fastened_with and self.runtime.inserted:
                facts.append(("Fastened", str(fastened_with), self.runtime.inserted[0], entity.entity_id))
        return facts

    @staticmethod
    def _goal(subgoal: Subgoal) -> tuple[Any, ...]:
        values = subgoal.arguments
        if subgoal.predicate == "INSPECTED":
            return "Inspected", values["region_id"]
        if subgoal.predicate == "HOLDING":
            return "Holding", values["object_id"]
        if subgoal.predicate == "PLACED":
            return "At", values["object_id"], values["region_id"]
        if subgoal.predicate == "INSERTED":
            return "Inserted", values["fastener_id"], values["target_id"]
        if subgoal.predicate == "FASTENED":
            return "Fastened", values["tool_id"], values["fastener_id"], values["target_id"]
        raise ValueError(f"Workshop PDDLStream cannot refine {subgoal.predicate}")

    @staticmethod
    def _to_actions(plan: Sequence[tuple[str, Sequence[Any]]]) -> list[Action]:
        mapped = {
            "inspect": ("INSPECT", ("region_id",)),
            "pick": ("PICK", ("object_id",)),
            "place": ("PLACE", ("object_id", "region_id")),
            "insert": ("INSERT", ("fastener_id", "target_id")),
            "fasten": ("FASTEN", ("tool_id", "fastener_id", "target_id")),
        }
        actions = []
        for operator, values in plan:
            key = str(operator).lower()
            if key not in mapped:
                raise ValueError(f"Unsupported Workshop PDDL operator {operator!r}")
            skill, names = mapped[key]
            actions.append(Action(skill, {name: str(value) for name, value in zip(names, values)}))
        return actions

    def _protocol_dict(self) -> dict[str, Any]:
        return {
            "max_tamp_trials": 1,
            "max_skeletons": self.protocol.max_skeletons,
            "max_complexity": self.protocol.max_complexity,
            "timeout_seconds": self.protocol.timeout_seconds,
            "algorithm": self.protocol.algorithm,
            "planner": self.protocol.planner,
            "continuous_streams": "NONE_PLANNING_ONLY",
        }
