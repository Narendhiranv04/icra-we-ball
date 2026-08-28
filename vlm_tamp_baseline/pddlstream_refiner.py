"""PDDLStream task-and-motion refinement for grounded VLM-TAMP subgoals."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

from baseline_common.models import Action, Observation
from mujoco_scenes.kitchen_execution_policy import (
    CONTAINER_WORKSPACES as EXECUTION_CONTAINER_WORKSPACES,
    KitchenWorkspace,
)

from .models import RefinementFailure, Subgoal
from .pddlstream_dependency import activate_pddlstream
from .refiner import RefinementResult, held_object, subgoal_satisfied


PAPER_MAX_TAMP_TRIALS = 3
PAPER_MAX_SKELETONS = 12
PAPER_MAX_COMPLEXITY = 5
PAPER_PLANNING_TIMEOUT_SECONDS = 60.0
WORKSPACES = tuple(workspace.value for workspace in KitchenWorkspace)
CONTAINER_WORKSPACES = {
    region: workspace.value
    for region, workspace in EXECUTION_CONTAINER_WORKSPACES.items()
}
RECEPTACLE_LABELS = frozenset({"bowl", "cup", "glass", "mug"})
INITIAL_TABLE_REGIONS = frozenset({"INITIAL", "TABLE", "TABLETOP"})


def _observed_source_region(context: Mapping[str, Any]) -> str:
    """Translate Phase-1 provenance into the execution domain's table region."""
    container = context.get("source_container")
    if container:
        return str(container)
    region = str(context.get("observed_source_region") or "countertop")
    source_kind = str(context.get("source_kind") or "").upper()
    if region.upper() in INITIAL_TABLE_REGIONS or source_kind == "TABLE":
        return "countertop"
    return region


@dataclass(frozen=True)
class StreamCertificate:
    """Hashable record of one sampled continuous parameter/collision check."""

    certificate_id: str
    stream: str
    arguments: tuple[str, ...]
    payload_json: str = field(compare=False, hash=False, repr=False)

    @property
    def payload(self) -> Mapping[str, Any]:
        return json.loads(self.payload_json)

    def __repr__(self) -> str:
        return self.certificate_id


class StreamOracle(Protocol):
    def certify(
        self,
        stream: str,
        arguments: tuple[str, ...],
        *,
        trial: int,
        observation: Observation,
    ) -> Mapping[str, Any] | None:
        """Return sampled parameters/telemetry, or None when infeasible."""


class KitchenGeometryOracle:
    """Measured-geometry stream boundary used by the MuJoCo kitchen adapter.

    The oracle deliberately samples only parameters that are already grounded
    in the frozen observed inventory. Collision/IK failures from the live skill
    backend are returned to the VLM-TAMP reprompt loop. A custom oracle may
    perform stronger offline checks without changing the PDDL domain.
    """

    def __init__(self, inventory: Mapping[str, Any], runtime: Any | None = None):
        self.runtime = runtime
        self.rows = {
            str(row["generic_object_id"]): row
            for row in inventory.get("objects", ())
        }

    def certify(
        self,
        stream: str,
        arguments: tuple[str, ...],
        *,
        trial: int,
        observation: Observation,
    ) -> Mapping[str, Any] | None:
        if stream == "motion":
            source, target = arguments
            if source not in WORKSPACES or target not in WORKSPACES:
                return None
            probe = getattr(self.runtime, "probe_base_motion", None)
            if callable(probe):
                return probe(source, target)
            return {"source": source, "target": target, "validated": "execution"}
        if stream == "inspect":
            region, workspace = arguments
            if CONTAINER_WORKSPACES.get(region) != workspace:
                return None
            return {"region": region, "workspace": workspace}
        if stream == "pick":
            object_id, location, workspace = arguments
            row = self.rows.get(object_id)
            if row is None:
                return None
            expected_location = self._object_location(object_id, observation)
            if location != expected_location:
                return None
            if self._workspace_for_location(location, observation) != workspace:
                return None
            return self._pick_payload(object_id, location, workspace)
        if stream == "pick-object":
            object_id, target, location, workspace = arguments
            if (
                object_id not in self.rows
                or not self._is_receptacle(target)
                or self._object_location(object_id, observation) != target
                or self._object_location(target, observation) != location
                or self._workspace_for_location(location, observation) != workspace
            ):
                return None
            payload = self._pick_payload(object_id, target, workspace)
            if payload is None:
                return None
            return {
                **payload,
                "target_receptacle": target,
                "target_location": location,
            }
        if stream == "place":
            object_id, region, workspace = arguments
            if object_id not in self.rows:
                return None
            if self._workspace_for_location(region, observation) != workspace:
                return None
            return self._place_payload(object_id, region, workspace)
        if stream == "place-object":
            object_id, target, location, workspace = arguments
            if (
                object_id not in self.rows
                or not self._is_receptacle(target)
                or self._object_location(target, observation) != location
                or self._workspace_for_location(location, observation) != workspace
            ):
                return None
            payload = self._place_payload(object_id, target, workspace)
            if payload is None:
                return None
            return {
                **payload,
                "target_receptacle": target,
                "target_location": location,
            }
        if stream in {"pour", "stir"}:
            first, target, location, workspace = arguments
            if (
                first not in self.rows
                or not self._is_receptacle(target)
                or self._object_location(target, observation) != location
            ):
                return None
            if self._workspace_for_location(location, observation) != workspace:
                return None
            return {
                "source_or_tool": first,
                "target": target,
                "target_location": location,
                "workspace": workspace,
                "geometry_validation": "live_phase_c",
            }
        return None

    def _pick_payload(
        self, object_id: str, location: str, workspace: str
    ) -> Mapping[str, Any] | None:
        row = self.rows[object_id]
        dimensions = row.get("observed_dimensions_m", {})
        if not dimensions or any(value is None for value in dimensions.values()):
            return None
        return {
            "object_id": object_id,
            "location": location,
            "workspace": workspace,
            "observed_dimensions_m": dimensions,
            "grasp_family": row.get("selected_functions", ()),
        }

    def _place_payload(
        self, object_id: str, destination: str, workspace: str
    ) -> Mapping[str, Any] | None:
        resolver = getattr(
            getattr(getattr(self.runtime, "phase_b", None), "manipulation", None),
            "placement_resolver",
            None,
        )
        if resolver is not None:
            try:
                target = resolver.resolve(object_id, destination)
            except ValueError:
                return None
            return {
                "object_id": object_id,
                "region": destination,
                "workspace": workspace,
                "target_position_world_m": list(target.target_position_world_m),
                "target_yaw_world_rad": target.target_yaw_world_rad,
            }
        return {
            "object_id": object_id,
            "region": destination,
            "workspace": workspace,
        }

    def _is_receptacle(self, object_id: str) -> bool:
        row = self.rows.get(object_id)
        return bool(
            row
            and str(row.get("semantic_label", "")).lower() in RECEPTACLE_LABELS
        )

    def _object_location(
        self, object_id: str, observation: Observation
    ) -> str:
        entity = next(
            (item for item in observation.entities if item.entity_id == object_id),
            None,
        )
        if entity is not None:
            dynamic = (
                entity.facts.get("region_id")
                or entity.facts.get("location")
                or entity.facts.get("source_region")
            )
            if dynamic:
                normalized = str(dynamic)
                return (
                    "countertop"
                    if normalized.upper() in INITIAL_TABLE_REGIONS
                    else normalized
                )
        context = self.rows[object_id].get("source_context", {})
        return _observed_source_region(context)

    def _workspace_for_location(
        self, location: str, observation: Observation
    ) -> str:
        visited: set[str] = set()
        while location in self.rows and location not in visited:
            visited.add(location)
            location = self._object_location(location, observation)
        return CONTAINER_WORKSPACES.get(location, "home")


class LivingRoomGeometryOracle(KitchenGeometryOracle):
    """Observed-footprint stream oracle for planning-only Living Room trials.

    This adapter uses measured payload and support dimensions only.  It does
    not receive the functional assignment or the expected GT action sequence.
    """

    def __init__(
        self,
        inventory: Mapping[str, Any],
        region_registry: Mapping[str, Any],
    ):
        super().__init__(inventory)
        self.regions = dict(region_registry.get("regions", {}))

    def certify(
        self,
        stream: str,
        arguments: tuple[str, ...],
        *,
        trial: int,
        observation: Observation,
    ) -> Mapping[str, Any] | None:
        if stream == "place":
            object_id, region_id, workspace = arguments
            row = self.rows.get(object_id)
            region = self.regions.get(region_id)
            if row is None or region is None or workspace != "home":
                return None
            dimensions = row.get("observed_dimensions_m", {})
            geometry = region.get("geometry", {})
            length = geometry.get("support_length_m", {}).get("value")
            width = geometry.get("support_width_m", {}).get("value")
            object_length = dimensions.get("length")
            object_width = dimensions.get("width")
            if None in {length, width, object_length, object_width}:
                return None
            fits = (
                object_length <= length and object_width <= width
            ) or (
                object_length <= width and object_width <= length
            )
            if not fits:
                return None
            return {
                "object_id": object_id,
                "region": region_id,
                "workspace": workspace,
                "geometry_validation": "observed_footprint_inside_support_bounds",
                "payload_footprint_m": [object_length, object_width],
                "support_footprint_m": [length, width],
            }
        return super().certify(
            stream,
            arguments,
            trial=trial,
            observation=observation,
        )


@dataclass(frozen=True)
class PDDLStreamProtocol:
    max_tamp_trials: int = PAPER_MAX_TAMP_TRIALS
    max_skeletons: int = PAPER_MAX_SKELETONS
    max_complexity: int = PAPER_MAX_COMPLEXITY
    timeout_seconds: float = PAPER_PLANNING_TIMEOUT_SECONDS
    algorithm: str = "adaptive"
    planner: str = "ff-astar"

    def __post_init__(self) -> None:
        if not 1 <= self.max_tamp_trials <= 3:
            raise ValueError("max_tamp_trials must be between 1 and 3")
        if self.max_skeletons < 1 or self.max_complexity < 1:
            raise ValueError("PDDLStream limits must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("PDDLStream timeout must be positive")
        if not self.algorithm or not self.planner:
            raise ValueError("PDDLStream algorithm and planner must not be empty")


class PDDLStreamSubgoalRefiner:
    """Solve each formal subgoal with the paper's PDDLStream retry protocol."""

    def __init__(
        self,
        inventory: Mapping[str, Any],
        oracle: StreamOracle,
        *,
        protocol: PDDLStreamProtocol = PDDLStreamProtocol(),
    ):
        self.inventory = inventory
        self.rows = {
            str(row["generic_object_id"]): row
            for row in inventory.get("objects", ())
        }
        self.oracle = oracle
        self.protocol = protocol
        self.last_trace: dict[str, Any] = {}
        self._certificate_counter = 0
        pddl_directory = Path(__file__).parent / "pddl"
        self._domain_pddl = (pddl_directory / "domain.pddl").read_text(
            encoding="utf-8"
        )
        self._stream_pddl = (pddl_directory / "stream.pddl").read_text(
            encoding="utf-8"
        )
        self._dependency_activated = False

    def refine(self, subgoal: Subgoal, observation: Observation) -> RefinementResult:
        if subgoal_satisfied(subgoal, observation):
            return RefinementResult()
        if not self._dependency_activated:
            activate_pddlstream()
            self._dependency_activated = True
        attempts: list[dict[str, Any]] = []
        started = time.perf_counter()
        for trial in range(1, self.protocol.max_tamp_trials + 1):
            object_ids = self._object_set(trial, subgoal, observation)
            solution = self._solve(trial, object_ids, subgoal, observation)
            attempts.append(solution["trace"])
            if solution["actions"] is not None:
                self.last_trace = {
                    "backend": "pddlstream",
                    "protocol": self._protocol_dict(),
                    "attempts": attempts,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                return RefinementResult(tuple(solution["actions"]))
        self.last_trace = {
            "backend": "pddlstream",
            "protocol": self._protocol_dict(),
            "attempts": attempts,
            "elapsed_seconds": time.perf_counter() - started,
        }
        collided = tuple(
            sorted(
                {
                    str(item)
                    for attempt in attempts
                    for item in attempt.get("collided_objects", ())
                }
            )
        )
        return RefinementResult(
            failure=RefinementFailure(
                "tamp_refinement_failed",
                f"PDDLStream found no feasible refinement after "
                f"{self.protocol.max_tamp_trials} trials.",
                subgoal,
                collided,
            )
        )

    def _object_set(
        self, trial: int, subgoal: Subgoal, observation: Observation
    ) -> tuple[str, ...]:
        goal_objects = {
            value
            for value in subgoal.arguments.values()
            if value in self.rows
        }
        held = held_object(observation)
        if held in self.rows:
            goal_objects.add(held)
        if trial >= 2:
            goal_objects.update(observation.object_ids & self.rows.keys())
        if trial >= 3:
            goal_objects.update(self.rows)
        return tuple(sorted(goal_objects))

    def _solve(
        self,
        trial: int,
        object_ids: Sequence[str],
        subgoal: Subgoal,
        observation: Observation,
    ) -> dict[str, Any]:
        from pddlstream.algorithms.meta import solve
        from pddlstream.language.constants import PDDLProblem
        from pddlstream.language.generator import from_fn

        sampled: list[StreamCertificate] = []
        stream_errors: list[dict[str, Any]] = []

        def certificate(stream: str, *arguments: str):
            normalized = tuple(map(str, arguments))
            # ``place-object`` is a task relation, not a generic way to free
            # the gripper.  Without this guard, a STIR refinement can park a
            # held liquid vessel inside the target mug because ``place`` and
            # ``place-object`` have equal unit cost.  Only expose this stream
            # when it is the exact relation requested by a PLACED subgoal.
            if stream == "place-object" and not (
                subgoal.predicate == "PLACED"
                and subgoal.arguments.get("object_id") == normalized[0]
                and subgoal.arguments.get("region_id") == normalized[1]
            ):
                return None
            if (
                stream == "place"
                and subgoal.predicate != "PLACED"
                and normalized[0] == held_object(observation)
                and normalized[1] != self._source_region(normalized[0])
            ):
                return None
            try:
                payload = self.oracle.certify(
                    stream,
                    normalized,
                    trial=trial,
                    observation=observation,
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                stream_errors.append(
                    {
                        "stream": stream,
                        "arguments": list(normalized),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return None
            if payload is None:
                return None
            self._certificate_counter += 1
            item = StreamCertificate(
                f"{stream}_{self._certificate_counter:05d}",
                stream,
                tuple(map(str, arguments)),
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
            sampled.append(item)
            return item

        stream_map = {
            "sample-motion": from_fn(
                lambda source, target: self._one(certificate("motion", source, target))
            ),
            "sample-inspect": from_fn(
                lambda region, workspace: self._one(
                    certificate("inspect", region, workspace)
                )
            ),
            "sample-pick": from_fn(
                lambda object_id, location, workspace: self._two_pick(
                    object_id,
                    certificate("pick", object_id, location, workspace),
                )
            ),
            "sample-pick-object": from_fn(
                lambda object_id, target, location, workspace: self._two_pick(
                    object_id,
                    certificate(
                        "pick-object", object_id, target, location, workspace
                    ),
                )
            ),
            "sample-place": from_fn(
                lambda object_id, region, workspace: self._two_place(
                    region,
                    certificate("place", object_id, region, workspace),
                )
            ),
            "sample-place-object": from_fn(
                lambda object_id, target, location, workspace: self._two_place(
                    target,
                    certificate(
                        "place-object", object_id, target, location, workspace
                    ),
                )
            ),
            "sample-pour": from_fn(
                lambda source, target, location, workspace: self._one(
                    certificate("pour", source, target, location, workspace)
                )
            ),
            "sample-stir": from_fn(
                lambda tool, target, location, workspace: self._one(
                    certificate("stir", tool, target, location, workspace)
                )
            ),
        }
        init = self._initial_facts(object_ids, observation)
        goal = self._goal(subgoal)
        problem = PDDLProblem(
            self._domain_pddl,
            {},
            self._stream_pddl,
            stream_map,
            init,
            goal,
        )
        planner_output = StringIO()
        with redirect_stdout(planner_output), redirect_stderr(planner_output):
            raw_solution = solve(
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
        if raw_solution is None:
            plan, cost, _evaluations = None, float("inf"), None
        else:
            plan, cost, _evaluations = raw_solution
        actions = None if plan is None else self._to_actions(plan)
        return {
            "actions": actions,
            "trace": {
                "trial": trial,
                "object_reducer": ("goal_related", "visible", "all")[trial - 1],
                "planning_objects": list(object_ids),
                "status": "SOLVED" if plan is not None else "INFEASIBLE",
                "cost": None if plan is None else float(cost),
                "pddl_plan": [] if plan is None else [
                    {
                        "operator": str(name),
                        "arguments": [str(value) for value in arguments],
                    }
                    for name, arguments in plan
                ],
                "stream_certificates": [
                    {
                        "id": item.certificate_id,
                        "stream": item.stream,
                        "arguments": list(item.arguments),
                        "payload": item.payload,
                    }
                    for item in sampled
                ],
                "stream_errors": stream_errors,
                "planner_log": planner_output.getvalue(),
            },
        }

    @staticmethod
    def _one(value: StreamCertificate | None):
        return None if value is None else (value,)

    @staticmethod
    def _two_pick(object_id: str, value: StreamCertificate | None):
        return None if value is None else (f"grasp_{object_id}", value)

    @staticmethod
    def _two_place(region: str, value: StreamCertificate | None):
        return None if value is None else (f"pose_{region}", value)

    def _initial_facts(
        self, object_ids: Sequence[str], observation: Observation
    ) -> list[tuple[Any, ...]]:
        facts: list[tuple[Any, ...]] = [("=", ("total-cost",), 0)]
        current_workspace = str(
            observation.robot.get("workspace")
            or observation.robot.get("location")
            or "home"
        )
        if current_workspace not in WORKSPACES:
            current_workspace = "home"
        for workspace in WORKSPACES:
            facts.append(("Workspace", workspace))
        facts.append(("AtRobot", current_workspace))

        region_state = {item.region_id: item for item in observation.regions}
        regions = set(region_state) | {"countertop", "serving_area"}
        for row in self.rows.values():
            context = row.get("source_context", {})
            regions.add(_observed_source_region(context))
        for region in sorted(regions):
            facts.extend(
                [
                    ("Region", region),
                    ("RequiresWorkspace", region, self._workspace(region)),
                ]
            )
            state = region_state.get(region)
            is_open = region in {"countertop", "serving_area"} or (
                state is not None and state.state.lower() == "open"
            )
            if is_open:
                facts.extend([("Open", region), ("Accessible", region)])
            else:
                facts.append(("Closed", region))
            if state is not None and state.inspected:
                facts.append(("Inspected", region))

        held = held_object(observation)
        if held is None:
            facts.append(("HandEmpty",))
        else:
            facts.append(("Holding", held))
        for object_id in object_ids:
            facts.append(("Movable", object_id))
            if self._is_receptacle(object_id):
                facts.append(("Receptacle", object_id))
            if object_id == held:
                continue
            location = self._object_location(object_id, observation)
            facts.append(("At", object_id, location))
        for entity in observation.entities:
            poured = entity.facts.get("poured_from", ())
            stirred = entity.facts.get("stirred_with", ())
            for source in ([poured] if isinstance(poured, str) else poured):
                facts.append(("Poured", str(source), entity.entity_id))
            for tool in ([stirred] if isinstance(stirred, str) else stirred):
                facts.append(("Stirred", str(tool), entity.entity_id))
        return facts

    def _source_region(self, object_id: str) -> str:
        context = self.rows[object_id].get("source_context", {})
        return _observed_source_region(context)

    def _is_receptacle(self, object_id: str) -> bool:
        row = self.rows.get(object_id)
        return bool(
            row
            and str(row.get("semantic_label", "")).lower() in RECEPTACLE_LABELS
        )

    def _object_location(
        self, object_id: str, observation: Observation
    ) -> str:
        entity = next(
            (item for item in observation.entities if item.entity_id == object_id),
            None,
        )
        if entity is not None:
            dynamic = (
                entity.facts.get("region_id")
                or entity.facts.get("location")
                or entity.facts.get("source_region")
            )
            if dynamic:
                normalized = str(dynamic)
                return (
                    "countertop"
                    if normalized.upper() in INITIAL_TABLE_REGIONS
                    else normalized
                )
        return self._source_region(object_id)

    def _workspace(self, location: str) -> str:
        if location in CONTAINER_WORKSPACES:
            return CONTAINER_WORKSPACES[location]
        row = self.rows.get(location)
        if row is not None:
            return str(
                row.get("source_context", {}).get("required_workspace", "home")
            )
        return "home"

    @staticmethod
    def _goal(subgoal: Subgoal) -> tuple[Any, ...]:
        values = subgoal.arguments
        goals = {
            "INSPECTED": ("Inspected", values.get("region_id")),
            "HOLDING": ("Holding", values.get("object_id")),
            "PLACED": ("At", values.get("object_id"), values.get("region_id")),
            "POURED": ("Poured", values.get("source_id"), values.get("target_id")),
            "STIRRED": ("Stirred", values.get("tool_id"), values.get("target_id")),
        }
        if subgoal.predicate not in goals:
            raise ValueError(f"PDDLStream kitchen domain cannot refine {subgoal.predicate}")
        return goals[subgoal.predicate]

    @staticmethod
    def _to_actions(plan: Sequence[tuple[str, Sequence[Any]]]) -> list[Action]:
        actions: list[Action] = []
        for operator, raw in plan:
            name = str(operator).lower()
            values = list(raw)
            if name == "move":
                continue
            if name.startswith("inspect-"):
                actions.append(Action("INSPECT", {"region_id": str(values[0])}))
            elif name in {"pick", "pick-object"}:
                actions.append(Action("PICK", {"object_id": str(values[0])}))
            elif name in {"place", "place-object"}:
                actions.append(
                    Action(
                        "PLACE",
                        {"object_id": str(values[0]), "region_id": str(values[1])},
                    )
                )
            elif name == "pour":
                actions.append(
                    Action(
                        "POUR",
                        {"source_id": str(values[0]), "target_id": str(values[1])},
                    )
                )
            elif name == "stir":
                actions.append(
                    Action(
                        "STIR",
                        {"tool_id": str(values[0]), "target_id": str(values[1])},
                    )
                )
            else:
                raise ValueError(f"Unsupported PDDLStream operator {operator!r}")
        return actions

    def _protocol_dict(self) -> dict[str, Any]:
        return {
            "max_tamp_trials": self.protocol.max_tamp_trials,
            "object_reducers": ["goal_related", "visible", "all"],
            "max_skeletons": self.protocol.max_skeletons,
            "max_complexity": self.protocol.max_complexity,
            "timeout_seconds_per_trial": self.protocol.timeout_seconds,
            "algorithm": self.protocol.algorithm,
            "planner": self.protocol.planner,
        }
