"""Observed-witness to classical symbolic planning for the kitchen task.

This module deliberately starts at the perception/planning boundary. It never
imports MuJoCo, scene object names, robot code, or hidden scene metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from heapq import heappop, heappush
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from PIL import Image
import yaml

from mujoco_scenes.semantic_grounding import YOLOWorldSemanticDetector
from mujoco_scenes.task_witness import load_task_requirements


class SymbolicCompilationError(RuntimeError):
    """Raised when observed evidence is insufficient for safe compilation."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    def serializable(item: Any) -> Any:
        if isinstance(item, (set, frozenset)):
            return sorted(serializable(child) for child in item)
        if isinstance(item, tuple):
            return [serializable(child) for child in item]
        if isinstance(item, list):
            return [serializable(child) for child in item]
        if isinstance(item, dict):
            return {key: serializable(child) for key, child in item.items()}
        return item

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(serializable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _symbol(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()
    if not result:
        raise SymbolicCompilationError(f"Invalid empty PDDL symbol: {value!r}")
    if result[0].isdigit():
        result = f"id_{result}"
    return result


def _validated_label(record: dict[str, Any]) -> tuple[str | None, float]:
    semantic = record.get("semantics", {}).get("validated", {})
    if semantic.get("status") != "SUPPORTED":
        return None, -math.inf
    label = semantic.get("canonical_label")
    if not isinstance(label, str) or not label:
        return None, -math.inf
    return label.strip().lower(), float(semantic.get("mean_confidence") or 0.0)


def ground_symbolic_sources(
    run_dir: str | Path,
    *,
    checkpoint: str,
    vocabulary_path: str | Path | None = None,
    confidence_threshold: float = 0.03,
) -> dict[str, Any]:
    """Ground preparation sources from stage-zero RGB instance crops only.

    Crops were created by the normal one-to-one detector/mask association
    path. This source-specific open vocabulary is intentionally separate from
    vessel/tool vocabulary so source prompts cannot alter the functional
    witness used by the search controller.
    """
    run_path = Path(run_dir).resolve()
    vocabulary_path = Path(vocabulary_path or (
        Path(__file__).resolve().parent / "configs" /
        "symbolic_source_vocabulary.yaml"
    ))
    config = yaml.safe_load(vocabulary_path.read_text(encoding="utf-8"))
    aliases = config["canonical_labels"]
    alias_to_canonical = {
        alias.strip().lower(): canonical
        for canonical, values in aliases.items()
        for alias in values
    }
    prompts = list(alias_to_canonical)
    stage_dirs = sorted((run_path / "stages").glob("000_*"))
    if len(stage_dirs) != 1:
        raise SymbolicCompilationError("Expected exactly one stage-000 observation")
    crop_paths = sorted(stage_dirs[0].glob("semantics/cameras/*/crops/*.png"))
    by_object: dict[str, list[dict[str, Any]]] = {}
    detector = YOLOWorldSemanticDetector(
        checkpoint,
        confidence_threshold=confidence_threshold,
        inference_size=640,
        device="cpu",
        max_detections=20,
    )
    try:
        for path in crop_paths:
            match = re.search(r"(object_[A-Za-z0-9_-]+)\.png$", path.name)
            if not match:
                continue
            object_id = match.group(1)
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            detections = detector.detect(image, prompts)
            best_by_label = {}
            for detection in detections:
                canonical = alias_to_canonical.get(
                    detection.raw_label.strip().lower()
                )
                if canonical is None:
                    continue
                previous = best_by_label.get(canonical)
                if previous is None or detection.confidence > previous.confidence:
                    best_by_label[canonical] = detection
            for canonical, detection in best_by_label.items():
                by_object.setdefault(object_id, []).append({
                    "canonical_label": canonical,
                    "raw_label": detection.raw_label,
                    "confidence": detection.confidence,
                    "camera_id": path.parts[-3],
                    "rgb_crop_path": path.relative_to(run_path).as_posix(),
                })
    finally:
        detector.close()
    records = {}
    minimum_views = int(config.get("minimum_supporting_views", 2))
    for object_id, detections in sorted(by_object.items()):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for detection in detections:
            grouped.setdefault(detection["canonical_label"], []).append(detection)
        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                -len({record["camera_id"] for record in item[1]}),
                -sum(record["confidence"] for record in item[1]) / len(item[1]),
                item[0],
            ),
        )
        hypotheses = {}
        for hypothesis_label, hypothesis_support in ranked:
            hypothesis_views = len({
                record["camera_id"] for record in hypothesis_support
            })
            hypotheses[hypothesis_label] = {
                "status": (
                    "SUPPORTED" if hypothesis_views >= minimum_views
                    else "UNKNOWN"
                ),
                "mean_confidence": sum(
                    record["confidence"] for record in hypothesis_support
                ) / len(hypothesis_support),
                "supporting_view_count": hypothesis_views,
                "evidence": hypothesis_support,
            }
        label, supporting = ranked[0]
        view_count = len({record["camera_id"] for record in supporting})
        records[object_id] = {
            "status": "SUPPORTED" if view_count >= minimum_views else "UNKNOWN",
            "canonical_label": label if view_count >= minimum_views else None,
            "mean_confidence": sum(record["confidence"] for record in supporting) / len(supporting),
            "supporting_view_count": view_count,
            "evidence": supporting,
            "observation_source": "RGB_DETECTOR_INSTANCE_CROPS",
            "detector_name": detector.name,
            "checkpoint": checkpoint,
            "detector_version": detector.version,
            "label_hypotheses": hypotheses,
        }
    result = {
        "schema_version": 1,
        "inference_basis": "RGB_ONLY_SOURCE_GROUNDING",
        "vocabulary_path": str(vocabulary_path),
        "minimum_supporting_views": minimum_views,
        "objects": records,
    }
    _write_json(run_path / "symbolic_source_semantics.json", result)
    return result


def _observed_location(record: dict[str, Any], initial_region: str) -> dict[str, Any]:
    """Compile location only from accepted inspection provenance.

    The intentionally narrow key allow-list prevents legacy ``source_region``
    and ``oracle_source_region`` metadata from entering the planner state.
    """
    evidence_region = record.get("last_evidence_source_region")
    evidence_stage = record.get("last_evidence_stage")
    if not isinstance(evidence_region, str) or evidence_stage is None:
        raise SymbolicCompilationError(
            f"{record.get('object_id')} lacks stage-local location provenance"
        )
    region = initial_region if evidence_region == "INITIAL" else evidence_region
    return {
        "region_id": region,
        "basis": "REGION_GATED_INSPECTION_EVIDENCE",
        "source_stage": int(evidence_stage),
        "source_region": evidence_region,
        "measurement_cloud_path": record.get("measurement_cloud_path"),
    }


def _verified_assignments(
    witness: dict[str, Any], group_id: str
) -> list[dict[str, Any]]:
    assignments = []
    for assignment in witness.get("operation_assignments", []):
        if assignment.get("function_group_id") != group_id:
            continue
        checks = assignment.get("relation_checks", [])
        if (
            assignment.get("assignment_status") == "TRUE"
            and assignment.get("pair_geometry_status") == "TRUE"
            and checks
            and all(check.get("status") == "TRUE" for check in checks)
        ):
            assignments.append(assignment)
    return assignments


def compile_observed_symbolic_state(
    run_dir: str | Path,
    task_requirements: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Compile observed registry + complete witness into planner input."""
    run_path = Path(run_dir).resolve()
    registry = _read_json(run_path / "object_registry.json")
    graph = _read_json(run_path / "observed_graph.json")
    witness = _read_json(run_path / "latest_witness.json")
    task = load_task_requirements(task_requirements)
    symbolic = task.get("symbolic_task")
    if not isinstance(symbolic, dict):
        raise SymbolicCompilationError("Task has no symbolic_task specification")
    if witness.get("status") != "COMPLETE":
        raise SymbolicCompilationError(
            f"Functional witness is {witness.get('status')}, not COMPLETE"
        )

    objects = registry.get("objects", {})
    source_semantics_path = run_path / "symbolic_source_semantics.json"
    source_semantics = (
        _read_json(source_semantics_path).get("objects", {})
        if source_semantics_path.exists()
        else {}
    )
    initial_region = str(symbolic.get("initial_observation_region", "countertop"))
    observed_objects = {}
    for object_id, record in sorted(objects.items()):
        if object_id != record.get("object_id"):
            raise SymbolicCompilationError(f"Registry ID mismatch for {object_id}")
        label, confidence = _validated_label(record)
        observed_objects[object_id] = {
            "object_id": object_id,
            "semantic_label": label,
            "semantic_confidence": None if not math.isfinite(confidence) else confidence,
            "location": _observed_location(record, initial_region),
            "first_seen_stage": record.get("first_seen_stage"),
            "last_seen_stage": record.get("last_seen_stage"),
        }

    region_states = {}
    for node in graph.get("nodes", []):
        if node.get("type") != "region":
            continue
        attributes = node.get("attributes", {})
        region_id = attributes.get("region_id")
        if isinstance(region_id, str):
            region_states[region_id] = {
                "open": bool(attributes.get("open", False)),
                "inspected": bool(attributes.get("inspected", False)),
            }
    region_states.setdefault(initial_region, {"open": True, "inspected": True})
    region_states[initial_region]["open"] = True

    selected = witness.get("selected_witness") or {}
    coffee_targets = list(selected.get("coffee_container", []))
    soup_targets = list(selected.get("soup_container", []))
    if len(coffee_targets) != 3 or len(soup_targets) != 3:
        raise SymbolicCompilationError("Witness must bind three coffee and three soup targets")
    function_bound_objects = {
        object_id
        for object_ids in selected.values()
        for object_id in object_ids
    }

    source_bindings: dict[str, str] = {}
    source_capabilities: dict[str, str] = {}
    used_sources: set[str] = set()
    for role, requirement in symbolic.get("source_roles", {}).items():
        labels = {
            str(value).strip().lower()
            for value in requirement.get("accepted_semantic_labels", [])
        }
        candidates = []
        for record in observed_objects.values():
            source_record = source_semantics.get(record["object_id"])
            if source_record:
                matching_hypotheses = [
                    (label, source_record.get("label_hypotheses", {}).get(label))
                    for label in labels
                ]
                matching_hypotheses = [
                    (label, hypothesis)
                    for label, hypothesis in matching_hypotheses
                    if isinstance(hypothesis, dict)
                    and hypothesis.get("status") == "SUPPORTED"
                ]
                if matching_hypotheses:
                    label, hypothesis = max(
                        matching_hypotheses,
                        key=lambda item: (
                            item[1].get("supporting_view_count", 0),
                            item[1].get("mean_confidence", 0.0),
                            item[0],
                        ),
                    )
                    confidence = hypothesis.get("mean_confidence")
                    supported = True
                else:
                    label = None
                    confidence = None
                    supported = False
            else:
                label = record["semantic_label"]
                confidence = record["semantic_confidence"]
                supported = label is not None
            if (
                supported and label in labels
                and record["object_id"] not in used_sources
                and record["object_id"] not in function_bound_objects
            ):
                candidates.append({
                    **record,
                    "source_semantic_label": label,
                    "source_semantic_confidence": confidence,
                })
        candidates.sort(
            key=lambda item: (
                -(item["source_semantic_confidence"] or 0.0), item["object_id"]
            )
        )
        if not candidates:
            raise SymbolicCompilationError(
                f"No validated observed RGB semantic evidence for source role {role}"
            )
        chosen = candidates[0]["object_id"]
        used_sources.add(chosen)
        source_bindings[role] = chosen
        source_capabilities[chosen] = str(requirement["provides"])

    # Some targets can be observed already containing material (the primary
    # scene's soup bowls are visibly pre-filled). This is declarative task
    # state, not a hidden simulator lookup: target identity still comes from
    # the validated witness and the content type comes from the replaceable
    # ground-truth functional specification.
    initial_target_contents: set[tuple[str, str]] = set()
    declared_contents = set(map(str, symbolic.get("contents", [])))
    for requirement in symbolic.get("target_requirements", {}).values():
        witness_role = str(requirement.get("witness_role", ""))
        targets = list(selected.get(witness_role, []))
        required_contents = set(map(str, requirement.get("required_contents", [])))
        for content in map(str, requirement.get("initial_contents", [])):
            if content not in declared_contents or content not in required_contents:
                raise SymbolicCompilationError(
                    f"Invalid initial target content {content!r} for {witness_role}"
                )
            initial_target_contents.update((target, content) for target in targets)

    coffee_assignments = _verified_assignments(witness, "coffee_stirring")
    soup_assignments = _verified_assignments(witness, "soup_serving")
    can_stir = sorted({
        (item["utensil_object_id"], item["target_object_id"])
        for item in coffee_assignments
    })
    soup_tool_assignments = sorted({
        (item["utensil_object_id"], item["target_object_id"])
        for item in soup_assignments
    })
    if {target for _, target in can_stir} != set(coffee_targets):
        raise SymbolicCompilationError("Verified CAN_STIR facts do not cover all coffee targets")
    if {target for _, target in soup_tool_assignments} != set(soup_targets):
        raise SymbolicCompilationError("Verified soup assignments do not cover every target")
    soup_tools = [tool for tool, _ in soup_tool_assignments]
    if len(set(soup_tools)) != len(soup_tools):
        raise SymbolicCompilationError("Dedicated soup assignment reused a physical tool")

    relevant = set(coffee_targets) | set(soup_targets) | set(source_bindings.values())
    relevant |= {tool for tool, _ in can_stir} | set(soup_tools)
    missing = relevant - observed_objects.keys()
    if missing:
        raise SymbolicCompilationError(f"Witness references unobserved objects: {sorted(missing)}")

    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "inference_basis": "OBSERVED_REGISTRY_PLUS_VERIFIED_FUNCTIONAL_WITNESS",
        "location_basis": "REGION_GATED_INSPECTION_EVIDENCE_ONLY",
        "privileged_location_keys_rejected": ["source_region", "oracle_source_region"],
        "objects": {key: observed_objects[key] for key in sorted(relevant)},
        "regions": region_states,
        "role_assignments": {
            "coffee_targets": coffee_targets,
            "soup_targets": soup_targets,
            "source_roles": source_bindings,
            "coffee_stirring": coffee_assignments,
            "soup_serving": soup_assignments,
        },
        "capabilities": {
            "source_contains": [
                [source, content]
                for source, content in sorted(source_capabilities.items())
            ],
            "initial_target_contents": [
                list(pair) for pair in sorted(initial_target_contents)
            ],
            "can_stir": [list(pair) for pair in can_stir],
            "assigned_soup_utensil": [
                list(pair) for pair in soup_tool_assignments
            ],
        },
        "requirements": symbolic,
        "witness_stage": witness.get("stage"),
        "source_grounding_path": (
            "symbolic_source_semantics.json"
            if source_semantics_path.exists() else None
        ),
    }


@dataclass(frozen=True)
class PlannerState:
    locations: tuple[tuple[str, str], ...]
    opened: frozenset[str]
    held: str | None
    contents: frozenset[tuple[str, str]]
    stirred: frozenset[str]
    utensil_placed: frozenset[tuple[str, str]]
    served: frozenset[str]

    def location_map(self) -> dict[str, str]:
        return dict(self.locations)


@dataclass(frozen=True)
class GroundAction:
    name: str
    arguments: tuple[str, ...]

    def render(self) -> str:
        return f"{self.name.upper()}({', '.join(self.arguments)})"


class KitchenSymbolicProblem:
    """Small grounded STRIPS problem compiled without scene-specific IDs."""

    def __init__(self, compiled: dict[str, Any]):
        self.compiled = compiled
        assignments = compiled["role_assignments"]
        capabilities = compiled["capabilities"]
        self.home = compiled["requirements"].get("home_region", "countertop")
        self.coffee_targets = frozenset(assignments["coffee_targets"])
        self.soup_targets = frozenset(assignments["soup_targets"])
        self.source_contents = dict(map(tuple, capabilities["source_contains"]))
        self.can_stir = frozenset(map(tuple, capabilities["can_stir"]))
        self.soup_assignments = frozenset(
            map(tuple, capabilities["assigned_soup_utensil"])
        )
        self.manipulable = frozenset(self.source_contents) | {
            tool for tool, _ in self.can_stir | self.soup_assignments
        }
        locations = tuple(sorted(
            (object_id, record["location"]["region_id"])
            for object_id, record in compiled["objects"].items()
        ))
        opened = frozenset(
            region for region, state in compiled["regions"].items()
            if state["open"]
        ) | {self.home}
        self.initial = PlannerState(
            locations=locations,
            opened=opened,
            held=None,
            contents=frozenset(
                map(tuple, capabilities.get("initial_target_contents", []))
            ),
            stirred=frozenset(),
            utensil_placed=frozenset(),
            served=frozenset(),
        )

    def goals_satisfied(self, state: PlannerState) -> bool:
        return self.coffee_targets | self.soup_targets <= state.served

    def heuristic(self, state: PlannerState) -> int:
        missing_contents = sum(
            (target, content) not in state.contents
            for target in self.coffee_targets for content in ("coffee", "water")
        ) + sum((target, "soup") not in state.contents for target in self.soup_targets)
        missing_stir = len(self.coffee_targets - state.stirred)
        placed_targets = {target for _, target in state.utensil_placed}
        missing_utensils = len(self.soup_targets - placed_targets)
        missing_served = len((self.coffee_targets | self.soup_targets) - state.served)
        return missing_contents + missing_stir + missing_utensils + missing_served

    def applicable_actions(self, state: PlannerState) -> list[GroundAction]:
        actions: list[GroundAction] = []
        locations = state.location_map()
        if state.held is not None:
            held = state.held
            if held in self.source_contents:
                content = self.source_contents[held]
                targets = self.soup_targets if content == "soup" else self.coffee_targets
                for target in sorted(targets):
                    if (target, content) not in state.contents:
                        actions.append(GroundAction("pour", (held, target, content)))
            for tool, target in sorted(self.can_stir):
                if tool == held and target not in state.stirred and {
                    (target, "coffee"), (target, "water")
                } <= state.contents:
                    actions.append(GroundAction("stir", (tool, target)))
            for tool, target in sorted(self.soup_assignments):
                if (
                    tool == held
                    and (target, "soup") in state.contents
                    and not any(existing_target == target for _, existing_target in state.utensil_placed)
                    and all(
                        coffee_target in state.stirred
                        for stir_tool, coffee_target in self.can_stir
                        if stir_tool == tool
                    )
                ):
                    actions.append(
                        GroundAction("place_serving_utensil", (tool, target))
                    )
            actions.append(GroundAction("place", (held, self.home)))
            return actions

        for target in sorted(self.coffee_targets - state.served):
            if target in state.stirred:
                actions.append(GroundAction("serve_coffee", (target,)))
        placed_targets = {target for _, target in state.utensil_placed}
        for target in sorted(self.soup_targets - state.served):
            if (target, "soup") in state.contents and target in placed_targets:
                tool = next(
                    tool for tool, placed_target in sorted(state.utensil_placed)
                    if placed_target == target
                )
                actions.append(GroundAction("serve_soup", (target, tool)))

        needed_objects = set()
        for source, content in self.source_contents.items():
            targets = self.soup_targets if content == "soup" else self.coffee_targets
            if any((target, content) not in state.contents for target in targets):
                needed_objects.add(source)
        needed_objects.update(
            tool for tool, target in self.can_stir if target not in state.stirred
        )
        needed_objects.update(
            tool for tool, target in self.soup_assignments if target not in placed_targets
        )
        for object_id in sorted(needed_objects):
            region = locations.get(object_id)
            if region in state.opened:
                actions.append(GroundAction("pick", (object_id, region)))
            elif region is not None:
                actions.append(GroundAction("open", (region,)))
        return sorted(set(actions), key=lambda item: (item.name, item.arguments))

    def apply(self, state: PlannerState, action: GroundAction) -> PlannerState:
        locations = state.location_map()
        name, args = action.name, action.arguments
        if name == "open":
            region, = args
            if region in state.opened:
                raise ValueError("OPEN requires a closed region")
            return replace(state, opened=state.opened | {region})
        if name == "close":
            region, = args
            if region not in state.opened or region == self.home:
                raise ValueError("CLOSE requires an open non-home region")
            return replace(state, opened=state.opened - {region})
        if name == "pick":
            object_id, region = args
            if state.held is not None or locations.get(object_id) != region or region not in state.opened:
                raise ValueError("PICK precondition failed")
            del locations[object_id]
            return replace(state, held=object_id, locations=tuple(sorted(locations.items())))
        if name == "place":
            object_id, region = args
            if state.held != object_id or region not in state.opened:
                raise ValueError("PLACE precondition failed")
            locations[object_id] = region
            return replace(state, held=None, locations=tuple(sorted(locations.items())))
        if name == "pour":
            source, target, content = args
            if state.held != source or self.source_contents.get(source) != content:
                raise ValueError("POUR precondition failed")
            return replace(state, contents=state.contents | {(target, content)})
        if name == "stir":
            tool, target = args
            if (
                state.held != tool or (tool, target) not in self.can_stir
                or not {(target, "coffee"), (target, "water")} <= state.contents
            ):
                raise ValueError("STIR precondition failed")
            return replace(state, stirred=state.stirred | {target})
        if name == "place_serving_utensil":
            tool, target = args
            if state.held != tool or (tool, target) not in self.soup_assignments or (target, "soup") not in state.contents:
                raise ValueError("PLACE_SERVING_UTENSIL precondition failed")
            locations[tool] = target
            return replace(
                state,
                held=None,
                locations=tuple(sorted(locations.items())),
                utensil_placed=state.utensil_placed | {(tool, target)},
            )
        if name == "serve_coffee":
            target, = args
            if target not in state.stirred:
                raise ValueError("SERVE_COFFEE precondition failed")
            return replace(state, served=state.served | {target})
        if name == "serve_soup":
            target, tool = args
            if (target, "soup") not in state.contents or (
                tool, target
            ) not in state.utensil_placed:
                raise ValueError("SERVE_SOUP precondition failed")
            return replace(state, served=state.served | {target})
        raise ValueError(f"Unknown action {name}")


def plan_symbolic_task(problem: KitchenSymbolicProblem) -> list[GroundAction]:
    """Run deterministic best-first classical state-space search."""
    frontier: list[tuple[int, int, int, PlannerState]] = []
    serial = 0
    heappush(frontier, (problem.heuristic(problem.initial), 0, serial, problem.initial))
    parent: dict[PlannerState, tuple[PlannerState, GroundAction] | None] = {
        problem.initial: None
    }
    best_cost = {problem.initial: 0}
    while frontier:
        _score, cost, _serial, state = heappop(frontier)
        if cost != best_cost.get(state):
            continue
        if problem.goals_satisfied(state):
            plan = []
            while parent[state] is not None:
                previous, action = parent[state]
                plan.append(action)
                state = previous
            return list(reversed(plan))
        for action in problem.applicable_actions(state):
            successor = problem.apply(state, action)
            next_cost = cost + 1
            if next_cost >= best_cost.get(successor, 10**9):
                continue
            best_cost[successor] = next_cost
            parent[successor] = (state, action)
            serial += 1
            heappush(
                frontier,
                (next_cost + problem.heuristic(successor), next_cost, serial, successor),
            )
    raise SymbolicCompilationError("Classical planner found no valid plan")


def validate_symbolic_plan(
    problem: KitchenSymbolicProblem, plan: Iterable[GroundAction]
) -> dict[str, Any]:
    state = problem.initial
    steps = []
    for index, action in enumerate(plan, 1):
        before = state
        try:
            state = problem.apply(state, action)
        except ValueError as error:
            return {
                "valid": False,
                "all_goals_satisfied": False,
                "failed_step": index,
                "action": action.render(),
                "error": str(error),
            }
        locations = state.location_map()
        if len(locations) != len(set(locations)):
            raise AssertionError("An object has multiple symbolic locations")
        if state.held is not None and state.held in locations:
            raise AssertionError("Held object also has a location")
        steps.append({
            "step": index,
            "action": action.render(),
            "held_before": before.held,
            "held_after": state.held,
        })
    soup_tools = [tool for tool, _ in state.utensil_placed]
    all_goals = problem.goals_satisfied(state)
    return {
        "valid": all_goals and len(soup_tools) == len(set(soup_tools)),
        "all_goals_satisfied": all_goals,
        "goal_count": len(problem.coffee_targets | problem.soup_targets),
        "served_targets": sorted(state.served),
        "coffee_reuse_verified": len({tool for tool, _ in problem.can_stir}) == 1,
        "soup_distinctness_verified": len(soup_tools) == len(set(soup_tools)) == 3,
        "steps": steps,
    }


def render_domain_pddl() -> str:
    """Return the generic task-level domain consumed by replaceable backends."""
    return """(define (domain observed-kitchen-preparation)
  (:requirements :strips :typing)
  (:types physical_object region content - object
          vessel utensil source - physical_object)
  (:predicates
    (at ?o - physical_object ?r - region) (open ?r - region)
    (handempty) (holding ?o - physical_object)
    (source_contains ?s - source ?c - content)
    (can_stir ?u - utensil ?v - vessel)
    (assigned_soup_utensil ?u - utensil ?v - vessel)
    (has_content ?v - vessel ?c - content) (stirred ?v - vessel)
    (utensil_placed ?u - utensil ?v - vessel) (served ?v - vessel))
  (:action pick :parameters (?o - physical_object ?r - region)
    :precondition (and (handempty) (at ?o ?r) (open ?r))
    :effect (and (holding ?o) (not (handempty)) (not (at ?o ?r))))
  (:action place :parameters (?o - physical_object ?r - region)
    :precondition (and (holding ?o) (open ?r))
    :effect (and (at ?o ?r) (handempty) (not (holding ?o))))
  (:action open :parameters (?r - region)
    :precondition (and) :effect (open ?r))
  (:action close :parameters (?r - region)
    :precondition (open ?r) :effect (not (open ?r)))
  (:action pour :parameters (?s - source ?v - vessel ?c - content)
    :precondition (and (holding ?s) (source_contains ?s ?c))
    :effect (has_content ?v ?c))
  (:action stir :parameters (?u - utensil ?v - vessel)
    :precondition (and (holding ?u) (can_stir ?u ?v)
      (has_content ?v coffee) (has_content ?v water))
    :effect (stirred ?v))
  (:action place_serving_utensil :parameters (?u - utensil ?v - vessel)
    :precondition (and (holding ?u) (assigned_soup_utensil ?u ?v)
      (has_content ?v soup))
    :effect (and (utensil_placed ?u ?v) (handempty) (not (holding ?u))))
  (:action serve_coffee :parameters (?v - vessel)
    :precondition (stirred ?v) :effect (served ?v))
  (:action serve_soup :parameters (?v - vessel ?u - utensil)
    :precondition (and (has_content ?v soup) (utensil_placed ?u ?v))
    :effect (served ?v))
)\n"""


def render_problem_pddl(problem: KitchenSymbolicProblem, task_id: str) -> str:
    all_vessels = sorted(problem.coffee_targets | problem.soup_targets)
    utensils = sorted({tool for tool, _ in problem.can_stir | problem.soup_assignments})
    sources = sorted(problem.source_contents)
    regions = sorted(problem.compiled["regions"])
    lines = [
        f"(define (problem {_symbol(task_id)})",
        "  (:domain observed-kitchen-preparation)",
        "  (:objects",
        f"    {' '.join(map(_symbol, all_vessels))} - vessel",
        f"    {' '.join(map(_symbol, utensils))} - utensil",
        f"    {' '.join(map(_symbol, sources))} - source",
        f"    {' '.join(map(_symbol, regions))} - region",
        "    coffee water soup - content",
        "  )",
        "  (:init",
        "    (handempty)",
    ]
    for object_id, region in problem.initial.locations:
        if object_id in problem.manipulable:
            lines.append(f"    (at {_symbol(object_id)} {_symbol(region)})")
    for region in sorted(problem.initial.opened):
        lines.append(f"    (open {_symbol(region)})")
    for source, content in sorted(problem.source_contents.items()):
        lines.append(f"    (source_contains {_symbol(source)} {_symbol(content)})")
    for target, content in sorted(problem.initial.contents):
        lines.append(f"    (has_content {_symbol(target)} {_symbol(content)})")
    for tool, target in sorted(problem.can_stir):
        lines.append(f"    (can_stir {_symbol(tool)} {_symbol(target)})")
    for tool, target in sorted(problem.soup_assignments):
        lines.append(
            f"    (assigned_soup_utensil {_symbol(tool)} {_symbol(target)})"
        )
    lines.extend(["  )", "  (:goal (and"])
    for target in all_vessels:
        lines.append(f"    (served {_symbol(target)})")
    lines.extend(["  ))", ")"])
    return "\n".join(lines) + "\n"


def _search_trace(run_path: Path) -> list[dict[str, Any]]:
    trace = []
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        return trace
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "REGION_OPENED":
            trace.append({
                "action": "INSPECT",
                "region_id": event.get("region_id"),
                "stage": event.get("stage"),
            })
    return trace


def _scientific_validation(
    run_path: Path,
    compiled: dict[str, Any],
    problem_pddl: str,
    plan_records: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Audit the perception/planning boundary using only saved run evidence."""
    registry = _read_json(run_path / "object_registry.json").get("objects", {})
    early_visibility_violations = []
    stage_snapshots = []
    for graph_path in sorted((run_path / "stages").glob("*/graph.json")):
        match = re.match(r"(\d+)", graph_path.parent.name)
        if match is None:
            continue
        stage = int(match.group(1))
        graph = _read_json(graph_path)
        visible_ids = {
            str(node.get("id", "")).removeprefix("object:")
            for node in graph.get("nodes", [])
            if node.get("type") == "object"
        }
        stage_snapshots.append({
            "stage": stage,
            "path": str(graph_path.relative_to(run_path)),
            "observed_object_count": len(visible_ids),
        })
        for object_id, record in registry.items():
            first_seen = record.get("first_seen_stage")
            if (
                isinstance(first_seen, int)
                and stage < first_seen
                and object_id in visible_ids
            ):
                early_visibility_violations.append({
                    "object_id": object_id,
                    "first_seen_stage": first_seen,
                    "leaked_at_stage": stage,
                })

    location_violations = [
        object_id
        for object_id, record in compiled.get("objects", {}).items()
        if record.get("location", {}).get("basis")
        != "REGION_GATED_INSPECTION_EVIDENCE"
    ]
    pddl_lower = problem_pddl.lower()
    forbidden_tokens = [
        token for token in (
            "oracle_source_region",
            "privileged_wrong_region",
            "s1i_",
            "ab3_",
            "pot_with_soup",
            "coffee_jar",
        ) if token in pddl_lower
    ]
    physical_ids = sorted(compiled.get("objects", {}))
    non_generic_ids = [
        object_id
        for object_id in physical_ids
        if re.fullmatch(r"object_\d+", object_id) is None
    ]
    plan_steps_match = all(
        record.get("step") == index and record.get("rendered")
        for index, record in enumerate(plan_records, 1)
    )
    return {
        "schema_version": 1,
        "check_1_observed_symbolic_state": {
            "passed": not early_visibility_violations and not location_violations,
            "hidden_object_exclusion_checked": bool(stage_snapshots),
            "early_visibility_violations": early_visibility_violations,
            "location_basis": compiled.get("location_basis"),
            "location_basis_violations": location_violations,
            "stage_snapshots": stage_snapshots,
        },
        "check_2_pddl_problem_cleanliness": {
            "passed": not forbidden_tokens and not non_generic_ids,
            "forbidden_privileged_tokens_found": forbidden_tokens,
            "non_generic_physical_ids": non_generic_ids,
            "problem_state_stage": compiled.get("witness_stage"),
            "problem_state_basis": compiled.get("inference_basis"),
        },
        "check_3_planner_origin_and_validation": {
            "passed": bool(
                plan_records
                and plan_steps_match
                and validation.get("valid")
                and validation.get("all_goals_satisfied")
            ),
            "planner_entry_point": (
                "mujoco_scenes.symbolic_planning.plan_symbolic_task"
            ),
            "algorithm": "deterministic_best_first_classical_state_space_search",
            "renderer_role": "serialization_only_after_planner_returns",
            "action_count": len(plan_records),
            "plan_steps_well_formed": plan_steps_match,
            "validator_valid": validation.get("valid"),
            "all_goals_satisfied": validation.get("all_goals_satisfied"),
        },
    }


def compile_plan_and_save(
    run_dir: str | Path,
    task_requirements: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Compile, plan, validate, and atomically save all boundary artifacts."""
    run_path = Path(run_dir).resolve()
    compiled = compile_observed_symbolic_state(run_path, task_requirements)
    problem = KitchenSymbolicProblem(compiled)
    plan = plan_symbolic_task(problem)
    validation = validate_symbolic_plan(problem, plan)
    if not validation["valid"]:
        raise SymbolicCompilationError(f"Generated plan failed validation: {validation}")
    task_id = compiled["task_id"]
    search_trace = _search_trace(run_path)
    assignments = compiled["role_assignments"]
    plan_records = [
        {"step": index, "action": action.name, "arguments": list(action.arguments), "rendered": action.render()}
        for index, action in enumerate(plan, 1)
    ]
    _write_json(run_path / "search_trace.json", search_trace)
    _write_json(run_path / "observed_symbolic_state.json", compiled)
    _write_json(run_path / "grounded_role_assignments.json", assignments)
    _write_json(run_path / "symbolic_initial_state.json", asdict(problem.initial))
    _write_json(run_path / "symbolic_goal.json", {
        "served": sorted(problem.coffee_targets | problem.soup_targets)
    })
    _write_json(run_path / "plan.json", plan_records)
    _write_json(run_path / "plan_validation.json", validation)
    domain_pddl = render_domain_pddl()
    problem_pddl = render_problem_pddl(problem, task_id)
    (run_path / "domain.pddl").write_text(domain_pddl, encoding="utf-8")
    (run_path / "problem.pddl").write_text(problem_pddl, encoding="utf-8")
    planner_provenance = {
        "schema_version": 1,
        "planner_entry_point": (
            "mujoco_scenes.symbolic_planning.plan_symbolic_task"
        ),
        "algorithm": "deterministic_best_first_classical_state_space_search",
        "domain_source": "generated_from_generic_action_schemas",
        "problem_source": "compiled_observed_state_and_verified_witness",
        "plan_renderer_role": "serialization_only_after_planner_returns",
        "action_count": len(plan_records),
    }
    _write_json(run_path / "planner_provenance.json", planner_provenance)
    _write_json(
        run_path / "scientific_validation.json",
        _scientific_validation(
            run_path,
            compiled,
            problem_pddl,
            plan_records,
            validation,
        ),
    )
    plan_text = "\n".join(
        f"{record['step']:03d} {record['rendered']}" for record in plan_records
    ) + "\n"
    (run_path / "plan.txt").write_text(plan_text, encoding="utf-8")
    witness_lines = [
        f"coffee targets: {', '.join(assignments['coffee_targets'])}",
        "coffee reusable tool: " + ", ".join(sorted({
            item['utensil_object_id'] for item in assignments['coffee_stirring']
        })),
        "soup dedicated assignments: " + "; ".join(
            f"{item['target_object_id']} -> {item['utensil_object_id']}"
            for item in assignments['soup_serving']
        ),
    ]
    combined = ["=== SEARCH / INSPECTION TRACE ===", ""]
    combined.extend(
        f"INSPECT({item['region_id']})" for item in search_trace
    )
    combined.extend([
        "", "=== GLOBAL FUNCTIONAL WITNESS ===", "", *witness_lines,
        "", "=== GENERATED SYMBOLIC TASK PLAN ===", "", plan_text.rstrip(),
        "", "=== VALIDATION ===", "", "PLAN VALID", "ALL GOALS SATISFIED", "",
    ])
    (run_path / "combined_action_sequence.txt").write_text(
        "\n".join(combined), encoding="utf-8"
    )
    return {
        "compiled": compiled,
        "plan": plan_records,
        "validation": validation,
        "search_trace": search_trace,
        "run_dir": str(run_path),
    }
