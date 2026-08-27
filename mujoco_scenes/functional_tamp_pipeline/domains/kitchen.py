"""Kitchen adapter around existing observed-state grounding and planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mujoco_scenes.scene_loader import KitchenScene
from mujoco_scenes.sequential_inspection import run_sequential_inspection
from mujoco_scenes.symbolic_planning import (
    KitchenSymbolicProblem, compile_observed_symbolic_state,
)
from mujoco_scenes.symbolic_planning_core import SymbolicAction, SymbolicProblem

from ..models import FunctionalSpecification, PipelineResult
from ..planning import plan_with_common_astar


TASK = (
    "Prepare and serve coffee and soup for two people using the available "
    "kitchenware. Stir both coffees and provide each soup bowl with a suitable "
    "utensil. Search closed kitchen storage for anything still required."
)
LOCAL_YOLO_WORLD = (
    Path(__file__).resolve().parents[3]
    / "semantic_model_cache" / "yolov8m-worldv2.pt"
)


def _action(
    name: str,
    arguments: tuple[str, ...],
    positive: set[tuple[str, ...]],
    add: set[tuple[str, ...]],
    delete: set[tuple[str, ...]],
) -> SymbolicAction:
    return SymbolicAction(
        name=name, arguments=arguments,
        positive_preconditions=frozenset(positive),
        negative_preconditions=frozenset(),
        add_effects=frozenset(add), delete_effects=frozenset(delete),
    )


class KitchenPlanningCompiler:
    """Translate the existing observed Kitchen witness into common STRIPS."""

    def compile_problem(
        self, assignment: dict[str, Any], context: dict[str, Any]
    ) -> SymbolicProblem:
        del assignment
        legacy = KitchenSymbolicProblem(context["compiled_observed_state"])
        initial: set[tuple[str, ...]] = {("hand_empty",)}
        initial.update(("at", obj, region) for obj, region in legacy.initial.locations)
        initial.update(("contains", target, content) for target, content in legacy.initial.contents)
        initial.update(("stirred", target) for target in legacy.initial.stirred)

        actions: list[SymbolicAction] = []
        for obj in sorted(legacy.manipulable):
            destinations = set(legacy._allowed_destinations(obj))
            locations = destinations | {
                region for candidate, region in legacy.initial.locations if candidate == obj
            }
            for region in sorted(locations):
                actions.append(_action(
                    "PICK", (obj,),
                    {("hand_empty",), ("at", obj, region)},
                    {("holding", obj)},
                    {("hand_empty",), ("at", obj, region)},
                ))
            for destination in sorted(destinations):
                preconditions = {("holding", obj)}
                if (obj, destination) in legacy.soup_assignments:
                    preconditions.add(("contains", destination, "soup"))
                actions.append(_action(
                    "PLACE", (obj, destination), preconditions,
                    {("hand_empty",), ("at", obj, destination)},
                    {("holding", obj)},
                ))

        for source, content in sorted(legacy.source_contents.items()):
            targets = legacy.soup_targets if content == "soup" else legacy.coffee_targets
            for target in sorted(targets):
                actions.append(_action(
                    "POUR", (source, target), {("holding", source)},
                    {("contains", target, content)}, set(),
                ))
        for tool, target in sorted(legacy.can_stir):
            actions.append(_action(
                "STIR", (tool, target),
                {
                    ("holding", tool), ("contains", target, "coffee"),
                    ("contains", target, "water"),
                },
                {("stirred", target)}, set(),
            ))
        actions.sort(key=lambda item: (
            item.name, item.arguments, tuple(sorted(item.positive_preconditions)),
        ))
        return SymbolicProblem(
            initial_atoms=frozenset(initial),
            goal_atoms=frozenset(legacy.goal_facts()),
            actions=tuple(actions),
        )


def scene_for_variant(internal_variant: str) -> KitchenScene:
    code = internal_variant.split("_", 1)[0]
    return KitchenScene(
        f"S1_integrated_kitchen_object_function_feasibility_{code}",
        include_robot=False,
        robot="none",
    )


def run_to_plan(
    *,
    variant_label: str,
    internal_variant: str,
    mode: str,
    specification: FunctionalSpecification,
    output_dir: Path,
    scene: KitchenScene | None = None,
) -> PipelineResult:
    scene = scene or scene_for_variant(internal_variant)
    contract = specification.raw_requirements[0]
    vocabulary_path: Path
    if mode == "vlm":
        vocabulary_path = output_dir / "object_vocabulary.yaml"
        vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        vocabulary_path.write_text(
            yaml.safe_dump(specification.metadata["object_vocabulary"], sort_keys=False),
            encoding="utf-8",
        )
    else:
        vocabulary_path = Path(specification.metadata["semantic_vocabulary_path"])

    session = run_sequential_inspection(
        scene,
        specification.region_ranking,
        runs_root=output_dir / "observed_search",
        run_id="phase1",
        width=1280,
        height=960,
        task_requirements=contract,
        stop_on_complete=True,
        semantic_backend="yolo_world",
        semantic_model=str(LOCAL_YOLO_WORLD),
        semantic_vocabulary_path=vocabulary_path,
        semantic_min_supporting_views=2,
        grounding_mode="joint",
        completion_predicate=lambda current: (
            (current.latest_witness or {}).get("status") == "COMPLETE"
        ),
        record_oracle_diagnostics=False,
    )
    witness = session.latest_witness or {}
    events = [
        json.loads(line) for line in session.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    opened = tuple(
        event["region_id"] for event in events if event.get("event") == "REGION_OPENED"
    )
    if witness.get("status") != "COMPLETE":
        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status="INFEASIBLE", inspected_regions=opened,
            failure_reason=str(witness.get("reason", "NO_COMPLETE_FUNCTIONAL_WITNESS")),
        )
    compiled = compile_observed_symbolic_state(session.run_dir, contract)
    assignments = compiled["role_assignments"]
    planned = plan_with_common_astar(
        KitchenPlanningCompiler(), assignments,
        {"compiled_observed_state": compiled},
    )
    plan_dir = output_dir / "action_sequence"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "action_plan.json").write_text(
        json.dumps({
            "planner": planned.search.statistics,
            "actions": list(planned.actions),
            "validation": planned.validation,
            "exploratory_open_actions_excluded": True,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PipelineResult(
        domain="kitchen", variant=variant_label, mode=mode,
        status="ACTION_SEQUENCE_READY", inspected_regions=opened,
        assignment=assignments, plan=planned.actions,
        search_statistics=planned.search.statistics,
    )
