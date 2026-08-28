"""Planning-only Kitchen state rollout and private GT sequence evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from baseline_common.models import Action, ActionResult, Entity, Observation, Region
from mujoco_scenes.baseline_kitchen_runtime import BaselineKitchenRuntime

from .living_room_runtime import compare_action_sequences


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_ROOT = REPOSITORY_ROOT / "EXPECTED_GT_ACTIONS" / "kitchen"


class KitchenPlanningState:
    """Apply symbolic effects without advancing MuJoCo or moving the robot."""

    def __init__(self, runtime: BaselineKitchenRuntime):
        self.runtime = runtime
        self.held_object: str | None = None
        self.inspected_regions: set[str] = set()
        self.labels = {
            str(row["generic_object_id"]): str(row.get("semantic_label", ""))
            for row in runtime.bundle.resolution.get("accepted", ())
        }

    def observe(self):
        observation, images = self.runtime.observe()
        return self._with_robot_state(observation), images

    def observe_state(self) -> Observation:
        return self._with_robot_state(self.runtime.observe_state())

    def _with_robot_state(self, observation: Observation) -> Observation:
        robot = dict(observation.robot)
        robot["holding"] = self.held_object
        entities = {item.entity_id: item for item in observation.entities}
        for row in self.runtime.bundle.inventory.get("objects", ()):
            object_id = str(row["generic_object_id"])
            context = row.get("source_context", {})
            source = str(context.get("observed_source_region") or "countertop")
            if source not in self.inspected_regions or object_id in entities:
                continue
            entities[object_id] = Entity(
                object_id,
                "object",
                object_id,
                {
                    "dimensions_m": dict(row.get("observed_dimensions_m", {})),
                    "source_region": source,
                },
            )
        regions = tuple(
            Region(
                item.region_id,
                item.label,
                "open" if item.region_id in self.inspected_regions else item.state,
                item.inspected or item.region_id in self.inspected_regions,
            )
            for item in observation.regions
        )
        augmented = self.runtime.ledger.augment(
            Observation(
                observation.scene,
                observation.revision,
                tuple(entities[key] for key in sorted(entities)),
                regions,
                robot,
                False,
            )
        )
        return Observation(
            augmented.scene,
            augmented.revision,
            augmented.entities,
            augmented.regions,
            augmented.robot,
            self.goal_verifier(augmented),
        )

    def goal_verifier(self, _observation: Observation | None = None) -> bool:
        parsed = [
            self.runtime.ledger._parse(effect)
            for effect in self.runtime.ledger.effects
        ]
        effects = [row for row in parsed if row is not None]

        def has(name: str, *arguments: str) -> bool:
            return (name, tuple(arguments)) in effects

        coffee_targets = [
            object_id
            for object_id, label in self.labels.items()
            if label in {"cup", "mug"}
        ]
        soup_targets = [
            object_id for object_id, label in self.labels.items() if label == "bowl"
        ]
        water_sources = [
            object_id for object_id, label in self.labels.items() if label == "kettle"
        ]
        coffee_sources = [
            object_id
            for object_id, label in self.labels.items()
            if label == "coffee_source"
        ]
        tools = {
            object_id for object_id, label in self.labels.items() if label == "spoon"
        }
        if not (
            len(coffee_targets) >= 2
            and len(soup_targets) >= 2
            and water_sources
            and coffee_sources
        ):
            return False
        prepared_coffee = [
            target
            for target in coffee_targets
            if has("placed", target, "serving_area")
            and any(has("poured", source, target) for source in water_sources)
            and any(has("poured", source, target) for source in coffee_sources)
            and any(has("stirred", tool, target) for tool in tools)
        ]
        served_soup: dict[str, set[str]] = {
            target: {
                tool
                for tool in tools
                if has("placed", tool, target)
            }
            for target in soup_targets
            if has("placed", target, "serving_area")
        }
        if len(prepared_coffee) < 2 or len(served_soup) < 2:
            return False
        first_two = list(served_soup.values())[:2]
        return bool(first_two[0] and first_two[1]) and any(
            left != right for left in first_two[0] for right in first_two[1]
        )

    def execute(self, action: Action) -> ActionResult:
        skill = action.skill.upper()
        arguments = action.arguments
        visible = self.observe_state().object_ids
        if skill == "INSPECT":
            region_id = str(arguments.get("region_id", ""))
            if region_id not in self.runtime.scene.get_region_observation_states():
                return ActionResult.failed("unknown_region", f"Unknown region {region_id}")
            if self.held_object is not None:
                return ActionResult.failed(
                    "gripper_occupied",
                    f"Cannot inspect while holding {self.held_object}",
                )
            self.inspected_regions.add(region_id)
            return ActionResult.succeeded(f"inspected({region_id})")
        if skill == "PICK":
            object_id = str(arguments.get("object_id", ""))
            if object_id not in visible:
                return ActionResult.failed(
                    "unknown_object", f"Object {object_id} is not currently visible"
                )
            if self.held_object is not None:
                return ActionResult.failed(
                    "gripper_occupied", f"Already holding {self.held_object}"
                )
            self.held_object = object_id
            effects = (f"holding({object_id})",)
        elif skill == "PLACE":
            object_id = str(arguments.get("object_id", ""))
            region_id = str(arguments.get("region_id", ""))
            destinations = visible | self.observe_state().region_ids
            if self.held_object != object_id:
                return ActionResult.failed(
                    "not_holding_object", f"The robot is not holding {object_id}"
                )
            if region_id not in destinations:
                return ActionResult.failed(
                    "unknown_destination", f"Unknown destination {region_id}"
                )
            self.held_object = None
            effects = (f"placed({object_id},{region_id})",)
        elif skill in {"POUR", "STIR"}:
            first_key = "source_id" if skill == "POUR" else "tool_id"
            first = str(arguments.get(first_key, ""))
            target = str(arguments.get("target_id", ""))
            if self.held_object != first:
                return ActionResult.failed(
                    "not_holding_object", f"The robot is not holding {first}"
                )
            if target not in visible:
                return ActionResult.failed(
                    "unknown_target", f"Target {target} is not currently visible"
                )
            predicate = "poured" if skill == "POUR" else "stirred"
            effects = (f"{predicate}({first},{target})",)
        else:
            return ActionResult.failed(
                "unsupported_planning_action",
                f"Kitchen planning-only mode cannot apply {action.skill}",
                recoverable=False,
            )
        self.runtime.accept_effects(effects)
        return ActionResult.succeeded(*effects)


def load_expected(root: Path, variant: str) -> dict[str, Any]:
    path = root / variant / "expected_gt_actions.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected GT file must be an object: {path}")
    return value


def canonical_kitchen_actions(
    history: Sequence[Mapping[str, Any]],
    backend_by_id: Mapping[str, str],
) -> list[dict[str, Any]]:
    argument_order = {
        "INSPECT": ("region_id",),
        "PICK": ("object_id",),
        "PLACE": ("object_id", "region_id"),
        "POUR": ("source_id", "target_id"),
        "STIR": ("tool_id", "target_id"),
    }
    result = []
    for row in history:
        if not row.get("success"):
            continue
        action = row.get("action", {})
        skill = str(action.get("skill", "")).upper()
        if skill not in argument_order:
            continue
        arguments = action.get("arguments", {})
        values = [
            backend_by_id.get(str(arguments[key]), str(arguments[key]))
            for key in argument_order[skill]
        ]
        result.append({"operator": skill, "arguments": values})
    return result


def normalize_kitchen_actions(
    actions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map execution-only GT operators to the shared task-level vocabulary."""
    result = []
    for row in actions:
        operator = str(row["operator"]).upper()
        arguments = list(map(str, row.get("arguments", ())))
        if operator == "CLOSE":
            continue
        if operator == "OPEN":
            operator = "INSPECT"
        elif operator == "PLACE_SERVING_UTENSIL":
            operator = "PLACE"
        result.append({"operator": operator, "arguments": arguments})
    return result


def compare_kitchen_actions(
    predicted: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw = compare_action_sequences(predicted, expected)
    normalized_predicted = normalize_kitchen_actions(predicted)
    normalized_expected = normalize_kitchen_actions(expected)
    task_level = compare_action_sequences(normalized_predicted, normalized_expected)
    return {
        "raw_execution_vocabulary": raw,
        "shared_task_vocabulary": task_level,
        "normalization": {
            "OPEN": "INSPECT",
            "CLOSE": "excluded_execution_cleanup",
            "PLACE_SERVING_UTENSIL": "PLACE",
        },
    }


__all__ = [
    "DEFAULT_EXPECTED_ROOT",
    "KitchenPlanningState",
    "canonical_kitchen_actions",
    "compare_kitchen_actions",
    "load_expected",
    "normalize_kitchen_actions",
]
