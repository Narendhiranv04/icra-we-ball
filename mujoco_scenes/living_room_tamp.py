"""Autonomous functional storage task for the living room."""

from __future__ import annotations

import os
from pathlib import Path

from mujoco_scenes.foundation_model import (
    AssessmentBackend,
    Candidate,
    FixedAssessmentBackend,
    OpenAICompatibleRanker,
)
from mujoco_scenes.living_room_manipulation import (
    LivingRoomManipulationExecutor,
)
from mujoco_scenes.tamp.events import EventLog
from mujoco_scenes.tamp.executive import (
    FunctionalTask,
    PlanRejected,
    TaskExecutive,
)
from mujoco_scenes.tamp.functions import FunctionRegistry
from mujoco_scenes.tamp.skills import (
    FailureCode,
    SkillAction,
    SkillResult,
    SkillStartError,
)
from mujoco_scenes.tamp.state import (
    ObjectObservation,
    ObservedState,
    RegionObservation,
    Relation,
    RobotObservation,
)


STORE_CONTROLLER_TASK = FunctionalTask(
    "store_game_controller",
    "game_controller",
    "can_store",
    "stored_in",
)

STORAGE_TARGETS = {
    "media_console_right_drawer": {
        "category": "drawer",
        "navigation_location": "drawer_right",
        "place_site": "drawer_place_controller",
        "drawer_side": "right",
    },
    "media_console_left_drawer": {
        "category": "drawer",
        "navigation_location": "drawer_left",
        "place_site": "left_drawer_place_controller",
        "drawer_side": "left",
    },
    "media_shelf": {
        "category": "shelf",
        "navigation_location": "bookshelf",
        "place_site": "media_shelf_controller_place",
    },
}

OBJECT_CATEGORIES = {
    "remote_control": "remote_control",
    "living_room_mug": "mug",
    "hardback_book": "book",
    "game_controller": "game_controller",
    "rigid_duster": "rigid_duster",
}


def semantic_storage_location(location: str) -> str:
    return {
        "drawer": "media_console_right_drawer",
        "drawer_left": "media_console_left_drawer",
        "drawer_right": "media_console_right_drawer",
        "bookshelf": "media_shelf",
    }.get(location, location)


class LivingRoomObserver:
    def __init__(
        self,
        navigation,
        manipulation: LivingRoomManipulationExecutor,
        left_drawer,
        right_drawer,
    ):
        self.navigation = navigation
        self.manipulation = manipulation
        self.drawers = {
            "left": left_drawer,
            "right": right_drawer,
        }
        self.inspected_regions = {"media_shelf"}
        self.revision = 0

    def _region_occupants(self, region_id: str) -> tuple[str, ...]:
        return tuple(
            object_id
            for object_id, location in self.manipulation.object_locations.items()
            if semantic_storage_location(location) == region_id
        )

    def __call__(self) -> ObservedState:
        self.revision += 1
        for side, drawer in self.drawers.items():
            if drawer.is_open:
                self.inspected_regions.add(
                    f"media_console_{side}_drawer"
                )

        regions = {}
        for region_id, values in STORAGE_TARGETS.items():
            side = values.get("drawer_side")
            is_open = self.drawers[side].is_open if side else True
            inspected = region_id in self.inspected_regions
            regions[region_id] = RegionObservation(
                region_id,
                str(values["category"]),
                True,
                inspected=inspected,
                open=is_open,
                occupied_by=(
                    self._region_occupants(region_id)
                    if inspected
                    else None
                ),
                facts={
                    "_navigation_location": values[
                        "navigation_location"
                    ],
                    "_place_site": values["place_site"],
                    "_drawer_side": side,
                    "rigid": True,
                },
            )

        held_object = self.manipulation.held_object
        objects = {}
        for object_id, category in OBJECT_CATEGORIES.items():
            location = self.manipulation.object_locations[object_id]
            semantic_location = semantic_storage_location(location)
            stored_region = regions.get(semantic_location)
            visible = bool(
                object_id == held_object
                or stored_region is None
                or (
                    stored_region.inspected
                    and stored_region.open is not False
                )
            )
            objects[object_id] = ObjectObservation(
                object_id,
                category,
                visible,
                semantic_location,
                {"rigid": True},
            )

        relations = tuple(
            Relation(object_id, "observed_in", region_id)
            for region_id, region in regions.items()
            if region.occupied_by is not None
            for object_id in region.occupied_by
        )
        motion_ready = bool(
            self.manipulation.navigation_safe
            and all(
                drawer.navigation_safe
                for drawer in self.drawers.values()
            )
        )
        return ObservedState(
            objects,
            regions,
            RobotObservation(
                self.navigation.current_location,
                held_object,
                motion_ready,
            ),
            relations,
            self.revision,
        )


class LivingRoomStoragePlanner:
    def __call__(
        self,
        task: FunctionalTask,
        candidate: Candidate,
        state: ObservedState,
    ) -> tuple[SkillAction, ...]:
        if task.required_function != "can_store":
            raise PlanRejected(
                FailureCode.FUNCTION_UNSATISFIED,
                f"Unsupported function: {task.required_function}",
            )
        region = state.regions[candidate.candidate_id]
        occupied_by = region.occupied_by
        if occupied_by and any(
            object_id != task.subject_id for object_id in occupied_by
        ):
            raise PlanRejected(
                FailureCode.TARGET_OCCUPIED,
                f"{candidate.candidate_id} is occupied",
            )

        actions = []
        current_location = state.robot.location
        held_object = state.robot.held_object
        if held_object not in {None, task.subject_id}:
            raise PlanRejected(
                FailureCode.PRECONDITION_FAILED,
                f"Robot is already holding {held_object}",
            )

        if held_object is None:
            subject = state.visible_object(task.subject_id)
            if subject is None:
                raise PlanRejected(
                    FailureCode.OBJECT_NOT_VISIBLE,
                    f"{task.subject_id} is not visible",
                )
            source_region = state.regions.get(subject.location or "")
            source_location = (
                str(source_region.facts["_navigation_location"])
                if source_region is not None
                else str(subject.location)
            )
            if current_location != source_location:
                actions.append(
                    SkillAction(
                        "move", {"destination": source_location}
                    )
                )
                current_location = source_location
            if (
                source_region is not None
                and source_region.category == "drawer"
                and not source_region.open
            ):
                actions.append(
                    SkillAction(
                        "open",
                        {
                            "side": source_region.facts["_drawer_side"],
                        },
                    )
                )
            actions.append(
                SkillAction("pick", {"object_id": task.subject_id})
            )

        target_location = str(region.facts["_navigation_location"])
        for other_id, other_region in state.regions.items():
            if (
                other_id != candidate.candidate_id
                and other_region.category == "drawer"
                and other_region.open
                and current_location
                == other_region.facts["_navigation_location"]
            ):
                actions.append(
                    SkillAction(
                        "close",
                        {"side": other_region.facts["_drawer_side"]},
                    )
                )
        if current_location != target_location:
            actions.append(
                SkillAction("move", {"destination": target_location})
            )
        if region.category == "drawer" and not region.open:
            actions.append(
                SkillAction(
                    "open", {"side": region.facts["_drawer_side"]}
                )
            )
        actions.append(
            SkillAction(
                "place",
                {
                    "destination": candidate.candidate_id,
                    "place_site": region.facts["_place_site"],
                },
            )
        )
        if region.category == "drawer":
            actions.append(
                SkillAction(
                    "close", {"side": region.facts["_drawer_side"]}
                )
            )
        return tuple(actions)


class LivingRoomSkillDispatcher:
    def __init__(
        self,
        navigation,
        manipulation: LivingRoomManipulationExecutor,
        left_drawer,
        right_drawer,
    ):
        self.navigation = navigation
        self.manipulation = manipulation
        self.drawers = {
            "left": left_drawer,
            "right": right_drawer,
        }
        self.action: SkillAction | None = None
        self.controller = None

    def _target_occupants(self, destination: str) -> tuple[str, ...]:
        return tuple(
            object_id
            for object_id, location in self.manipulation.object_locations.items()
            if semantic_storage_location(location) == destination
        )

    def start(self, action: SkillAction) -> None:
        if self.action is not None:
            raise SkillStartError(
                FailureCode.PRECONDITION_FAILED,
                "Another skill is active",
                recoverable=False,
            )
        arguments = action.arguments
        try:
            if action.name == "move":
                self.controller = self.navigation
                self.navigation.request_move(
                    str(arguments["destination"])
                )
            elif action.name in {"open", "close"}:
                side = str(arguments["side"])
                self.controller = self.drawers[side]
                self.controller.request(
                    action.name, self.navigation.current_location
                )
            elif action.name == "pick":
                self.controller = self.manipulation
                self.manipulation.request_pick(
                    str(arguments["object_id"]),
                    self.navigation.current_location,
                )
            elif action.name == "place":
                destination = str(arguments["destination"])
                occupants = tuple(
                    object_id
                    for object_id in self._target_occupants(destination)
                    if object_id != self.manipulation.held_object
                )
                if occupants:
                    raise SkillStartError(
                        FailureCode.TARGET_OCCUPIED,
                        f"{destination} contains {', '.join(occupants)}",
                    )
                self.controller = self.manipulation
                self.manipulation.request_place_at(
                    self.navigation.current_location,
                    destination,
                    str(arguments["place_site"]),
                )
            else:
                raise SkillStartError(
                    FailureCode.PRECONDITION_FAILED,
                    f"Unknown living-room skill: {action.name}",
                    recoverable=False,
                )
        except SkillStartError:
            raise
        except (KeyError, RuntimeError, ValueError) as error:
            raise SkillStartError(
                FailureCode.PRECONDITION_FAILED, str(error)
            ) from error
        self.action = action

    @staticmethod
    def _failure_code(message: str) -> FailureCode:
        lowered = message.lower()
        if "collision" in lowered:
            return FailureCode.COLLISION
        if "ik" in lowered:
            return FailureCode.IK_FAILED
        if "contact" in lowered or "gripper" in lowered:
            return FailureCode.GRASP_FAILED
        if "place" in lowered:
            return FailureCode.PLACEMENT_FAILED
        if "rrt" in lowered or "path" in lowered:
            return FailureCode.PATH_BLOCKED
        return FailureCode.INTERNAL_ERROR

    def update(self) -> SkillResult | None:
        if self.action is None or self.controller is None:
            return None
        self.controller.update()
        failure = getattr(self.controller, "failure", None)
        if failure:
            code = self._failure_code(str(failure))
            self.action = None
            self.controller = None
            return SkillResult.failed(
                code,
                str(failure),
                recoverable=False,
            )
        if bool(getattr(self.controller, "busy", False)):
            return None

        action = self.action
        if (
            action.name == "pick"
            and self.manipulation.held_object
            != action.arguments["object_id"]
        ):
            return None
        if (
            action.name == "place"
            and self.manipulation.held_object is not None
        ):
            return None
        self.action = None
        self.controller = None
        return SkillResult.succeeded()


def storage_goal_satisfied(
    task: FunctionalTask,
    candidate: Candidate,
    state: ObservedState,
) -> bool:
    subject = state.objects[task.subject_id]
    region = state.regions[candidate.candidate_id]
    return bool(
        subject.location == candidate.candidate_id
        and task.subject_id in (region.occupied_by or ())
        and (region.category != "drawer" or region.open is False)
    )


def _assessment_backend_from_env() -> AssessmentBackend:
    if os.environ.get("TAMP_FM_BACKEND", "remote") == "fixed":
        target_ids = tuple(STORAGE_TARGETS)
        return FixedAssessmentBackend(target_ids, target_ids)
    return OpenAICompatibleRanker.from_env()


class LivingRoomTampController:
    def __init__(
        self,
        navigation,
        manipulation: LivingRoomManipulationExecutor,
        left_drawer,
        right_drawer,
        assessment_backend: AssessmentBackend | None = None,
    ):
        self.navigation = navigation
        self.manipulation = manipulation
        self.drawers = (left_drawer, right_drawer)
        self.backend = assessment_backend or _assessment_backend_from_env()
        self.observer = LivingRoomObserver(
            navigation, manipulation, left_drawer, right_drawer
        )
        self.dispatcher = LivingRoomSkillDispatcher(
            navigation, manipulation, left_drawer, right_drawer
        )
        event_path = os.environ.get("TAMP_EVENT_LOG")
        self.events = EventLog(Path(event_path) if event_path else None)
        self.executive = TaskExecutive(
            FunctionRegistry.load(),
            self.backend,
            self.observer,
            self.dispatcher,
            LivingRoomStoragePlanner(),
            storage_goal_satisfied,
            event_log=self.events,
        )

    @property
    def busy(self) -> bool:
        return self.executive.busy

    @property
    def status(self) -> str:
        return self.executive.status

    @property
    def failure(self):
        return self.executive.failure_code

    @property
    def navigation_safe(self) -> bool:
        return bool(
            not self.busy
            and self.manipulation.navigation_safe
            and all(drawer.navigation_safe for drawer in self.drawers)
        )

    def progress(self) -> float:
        return self.executive.progress

    def request_store_controller(self) -> None:
        self.executive.start(STORE_CONTROLLER_TASK)

    def update(self) -> None:
        self.executive.update()

    def close(self) -> None:
        self.executive.close()
        self.events.close()
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
