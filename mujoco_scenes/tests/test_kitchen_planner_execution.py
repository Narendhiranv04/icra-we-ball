import pytest

from mujoco_scenes import run_kitchen_planner_execution as runner
from mujoco_scenes.run_kitchen_planner_execution import execute_plan, planner_actions
from mujoco_scenes.tamp.skills import SkillAction


def test_llm3_plan_is_accepted_without_rewriting_ids():
    actions = planner_actions(
        {
            "plan": {
                "status": "PLAN",
                "actions": [
                    {"skill": "PICK", "arguments": {"object_id": "spoon_7"}},
                    {
                        "skill": "STIR",
                        "arguments": {"tool_id": "spoon_7", "target_id": "mug_2"},
                    },
                ],
            }
        }
    )
    assert [action.name for action in actions] == ["PICK", "STIR"]
    assert actions[1].arguments["target_id"] == "mug_2"


def test_phase_c_list_arguments_are_normalized():
    actions = planner_actions(
        [{"action": "POUR", "arguments": ["kettle", "mug", "water"]}]
    )
    assert actions[0].arguments == {
        "source_id": "kettle",
        "target_id": "mug",
        "content": "water",
    }


def test_unknown_action_fails_closed():
    with pytest.raises(ValueError, match="Unsupported kitchen action"):
        planner_actions([{"skill": "TELEPORT", "arguments": {}}])


def test_malformed_action_arguments_fail_closed():
    rows = (
        {"action": "STIR", "arguments": ["spoon"]},
        {"action": "PICK", "arguments": ["spoon", "extra"]},
        {"skill": "PLACE", "arguments": {"object_id": "mug"}},
        {
            "skill": "PICK",
            "arguments": {"object_id": "mug", "unexpected": "value"},
        },
    )
    for row in rows:
        with pytest.raises(ValueError):
            planner_actions([row])


def test_goal_and_live_status_are_carried_through_execution(monkeypatch):
    seen = {}

    class FakeExecutive:
        mode = "complete"
        history = []
        executed_actions = 1
        replans = 0
        status = "Executing STIR (1/1)"

    class FakeExecution:
        def __init__(self, *_args, step_callback=None, **_kwargs):
            self.executive = FakeExecutive()
            self.step_callback = step_callback

        def start(self, _task_id, goal, _witness):
            seen["goal"] = goal

        def run(self):
            self.step_callback()
            return self.executive

    monkeypatch.setattr(runner, "KitchenGroundedExecution", FakeExecution)
    statuses = []
    scene = type(
        "Scene",
        (),
        {
            "config": type("Config", (), {"name": "scene", "goal": "default"})(),
        },
    )()
    result = execute_plan(
        scene,
        {},
        {},
        {},
        {"status": "COMPLETE"},
        (SkillAction("STIR", {"tool_id": "tool", "target_id": "mug"}),),
        goal="Stir the visible mug",
        step_callback=statuses.append,
    )
    assert seen["goal"] == "Stir the visible mug"
    assert result["goal"] == "Stir the visible mug"
    assert statuses == ["Executing STIR (1/1)"]
