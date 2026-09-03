from __future__ import annotations

import json

import pytest

from baseline_common.models import Entity, Observation, Region
from mujoco_scenes.tamp.discovery_replanning import PlanStatus, PlannerResult
from mujoco_scenes.tamp.skills import SkillAction, SkillResult


class FakeDispatcher:
    def __init__(self):
        self.action = None

    def prepare(self, _actions):
        return SkillResult.succeeded()

    def start(self, action):
        self.action = action

    def update(self):
        if self.action is None:
            return None
        action = self.action
        self.action = None
        return SkillResult.succeeded(f"holding({action.arguments['object_id']})")


class FakeRuntime:
    created = None

    def __init__(self):
        self.dispatcher = FakeDispatcher()
        self.closed = False
        self.effects = ()

    @classmethod
    def from_variant(cls, *_args, **_kwargs):
        cls.created = cls()
        return cls.created

    @staticmethod
    def _observation():
        return Observation(
            "kitchen",
            1,
            (Entity("mug", "object", "mug", {"region_id": "countertop"}),),
            (Region("countertop", "countertop", "open", True),),
            {"workspace": "home", "holding": None},
        )

    def observe(self):
        return self._observation(), ()

    def observe_state(self):
        return self._observation()

    def goal_verifier(self, observation):
        return observation.scene == "kitchen"

    def accept_effects(self, effects):
        self.effects = effects

    def open(self):
        return None

    def sync(self, _status):
        return None

    def close(self):
        self.closed = True


class FakePlanner:
    def __init__(self, _config):
        pass

    def plan(self, _request):
        return PlannerResult(
            PlanStatus.PLAN,
            (SkillAction("PICK", {"object_id": "mug"}),),
        )


def test_runner_wires_observation_planner_effects_and_goal_verifier(tmp_path, monkeypatch):
    import mujoco_scenes.run_kitchen_discovery_replanning as runner

    monkeypatch.setattr(runner, "BaselineKitchenRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "OpenAIDiscoveryPlanner", FakePlanner)

    output = tmp_path / "episode"
    result = runner.run_episode(
        variant="K1",
        output_dir=output,
        goal="Pick the mug.",
        base_url="http://unused/v1",
        model="unused",
        show_viewer=False,
        preflight_model_server=False,
    )

    assert result["success"] is True
    assert result["executed_actions"] == 1
    assert result["model_calls"] == 1
    assert result["protocol"] == "native"
    assert result["raw_vlm_requests"] == 1
    assert result["planning_latency_s"] == 0.0
    assert result["elapsed_seconds"] >= 0.0
    assert FakeRuntime.created.effects == ("holding(mug)",)
    assert FakeRuntime.created.closed is True
    assert json.loads((output / "discovery_replanning_result.json").read_text())["success"] is True


def test_runner_rejects_a_nonempty_output_directory(tmp_path):
    from mujoco_scenes.run_kitchen_discovery_replanning import run_episode

    output = tmp_path / "episode"
    output.mkdir()
    (output / "old_result.json").write_text("{}")

    with pytest.raises(ValueError, match="must be empty"):
        run_episode(
            variant="K1",
            output_dir=output,
            goal="test",
            base_url="http://unused/v1",
            model="unused",
            preflight_model_server=False,
        )


def test_single_call_protocol_rejects_a_larger_call_budget(tmp_path):
    from mujoco_scenes.run_kitchen_discovery_replanning import run_episode

    with pytest.raises(ValueError, match="requires max_model_calls=1"):
        run_episode(
            variant="K1",
            output_dir=tmp_path / "episode",
            goal="test",
            base_url="http://unused/v1",
            model="unused",
            protocol="single_call",
            max_model_calls=2,
            preflight_model_server=False,
        )
