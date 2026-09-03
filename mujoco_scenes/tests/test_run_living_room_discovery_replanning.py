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

    def __init__(self, *_args, **_kwargs):
        type(self).created = self
        self.dispatcher = FakeDispatcher()
        self.object_annotation_aliases = {"cup": "cup"}
        self.closed = False

    @staticmethod
    def _observation():
        return Observation(
            "living_room",
            1,
            (Entity("cup", "object", "cup", {"region_id": "staging"}),),
            (Region("staging", "staging", "open", True),),
            {"workspace": "home", "holding": None},
        )

    def observe(self):
        return self._observation(), ()

    def goal_verifier(self):
        return True

    def accept_effects(self, _effects):
        return None

    def open(self):
        return None

    def sync(self, _status):
        return None

    def close(self):
        self.closed = True


class FakePlanner:
    def __init__(self, *_args, **_kwargs):
        pass

    def plan(self, _request):
        return PlannerResult(PlanStatus.PLAN, (SkillAction("PICK", {"object_id": "cup"}),))


def test_runner_wires_living_room_physical_adapter(tmp_path, monkeypatch):
    import mujoco_scenes.run_living_room_discovery_replanning as runner

    monkeypatch.setattr(runner, "LivingRoomDiscoveryRuntime", FakeRuntime)
    monkeypatch.setattr(runner, "OpenAIDiscoveryPlanner", FakePlanner)
    output = tmp_path / "episode"

    result = runner.run_episode(
        variant="L1",
        output_dir=output,
        goal="Put the cup away.",
        base_url="http://unused/v1",
        model="unused",
        show_viewer=False,
        preflight_model_server=False,
    )

    assert result["success"]
    assert result["executed_actions"] == 1
    assert result["model_calls"] == 1
    assert result["protocol"] == "native"
    assert result["raw_vlm_requests"] == 1
    assert result["planning_latency_s"] == 0.0
    assert result["elapsed_seconds"] >= 0.0
    assert FakeRuntime.created.closed
    assert json.loads((output / "discovery_replanning_result.json").read_text())["scene"] == "living_room"


def test_runner_rejects_nonempty_output_directory(tmp_path):
    from mujoco_scenes.run_living_room_discovery_replanning import run_episode

    output = tmp_path / "episode"
    output.mkdir()
    (output / "old.json").write_text("{}")
    with pytest.raises(ValueError, match="must be empty"):
        run_episode(
            variant="L1",
            output_dir=output,
            goal="test",
            base_url="http://unused/v1",
            model="unused",
            preflight_model_server=False,
        )
