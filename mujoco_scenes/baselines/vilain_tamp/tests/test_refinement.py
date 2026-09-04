from __future__ import annotations

from dataclasses import dataclass, replace
import json

import pytest

from mujoco_scenes.baselines.vilain_tamp.contracts import (
    ExecutionProjection,
    RefinementStage,
    SymbolicAction,
)
from mujoco_scenes.baselines.vilain_tamp.refinement import (
    ADAPTATION_LABEL,
    MuJoCoSequenceRefiner,
    RefinementContractError,
    RefinementStageContext,
    RefinementStageOutcome,
)


@dataclass
class FakePlanningScene:
    state: dict[str, object]


class FakeStateAdapter:
    def snapshot(self, planning_scene: FakePlanningScene):
        return dict(planning_scene.state)

    def apply_predicted_transition(
        self,
        planning_scene: FakePlanningScene,
        action: SymbolicAction,
        predicted_terminal_state,
    ) -> None:
        del action
        planning_scene.state.update(predicted_terminal_state)


class FakeStageBackend:
    def __init__(self, *, fail_at=None, raise_at=None):
        self.fail_at = fail_at
        self.raise_at = raise_at
        self.calls: list[tuple[FakePlanningScene, RefinementStageContext]] = []

    def refine_stage(
        self,
        planning_scene: FakePlanningScene,
        context: RefinementStageContext,
    ) -> RefinementStageOutcome:
        self.calls.append((planning_scene, context))
        if context.stage is self.raise_at:
            raise RuntimeError("synthetic backend crash")
        if context.stage is self.fail_at:
            return RefinementStageOutcome(
                stage=context.stage,
                success=False,
                resolved_entities=context.projection.resolved_entities,
                collision_free=False,
                numeric_evidence={"signed_distance_m": -0.012},
                reason_code="PATH_COLLISION",
                summary="approach path intersects the cabinet",
                robot_or_arm="arm",
                involved_entities=context.projection.resolved_entities,
                collision_pair=("gripper", "cabinet"),
                backend_trace_artifact="traces/collision.json",
                recoverable_by_problem_revision=True,
            )
        terminal = None
        if context.stage is RefinementStage.STATE_TRANSITION:
            terminal = {
                "last_action": context.action.action_instance_id,
                "completed_actions": int(context.predicted_state.get("completed_actions", 0))
                + 1,
            }
        return RefinementStageOutcome(
            stage=context.stage,
            success=True,
            candidate_artifacts=(f"candidates/{context.action.action_instance_id}.json",),
            trajectory_artifacts=(f"paths/{context.action.action_instance_id}.json",),
            resolved_entities=context.projection.resolved_entities,
            chosen_grasp={"candidate_id": "grasp_0"},
            chosen_target_pose={"frame": "target", "pose_id": "pose_0"},
            collision_free=True,
            reachable=True,
            numeric_evidence={"clearance_m": 0.03},
            predicted_terminal_state=terminal,
            backend_trace_artifact=f"traces/{context.action.action_instance_id}.json",
        )


def action(index: int, operator: str, arguments=("object_1", "surface")):
    return SymbolicAction(
        action_index=index,
        action_instance_id=f"vilain_00_{index + 1:03d}_{operator.replace('-', '_')}",
        operator=operator,
        arguments=tuple(arguments),
    )


def projection(symbolic_action: SymbolicAction):
    entities = tuple(f"body_{item}" for item in symbolic_action.arguments)
    return ExecutionProjection(
        action_instance_id=symbolic_action.action_instance_id,
        pddl_operator=symbolic_action.operator,
        pddl_arguments=symbolic_action.arguments,
        controller_operator="TEST",
        controller_arguments=entities,
        resolved_entities=entities,
        binding_method="ONE_TO_ONE_CLASS_CENTROID_AABB",
        binding_confidence=0.9,
        binding_evidence_artifacts=("identity/evidence.json",),
        skill_parameters={},
    )


def make_refiner(backend: FakeStageBackend, clock=None):
    return MuJoCoSequenceRefiner(
        stage_backends={stage: backend for stage in RefinementStage},
        state_adapter=FakeStateAdapter(),
        clock=clock or __import__("time").perf_counter,
    )


def test_sequence_propagates_predicted_state_on_one_fresh_copy(tmp_path) -> None:
    live_state = {"completed_actions": 0, "live_marker": "untouched"}
    planning_scene = FakePlanningScene(dict(live_state))
    factory_calls = []

    def planning_scene_factory():
        factory_calls.append(True)
        return planning_scene

    backend = FakeStageBackend()
    actions = (action(0, "pick-from"), action(1, "place-on"))
    result = make_refiner(backend).refine(
        attempt_index=0,
        actions=actions,
        projections=tuple(projection(item) for item in actions),
        planning_scene_factory=planning_scene_factory,
        output_root=tmp_path,
    )

    assert result.success
    assert result.failure is None
    assert result.certificate is not None
    assert result.certificate.adaptation_label == ADAPTATION_LABEL
    assert result.certificate.final_predicted_state["completed_actions"] == 2
    assert len(result.certificate.actions) == 2
    assert len(result.certificate.actions[0].stages) == len(RefinementStage)
    second_start = next(
        context
        for _, context in backend.calls
        if context.action.action_index == 1
        and context.stage is RefinementStage.ENTITY_RESOLUTION
    )
    assert second_start.predicted_state["completed_actions"] == 1
    assert factory_calls == [True]
    assert live_state == {"completed_actions": 0, "live_marker": "untouched"}
    persisted = json.loads((tmp_path / "refinement.json").read_text(encoding="utf-8"))
    assert persisted["success"] is True


def test_first_failure_stops_sequence_and_serializes_detail(tmp_path) -> None:
    backend = FakeStageBackend(fail_at=RefinementStage.COLLISION)
    actions = (action(0, "pick-from"), action(1, "place-on"))
    result = make_refiner(backend).refine(
        attempt_index=2,
        actions=actions,
        projections=tuple(projection(item) for item in actions),
        planning_scene_factory=lambda: FakePlanningScene({"completed_actions": 0}),
        output_root=tmp_path,
    )

    assert not result.success
    assert result.certificate is None
    assert result.failure is not None
    assert result.failure.attempt_index == 2
    assert result.failure.action_index == 0
    assert result.failure.action_instance_id == actions[0].action_instance_id
    assert result.failure.stage is RefinementStage.COLLISION
    assert result.failure.reason_code == "PATH_COLLISION"
    assert result.failure.collision_pair == ("gripper", "cabinet")
    assert result.failure.numeric_evidence == {"signed_distance_m": -0.012}
    assert result.failure.backend_trace_artifact == "traces/collision.json"
    assert len(backend.calls) == 5
    persisted = json.loads((tmp_path / "refinement.json").read_text(encoding="utf-8"))
    assert persisted["failure"]["stage"] == "COLLISION"


@pytest.mark.parametrize(
    "operator",
    [
        "open-storage",
        "pick-from",
        "place-on",
        "insert",
        "pour",
        "stir",
        "drive",
        "place-in",
    ],
)
def test_every_supported_action_runs_all_stage_categories(operator: str) -> None:
    backend = FakeStageBackend()
    current = action(0, operator)
    result = make_refiner(backend).refine(
        attempt_index=0,
        actions=(current,),
        projections=(projection(current),),
        planning_scene_factory=lambda: FakePlanningScene({}),
    )
    assert result.success
    assert [context.stage for _, context in backend.calls] == list(RefinementStage)
    assert all(context.stage_purpose for _, context in backend.calls)


@pytest.mark.parametrize("operator", ["pour", "stir", "drive"])
def test_contact_rich_skills_use_only_black_box_envelopes(operator: str) -> None:
    backend = FakeStageBackend()
    current = action(0, operator)
    make_refiner(backend).refine(
        attempt_index=0,
        actions=(current,),
        projections=(projection(current),),
        planning_scene_factory=lambda: FakePlanningScene({}),
    )
    envelope = next(
        context
        for _, context in backend.calls
        if context.stage is RefinementStage.SKILL_ENVELOPE
    )
    assert envelope.interaction_mode == "BLACK_BOX_SKILL_ENVELOPE"
    assert "object_centric_envelope" in envelope.stage_purpose


def test_backend_exception_becomes_terminal_structured_failure() -> None:
    backend = FakeStageBackend(raise_at=RefinementStage.IK)
    current = action(0, "pick-from")
    result = make_refiner(backend).refine(
        attempt_index=0,
        actions=(current,),
        projections=(projection(current),),
        planning_scene_factory=lambda: FakePlanningScene({}),
    )
    assert result.failure is not None
    assert result.failure.stage is RefinementStage.IK
    assert result.failure.reason_code == "REFINEMENT_BACKEND_EXCEPTION"
    assert not result.failure.recoverable_by_problem_revision


def test_contract_rejects_foreign_artifacts_and_projection_mismatch() -> None:
    current = action(0, "pick-from")
    refiner = make_refiner(FakeStageBackend())
    with pytest.raises(RefinementContractError, match="external method artifacts"):
        refiner.refine(
            attempt_index=0,
            actions=(current,),
            projections=(projection(current),),
            planning_scene_factory=lambda: FakePlanningScene({}),
            external_method_artifacts={"artifact": "foreign"},
        )
    wrong = action(1, "pick-from")
    with pytest.raises(RefinementContractError, match="identity mismatch"):
        refiner.refine(
            attempt_index=0,
            actions=(current,),
            projections=(projection(wrong),),
            planning_scene_factory=lambda: FakePlanningScene({}),
        )
    mismatched_content = projection(current)
    mismatched_content = replace(
        mismatched_content,
        pddl_arguments=("different_object", "surface"),
    )
    with pytest.raises(RefinementContractError, match="content mismatch"):
        refiner.refine(
            attempt_index=0,
            actions=(current,),
            projections=(mismatched_content,),
            planning_scene_factory=lambda: FakePlanningScene({}),
        )


def test_missing_stage_backend_is_rejected() -> None:
    backend = FakeStageBackend()
    with pytest.raises(RefinementContractError, match="missing refinement"):
        MuJoCoSequenceRefiner(
            stage_backends={RefinementStage.IK: backend},
            state_adapter=FakeStateAdapter(),
        )


def test_refinement_module_has_no_simulator_or_controller_imports() -> None:
    from mujoco_scenes.baselines.vilain_tamp import refinement

    source = __import__("inspect").getsource(refinement)
    assert "import mujoco" not in source
    assert "grounding" not in source.lower()
    assert "controller" not in source.lower()
