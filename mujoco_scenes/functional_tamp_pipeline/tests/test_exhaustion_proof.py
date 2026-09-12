"""An exhausted search over a complete contract is a conclusion, not a partial.

Before this, a run that stated the task fully, enumerated every candidate it
could observe and rejected all of them reported PARTIAL_ACTION_SEQUENCE_READY --
the same status as a run that simply stopped early.  The conclusion the evidence
supported was discarded, and infeasibility was never validly concluded on any of
the 36 archived infeasible trials.

The three conditions are jointly required, and the branch sits below the
full-plan branch so it can never displace a success or claim a completion.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "mujoco_scenes" / "functional_tamp_pipeline" / "run.py"

from mujoco_scenes.evaluation_outcome import (
    INFEASIBILITY_CONCLUDED_STATUSES,
    is_false_completion,
    outcome_is_correct,
)


def _decide(is_full_plan, contract_complete, exhausted, grounded, has_actions):
    """The decision the patched branch makes, mirrored for testing."""
    if is_full_plan:
        return "ACTION_SEQUENCE_READY"
    if contract_complete and exhausted and not grounded:
        return "EXHAUSTED_NO_VALID_GROUNDING"
    if has_actions:
        return "PARTIAL_ACTION_SEQUENCE_READY"
    return "NO_MEANINGFUL_CANDIDATE_PLAN"


def test_all_three_conditions_are_required():
    assert _decide(False, True, True, False, True) == "EXHAUSTED_NO_VALID_GROUNDING"
    # No complete contract: the run never stated the task.
    assert _decide(False, False, True, False, True) == "PARTIAL_ACTION_SEQUENCE_READY"
    # Not exhausted: the run has not finished looking.
    assert _decide(False, True, False, False, True) == "PARTIAL_ACTION_SEQUENCE_READY"
    # Grounded: there is a valid assignment, so nothing was proven impossible.
    assert _decide(False, True, True, True, True) == "PARTIAL_ACTION_SEQUENCE_READY"


def test_it_can_never_displace_a_success():
    assert _decide(True, True, True, False, True) == "ACTION_SEQUENCE_READY"


def test_it_can_never_create_a_false_completion():
    status = _decide(False, True, True, False, True)
    assert not is_false_completion(pipeline_status=status, gt_full_task_satisfied=False)


def test_the_conclusion_is_accepted_only_with_a_complete_contract():
    status = "EXHAUSTED_NO_VALID_GROUNDING"
    assert status in INFEASIBILITY_CONCLUDED_STATUSES
    assert outcome_is_correct(gt_feasible=False, gt_full_task_satisfied=False,
                              pipeline_status=status, runtime_contract_complete=True)
    assert not outcome_is_correct(gt_feasible=False, gt_full_task_satisfied=False,
                                  pipeline_status=status, runtime_contract_complete=False)


def test_a_feasible_variant_does_not_become_correct_by_concluding_exhaustion():
    """Concluding exhaustion on a possible task is honest about the evidence
    but is not a success, and must never be scored as one."""
    assert not outcome_is_correct(gt_feasible=True, gt_full_task_satisfied=False,
                                  pipeline_status="EXHAUSTED_NO_VALID_GROUNDING",
                                  runtime_contract_complete=True)


def test_every_domain_applies_it():
    """Three domains decide terminal status in three different places.

    The first implementation went into run.py only, so workshop got the
    conclusion and kitchen and living room silently did not -- four trials met
    every condition and still reported a partial plan.
    """
    for rel in ("mujoco_scenes/functional_tamp_pipeline/run.py",
                "mujoco_scenes/functional_tamp_pipeline/domains/kitchen.py",
                "mujoco_scenes/functional_tamp_pipeline/domains/living_room.py"):
        source = (REPO / rel).read_text()
        assert "exhaustion_proves_no_valid_grounding" in source, (
            f"{rel} decides a terminal status without the exhaustion conclusion")
        full = source.index('status = "ACTION_SEQUENCE_READY"')
        exhausted = source.index('status = "EXHAUSTED_NO_VALID_GROUNDING"')
        partial = source.index('status = "PARTIAL_ACTION_SEQUENCE_READY"')
        assert full < exhausted < partial, f"{rel}: branch order is wrong"


def test_the_helper_requires_all_three_conditions():
    from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
        exhaustion_proves_no_valid_grounding as proves,
    )

    class Spec:
        def __init__(self, c): self.online_executable_contract_complete = c

    class Sat:
        def __init__(self, ex, comp): self.evidence = {"search_exhausted": ex}; self.complete = comp

    assert proves(Spec(True), Sat(True, False))
    assert not proves(Spec(False), Sat(True, False))
    assert not proves(Spec(True), Sat(False, False))
    assert not proves(Spec(True), Sat(True, True))


def test_the_branch_is_ordered_below_the_full_plan_branch_in_source():
    source = RUN.read_text()
    full = source.index('status = "ACTION_SEQUENCE_READY"')
    exhausted = source.index('status = "EXHAUSTED_NO_VALID_GROUNDING"')
    partial = source.index('status = "PARTIAL_ACTION_SEQUENCE_READY"')
    assert full < exhausted < partial, (
        "order must be full plan, then exhaustion proof, then partial: "
        "above the full-plan branch it would displace successes, below the "
        "partial branch it would never fire")
