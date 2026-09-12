"""The FM evidence ablation must withhold evidence, and nothing else.

The failure mode these guard against is an ablation that looks like it ran but
did not: a channel left reachable through a fallback, a mask that never got
applied, or a mask that removed something which is not evidence and so measured
the wrong thing.
"""
from __future__ import annotations

import pytest

from mujoco_scenes.fm_ablation_shadow import (
    evidence_masked, mask_observed_graph, observed_evidence_present,
)
from mujoco_scenes.fm_evidence_ablation import (
    COMPONENT_MASKS, CONDITION_LABELS, CONDITION_ORDER, EVIDENCE_COMPONENTS,
    numeric_properties_referenced, resolve_components,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation, FunctionalRequirementGraph, FunctionalRole,
    NumericConstraint, OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode, ObservedRelation, ObservedSceneGraph,
)


def _spec() -> FunctionalRequirementGraph:
    return FunctionalRequirementGraph(
        domain="kitchen", task_instruction="t",
        nodes={
            "tool": FunctionalRole(name="tool", semantic_categories=("spoon",),
                                   unary_predicates=("is_graspable",),
                                   numeric_constraints=(
                                       NumericConstraint(property_name="length",
                                                         operator=">=", threshold=0.1,
                                                         unit="m"),)),
            "target": FunctionalRole(name="target", semantic_categories=("mug",)),
        },
        relations=(FunctionalRelation(subject_role="tool", predicate="INSERTABLE_IN",
                                      object_role="target"),),
        operation_groups=(
            OperationGroup(id="g1", function="stir", tool_role="tool",
                           target_role="target", required_target_count=1,
                           usage_policy="DEDICATED_PER_TARGET",
                           required_relations=("INSERTABLE_IN",)),),
        source="VLM_CANONICAL_G_F")


def _graph() -> ObservedSceneGraph:
    graph = ObservedSceneGraph()
    graph.add_node(ObservedNode(
        instance_id="o1", entity_kind="OBJECT", canonical_category="spoon",
        semantic_labels={"status": "TRUE"},
        unary_properties={"length": 0.2}, unary_predicates={"is_graspable": "TRUE"},
        # `length` is duplicated into geometry on purpose: this is the fallback
        # path that made an earlier unary mask a no-op.
        geometry={"length": 0.2, "pose": [0, 0, 0], "properties": {"length": 0.2}}))
    graph.add_node(ObservedNode(
        instance_id="o2", entity_kind="OBJECT", canonical_category="mug",
        geometry={"pose": [1, 0, 0]}))
    graph.add_relation(ObservedRelation(
        predicate="INSERTABLE_IN", subject_id="o1", object_id="o2", status="TRUE"))
    return graph


# --- the conditions ---------------------------------------------------------

def test_conditions_are_cumulative():
    """Each condition adds exactly one channel to the previous one.

    A difference between two rows is then attributable to the single channel
    that was added, which is the whole point of reporting it this way.
    """
    assert CONDITION_ORDER == ("semantic_only", "semantic_unary", "full")
    previous: tuple[str, ...] = ()
    for name in CONDITION_ORDER:
        current = COMPONENT_MASKS[name]
        assert set(previous).issubset(set(current))
        assert len(current) == len(previous) + 1
        previous = current
    assert COMPONENT_MASKS["full"] == EVIDENCE_COMPONENTS


def test_every_condition_has_a_label():
    assert set(CONDITION_LABELS) == set(COMPONENT_MASKS)


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        resolve_components("no_semantic")


# --- masking the observation ------------------------------------------------

@pytest.mark.parametrize("condition", list(COMPONENT_MASKS))
def test_mask_leaves_exactly_the_enabled_channels(condition):
    masked = mask_observed_graph(_graph(), condition, _spec())
    present = observed_evidence_present(masked)
    for channel in EVIDENCE_COMPONENTS:
        assert present[channel] is (channel in COMPONENT_MASKS[condition]), (
            f"{condition}: {channel} should be "
            f"{'present' if channel in COMPONENT_MASKS[condition] else 'absent'}")


def test_unary_mask_closes_the_geometry_fallback():
    """Grounding reads numerics from unary_properties, then geometry.

    Clearing only unary_properties leaves the value reachable, so the ablation
    would report a condition it never actually ran.
    """
    masked = mask_observed_graph(_graph(), "semantic_only", _spec())
    node = masked.get_node("o1")
    assert node.unary_properties == {}
    assert "length" not in node.geometry
    assert "length" not in node.geometry.get("properties", {})


def test_unary_mask_keeps_geometry_the_planner_needs():
    """Withhold the measurements the checks read, not all of geometry.

    Clearing geometry wholesale would break planning, and the run would fail
    for a reason unrelated to evidence.
    """
    masked = mask_observed_graph(_graph(), "semantic_only", _spec())
    assert masked.get_node("o1").geometry.get("pose") == [0, 0, 0]
    assert masked.get_node("o2").geometry.get("pose") == [1, 0, 0]


def test_only_referenced_numeric_properties_are_withheld():
    assert numeric_properties_referenced(_spec()) == {"length"}


def test_full_condition_withholds_nothing():
    """`full` is the reference: it must be an identity on the observation,
    or every delta is measured against something that is not the pipeline."""
    before = observed_evidence_present(_graph())
    after = observed_evidence_present(mask_observed_graph(_graph(), "full", _spec()))
    assert before == after == {"semantic": True, "unary": True, "binary": True}
    masked = mask_observed_graph(_graph(), "full", _spec())
    assert masked.get_node("o1").unary_properties == {"length": 0.2}
    assert masked.get_node("o1").geometry["length"] == 0.2
    assert len(masked.relations) == 1


def test_masking_does_not_touch_the_functional_graph():
    """G_F states the task and is not evidence.

    Masking it is what broke the earlier design: the task-interface validator
    rejects a graph with required_relations stripped, correctly.
    """
    spec = _spec()
    before = spec.to_dict()
    mask_observed_graph(_graph(), "semantic_only", spec)
    assert spec.to_dict() == before


# --- the patch itself -------------------------------------------------------

def test_patch_applies_and_restores():
    from mujoco_scenes.functional_tamp_pipeline import grounding

    original = grounding.ground_graph
    with evidence_masked("semantic_only"):
        assert grounding.ground_graph is not original
    assert grounding.ground_graph is original


def test_patch_restores_even_if_the_run_raises():
    from mujoco_scenes.functional_tamp_pipeline import grounding

    original = grounding.ground_graph
    with pytest.raises(RuntimeError):
        with evidence_masked("full"):
            raise RuntimeError("pipeline blew up")
    assert grounding.ground_graph is original, "a failed run must not leak the patch"


def test_grounding_through_the_patch_sees_a_masked_observation():
    """End to end through the real ground_graph, with the mask in force."""
    from mujoco_scenes.functional_tamp_pipeline import grounding

    seen = {}
    original = grounding.ground_graph

    def spy(graph_f, graph_o, context=None, *a, **k):
        seen.update(observed_evidence_present(graph_o))
        return original(graph_f, graph_o, context, *a, **k)

    grounding.ground_graph = spy
    try:
        with evidence_masked("semantic_unary") as stats:
            grounding.ground_graph(_spec(), _graph(), {"search_exhausted": True})
    finally:
        grounding.ground_graph = original
    assert seen == {"semantic": True, "unary": True, "binary": False}
    assert stats["groundings"] == 1


def test_unknown_condition_rejected_before_anything_runs():
    with pytest.raises(ValueError):
        with evidence_masked("binary_only"):
            pass
