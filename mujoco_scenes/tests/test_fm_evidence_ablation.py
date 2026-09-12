"""The FM evidence ablation must ablate evidence, and nothing else.

The failure mode these guard against is an ablation that looks like it ran but
did not: a mask that leaves a channel in force, or one that removes something
which is not evidence at all and so measures the wrong thing.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mujoco_scenes.fm_evidence_ablation import (
    COMPONENT_MASKS, EVIDENCE_COMPONENTS, evidence_channels_present,
    mask_specification, resolve_components,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation, FunctionalRequirementGraph, FunctionalRole,
    NumericConstraint, OperationGroup,
)


def _spec() -> FunctionalRequirementGraph:
    """A graph carrying all three evidence channels, including the tricky ones."""
    return FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="t",
        nodes={
            "tool": FunctionalRole(name="tool", semantic_categories=("spoon",),
                                   unary_predicates=("is_graspable",),
                                   numeric_constraints=(
                                       NumericConstraint(property_name="length",
                                                         operator=">=", threshold=0.1,
                                                         unit="m"),),
                                   semantic_hints=("stirrer",)),
            "target": FunctionalRole(name="target", semantic_categories=("mug",)),
        },
        relations=(FunctionalRelation(subject_role="tool", predicate="INSERTABLE_IN",
                                      object_role="target"),),
        operation_groups=(
            OperationGroup(id="g1", function="stir", tool_role="tool",
                           target_role="target", required_target_count=1,
                           usage_policy="DEDICATED_PER_TARGET",
                           required_relations=("INSERTABLE_IN",),
                           physical_preconditions=(("tool", "REACHES_BOTTOM", "target"),)),
        ),
        source="VLM_CANONICAL_G_F",
    )


@pytest.mark.parametrize("condition", sorted(COMPONENT_MASKS))
def test_every_mask_leaves_exactly_the_enabled_channels(condition):
    enabled = resolve_components(condition)
    present = evidence_channels_present(mask_specification(_spec(), condition))
    for channel in EVIDENCE_COMPONENTS:
        assert present[channel] is (channel in enabled), (
            f"{condition}: channel {channel} should be "
            f"{'present' if channel in enabled else 'absent'}")


def test_masks_are_the_seven_non_empty_subsets():
    assert len(COMPONENT_MASKS) == 7
    assert {frozenset(v) for v in COMPONENT_MASKS.values()} == {
        frozenset(s) for s in
        [("semantic",), ("unary",), ("binary",), ("semantic", "unary"),
         ("semantic", "binary"), ("unary", "binary"), EVIDENCE_COMPONENTS]}


def test_physical_preconditions_are_cleared_by_the_binary_mask():
    """They override required_relations in the grounder.

    Clearing only required_relations would leave binary geometry fully in force
    for every group declaring preconditions -- an ablation that silently did
    nothing, while reporting that it had.
    """
    masked = mask_specification(_spec(), "no_binary")
    assert all(not g.physical_preconditions for g in masked.operation_groups)
    assert all(not g.required_relations for g in masked.operation_groups)


def test_masking_does_not_ablate_the_task_itself():
    """Entity kind, cardinality and binding policy are the task, not evidence."""
    spec = _spec()
    for condition in COMPONENT_MASKS:
        masked = mask_specification(spec, condition)
        assert set(masked.nodes) == set(spec.nodes)
        for name, role in masked.nodes.items():
            original = spec.nodes[name]
            assert role.entity_kind == original.entity_kind
            assert role.minimum_count == original.minimum_count
            assert role.binding_policy == original.binding_policy
        assert len(masked.operation_groups) == len(spec.operation_groups)
        assert masked.operation_groups[0].required_target_count == 1


def test_source_survives_so_the_pipeline_still_accepts_the_graph():
    for condition in COMPONENT_MASKS:
        assert mask_specification(_spec(), condition).source == "VLM_CANONICAL_G_F"


def test_full_condition_is_an_evidence_preserving_identity():
    """`full` is the reference; if it changed the graph, every delta would be
    measured against something that is not the shipped system."""
    spec = _spec()
    masked = mask_specification(spec, "full")
    assert masked.nodes == spec.nodes
    assert masked.relations == spec.relations
    assert masked.operation_groups == spec.operation_groups


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        resolve_components("geometry_only")


def test_masking_matches_component_gating():
    """Input masking must equal the GT ablation's in-grounder gating.

    The grounder skips a check exactly when the field carrying it is empty, so
    clearing the field is the same experiment as branching on a component set.
    This pins that equivalence, because it is the assumption the whole shadow
    design rests on.
    """
    from mujoco_scenes.functional_tamp_pipeline.grounding import evaluate_node_for_role
    from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode

    node = ObservedNode(instance_id="o1", entity_kind="OBJECT",
                        canonical_category="spoon",
                        unary_properties={"length": 0.05})
    role = _spec().nodes["tool"]
    # Unary evidence refutes this node (length 0.05 < 0.1).
    assert evaluate_node_for_role(node, role)[0] == "FALSE"
    # With the unary channel masked out of the role, the same check cannot fire.
    unary_masked = mask_specification(_spec(), "semantic_only").nodes["tool"]
    assert evaluate_node_for_role(node, unary_masked)[0] != "FALSE"
