"""One context instance per application, when the context is one per person.

Three quantities were being written as one number: how many times the task
applies an operation, how many physical instances that needs, and whether one
instance may serve several applications.  Only the last is a fact about the
runtime's own role, and the runtime declares it per canonical role.

The failure these tests pin down was a reported completion of a two-person task
against one person's seat: the personal placement compiled a single shared
seating context, grounding paired the first placement with that seat, and the
second placement was evaluated with no context at all -- so its proximity
precondition was never checked and the binding was recorded satisfied with an
empty context.

Nothing here reads a reference graph, an expected plan, or a benchmark variant.
"""

from __future__ import annotations

from dataclasses import replace

from mujoco_scenes.functional_tamp_pipeline.grounding import (
    _evaluate_operation_group,
    _operation_context_schedules,
    ground_graph,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.operation_slot_completion import _slot_cardinality
from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (
    ONE_PER_APPLICATION,
    REUSABLE_ACROSS_APPLICATIONS,
    role_reuse_admissibility,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)


# ---------------------------------------------------------------------------
# What the runtime declares about its own roles
# ---------------------------------------------------------------------------


def test_personal_roles_are_one_per_application_and_shared_ones_are_not():
    assert role_reuse_admissibility("living_room", "SEATING_POSITION") == ONE_PER_APPLICATION
    assert role_reuse_admissibility("living_room", "PERSONAL_CUP_SAUCER_REGION") == ONE_PER_APPLICATION
    assert role_reuse_admissibility("living_room", "SEATING_PAIR") == REUSABLE_ACROSS_APPLICATIONS
    assert role_reuse_admissibility("living_room", "SHARED_REMOTE_REGION") == REUSABLE_ACROSS_APPLICATIONS


def test_material_sources_and_implements_are_reusable_but_servings_are_not():
    assert role_reuse_admissibility("kitchen", "coffee_source") == REUSABLE_ACROSS_APPLICATIONS
    assert role_reuse_admissibility("kitchen", "water_source") == REUSABLE_ACROSS_APPLICATIONS
    assert role_reuse_admissibility("kitchen", "coffee_stirrer") == REUSABLE_ACROSS_APPLICATIONS
    assert role_reuse_admissibility("kitchen", "coffee_container") == ONE_PER_APPLICATION
    assert role_reuse_admissibility("kitchen", "soup_eating_utensil") == ONE_PER_APPLICATION


def test_a_role_the_runtime_declares_nothing_about_is_not_assumed_reusable():
    assert role_reuse_admissibility("kitchen", "countertop") is None
    assert role_reuse_admissibility("living_room", "not_a_role") is None


# ---------------------------------------------------------------------------
# Induced slot cardinality
# ---------------------------------------------------------------------------


def test_two_personal_placements_induce_two_distinct_seats():
    assert _slot_cardinality("living_room", "anchor", "SEATING_POSITION", 2) == (2, "DISTINCT")


def test_one_personal_placement_induces_one_seat():
    assert _slot_cardinality("living_room", "anchor", "SEATING_POSITION", 1) == (1, "SHARED")


def test_the_seating_pair_stays_one_shared_compound_reference():
    assert _slot_cardinality("living_room", "anchor", "SEATING_PAIR", 1) == (1, "SHARED")
    assert _slot_cardinality("living_room", "anchor", "SEATING_PAIR", 2) == (1, "SHARED")


def test_the_marked_fastening_location_stays_one_however_many_fastenings():
    assert _slot_cardinality("workshop", "anchor", "repair_target", 1) == (1, "SHARED")
    assert _slot_cardinality("workshop", "anchor", "repair_target", 3) == (1, "SHARED")


def test_a_reusable_source_poured_twice_is_not_two_objects():
    count, policy = _slot_cardinality("kitchen", "source", "coffee_source", 2)
    assert (count, policy) == (2, "REUSABLE"), "two applications, one jar admissible"


def test_a_serving_container_filled_twice_is_two_objects():
    assert _slot_cardinality("kitchen", "target", "coffee_container", 2) == (2, "DISTINCT")


# ---------------------------------------------------------------------------
# The context schedule: a checked context may never go missing
# ---------------------------------------------------------------------------


def _group(context_relations=("NEAR_SEAT",), policy="DEDICATED_PER_TARGET"):
    return OperationGroup(
        id="personal", function="SUPPORT_DRINKWARE",
        tool_role="PERSONAL_CUP_SAUCER_REGION", target_role="CUP_SAUCER_SET",
        context_role="SEATING_POSITION", required_target_count=2, usage_policy=policy,
        required_relations=("FITS_SET_ON",), context_relations=tuple(context_relations),
        capability_id="SUPPORT_DRINKWARE",
    )


def test_two_applications_and_two_contexts_are_matched_in_every_order():
    schedules, refusal = _operation_context_schedules(_group(), ["seat_0001", "seat_0002"], 2)
    assert refusal is None
    assert set(schedules) == {("seat_0001", "seat_0002"), ("seat_0002", "seat_0001")}


def test_one_context_is_shared_by_every_application_and_still_checked():
    schedules, refusal = _operation_context_schedules(_group(), ["seat_0001"], 2)
    assert refusal is None
    assert schedules == [("seat_0001", "seat_0001")], "shared, never dropped"


def test_a_checked_context_with_nothing_selected_is_refused():
    schedules, refusal = _operation_context_schedules(_group(), [], 2)
    assert schedules == []
    assert refusal == "MISSING_REQUIRED_OPERATION_CONTEXT"


def test_several_contexts_short_of_the_applications_fail_closed():
    schedules, refusal = _operation_context_schedules(_group(), ["a", "b"], 3)
    assert schedules == []
    assert refusal == "TOO_FEW_OPERATION_CONTEXTS_FOR_APPLICATIONS"


def test_a_group_whose_preconditions_never_mention_its_context_needs_none():
    schedules, refusal = _operation_context_schedules(_group(context_relations=()), [], 2)
    assert refusal is None
    assert schedules == [(None, None)]


# ---------------------------------------------------------------------------
# The whole failure, end to end in the grounder
# ---------------------------------------------------------------------------


def _living_scene():
    graph_o = ObservedSceneGraph()
    for region, near in (("region_0001", "seat_0001"), ("region_0003", "seat_0002")):
        graph_o.add_node(ObservedNode(instance_id=region, entity_kind="REGION",
                                      canonical_category="side_table"))
        for seat in ("seat_0001", "seat_0002"):
            graph_o.add_relation(ObservedRelation(
                subject_id=region, predicate="NEAR_SEAT", object_id=seat,
                status="TRUE" if seat == near else "FALSE"))
    for slot in ("slot_1", "slot_2"):
        graph_o.add_node(ObservedNode(instance_id=slot, entity_kind="OBJECT",
                                      canonical_category="cup_saucer_set"))
        for region in ("region_0001", "region_0003"):
            graph_o.add_relation(ObservedRelation(
                subject_id=region, predicate="FITS_SET_ON", object_id=slot, status="TRUE"))
    for seat in ("seat_0001", "seat_0002"):
        graph_o.add_node(ObservedNode(instance_id=seat, entity_kind="FIXED_TARGET",
                                      canonical_category="seating_position"))
    return graph_o


def _living_graph(seat_count: int):
    return FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="two personal refreshment settings",
        nodes={
            "PERSONAL_CUP_SAUCER_REGION": FunctionalRole(
                name="PERSONAL_CUP_SAUCER_REGION", entity_kind="REGION", count=2,
                binding_policy="DISTINCT", semantic_categories=("side_table",)),
            "CUP_SAUCER_SET": FunctionalRole(
                name="CUP_SAUCER_SET", entity_kind="OBJECT", count=2,
                binding_policy="DISTINCT", semantic_categories=("cup_saucer_set",)),
            "SEATING_POSITION": FunctionalRole(
                name="SEATING_POSITION", entity_kind="FIXED_TARGET", count=seat_count,
                binding_policy="DISTINCT" if seat_count > 1 else "SHARED",
                semantic_categories=("seating_position",)),
        },
        operation_groups=(
            OperationGroup(
                id="personal", function="SUPPORT_DRINKWARE",
                tool_role="PERSONAL_CUP_SAUCER_REGION", target_role="CUP_SAUCER_SET",
                context_role="SEATING_POSITION", required_target_count=2,
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("FITS_SET_ON",), context_relations=("NEAR_SEAT",),
                capability_id="SUPPORT_DRINKWARE",
                physical_preconditions=(
                    ("PERSONAL_CUP_SAUCER_REGION", "FITS_SET_ON", "CUP_SAUCER_SET"),
                    ("PERSONAL_CUP_SAUCER_REGION", "NEAR_SEAT", "SEATING_POSITION"),
                ),
            ),
        ),
    )


def test_two_personal_placements_bind_two_distinct_seats():
    result = ground_graph(_living_graph(2), _living_scene(), {"search_exhausted": True})
    assert result.complete, result.unresolved_constraints
    seats = result.assignment["SEATING_POSITION"]
    assert sorted(seats) == ["seat_0001", "seat_0002"]
    bound = [binding["context"].get("SEATING_POSITION")
             for bindings in result.operation_bindings.values() for binding in bindings]
    assert sorted(bound) == ["seat_0001", "seat_0002"], "one seat per placement, both checked"


def test_two_personal_placements_cannot_share_one_seat():
    """The regression: one seat must not satisfy two personal placements.

    Before, the second application was evaluated with no context, so its
    proximity precondition went unchecked and the group reported satisfied.
    """
    result = ground_graph(_living_graph(1), _living_scene(), {"search_exhausted": True})
    assert not result.complete
    for bindings in (result.operation_bindings or {}).values():
        for binding in bindings:
            assert binding.get("context", {}).get("SEATING_POSITION"), (
                "no binding may be recorded without the seat its precondition checks")


def test_the_group_itself_refuses_a_binding_with_no_context():
    status, diagnostics, matching = _evaluate_operation_group(
        _group(), ["region_0001", "region_0003"], ["slot_1", "slot_2"],
        _living_scene(), selected_contexts=[], required_distinct_tools=2)
    assert status == "FALSE"
    assert matching == []
    assert any(row.get("status") == "MISSING_REQUIRED_OPERATION_CONTEXT" for row in diagnostics)


def test_one_personal_placement_with_one_seat_still_succeeds():
    graph = _living_graph(1)
    group = graph.operation_groups[0]
    single = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="one personal refreshment setting",
        nodes={**graph.nodes,
               "CUP_SAUCER_SET": FunctionalRole(
                   name="CUP_SAUCER_SET", entity_kind="OBJECT", count=1,
                   binding_policy="DISTINCT", semantic_categories=("cup_saucer_set",)),
               "PERSONAL_CUP_SAUCER_REGION": FunctionalRole(
                   name="PERSONAL_CUP_SAUCER_REGION", entity_kind="REGION", count=1,
                   binding_policy="DISTINCT", semantic_categories=("side_table",))},
        operation_groups=(replace(group, required_target_count=1),),
    )
    result = ground_graph(single, _living_scene(), {"search_exhausted": True})
    assert result.complete, result.unresolved_constraints
    binding = next(iter(result.operation_bindings.values()))[0]
    assert binding["context"]["SEATING_POSITION"] in {"seat_0001", "seat_0002"}
