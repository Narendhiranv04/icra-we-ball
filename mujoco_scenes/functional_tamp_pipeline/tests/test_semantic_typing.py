from mujoco_scenes.functional_tamp_pipeline.semantic_typing import (
    build_role_type_hypotheses,
    operation_participant_satisfiable,
)


def _role(role_id, function, kind="OBJECT"):
    return {
        "id": role_id, "function": function, "description": "",
        "entity_kind": kind, "required_count": 1,
        "binding_policy": "REUSABLE", "candidate_categories": [],
        "required_properties": [],
    }


def test_wrong_positive_alias_is_overridden_by_transfer_structure():
    doc = {
        "functional_roles": [
            _role("source", "coffee ingredient source stored in a canister"),
            _role("receiver", "receiving vessel for prepared coffee"),
        ],
        "functional_relations": [],
        "interaction_groups": [{
            "id": "transfer", "function": "transfer material into container",
            "tool_role": "source", "target_role": "receiver",
        }],
    }
    wrong_alias = lambda _d, role, _doc: (
        "coffee_container" if role["id"] == "source" else "coffee_container", "TEST_WEAK_ALIAS"
    )
    hypotheses = build_role_type_hypotheses("kitchen", doc, weak_mapper=wrong_alias)
    assert hypotheses["source"].canonical_role_candidates == ("coffee_source",)
    assert hypotheses["receiver"].canonical_role_candidates == ("coffee_container",)
    assert hypotheses["source"].status == "STRUCTURAL_OVERRIDE_OF_WEAK_FUNCTION_ALIAS"


def test_explicit_receiving_container_cannot_be_transfer_source():
    doc = {
        "functional_roles": [
            _role("receiver", "receiving destination container for prepared coffee"),
            _role("other", "receiving destination container for prepared coffee"),
        ],
        "functional_relations": [],
        "interaction_groups": [{
            "id": "transfer", "function": "transfer material into container",
            "tool_role": "receiver", "target_role": "other",
        }],
    }
    hypotheses = build_role_type_hypotheses("kitchen", doc)
    assert not operation_participant_satisfiable("kitchen", doc["interaction_groups"][0], hypotheses)


def test_zero_relation_transfer_capability_still_checks_participants():
    doc = {
        "functional_roles": [_role("tool", "reusable mixing tool"), _role("cup", "receiving coffee vessel")],
        "functional_relations": [],
        "interaction_groups": [{
            "id": "transfer", "function": "transfer content to container",
            "tool_role": "tool", "target_role": "cup",
        }],
    }
    hypotheses = build_role_type_hypotheses("kitchen", doc)
    assert not operation_participant_satisfiable("kitchen", doc["interaction_groups"][0], hypotheses)


def test_living_side_table_near_seat_is_support_not_seating():
    doc = {"functional_roles": [_role("table", "side table near a seating position")],
           "functional_relations": [], "interaction_groups": []}
    hypothesis = build_role_type_hypotheses("living_room", doc)["table"]
    assert "PERSONAL_CUP_SAUCER_REGION" in hypothesis.canonical_role_candidates
    assert "SEATING_POSITION" not in hypothesis.canonical_role_candidates
    assert "SEATING_PAIR" not in hypothesis.canonical_role_candidates


def test_workshop_receiving_assembly_is_repair_target_not_fastener():
    doc = {"functional_roles": [_role("assembly", "fixed assembly receiving installed component")],
           "functional_relations": [], "interaction_groups": []}
    hypothesis = build_role_type_hypotheses("workshop", doc)["assembly"]
    assert hypothesis.canonical_role_candidates == ("repair_target",)


def test_coherent_ambiguous_role_is_not_task_spec_contradiction():
    doc = {"functional_roles": [_role("implement", "long implement for preparation")],
           "functional_relations": [], "interaction_groups": []}
    hypothesis = build_role_type_hypotheses("kitchen", doc)["implement"]
    assert set(hypothesis.canonical_role_candidates) == {"coffee_stirrer", "soup_eating_utensil"}
    assert hypothesis.status == "UNCONSTRAINED_ROLE_TYPE"
