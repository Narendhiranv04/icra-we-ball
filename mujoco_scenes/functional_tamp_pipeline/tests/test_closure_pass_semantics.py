"""Tests for the final deterministic closure pass.

Three generic readings are exercised here, each with the case it was added for
and the adversarial case it must refuse:

  * a binding policy read from the whole contract rather than one field;
  * a relation read against the form of a participant its own operation uses;
  * an unverifiable required property treated as the model's proposal unless the
    instruction asks for it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document, normalize_v3_live_document,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    compile_candidate_graph, instruction_expresses_property,
    resolve_binding_policy, separate_identity_evidence,
)

KITCHEN = (
    "Prepare and serve one coffee and one soup for each of two people. Make each "
    "coffee using coffee and water and stir it before serving. Serve each soup "
    "bowl with its own suitable eating utensil."
)
LIVING = (
    "Prepare the living room for two people to enjoy refreshments while watching "
    "television. Provide each person with their own refreshment setting nearby, "
    "and place the entertainment control where it is accessible to both people."
)
ARCHIVE = Path(__file__).resolve().parents[3] / "benchmark_reports/v3_qwen_distribution_3x32_20260910T053937"


def _role(rid, canonical, *, count=2, policy="DISTINCT", function="a thing"):
    return {"id": rid, "canonical_role": canonical, "function": function,
            "required_count": count, "binding_policy": policy,
            "entity_kind": "OBJECT", "candidate_categories": [],
            "required_properties": []}


# ---------------------------------------------------------------- binding policy

def test_reusable_source_used_twice_needs_one_object():
    role = _role("jar", "coffee_source", function="supply of coffee")
    document = {"functional_roles": [role], "functional_relations": [], "operation_pairings": []}
    policy, trace = resolve_binding_policy("kitchen", "jar", role, document)
    assert policy == "REUSABLE"
    assert trace["code"] == "GLOBAL_GRAPH_BINDING_POLICY_CONSISTENCY_OVERRIDE"
    assert trace["raw_binding_policy"] == "DISTINCT"


def test_runtime_one_per_application_role_is_never_re_read():
    """The reading never touches a role the runtime calls one-per-application."""
    role = _role("utensil", "soup_eating_utensil", function="tool to eat soup")
    document = {"functional_roles": [role], "functional_relations": [], "operation_pairings": []}
    policy, trace = resolve_binding_policy("kitchen", "utensil", role, document)
    assert policy == "DISTINCT"
    assert trace is None


@pytest.mark.parametrize("wording", [
    "each person uses their own stirrer",
    "a separate spoon for every cup",
    "one for each serving",
    "individual stirring implements",
    "per person stirring tool",
])
def test_separate_identity_wording_defeats_the_override(wording):
    role = _role("stirrer", "coffee_stirrer", function=wording)
    document = {"functional_roles": [role], "functional_relations": [], "operation_pairings": []}
    assert separate_identity_evidence("kitchen", "stirrer", role, document)
    policy, trace = resolve_binding_policy("kitchen", "stirrer", role, document)
    assert policy == "DISTINCT"
    assert trace is None


def test_repeated_use_alone_is_not_separate_identity():
    """"each" on its own is a distributive quantifier, not a claim about identity."""
    role = _role("stirrer", "coffee_stirrer", function="stir each coffee before serving")
    document = {"functional_roles": [role], "functional_relations": [],
                "operation_pairings": [{"id": "stir", "operation": "stir each coffee",
                                        "participant_roles": ["stirrer"], "operation_count": 2}]}
    policy, _ = resolve_binding_policy("kitchen", "stirrer", role, document)
    assert policy == "REUSABLE"


def test_wording_on_an_unrelated_operation_is_not_evidence():
    """Operation scope stays local: another operation's words are not about this role."""
    role = _role("stirrer", "coffee_stirrer", function="tool to mix coffee")
    document = {
        "functional_roles": [role, _role("utensil", "soup_eating_utensil")],
        "functional_relations": [],
        "operation_pairings": [
            {"id": "stir", "operation": "stir the coffee", "participant_roles": ["stirrer"],
             "operation_count": 2},
            {"id": "serve", "operation": "give each bowl its own separate utensil",
             "participant_roles": ["utensil"], "operation_count": 2},
        ],
    }
    assert not separate_identity_evidence("kitchen", "stirrer", role, document)
    assert resolve_binding_policy("kitchen", "stirrer", role, document)[0] == "REUSABLE"


def test_two_declarations_of_one_canonical_role_stay_distinct():
    left = _role("stirrer_a", "coffee_stirrer", count=2)
    right = _role("stirrer_b", "coffee_stirrer", count=2)
    document = {"functional_roles": [left, right], "functional_relations": [], "operation_pairings": []}
    assert separate_identity_evidence("kitchen", "stirrer_a", left, document)
    assert resolve_binding_policy("kitchen", "stirrer_a", left, document)[0] == "DISTINCT"


def test_single_use_role_is_left_alone():
    role = _role("jar", "coffee_source", count=1)
    document = {"functional_roles": [role], "functional_relations": [], "operation_pairings": []}
    assert resolve_binding_policy("kitchen", "jar", role, document) == ("DISTINCT", None)


# ------------------------------------------------- unverifiable required property

@pytest.mark.parametrize("prop", ["heat resistant", "edible safe", "must be sterile"])
def test_a_quality_the_instruction_never_asks_for_is_the_models_proposal(prop):
    assert not instruction_expresses_property(KITCHEN, prop)


def test_a_quality_the_instruction_does_ask_for_still_blocks():
    assert instruction_expresses_property("Use a heat resistant bowl", "heat resistant")
    assert instruction_expresses_property(KITCHEN, "safe for soup")


def test_model_proposed_property_does_not_block_a_binding():
    raw = json.loads(
        (ARCHIVE / "kitchen/K1/trial_01/raw_v3.json").read_text())
    document = json.loads(raw["content"]) if isinstance(raw.get("content"), str) else raw
    normalized, _ = normalize_v3_live_document(document, domain="kitchen", task_instruction=KITCHEN)
    canonical = convert_v3_to_canonical_document(normalized, domain="kitchen", task_instruction=KITCHEN)
    graph = compile_candidate_graph("kitchen", KITCHEN, canonical)
    blocking = [row["property"] for row in graph.metadata["unverified_required_properties"]]
    proposed = [row["property"] for row in graph.metadata["model_proposed_unverifiable_properties"]]
    assert "heat resistant" not in blocking
    assert "heat resistant" in proposed


# ------------------------------------------------------- relation form projection

@pytest.mark.skipif(not ARCHIVE.exists(), reason="frozen archive not present")
def test_relation_is_read_against_the_form_its_own_operation_uses():
    """One "surface" role serving both placements.

    The control's placement is compiled against the shared support, so the
    relation the model wrote about the control and the surface must be checked
    against that same shared support -- not against the personal support the raw
    role kept for the refreshment placement.
    """
    raw = json.loads((ARCHIVE / "living_room/L4/trial_02/raw_v3.json").read_text())
    document = json.loads(raw["content"]) if isinstance(raw.get("content"), str) else raw
    normalized, _ = normalize_v3_live_document(document, domain="living_room", task_instruction=LIVING)
    canonical = convert_v3_to_canonical_document(normalized, domain="living_room", task_instruction=LIVING)

    rerouted = [
        row for row in canonical["functional_constraint_interpretation"]
        if row.get("code") == "RELATION_READ_AGAINST_THE_FORM_ITS_OWN_OPERATION_USES"
    ]
    assert rerouted, "the control's support relation should follow the control's own operation"
    assert all(row["form_role"].startswith("fm_form__surface__") for row in rerouted)

    graph = compile_candidate_graph("living_room", LIVING, canonical)
    assert graph.metadata["online_executable_contract_complete"] is True
    sources = {group.tool_role for group in graph.operation_groups}
    assert {"PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION"} <= sources


def test_binding_policy_resolution_cannot_see_the_scene():
    """Structural guarantee, not a behavioural sample.

    The whole point of resolving the policy while compiling is that the observed
    scene is not available yet, so a scene short of objects can never be the
    reason a stated requirement is re-read.  ``resolve_binding_policy`` takes the
    domain, the role and the document and nothing else, and ``compile_candidate_graph``
    -- which calls it -- takes no observed graph at all.
    """
    import inspect

    accepted = set(inspect.signature(resolve_binding_policy).parameters)
    assert accepted == {"domain", "role_id", "role", "document"}
    compile_parameters = set(inspect.signature(compile_candidate_graph).parameters)
    assert compile_parameters == {"domain", "task", "raw"}
    for name in ("graph_o", "observed", "scene", "candidates", "inventory"):
        assert name not in accepted and name not in compile_parameters


def test_resolution_is_identical_whatever_the_scene_would_hold():
    """The same contract resolves the same way twice, with nothing else supplied."""
    role = _role("stirrer", "coffee_stirrer", function="tool to mix the coffee")
    document = {"functional_roles": [role], "functional_relations": [],
                "operation_pairings": [{"id": "stir", "operation": "stir the coffee",
                                        "participant_roles": ["stirrer", "cup"],
                                        "operation_count": 2}]}
    first = resolve_binding_policy("kitchen", "stirrer", role, document)
    second = resolve_binding_policy("kitchen", "stirrer", role, document)
    assert first[0] == second[0] == "REUSABLE"
