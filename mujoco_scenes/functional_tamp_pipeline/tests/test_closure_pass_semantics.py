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
    assert accepted == {"domain", "role_id", "role", "document", "canonical_by_raw_id"}
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


# ------------------------------------------- resolved policy reaches operations

def test_operation_reuse_follows_the_resolved_role_policy():
    """The Kitchen failure this was written for.

    The stirrer resolves to reusable, so one spoon serves both stirrings.  The
    stirring operation was created before that resolution and kept asking for
    one spoon per application; the grounder believed the operation, took two of
    the scene's three spoons, and the two soup utensils the task does want
    severally were reported undiscovered.
    """
    raw = json.loads((ARCHIVE / "kitchen/K1/trial_01/raw_v3.json").read_text())
    document = json.loads(raw["content"]) if isinstance(raw.get("content"), str) else raw
    normalized, _ = normalize_v3_live_document(document, domain="kitchen", task_instruction=KITCHEN)
    canonical = convert_v3_to_canonical_document(normalized, domain="kitchen", task_instruction=KITCHEN)
    graph = compile_candidate_graph("kitchen", KITCHEN, canonical)

    by_tool = {group.tool_role: group for group in graph.operation_groups}
    stirring = by_tool["coffee_stirrer"]
    assert graph.nodes["coffee_stirrer"].binding_policy == "REUSABLE"
    assert graph.nodes["coffee_stirrer"].minimum_count == 1
    assert stirring.usage_policy == "SEQUENTIAL_REUSE_ALLOWED"
    assert stirring.distinct_within_group is False

    # ... and the utensil the instruction says each bowl gets its own of is
    # untouched, so this cannot be mistaken for a blanket relaxation.
    serving = by_tool["soup_eating_utensil"]
    assert graph.nodes["soup_eating_utensil"].binding_policy == "DISTINCT"
    assert serving.usage_policy == "DEDICATED_PER_TARGET"
    assert serving.distinct_within_group is True

    codes = {row["code"] for row
             in graph.metadata["canonicalization_trace"]["role_operation_reconciliations"]}
    assert "OPERATION_REUSE_FOLLOWS_RESOLVED_ROLE_POLICY" in codes


def test_a_distinct_source_keeps_its_operation_dedicated():
    """The adversarial direction: nothing relaxes an operation over a role the
    runtime says cannot be reused."""
    raw = json.loads((ARCHIVE / "kitchen/K1/trial_01/raw_v3.json").read_text())
    document = json.loads(raw["content"]) if isinstance(raw.get("content"), str) else raw
    normalized, _ = normalize_v3_live_document(document, domain="kitchen", task_instruction=KITCHEN)
    canonical = convert_v3_to_canonical_document(normalized, domain="kitchen", task_instruction=KITCHEN)
    graph = compile_candidate_graph("kitchen", KITCHEN, canonical)
    for group in graph.operation_groups:
        if graph.nodes[group.tool_role].binding_policy == "DISTINCT":
            assert group.usage_policy == "DEDICATED_PER_TARGET"


# --------------------------------------- aggregate counts partition by function

def test_aggregate_participant_count_is_partitioned_across_forms():
    """One "spoon" role counted four times, for two stirrings and two servings.

    Each projected form takes the count of the operations that induced it.  The
    form left on the raw role used to keep the aggregate four and was then
    reported short of objects for a demand the contract never made.
    """
    raw = json.loads((ARCHIVE / "kitchen/K7/trial_01/raw_v3.json").read_text())
    document = json.loads(raw["content"]) if isinstance(raw.get("content"), str) else raw
    normalized, _ = normalize_v3_live_document(document, domain="kitchen", task_instruction=KITCHEN)
    canonical = convert_v3_to_canonical_document(normalized, domain="kitchen", task_instruction=KITCHEN)
    partitions = [row for row in canonical.get("functional_constraint_interpretation", ())
                  if row.get("code") == "AGGREGATE_PARTICIPANT_COUNT_PARTITIONED_BY_FUNCTIONAL_FORM"]
    assert partitions, "the four-times spoon should have been partitioned"
    row = partitions[0]
    assert row["declared_required_count"] == 4
    assert row["resolved_required_count"] < 4
    graph = compile_candidate_graph("kitchen", KITCHEN, canonical)
    for name in ("coffee_stirrer", "soup_eating_utensil"):
        if name in graph.nodes:
            assert graph.nodes[name].minimum_count <= 2


# ----------------------------------------------- symmetric binding resolution

def test_two_declarations_of_one_canonical_role_are_seen_through_the_mapping():
    """Raw roles carry no canonical name, so the evidence needs the mapping.

    Without it the check compared a missing field with itself, found nothing,
    and two declarations of one role looked like one.
    """
    left = _role("cup_one", None, count=1, function="cup for the first person")
    right = _role("cup_two", None, count=1, function="cup for the second person")
    document = {"functional_roles": [left, right], "functional_relations": [],
                "operation_pairings": []}
    mapping = {"cup_one": "coffee_container", "cup_two": "coffee_container"}
    evidence = separate_identity_evidence(
        "kitchen", "cup_one", {**left, "canonical_role": "coffee_container"},
        document, mapping)
    assert any(row["source"] == "TWO_DECLARATIONS_OF_ONE_CANONICAL_ROLE" for row in evidence)


def test_a_permissive_policy_is_tightened_when_the_contract_asks_severally():
    """The other correction direction, which costs score and buys soundness.

    The model calls the eating utensil SHARED while saying each bowl gets its
    own; the runtime says one utensil cannot serve two servings.  Honouring
    SHARED would satisfy a two-utensil requirement with one object.
    """
    role = _role("utensil", "soup_eating_utensil", count=2, policy="SHARED",
                 function="each bowl gets its own eating utensil")
    document = {"functional_roles": [role], "functional_relations": [],
                "operation_pairings": []}
    resolved, trace = resolve_binding_policy("kitchen", "utensil", role, document)
    assert resolved == "DISTINCT"
    assert trace["direction"] == "TIGHTENED"
    assert trace["separate_identity_evidence"]


def test_a_permissive_policy_over_a_reusable_role_is_left_alone():
    """Tightening applies only where the runtime says reuse is impossible."""
    role = _role("jar", "coffee_source", count=2, policy="REUSABLE",
                 function="its own supply of coffee for each cup")
    document = {"functional_roles": [role], "functional_relations": [],
                "operation_pairings": []}
    resolved, trace = resolve_binding_policy("kitchen", "jar", role, document)
    assert resolved == "REUSABLE" and trace is None


# --------------------------------------------- a role name is not an action

@pytest.mark.parametrize("phrase,participants,declared,expected", [
    # The case this was written for: the component's name is not a fastening.
    ("The robot arm opens and inspects storage containers to find the "
     "compatible fastening component.", ["robot_arm", "storage_container"],
     ["robot_arm", "storage_container", "fastening_component", "workbench"], True),
    ("search for fastening tool", ["fastening_tool"], ["fastening_tool"], True),
    ("Inspect the storage to locate the mounting bracket", ["storage"],
     ["storage", "mounting_bracket"], True),
    # ... and the adversarial half: a real act coordinated with a search stays.
    ("Search the drawers and then fasten the screw into the joint",
     ["drawer", "screw"], ["drawer", "screw", "joint"], False),
    ("Open the cabinet and place the tool on the bench", ["cabinet", "tool"],
     ["cabinet", "tool", "bench"], False),
    ("Locate the parts and perform the fastening", ["part"], ["part"], False),
    ("find the screw and then drive it into the hole", ["screw", "hole"],
     ["screw", "hole"], False),
])
def test_a_declared_role_name_is_never_read_as_an_action(
        phrase, participants, declared, expected):
    """Every declared role's name is stripped, not only this operation's own.

    A phrase may name another participant descriptively.  Because that one is
    not listed in this operation it was never stripped, so "fastening" inside
    "the compatible fastening component" matched as an act coordinated with the
    search, and an acquisition directive the runtime already performs was
    compiled as a physical operation it could not seat.
    """
    from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
        is_non_physical_operation_phrase,
    )
    assert is_non_physical_operation_phrase(phrase, participants, declared) is expected


def test_an_operation_that_only_moves_the_robot_is_not_a_task_requirement():
    """"The robot arm moves to a safe resting position on the workbench."

    The executor and a place are the only participants, so no task object ends
    up anywhere.  Read as a task operation it had one participant left once the
    executor was removed, could state no relation, and blocked a contract whose
    goals were otherwise complete.
    """
    raw = json.loads((ARCHIVE / "workshop/W7/trial_02/raw_v3.json").read_text())
    document = json.loads(raw["content"]) if isinstance(raw.get("content"), str) else raw
    workshop = ("Identify the compatible components required to complete the fastening at "
                "the marked workbench location, complete the fastening, and leave any "
                "reusable equipment used for the task safely on the workbench.")
    normalized, trace = normalize_v3_live_document(
        document, domain="workshop", task_instruction=workshop)
    notes = trace if isinstance(trace, list) else (trace or {}).get("repairs", []) or []
    assert any(row.get("code") == "OPERATION_STATES_WHERE_THE_ROBOT_ITSELF_ENDS_UP"
               for row in notes)
    canonical = convert_v3_to_canonical_document(
        normalized, domain="workshop", task_instruction=workshop)
    graph = compile_candidate_graph("workshop", workshop, canonical)
    assert graph.metadata["online_executable_contract_complete"] is True
