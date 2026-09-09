"""Task requirement provenance: instruction clauses vs visual observation.

One FM call sees the instruction and the images together, so the returned
contract does not say which produced any given element.  An instruction clause
may create a task requirement; a visual observation may only propose a
candidate, a current location, or somewhere worth searching.  These tests pin
the evidence the runtime records for that distinction.
"""

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    _provenance_terms,
    classify_requirement_provenance,
    convert_v3_to_canonical_document,
)

KITCHEN_INSTRUCTION = (
    "Prepare and serve one coffee and one soup for each of two people. "
    "Make each coffee using coffee and water and stir it before serving."
)


def _role(rid, function, categories=(), kind="OBJECT", count=1):
    return {
        "id": rid, "entity_kind": kind, "function": function,
        "required_count": count, "binding_policy": "DISTINCT",
        "candidate_categories": list(categories), "required_properties": [],
    }


def _document(roles, relations=(), operations=()):
    return {
        "schema_version": 3, "status": "SUPPORTED", "task_summary": "t",
        "task_contract": {
            "functional_roles": list(roles),
            "functional_relations": list(relations),
            "operation_pairings": list(operations),
        },
        "observation_guidance": {
            "visible_candidates_per_role": {}, "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }


def test_instruction_wording_yields_clause_support():
    terms = _provenance_terms(KITCHEN_INSTRUCTION)
    assert classify_requirement_provenance(
        _provenance_terms("a container that holds the prepared coffee"),
        instruction_terms=terms, participates_in_expressed_operation=False,
    ) == "INSTRUCTION_CLAUSE_SUPPORT"


def test_observation_only_role_is_not_credited_to_the_instruction():
    """A participant the instruction never mentions is observation evidence."""
    terms = _provenance_terms(KITCHEN_INSTRUCTION)
    assert classify_requirement_provenance(
        _provenance_terms("a closed cabinet seen at the back of the room"),
        instruction_terms=terms, participates_in_expressed_operation=False,
    ) == "OBSERVATION_ONLY"


def test_operation_participation_derives_requirement():
    """No instruction wording, but an expressed operation needs the participant."""
    terms = _provenance_terms(KITCHEN_INSTRUCTION)
    assert classify_requirement_provenance(
        _provenance_terms("a flat destination surface"),
        instruction_terms=terms, participates_in_expressed_operation=True,
    ) == "DERIVED_FROM_EXPLICIT_OPERATION"


def test_every_role_and_relation_carries_requirement_provenance():
    doc = _document(
        roles=[
            _role("cup", "a container for the served coffee", ["cup"]),
            _role("shelf", "a shelf noticed along the far wall", ["shelf"], kind="REGION"),
        ],
        relations=[{"id": "r1", "relation": "coffee fits in cup",
                    "participant_roles": ["cup", "shelf"], "required": True}],
    )
    canonical = convert_v3_to_canonical_document(
        doc, domain="kitchen", task_instruction=KITCHEN_INSTRUCTION)
    rows = canonical["fm_semantic_accounting"]
    assert rows, "accounting must not be empty"
    assert all("requirement_provenance" in row for row in rows)

    by_id = {row["raw_id"]: row["requirement_provenance"] for row in rows}
    assert by_id["cup"] == "INSTRUCTION_CLAUSE_SUPPORT"
    assert by_id["shelf"] == "OBSERVATION_ONLY"


def test_provenance_is_recorded_not_enforced():
    """Observation-only elements are still represented, never silently deleted."""
    doc = _document(roles=[
        _role("cup", "a container for the served coffee", ["cup"]),
        _role("shelf", "a shelf noticed along the far wall", ["shelf"], kind="REGION"),
    ])
    canonical = convert_v3_to_canonical_document(
        doc, domain="kitchen", task_instruction=KITCHEN_INSTRUCTION)
    accounted = {row["raw_id"] for row in canonical["fm_semantic_accounting"]
                 if row["element_kind"] == "role"}
    assert accounted == {"cup", "shelf"}


def test_every_raw_element_is_accounted_for():
    """No raw role, relation or operation may vanish without a recorded disposition."""
    doc = _document(
        roles=[
            _role("cup", "a container for the served coffee", ["cup"]),
            _role("spoon", "a tool for stirring the coffee", ["spoon"]),
            _role("odd", "an unclassifiable curiosity", ["curio"]),
        ],
        relations=[{"id": "r1", "relation": "spoon fits in cup",
                    "participant_roles": ["spoon", "cup"], "required": True}],
        operations=[{"id": "o1", "operation": "stir coffee",
                     "participant_roles": ["spoon", "cup"], "operation_count": 1}],
    )
    canonical = convert_v3_to_canonical_document(
        doc, domain="kitchen", task_instruction=KITCHEN_INSTRUCTION)
    accounted = {(row["element_kind"], row["raw_id"])
                 for row in canonical["fm_semantic_accounting"]}
    expected = {("role", "cup"), ("role", "spoon"), ("role", "odd"),
                ("relation", "r1"), ("operation", "o1")}
    assert expected <= accounted, f"unaccounted: {expected - accounted}"
