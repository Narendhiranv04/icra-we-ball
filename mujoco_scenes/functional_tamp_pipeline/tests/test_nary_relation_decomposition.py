"""Relations over more than two participants are represented, not rejected.

The wire contract admits two to four participants and the prompt asks the model
to name every member a quantified phrase covers, so n-ary relations are expected.
Previously any such relation outside one declared context-set family aborted the
whole contract as a task-specification failure, while a *binary* relation the
runtime could not interpret was carried through harmlessly. Arity alone decided
whether the same uninterpretability was fatal.
"""



from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import convert_v3_to_canonical_document

INSTRUCTION = "Prepare and serve one coffee for each of two people."


def _role(rid, function, categories=()):
    return {"id": rid, "entity_kind": "OBJECT", "function": function,
            "required_count": 1, "binding_policy": "DISTINCT",
            "candidate_categories": list(categories), "required_properties": []}


def _document(roles, relations=(), operations=()):
    return {
        "schema_version": 3, "status": "SUPPORTED", "task_summary": "t",
        "task_contract": {"functional_roles": list(roles),
                          "functional_relations": list(relations),
                          "operation_pairings": list(operations)},
        "observation_guidance": {"visible_candidates_per_role": {},
                                 "inspectable_regions": [], "inspection_order": []},
        "unsupported_reason": "",
    }


ROLES = [
    _role("beverage", "the prepared coffee", ["coffee"]),
    _role("grounds", "solid ingredient for coffee", ["coffee_jar"]),
    _role("liquid", "liquid ingredient for coffee", ["kettle"]),
]


def _convert(relations):
    return convert_v3_to_canonical_document(
        _document(ROLES, relations=relations), domain="kitchen",
        task_instruction=INSTRUCTION)


def test_uninterpretable_ternary_relation_does_not_abort_the_contract():
    """The same predicate is harmless with two participants; three must not be fatal."""
    canonical = _convert([{
        "id": "ingredients", "relation": "made_from",
        "participant_roles": ["beverage", "grounds", "liquid"], "required": True,
    }])
    assert canonical is not None


def test_binary_and_nary_uninterpretable_relations_agree():
    binary = _convert([{"id": "r", "relation": "made_from",
                        "participant_roles": ["beverage", "grounds"], "required": True}])
    ternary = _convert([{"id": "r", "relation": "made_from",
                         "participant_roles": ["beverage", "grounds", "liquid"],
                         "required": True}])
    assert (binary is not None) == (ternary is not None)


def test_every_nary_participant_survives_into_the_accounting():
    """Decomposition must not quietly discard a named participant."""
    canonical = _convert([{
        "id": "ingredients", "relation": "made_from",
        "participant_roles": ["beverage", "grounds", "liquid"], "required": True,
    }])
    accounted = {row["raw_id"] for row in canonical["fm_semantic_accounting"]}
    assert {"beverage", "grounds", "liquid", "ingredients"} <= accounted


def test_decomposition_emits_one_edge_per_pair_and_keeps_the_predicate():
    """A predicate the runtime cannot read at all is decomposed over every pair.

    With no reading to constrain it, no pair can be preferred, so none is
    dropped and the phrase travels with each edge for the stages downstream.
    """
    canonical = _convert([{
        "id": "ingredients", "relation": "belongs_with",
        "participant_roles": ["beverage", "grounds", "liquid"], "required": True,
    }])
    edges = [r for r in canonical["functional_relations"]
             if r["id"].startswith("ingredients__")]
    pairs = {frozenset((e["subject_role"], e["object_role"])) for e in edges}
    assert pairs == {
        frozenset(("beverage", "grounds")),
        frozenset(("beverage", "liquid")),
        frozenset(("grounds", "liquid")),
    }
    assert all(e["relation"] == "belongs_with" for e in edges)


def test_a_readable_predicate_only_keeps_the_pairs_it_can_be_read_over():
    """"The beverage is made from grounds and water" says nothing about the two.

    Once the runtime can read the phrase, the readings are what decide which
    pairs survive: each ingredient supplies the carrier, and neither supplies
    the other.  Emitting that third edge would invent a claim the model never
    made.
    """
    canonical = _convert([{
        "id": "ingredients", "relation": "made_from",
        "participant_roles": ["beverage", "grounds", "liquid"], "required": True,
    }])
    edges = [r for r in canonical["functional_relations"]
             if r["id"].startswith("ingredients__")]
    pairs = {frozenset((e["subject_role"], e["object_role"])) for e in edges}
    assert frozenset(("grounds", "liquid")) not in pairs
    assert pairs == {frozenset(("beverage", "grounds")), frozenset(("beverage", "liquid"))}


def test_genuine_contradiction_path_is_still_reachable():
    """A recognised predicate with no legal pair yields nothing, which the
    caller turns into a task-specification failure. Arity alone never does."""
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import _decompose_nary_relation

    relation = {"id": "x", "relation": "contained_in",
                "participant_roles": ["a", "b", "c"], "required": True}
    # No hypotheses for any participant: no pair can be given a legal reading.
    assert _decompose_nary_relation("kitchen", relation, {}) == []
