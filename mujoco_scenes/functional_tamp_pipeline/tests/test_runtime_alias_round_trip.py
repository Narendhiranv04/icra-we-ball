"""The two semantic layers must not drift apart.

There are two independent mappings between free text and runtime roles:

1. an FM role's own wording is mapped to a canonical functional role, and
2. a detector's open-vocabulary label is checked for compatibility with the
   canonical role that role was bound to.

Nothing forces them to agree.  A category the runtime ontology declares
acceptable for a role could fail to be recognised by the compile-time typing,
or a label that types a role could then be rejected at grounding, and either
way a trial fails for a reason that is only a vocabulary gap.  These tests
close the loop in both directions, using nothing but the runtime ontology
configuration and the capability registry: no ground-truth object list and no
benchmark variant appears here.

Where a domain deliberately keeps a role open-world, the expected verdict is
UNKNOWN rather than TRUE.  What must never happen is FALSE -- a claim of
positive incompatibility -- for a label the runtime itself declares acceptable.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.grounding import (
    check_semantic_role_compatibility,
)
from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (
    get_all_system_role_semantic_categories,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode
from mujoco_scenes.functional_tamp_pipeline.semantic_typing import (
    build_role_type_hypotheses,
    canonical_role_family,
)
from mujoco_scenes.functional_tamp_pipeline.system_context_registry import (
    get_domain_selectable_roles,
    get_domain_system_fixed_anchors,
)


DOMAINS = ("kitchen", "living_room", "workshop")


def _declared_roles(domain: str) -> dict[str, tuple[str, ...]]:
    """Canonical roles the runtime declares acceptance categories for."""
    declared = get_all_system_role_semantic_categories(domain)
    bindable = set(get_domain_selectable_roles(domain)) | set(get_domain_system_fixed_anchors(domain))
    return {role: cats for role, cats in declared.items() if role in bindable and cats}


def _spellings(label: str) -> list[str]:
    """The ways a detector or a model might write the same label."""
    base = label.strip()
    spaced = base.replace("_", " ").replace("-", " ")
    return list(dict.fromkeys([
        base, base.upper(), base.title(),
        spaced, spaced.upper(), spaced.title(),
        spaced.replace(" ", "-"), spaced.replace(" ", "_"),
        f"  {spaced}  ",
    ]))


def _observed(label: str, kind: str = "OBJECT") -> ObservedNode:
    return ObservedNode(
        instance_id="observed_0001", entity_kind=kind, canonical_category=label,
        semantic_labels={"status": "SUPPORTED", "canonical_label": label},
    )


def test_every_bindable_role_declares_acceptance_categories():
    """A role with no declared vocabulary accepts anything, which hides drift."""
    missing = {
        (domain, role)
        for domain in DOMAINS
        for role in set(get_domain_selectable_roles(domain))
        if not get_all_system_role_semantic_categories(domain).get(role)
    }
    assert missing == set(), f"selectable roles without declared categories: {sorted(missing)}"


@pytest.mark.parametrize("domain", DOMAINS)
def test_declared_category_is_never_positively_incompatible(domain):
    """A label the runtime declares acceptable must never be judged FALSE.

    TRUE is the expected verdict; UNKNOWN is tolerated for a category the
    runtime keeps open-world.  FALSE would mean the grounding layer positively
    rejects a label the ontology layer offers, which is drift.
    """
    rejected = []
    for role, categories in _declared_roles(domain).items():
        kind = "REGION" if canonical_role_family(domain, role) == "SUPPORT" else (
            "FIXED_TARGET" if canonical_role_family(domain, role) == "FIXED_TARGET" else "OBJECT")
        for category in categories:
            for spelling in _spellings(category):
                status, _ = check_semantic_role_compatibility(_observed(spelling, kind), categories)
                if status == "FALSE":
                    rejected.append((role, category, spelling))
    assert rejected == [], f"{domain}: declared categories rejected as incompatible: {rejected}"


@pytest.mark.parametrize("domain", DOMAINS)
def test_declared_category_matches_in_every_ordinary_spelling(domain):
    """Case, spacing, underscores and hyphens must not change the verdict."""
    inconsistent = []
    for role, categories in _declared_roles(domain).items():
        for category in categories:
            verdicts = {
                check_semantic_role_compatibility(_observed(spelling), categories)[0]
                for spelling in _spellings(category)
            }
            if len(verdicts) > 1:
                inconsistent.append((role, category, sorted(verdicts)))
    assert inconsistent == [], f"{domain}: spelling changed the verdict: {inconsistent}"


@pytest.mark.parametrize("domain", DOMAINS)
def test_unrelated_label_is_not_silently_accepted(domain):
    """The open-world rule must not turn an unseen label into a match."""
    for role, categories in _declared_roles(domain).items():
        status, _ = check_semantic_role_compatibility(
            _observed("zzzq_unregistered_thing"), categories)
        assert status in {"FALSE", "UNKNOWN"}, (domain, role, status)


@pytest.mark.parametrize("domain", DOMAINS)
def test_declared_categories_type_a_role_onto_the_role_that_declares_them(domain):
    """A model role offering a role's own categories types onto that role.

    The wording is the runtime's own: each role's declared acceptance
    categories, plus its canonical name read as ordinary words.  If the
    compile-time typing cannot reach the role from those, the two layers
    disagree about what the role is.
    """
    drifted = []
    for role, categories in _declared_roles(domain).items():
        words = role.lower().replace("_", " ")
        document = {
            "functional_roles": [{
                "id": "probe",
                "entity_kind": (
                    "REGION" if canonical_role_family(domain, role) == "SUPPORT"
                    else "FIXED_TARGET" if canonical_role_family(domain, role) == "FIXED_TARGET"
                    else "OBJECT"
                ),
                "function": words,
                "description": "",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": list(categories),
                "required_properties": [],
            }],
            "functional_relations": [],
            "interaction_groups": [],
        }
        candidates = build_role_type_hypotheses(domain, document)["probe"].canonical_role_candidates
        if role not in candidates:
            drifted.append((role, list(categories), list(candidates)))
    assert drifted == [], f"{domain}: role wording no longer types onto its own role: {drifted}"


@pytest.mark.parametrize("domain", DOMAINS)
def test_capability_slots_only_name_roles_the_domain_declares(domain):
    """A capability slot naming a role the domain has no ontology for is drift."""
    from mujoco_scenes.functional_tamp_pipeline.operation_slot_completion import capability_slots
    from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
        get_robot_capabilities,
    )
    from mujoco_scenes.functional_tamp_pipeline.system_context_registry import (
        get_domain_planner_context_constants,
    )
    known = (
        set(get_domain_selectable_roles(domain))
        | set(get_domain_system_fixed_anchors(domain))
        | set(get_domain_planner_context_constants(domain))
    )
    for capability in get_robot_capabilities(domain):
        for slot, roles in capability_slots(domain, capability).items():
            assert set(roles) & known, (
                f"{domain}/{capability.capability_id} slot {slot!r} names no role this "
                f"domain declares: {sorted(roles)}"
            )


@pytest.mark.parametrize("domain", DOMAINS)
def test_capability_anchor_slots_agree_with_their_own_verifier(domain):
    """A capability may only offer anchors its own predicates can verify.

    Listing both the individual seat and the pair made them look
    interchangeable while the predicate that has to check the result accepts
    only one of them.
    """
    from mujoco_scenes.functional_tamp_pipeline.operation_slot_completion import (
        capability_anchor_roles,
    )
    from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
        get_robot_capabilities,
    )
    for capability in get_robot_capabilities(domain):
        admissible = capability_anchor_roles(domain, capability)
        assert set(admissible) == set(capability.allowed_anchor_roles), (
            f"{domain}/{capability.capability_id} lists anchors its predicates reject: "
            f"declared={list(capability.allowed_anchor_roles)} admissible={list(admissible)}"
        )
