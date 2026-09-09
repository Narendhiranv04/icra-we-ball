"""Operation-induced abstract role requirements.

When the FM expresses a physical operation but enumerates its participants
imperfectly, the operation's own semantics still say which functional
participants must exist.  Those may be represented existentially, as typed
slots resolved later against the observed scene, but never bound here to a
concrete object or scene category, and never induced from an operation the FM
did not express.
"""

import pytest

from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    ABSTRACT_ROLE_PROVENANCE,
    abstract_slots_for_operation,
)


def _slots(domain, phrase):
    return {s.slot: s for s in abstract_slots_for_operation(domain, phrase)}


def test_expressed_fastening_induces_three_typed_participants():
    """A fastening needs an instrument, a joining component and a fixed target."""
    slots = _slots("workshop", "fasten the joint")
    assert set(slots) == {"source", "target", "anchor"}
    assert slots["source"].semantic_family == "INSTRUMENT"
    assert slots["target"].semantic_family == "COMPONENT"
    assert slots["anchor"].semantic_family == "FIXED_TARGET"


def test_synonymous_phrasing_induces_the_same_requirements():
    assert _slots("workshop", "tighten screw").keys() == _slots("workshop", "fasten the joint").keys()


def test_transfer_induces_a_source_and_a_receiver():
    slots = _slots("kitchen", "pour water into cup")
    assert slots["source"].semantic_family == "SOURCE"
    assert slots["target"].semantic_family == "DESTINATION"


def test_non_physical_directive_induces_nothing():
    """The runtime must never invent an operation in order to invent participants."""
    assert abstract_slots_for_operation("workshop", "identify compatible components") == ()
    assert abstract_slots_for_operation("kitchen", "decide which cup to use") == ()


def test_unrecognised_phrase_induces_nothing():
    assert abstract_slots_for_operation("kitchen", "perform the zzzzq procedure") == ()


def test_requirements_are_abstract_and_bind_no_concrete_object():
    """Slots carry role types for later resolution, never instances or objects."""
    for slot in abstract_slots_for_operation("workshop", "fasten the joint"):
        assert slot.provenance == ABSTRACT_ROLE_PROVENANCE
        assert slot.candidate_canonical_roles
        for candidate in slot.candidate_canonical_roles:
            # role type names, never observed-instance identifiers
            assert not candidate.startswith("object_")
            assert not candidate[:1].isdigit()


@pytest.mark.parametrize("domain,phrase", [
    ("workshop", "fasten the joint"),
    ("kitchen", "stir coffee"),
    ("living_room", "place refreshment setting"),
])
def test_every_induced_slot_is_typed(domain, phrase):
    slots = abstract_slots_for_operation(domain, phrase)
    assert slots
    assert all(s.semantic_family and s.slot in {"source", "target", "anchor"} for s in slots)
