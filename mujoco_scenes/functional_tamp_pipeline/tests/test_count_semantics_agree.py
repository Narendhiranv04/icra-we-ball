"""The prompt and the compiler must mean the same thing by required_count.

They did not.  The prompt asked for "how many physical instances the task
needs" while the compiler reads the field as how many *times* the task needs
the participant and lets binding_policy say how those come down to instances --
which is why a reusable coffee jar declared "2 DISTINCT" was read as a demand
for two jars, and a scene holding the one the task needs was reported short of
an object.

The compiler's reading is the one worth keeping: the model is being asked for
the task's meaning, and how many separate things a reusable function needs is a
physical fact about the runtime rather than something to ask the model for.  So
the prompt was changed to match the code, and this pins the pair together.
"""

from __future__ import annotations

import re

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import SYSTEM_PROMPT_V3
from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRole
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import _minimum_distinct_objects


def test_the_prompt_says_count_is_occurrences_not_instances():
    prompt = re.sub(r"\s+", " ", SYSTEM_PROMPT_V3)
    assert "required_count is how many times the task needs this participant" in prompt
    assert "how many physical instances the task needs" not in prompt
    # And it says which field carries the relation to physical things.
    assert "binding_policy is what says how those come down to physical things" in prompt


def test_operation_count_stays_separate_from_required_count():
    prompt = re.sub(r"\s+", " ", SYSTEM_PROMPT_V3)
    assert "operation_count is how many times the change happens" in prompt
    assert "independent of required_count" in prompt


def _role(count, policy):
    return {"id": "r", "entity_kind": "OBJECT", "function": "f", "description": "",
            "required_count": count, "binding_policy": policy,
            "candidate_categories": [], "required_properties": []}


def test_the_compiler_reads_the_field_the_way_the_prompt_now_describes():
    """Two applications, and how many things they need depends on the policy."""
    # A single instance can serve every application: one physical thing suffices,
    # and the minimum is stated because it differs from the count.
    assert _minimum_distinct_objects("kitchen", "coffee_source", _role(2, "REUSABLE")) == 1
    assert _minimum_distinct_objects("kitchen", "coffee_source", _role(2, "SHARED")) == 1
    # The model said each application needs its own, so the count *is* the
    # minimum; None means exactly that, and restating it would change the
    # compiled graph's fingerprint without changing what it means.
    assert _minimum_distinct_objects("kitchen", "coffee_container", _role(2, "DISTINCT")) is None
    node = FunctionalRole(name="coffee_container", entity_kind="OBJECT", count=2,
                          binding_policy="DISTINCT", min_count=None)
    assert node.minimum_count == 2
    reusable = FunctionalRole(name="coffee_source", entity_kind="OBJECT", count=2,
                              binding_policy="REUSABLE", min_count=1)
    assert reusable.minimum_count == 1


def test_one_application_states_no_separate_minimum():
    assert _minimum_distinct_objects("kitchen", "coffee_container", _role(1, "DISTINCT")) is None


def test_an_explicit_binding_cardinality_from_the_model_wins():
    role = _role(3, "DISTINCT")
    role["binding_cardinality"] = {"minimum_distinct_physical_objects": 1}
    assert _minimum_distinct_objects("kitchen", "coffee_container", role) == 1
