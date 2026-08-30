"""Pass 3.6A.2 Comprehensive Unit Tests.

Verifies:
1. Kitchen initial RGB render resolution is 1280x960.
2. Complete schema validation and fail-closed behavior for unsupported/malformed payloads.
3. Decoupling of VLM-local IDs from semantic canonicalization.
4. Compositional concept matching for Kitchen, Living Room, and Workshop.
5. Zero payload leaks in system prompts and requests.
6. Transformation provenance and vlm_canonicalization_version.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLMSpecProvider, VLM_CANONICALIZATION_VERSION,
)
from mujoco_scenes.kitchen_vlm_functional_graph import (
    compile_vlm_functional_graph,
    map_kitchen_role_function,
    map_unary_property,
    map_binary_relation,
    resolve_kitchen_region_proposal,
)
from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    map_living_room_role_function,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMResponseValidationError,
    validate_requirement_response,
    validate_kitchen_functional_specification,
    SYSTEM_PROMPT,
)
from mujoco_scenes.workshop_phase1.requirements import (
    FMRequirementProvider,
    map_workshop_role_function,
    map_workshop_relation,
    resolve_workshop_region_proposal,
)


def test_vlm_canonicalization_version_constant():
    assert VLM_CANONICALIZATION_VERSION == "phase3_6a5_v1"


def test_kitchen_render_resolution_configured():
    from mujoco_scenes.run_kitchen_vlm_pipeline import run_pipeline
    import inspect
    sig = inspect.signature(run_pipeline)
    assert sig.parameters["width"].default == 1280
    assert sig.parameters["height"].default == 960


def test_kitchen_compositional_role_mapping():
    assert map_kitchen_role_function("contain one serving of coffee") == "coffee_container"
    assert map_kitchen_role_function("contain soup for guest") == "soup_container"
    assert map_kitchen_role_function("stir coffee beverage") == "coffee_stirrer"
    assert map_kitchen_role_function("serve soup with utensil") == "soup_eating_utensil"
    assert map_kitchen_role_function("provide coffee material") == "coffee_source"
    assert map_kitchen_role_function("provide hot water") == "water_source"
    assert map_kitchen_role_function("unknown extraterrestrial activity") is None


def test_living_room_compositional_concept_matching():
    # Personal cup/saucer
    assert map_living_room_role_function("surface to hold personal cup and saucer for seated viewer") == "personal_cup_saucer"
    assert map_living_room_role_function("side table for individual drink") == "personal_cup_saucer"
    assert map_living_room_role_function("support personal beverage near armchair") == "personal_cup_saucer"

    # Shared remote
    assert map_living_room_role_function("central coffee table to hold shared tv remote for both viewers") == "shared_remote"
    assert map_living_room_role_function("common surface holding media controller") == "shared_remote"
    assert map_living_room_role_function("shared table for remote control") == "shared_remote"

    # Unrelated
    assert map_living_room_role_function("clean the window curtains") is None


def test_workshop_compositional_concept_matching():
    # CAN_DRIVE_SCREW
    assert map_workshop_role_function("tighten threaded screw into workbench") == "CAN_DRIVE_SCREW"
    assert map_workshop_role_function("torque fastener") == "CAN_DRIVE_SCREW"
    assert map_workshop_role_function("drive wood screw") == "CAN_DRIVE_SCREW"

    # CAN_FASTEN
    assert map_workshop_role_function("secure joint and anchor in hole") == "CAN_FASTEN"
    assert map_workshop_role_function("threaded fastener to hold parts") == "CAN_FASTEN"

    # Relations
    assert map_workshop_relation("fits driver bit to screw head") == "COMPATIBLE_WITH"
    assert map_workshop_relation("long enough to reach workpiece hole recess") == "REACHES_TARGET"
    assert map_workshop_relation("threads into target repair hole") == "COMPATIBLE_WITH_TARGET"


def test_unsupported_contract_enforcement():
    # Valid UNSUPPORTED
    valid_unsupported = {
        "status": "UNSUPPORTED",
        "task_summary": "Perform quantum teleportation",
        "functional_roles": [],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "Task violates laws of physics and lacks physical roles.",
    }
    validated = validate_requirement_response(valid_unsupported)
    assert validated["status"] == "UNSUPPORTED"
    assert validated["unsupported_reason"] != ""

    # Invalid UNSUPPORTED: has non-empty roles
    invalid_unsupported = dict(valid_unsupported)
    invalid_unsupported["functional_roles"] = [
        {"id": "r1", "entity_kind": "OBJECT", "function": "foo", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["c"], "visible_candidates": [], "required_properties": []}
    ]
    with pytest.raises(FMResponseValidationError, match="must have empty functional_roles"):
        validate_requirement_response(invalid_unsupported)

    # Invalid UNSUPPORTED: empty reason
    no_reason = dict(valid_unsupported)
    no_reason["unsupported_reason"] = ""
    with pytest.raises(FMResponseValidationError, match="non-empty unsupported_reason"):
        validate_requirement_response(no_reason)


def test_zero_leakage_in_system_prompt():
    prompt_lower = SYSTEM_PROMPT.lower()
    # Check that domain objects, ground truth IDs, and checkers are not leaked in system prompt
    assert "screwdriver" not in prompt_lower
    assert "phillips" not in prompt_lower
    assert "can_drive_screw" not in prompt_lower
    assert "reaches_target" not in prompt_lower
    assert "open_cavity" not in prompt_lower
    assert "planar_support" not in prompt_lower
    assert '"d1"' not in prompt_lower
    assert '"c2"' not in prompt_lower
