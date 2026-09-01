"""Tests for image-conditioned Workshop Qwen requirement generation."""

from __future__ import annotations

import base64
import json

import pytest

from mujoco_scenes.run_workshop_vlm_requirements import build_result
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMBackendNotConfiguredError,
    FMResponseValidationError,
    validate_requirement_response,
)
from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
from mujoco_scenes.workshop_phase1.types import RequirementSource


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture
def observation_image(tmp_path):
    path = tmp_path / "workshop_initial.png"
    path.write_bytes(PNG_1X1)
    return path


def candidate(label: str, description: str) -> dict:
    return {
        "label": label,
        "visual_description": description,
        "suitability_reason": "Visually plausible but not geometrically verified.",
    }


def natural_decomposition() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Choose a compatible driver and threaded fastener.",
        "functional_roles": [
            {
                "id": "rotating_tool",
                "entity_kind": "OBJECT",
                "function": "tighten a screw",
                "description": "A device that rotates the screw into the repair joint.",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [
                    candidate("Phillips screwdriver", "hand tool on the workbench")
                ],
                "required_properties": [],
            },
            {
                "id": "threaded_joiner",
                "entity_kind": "OBJECT",
                "function": "secure the joint",
                "description": "A threaded component that holds the repair joint.",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "visible_candidates": [
                    candidate("Phillips screw", "small threaded fastener in the tray")
                ],
                "required_properties": [],
            },
            {
                "id": "repair_hole",
                "entity_kind": "FIXED_TARGET",
                "function": "target repair hole on workpiece",
                "description": "hole in the frame joint to receive screw",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["workbench_hole"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "rotating_tool",
                "relation": "tip must fit the screw head and transmit torque",
                "object_role": "threaded_joiner",
            },
            {
                "subject_role": "rotating_tool",
                "relation": "must reach the workpiece hole recess",
                "object_role": "repair_hole",
            },
            {
                "subject_role": "threaded_joiner",
                "relation": "must fit the workbench target hole and thread into the hole",
                "object_role": "repair_hole",
            },
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


class FakeTransport:
    def __init__(self, document: dict) -> None:
        self.document = document
        self.calls = 0
        self.payload = None

    def complete(self, payload):
        self.calls += 1
        self.payload = payload
        return {"choices": [{"message": {"content": json.dumps(self.document)}}]}


def provider_for(document: dict) -> tuple[FMRequirementProvider, FakeTransport]:
    transport = FakeTransport(document)
    adapter = FMAdapter(
        base_url="http://unused/v1", model="qwen35-9b", transport=transport
    )
    return FMRequirementProvider(adapter), transport


def test_adapter_builds_multimodal_planning_free_request(observation_image):
    provider, transport = provider_for(natural_decomposition())
    provider.get_requirements(observation_images=[observation_image])
    payload = transport.payload
    assert payload["model"] == "qwen35-9b"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["type"] == "json_schema"
    prompt = payload["messages"][0]["content"].lower()
    assert "infer the complete set" in prompt
    assert "do not produce an action sequence" in prompt
    assert payload["messages"][1]["content"][1]["type"] == "image_url"


def test_prompt_does_not_contain_expected_answers(observation_image):
    provider, transport = provider_for(natural_decomposition())
    provider.get_requirements(observation_images=[observation_image])
    text = transport.payload["messages"][1]["content"][0]["text"]
    system = transport.payload["messages"][0]["content"]
    request = json.loads(text)
    assert set(request) == {"task_instruction", "request"}
    assert "role_envelopes" not in text
    combined = f"{system}\n{text}"
    assert "CAN_DRIVE_SCREW" not in combined
    assert "REACHES_TARGET" not in combined
    assert "screwdriver" not in combined.casefold()
    assert "power drill" not in combined.casefold()


def test_model_roles_and_visible_candidates_normalize_once(observation_image):
    provider, transport = provider_for(natural_decomposition())
    requirements = provider.get_requirements(observation_images=[observation_image])
    assert transport.calls == 1
    assert [requirement.function_name for requirement in requirements] == [
        "CAN_DRIVE_SCREW", "CAN_FASTEN",
    ]
    assert all(requirement.source == RequirementSource.FM for requirement in requirements)
    assert requirements[0].accepted_categories == ["screwdriver"]
    assert requirements[0].semantic_hints == ["Phillips screwdriver"]
    assert set(requirements[0].required_relations) == {
        "REACHES_TARGET", "COMPATIBLE_WITH",
    }

    # The reviewed detector vocabulary can include unseen alternatives for later search.
    assert provider.get_detector_prompts() == [
        "screwdriver", "screw", "power drill", "wooden hammer",
    ]
    assert transport.calls == 1


def test_result_records_image_provenance_and_no_execution(observation_image):
    provider, transport = provider_for(natural_decomposition())
    result = build_result(
        provider, "Find and install a compatible screw", [observation_image]
    )
    assert result["fm_calls"] == 1
    assert result["initial_observation_images"][0]["sha256"]
    assert result["planning_started"] is False
    assert result["execution_started"] is False
    assert transport.calls == 1


def test_vlm_unsupported_property_on_object_role_fails_closed(observation_image):
    document = natural_decomposition()
    document["functional_roles"][0]["required_properties"] = [
        "planar support"
    ]
    provider, _ = provider_for(document)
    from mujoco_scenes.functional_tamp_pipeline.errors import UnsupportedCheckerCapabilityError
    with pytest.raises(UnsupportedCheckerCapabilityError, match="not supported in canonical Workshop G_F"):
        provider.get_requirements(observation_images=[observation_image])


def test_vlm_unmapped_unary_property_fails_closed(observation_image):
    document = natural_decomposition()
    document["functional_roles"][0]["required_properties"] = [
        "completely unmapped property"
    ]
    provider, _ = provider_for(document)
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError
    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any Workshop unary property"):
        provider.get_requirements(observation_images=[observation_image])


def test_role_authority_function_and_description_only():
    from mujoco_scenes.workshop_phase1.requirements import map_workshop_role_function
    # Driver function with non-driver categories
    raw_driver = {
        "function": "tool to drive screws into frame",
        "description": "hand tool for tightening screws",
        "candidate_categories": ["wooden hammer", "mallet"],
    }
    assert map_workshop_role_function(raw_driver) == "CAN_DRIVE_SCREW"

    # Fastener function with non-fastener categories
    raw_fastener = {
        "function": "threaded fastener to secure the loose joint",
        "description": "fastener to hold joint together",
        "candidate_categories": ["power drill", "wrench"],
    }
    assert map_workshop_role_function(raw_fastener) == "CAN_FASTEN"


def test_candidate_categories_cannot_manufacture_driver_or_fastener():
    from mujoco_scenes.workshop_phase1.requirements import map_workshop_role_function
    # Non-driver function with driver candidate categories
    raw_unmapped = {
        "function": "paint the wall surface",
        "description": "decorative coating applicator",
        "candidate_categories": ["screwdriver", "power driver"],
    }
    assert map_workshop_role_function(raw_unmapped) is None

    # Non-fastener function with fastener candidate categories
    raw_unmapped_2 = {
        "function": "illuminate workspace",
        "description": "lighting fixture",
        "candidate_categories": ["screw", "Phillips screw"],
    }
    assert map_workshop_role_function(raw_unmapped_2) is None


def test_unknown_fixed_target_fails_closed():
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError
    doc = natural_decomposition()
    doc["functional_roles"][2] = {
        "id": "unknown_target",
        "entity_kind": "FIXED_TARGET",
        "function": "display shelf target",
        "description": "shelf for ornaments",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["shelf"],
        "visible_candidates": [],
        "required_properties": [],
    }
    doc["functional_relations"][1]["object_role"] = "unknown_target"
    doc["functional_relations"][2]["object_role"] = "unknown_target"
    provider = FMRequirementProvider()
    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any Workshop fixed target"):
        provider.generate_canonical(raw_document=doc)


def test_unknown_region_fails_closed():
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    from mujoco_scenes.functional_tamp_pipeline.errors import UnmappedFunctionalConceptError
    doc = natural_decomposition()
    doc["functional_roles"].append({
        "id": "unknown_region",
        "entity_kind": "REGION",
        "function": "seating rest",
        "description": "cushioned armchair",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["chair"],
        "visible_candidates": [],
        "required_properties": [],
    })
    provider = FMRequirementProvider()
    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped to any Workshop context region"):
        provider.generate_canonical(raw_document=doc)


def test_duplicate_roles_fail_closed_without_alternative_evidence():
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    from mujoco_scenes.functional_tamp_pipeline.errors import AmbiguousCanonicalizationError
    doc = natural_decomposition()
    doc["functional_roles"].append({
        "id": "second_driver",
        "entity_kind": "OBJECT",
        "function": "tighten a screw",
        "description": "another screwdriver",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["screwdriver"],
        "visible_candidates": [],
        "required_properties": [],
    })
    provider = FMRequirementProvider()
    with pytest.raises(AmbiguousCanonicalizationError, match="without explicit alternative evidence"):
        provider.generate_canonical(raw_document=doc)


def test_explicit_alternative_driver_roles_merged():
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    doc = natural_decomposition()
    doc["functional_roles"][0] = {
        "id": "driver_option_1",
        "entity_kind": "OBJECT",
        "function": "tighten screw with alternative driver option",
        "description": "manual driver tool candidate",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["screwdriver", "Phillips screwdriver"],
        "visible_candidates": [],
        "required_properties": [],
    }
    doc["functional_roles"].append({
        "id": "driver_option_2",
        "entity_kind": "OBJECT",
        "function": "tighten screw with interchangeable tool candidate",
        "description": "power driver tool candidate",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["power drill", "cordless power drill"],
        "visible_candidates": [],
        "required_properties": [],
    })
    doc["functional_relations"][0]["subject_role"] = "driver_option_1"
    doc["functional_relations"][1]["subject_role"] = "driver_option_1"
    provider = FMRequirementProvider()
    res = provider.generate_canonical(raw_document=doc)
    assert res["status"] == "CANONICALIZED"
    driver_roles = [r for r in provider.normalized_roles if r.canonical_role_id == "driver"]
    assert len(driver_roles) == 1
    assert driver_roles[0].required_count == 1
    assert "screwdriver" in driver_roles[0].candidate_categories
    assert "power drill" in driver_roles[0].candidate_categories
    accounting = provider.canonicalization_trace["concept_accounting"]
    assert accounting["roles"]["driver_option_1"]["status"] == "MERGED_BY_EXPLICIT_ALTERNATIVE_RULE"
    assert accounting["roles"]["driver_option_2"]["status"] == "MERGED_BY_EXPLICIT_ALTERNATIVE_RULE"


def test_schema_missing_required_fields_fail_closed():
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError

    # Missing required_count
    doc = natural_decomposition()
    del doc["functional_roles"][0]["required_count"]
    provider = FMRequirementProvider()
    with pytest.raises(MalformedVLMSpecificationError, match="missing required schema field 'required_count'"):
        provider.generate_canonical(raw_document=doc)

    # Missing binding_policy
    doc = natural_decomposition()
    del doc["functional_roles"][0]["binding_policy"]
    with pytest.raises(MalformedVLMSpecificationError, match="missing required schema field 'binding_policy'"):
        provider.generate_canonical(raw_document=doc)


def test_signature_aware_relation_canonicalization():
    from mujoco_scenes.workshop_phase1.requirements import canonicalize_workshop_relation
    # Forward relations
    assert canonicalize_workshop_relation("d", "driver", "compatible with fastener", "f", "fastener") == (
        "driver", "COMPATIBLE_WITH", "fastener", "PRESERVED", "GRAPH_RELATION"
    )
    assert canonicalize_workshop_relation("d", "driver", "reaches target hole", "t", "repair_target") == (
        "driver", "REACHES_TARGET", "repair_target", "PRESERVED", "GRAPH_RELATION"
    )
    assert canonicalize_workshop_relation("f", "fastener", "threads into target repair hole", "t", "repair_target") == (
        "fastener", "COMPATIBLE_WITH_TARGET", "repair_target", "PRESERVED", "GRAPH_RELATION"
    )

    # Reverse relations normalized
    assert canonicalize_workshop_relation("f", "fastener", "is driven by tool", "d", "driver") == (
        "driver", "COMPATIBLE_WITH", "fastener", "NORMALIZED_TO_CANONICAL_SIGNATURE", "GRAPH_RELATION"
    )
    assert canonicalize_workshop_relation("t", "repair_target", "is reached by driver", "d", "driver") == (
        "driver", "REACHES_TARGET", "repair_target", "NORMALIZED_TO_CANONICAL_SIGNATURE", "GRAPH_RELATION"
    )
    assert canonicalize_workshop_relation("t", "repair_target", "receives fastener", "f", "fastener") == (
        "fastener", "COMPATIBLE_WITH_TARGET", "repair_target", "NORMALIZED_TO_CANONICAL_SIGNATURE", "GRAPH_RELATION"
    )


def test_self_relations_fail_closed():
    from mujoco_scenes.workshop_phase1.requirements import canonicalize_workshop_relation
    from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
    with pytest.raises(MalformedVLMSpecificationError, match="Self-relations are not supported"):
        canonicalize_workshop_relation("role_1", "driver", "compatible with", "role_1", "driver")


def test_unsupported_functional_located_on_fails_closed():
    from mujoco_scenes.workshop_phase1.requirements import canonicalize_workshop_relation
    from mujoco_scenes.functional_tamp_pipeline.errors import UnsupportedCheckerCapabilityError
    with pytest.raises(UnsupportedCheckerCapabilityError, match="LOCATED_ON on role 'role_1' is not supported"):
        canonicalize_workshop_relation("role_1", "driver", "located on workbench", "role_2", "MAIN_WORKBENCH_ZONE")


def test_operation_group_redundancy_validation():
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
    from mujoco_scenes.functional_tamp_pipeline.tests.test_ideal_fixtures import load_ideal_fixture, MockFMAdapter
    data = load_ideal_fixture("workshop")
    adapter = MockFMAdapter(data)
    provider = FMRequirementProvider(fm_adapter=adapter)
    gf = VLMSpecProvider._workshop("Repair frame", [], provider=provider)

    # Runtime G_F has NO operation groups
    assert len(gf.operation_groups) == 0

    # Operation group validated and recorded in concept accounting
    accounting = provider.canonicalization_trace["concept_accounting"]
    assert len(accounting["operation_groups"]) == 1
    grp_entry = accounting["operation_groups"][0]
    assert grp_entry["raw_group_id"] == "group_1"
    assert grp_entry["status"] == "MERGED_BY_EXPLICIT_RULE"
    assert grp_entry["structural_destination"] == "REDUNDANT_WITH_CANONICAL_GRAPH_RELATIONS"
    assert set(grp_entry["represented_relations"]) == {"COMPATIBLE_WITH", "REACHES_TARGET"}


def test_workshop_vlm_canonicalization_version():
    from mujoco_scenes.workshop_phase1.requirements import (
        FMRequirementProvider,
        WORKSHOP_VLM_CANONICALIZATION_VERSION,
    )
    from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
    from mujoco_scenes.functional_tamp_pipeline.tests.test_ideal_fixtures import load_ideal_fixture, MockFMAdapter
    assert WORKSHOP_VLM_CANONICALIZATION_VERSION == "phase3_p3g_v1"
    data = load_ideal_fixture("workshop")
    adapter = MockFMAdapter(data)
    provider = FMRequirementProvider(fm_adapter=adapter)
    gf = VLMSpecProvider._workshop("Repair frame", [], provider=provider)
    assert gf.metadata["vlm_canonicalization_version"] == "phase3_p3g_v1"
    assert provider.canonicalization_trace["vlm_canonicalization_version"] == "phase3_p3g_v1"


def test_transport_schema_rejects_old_candidate_types_and_extra_fields():
    document = natural_decomposition()
    document["actions"] = []
    with pytest.raises(FMResponseValidationError, match="Unexpected top-level fields"):
        validate_requirement_response(document)

    document = natural_decomposition()
    role = document["functional_roles"][0]
    role["candidate_types"] = ["screwdriver"]
    with pytest.raises(FMResponseValidationError, match="invalid fields"):
        validate_requirement_response(document)


def test_image_required_before_network(monkeypatch):
    for variable in ("TAMP_FM_BASE_URL", "FM_BASE_URL", "TAMP_FM_MODEL", "FM_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    adapter = FMAdapter()
    with pytest.raises(ValueError, match="initial-observation image"):
        adapter.generate_task_requirements(
            "Find a compatible screw and driver", observation_images=[]
        )


def test_missing_endpoint_after_image_validation(observation_image, monkeypatch):
    for variable in ("TAMP_FM_BASE_URL", "FM_BASE_URL", "TAMP_FM_MODEL", "FM_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(FMBackendNotConfiguredError, match="TAMP_FM_BASE_URL"):
        FMAdapter().generate_task_requirements(
            "Find a compatible screw and driver",
            observation_images=[observation_image],
        )


def test_workshop_local_id_collision_independence():
    from mujoco_scenes.workshop_phase1.requirements import resolve_workshop_region_proposal
    proposal = {
        "id": "LEFT_DRAWER",
        "label": "right storage drawer",
        "visual_description": "drawer on right side of workbench",
    }
    resolved = resolve_workshop_region_proposal(proposal)
    assert resolved == "RIGHT_DRAWER", f"Expected RIGHT_DRAWER from visual label, got {resolved}"


def test_workshop_fixed_target_role_tracing(observation_image):
    doc = natural_decomposition()
    provider, _ = provider_for(doc)
    reqs = provider.get_requirements(observation_images=[observation_image])
    assert len(reqs) == 2
    assert any(t.get("transformation") == "SYSTEM_OWNED_FIXED_TARGET_REPRESENTATION" for t in provider.transformation_trace)
