"""Tests for image-conditioned Kitchen and Living-Room Qwen requirements."""

from __future__ import annotations

import base64
import json

import pytest
import yaml

from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider
from mujoco_scenes.living_room_region_function import load_integrated_task
from mujoco_scenes.run_environment_vlm_requirements import (
    DIRECT_SCENE_CAMERAS,
    available_variants,
)
from mujoco_scenes.functional_tamp_pipeline.errors import TransportOrStructuredOutputError
from mujoco_scenes.task_witness import load_task_requirements
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter, FMResponseValidationError


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture
def observation_image(tmp_path):
    path = tmp_path / "initial_observation.png"
    path.write_bytes(PNG_1X1)
    return path


def candidate(label: str, description: str) -> dict:
    return {
        "label": label,
        "visual_description": description,
        "suitability_reason": "Visually plausible pending geometry checks.",
    }


def kitchen_decomposition() -> dict:
    # IDs and ordering are chosen by the model, not supplied by the client.
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two servings each of coffee and soup.",
        "functional_roles": [
            {
                "id": "utensil_for_soup",
                "entity_kind": "OBJECT",
                "function": "provide a soup eating utensil",
                "description": "A utensil suitable for consuming soup from a bowl.",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["soup_spoon", "spoon"],
                "visible_candidates": [candidate("metal spoon", "silver utensil")],
                "required_properties": ["elongated object"],
            },
            {
                "id": "vessel_for_coffee",
                "entity_kind": "OBJECT",
                "function": "hold coffee",
                "description": "An individual drinking vessel for coffee.",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup", "coffee mug"],
                "visible_candidates": [
                    candidate("cup", "small cup near the counter edge"),
                    candidate("coffee mug", "handled mug beside the bowl"),
                ],
                "required_properties": ["open cavity"],
            },
            {
                "id": "mixing_implement",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "description": "An implement for mixing coffee inside its vessel.",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["spoon", "stirrer"],
                "visible_candidates": [candidate("spoon", "long silver spoon")],
                "required_properties": ["elongated object"],
            },
            {
                "id": "vessel_for_soup",
                "entity_kind": "OBJECT",
                "function": "contain soup",
                "description": "An individual vessel used to serve soup.",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["bowl", "soup bowl"],
                "visible_candidates": [candidate("soup bowl", "round bowl on the worktop")],
                "required_properties": ["open cavity"],
            },
            {
                "id": "coffee_material_source",
                "entity_kind": "OBJECT",
                "function": "provide coffee",
                "description": "A source of coffee material for both drinks.",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["coffee jar", "coffee material"],
                "visible_candidates": [candidate("coffee jar", "green coffee container")],
                "required_properties": [],
            },
            {
                "id": "pourable_water_source",
                "entity_kind": "OBJECT",
                "function": "provide water",
                "description": "A kettle that supplies water to both drinks.",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["kettle", "water source"],
                "visible_candidates": [candidate("kettle", "white handled kettle")],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "utensil_for_soup",
                "relation": "enter opening",
                "object_role": "vessel_for_soup",
            },
            {
                "subject_role": "mixing_implement",
                "relation": "reaches bottom",
                "object_role": "vessel_for_coffee",
            },
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


def living_room_decomposition() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Find personal drinkware surfaces and a shared remote surface.",
        "functional_roles": [
            {
                "id": "central_control_surface",
                "entity_kind": "REGION",
                "function": "support the television remote for both viewers",
                "description": "A shared central surface reachable from both seats.",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee table"],
                "visible_candidates": [candidate("coffee table", "low central table")],
                "required_properties": ["planar support"],
            },
            {
                "id": "individual_drink_surface",
                "entity_kind": "REGION",
                "function": "support a cup and saucer near a seat",
                "description": "A personal surface adjacent to each viewer.",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table", "end table"],
                "visible_candidates": [
                    candidate("side table", "small table beside the left chair"),
                    candidate("end table", "small table beside the right chair"),
                ],
                "required_properties": ["planar support"],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "central_control_surface",
                "relation": "accessible from both seats",
                "object_role": "central_control_surface",
            },
            {
                "subject_role": "individual_drink_surface",
                "relation": "near the assigned seat",
                "object_role": "individual_drink_surface",
            },
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


class FakeTransport:
    def __init__(self, document: Any) -> None:
        self.document = document
        self.calls = 0
        self.payload = None

    def complete(self, payload):
        self.calls += 1
        self.payload = payload
        if isinstance(self.document, dict) and "choices" in self.document:
            return self.document
        content = json.dumps(self.document) if not isinstance(self.document, str) else self.document
        return {"choices": [{"message": {"content": content}}]}


def provider_for(environment: str, document: dict):
    transport = FakeTransport(document)
    adapter = FMAdapter(
        base_url="http://unused/v1", model="qwen35-9b", transport=transport
    )
    return EnvironmentVLMRequirementProvider(environment, fm_adapter=adapter), transport


@pytest.mark.parametrize(
    ("environment", "document", "expected_ids"),
    [
        (
            "kitchen", kitchen_decomposition(),
            [
                "coffee_container", "soup_container", "coffee_stirrer",
                "soup_eating_utensil", "coffee_source", "water_source",
            ],
        ),
        (
            "living_room", living_room_decomposition(),
            ["personal_cup_saucer", "shared_remote"],
        ),
    ],
)
def test_model_chosen_roles_normalize_after_the_call(
    environment, document, expected_ids, observation_image
):
    provider, transport = provider_for(environment, document)
    result = provider.generate(observation_images=[observation_image])

    assert transport.calls == 1
    assert [row["role_id"] for row in result["normalized_requirements"]] == expected_ids
    raw_id = result["normalized_requirements"][0].get("raw_vlm_role_id") or result["normalized_requirements"][0]["raw_vlm_role_ids"][0]
    assert raw_id != expected_ids[0]
    assert result["initial_observation_images"][0]["sha256"]
    if environment == "kitchen":
        assert result["normalized_task_contract"]["generated_from_foundation_model"] is True
    assert result["raw_vlm_requirement_response"] == document
    for boundary in (
        "observation_search_started", "semantic_grounding_started", "allocation_started",
        "geometry_verification_started", "planning_started", "execution_started",
    ):
        assert result[boundary] is False

    assert provider.generate(observation_images=[]) == result
    assert transport.calls == 1


def test_prompt_contains_only_goal_generic_request_and_images(observation_image):
    provider, transport = provider_for("kitchen", kitchen_decomposition())
    provider.generate(observation_images=[observation_image])
    content = transport.payload["messages"][1]["content"]
    request = json.loads(content[0]["text"])
    assert set(request) == {"task_instruction", "request"}
    assert "role_envelopes" not in content[0]["text"]
    leaked_answers = (
        "coffee_container", "soup_container", "coffee_stirrer", "open cavity",
        "insert into the target", "reach the bottom", "PLANAR_SUPPORT", "side table",
    )
    assert not any(answer in content[0]["text"] for answer in leaked_answers)
    system_prompt = transport.payload["messages"][0]["content"]
    assert not any(answer in system_prompt for answer in leaked_answers)
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_visible_candidates_are_not_frozen_alternatives(observation_image):
    document = kitchen_decomposition()
    document["functional_roles"][1]["visible_candidates"] = [
        candidate("cup", "the only visible drinking vessel")
    ]
    provider, _ = provider_for("kitchen", document)
    coffee = provider.generate(observation_images=[observation_image])[
        "normalized_requirements"
    ][0]
    assert coffee["semantic_hints"] == ["cup"]
    assert coffee["accepted_categories"] == ["cup", "mug"]
    assert coffee["visible_candidates"][0]["label"] == "cup"


def test_kitchen_contract_remains_accepted_by_existing_loader(observation_image):
    provider, _ = provider_for("kitchen", kitchen_decomposition())
    result = provider.generate(observation_images=[observation_image])
    loaded = load_task_requirements(result["normalized_task_contract"])
    assert loaded["roles"]["coffee_container"]["count"] == 2
    assert loaded["operation_groups"]["soup_serving"]["required_target_count"] == 2


def test_living_contract_remains_accepted_by_existing_loader(tmp_path, observation_image):
    provider, _ = provider_for("living_room", living_room_decomposition())
    result = provider.generate_canonical(
        "Prepare living room", observation_images=[observation_image]
    )
    assert len(result["normalized_requirements"]) == 2


def test_custom_instruction_is_the_only_task_content_sent(observation_image):
    provider, transport = provider_for("kitchen", kitchen_decomposition())
    result = provider.generate(
        "Prepare refreshments for the guests.", observation_images=[observation_image]
    )
    request = json.loads(transport.payload["messages"][1]["content"][0]["text"])
    assert request["task_instruction"] == result["task_instruction"]


def test_missing_required_property_is_saved_for_review_and_blocks_handoff(observation_image):
    document = kitchen_decomposition()
    document["functional_roles"][2]["required_properties"] = []
    provider, _ = provider_for("kitchen", document)
    result = provider.generate(observation_images=[observation_image])
    assert result["ready_for_grounding"] is False
    assert result["normalized_task_contract"] is None
    assert any(
        "omitted required properties" in issue
        for issue in result["reviewed_ontology_audit"]["issues"]
    )

    strict_provider, _ = provider_for("kitchen", document)
    with pytest.raises(ValueError, match="not ready for grounding"):
        strict_provider.generate(
            observation_images=[observation_image], require_reviewed_contract=True
        )


def test_unrelated_function_is_unmapped_post_response(observation_image):
    document = kitchen_decomposition()
    document["functional_roles"][1].update(
        {"function": "decorate a shelf", "description": "Pure decoration."}
    )
    provider, _ = provider_for("kitchen", document)
    result = provider.generate(observation_images=[observation_image])
    assert result["ready_for_grounding"] is False
    assert any(
        "unmapped or ambiguous" in issue
        for issue in result["reviewed_ontology_audit"]["issues"]
    )


def test_image_is_required_before_model_call():
    provider, transport = provider_for("kitchen", kitchen_decomposition())
    with pytest.raises(ValueError, match="initial-observation image"):
        provider.generate(observation_images=[])
    assert transport.calls == 0


def test_invalid_environment_is_rejected():
    with pytest.raises(ValueError, match="environment must be one of"):
        EnvironmentVLMRequirementProvider("bedroom")


def test_direct_scene_input_is_variant_specific_and_five_view():
    assert "F0_ALL_VISIBLE" in available_variants("kitchen")
    assert "F0_ALL_OBJECTS_IN_STAGING" in available_variants("living_room")
    assert len(DIRECT_SCENE_CAMERAS["kitchen"]) == 5
    assert len(DIRECT_SCENE_CAMERAS["living_room"]) == 5
    assert all("yolo" not in camera.casefold() for cameras in DIRECT_SCENE_CAMERAS.values() for camera in cameras)


def test_kitchen_inspection_policy_is_qwen_ranked_without_hidden_contents(observation_image):
    regions = {
        "D1": "upper kitchen drawer",
        "D2": "lower kitchen drawer",
        "C2": "upper wall cupboard",
        "B1": "countertop storage box",
        "C1": "lower kitchen cupboard",
    }
    policy = {
        "initial_requirements_satisfied": False,
        "decision_reason": "Some required candidates are not visible.",
        "inspection_order": [
            {"region_id": region, "reason": "Gather missing visual evidence."}
            for region in ("C2", "B1", "D1", "D2", "C1")
        ],
    }
    transport = FakeTransport(policy)
    adapter = FMAdapter(base_url="http://unused/v1", model="qwen35-9b", transport=transport)
    result = adapter.generate_inspection_priors(
        "Prepare coffee and soup.", regions, observation_images=[observation_image]
    )
    assert [row["region_id"] for row in result["inspection_order"]] == [
        "C2", "B1", "D1", "D2", "C1"
    ]
    assert adapter.last_raw_inspection_response == policy
    prompt = json.loads(transport.payload["messages"][1]["content"][0]["text"])
    assert set(prompt) == {"task_instruction", "request"}
    serialized = json.dumps(prompt).casefold()
    assert "hidden objects" not in serialized
    assert "intended_outcome" not in serialized
    assert "expected_gt" not in serialized


def test_living_room_canonical_role_consolidation(observation_image):
    decomposition = {
        "status": "SUPPORTED",
        "task_summary": "Prepare living room seating surfaces for drinks and remote.",
        "functional_roles": [
            {
                "id": "viewer_1_side_table",
                "entity_kind": "REGION",
                "function": "personal cup and saucer support",
                "description": "hold viewer 1 drinkware",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table"],
                "visible_candidates": [
                    candidate("small side table", "left viewer individual side table"),
                ],
                "required_properties": ["planar support"],
            },
            {
                "id": "viewer_2_side_table",
                "entity_kind": "REGION",
                "function": "personal cup and saucer support",
                "description": "hold viewer 2 drinkware",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table"],
                "visible_candidates": [
                    candidate("small side table", "right viewer individual side table"),
                ],
                "required_properties": ["planar support"],
            },
            {
                "id": "shared_coffee_table",
                "entity_kind": "REGION",
                "function": "shared remote support",
                "description": "hold television remote for both viewers",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee table"],
                "visible_candidates": [
                    candidate("coffee table", "central low coffee table"),
                ],
                "required_properties": ["planar support"],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "viewer_1_side_table",
                "relation": "within reach of one seated person",
                "object_role": "viewer_1_side_table",
            },
            {
                "subject_role": "viewer_2_side_table",
                "relation": "within reach of one seated person",
                "object_role": "viewer_2_side_table",
            },
            {
                "subject_role": "shared_coffee_table",
                "relation": "accessible from both seats",
                "object_role": "shared_coffee_table",
            },
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    provider, _ = provider_for("living_room", decomposition)
    result = provider.generate_canonical(
        "Prepare living room for two people",
        observation_images=[observation_image],
    )
    reqs = result["normalized_requirements"]
    # Exactly 2 consolidated canonical roles: personal_cup_saucer (count 2) and shared_remote (count 1)
    assert len(reqs) == 2
    personal_req = next(r for r in reqs if r["role_id"] == "personal_cup_saucer")
    assert personal_req["vlm_required_count"] == 2
    assert personal_req["raw_vlm_role_ids"] == ["viewer_1_side_table", "viewer_2_side_table"]
    assert personal_req["function"] == "PERSONAL_CUP_SAUCER_REGION"

    shared_req = next(r for r in reqs if r["role_id"] == "shared_remote")
    assert shared_req["vlm_required_count"] == 1
    assert shared_req["raw_vlm_role_ids"] == ["shared_coffee_table"]
    assert shared_req["function"] == "SHARED_REMOTE_REGION"

    # Now verify FunctionalRequirementGraph construction and validation
    from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
    vlm_spec = VLMSpecProvider()
    graph = vlm_spec._living_room("Prepare living room", [observation_image], provider=provider)
    graph.validate()
    assert "PERSONAL_CUP_SAUCER_REGION" in graph.nodes or "personal_cup_saucer" in graph.nodes
    assert "SHARED_REMOTE_REGION" in graph.nodes or "shared_remote" in graph.nodes


def test_fm_diagnostics_saved_on_failure_and_success(tmp_path, monkeypatch):
    import os
    diag_dir = tmp_path / "test_diag"
    monkeypatch.setenv("TAMP_FM_DIAGNOSTIC_DIR", str(diag_dir))

    # Test 1: finish_reason="length", truncated malformed JSON
    mock_resp_length = {
        "model": "qwen35-9b",
        "choices": [{"finish_reason": "length", "message": {"content": '{"status": "SUPPORTED", "roles": ['}}],
        "usage": {"total_tokens": 4096},
    }
    transport_err = FakeTransport(mock_resp_length)
    adapter_err = FMAdapter(transport=transport_err)
    img_err = tmp_path / "img_err.png"
    img_err.write_bytes(PNG_1X1)
    with pytest.raises((FMResponseValidationError, TransportOrStructuredOutputError)):
        adapter_err.generate_kitchen_functional_graph("task", {}, observation_images=[img_err])

    f1 = diag_dir / "fm_call_001.json"
    assert f1.exists()
    d1 = json.loads(f1.read_text(encoding="utf-8"))
    assert d1["finish_reason"] == "length"
    assert d1["json_parse_success"] is False
    assert d1["content_length_chars"] > 0
    assert d1["content_sha256"] is not None

    # Test 2: finish_reason="stop", valid JSON
    valid_doc = {
        "status": "SUPPORTED",
        "task_summary": "sum",
        "functional_roles": [
            {
                "id": "r1", "entity_kind": "OBJECT", "function": "func", "description": "desc",
                "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["cand"],
                "visible_candidates": [], "required_properties": ["planar support"],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    mock_resp_ok = {
        "model": "qwen35-9b",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_doc)}}],
        "usage": {"total_tokens": 500},
    }
    transport_ok = FakeTransport(mock_resp_ok)
    adapter_ok = FMAdapter(transport=transport_ok)
    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    res = adapter_ok.generate_task_requirements("task", observation_images=[img])
    assert res["status"] == "SUPPORTED"

    f2 = diag_dir / "fm_call_002.json"
    assert f2.exists()
    d2 = json.loads(f2.read_text(encoding="utf-8"))
    assert d2["finish_reason"] == "stop"
    assert d2["json_parse_success"] is True


def test_fm_diagnostics_not_saved_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("TAMP_FM_DIAGNOSTIC_DIR", raising=False)
    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    valid_doc = {
        "status": "SUPPORTED",
        "task_summary": "sum",
        "functional_roles": [
            {
                "id": "r1", "entity_kind": "OBJECT", "function": "func", "description": "desc",
                "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["cand"],
                "visible_candidates": [], "required_properties": ["planar support"],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    mock_resp = {
        "model": "qwen35-9b",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_doc)}}],
    }
    transport = FakeTransport(mock_resp)
    adapter = FMAdapter(transport=transport)
    adapter.generate_task_requirements("task", observation_images=[img])
    assert not list(tmp_path.glob("fm_call_*.json"))
