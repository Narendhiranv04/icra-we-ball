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
from mujoco_scenes.task_witness import load_task_requirements
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter


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
        "functional_requirements": [
            {
                "id": "utensil_for_soup",
                "entity_kind": "OBJECT",
                "function": "provide a soup eating utensil",
                "description": "A utensil suitable for consuming soup from a bowl.",
                "required_count": 2,
                "candidate_objects": [candidate("metal spoon", "silver utensil")],
                "required_properties": [
                    "elongated object", "enter the container opening", "reach the contents",
                ],
            },
            {
                "id": "vessel_for_coffee",
                "entity_kind": "OBJECT",
                "function": "hold coffee",
                "description": "An individual drinking vessel for coffee.",
                "required_count": 2,
                "candidate_objects": [
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
                "candidate_objects": [candidate("spoon", "long silver spoon")],
                "required_properties": [
                    "long narrow shape", "fit through the opening", "reach the bottom",
                ],
            },
            {
                "id": "vessel_for_soup",
                "entity_kind": "OBJECT",
                "function": "contain soup",
                "description": "An individual vessel used to serve soup.",
                "required_count": 2,
                "candidate_objects": [candidate("soup bowl", "round bowl on the worktop")],
                "required_properties": ["open cavity"],
            },
            {
                "id": "coffee_material_source",
                "entity_kind": "OBJECT",
                "function": "provide coffee",
                "description": "A source of coffee material for both drinks.",
                "required_count": 1,
                "candidate_objects": [candidate("coffee jar", "green coffee container")],
                "required_properties": ["dispenses coffee material"],
            },
            {
                "id": "pourable_water_source",
                "entity_kind": "OBJECT",
                "function": "provide water",
                "description": "A kettle that supplies water to both drinks.",
                "required_count": 1,
                "candidate_objects": [candidate("kettle", "white handled kettle")],
                "required_properties": ["holds and pours water"],
            },
        ],
        "unsupported_reason": "",
    }


def living_room_decomposition() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Find personal drinkware surfaces and a shared remote surface.",
        "functional_requirements": [
            {
                "id": "central_control_surface",
                "entity_kind": "REGION",
                "function": "support the television remote for both viewers",
                "description": "A shared central surface reachable from both seats.",
                "required_count": 1,
                "candidate_objects": [candidate("coffee table", "low central table")],
                "required_properties": [
                    "planar support", "fit the remote", "accessible from both seats",
                ],
            },
            {
                "id": "individual_drink_surface",
                "entity_kind": "REGION",
                "function": "support a cup and saucer near a seat",
                "description": "A personal surface adjacent to each viewer.",
                "required_count": 2,
                "candidate_objects": [
                    candidate("side table", "small table beside the left chair"),
                    candidate("end table", "small table beside the right chair"),
                ],
                "required_properties": [
                    "flat stable support", "fit a cup and saucer together",
                    "near the assigned seat",
                ],
            },
        ],
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
    assert result["normalized_requirements"][0]["raw_vlm_role_id"] != expected_ids[0]
    assert result["initial_observation_images"][0]["sha256"]
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
    document["functional_requirements"][1]["candidate_objects"] = [
        candidate("cup", "the only visible drinking vessel")
    ]
    provider, _ = provider_for("kitchen", document)
    coffee = provider.generate(observation_images=[observation_image])[
        "normalized_requirements"
    ][0]
    assert coffee["semantic_hints"] == ["cup"]
    assert coffee["accepted_categories"] == ["cup", "mug"]
    assert coffee["visible_candidate_objects"][0]["label"] == "cup"


def test_kitchen_contract_remains_accepted_by_existing_loader(observation_image):
    provider, _ = provider_for("kitchen", kitchen_decomposition())
    result = provider.generate(observation_images=[observation_image])
    loaded = load_task_requirements(result["normalized_task_contract"])
    assert loaded["roles"]["coffee_container"]["count"] == 2
    assert loaded["operation_groups"]["soup_serving"]["required_target_count"] == 2


def test_living_contract_remains_accepted_by_existing_loader(tmp_path, observation_image):
    provider, _ = provider_for("living_room", living_room_decomposition())
    result = provider.generate(observation_images=[observation_image])
    path = tmp_path / "living_room_vlm_task.yaml"
    path.write_text(yaml.safe_dump(result["normalized_task_contract"], sort_keys=False))
    loaded = load_integrated_task(path)
    assert loaded["requirement_entity_kind"] == "REGION"
    assert loaded["allocation"]["production_policy"] == "global_target_specific"


def test_custom_instruction_is_the_only_task_content_sent(observation_image):
    provider, transport = provider_for("kitchen", kitchen_decomposition())
    result = provider.generate(
        "Prepare refreshments for the guests.", observation_images=[observation_image]
    )
    request = json.loads(transport.payload["messages"][1]["content"][0]["text"])
    assert request["task_instruction"] == result["task_instruction"]


def test_missing_required_property_is_saved_for_review_and_blocks_handoff(observation_image):
    document = kitchen_decomposition()
    document["functional_requirements"][2]["required_properties"] = ["long narrow shape"]
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
    document["functional_requirements"][1].update(
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
    assert set(prompt) == {
        "task_instruction", "observable_closed_storage_regions", "request"
    }
    serialized = json.dumps(prompt).casefold()
    assert "hidden objects" not in serialized
    assert "intended_outcome" not in serialized
    assert "expected_gt" not in serialized
