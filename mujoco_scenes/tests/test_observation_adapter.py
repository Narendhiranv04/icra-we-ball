from mujoco_scenes.tamp.observation_adapter import observed_registry_state


INVENTORY = {
    "objects": [
        {
            "generic_object_id": "mug_1",
            "semantic_label": "mug",
            "semantic_label_source": "OBSERVED_SEMANTIC_DETECTOR",
            "observed_dimensions_m": {"height": 0.1},
            "selected_functions": ["coffee_vessel"],
            "source_context": {"observed_source_region": "countertop"},
        },
        {
            "generic_object_id": "spoon_1",
            "semantic_label": "spoon",
            "source_context": {"observed_source_region": "D1"},
        },
    ]
}


def test_closed_region_does_not_reveal_observed_inventory_item():
    state = observed_registry_state(
        INVENTORY,
        {"D1": {"open": False, "inspected": False}},
        robot_location="home",
        held_object=None,
        revision=1,
    )
    assert state.objects["mug_1"].visible
    assert not state.objects["spoon_1"].visible
    assert "physical_backend_body" not in state.objects["mug_1"].facts


def test_explicit_visibility_mask_is_authoritative():
    state = observed_registry_state(
        INVENTORY,
        {"D1": {"open": True, "inspected": True}},
        robot_location="home",
        held_object=None,
        revision=2,
        visible_object_ids={"spoon_1"},
    )
    assert not state.objects["mug_1"].visible
    assert state.objects["spoon_1"].visible


def test_live_location_overrides_discovery_region_but_preserves_provenance():
    state = observed_registry_state(
        INVENTORY,
        {"D1": {"open": False}, "countertop": {"open": True}},
        robot_location="home",
        held_object=None,
        revision=3,
        live_locations={"spoon_1": "countertop"},
    )
    observed = state.objects["spoon_1"]
    assert observed.location == "countertop"
    assert observed.visible
    assert observed.facts["source_region"] == "countertop"
    assert observed.facts["discovered_source_region"] == "D1"
