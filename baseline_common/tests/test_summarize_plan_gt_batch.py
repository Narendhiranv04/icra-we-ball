from baseline_common.summarize_plan_gt_batch import (
    _apply_planned_placements,
    _living_room_placement_metrics,
)


def _snapshot(placements):
    regions = [
        {"id": "left", "label": "personal_table_left"},
        {"id": "shared", "label": "shared_table"},
        {"id": "right", "label": "personal_table_right"},
        {"id": "staging", "label": "staging_area"},
    ]
    entities = [
        {
            "id": f"object_{index}",
            "kind": "object",
            "label": label,
            "facts": {"region_id": region},
        }
        for index, (label, region) in enumerate(placements, 1)
    ]
    return {"known_regions": regions, "visible_entities": entities}


def test_living_room_placement_metrics_score_complete_goal():
    snapshot = _snapshot(
        [
            ("cup_1", "left"),
            ("saucer_1", "left"),
            ("cup_2", "right"),
            ("saucer_2", "right"),
            ("tv_remote", "shared"),
        ]
    )
    assert _living_room_placement_metrics(snapshot) == (1.0, 1.0)


def test_living_room_placement_metrics_penalize_duplicate_and_missing_pairs():
    snapshot = _snapshot(
        [
            ("cup_1", "left"),
            ("saucer_1", "left"),
            ("cup_2", "left"),
            ("saucer_2", "staging"),
            ("tv_remote", "shared"),
        ]
    )
    correctness, coverage = _living_room_placement_metrics(snapshot)
    assert correctness == 3 / 4
    assert coverage == 3 / 5


def test_apply_planned_placements_supports_planning_only_owl_output():
    snapshot = _snapshot([("cup_1", "staging")])
    payload = {
        "result": {
            "actions": [
                {"operator": "PICK", "arguments": ["object_1"]},
                {"operator": "PLACE", "arguments": ["object_1", "left"]},
            ]
        }
    }
    _apply_planned_placements(snapshot, payload)
    assert snapshot["visible_entities"][0]["facts"]["region_id"] == "left"
