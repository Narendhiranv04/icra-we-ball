import pytest
from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker

def test_winner_policies():
    tracker = PersistentInstanceTracker()
    
    observations = [
        {"camera_id": "CAM1", "canonical_label": "A", "confidence": 0.1, "physical_support_quality": 1.0, "inference_source": "full"},
        {"camera_id": "CAM2", "canonical_label": "A", "confidence": 0.1, "physical_support_quality": 1.0, "inference_source": "full"},
        {"camera_id": "CAM3", "canonical_label": "A", "confidence": 0.1, "physical_support_quality": 1.0, "inference_source": "full"},
        {"camera_id": "CAM4", "canonical_label": "B", "confidence": 0.05, "physical_support_quality": 1.0, "inference_source": "proposal_crop"},
        {"camera_id": "CAM5", "canonical_label": "B", "confidence": 0.05, "physical_support_quality": 1.0, "inference_source": "proposal_crop"}
    ]
    
    config1 = {
        "winner_policy": "supporting_views_then_weighted_score",
        "minimum_supporting_views": 1,
        "proposal_crop_score_multiplier": 20.0
    }
    belief1 = tracker._compute_consensus_semantic_belief(observations, config1)
    assert belief1["canonical_label"] == "a"
    
    config2 = {
        "winner_policy": "weighted_score_then_supporting_views",
        "minimum_supporting_views": 1,
        "proposal_crop_score_multiplier": 20.0
    }
    belief2 = tracker._compute_consensus_semantic_belief(observations, config2)
    assert belief2["canonical_label"] == "b"

    config3 = {
        "winner_policy": "weighted_score_then_supporting_views",
        "minimum_supporting_views": 3,
        "proposal_crop_score_multiplier": 20.0
    }
    belief3 = tracker._compute_consensus_semantic_belief(observations, config3)
    assert belief3["status"] == "UNKNOWN"

