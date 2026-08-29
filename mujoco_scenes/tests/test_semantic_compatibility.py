from mujoco_scenes.functional_tamp_pipeline.grounding import check_semantic_role_compatibility
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode

def test_check_semantic_role_compatibility_mixed_plausible():
    accepted = ["screwdriver", "power_driver"]
    
    # 1. Mixed overlap -> UNKNOWN
    belief = {
        "status": "UNKNOWN", 
        "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"],
        "latest_observation": {
            "plausible_labels": ["screwdriver", "hammer"]
        }
    }
    status, _ = check_semantic_role_compatibility(belief, accepted)
    assert status == "UNKNOWN"

    # 2. Subset -> TRUE
    belief_true = {
        "status": "UNKNOWN", 
        "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"],
        "latest_observation": {
            "plausible_labels": ["screwdriver"]
        }
    }
    status, _ = check_semantic_role_compatibility(belief_true, accepted)
    assert status == "TRUE"

    # 3. Disjoint -> FALSE
    belief_false = {
        "status": "UNKNOWN", 
        "reason_codes": ["CONFLICTING_MULTI_VIEW_LABELS"],
        "latest_observation": {
            "plausible_labels": ["hammer"]
        }
    }
    status, _ = check_semantic_role_compatibility(belief_false, accepted)
    assert status == "FALSE"
