from types import SimpleNamespace

import numpy as np

from mujoco_scenes.semantic_grounding import (
    Detection,
    _process_isolation_requested,
    associate_detections_to_masks,
    detector_vocabulary,
    fuse_semantic_observations,
    load_semantic_config,
    run_semantic_inspection,
)
from mujoco_scenes.task_witness import evaluate_semantic_compatibility


def _config():
    config = load_semantic_config()
    config["association"]["minimum_association_score"] = 0.1
    return config


def _detection(label, box, confidence=0.8, camera="inspection_left"):
    return Detection(
        raw_label=label,
        canonical_label=label,
        confidence=confidence,
        bbox_xyxy=box,
        source_camera=camera,
        detector_name="mock_detector",
        checkpoint="mock_checkpoint",
        detector_version="1",
        inference_resolution=(100, 100),
        input_image_path=f"cameras/{camera}/rgb.png",
    )


def _accepted_observation(
    label,
    camera,
    confidence=0.8,
    association_score=0.9,
    visible_pixels=400,
):
    detection = _detection(
        label,
        (10, 10, 30, 30),
        confidence=confidence,
        camera=camera,
    ).to_dict()
    return {
        "detection_index": 0,
        "object_id": "object_0001",
        "association_score": association_score,
        "metrics": {"visible_mask_pixels": visible_pixels},
        "detection": detection,
    }


def test_semantic_process_isolation_environment_is_explicit(monkeypatch):
    monkeypatch.delenv("MUJOCO_SEMANTIC_PROCESS_ISOLATION", raising=False)
    assert not _process_isolation_requested()
    monkeypatch.setenv("MUJOCO_SEMANTIC_PROCESS_ISOLATION", "1")
    assert _process_isolation_requested()
    monkeypatch.setenv("MUJOCO_SEMANTIC_PROCESS_ISOLATION", "false")
    assert not _process_isolation_requested()


def test_detector_vocabulary_is_configurable_and_broader_than_roles():
    vocabulary = detector_vocabulary(_config())
    assert "spoon" in vocabulary
    assert "marker" in vocabulary
    assert "bottle" in vocabulary


def test_detection_serializes_normalized_required_fields():
    record = _detection("fork", (1, 2, 30, 40)).to_dict()
    assert record["canonical_label"] == "fork"
    assert record["bbox_xyxy"] == [1, 2, 30, 40]
    assert record["inference_resolution"] == [100, 100]
    assert record["detector_name"] == "mock_detector"


def test_strong_box_mask_overlap_associates_correct_object():
    mask = np.zeros((100, 100), bool)
    mask[20:50, 30:60] = True
    result = associate_detections_to_masks(
        [_detection("fork", (29, 19, 61, 51))],
        {"object_0001": mask},
        _config(),
    )
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["object_id"] == "object_0001"


def test_weak_box_mask_overlap_produces_no_assignment():
    mask = np.zeros((100, 100), bool)
    mask[20:50, 30:60] = True
    result = associate_detections_to_masks(
        [_detection("fork", (80, 80, 95, 95))],
        {"object_0001": mask},
        _config(),
    )
    assert result["accepted"] == []
    assert result["unmatched_object_ids"] == ["object_0001"]


def test_ambiguous_overlap_is_rejected():
    mask = np.zeros((100, 100), bool)
    mask[20:50, 30:60] = True
    result = associate_detections_to_masks(
        [_detection("fork", (29, 19, 61, 51))],
        {"object_0001": mask, "object_0002": mask.copy()},
        _config(),
    )
    assert result["accepted"] == []
    assert {
        record["reason"] for record in result["rejected"]
    } == {"AMBIGUOUS_ASSOCIATION"}


def test_multiple_same_category_detections_are_not_collapsed():
    first = np.zeros((100, 100), bool)
    second = np.zeros((100, 100), bool)
    first[10:30, 10:30] = True
    second[60:85, 60:85] = True
    result = associate_detections_to_masks(
        [
            _detection("fork", (8, 8, 32, 32)),
            _detection("fork", (58, 58, 87, 87)),
        ],
        {"object_0001": first, "object_0002": second},
        _config(),
    )
    assert {
        record["object_id"] for record in result["accepted"]
    } == {"object_0001", "object_0002"}


def test_no_detection_fuses_to_unknown_not_false():
    record = fuse_semantic_observations(
        [],
        config=_config(),
        stage=0,
        region_id="INITIAL",
        detector_metadata={
            "name": "mock",
            "checkpoint": "mock",
            "version": "1",
        },
    )
    assert record["status"] == "UNKNOWN"
    assert record["canonical_label"] is None
    assert record["reason_codes"] == ["NO_ASSOCIATED_DETECTION"]


def test_multiview_agreement_produces_supported_label():
    config = _config()
    record = fuse_semantic_observations(
        [
            _accepted_observation("fork", "inspection_left"),
            _accepted_observation("fork", "inspection_right"),
        ],
        config=config,
        stage=2,
        region_id="D2",
        detector_metadata={
            "name": "mock",
            "checkpoint": "mock",
            "version": "1",
        },
    )
    assert record["status"] == "SUPPORTED"
    assert record["canonical_label"] == "fork"
    assert record["contributing_camera_ids"] == [
        "inspection_left",
        "inspection_right",
    ]


def test_multiview_consensus_outweighs_one_high_confidence_outlier():
    config = _config()
    config["fusion"][
        "winner_policy"
    ] = "supporting_views_then_weighted_score"
    record = fuse_semantic_observations(
        [
            _accepted_observation("fork", "inspection_left", 0.30),
            _accepted_observation("fork", "inspection_right", 0.31),
            _accepted_observation("fork", "inspection_top", 0.29),
            _accepted_observation("pen", "inspection_front", 0.95),
            _accepted_observation("pen", "inspection_close", 0.90),
        ],
        config=config,
        stage=0,
        region_id="INITIAL",
        detector_metadata={
            "name": "mock",
            "checkpoint": "mock",
            "version": "1",
        },
    )
    assert record["status"] == "SUPPORTED"
    assert record["canonical_label"] == "fork"
    assert record["supporting_view_margin"] == 1
    assert record["winning_label_margin_kind"] == "supporting_view_count"


def test_multiview_disagreement_is_retained_and_can_be_unknown():
    config = _config()
    config["fusion"]["minimum_supporting_views"] = 1
    config["fusion"]["minimum_winning_label_margin"] = 10.0
    record = fuse_semantic_observations(
        [
            _accepted_observation("fork", "inspection_left", 0.8),
            _accepted_observation("spoon", "inspection_right", 0.79),
        ],
        config=config,
        stage=2,
        region_id="D2",
        detector_metadata={
            "name": "mock",
            "checkpoint": "mock",
            "version": "1",
        },
    )
    assert record["status"] == "UNKNOWN"
    assert {item["label"] for item in record["alternatives"]} == {
        "fork",
        "spoon",
    }
    assert "CONFLICTING_MULTI_VIEW_LABELS" in record["reason_codes"]


def test_confident_excluded_label_is_semantic_false():
    role = {
        "semantic_preferences": [
            {
                "rank": 1,
                "canonical_label": "spoon",
                "detector_aliases": ["spoon"],
            }
        ]
    }
    result = evaluate_semantic_compatibility(
        {
            "semantics": {
                "validated": {
                    "status": "SUPPORTED",
                    "canonical_label": "marker",
                    "mean_confidence": 0.9,
                }
            }
        },
        role,
    )
    assert result["status"] == "FALSE"
    assert result["reason"] == "SUPPORTED_EXCLUDED_LABEL"


def test_unmatched_visible_instance_remains_semantically_unknown():
    role = {
        "semantic_preferences": [
            {
                "rank": 1,
                "canonical_label": "fork",
                "detector_aliases": ["fork"],
            }
        ]
    }
    assert evaluate_semantic_compatibility({}, role)["status"] == "UNKNOWN"


def test_semantic_record_contains_detector_and_rgb_provenance():
    record = fuse_semantic_observations(
        [
            _accepted_observation("fork", "inspection_left"),
            _accepted_observation("fork", "inspection_right"),
        ],
        config=_config(),
        stage=2,
        region_id="D2",
        detector_metadata={
            "name": "mock_detector",
            "checkpoint": "mock_checkpoint",
            "version": "1.2.3",
        },
    )
    assert record["observation_source"] == "RGB_DETECTOR"
    assert record["checkpoint"] == "mock_checkpoint"
    assert record["detector_version"] == "1.2.3"
    assert record["semantic_evidence_paths"]


class _SpyDetector:
    name = "spy"
    checkpoint = "spy.pt"
    version = "1"

    def __init__(self):
        self.calls = []

    def detect(self, image, vocabulary):
        self.calls.append((np.asarray(image).copy(), tuple(vocabulary)))
        return ()


def test_detector_receives_only_rgb_and_vocabulary_not_simulator_names(tmp_path):
    detector = _SpyDetector()
    mask = np.zeros((64, 64), bool)
    mask[20:40, 20:40] = True
    capture = SimpleNamespace(
        validation={"usable": True},
        rgb=np.zeros((64, 64, 3), np.uint8),
        instance_masks={"secret_simulator_body_name": mask},
    )
    inspection = SimpleNamespace(
        cameras={"inspection_left": capture}
    )
    config = _config()
    config["mask_crop"]["enabled"] = False
    config["fusion"]["minimum_supporting_views"] = 1
    result = run_semantic_inspection(
        inspection,
        accepted_instance_to_object_id={
            "secret_simulator_body_name": "object_0001"
        },
        detector=detector,
        config=config,
        stage=0,
        region_id="INITIAL",
        stage_dir=tmp_path,
        save_overlays=True,
    )
    assert len(detector.calls) == 1
    image, vocabulary = detector.calls[0]
    assert image.shape == (64, 64, 3)
    assert "secret_simulator_body_name" not in vocabulary
    serialized = (tmp_path / "semantics" / "detections.json").read_text()
    assert "secret_simulator_body_name" not in serialized
    assert (tmp_path / "semantic_overview.png").exists()
    camera_summary = result["camera_summaries"][0]
    assert camera_summary["camera_id"] == "inspection_left"
    assert camera_summary["inference_seconds"] >= 0.0
    assert camera_summary["full_frame_inference_seconds"] >= 0.0
    assert camera_summary["crop_inference_seconds"] == 0.0
    assert camera_summary["crop_inference_count"] == 0
