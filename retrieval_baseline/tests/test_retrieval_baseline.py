from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from retrieval_baseline.retrieval import RetrievalScores, assign_distinct
from retrieval_baseline.roles import LIVING_ROOM_ROLES


def test_role_phrases_contain_no_category_nouns():
    """Prompting with the answer would make this a label lookup, not retrieval."""
    banned = ("cup", "mug", "saucer", "remote", "table", "sofa", "chair", "plate")
    for role in LIVING_ROOM_ROLES:
        lowered = role.phrase.lower()
        found = [word for word in banned if word in lowered]
        assert not found, f"{role.key} phrase leaks category noun(s) {found}"


def test_assign_distinct_never_reuses_a_candidate():
    scores = RetrievalScores(by_phrase={"p": {"a": 0.9, "b": 0.8, "c": 0.7}})
    first = assign_distinct(scores, "p", 2, taken=())
    assert first == ["a", "b"]
    second = assign_distinct(scores, "p", 2, taken=first)
    assert second == ["c"]
    assert not set(first) & set(second)


def test_ranking_is_deterministic_under_ties():
    scores = RetrievalScores(by_phrase={"p": {"b": 0.5, "a": 0.5}})
    assert scores.ranking("p") == [("a", 0.5), ("b", 0.5)]


def test_crops_come_from_the_raw_unannotated_frame(tmp_path):
    """The annotated frame has aliases printed on it; CLIP reads text."""
    from retrieval_baseline.retrieval import _crops

    annotations = {
        "cameras": {
            "cam": {"regions": [{"id": "region_0001", "bbox_xyxy": [10, 10, 60, 60]}]}
        }
    }
    Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8)).save(tmp_path / "raw_cam.png")
    got = list(_crops(annotations, tmp_path, "region"))
    assert [row[0] for row in got] == ["region_0001"]

    # With only the annotated frame present, nothing is scored.
    (tmp_path / "raw_cam.png").unlink()
    Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8)).save(tmp_path / "cam.png")
    assert list(_crops(annotations, tmp_path, "region")) == []


def test_baseline_declares_it_uses_no_language_model():
    source = (
        __import__("pathlib").Path("retrieval_baseline/run_living_room.py")
        .read_text(encoding="utf-8")
    )
    assert '"uses_language_model": False' in source
    assert '"raw_vlm_requests": 0' in source
