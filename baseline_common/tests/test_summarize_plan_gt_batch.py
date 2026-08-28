from baseline_common.summarize_plan_gt_batch import _metrics


def test_metrics_score_sequence_not_outcome_classification():
    payload = {
        "gt_comparison": {
            "outcome_match": True,
            "shared_task_vocabulary": {
                "exact_sequence_match": False,
                "ordered_f1": 0.625,
            },
        }
    }

    assert _metrics(payload) == (0.0, 0.625)


def test_metrics_support_living_room_comparison_shape():
    payload = {
        "gt_comparison": {
            "outcome_match": False,
            "exact_sequence_match": True,
            "ordered_f1": 1.0,
        }
    }

    assert _metrics(payload) == (1.0, 1.0)
