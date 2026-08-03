import json

from mujoco_scenes.generate_pairing_strategy_report import (
    generate_pairing_strategy_report,
)


def _run(tmp_path, name, strategy, checks, skipped, elapsed):
    run = tmp_path / name
    stage = run / "stages" / "000_initial"
    stage.mkdir(parents=True)
    payload = {
        "stage": 0,
        "region_id": "INITIAL",
        "pairing_strategy": strategy,
        "observed_object_ids": ["object_0001", "object_0002"],
        "ordered_distinct_object_pair_count": 2,
        "relation_evaluation_count": checks,
        "skipped_relation_pair_count": skipped,
        "elapsed_seconds": elapsed,
    }
    (stage / "pair_relation_evaluations.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return run


def test_pairing_report_compares_executed_checks_and_time(tmp_path):
    exhaustive = _run(
        tmp_path, "all", "exhaustive_all_pairs", 4, 0, 0.004
    )
    scoped = _run(
        tmp_path, "scoped", "semantic_role_scoped", 1, 3, 0.001
    )
    output = tmp_path / "report"
    generate_pairing_strategy_report(exhaustive, scoped, output)
    report = json.loads(
        (output / "pairing_strategy_ablation.json").read_text()
    )
    assert report["comparison"]["relation_check_reduction_percent"] == 75.0
    assert report["comparison"]["binary_evaluation_speedup"] == 4.0
    assert (output / "pairing_strategy_ablation.png").exists()
    assert (output / "pairing_strategy_ablation.html").exists()
