from pathlib import Path

from mujoco_scenes.baselines.vilain_tamp.production_execution import (
    BenchmarkRegistryHiddenContextProvider,
)


def test_kitchen_hidden_context_uses_benchmark_physical_vocabulary() -> None:
    config_root = Path(__file__).resolve().parents[3] / "configs"
    context = BenchmarkRegistryHiddenContextProvider(config_root).load(
        "kitchen", "F0_ALL_VISIBLE"
    )

    assert context.ground_truth_feasibility is True
    assert context.requirements["coffee_vessels"] == [
        "ab3_narrow_deep_cup",
        "ab3_medium_deep_mug",
    ]
    assert context.requirements["water_sources"] == ["s1i_compact_kettle"]
    assert context.requirements["suitable_stirrers"] == [
        "s1i_final_long_narrow_spoon"
    ]
    assert not any(
        str(value).startswith("missing-")
        for values in context.requirements.values()
        for value in (values if isinstance(values, list) else [values])
    )


def test_kitchen_hidden_context_marks_predetermined_infeasible_variant() -> None:
    config_root = Path(__file__).resolve().parents[3] / "configs"
    context = BenchmarkRegistryHiddenContextProvider(config_root).load(
        "kitchen", "I0_MISSING_COFFEE_VESSEL"
    )

    assert context.ground_truth_feasibility is False
    assert context.requirements["coffee_vessels"] == [
        "ab3_narrow_deep_cup",
        "ab3_medium_deep_mug",
    ]
