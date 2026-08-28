from pathlib import Path


ROOT = Path(__file__).parents[2]


def _production_sources(folder: str):
    return (
        path
        for path in (ROOT / folder).glob("*.py")
        if path.name != "__init__.py"
    )


def test_baseline_implementations_do_not_import_each_other():
    for path in _production_sources("llm3_baseline"):
        assert "vlm_tamp_baseline" not in path.read_text(encoding="utf-8")
    for path in _production_sources("vlm_tamp_baseline"):
        assert "llm3_baseline" not in path.read_text(encoding="utf-8")


def test_neutral_layer_contains_no_baseline_policy_imports():
    for path in _production_sources("baseline_common"):
        source = path.read_text(encoding="utf-8")
        assert "llm3_baseline" not in source
        assert "vlm_tamp_baseline" not in source
