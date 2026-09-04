from __future__ import annotations

import ast
from pathlib import Path
import re


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_FILES = tuple(
    path
    for path in PACKAGE_ROOT.rglob("*.py")
    if "tests" not in path.relative_to(PACKAGE_ROOT).parts
)
FORBIDDEN_IMPORT_PARTS = ("functional_tamp_pipeline", "phase4_")
FORBIDDEN_SOURCE_PATTERNS = {
    "method-specific package": re.compile(r"\bfunctional_tamp_pipeline\b"),
    "method-specific execution prefix": re.compile(r"\bphase4_"),
    "ground graph entry point": re.compile(r"\bground_graph\b"),
    "graph result type": re.compile(r"\bGraphGroundingResult\b"),
    "functional graph symbol": re.compile(r"\bG_F\b"),
    "observation graph symbol": re.compile(r"\bG_O\b"),
    "assignment symbol": re.compile(r"\bphi(?:\*)?\b", re.IGNORECASE),
    "handoff type": re.compile(r"\bPhase3Handoff\b"),
    "graph result artifact": re.compile(r"\bgraph_grounding_result\b"),
    "planning audit artifact": re.compile(r"\bplan_grounding_audit\b"),
    "functional witness": re.compile(
        r"\b(?:functional|canonical)(?:_|\s+)(?:grounding(?:_|\s+))?witness(?:es)?\b",
        re.IGNORECASE,
    ),
}


def test_production_imports_do_not_cross_method_boundary() -> None:
    violations: list[str] = []
    for path in PRODUCTION_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(part in name for part in FORBIDDEN_IMPORT_PARTS):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {name}")
    assert not violations, "forbidden imports:\n" + "\n".join(violations)


def test_production_source_has_no_proposed_method_references() -> None:
    violations: list[str] = []
    for path in PRODUCTION_FILES:
        source = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
            if pattern.search(source):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {label}")
    assert not violations, "forbidden source references:\n" + "\n".join(violations)


def test_stage_one_production_has_no_simulator_imports() -> None:
    for path in PRODUCTION_FILES:
        source = path.read_text(encoding="utf-8")
        assert "import mujoco" not in source
