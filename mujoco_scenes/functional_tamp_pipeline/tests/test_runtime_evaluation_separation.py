"""The online path may not reach the code that knows the answers.

Scoring knows the feasibility label, the reference graph and the expected action
sequence.  An online module that can import it can reach all of that, whatever
it does with it today, so the boundary is enforced structurally rather than by
review.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
AUDIT = REPO / "scripts" / "audit_no_gt_leakage.py"


def _audit_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gt_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_audit_reports_nothing_on_the_current_tree():
    result = subprocess.run([sys.executable, str(AUDIT)], capture_output=True,
                            text=True, cwd=str(REPO))
    assert "FINDINGS: 0" in result.stdout, result.stdout[-3000:]


def test_the_scorer_is_not_importable_from_the_online_path():
    """The specific module this project scores with."""
    module = _audit_module()
    assert "evaluation_outcome" in module.EVALUATION_ONLY_MODULES
    for path in module.sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                assert "evaluation_outcome" not in str(name), (
                    f"{path.relative_to(REPO)} imports the GT scorer")


def test_the_import_rule_actually_catches_a_violation(tmp_path):
    """A rule that cannot fail is not a rule.

    Plants an online-looking module that imports the scorer and checks the audit
    reports it, so the clean result above means something.
    """
    module = _audit_module()
    offender = tmp_path / "offender.py"
    offender.write_text(
        "from mujoco_scenes.evaluation_outcome import outcome_is_correct\n"
        "def decide():\n    return outcome_is_correct\n")
    tree = ast.parse(offender.read_text())
    # The audit reports paths relative to the repo, so evaluate the rule directly.
    findings = module.evaluation_imports(REPO / "mujoco_scenes" / "offender.py", tree)
    assert findings, "the audit does not catch an online import of the scorer"
    assert "evaluation_outcome" in findings[0]


def test_each_violation_is_reported_once(tmp_path):
    """A from-import used to be counted twice, once for the module and once per name."""
    module = _audit_module()
    tree = ast.parse("from mujoco_scenes.evaluation_outcome import a, b, c\n")
    findings = module.evaluation_imports(REPO / "mujoco_scenes" / "x.py", tree)
    assert len(findings) == 1, findings


def test_the_only_allowance_is_the_oracle_mode_dispatcher():
    """The exception is narrow, named, and cannot silently grow."""
    module = _audit_module()
    assert module.EVALUATION_IMPORT_ALLOWANCES == {
        ("mujoco_scenes/functional_tamp_pipeline/spec_provider.py", "gt_spec_provider"),
    }


def test_the_oracle_import_stays_lazy_and_mode_guarded():
    """If it became a module-level import the VLM path would execute it."""
    source = (REPO / "mujoco_scenes" / "functional_tamp_pipeline" / "spec_provider.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            assert "gt_spec_provider" not in text, (
                "the oracle provider is imported at module level; it must stay "
                "inside the mode-guarded branch")
    assert 'if mode == "gt"' in source
