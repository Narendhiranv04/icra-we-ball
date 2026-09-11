"""The online path may not consult the reference, nor know which variant it is.

Two privileged instruments exist on purpose and each is checked here rather
than trusted: the variant-keyed oracle search order, which the contract must
refuse outright in vlm mode, and the privileged perception evaluator, which
must not be reachable from the pipeline package at all.

The static audit is the third check, run as a subprocess so its report is the
same artifact the closure report carries.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import SearchRegionContractError
from mujoco_scenes.functional_tamp_pipeline.search_contract import (
    ORACLE_SEARCH_ORDERS,
    freeze_search_region_contract,
    validate_search_order_preflight,
)


REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent.parent


def test_the_static_audit_reports_nothing():
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "audit_no_gt_leakage.py")],
                         env=env, cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert "FINDINGS: 0" in out.stdout, out.stdout[-4000:]
    assert out.returncode == 0, out.stdout[-4000:]


@pytest.mark.parametrize("domain,variant", [("kitchen", "K1"), ("workshop", "W5")])
def test_the_variant_keyed_oracle_is_refused_in_vlm_mode(domain, variant):
    """The table exists and is a benchmark answer; vlm mode may not reach it."""
    assert variant in ORACLE_SEARCH_ORDERS[domain]
    with pytest.raises(SearchRegionContractError, match="privileged"):
        validate_search_order_preflight(domain, "oracle", mode="vlm")
    with pytest.raises(SearchRegionContractError, match="privileged"):
        freeze_search_region_contract(None, domain, "oracle", mode="vlm", variant=variant)


def test_the_privileged_perception_evaluator_is_not_reachable_from_the_pipeline():
    """It matches detector tracks against simulator bodies; nothing may read it."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "evaluation" == str(node.module or "").rsplit(".", 1)[-1]:
                offenders.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("workshop_phase1.evaluation"):
                        offenders.append(f"{path.name}:{node.lineno}: import {alias.name}")
            for name in (getattr(node, "attr", None), getattr(node, "id", None)):
                if name == "PrivilegedPhase1Evaluator":
                    offenders.append(f"{path.name}:{getattr(node, 'lineno', 0)}: {name}")
    assert not offenders, offenders
