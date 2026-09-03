"""The ablation runner must evaluate the condition it labels a row with."""

from __future__ import annotations

import json

from mujoco_scenes import run_gt_evidence_ablation as runner


def _install_stubs(monkeypatch, recorded):
    monkeypatch.setattr(runner, "_variants", lambda domain: ("V1",))
    monkeypatch.setattr(runner, "GTSpecProvider", lambda: _Provider())
    monkeypatch.setattr(runner, "build_oracle_graph", lambda *a, **k: object())

    def fake_evaluate_one(domain, variant, mode, *, evidence_components=None, **_):
        recorded.append((mode, evidence_components))
        return {
            "domain": domain,
            "variant": variant,
            "evidence_mode": mode,
            "evidence_components": sorted(
                runner.resolve_evidence_components(mode, evidence_components)
            ),
            "intended_outcome": "FEASIBLE",
            "predicted_outcome": "FEASIBLE",
            "outcome_correct": True,
            "grounding_complete": True,
            "ground_truth_valid_complete": True,
            "runtime_ms": 0.0,
            "role_slots_selected": 0,
            "operation_bindings_total": 0,
            "semantic_valid_role_slots": 0,
            "geometric_valid_role_slots": 0,
            "ground_truth_valid_operation_bindings": 0,
        }

    monkeypatch.setattr(runner, "evaluate_one", fake_evaluate_one)


class _Provider:
    def provide(self, domain, task):
        return object()


def test_evidence_modes_are_not_all_evaluated_as_joint(tmp_path, monkeypatch):
    recorded: list[tuple[str, tuple[str, ...] | None]] = []
    _install_stubs(monkeypatch, recorded)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gt_evidence_ablation",
            "--domains", "kitchen",
            "--output-root", str(tmp_path / "modes"),
        ],
    )

    assert runner.main() == 0

    assert [mode for mode, _ in recorded] == list(runner.MODES)
    assert all(components is None for _, components in recorded)
    rows = json.loads((tmp_path / "modes" / "results.json").read_text())
    assert {row["condition"]: row["evidence_components"] for row in rows} == {
        "semantic_only": ["semantic"],
        "geometric_only": ["binary", "unary"],
        "joint": ["binary", "semantic", "unary"],
    }


def test_component_masks_are_evaluated_with_an_explicit_mask(tmp_path, monkeypatch):
    recorded: list[tuple[str, tuple[str, ...] | None]] = []
    _install_stubs(monkeypatch, recorded)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gt_evidence_ablation",
            "--domains", "kitchen",
            "--component-masks", "no_binary",
            "--output-root", str(tmp_path / "masks"),
        ],
    )

    assert runner.main() == 0

    assert recorded == [("joint", ("semantic", "unary"))]
