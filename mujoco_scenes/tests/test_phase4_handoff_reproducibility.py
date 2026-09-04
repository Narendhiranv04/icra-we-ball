"""Fresh-clone reproducibility smoke tests for Phase-4 handoff generation.

Verifies deterministic reconstruction of Phase-3 GT handoffs from tracked inputs
and ensures symbolic plan constraints without launching any MuJoCo simulation.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from mujoco_scenes.phase4_execution import load_phase3_handoff
from mujoco_scenes.prepare_phase4_handoff import (
    prepare_kitchen_gt_handoff,
    prepare_phase4_handoff,
)


REQUIRED_ARTIFACT_FILES = (
    "run_manifest.json",
    "result.json",
    "graph_grounding_result.json",
    "plan_grounding_audit.json",
    "observed_scene_graph.json",
    "functional_specification.json",
    "action_sequence/action_plan.json",
    "observed_search/phase1/object_registry.json",
)


def _find_action_step(actions: list[dict], op: str, args: list[str]) -> int:
    for idx, act in enumerate(actions):
        if act.get("operator") == op and list(act.get("arguments", [])) == args:
            return idx
    raise ValueError(f"Action {op}({args}) not found in plan")


def test_k4_fresh_clone_reproducibility(tmp_path: Path):
    """Test K4 GT handoff generation and verify Fix 1 B1-bowl relocation ordering."""
    handoff = prepare_kitchen_gt_handoff("K4", output_root=tmp_path)
    assert handoff.domain == "kitchen"
    assert handoff.variant == "K4"
    assert len(handoff.actions) == 26

    # Verify all mandatory Phase-4 execution handoff artifacts were written
    run_dir = tmp_path / "kitchen" / "K4" / "gt"
    for rel_path in REQUIRED_ARTIFACT_FILES:
        assert (run_dir / rel_path).is_file(), f"Missing artifact {rel_path}"

    # Verify fail-closed validation passes
    reloaded = load_phase3_handoff(run_dir)
    assert len(reloaded.actions) == 26

    # In K4: object_0009 is the bowl starting in B1; object_0005 is its assigned spoon.
    # The symbolic sequence must be:
    # PICK(object_0009) -> PLACE(object_0009, serving_area) -> later PICK(object_0005) -> PLACE(object_0005, object_0009)
    step_pick_bowl = _find_action_step(reloaded.actions, "PICK", ["object_0009"])
    step_place_bowl_served = _find_action_step(reloaded.actions, "PLACE", ["object_0009", "serving_area"])
    step_pick_spoon = _find_action_step(reloaded.actions, "PICK", ["object_0005"])
    step_place_spoon_in_bowl = _find_action_step(reloaded.actions, "PLACE", ["object_0005", "object_0009"])

    assert step_pick_bowl < step_place_bowl_served, "Bowl must be picked before being served"
    assert step_place_bowl_served < step_pick_spoon, "B1 bowl must reach serving_area before spoon is picked"
    assert step_pick_spoon < step_place_spoon_in_bowl, "Spoon must be picked before placing in bowl"


def test_k6_fresh_clone_reproducibility(tmp_path: Path):
    """Test K6 GT handoff generation and verify composite drawer + B1 bowl sequencing."""
    handoff = prepare_kitchen_gt_handoff("K6", output_root=tmp_path)
    assert handoff.domain == "kitchen"
    assert handoff.variant == "K6"
    assert len(handoff.actions) == 26

    run_dir = tmp_path / "kitchen" / "K6" / "gt"
    for rel_path in REQUIRED_ARTIFACT_FILES:
        assert (run_dir / rel_path).is_file(), f"Missing artifact {rel_path}"

    reloaded = load_phase3_handoff(run_dir)
    assert len(reloaded.actions) == 26

    # In K6:
    # 1. Countertop bowl branch: object_0002 is on counter; drawer spoon object_0005 placed in it, then served.
    step_place_c_spoon = _find_action_step(reloaded.actions, "PLACE", ["object_0005", "object_0002"])
    step_place_c_bowl_served = _find_action_step(reloaded.actions, "PLACE", ["object_0002", "serving_area"])
    assert step_place_c_spoon < step_place_c_bowl_served, "Countertop bowl receives spoon then is served"

    # 2. Box B1 bowl branch: object_0008 starts in B1; must reach serving_area before drawer spoon object_0006 is placed.
    step_pick_b1_bowl = _find_action_step(reloaded.actions, "PICK", ["object_0008"])
    step_place_b1_bowl_served = _find_action_step(reloaded.actions, "PLACE", ["object_0008", "serving_area"])
    step_pick_b1_spoon = _find_action_step(reloaded.actions, "PICK", ["object_0006"])
    step_place_b1_spoon_in_bowl = _find_action_step(reloaded.actions, "PLACE", ["object_0006", "object_0008"])

    assert step_pick_b1_bowl < step_place_b1_bowl_served
    assert step_place_b1_bowl_served < step_pick_b1_spoon
    assert step_pick_b1_spoon < step_place_b1_spoon_in_bowl


@pytest.mark.parametrize("variant,expected_len", [
    ("K1", 24),
    ("K2", 26),
    ("K3", 24),
    ("K5", 24),
])
def test_k1_k2_k3_k5_preservation(tmp_path: Path, variant: str, expected_len: int):
    """Verify K1-K3 and K5 plans are preserved without unintended modifications."""
    handoff = prepare_kitchen_gt_handoff(variant, output_root=tmp_path)
    assert len(handoff.actions) == expected_len

    run_dir = tmp_path / "kitchen" / variant / "gt"
    reloaded = load_phase3_handoff(run_dir)
    assert len(reloaded.actions) == expected_len


def test_prepare_phase4_handoff_cli_batch(tmp_path: Path):
    """Verify batch preparation of K4,K5,K6 via prepare_phase4_handoff."""
    results = prepare_phase4_handoff(
        domain="kitchen",
        variant="K4,K5,K6",
        mode="gt",
        output_root=tmp_path,
    )
    assert len(results) == 3
    variants_prepared = [h.variant for h in results]
    assert variants_prepared == ["K4", "K5", "K6"]
