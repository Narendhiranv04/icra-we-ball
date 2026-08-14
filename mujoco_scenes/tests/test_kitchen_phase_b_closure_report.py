from mujoco_scenes.generate_kitchen_phase_b_closure_report import (
    CARRY_EVIDENCE,
    closure_from_artifacts,
    git_provenance,
    guards_are_verifiable,
    normalized_assignment_ids,
)


REQUIRED = {
    "frozen_phase1_input_integrity_pass": True,
    "frozen_phase2_input_integrity_pass": True,
    "execution_scene_calibration_audit_pass": True,
    "no_runtime_functional_substitution_pass": True,
    "primary_b1_corrected_layout_repeatability_pass": True,
    "pick_coverage_pass": True,
    "place_coverage_pass": True,
    "carried_move_pass": True,
    "storage_repeatability_pass": True,
    "c2_unrestricted_grasp_pass": True,
    "entity_resolution_pass": True,
    "extraction_pass": True,
    "destination_coverage_pass": True,
    "variant_coverage_pass": True,
    "isolated_operator_coverage_pass": True,
    "scientific_guards_pass": True,
    "multi_object_pass": True,
    "final_relation_pass": True,
    "tests_pass": True,
    "reproduction_manifest_valid": True,
}


def test_pick_success_cannot_close_phase_b_when_place_fails():
    evidence = dict(REQUIRED, place_coverage_pass=False)
    assert closure_from_artifacts(evidence)["phase_b_closed"] is False


def test_missing_carried_move_cannot_close_phase_b():
    evidence = dict(REQUIRED, carried_move_pass=False)
    assert closure_from_artifacts(evidence)["phase_b_closed"] is False


def test_missing_production_operator_coverage_cannot_close_phase_b():
    evidence = dict(REQUIRED, isolated_operator_coverage_pass=False)
    assert closure_from_artifacts(evidence)["phase_b_closed"] is False


def test_missing_artifact_gate_defaults_to_false():
    evidence = dict(REQUIRED)
    del evidence["multi_object_pass"]
    result = closure_from_artifacts(evidence)
    assert result["phase_b_closed"] is False
    assert "multi_object_pass" in result["missing_or_failed"]


def test_c2_fixture_active_during_contact_cannot_close_phase_b():
    evidence = dict(REQUIRED, c2_unrestricted_grasp_pass=False)
    assert closure_from_artifacts(evidence)["phase_b_closed"] is False


def test_phase_c_is_not_a_phase_b_gate():
    # POUR/STIR support deliberately does not appear in the Phase-B closure
    # inputs; explicit non-fabrication is covered by the operator audit.
    assert closure_from_artifacts(REQUIRED)["phase_b_closed"] is True


def test_bare_scientific_guard_is_unverifiable():
    assert not guards_are_verifiable({"claimed": {"passed": True}})
    assert guards_are_verifiable({
        "measured": {
            "passed": True,
            "validation_method": "compare telemetry",
            "evidence": ["artifact.json"],
        }
    })


def test_canonical_jar_carry_evidence_path_matches_reproduction_output():
    assert CARRY_EVIDENCE["JAR_SOURCE"] == (
        "runs/phaseB_freeze_carried_move/JAR_SOURCE/carried_move_result.json"
    )


def test_git_provenance_reports_real_head_and_cleanliness():
    provenance = git_provenance()
    assert len(provenance["git_head"]) == 40
    assert provenance["worktree_status"] in {"clean", "dirty"}
    assert provenance["worktree_dirty"] == bool(
        provenance["worktree_status_short"]
    )


def test_missing_legacy_sources_remain_explicitly_empty():
    legacy = normalized_assignment_ids({"coffee": [], "soup": []})
    current = normalized_assignment_ids({
        "coffee_targets": [],
        "coffee_stirring": [],
        "soup_targets": [],
        "soup_serving": [],
        "source_roles": {"water_source": "object_1"},
    })
    assert legacy["sources"] == {}
    assert current["sources"] == {"water_source": "object_1"}
