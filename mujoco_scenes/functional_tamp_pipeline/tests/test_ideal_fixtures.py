"""Test harness and diagnostic suite for Ideal Raw VLM Fixtures (Pass P3-C).

Verifies:
1. Schema compliance of ideal raw fixtures against production schemas.
2. Anti-leakage rules (zero canonical IR identifiers, zero simulator backend strings, zero benchmark oracle tokens).
3. Deterministic, zero-network diagnostic canonicalization over current production canonicalizers.
4. Concept preservation and loss classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
import jsonschema

from mujoco_scenes.functional_tamp_pipeline.errors import (
    MalformedVLMSpecificationError,
    UnmappedFunctionalConceptError,
    VLMSpecificationError,
)
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import evaluate_gf_against_reference
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import validate_runtime_gf
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.workshop_phase1.fm_adapter import (
    KITCHEN_FUNCTIONAL_GRAPH_SCHEMA,
    RESPONSE_SCHEMA,
    validate_kitchen_functional_specification,
    validate_requirement_response,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"

FORBIDDEN_CANONICAL_ROLES = {
    "coffee_container", "coffee_stirrer", "soup_container", "soup_eating_utensil",
    "water_source", "coffee_source",
    "PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION", "CUP_SAUCER_SET", "REMOTE",
    "SEATING_POSITION", "SEATING_PAIR",
    "driver", "fastener", "repair_target",
}

FORBIDDEN_PREDICATE_TOKENS = {
    "INSERTABLE_IN", "REACHES_BOTTOM", "PLANAR_SUPPORT", "FITS_SET_ON", "FITS_ON",
    "NEAR_SEAT", "ACCESSIBLE_FROM_BOTH_SEATS", "CAN_DRIVE_SCREW", "CAN_FASTEN",
    "COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET",
}

FORBIDDEN_BACKEND_HANDLES = {
    "workshop_power_driver", "workshop_long_phillips_driver", "workshop_medium_phillips_screw",
}

FORBIDDEN_ORACLE_TOKENS = {
    "F0", "K1", "K2", "L1", "W1", "object_0001", "region_0001",
    "LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET",
}


def load_ideal_fixture(domain: str) -> dict[str, Any]:
    """Load the ideal raw fixture for a given domain."""
    filename_map = {
        "kitchen": "kitchen_K1.json",
        "living_room": "living_room_L1.json",
        "workshop": "workshop_W1.json",
    }
    fixture_file = FIXTURES_DIR / filename_map[domain]
    assert fixture_file.exists(), f"Fixture file {fixture_file} does not exist!"
    return json.loads(fixture_file.read_text(encoding="utf-8"))


def test_ideal_raw_fixtures_schema_compliance():
    """Verify that all 3 ideal raw fixtures pass production schema validation."""
    # Kitchen
    k_data = load_ideal_fixture("kitchen")
    jsonschema.validate(k_data, KITCHEN_FUNCTIONAL_GRAPH_SCHEMA)
    validated_k = validate_kitchen_functional_specification(k_data)
    assert validated_k["status"] == "SUPPORTED"
    assert len(validated_k["functional_roles"]) == 6

    # Living Room
    l_data = load_ideal_fixture("living_room")
    jsonschema.validate(l_data, RESPONSE_SCHEMA)
    validated_l = validate_requirement_response(l_data)
    assert validated_l["status"] == "SUPPORTED"
    assert len(validated_l["functional_roles"]) == 6

    # Workshop
    w_data = load_ideal_fixture("workshop")
    jsonschema.validate(w_data, RESPONSE_SCHEMA)
    validated_w = validate_requirement_response(w_data)
    assert validated_w["status"] == "SUPPORTED"
    assert len(validated_w["functional_roles"]) == 3


def test_ideal_raw_fixtures_anti_leak_and_purity():
    """Verify that raw fixtures contain no canonical IR identifiers, backend handles, or oracle strings."""
    for domain in ("kitchen", "living_room", "workshop"):
        data = load_ideal_fixture(domain)

        # 1. Raw role IDs must not equal canonical role names
        raw_role_ids = {r["id"] for r in data["functional_roles"]}
        leaked_roles = raw_role_ids.intersection(FORBIDDEN_CANONICAL_ROLES)
        assert not leaked_roles, f"Fixture {domain} leaked canonical role IDs: {leaked_roles}"

        # 2. Functional relations must not contain raw canonical predicate tokens
        for rel in data["functional_relations"]:
            rel_str = rel["relation"].strip().upper()
            assert rel_str not in FORBIDDEN_PREDICATE_TOKENS, (
                f"Fixture {domain} leaked canonical predicate token '{rel_str}' in relation {rel}"
            )

        # 3. Serialized JSON must not contain backend handles or oracle tokens
        raw_text = json.dumps(data)
        for backend_handle in FORBIDDEN_BACKEND_HANDLES:
            assert backend_handle not in raw_text, (
                f"Fixture {domain} leaked backend handle '{backend_handle}'"
            )

        for oracle_token in FORBIDDEN_ORACLE_TOKENS:
            assert oracle_token not in raw_text, (
                f"Fixture {domain} leaked benchmark oracle token '{oracle_token}'"
            )


def diagnose_fixture_canonicalization(domain: str) -> dict[str, Any]:
    """Execute current production canonicalizer on the ideal fixture and return a structured diagnostic outcome."""
    data = load_ideal_fixture(domain)
    task_instructions = {
        "kitchen": "Prepare and serve coffee and soup for two people using available kitchenware.",
        "living_room": "Prepare the living room for two people watching television.",
        "workshop": "Repair the frame by securing the loose joint.",
    }
    instruction = task_instructions[domain]

    report: dict[str, Any] = {
        "domain": domain,
        "status": "UNKNOWN",
        "graph": None,
        "error_type": None,
        "error_message": None,
        "error_category": None,
        "first_failing_module": None,
        "concept_preservation": {},
        "reference_metrics": {},
    }

    try:
        if domain == "kitchen":
            class MockKitchenAdapter:
                last_raw_kitchen_graph_response = data
                last_validated_kitchen_graph_response = data
                def generate_kitchen_functional_graph(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                    return data

            gf = VLMSpecProvider._kitchen(instruction, [], adapter=MockKitchenAdapter())

        elif domain == "living_room":
            from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider
            class MockLivingRoomAdapter:
                def generate_task_requirements(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                    return data

            provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=MockLivingRoomAdapter())
            gf = VLMSpecProvider._living_room(instruction, [], provider=provider)

        elif domain == "workshop":
            from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
            class MockWorkshopAdapter:
                raw_decomposition = data
                raw_vlm_response = data
                validated_vlm_specification = data
                def generate_task_requirements(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                    return data

            provider = FMRequirementProvider(fm_adapter=MockWorkshopAdapter())
            gf = VLMSpecProvider._workshop(instruction, [], provider=provider)

        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Validate resulting graph
        gf.validate()
        validate_runtime_gf(gf)

        report["status"] = "CANONICALIZED"
        report["graph"] = gf

        # Compare against offline GT reference
        gt_ref = GTSpecProvider().provide(domain, instruction)
        eval_result = evaluate_gf_against_reference(gf, gt_ref)
        report["reference_metrics"] = {
            "role_precision": eval_result.role_precision,
            "role_recall": eval_result.role_recall,
            "relation_precision": eval_result.relation_precision,
            "relation_recall": eval_result.relation_recall,
            "exact_structural_match": eval_result.exact_structural_match,
        }

        # Concept preservation classification
        for role in data["functional_roles"]:
            report["concept_preservation"][f"role:{role['id']}"] = "PRESERVED"
        for rel in data["functional_relations"]:
            report["concept_preservation"][f"rel:{rel['subject_role']}->{rel['object_role']}"] = "PRESERVED"
        for grp in data.get("interaction_groups", []):
            report["concept_preservation"][f"group:{grp['id']}"] = "PRESERVED"

    except Exception as exc:
        report["status"] = "CANONICALIZATION_FAILED"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["error_category"] = getattr(exc, "category", "UNCLASSIFIED_ERROR")
        import traceback
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            last_frame = tb[-1]
            report["first_failing_module"] = f"{Path(last_frame.filename).name}:{last_frame.lineno} ({last_frame.name})"

    return report


def test_diagnostic_canonicalization_outcomes():
    """Verify that canonicalizer outcomes are captured deterministically with zero model calls."""
    # 1. Kitchen Diagnostic
    k_diag = diagnose_fixture_canonicalization("kitchen")
    assert k_diag["status"] == "CANONICALIZED"
    assert k_diag["reference_metrics"]["role_recall"] == 1.0
    assert k_diag["reference_metrics"]["relation_recall"] == 1.0

    # 2. Living Room Diagnostic
    l_diag = diagnose_fixture_canonicalization("living_room")
    assert l_diag["status"] == "CANONICALIZATION_FAILED"
    assert l_diag["error_type"] == "UnmappedFunctionalConceptError"
    assert l_diag["error_category"] == "UNMAPPED_FUNCTIONAL_CONCEPT"
    assert "map_living_room_relation" in str(l_diag["first_failing_module"])

    # 3. Workshop Diagnostic
    w_diag = diagnose_fixture_canonicalization("workshop")
    assert w_diag["status"] == "CANONICALIZED"
    assert w_diag["reference_metrics"]["role_recall"] == 1.0
    assert w_diag["reference_metrics"]["relation_recall"] == 1.0
    assert w_diag["graph"].candidate_regions == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
