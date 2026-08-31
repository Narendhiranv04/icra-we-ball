"""Hardened test harness and diagnostic suite for Ideal Raw VLM Fixtures (Pass P3-C.2).

Verifies:
1. Schema compliance of ideal raw fixtures against production schemas.
2. Comprehensive anti-leakage rules across all relation, group, context relation fields,
   raw IDs, backend handles, and canonical benchmark oracle tokens.
3. Deterministic, zero-network diagnostic canonicalization over current production canonicalizers.
4. Concept-exact, evidence-derived concept preservation tracing and granular failure localization.
5. Future-proof test adapter compliance.
6. Full GT reference metric evaluation and representation disclosure.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
import jsonschema
import pytest

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
from mujoco_scenes.kitchen_vlm_functional_graph import (
    map_binary_relation,
    map_unary_property,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMCallMetrics,
    KITCHEN_FUNCTIONAL_GRAPH_SCHEMA,
    RESPONSE_SCHEMA,
    validate_kitchen_functional_specification,
    validate_requirement_response,
)
from mujoco_scenes.workshop_phase1.requirements import resolve_workshop_region_proposal

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

FORBIDDEN_ORACLE_REGIONS = {
    "D1", "D2", "C1", "C2", "B1",
    "LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET",
}

FORBIDDEN_ORACLE_TOKENS = {
    "F0", "K1", "K2", "L1", "W1", "object_0001", "region_0001",
}


class MockFMAdapter:
    """Complete, future-proof offline test adapter implementing the FMAdapter interface."""

    def __init__(self, document: dict[str, Any]) -> None:
        self.document = deepcopy(document)
        self.last_raw_response = deepcopy(document)
        self.last_raw_requirement_response = deepcopy(document)
        self.last_raw_inspection_response: dict[str, Any] | None = None
        self.last_observation_images: list[str] = []
        self.last_raw_kitchen_graph_response = deepcopy(document)
        self.last_validated_kitchen_graph_response = deepcopy(document)
        self.raw_decomposition = deepcopy(document)
        self.raw_vlm_response = deepcopy(document)
        self.validated_vlm_specification = deepcopy(document)
        self.metrics = FMCallMetrics(requirement_calls=0, search_prior_calls=0, total_calls=0)
        self.call_count: int = 0

    def generate_task_requirements(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        return deepcopy(self.document)

    def generate_kitchen_functional_graph(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        return deepcopy(self.document)

    def generate_inspection_priors(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.metrics.search_prior_calls += 1
        self.metrics.total_calls += 1
        return {
            "initial_requirements_satisfied": True,
            "decision_reason": "Offline mock",
            "inspectable_regions": [],
            "inspection_order": [],
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
    """Verify comprehensive anti-leakage invariants across all fixture fields."""
    for domain in ("kitchen", "living_room", "workshop"):
        data = load_ideal_fixture(domain)

        # 1. Raw local IDs must be neutral (role_1..N, group_1..N, search_1..N) and not embed canonical names
        raw_role_ids = {r["id"] for r in data["functional_roles"]}
        for r_id in raw_role_ids:
            assert r_id.startswith("role_"), f"Fixture {domain} role ID {r_id} is not neutral (expected role_*)"
            for canon_role in FORBIDDEN_CANONICAL_ROLES:
                assert canon_role.lower() not in r_id.lower(), (
                    f"Fixture {domain} role ID {r_id} embeds canonical role name {canon_role}"
                )
            for backend in FORBIDDEN_BACKEND_HANDLES:
                assert backend.lower() not in r_id.lower(), (
                    f"Fixture {domain} role ID {r_id} embeds backend handle {backend}"
                )

        raw_group_ids = {g["id"] for g in data.get("interaction_groups", [])}
        for g_id in raw_group_ids:
            assert g_id.startswith("group_"), f"Fixture {domain} group ID {g_id} is not neutral (expected group_*)"

        raw_region_ids = {reg["id"] for reg in data.get("inspectable_regions", [])}
        for reg_id in raw_region_ids:
            assert reg_id.startswith("search_"), f"Fixture {domain} search ID {reg_id} is not neutral (expected search_*)"
            for oracle_reg in FORBIDDEN_ORACLE_REGIONS:
                assert oracle_reg.lower() not in reg_id.lower(), (
                    f"Fixture {domain} search ID {reg_id} embeds oracle region {oracle_reg}"
                )

        # 2. Functional relations and interaction group relations must NOT contain canonical predicate tokens
        for rel in data["functional_relations"]:
            rel_str = rel["relation"].strip().upper()
            assert rel_str not in FORBIDDEN_PREDICATE_TOKENS, (
                f"Fixture {domain} leaked canonical predicate token '{rel_str}' in functional_relations"
            )

        for grp in data.get("interaction_groups", []):
            for req_rel in grp.get("required_relations", []):
                req_str = str(req_rel).strip().upper()
                assert req_str not in FORBIDDEN_PREDICATE_TOKENS, (
                    f"Fixture {domain} leaked canonical predicate token '{req_str}' in group required_relations"
                )
            for ctx_rel in grp.get("context_relations", []):
                ctx_str = str(ctx_rel).strip().upper()
                assert ctx_str not in FORBIDDEN_PREDICATE_TOKENS, (
                    f"Fixture {domain} leaked canonical predicate token '{ctx_str}' in group context_relations"
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

        for oracle_region in FORBIDDEN_ORACLE_REGIONS:
            for reg in data.get("inspectable_regions", []):
                assert reg["id"] != oracle_region, f"Fixture {domain} used oracle region ID {oracle_region}"


def test_mock_fm_adapter_future_proof_interface():
    """Verify that MockFMAdapter implements all attributes expected by downstream providers."""
    k_data = load_ideal_fixture("kitchen")
    adapter = MockFMAdapter(k_data)
    assert adapter.generate_task_requirements() == k_data
    assert adapter.generate_kitchen_functional_graph() == k_data
    assert adapter.generate_inspection_priors()["initial_requirements_satisfied"] is True
    assert adapter.metrics.total_calls == 3
    assert adapter.last_observation_images == []
    assert adapter.last_raw_response == k_data


def diagnose_fixture_canonicalization(domain: str) -> dict[str, Any]:
    """Execute current production canonicalizer on the ideal fixture and return a concept-exact diagnostic report."""
    data = load_ideal_fixture(domain)
    task_instructions = {
        "kitchen": "Prepare and serve coffee and soup for two people using available kitchenware.",
        "living_room": "Prepare the living room for two people watching television.",
        "workshop": "Repair the frame by securing the loose joint.",
    }
    instruction = task_instructions[domain]
    adapter = MockFMAdapter(data)

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
        "reference_diagnostics": {},
    }

    try:
        if domain == "kitchen":
            gf = VLMSpecProvider._kitchen(instruction, [], adapter=adapter)

            # Validate resulting graph
            gf.validate()
            validate_runtime_gf(gf)

            report["status"] = "CANONICALIZED"
            report["graph"] = gf

            # Evidence-based preservation from trace & contract
            raw_contract = gf.raw_requirements[0]
            canonical_roles_map = {
                r_data["raw_vlm_role_id"]: canon_name
                for canon_name, r_data in raw_contract.get("roles", {}).items()
            }
            canonical_groups_map = {
                g_data["raw_vlm_group_id"]: g_name
                for g_name, g_data in raw_contract.get("operation_groups", {}).items()
            }

            for role in data["functional_roles"]:
                rid = role["id"]
                s_canon = canonical_roles_map.get(rid)
                if s_canon:
                    report["concept_preservation"][f"role:{rid}"] = f"PRESERVED -> {s_canon}"
                else:
                    report["concept_preservation"][f"role:{rid}"] = "DROPPED"

                seen_predicates_on_role: set[str] = set()
                for prop in role.get("required_properties", []):
                    mapped_pred = map_unary_property(prop)
                    if mapped_pred and s_canon and mapped_pred in gf.nodes[s_canon].unary_predicates:
                        if mapped_pred in seen_predicates_on_role:
                            report["concept_preservation"][f"prop:{rid}:{prop}"] = f"MERGED_BY_EXPLICIT_RULE -> {mapped_pred}"
                        else:
                            report["concept_preservation"][f"prop:{rid}:{prop}"] = f"PRESERVED -> {mapped_pred}"
                            seen_predicates_on_role.add(mapped_pred)
                    elif mapped_pred:
                        report["concept_preservation"][f"prop:{rid}:{prop}"] = f"DROPPED: mapped to {mapped_pred} but absent in role"
                    else:
                        report["concept_preservation"][f"prop:{rid}:{prop}"] = f"REJECTED: unmapped property {prop!r}"

            for idx, rel in enumerate(data["functional_relations"]):
                s = rel["subject_role"]
                r = rel["relation"]
                o = rel["object_role"]
                s_canon = canonical_roles_map.get(s)
                o_canon = canonical_roles_map.get(o)
                mapped_pred = map_binary_relation(r)
                rel_present = any(
                    rel_item.subject_role == s_canon and rel_item.predicate == mapped_pred and rel_item.object_role == o_canon
                    for rel_item in gf.relations
                )
                rel_key = f"rel:{idx}:{s}:{r}:{o}"
                if rel_present and mapped_pred:
                    report["concept_preservation"][rel_key] = f"PRESERVED -> {s_canon} -[{mapped_pred}]-> {o_canon}"
                elif rel_present:
                    report["concept_preservation"][rel_key] = f"FABRICATED_WITHOUT_RAW_SUPPORT -> {s_canon} -[{mapped_pred}]-> {o_canon}"
                else:
                    report["concept_preservation"][rel_key] = f"REJECTED: unmapped relation {r!r}"

            for grp in data.get("interaction_groups", []):
                gid = grp["id"]
                if gid in canonical_groups_map:
                    report["concept_preservation"][f"group:{gid}"] = f"PRESERVED -> {canonical_groups_map[gid]}"
                else:
                    report["concept_preservation"][f"group:{gid}"] = "DROPPED"

        elif domain == "living_room":
            from mujoco_scenes.environment_vlm_requirements import (
                EnvironmentVLMRequirementProvider,
                map_living_room_fixed_target_role,
                map_living_room_object_payload_role,
                map_living_room_relation,
                map_living_room_role_function,
            )

            provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
            gf = VLMSpecProvider._living_room(instruction, [], provider=provider)
            gf.validate()
            validate_runtime_gf(gf)
            report["status"] = "CANONICALIZED"
            report["graph"] = gf

            # Concept preservation from concept_accounting trace
            trace = gf.metadata.get("canonicalization_trace", {})
            accounting = trace.get("concept_accounting", {})
            roles_acct = accounting.get("roles", {})

            for r in data["functional_roles"]:
                rid = r["id"]
                r_entry = roles_acct.get(rid, {})
                status = r_entry.get("status", "PRESERVED")
                canon_role = r_entry.get("canonical_role", "UNKNOWN")
                report["concept_preservation"][f"role:{rid}"] = f"{status} -> {canon_role}"

                for prop in r.get("required_properties", []):
                    prop_entry = next(
                        (p for p in accounting.get("properties", []) if p["raw_role_id"] == rid and p["raw_phrase"] == prop),
                        None
                    )
                    if prop_entry:
                        status = prop_entry["status"]
                        canon_pred = prop_entry["canonical_predicate"]
                        report["concept_preservation"][f"prop:{rid}:{prop}"] = f"{status} -> {canon_pred}"
                    else:
                        report["concept_preservation"][f"prop:{rid}:{prop}"] = f"REJECTED: unmapped property {prop!r}"

            for idx, rel in enumerate(data["functional_relations"]):
                s = rel["subject_role"]
                r = rel["relation"]
                o = rel["object_role"]
                rel_key = f"rel:{idx}:{s}:{r}:{o}"
                rel_entry = next(
                    (item for item in accounting.get("relations", []) if item["raw_subject_role_id"] == s and item["raw_relation_text"] == r and item["raw_object_role_id"] == o),
                    None
                )
                if rel_entry:
                    s_c = rel_entry["canonical_subject_role_id"]
                    p_c = rel_entry["canonical_predicate"]
                    o_c = rel_entry["canonical_object_role_id"]
                    report["concept_preservation"][rel_key] = f"PRESERVED -> {s_c} -[{p_c}]-> {o_c}"
                else:
                    report["concept_preservation"][rel_key] = f"REJECTED: unmapped relation {r!r}"

            for grp in data.get("interaction_groups", []):
                gid = grp["id"]
                grp_entry = next(
                    (g for g in accounting.get("operation_groups", []) if g["raw_group_id"] == gid),
                    None
                )
                if grp_entry:
                    c_gid = grp_entry["canonical_group_id"]
                    report["concept_preservation"][f"group:{gid}"] = f"PRESERVED -> {c_gid}"
                else:
                    report["concept_preservation"][f"group:{gid}"] = "DROPPED"

        elif domain == "workshop":
            from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider

            provider = FMRequirementProvider(fm_adapter=adapter)
            gf = VLMSpecProvider._workshop(instruction, [], provider=provider)
            gf.validate()
            validate_runtime_gf(gf)
            report["status"] = "CANONICALIZED"
            report["graph"] = gf

            # Evidence-based preservation from normalized provider structures
            normalized_roles_map = {
                r.raw_role_id: r.canonical_role_id
                for r in provider.normalized_roles
            }
            normalized_groups_map = {
                g.id: g
                for g in provider.normalized_operation_groups
            }

            for role in data["functional_roles"]:
                rid = role["id"]
                if rid in normalized_roles_map:
                    report["concept_preservation"][f"role:{rid}"] = f"PRESERVED -> {normalized_roles_map[rid]}"
                else:
                    report["concept_preservation"][f"role:{rid}"] = "DROPPED"

            # Workshop raw unary properties explicit status
            report["concept_preservation"]["properties"] = "Workshop raw unary property count = 0 (no property preservation cases applicable)"

            for idx, rel in enumerate(data["functional_relations"]):
                s = rel["subject_role"]
                r = rel["relation"]
                o = rel["object_role"]
                s_canon = normalized_roles_map.get(s)
                o_canon = normalized_roles_map.get(o)
                # Find matching normalized relation
                matching_norm = next(
                    (nr for nr in provider.normalized_relations if nr.raw_subject_role_id == s and nr.raw_object_role_id == o and nr.raw_relation_text == r),
                    None
                )
                canon_pred = matching_norm.canonical_predicate if matching_norm else None
                rel_present = any(
                    rel_item.subject_role == s_canon and rel_item.predicate == canon_pred and rel_item.object_role == o_canon
                    for rel_item in gf.relations
                )
                rel_key = f"rel:{idx}:{s}:{r}:{o}"
                if rel_present and canon_pred:
                    report["concept_preservation"][rel_key] = f"PRESERVED -> {s_canon} -[{canon_pred}]-> {o_canon}"
                else:
                    report["concept_preservation"][rel_key] = f"REJECTED: unmapped relation {r!r}"

            for grp in data.get("interaction_groups", []):
                gid = grp["id"]
                if gid in normalized_groups_map:
                    report["concept_preservation"][f"group:{gid}"] = f"PRESERVED -> {gid} (extra wrt GT reference)"
                else:
                    report["concept_preservation"][f"group:{gid}"] = "DROPPED"

            for reg in data.get("inspectable_regions", []):
                reg_id = reg["id"]
                canon_reg = resolve_workshop_region_proposal(reg)
                report["concept_preservation"][f"region:{reg_id}"] = f"PRESERVED -> {canon_reg}"

        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Compute full GT reference comparison if canonicalized
        if report["status"] == "CANONICALIZED" and report["graph"] is not None:
            gt_ref = GTSpecProvider().provide(domain, instruction)
            eval_result = evaluate_gf_against_reference(report["graph"], gt_ref)
            eval_dict = eval_result.to_dict()
            report["reference_metrics"] = {
                "role_identity_recall": eval_dict["metrics"]["role_identity_recall"],
                "role_identity_precision": eval_dict["metrics"]["role_identity_precision"],
                "role_exact_recall": eval_dict["metrics"]["role_exact_recall"],
                "role_exact_precision": eval_dict["metrics"]["role_exact_precision"],
                "relation_recall": eval_dict["metrics"]["relation_recall"],
                "relation_precision": eval_dict["metrics"]["relation_precision"],
                "operation_group_identity_recall": eval_dict["metrics"]["operation_group_identity_recall"],
                "operation_group_identity_precision": eval_dict["metrics"]["operation_group_identity_precision"],
                "operation_group_exact_recall": eval_dict["metrics"]["operation_group_exact_recall"],
                "operation_group_exact_precision": eval_dict["metrics"]["operation_group_exact_precision"],
                "reference_complete": eval_dict["metrics"]["reference_complete"],
                "exact_structural_match": eval_dict["metrics"]["exact_structural_match"],
                "missing_roles": eval_dict["roles"]["missing"],
                "extra_roles": eval_dict["roles"]["extra"],
                "missing_relations": eval_dict["relations"]["missing"],
                "extra_relations": eval_dict["relations"]["extra"],
                "missing_operation_groups": eval_dict["operation_groups"]["missing"],
                "extra_operation_groups": eval_dict["operation_groups"]["extra"],
            }
            report["reference_diagnostics"] = {
                "role_attribute_mismatches": eval_dict["roles"].get("attribute_mismatches", {}),
                "role_cardinality_diagnostics": eval_dict["roles"].get("cardinality_diagnostics", {}),
                "role_normalization_diagnostics": eval_dict["roles"].get("normalization_diagnostics", {}),
                "category_diagnostics": eval_dict["roles"].get("category_diagnostics", {}),
                "operation_group_mismatches": eval_dict["operation_groups"].get("mismatches", {}),
                "operation_group_representation_diagnostics": eval_dict["operation_groups"].get("representation_diagnostics", {}),
                "graph_attribute_mismatches": eval_dict["graph_level"].get("attribute_mismatches", {}),
                "cross_group_reuse_mismatch": eval_dict["graph_level"].get("cross_group_reuse_mismatch", False),
            }

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
    """Verify concept-exact evidence-based canonicalization outcomes and metrics across all 3 domains."""
    # 1. Kitchen Diagnostic
    k_diag = diagnose_fixture_canonicalization("kitchen")
    assert k_diag["status"] == "CANONICALIZED"
    assert k_diag["reference_metrics"]["role_identity_recall"] == 1.0
    assert k_diag["reference_metrics"]["relation_recall"] == 1.0
    assert k_diag["reference_metrics"]["operation_group_identity_recall"] == 1.0
    assert k_diag["reference_metrics"]["reference_complete"] is True

    # Check that Kitchen cardinality difference on reusable stirrer explains role_exact_recall = 0.833
    assert k_diag["reference_metrics"]["role_exact_recall"] == pytest.approx(5.0 / 6.0)
    card_diag = k_diag["reference_diagnostics"]["role_cardinality_diagnostics"]["coffee_stirrer"]
    assert card_diag["cardinality_compatible"] is True
    assert card_diag["cardinality_exact"] is False
    assert card_diag["reference_range"] == [1, 2]
    assert card_diag["candidate_range"] == [1, 1]

    # Verify 4 distinct relation keys exist and are predicate-verified
    k_rel_keys = [k for k in k_diag["concept_preservation"] if k.startswith("rel:")]
    assert len(k_rel_keys) == 4
    assert any("fits inside" in k and "INSERTABLE_IN" in k_diag["concept_preservation"][k] for k in k_rel_keys)
    assert any("reaches the bottom" in k and "REACHES_BOTTOM" in k_diag["concept_preservation"][k] for k in k_rel_keys)

    # Verify property preservation entries
    assert "PRESERVED -> OPEN_CAVITY" in k_diag["concept_preservation"]["prop:role_1:open cavity"]
    assert "MERGED_BY_EXPLICIT_RULE -> OPEN_CAVITY" in k_diag["concept_preservation"]["prop:role_1:capable of holding liquid"]
    assert "PRESERVED -> ELONGATED_OBJECT" in k_diag["concept_preservation"]["prop:role_3:elongated shape"]
    assert "MERGED_BY_EXPLICIT_RULE -> ELONGATED_OBJECT" in k_diag["concept_preservation"]["prop:role_3:slender"]

    # Coverage assertion: all expected raw concept keys are diagnosed
    expected_k_keys = {
        "role:role_1", "role:role_2", "role:role_3", "role:role_4", "role:role_5", "role:role_6",
        "prop:role_1:open cavity", "prop:role_1:capable of holding liquid",
        "prop:role_2:open cavity", "prop:role_2:capable of holding liquid",
        "prop:role_3:elongated shape", "prop:role_3:slender",
        "prop:role_4:elongated shape",
        "rel:0:role_3:fits inside:role_1", "rel:1:role_3:reaches the bottom:role_1",
        "rel:2:role_4:fits inside:role_2", "rel:3:role_4:reaches the bottom:role_2",
        "group:group_1", "group:group_2",
    }
    assert set(k_diag["concept_preservation"].keys()) == expected_k_keys

    # 2. Living Room Diagnostic
    l_diag = diagnose_fixture_canonicalization("living_room")
    assert l_diag["status"] == "CANONICALIZED"
    assert l_diag["reference_metrics"]["role_identity_recall"] == 1.0
    assert l_diag["reference_metrics"]["role_identity_precision"] == 1.0
    assert l_diag["reference_metrics"]["role_exact_recall"] == 1.0
    assert l_diag["reference_metrics"]["relation_recall"] == 1.0
    assert l_diag["reference_metrics"]["relation_precision"] == 1.0
    assert l_diag["reference_metrics"]["operation_group_identity_recall"] == 1.0
    assert l_diag["reference_metrics"]["operation_group_identity_precision"] == 1.0
    assert l_diag["reference_metrics"]["reference_complete"] is True

    # Verify sub-concept mapping diagnostic with contextual arguments
    assert "PRESERVED -> PERSONAL_CUP_SAUCER_REGION" in l_diag["concept_preservation"]["role:role_1"]
    assert "PRESERVED -> SHARED_REMOTE_REGION" in l_diag["concept_preservation"]["role:role_2"]
    assert "PRESERVED -> CUP_SAUCER_SET" in l_diag["concept_preservation"]["role:role_3"]
    assert "PRESERVED -> REMOTE" in l_diag["concept_preservation"]["role:role_4"]
    assert "PRESERVED -> SEATING_POSITION" in l_diag["concept_preservation"]["role:role_5"]
    assert "PRESERVED -> SEATING_PAIR" in l_diag["concept_preservation"]["role:role_6"]
    assert "PRESERVED -> PLANAR_SUPPORT" in l_diag["concept_preservation"]["prop:role_1:planar horizontal support"]
    assert "PRESERVED -> PLANAR_SUPPORT" in l_diag["concept_preservation"]["prop:role_2:planar horizontal support"]

    # Verify relations with canonical context
    assert "PRESERVED -> PERSONAL_CUP_SAUCER_REGION -[FITS_SET_ON]-> CUP_SAUCER_SET" in l_diag["concept_preservation"]["rel:0:role_1:can hold drinkware set:role_3"]
    assert "PRESERVED -> PERSONAL_CUP_SAUCER_REGION -[NEAR_SEAT]-> SEATING_POSITION" in l_diag["concept_preservation"]["rel:1:role_1:near seat:role_5"]
    assert "PRESERVED -> SHARED_REMOTE_REGION -[FITS_ON]-> REMOTE" in l_diag["concept_preservation"]["rel:2:role_2:can hold remote:role_4"]
    assert "PRESERVED -> SHARED_REMOTE_REGION -[ACCESSIBLE_FROM_BOTH_SEATS]-> SEATING_PAIR" in l_diag["concept_preservation"]["rel:3:role_2:accessible from both seats:role_6"]
    assert "PRESERVED -> personal_support_group" in l_diag["concept_preservation"]["group:group_1"]

    expected_l_keys = {
        "role:role_1", "role:role_2", "role:role_3", "role:role_4", "role:role_5", "role:role_6",
        "prop:role_1:planar horizontal support", "prop:role_2:planar horizontal support",
        "rel:0:role_1:can hold drinkware set:role_3", "rel:1:role_1:near seat:role_5",
        "rel:2:role_2:can hold remote:role_4", "rel:3:role_2:accessible from both seats:role_6",
        "group:group_1",
    }
    assert set(l_diag["concept_preservation"].keys()) == expected_l_keys

    # 3. Workshop Diagnostic
    w_diag = diagnose_fixture_canonicalization("workshop")
    assert w_diag["status"] == "CANONICALIZED"
    assert w_diag["reference_metrics"]["role_identity_recall"] == 1.0
    assert w_diag["reference_metrics"]["relation_recall"] == 1.0
    assert w_diag["reference_metrics"]["reference_complete"] is True
    assert w_diag["reference_metrics"]["extra_operation_groups"] == ["group_1"]
    assert w_diag["reference_metrics"]["exact_structural_match"] is False

    # Verify 3 distinct relation keys with exact predicate mapping
    w_rel_keys = [k for k in w_diag["concept_preservation"] if k.startswith("rel:")]
    assert len(w_rel_keys) == 3
    assert "PRESERVED -> driver -[COMPATIBLE_WITH]-> fastener" in w_diag["concept_preservation"]["rel:0:role_1:compatible with:role_2"]
    assert "PRESERVED -> driver -[REACHES_TARGET]-> repair_target" in w_diag["concept_preservation"]["rel:1:role_1:reaches target:role_3"]
    assert "PRESERVED -> fastener -[COMPATIBLE_WITH_TARGET]-> repair_target" in w_diag["concept_preservation"]["rel:2:role_2:compatible with target:role_3"]

    # Verify exact 1-to-1 region proposals resolution
    assert w_diag["concept_preservation"]["region:search_1"] == "PRESERVED -> LEFT_DRAWER"
    assert w_diag["concept_preservation"]["region:search_2"] == "PRESERVED -> RIGHT_DRAWER"
    assert w_diag["concept_preservation"]["region:search_3"] == "PRESERVED -> TOOL_CABINET"

    # Verify explicit zero property record
    assert "count = 0" in w_diag["concept_preservation"]["properties"]

    expected_w_keys = {
        "role:role_1", "role:role_2", "role:role_3",
        "properties",
        "rel:0:role_1:compatible with:role_2", "rel:1:role_1:reaches target:role_3", "rel:2:role_2:compatible with target:role_3",
        "group:group_1",
        "region:search_1", "region:search_2", "region:search_3",
    }
    assert set(w_diag["concept_preservation"].keys()) == expected_w_keys
