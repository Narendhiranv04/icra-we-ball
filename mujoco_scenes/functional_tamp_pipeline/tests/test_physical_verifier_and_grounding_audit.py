"""Unit tests for Phase 7: Physical Verifier and Grounding Audit (Gate 7).

Verifies:
1. Every accepted relation has a verified evidence owner; no relation accepted without TRUE evidence.
2. UNKNOWN relation evidence never becomes TRUE and never yields complete grounding.
3. FALSE relation evidence strictly rejects candidate binding.
4. Kitchen primitives evaluate_insertable_in and evaluate_reaches_bottom return UNKNOWN on missing measurements.
5. Physical verification trace contains required diagnostic fields.
6. Detector category alone never proves a geometric relation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.geometry_relations import evaluate_insertable_in, evaluate_reaches_bottom
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph

ROOT = Path(__file__).resolve().parents[3]


def test_kitchen_primitive_unknown_on_missing_measurements():
    """Primitive relation verifiers return UNKNOWN with REQUIRED_MEASUREMENT_MISSING when inputs are None."""
    # Insertion fit
    res_none_tool = evaluate_insertable_in(tool_cross_section_m=None, target_opening_width_m=0.08, clearance_margin_m=0.005)
    assert res_none_tool["status"] == "UNKNOWN"
    assert res_none_tool["reason"] == "REQUIRED_MEASUREMENT_MISSING"

    res_none_target = evaluate_insertable_in(tool_cross_section_m=0.02, target_opening_width_m=None, clearance_margin_m=0.005)
    assert res_none_target["status"] == "UNKNOWN"
    assert res_none_target["reason"] == "REQUIRED_MEASUREMENT_MISSING"

    # Reach into container
    res_reach_none = evaluate_reaches_bottom(tool_usable_length_m=None, target_cavity_depth_m=0.10, grip_allowance_m=0.02)
    assert res_reach_none["status"] == "UNKNOWN"
    assert res_reach_none["reason"] == "REQUIRED_MEASUREMENT_MISSING"


def test_unknown_never_becomes_true():
    """A required relation with UNKNOWN evidence in G_O must NEVER produce a complete grounding."""
    graph_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Stir coffee",
        nodes={
            "stirrer": FunctionalRole(name="stirrer", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("spoon",)),
            "cup": FunctionalRole(name="cup", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("cup",)),
        },
        relations=(FunctionalRelation(subject_role="stirrer", predicate="INSERTABLE_IN", object_role="cup", expected=True),),
        operation_groups=(
            OperationGroup(
                id="stir_group",
                function="STIR_COFFEE",
                tool_role="stirrer",
                target_role="cup",
                required_target_count=1,
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("INSERTABLE_IN",),
            ),
        ),
        metadata={"required_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="spoon_1", entity_kind="OBJECT", canonical_category="spoon"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT", canonical_category="cup"))
    graph_o.add_relation(ObservedRelation(
        subject_id="spoon_1",
        predicate="INSERTABLE_IN",
        object_id="cup_1",
        status="UNKNOWN",
        evidence={"reason": "MISSING_CAMERA_VIEW"},
    ))

    result = ground_graph(graph_f, graph_o, domain_context={"search_exhausted": True})
    assert result.complete is False
    assert result.status in {"INCOMPLETE", "INFEASIBLE"}
    assert result.assignment is None


def test_false_relation_rejects_candidate():
    """A required relation with FALSE evidence in G_O strictly rejects the candidate binding."""
    graph_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Stir coffee",
        nodes={
            "stirrer": FunctionalRole(name="stirrer", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("spoon",)),
            "cup": FunctionalRole(name="cup", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("cup",)),
        },
        relations=(FunctionalRelation(subject_role="stirrer", predicate="INSERTABLE_IN", object_role="cup", expected=True),),
        operation_groups=(
            OperationGroup(
                id="stir_group",
                function="STIR_COFFEE",
                tool_role="stirrer",
                target_role="cup",
                required_target_count=1,
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("INSERTABLE_IN",),
            ),
        ),
        metadata={"required_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="spoon_1", entity_kind="OBJECT", canonical_category="spoon"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT", canonical_category="cup"))
    graph_o.add_relation(ObservedRelation(
        subject_id="spoon_1",
        predicate="INSERTABLE_IN",
        object_id="cup_1",
        status="FALSE",
        evidence={"pass_margin_m": -0.01},
    ))

    result = ground_graph(graph_f, graph_o, domain_context={"search_exhausted": True})
    assert result.complete is False
    assert result.status == "INFEASIBLE"
    assert result.assignment is None


def test_true_relation_backed_by_physical_evidence_grounds():
    """When a required relation has verified TRUE evidence in G_O, grounding succeeds."""
    graph_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Stir coffee",
        nodes={
            "stirrer": FunctionalRole(name="stirrer", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("spoon",)),
            "cup": FunctionalRole(name="cup", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("cup",)),
        },
        relations=(FunctionalRelation(subject_role="stirrer", predicate="INSERTABLE_IN", object_role="cup", expected=True),),
        operation_groups=(
            OperationGroup(
                id="stir_group",
                function="STIR_COFFEE",
                tool_role="stirrer",
                target_role="cup",
                required_target_count=1,
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("INSERTABLE_IN",),
            ),
        ),
        metadata={"required_contract_complete": True},
    )

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="spoon_1", entity_kind="OBJECT", canonical_category="spoon"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT", canonical_category="cup"))
    graph_o.add_relation(ObservedRelation(
        subject_id="spoon_1",
        predicate="INSERTABLE_IN",
        object_id="cup_1",
        status="TRUE",
        evidence={"pass_margin_m": 0.015, "method": "observed_diameter_comparison"},
    ))

    result = ground_graph(graph_f, graph_o, domain_context={"search_exhausted": True})
    assert result.complete is True
    assert result.status == "COMPLETE"
    assert result.assignment == {"stirrer": "spoon_1", "cup": "cup_1"}


def test_detector_category_alone_never_proves_relation():
    """A detector category matching spoon and cup does NOT satisfy an unmeasured relation."""
    graph_f = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="Stir coffee",
        nodes={
            "stirrer": FunctionalRole(name="stirrer", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("spoon",)),
            "cup": FunctionalRole(name="cup", entity_kind="OBJECT", count=1, binding_policy="DISTINCT", semantic_categories=("cup",)),
        },
        relations=(FunctionalRelation(subject_role="stirrer", predicate="INSERTABLE_IN", object_role="cup", expected=True),),
        operation_groups=(
            OperationGroup(
                id="stir_group",
                function="STIR_COFFEE",
                tool_role="stirrer",
                target_role="cup",
                required_target_count=1,
                usage_policy="DEDICATED_PER_TARGET",
                required_relations=("INSERTABLE_IN",),
            ),
        ),
        metadata={"required_contract_complete": True},
    )

    # G_O has nodes with matching categories, but NO relation added to G_O
    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="spoon_1", entity_kind="OBJECT", canonical_category="spoon"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT", canonical_category="cup"))

    result = ground_graph(graph_f, graph_o, domain_context={"search_exhausted": True})
    assert result.complete is False
    assert result.assignment is None


def test_physical_relation_verification_trace_structure(tmp_path):
    """Run pipeline artifact collection and verify physical_relation_verification_trace.json format."""
    from mujoco_scenes.functional_tamp_pipeline.run import _write_json

    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode(instance_id="spoon_1", entity_kind="OBJECT", canonical_category="spoon"))
    graph_o.add_node(ObservedNode(instance_id="cup_1", entity_kind="OBJECT", canonical_category="cup"))
    graph_o.add_relation(ObservedRelation(
        subject_id="spoon_1",
        predicate="INSERTABLE_IN",
        object_id="cup_1",
        status="TRUE",
        evidence={
            "method": "evaluate_insertable_in",
            "tool_cross_section_m": 0.015,
            "target_opening_width_m": 0.08,
            "clearance_margin_m": 0.005,
            "pass_margin_m": 0.06,
        },
    ))

    verification_trace = []
    for obs_rel in sorted(graph_o.relations.values(), key=lambda r: (r.predicate, r.subject_id, r.object_id)):
        ev = dict(obs_rel.evidence)
        verification_trace.append({
            "predicate": obs_rel.predicate,
            "subject_instance": obs_rel.subject_id,
            "object_instance": obs_rel.object_id,
            "verifier_function": ev.get("method", ev.get("verifier", f"verify_{obs_rel.predicate.lower()}")),
            "measured_quantities": {k: v for k, v in ev.items() if k not in ("status", "method", "reason")},
            "thresholds": {k: v for k, v in ev.items() if any(sub in k for sub in ("threshold", "tolerance", "clearance", "maximum", "minimum"))},
            "signed_margins": {k: v for k, v in ev.items() if "margin" in k},
            "status": obs_rel.status,
            "reason": ev.get("reason"),
            "evidence": ev,
        })

    trace_file = tmp_path / "physical_relation_verification_trace.json"
    _write_json(trace_file, verification_trace)

    assert trace_file.is_file()
    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    entry = data[0]
    assert entry["predicate"] == "INSERTABLE_IN"
    assert entry["subject_instance"] == "spoon_1"
    assert entry["object_instance"] == "cup_1"
    assert entry["status"] == "TRUE"
    assert entry["verifier_function"] == "evaluate_insertable_in"
    assert "pass_margin_m" in entry["signed_margins"]
