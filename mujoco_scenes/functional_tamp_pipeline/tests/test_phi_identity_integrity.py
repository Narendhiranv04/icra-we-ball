"""Regression tests for Phase 3 canonical phi* instance identity integrity and audit hardening."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.domains.workshop import (
    WorkshopDomainAdapter,
    WorkshopPlanningCompiler,
    compile_workshop_requirements_from_graph,
)
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    GraphGroundingResult,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedObject,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.audit import audit_plan_grounding
from mujoco_scenes.functional_tamp_pipeline.planning import plan_with_common_astar


def test_workshop_canonical_phi_preserves_go_instances():
    """Verify that Workshop satisfaction and planning preserve G_O instance IDs without rewriting into simulator body names."""
    # Build synthetic G_F
    graph_f = FunctionalRequirementGraph(
        task_instruction="Repair frame",
        domain="workshop",
        nodes={
            "driver": FunctionalRole(
                name="driver",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("power_driver", "screwdriver"),
                unary_predicates=("CAN_DRIVE_SCREW",),
            ),
            "fastener": FunctionalRole(
                name="fastener",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("screw",),
                unary_predicates=("CAN_FASTEN",),
            ),
            "repair_target": FunctionalRole(
                name="repair_target",
                entity_kind="FIXED_TARGET",
                count=1,
                semantic_categories=("repair_target",),
            ),
        },
        relations=(
            FunctionalRelation(subject_role="driver", predicate="REACHES_TARGET", object_role="repair_target"),
            FunctionalRelation(subject_role="fastener", predicate="COMPATIBLE_WITH_TARGET", object_role="repair_target"),
            FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener"),
        ),
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER"),
        source="GT_FUNCTIONAL_SPEC_ONLY",
    )

    # Build synthetic G_O with tracked instances
    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedObject(
        instance_id="object_track_101",
        entity_kind="OBJECT",
        canonical_category="power_driver",
        semantic_labels={"canonical_label": "power_driver", "status": "SUPPORTED"},
        source_region="RIGHT_DRAWER",
        unary_predicates={"CAN_DRIVE_SCREW": "TRUE", "CAN_FASTEN": "FALSE"},
    ))
    graph_o.add_node(ObservedObject(
        instance_id="object_track_202",
        entity_kind="OBJECT",
        canonical_category="screw",
        semantic_labels={"canonical_label": "screw", "status": "SUPPORTED"},
        source_region="LEFT_DRAWER",
        unary_predicates={"CAN_DRIVE_SCREW": "FALSE", "CAN_FASTEN": "TRUE"},
    ))
    graph_o.add_node(ObservedNode(
        instance_id="repair_target",
        entity_kind="FIXED_TARGET",
        canonical_category="repair_target",
    ))
    graph_o.add_node(ObservedNode(
        instance_id="MAIN_WORKBENCH_ZONE",
        entity_kind="REGION",
        canonical_category="MAIN_WORKBENCH_ZONE",
    ))
    graph_o.add_node(ObservedNode(
        instance_id="LEFT_DRAWER",
        entity_kind="REGION",
        canonical_category="LEFT_DRAWER",
    ))
    graph_o.add_node(ObservedNode(
        instance_id="RIGHT_DRAWER",
        entity_kind="REGION",
        canonical_category="RIGHT_DRAWER",
    ))

    # Add relations
    graph_o.add_relation(ObservedRelation("object_track_101", "REACHES_TARGET", "repair_target", "TRUE"))
    graph_o.add_relation(ObservedRelation("object_track_202", "COMPATIBLE_WITH_TARGET", "repair_target", "TRUE"))
    graph_o.add_relation(ObservedRelation("object_track_101", "COMPATIBLE_WITH", "object_track_202", "TRUE"))

    from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
    ground_res = ground_graph(graph_f, graph_o, {"search_exhausted": True})
    assert ground_res.complete is True
    assert ground_res.assignment["driver"] == "object_track_101"
    assert ground_res.assignment["fastener"] == "object_track_202"

    # Plan with WorkshopPlanningCompiler
    compiler = WorkshopPlanningCompiler()
    context = {
        "opened_regions": ("LEFT_DRAWER", "RIGHT_DRAWER"),
        "sources": {
            "object_track_101": "RIGHT_DRAWER",
            "object_track_202": "LEFT_DRAWER",
        },
        "work_surface": "MAIN_WORKBENCH_ZONE",
        "target_joint": "workshop_frame_joint",
    }
    planned = plan_with_common_astar(compiler, ground_res.assignment, context)
    assert planned.actions is not None
    assert len(planned.actions) == 5

    # Verify action plan uses the exact G_O instance IDs, NEVER backend simulator strings
    driver_args = [a["arguments"][0] for a in planned.actions if a["operator"] == "PICK" and "RIGHT_DRAWER" in a["arguments"]]
    fastener_args = [a["arguments"][0] for a in planned.actions if a["operator"] == "PICK" and "LEFT_DRAWER" in a["arguments"]]

    assert driver_args == ["object_track_101"]
    assert fastener_args == ["object_track_202"]

    for action in planned.actions:
        for arg in action["arguments"]:
            assert arg not in {"workshop_power_driver", "workshop_long_phillips_driver", "workshop_medium_phillips_screw"}, (
                f"Action {action} leaked simulator backend handle '{arg}'"
            )

    # Run plan grounding audit
    audit = audit_plan_grounding(graph_f, graph_o, ground_res, planned.actions, home_region="MAIN_WORKBENCH_ZONE")
    assert audit["grounding_complete"] is True
    assert audit["all_assignment_nodes_observed"] is True
    assert audit["plan_uses_only_grounded_task_objects"] is True
    assert len(audit["violations"]) == 0


def test_audit_plan_grounding_rejects_fake_prefixed_arguments():
    """Verify that audit_plan_grounding fails closed on unobserved region_*, seat_*, or object_* names."""
    graph_f = FunctionalRequirementGraph(
        task_instruction="Test task",
        domain="living_room",
        nodes={
            "item": FunctionalRole(name="item", entity_kind="OBJECT", count=1, semantic_categories=("cup",)),
        },
        source="GT_FUNCTIONAL_SPEC_ONLY",
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedObject(instance_id="object_0001", entity_kind="OBJECT", canonical_category="cup"))
    graph_o.add_node(ObservedNode(instance_id="region_0001", entity_kind="REGION", canonical_category="table"))

    ground_res = GraphGroundingResult(
        status="COMPLETE",
        complete=True,
        assignment={"item": "object_0001"},
    )

    # 1. Valid plan passes
    valid_plan = [{"action_index": 1, "operator": "PLACE", "arguments": ["object_0001", "region_0001"]}]
    audit_valid = audit_plan_grounding(graph_f, graph_o, ground_res, valid_plan, home_region="staging_tray")
    assert audit_valid["plan_uses_only_grounded_task_objects"] is True
    assert len(audit_valid["violations"]) == 0

    # 2. Fake region_nonexistent must FAIL
    fake_region_plan = [{"action_index": 1, "operator": "PLACE", "arguments": ["object_0001", "region_nonexistent"]}]
    audit_fake_region = audit_plan_grounding(graph_f, graph_o, ground_res, fake_region_plan, home_region="staging_tray")
    assert audit_fake_region["plan_uses_only_grounded_task_objects"] is False
    assert any("region_nonexistent" in v for v in audit_fake_region["violations"])

    # 3. Fake seat_nonexistent must FAIL
    fake_seat_plan = [{"action_index": 1, "operator": "NAVIGATE", "arguments": ["seat_nonexistent"]}]
    audit_fake_seat = audit_plan_grounding(graph_f, graph_o, ground_res, fake_seat_plan, home_region="staging_tray")
    assert audit_fake_seat["plan_uses_only_grounded_task_objects"] is False
    assert any("seat_nonexistent" in v for v in audit_fake_seat["violations"])

    # 4. Fake object_nonexistent must FAIL
    fake_obj_plan = [{"action_index": 1, "operator": "PICK", "arguments": ["object_nonexistent"]}]
    audit_fake_obj = audit_plan_grounding(graph_f, graph_o, ground_res, fake_obj_plan, home_region="staging_tray")
    assert audit_fake_obj["plan_uses_only_grounded_task_objects"] is False
    assert any("object_nonexistent" in v for v in audit_fake_obj["violations"])


def test_audit_plan_grounding_allowed_context_ids():
    """Verify that explicitly passed allowed_context_ids can authorize domain constants."""
    graph_f = FunctionalRequirementGraph(
        task_instruction="Test task",
        domain="custom",
        nodes={"item": FunctionalRole(name="item", entity_kind="OBJECT", count=1, semantic_categories=("item",))},
        source="GT_FUNCTIONAL_SPEC_ONLY",
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedObject(instance_id="object_0001", entity_kind="OBJECT", canonical_category="item"))
    ground_res = GraphGroundingResult(status="COMPLETE", complete=True, assignment={"item": "object_0001"})

    plan = [{"action_index": 1, "operator": "DISPATCH", "arguments": ["object_0001", "CUSTOM_STATION_A"]}]

    # Without allowed_context_ids, CUSTOM_STATION_A fails
    audit_fail = audit_plan_grounding(graph_f, graph_o, ground_res, plan, home_region="staging_tray")
    assert audit_fail["plan_uses_only_grounded_task_objects"] is False

    # With allowed_context_ids, CUSTOM_STATION_A passes
    audit_pass = audit_plan_grounding(
        graph_f, graph_o, ground_res, plan, home_region="staging_tray",
        allowed_context_ids=["CUSTOM_STATION_A"]
    )
    assert audit_pass["plan_uses_only_grounded_task_objects"] is True
    assert len(audit_pass["violations"]) == 0


def test_pipeline_result_assignment_matches_canonical_grounding_contract(tmp_path: Path):
    """Verify that across all 3 domains, PipelineResult.assignment equals GraphGroundingResult.assignment."""
    from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline

    for domain, variant in [("kitchen", "K1"), ("living_room", "L1"), ("workshop", "W1")]:
        out_root = tmp_path / domain / variant
        result = run_pipeline(domain=domain, variant=variant, mode="gt", output_root=out_root)
        assert result.status == "ACTION_SEQUENCE_READY"

        run_dir = out_root / domain / variant / "gt"
        ggr = json.loads((run_dir / "graph_grounding_result.json").read_text(encoding="utf-8"))
        res_json = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

        assert res_json["assignment"] == ggr["assignment"], f"Assignment mismatch in {domain} {variant}!"
        assert result.assignment == ggr["assignment"], f"Result object assignment mismatch in {domain} {variant}!"
