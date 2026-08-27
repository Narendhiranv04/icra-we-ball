"""Living-room one-observation grounding and common-core planning adapter."""

from __future__ import annotations

import json
from pathlib import Path
import yaml

from mujoco_scenes.living_room_region_function import (
    IntegratedLivingRoomRegionRun, write_resolved_integrated_rig,
)
from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene
from mujoco_scenes.living_room_symbolic_planning import run_living_room_symbolic_pipeline
from mujoco_scenes.living_room_variants import scene_name
from mujoco_scenes.region_ablation import create_region_semantic_detector
from mujoco_scenes.region_ablation2 import DEFAULT_EVALUATION_CONFIG

from ..models import FunctionalSpecification, PipelineResult


TASK = (
    "Prepare the living room for two people watching television: put one cup "
    "and saucer on each personal support and the remote on a shared support."
)
LOCAL_MODEL = Path(__file__).resolve().parents[3] / "semantic_model_cache/yolov8m-worldv2.pt"


def compile_living_room_task_from_graph(graph: FunctionalRequirementGraph) -> dict[str, Any]:
    """Deterministically compile living room task contract from G_F."""
    function_groups = {}
    semantic_region_roles = {}

    for name, node in graph.nodes.items():
        if node.entity_kind == "REGION":
            role_key = name.lower()
            policy = "SHARED_REGION_REQUIRED" if node.shared else "DEDICATED_REGION_PER_TARGET"
            target_role = "both_seating_positions" if node.shared else "seating_position"
            req_rels = ["PLANAR_SUPPORT"]
            if node.shared:
                req_rels.extend(["FITS_ON", "ACCESSIBLE_FROM_BOTH_SEATS"])
            else:
                req_rels.extend(["FITS_SET_ON", "NEAR_SEAT"])

            function_groups[name.lower()] = {
                "function_id": name,
                "candidate_entity_kind": "REGION",
                "region_role": role_key,
                "usage_policy": policy,
                "target_assignment_policy": "TARGET_SPECIFIC" if not node.shared else "SHARED_REGION",
                "target_role": target_role,
                "required_target_count": node.count,
                "payload_groups": ["shared_remote"] if node.shared else ["personal_cup_saucer_sets"],
                "required_relations": req_rels,
            }
            semantic_region_roles[role_key] = {
                "accepted_categories": {cat: 1 for cat in node.semantic_categories},
                "rejected_categories": {
                    cat: 1 for cat in ("floor", "rug", "media_console", "bookshelf", "chair", "armchair", "sofa")
                    if cat not in node.semantic_categories
                },
            }

    return {
        "schema_version": 1,
        "task_id": "l2_integrated_living_room_region_function_phase1",
        "requirement_entity_kind": "REGION",
        "natural_language_goal": graph.task_instruction,
        "specification_source": graph.source,
        "payload_groups": {
            "personal_cup_saucer_sets": {
                "count": 2,
                "each_requires": {"cup": 1, "saucer": 1},
                "grouping_policy": "MINIMUM_TOTAL_OBSERVED_CENTROID_DISTANCE",
                "target_assignment_policy": "OBSERVED_X_ORDER",
                "target_role": "seating_position",
            },
            "shared_remote": {
                "count": 1,
                "each_requires": {"tv_remote": 1},
                "grouping_policy": "ALL_REQUIRED_ROLES",
            },
        },
        "function_groups": function_groups,
        "allow_cross_function_region_sharing": False,
        "semantic_requirements": {
            "region_roles": semantic_region_roles,
            "payload_roles": {
                "cup": ["cup"],
                "saucer": ["saucer", "plate"],
                "tv_remote": ["remote_control", "tv_remote"],
            },
            "seating_categories": ["armchair", "chair", "sofa"],
            "minimum_supporting_views": 1,
            "minimum_mean_confidence": 0.03,
            "minimum_winning_score_margin": 0.015,
        },
        "geometric_requirements": {
            "unary_region": {
                "predicate": "PLANAR_SUPPORT",
                "maximum_normal_angle_degrees": 12.0,
                "minimum_planarity_score": 0.90,
                "minimum_usable_area_m2": 0.030,
            },
            "payload_region": {
                "relation": "FITS_ON",
                "edge_clearance_margin_m": 0.015,
                "allowed_orientations_degrees": [0, 90],
            },
            "payload_set_region": {
                "relation": "FITS_SET_ON",
                "edge_clearance_margin_m": 0.020,
                "inter_payload_clearance_m": 0.025,
                "allowed_orientations_degrees": [0, 90],
                "arrangements": ["ALONG_LENGTH", "ALONG_WIDTH"],
            },
            "personal_context": {
                "relation": "NEAR_SEAT",
                "maximum_centroid_distance_m": 1.20,
            },
            "control_context": {
                "relation": "ACCESSIBLE_FROM_BOTH_SEATS",
                "maximum_distance_to_each_seat_m": 1.65,
            },
        },
        "allocation": {
            "production_policy": "global_target_specific",
            "diagnostic_modes": [
                "semantic_only", "geometry_only", "joint", "target_agnostic_count",
                "greedy_target_specific", "global_target_specific",
            ],
            "deterministic_tie_break": [
                "complete_target_coverage",
                "total_signed_margin_descending",
                "total_candidate_rank",
                "persistent_region_ids",
            ],
        },
    }


def build_living_room_observed_scene_graph(run: Any) -> ObservedSceneGraph:
    """Build canonical ObservedSceneGraph G_O from Living Room perception run."""
    from ..scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph

    graph_o = ObservedSceneGraph()
    for region_id, reg_data in getattr(run, "region_evidence", {}).items():
        canonical = reg_data.get("semantic_classification", {}).get("canonical_label")
        unary_preds = {"PLANAR_SUPPORT": "TRUE" if reg_data.get("is_planar", True) else "FALSE"}
        node = ObservedNode(
            instance_id=region_id,
            entity_kind="REGION",
            canonical_category=canonical,
            semantic_labels=dict(reg_data.get("semantic_classification", {})),
            unary_predicates=unary_preds,
        )
        graph_o.add_node(node)

    # Add target nodes to G_O
    graph_o.add_node(ObservedNode(instance_id="cup_saucer_payload_target", entity_kind="OBJECT", canonical_category="cup_saucer"))
    graph_o.add_node(ObservedNode(instance_id="remote_payload_target", entity_kind="OBJECT", canonical_category="tv_remote"))
    graph_o.add_node(ObservedNode(instance_id="seating_target", entity_kind="FIXED_TARGET", canonical_category="seating"))
    graph_o.add_node(ObservedNode(instance_id="seating_pair_target", entity_kind="FIXED_TARGET", canonical_category="seating_pair"))

    # Populate relations from run.production_result
    for alloc in run.production_result.get("assignments", []):
        slot = alloc["slot_id"]
        reg_id = alloc["region_id"]
        if "personal" in slot.lower():
            graph_o.add_relation(ObservedRelation(subject_id=reg_id, predicate="FITS_SET_ON", object_id="cup_saucer_payload_target", status="TRUE"))
            graph_o.add_relation(ObservedRelation(subject_id=reg_id, predicate="NEAR_SEAT", object_id="seating_target", status="TRUE"))
        elif "shared" in slot.lower():
            graph_o.add_relation(ObservedRelation(subject_id=reg_id, predicate="FITS_ON", object_id="remote_payload_target", status="TRUE"))
            graph_o.add_relation(ObservedRelation(subject_id=reg_id, predicate="ACCESSIBLE_FROM_BOTH_SEATS", object_id="seating_pair_target", status="TRUE"))

    return graph_o


def run_to_plan(
    *, variant_label: str, internal_variant: str, mode: str,
    specification: FunctionalSpecification, output_dir: Path,
) -> PipelineResult:
    from ..grounding import ground_graph

    phase1 = output_dir / "observed_grounding"
    rig_path = output_dir / "resolved_rig.yaml"
    name = scene_name(internal_variant)
    write_resolved_integrated_rig(name, rig_path)
    detector, semantic_config = create_region_semantic_detector(
        checkpoint=str(LOCAL_MODEL), confidence_threshold=0.03,
        vocabulary_path=specification.metadata["semantic_vocabulary_path"],
    )
    scene = L2LivingRoomRegionScene(name, robot="none")
    compiled_task = compile_living_room_task_from_graph(specification)
    task_path = output_dir / "functional_task_contract.yaml"
    task_path.write_text(
        yaml.safe_dump(compiled_task, sort_keys=False),
        encoding="utf-8",
    )
    task_config = str(task_path)

    run = IntegratedLivingRoomRegionRun(
        phase1, scene_name=name,
        task_config=task_config,
        evaluation_config=DEFAULT_EVALUATION_CONFIG,
        rig_config=rig_path, semantic_detector=detector,
        semantic_config=semantic_config, width=1280, height=960,
    ).run(scene)

    graph_o = build_living_room_observed_scene_graph(run)
    ground_result = ground_graph(specification, graph_o)

    if not ground_result.complete or not ground_result.assignment:
        return PipelineResult(
            domain="living_room", variant=variant_label, mode=mode,
            status=ground_result.status, failure_reason=str(
                ground_result.unsatisfied_relations or ground_result.missing_roles or "NO_GLOBAL_REGION_ASSIGNMENT"
            ),
        )

    plan_dir = output_dir / "action_sequence"
    planning = run_living_room_symbolic_pipeline(phase1, plan_dir)
    plan_payload = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    actions = tuple({
        "action_index": index + 1,
        "operator": row["operator"],
        "arguments": list(row["arguments"].values()),
    } for index, row in enumerate(plan_payload["actions"]))
    assignment = {row["slot_id"]: row["region_id"] for row in run.production_result["assignments"]}
    return PipelineResult(
        domain="living_room", variant=variant_label, mode=mode,
        status="ACTION_SEQUENCE_READY", assignment=assignment, plan=actions,
        search_statistics=planning.get("search_statistics", {}),
    )
