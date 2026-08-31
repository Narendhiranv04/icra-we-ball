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
from ..scene_graph import ObservedNode, ObservedObject, ObservedRelation, ObservedSceneGraph


TASK = (
    "Prepare the living room for two people watching television: put one cup "
    "and saucer on each personal support and the remote on a shared support."
)
LOCAL_MODEL = Path(__file__).resolve().parents[3] / "semantic_model_cache/yolov8m-worldv2.pt"


def parse_planar_support(geometry_or_prop: Any) -> str:
    """Parse PLANAR_SUPPORT evidence into strict ternary status ('TRUE', 'FALSE', 'UNKNOWN')."""
    if isinstance(geometry_or_prop, dict):
        prop = geometry_or_prop.get("PLANAR_SUPPORT", geometry_or_prop)
        if isinstance(prop, dict):
            status = prop.get("status")
            if status == "UNKNOWN":
                return "UNKNOWN"
            val = prop.get("value")
            if val is None:
                return "UNKNOWN"
            return "TRUE" if bool(val) else "FALSE"
        elif isinstance(prop, bool):
            return "TRUE" if prop else "FALSE"
        elif isinstance(prop, str):
            return prop if prop in {"TRUE", "FALSE", "UNKNOWN"} else "UNKNOWN"
        return "UNKNOWN"
    elif isinstance(geometry_or_prop, bool):
        return "TRUE" if geometry_or_prop else "FALSE"
    elif isinstance(geometry_or_prop, str):
        return geometry_or_prop if geometry_or_prop in {"TRUE", "FALSE", "UNKNOWN"} else "UNKNOWN"
    return "UNKNOWN"


def compile_living_room_task_from_graph(graph: FunctionalRequirementGraph) -> dict[str, Any]:
    """Deterministically compile living room task contract from G_F."""
    function_groups = {}
    semantic_region_roles = {}

    personal_count = 2
    for name, node in graph.nodes.items():
        if node.entity_kind == "REGION":
            role_key = name.lower()
            policy = "SHARED_REGION_REQUIRED" if node.shared else "DEDICATED_REGION_PER_TARGET"
            target_role = "both_seating_positions" if node.shared else "seating_position"
            req_rels = ["PLANAR_SUPPORT"]
            if node.shared:
                req_rels.extend(["FITS_ON", "ACCESSIBLE_FROM_BOTH_SEATS"])
            else:
                personal_count = node.count
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
                "count": personal_count,
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
    """Build canonical ObservedSceneGraph G_O from raw Living Room perception run."""
    from ..scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph

    graph_o = ObservedSceneGraph()
    region_reg = getattr(run, "region_registry", {}) or getattr(run, "region_evidence", {})

    for region_id, reg_data in sorted(region_reg.items()):
        semantics = reg_data.get("semantics", {}) or reg_data.get("semantic_classification", {})
        canonical = semantics.get("canonical_label")
        geometry = reg_data.get("geometry", {})
        planar_str = parse_planar_support(geometry)

        node = ObservedNode(
            instance_id=region_id,
            entity_kind="REGION",
            canonical_category=canonical,
            semantic_labels=dict(semantics),
            unary_predicates={"PLANAR_SUPPORT": planar_str},
            geometry=dict(geometry),
        )
        graph_o.add_node(node)

    # Add payload bundle nodes, individual object nodes, and seat nodes from personal_rows
    personal_rows = getattr(run, "personal_rows", [])
    seen_slots: set[str] = set()
    seen_seats: set[str] = set()
    for row in personal_rows:
        slot_id = row["slot_id"]
        seat_id = row.get("seating_target_id")
        p_ids = list(row.get("payload_ids", []))
        for pid in p_ids:
            if pid not in graph_o.nodes:
                graph_o.add_node(ObservedNode(
                    instance_id=pid,
                    entity_kind="OBJECT",
                    canonical_category="cup_or_saucer",
                    source_region="staging_tray",
                ))
        if slot_id not in seen_slots:
            seen_slots.add(slot_id)
            graph_o.add_node(ObservedNode(
                instance_id=slot_id,
                entity_kind="OBJECT",
                canonical_category="cup_saucer_set",
                unary_properties={"payload_ids": p_ids},
            ))
        if seat_id and seat_id not in seen_seats:
            seen_seats.add(seat_id)
            seat_evidence = getattr(run, "seating_registry", {}).get(seat_id, {})
            graph_o.add_node(ObservedNode(
                instance_id=seat_id,
                entity_kind="FIXED_TARGET",
                canonical_category="seating_position",
                geometry=dict(seat_evidence.get("geometry", {})),
            ))

    # Add shared remote node and seating pair node from shared_rows
    shared_rows = getattr(run, "shared_rows", [])
    remote_ids: list[str] = []
    for row in shared_rows:
        p_ids = row.get("payload_ids", [])
        for pid in p_ids:
            if pid not in graph_o.nodes:
                graph_o.add_node(ObservedNode(
                    instance_id=pid,
                    entity_kind="OBJECT",
                    canonical_category="tv_remote",
                    source_region="staging_tray",
                ))
        if p_ids:
            remote_ids.extend(p_ids)
    remote_id = remote_ids[0] if remote_ids else "tv_remote"
    if remote_id not in graph_o.nodes:
        graph_o.add_node(ObservedNode(
            instance_id=remote_id,
            entity_kind="OBJECT",
            canonical_category="tv_remote",
            source_region="staging_tray",
        ))
    graph_o.add_node(ObservedNode(
        instance_id="SEATING_PAIR",
        entity_kind="FIXED_TARGET",
        canonical_category="seating_pair",
    ))

    # Populate relations from personal_rows
    for row in personal_rows:
        reg_id = row["region_id"]
        slot_id = row["slot_id"]
        seat_id = row.get("seating_target_id")
        fits_set_on = str(row.get("FITS_SET_ON", "UNKNOWN"))
        near_seat = str(row.get("NEAR_SEAT", "UNKNOWN"))
        graph_o.add_relation(ObservedRelation(
            subject_id=reg_id,
            predicate="FITS_SET_ON",
            object_id=slot_id,
            status=fits_set_on,
            evidence=dict(row.get("fit_evidence", {})),
        ))
        if seat_id:
            graph_o.add_relation(ObservedRelation(
                subject_id=reg_id,
                predicate="NEAR_SEAT",
                object_id=seat_id,
                status=near_seat,
                evidence=dict(row.get("context_evidence", {})),
            ))

    # Populate relations from shared_rows
    for row in shared_rows:
        reg_id = row["region_id"]
        fits_on = str(row.get("FITS_ON", "UNKNOWN"))
        accessible = str(row.get("ACCESSIBLE_FROM_BOTH_SEATS", "UNKNOWN"))
        graph_o.add_relation(ObservedRelation(
            subject_id=reg_id,
            predicate="FITS_ON",
            object_id=remote_id,
            status=fits_on,
            evidence=dict(row.get("fit_evidence", {})),
        ))
        graph_o.add_relation(ObservedRelation(
            subject_id=reg_id,
            predicate="ACCESSIBLE_FROM_BOTH_SEATS",
            object_id="SEATING_PAIR",
            status=accessible,
            evidence=dict(row.get("context_evidence", {})),
        ))

    return graph_o


def run_to_plan(
    *, variant_label: str, internal_variant: str, mode: str,
    specification: FunctionalSpecification, output_dir: Path,
    observer: Any = None,
) -> PipelineResult:
    from ..grounding import ground_graph
    from ..task_interface_validator import validate_runtime_gf
    validate_runtime_gf(specification)

    phase1 = output_dir / "observed_grounding"
    if phase1.exists():
        import shutil
        shutil.rmtree(phase1, ignore_errors=True)
    rig_path = output_dir / "resolved_rig.yaml"
    name = scene_name(internal_variant)
    write_resolved_integrated_rig(name, rig_path)

    vocabulary_path: Path
    if specification.detector_vocabulary:
        vocabulary_path = output_dir / "living_room_vocabulary.yaml"
        vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_labels: dict[str, list[str]] = {}
        base_canon: dict[str, list[str]] = {}
        alias_to_base: dict[str, str] = {}
        if "semantic_vocabulary_path" in specification.metadata:
            base_vocab_p = Path(specification.metadata["semantic_vocabulary_path"])
            if base_vocab_p.is_file():
                b_data = yaml.safe_load(base_vocab_p.read_text(encoding="utf-8")) or {}
                base_canon = dict(b_data.get("canonical_labels", {}))
                for c_k, a_list in base_canon.items():
                    for a in a_list:
                        alias_to_base[a.strip().lower()] = c_k

        for node in specification.nodes.values():
            for cat in node.semantic_categories:
                norm_c = cat.strip().lower()
                if norm_c in base_canon:
                    canonical_labels[norm_c] = list(base_canon[norm_c])
                elif norm_c in alias_to_base:
                    canon_k = alias_to_base[norm_c]
                    canonical_labels[canon_k] = list(base_canon[canon_k])
                else:
                    canonical_labels[norm_c] = [norm_c]

        for term in specification.detector_vocabulary:
            norm_t = term.strip().lower()
            if norm_t not in canonical_labels:
                if norm_t in alias_to_base:
                    canon_k = alias_to_base[norm_t]
                    if canon_k not in canonical_labels:
                        canonical_labels[canon_k] = list(base_canon[canon_k])
                else:
                    canonical_labels[norm_t] = [norm_t]

        vocab_dict = {
            "schema_version": 1,
            "canonical_labels": canonical_labels,
        }
        vocabulary_path.write_text(yaml.safe_dump(vocab_dict, sort_keys=False), encoding="utf-8")
    elif "semantic_vocabulary_path" in specification.metadata:
        vocabulary_path = Path(specification.metadata["semantic_vocabulary_path"])
    else:
        vocabulary_path = output_dir / "living_room_vocabulary.yaml"
        vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        vocabulary_path.write_text(yaml.safe_dump({"schema_version": 1, "canonical_labels": {}}, sort_keys=False), encoding="utf-8")

    detector, semantic_config = create_region_semantic_detector(
        checkpoint=str(LOCAL_MODEL), confidence_threshold=0.03,
        vocabulary_path=vocabulary_path,
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
    if observer is not None:
        overview_path = phase1 / "observation" / "initial_scene_overview.png"
        frame_path_str = str(overview_path) if overview_path.exists() else None
        observer("observation_updated", {
            "stage": "initial",
            "inspected_regions": [],
            "scene_graph": graph_o.to_dict(),
            "frame_path": frame_path_str,
        })

    ground_result = ground_graph(specification, graph_o, {"search_exhausted": True})
    if observer is not None:
        observer("grounding_updated", {
            "grounding": ground_result.to_dict(),
            "satisfied": bool(ground_result.complete),
            "status": ground_result.status,
            "scene_graph": graph_o.to_dict(),
        })

    (output_dir / "observed_scene_graph.json").write_text(
        json.dumps(graph_o.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "graph_grounding_result.json").write_text(
        json.dumps(ground_result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not ground_result.complete or not ground_result.assignment:
        return PipelineResult(
            domain="living_room", variant=variant_label, mode=mode,
            status=ground_result.status, failure_reason=str(
                ground_result.unsatisfied_relations or ground_result.missing_roles or "NO_GLOBAL_REGION_ASSIGNMENT"
            ),
        )

    # Sync canonical assignment phi into planner input using exact operation_bindings
    personal_bindings = (
        ground_result.operation_bindings.get("personal_support_group")
        or ground_result.operation_bindings.get("personal_support")
    )
    if not personal_bindings:
        for g_id, b_list in ground_result.operation_bindings.items():
            if b_list and any(b.get("target_id", "").startswith("personal_table_slot") for b in b_list):
                personal_bindings = b_list
                break

    canonical_assignments = []
    if personal_bindings:
        for binding in personal_bindings:
            reg_id = binding["tool_id"]
            slot_id = binding["target_id"]
            seat_id = binding.get("context", {}).get("SEATING_POSITION", "")
            matching_row = next((r for r in getattr(run, "personal_rows", []) if r["region_id"] == reg_id and r["slot_id"] == slot_id), None)
            if not matching_row and not seat_id:
                raise ValueError(f"Grounding operation binding {binding} cannot be mapped to evidence")

            reg_node = graph_o.nodes.get(reg_id)
            planar = reg_node.unary_predicates.get("PLANAR_SUPPORT", "UNKNOWN") if reg_node else "UNKNOWN"
            fit_rel = graph_o.get_relation("FITS_SET_ON", reg_id, slot_id)
            fits_set_on = fit_rel.status if fit_rel else "UNKNOWN"
            near_rel = graph_o.get_relation("NEAR_SEAT", reg_id, seat_id) if seat_id else None
            near_seat = near_rel.status if near_rel else "UNKNOWN"
            sem_status = matching_row.get("semantic_role_status", "UNKNOWN") if matching_row else "TRUE"
            comp_status = "TRUE" if (planar == "TRUE" and fits_set_on == "TRUE" and near_seat == "TRUE" and sem_status == "TRUE") else ("FALSE" if "FALSE" in (planar, fits_set_on, near_seat, sem_status) else "UNKNOWN")

            canonical_assignments.append({
                "function_id": "PERSONAL_CUP_SAUCER_REGION",
                "slot_id": slot_id,
                "region_id": reg_id,
                "payload_ids": matching_row.get("payload_ids", ["cup", "saucer"]) if matching_row else ["cup", "saucer"],
                "selected_compatibility_evidence": {
                    "compatibility_status": comp_status,
                    "FITS_SET_ON": fits_set_on,
                    "NEAR_SEAT": near_seat,
                    "PLANAR_SUPPORT": planar,
                    "semantic_role_status": sem_status,
                },
            })
    else:
        personal_regions = ground_result.assignment.get("PERSONAL_CUP_SAUCER_REGION", ground_result.assignment.get("PERSONAL_SUPPORT", []))
        if isinstance(personal_regions, str):
            personal_regions = [personal_regions]
        for idx, reg_id in enumerate(personal_regions, start=1):
            slot_id = f"personal_table_slot_{idx}"
            matching_row = next((r for r in getattr(run, "personal_rows", []) if r["region_id"] == reg_id and r["slot_id"] == slot_id), None)
            if not matching_row:
                raise ValueError(f"Cannot resolve personal region {reg_id} for slot {slot_id}")
            reg_node = graph_o.nodes.get(reg_id)
            planar = reg_node.unary_predicates.get("PLANAR_SUPPORT", "UNKNOWN") if reg_node else "UNKNOWN"
            fit_rel = graph_o.get_relation("FITS_SET_ON", reg_id, slot_id)
            fits_set_on = fit_rel.status if fit_rel else "UNKNOWN"
            seat_id = matching_row.get("seating_target_id", "")
            near_rel = graph_o.get_relation("NEAR_SEAT", reg_id, seat_id)
            near_seat = near_rel.status if near_rel else "UNKNOWN"
            sem_status = matching_row.get("semantic_role_status", "UNKNOWN")
            comp_status = "TRUE" if (planar == "TRUE" and fits_set_on == "TRUE" and near_seat == "TRUE" and sem_status == "TRUE") else ("FALSE" if "FALSE" in (planar, fits_set_on, near_seat, sem_status) else "UNKNOWN")

            canonical_assignments.append({
                "function_id": "PERSONAL_CUP_SAUCER_REGION",
                "slot_id": slot_id,
                "region_id": reg_id,
                "payload_ids": matching_row.get("payload_ids", [f"cup_{idx}", f"saucer_{idx}"]),
                "selected_compatibility_evidence": {
                    "compatibility_status": comp_status,
                    "FITS_SET_ON": fits_set_on,
                    "NEAR_SEAT": near_seat,
                    "PLANAR_SUPPORT": planar,
                    "semantic_role_status": sem_status,
                },
            })

    shared_region = (
        ground_result.assignment.get("SHARED_REMOTE_REGION")
        or ground_result.assignment.get("SHARED_SUPPORT")
        or ground_result.assignment.get("shared_remote")
    )
    if isinstance(shared_region, list) and shared_region:
        shared_region = shared_region[0]

    if shared_region:
        matching_shared = next((r for r in getattr(run, "shared_rows", []) if r["region_id"] == shared_region), {})
        remote_id = matching_shared.get("payload_ids", ["tv_remote"])[0] if matching_shared.get("payload_ids") else "tv_remote"

        reg_node = graph_o.nodes.get(shared_region)
        planar = reg_node.unary_predicates.get("PLANAR_SUPPORT", "UNKNOWN") if reg_node else "UNKNOWN"
        fit_rel = graph_o.get_relation("FITS_ON", shared_region, remote_id)
        fits_on = fit_rel.status if fit_rel else "UNKNOWN"
        acc_rel = graph_o.get_relation("ACCESSIBLE_FROM_BOTH_SEATS", shared_region, "SEATING_PAIR")
        access = acc_rel.status if acc_rel else "UNKNOWN"
        sem_status = matching_shared.get("semantic_role_status", "UNKNOWN")
        comp_status = "TRUE" if (planar == "TRUE" and fits_on == "TRUE" and access == "TRUE" and sem_status == "TRUE") else ("FALSE" if "FALSE" in (planar, fits_on, access, sem_status) else "UNKNOWN")

        canonical_assignments.append({
            "function_id": "SHARED_REMOTE_REGION",
            "slot_id": "shared_remote_slot",
            "region_id": shared_region,
            "payload_ids": matching_shared.get("payload_ids", ["tv_remote_1"]),
            "selected_compatibility_evidence": {
                "compatibility_status": comp_status,
                "FITS_ON": fits_on,
                "ACCESSIBLE_FROM_BOTH_SEATS": access,
                "PLANAR_SUPPORT": planar,
                "semantic_role_status": sem_status,
            },
        })

    # Write deterministic compiler/planner projection artifacts before invoking symbolic planner (note: these are compiler projections, not canonical phi*)
    (phase1 / "region_assignments.json").write_text(
        json.dumps({"assignments": canonical_assignments}, indent=2), encoding="utf-8"
    )
    (phase1 / "functional_region_witness.json").write_text(
        json.dumps({
            "status": "COMPLETE",
            "functional_requirements": canonical_assignments,
        }, indent=2), encoding="utf-8"
    )

    plan_dir = output_dir / "action_sequence"
    planning = run_living_room_symbolic_pipeline(phase1, plan_dir)
    plan_payload = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    actions = tuple({
        "action_index": index + 1,
        "operator": row["operator"],
        "arguments": list(row["arguments"].values()),
    } for index, row in enumerate(plan_payload["actions"]))
    planner_projection = {row["slot_id"]: row["region_id"] for row in canonical_assignments}
    (output_dir / "planner_projection.json").write_text(
        json.dumps(planner_projection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    from ..audit import audit_plan_grounding
    plan_audit = audit_plan_grounding(
        specification, graph_o, ground_result, list(actions), home_region="staging_tray"
    )
    (output_dir / "plan_grounding_audit.json").write_text(
        json.dumps(plan_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PipelineResult(
        domain="living_room", variant=variant_label, mode=mode,
        status="ACTION_SEQUENCE_READY", assignment=ground_result.assignment, plan=actions,
        search_statistics=planning.get("search_statistics", {}),
    )
