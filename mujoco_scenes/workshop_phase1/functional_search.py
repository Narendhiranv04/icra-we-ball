"""Joint functional requirement satisfaction search and infeasibility diagnosis."""

from __future__ import annotations

from typing import Any

from mujoco_scenes.workshop_phase1.geometric_grounding import GeometricGrounder
from mujoco_scenes.workshop_phase1.types import (
    FunctionGroundingResult,
    FunctionalRequirement,
    FunctionalWitness,
    GroundingStatus,
    ObservedObjectTrack,
    ObservedRegion,
    combine_status,
)


class FunctionalSatisfactionSearch:
    """Performs search over joint (driver, fastener, work_surface, parts_container) tuples."""

    def __init__(self, geometric_grounder: GeometricGrounder | None = None) -> None:
        self.geometric_grounder = geometric_grounder or GeometricGrounder()

    def check_profile_compatibility(
        self,
        driver_track: ObservedObjectTrack,
        fastener_track: ObservedObjectTrack,
    ) -> tuple[GroundingStatus, dict[str, Any]]:
        """Check measured local interfaces; semantic names are never read."""
        relation = self.geometric_grounder.evaluate_compatible_with(driver_track, fastener_track)
        self.geometric_grounder.relation_call_counts["COMPATIBLE_WITH"] += 1
        status = {
            "TRUE": GroundingStatus.PASS,
            "FALSE": GroundingStatus.FAIL,
        }.get(relation["status"], GroundingStatus.UNKNOWN)
        return status, relation

    def search_witness(
        self,
        driver_candidates: list[ObservedObjectTrack],
        fastener_candidates: list[ObservedObjectTrack],
        work_surface_candidates: list[ObservedRegion],
        parts_container_candidates: list[ObservedRegion],
        requirements: list[FunctionalRequirement],
    ) -> tuple[FunctionalWitness | None, list[dict[str, Any]]]:
        """Search all candidate 4-tuples for a jointly feasible assignment."""
        evaluated_tuples: list[dict[str, Any]] = []
        valid_witnesses: list[FunctionalWitness] = []

        required = {
            requirement.function_name: set(requirement.required_relations)
            for requirement in requirements
        }
        require_profile = "COMPATIBLE_WITH" in required.get("CAN_DRIVE_SCREW", set())
        require_packing = "FITS_SET_ON" in required.get("WORK_SURFACE", set())
        require_container_fit = "FITS_IN" in required.get("SMALL_PARTS_CONTAINER", set())

        for d in driver_candidates:
            for f in fastener_candidates:
                prof_status, prof_relation = (
                    self.check_profile_compatibility(d, f) if require_profile else
                    (GroundingStatus.PASS, {"relation": "COMPATIBLE_WITH", "status": "NOT_REQUIRED"})
                )
                if prof_status != GroundingStatus.PASS:
                    evaluated_tuples.append({
                        "driver": d.instance_id,
                        "fastener": f.instance_id,
                        "status": "UNRESOLVED_PROFILE" if prof_status == GroundingStatus.UNKNOWN else "REJECTED_PROFILE_MISMATCH",
                        "relation_evidence": prof_relation,
                    })
                    continue

                for w in work_surface_candidates:
                    fits, pack_details = (
                        self.geometric_grounder.check_relational_packing(d, f, w) if require_packing else
                        (True, {"relation": "FITS_SET_ON", "status": "NOT_REQUIRED"})
                    )
                    if not fits:
                        evaluated_tuples.append({
                            "driver": d.instance_id,
                            "fastener": f.instance_id,
                            "work_surface": w.region_instance_id,
                            "status": "REJECTED_PACKING",
                            "packing_details": pack_details,
                        })
                        continue

                    for c in parts_container_candidates:
                        fits_in = (
                            self.geometric_grounder.evaluate_fits_in(f, c) if require_container_fit else
                            {"relation": "FITS_IN", "status": "NOT_REQUIRED"}
                        )
                        if require_container_fit and fits_in["status"] != "TRUE":
                            evaluated_tuples.append({
                                "driver": d.instance_id, "fastener": f.instance_id,
                                "work_surface": w.region_instance_id,
                                "parts_container": c.region_instance_id,
                                "status": "UNRESOLVED_CONTAINER_FIT" if fits_in["status"] == "UNKNOWN" else "REJECTED_CONTAINER_FIT",
                                "relation_evidence": fits_in,
                            })
                            continue
                        conf = float(
                            d.overall_confidence
                            * f.overall_confidence
                            * w.current_semantic_belief.get("confidence", 1.0)
                            * c.current_semantic_belief.get("confidence", 1.0)
                        )
                        wit = FunctionalWitness(
                            driver_id=d.instance_id,
                            fastener_id=f.instance_id,
                            work_surface_id=w.region_instance_id,
                            parts_container_id=c.region_instance_id,
                            overall_confidence=conf,
                            verification_details={
                                "packing": pack_details,
                                "compatible_with": prof_relation,
                                "fits_in": fits_in,
                                "driver_usable_length_m": d.current_geometric_properties.get("usable_length_m"),
                                "fastener_maximum_cross_section_m": f.current_geometric_properties.get("maximum_cross_section_m"),
                                "driver_footprint_area_m2": d.current_geometric_properties.get("footprint_area_m2"),
                                "fastener_length_m": f.current_geometric_properties.get("total_length_m"),
                                "surface_support_area_m2": w.current_geometric_properties.get("support_area_m2"),
                                "driver_camera_count": len(d.contributing_cameras),
                                "fastener_camera_count": len(f.contributing_cameras),
                            },
                        )
                        valid_witnesses.append(wit)
                        evaluated_tuples.append({
                            "driver": d.instance_id,
                            "fastener": f.instance_id,
                            "work_surface": w.region_instance_id,
                            "parts_container": c.region_instance_id,
                            "status": "ACCEPTED",
                        })

        if valid_witnesses:
            def witness_rank(witness: FunctionalWitness) -> tuple[float, float, float, float, float, float, float, float]:
                details = witness.verification_details
                driver_views = float(details.get("driver_camera_count", 0))
                fastener_views = float(details.get("fastener_camera_count", 0))
                packing_margin = float(
                    details.get("packing", {}).get("selected_arrangement", {}).get(
                        "signed_margin_m", float("-inf")))
                footprint = details.get("driver_footprint_area_m2")
                footprint_rank = -float(footprint) if footprint is not None else float("-inf")
                driver_length = details.get("driver_usable_length_m")
                # Prefer the complete observation of a valid driver. Partial
                # handle/shaft proposals can barely satisfy reach but are not
                # a defensible full-object witness.
                driver_length_rank = float(driver_length) if driver_length is not None else float("-inf")
                fastener_cross = details.get("fastener_maximum_cross_section_m")
                fastener_cross_rank = -float(fastener_cross) if fastener_cross is not None else float("-inf")
                fastener_length = details.get("fastener_length_m")
                fastener_length_rank = float(fastener_length) if fastener_length is not None else float("-inf")
                return (
                    driver_length_rank,
                    fastener_length_rank,
                    fastener_cross_rank,
                    min(driver_views, fastener_views),
                    driver_views + fastener_views,
                    packing_margin,
                    footprint_rank,
                    witness.overall_confidence,
                )

            valid_witnesses.sort(key=witness_rank, reverse=True)
            return valid_witnesses[0], evaluated_tuples

        return None, evaluated_tuples

    def diagnose_infeasibility(
        self,
        all_objects: list[ObservedObjectTrack],
        all_regions: list[ObservedRegion],
        driver_candidates: list[ObservedObjectTrack],
        fastener_candidates: list[ObservedObjectTrack],
        work_surface_candidates: list[ObservedRegion],
        parts_container_candidates: list[ObservedRegion],
        evaluated_tuples: list[dict[str, Any]],
        has_unresolved_evidence: bool = False,
        grounding_results: list[FunctionGroundingResult] | None = None,
        requirements: list[FunctionalRequirement] | None = None,
    ) -> str:
        """Diagnose exact grounded infeasibility reason matching benchmark categories."""
        # If no objects were detected or all object evidence is missing/unresolved -> INSUFFICIENT_EVIDENCE
        if not all_objects:
            return "INSUFFICIENT_EVIDENCE"

        results = grounding_results or []
        if results and all(result.semantic_status == GroundingStatus.UNKNOWN for result in results):
            return "INSUFFICIENT_EVIDENCE"

        missing_driver = (len(driver_candidates) == 0)
        missing_fastener = (len(fastener_candidates) == 0)
        missing_surface = (len(work_surface_candidates) == 0)
        missing_container = (len(parts_container_candidates) == 0)

        def role_evidence(function_name: str) -> dict[str, Any]:
            role = [result for result in results if result.function_name == function_name]
            unresolved = any(
                result.semantic_status == GroundingStatus.UNKNOWN
                or (result.semantic_status == GroundingStatus.PASS
                    and result.geometric_status == GroundingStatus.UNKNOWN)
                for result in role)
            return {
                "semantic_pass": any(result.semantic_status == GroundingStatus.PASS for result in role),
                "unary_fail": any(result.semantic_status == GroundingStatus.PASS
                                  and result.geometric_status == GroundingStatus.FAIL for result in role),
                "unresolved": unresolved,
                "resolved_negative": bool(role) and not unresolved and all(
                    result.combined_status == GroundingStatus.FAIL for result in role),
                "failed_relations": {
                    relation.get("relation")
                    for result in role
                    for relation in result.geometric_evidence.get("relations", [])
                    if relation.get("status") == "FALSE"
                },
            }

        driver_evidence = role_evidence("CAN_DRIVE_SCREW")
        fastener_evidence = role_evidence("CAN_FASTEN")
        surface_evidence = role_evidence("WORK_SURFACE")
        container_evidence = role_evidence("SMALL_PARTS_CONTAINER")

        relations_cfg = self.geometric_grounder.config.get("relations", {})

        def conclusively_obstructed(region: ObservedRegion) -> bool:
            obstruction = region.obstruction_evidence
            elevated_points = int(obstruction.get("elevated_point_count", 0))
            support_points = int(region.support_plane.get("point_count", 0))
            obstruction_fraction = elevated_points / max(1, support_points)
            return (
                bool(obstruction.get("is_obstructed", False))
                and elevated_points >= int(
                    relations_cfg.get("obstruction_minimum_elevated_points", 500))
                and obstruction_fraction >= float(
                    relations_cfg.get("obstruction_minimum_elevated_fraction", 0.15))
            )

        all_surface_candidates_obstructed = (
            bool(work_surface_candidates)
            and all(conclusively_obstructed(region)
                    for region in work_surface_candidates)
        )

        # Check for tool geometry failure: semantic driver exists, but failed reach
        tool_geometry_minimum_semantic_confidence = float(
            self.geometric_grounder.config.get("relations", {}).get(
                "tool_geometry_minimum_semantic_confidence", 0.0))
        has_confident_short_driver = any(
            result.function_name == "CAN_DRIVE_SCREW"
            and result.semantic_status == GroundingStatus.PASS
            and float(result.semantic_evidence.get("confidence", 0.0))
            >= tool_geometry_minimum_semantic_confidence
            and any(
                relation.get("relation") == "REACHES_TARGET"
                and relation.get("status") == "FALSE"
                for relation in result.geometric_evidence.get("relations", []))
            for result in results
        )
        if (missing_driver and driver_evidence["semantic_pass"]
                and "REACHES_TARGET" in driver_evidence["failed_relations"]
                and has_confident_short_driver
                and not driver_evidence["unresolved"]):
            return "TOOL_GEOMETRY_FAILURE"

        # Diagnose structural region deficits before movable-object deficits.
        # Region evidence is aggregated from all five views and is therefore
        # more reliable than a missed small-tool proposal in an otherwise
        # conclusive no-surface/no-container scene.
        if (missing_surface and surface_evidence["resolved_negative"]
                or all_surface_candidates_obstructed):
            return "NO_WORK_SURFACE"

        if missing_container and container_evidence["resolved_negative"]:
            return "NO_PARTS_CONTAINER"

        # Generic required-role failure. No object categories are inspected here.
        if missing_driver and driver_evidence["resolved_negative"]:
            return "NO_VALID_DRIVER"

        if missing_fastener and fastener_evidence["resolved_negative"]:
            return "NO_VALID_FASTENER"

        unresolved_missing_role = any((
            missing_driver and not driver_evidence["resolved_negative"],
            missing_fastener and not fastener_evidence["resolved_negative"],
            missing_surface and not surface_evidence["resolved_negative"],
            missing_container and not container_evidence["resolved_negative"],
        ))
        if unresolved_missing_role:
            return "INSUFFICIENT_EVIDENCE"

        # Joint tuple failure analysis
        has_packing_rejection = any(t.get("status") == "REJECTED_PACKING" for t in evaluated_tuples)
        has_profile_rejection = any(t.get("status") == "REJECTED_PROFILE_MISMATCH" for t in evaluated_tuples)
        has_unresolved_tuple = any(str(t.get("status", "")).startswith("UNRESOLVED") for t in evaluated_tuples)

        if has_unresolved_tuple and not (has_packing_rejection or has_profile_rejection):
            return "INSUFFICIENT_EVIDENCE"

        # A packing rejection is emitted only after a driver/fastener pair has
        # already passed measured interface compatibility. It is therefore the
        # deepest verified failure, even if unrelated candidate pairs also
        # produce profile mismatches.
        if has_packing_rejection:
            return "OBJECT_REGION_PACKING_FAILURE"

        if len(driver_candidates) > 0 and len(fastener_candidates) > 0:
            return "GLOBAL_CONFLICT"

        if missing_driver:
            return "INSUFFICIENT_EVIDENCE"
        if missing_fastener:
            return "INSUFFICIENT_EVIDENCE"

        return "GLOBAL_CONFLICT" if evaluated_tuples else "INSUFFICIENT_EVIDENCE"
