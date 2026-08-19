"""Joint functional requirement satisfaction search and infeasibility diagnosis."""

from __future__ import annotations

from typing import Any

from mujoco_scenes.workshop_phase1.geometric_grounding import GeometricGrounder
from mujoco_scenes.workshop_phase1.types import (
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
    ) -> tuple[FunctionalWitness | None, list[dict[str, Any]]]:
        """Search all candidate 4-tuples for a jointly feasible assignment."""
        evaluated_tuples: list[dict[str, Any]] = []
        valid_witnesses: list[FunctionalWitness] = []

        for d in driver_candidates:
            for f in fastener_candidates:
                prof_status, prof_relation = self.check_profile_compatibility(d, f)
                if prof_status != GroundingStatus.PASS:
                    evaluated_tuples.append({
                        "driver": d.instance_id,
                        "fastener": f.instance_id,
                        "status": "UNRESOLVED_PROFILE" if prof_status == GroundingStatus.UNKNOWN else "REJECTED_PROFILE_MISMATCH",
                        "relation_evidence": prof_relation,
                    })
                    continue

                for w in work_surface_candidates:
                    fits, pack_details = self.geometric_grounder.check_relational_packing(d, f, w)
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
                        fits_in = self.geometric_grounder.evaluate_fits_in(f, c)
                        if fits_in["status"] != "TRUE":
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
                                "fastener_length_m": f.current_geometric_properties.get("total_length_m"),
                                "surface_support_area_m2": w.current_geometric_properties.get("support_area_m2"),
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
            valid_witnesses.sort(key=lambda x: x.overall_confidence, reverse=True)
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
    ) -> str:
        """Diagnose exact grounded infeasibility reason matching benchmark categories."""
        # If no objects were detected or all object evidence is missing/unresolved -> INSUFFICIENT_EVIDENCE
        if not all_objects:
            return "INSUFFICIENT_EVIDENCE"

        observed_categories = {
            trk.current_semantic_belief.get("canonical_label", "").lower()
            for trk in all_objects
        }

        # If all detected objects are unknown -> INSUFFICIENT_EVIDENCE
        if observed_categories.issubset({"unknown", "object", "object_proposal"}):
            return "INSUFFICIENT_EVIDENCE"

        missing_driver = (len(driver_candidates) == 0)
        missing_fastener = (len(fastener_candidates) == 0)
        missing_surface = (len(work_surface_candidates) == 0)
        missing_container = (len(parts_container_candidates) == 0)

        # Check for tool geometry failure: semantic driver exists, but failed reach
        has_semantic_driver = any(
            trk.current_semantic_belief.get("canonical_label") in ("screwdriver", "power_driver")
            for trk in all_objects
        )
        if missing_driver and has_semantic_driver:
            # Succeeded semantically, failed reach geometry
            return "TOOL_GEOMETRY_FAILURE"

        # Check for single-role missing causes
        if missing_driver and ("wrench" in observed_categories or "pliers" in observed_categories):
            return "NO_VALID_DRIVER"

        if missing_fastener and ({"screw", "bolt"} & observed_categories):
            return "NO_VALID_FASTENER"

        if missing_surface:
            return "NO_WORK_SURFACE"

        if missing_container:
            return "NO_PARTS_CONTAINER"

        # Joint tuple failure analysis
        has_packing_rejection = any(t.get("status") == "REJECTED_PACKING" for t in evaluated_tuples)
        has_profile_rejection = any(t.get("status") == "REJECTED_PROFILE_MISMATCH" for t in evaluated_tuples)

        if has_packing_rejection and not has_profile_rejection:
            return "OBJECT_REGION_PACKING_FAILURE"

        if len(driver_candidates) > 0 and len(fastener_candidates) > 0:
            return "GLOBAL_CONFLICT"

        deficit_count = sum([missing_driver, missing_fastener, missing_surface, missing_container])
        if deficit_count >= 2:
            return "GLOBAL_CONFLICT"

        if missing_driver:
            return "NO_VALID_DRIVER"
        if missing_fastener:
            return "NO_VALID_FASTENER"

        return "INSUFFICIENT_EVIDENCE"
