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
    ) -> tuple[GroundingStatus, str]:
        """Check if driver interface mates with fastener interface."""
        d_belief = driver_track.current_semantic_belief
        f_belief = fastener_track.current_semantic_belief

        d_cat = str(d_belief.get("canonical_label", "")).lower()
        f_cat = str(f_belief.get("canonical_label", "")).lower()
        d_raw = str(d_belief.get("raw_label", "")).lower()
        f_raw = str(f_belief.get("raw_label", "")).lower()

        d_geo = driver_track.current_geometric_properties.get("interface_geometry", "UNKNOWN")
        f_geo = fastener_track.current_geometric_properties.get("interface_geometry", "UNKNOWN")

        is_slotted_driver = ("flathead" in d_cat or "flathead" in d_raw or "slotted" in d_cat or "slotted" in d_raw or d_geo == "SLOT_LIKE")
        is_cross_screw = ("phillips" in f_cat or "phillips" in f_raw or "cross" in f_raw or f_geo == "CROSS_RECESS" or f_cat == "screw")

        # Flathead/slotted driver cannot drive cross-recess screw
        if is_slotted_driver and is_cross_screw:
            return GroundingStatus.FAIL, "SLOT_DRIVER_MISMATCH_WITH_CROSS_RECESS_SCREW"

        # Wrench/pliers cannot drive machine screw
        if d_cat in ("wrench", "pliers") or "pliers" in d_raw or "wrench" in d_raw:
            return GroundingStatus.FAIL, "WRENCH_PLIERS_INCOMPATIBLE_WITH_SCREW"

        # Hex driver with hex bolt
        if ("hex" in d_cat or "hex" in d_raw) and ("bolt" in f_cat or "bolt" in f_raw):
            return GroundingStatus.PASS, "COMPATIBLE_HEX_DRIVER_AND_BOLT"

        # Screwdriver / power driver with cross or symmetric bit driving screw
        if (d_cat in ("screwdriver", "power_driver") or "driver" in d_raw) and ("screw" in f_cat or "screw" in f_raw):
            return GroundingStatus.PASS, "COMPATIBLE_FASTENER_DRIVER_MATCH"

        if not d_cat or not f_cat or d_cat == "unknown" or f_cat == "unknown":
            return GroundingStatus.UNKNOWN, "UNKNOWN_INTERFACE_COMPATIBILITY"

        return GroundingStatus.FAIL, f"INTERFACE_MISMATCH_{d_cat.upper()}_VS_{f_cat.upper()}"

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
                prof_status, prof_reason = self.check_profile_compatibility(d, f)
                if prof_status != GroundingStatus.PASS:
                    evaluated_tuples.append({
                        "driver": d.instance_id,
                        "fastener": f.instance_id,
                        "status": "UNRESOLVED_PROFILE" if prof_status == GroundingStatus.UNKNOWN else "REJECTED_PROFILE_MISMATCH",
                        "reason": prof_reason,
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
                                "driver_category": d.current_semantic_belief.get("canonical_label"),
                                "driver_reach_m": d.current_geometric_properties.get("usable_reach_m"),
                                "fastener_category": f.current_semantic_belief.get("canonical_label"),
                                "fastener_length_m": f.current_geometric_properties.get("length_m"),
                                "surface_usable_area_m2": w.current_geometric_properties.get("usable_area_m2"),
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

        if missing_fastener and ("bolt" in observed_categories):
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
