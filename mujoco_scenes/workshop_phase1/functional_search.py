"""Joint functional requirement satisfaction search and infeasibility diagnosis."""

from __future__ import annotations

from typing import Any

from mujoco_scenes.workshop_phase1.geometric_grounding import GeometricGrounder
from mujoco_scenes.workshop_phase1.types import (
    FunctionalWitness,
    GroundingStatus,
    ObservedObjectTrack,
    ObservedRegion,
)


class FunctionalSatisfactionSearch:
    """Performs exhaustive search over joint (driver, fastener, work_surface, parts_container) tuples."""

    def __init__(self, geometric_grounder: GeometricGrounder | None = None) -> None:
        self.geometric_grounder = geometric_grounder or GeometricGrounder()

    def check_profile_compatibility(
        self,
        driver_track: ObservedObjectTrack,
        fastener_track: ObservedObjectTrack,
    ) -> tuple[bool, str]:
        """Check if driver drive profile mates with fastener drive recess/head."""
        d_belief = driver_track.current_semantic_belief
        f_belief = fastener_track.current_semantic_belief

        d_interface = d_belief.get("normalized_tool_interface", "UNKNOWN_INTERFACE")
        f_interface = f_belief.get("normalized_fastener_interface", "UNKNOWN_INTERFACE")

        if d_interface == "CROSS_RECESS" and f_interface == "CROSS_RECESS":
            return True, "MATCHED_CROSS_RECESS"
        if d_interface == "SINGLE_SLOT" and f_interface == "SINGLE_SLOT":
            return True, "MATCHED_SINGLE_SLOT"
        if d_interface == "HEX_SOCKET" and f_interface == "HEX_HEAD":
            return True, "MATCHED_HEX"

        if d_interface == "UNKNOWN_INTERFACE" or f_interface == "UNKNOWN_INTERFACE":
            return False, "UNKNOWN_INTERFACE_COMPATIBILITY"

        return False, f"INTERFACE_MISMATCH_{d_interface}_VS_{f_interface}"

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
                prof_compat, prof_reason = self.check_profile_compatibility(d, f)
                if not prof_compat:
                    evaluated_tuples.append({
                        "driver": d.instance_id,
                        "fastener": f.instance_id,
                        "status": "REJECTED_PROFILE_MISMATCH",
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
                            * 1.0
                            * 1.0
                        )
                        wit = FunctionalWitness(
                            driver_id=d.instance_id,
                            fastener_id=f.instance_id,
                            work_surface_id=w.region_instance_id,
                            parts_container_id=c.region_instance_id,
                            overall_confidence=conf,
                            verification_details={
                                "packing": pack_details,
                                "driver_role": d.current_semantic_belief.get("inferred_role"),
                                "driver_reach_m": d.current_geometric_properties.get("usable_reach_m"),
                                "fastener_role": f.current_semantic_belief.get("inferred_role"),
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
        """Diagnose exact grounded infeasibility reason matching the 7 benchmark failure categories."""
        if has_unresolved_evidence and not all_objects:
            return "INSUFFICIENT_EVIDENCE"

        has_cross_driver_semantic = any(
            trk.current_semantic_belief.get("normalized_tool_interface") == "CROSS_RECESS"
            for trk in all_objects
        )
        has_cross_fastener_semantic = any(
            trk.current_semantic_belief.get("normalized_fastener_interface") == "CROSS_RECESS"
            for trk in all_objects
        )

        missing_driver = (len(driver_candidates) == 0)
        missing_fastener = (len(fastener_candidates) == 0)
        missing_surface = (len(work_surface_candidates) == 0)
        missing_container = (len(parts_container_candidates) == 0)

        deficit_count = sum([missing_driver, missing_fastener, missing_surface, missing_container])

        if deficit_count >= 2:
            return "GLOBAL_CONFLICT"

        if missing_driver:
            has_long_driver = any(
                trk.current_geometric_properties.get("total_length_m", 0.0) >= 0.14
                and trk.current_semantic_belief.get("broad_object_type") in ("manual screwdriver", "power driver")
                for trk in all_objects
            )
            if has_cross_driver_semantic and has_long_driver:
                return "GLOBAL_CONFLICT"
            elif has_cross_driver_semantic:
                return "TOOL_GEOMETRY_FAILURE"
            return "NO_VALID_DRIVER"

        if missing_fastener:
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

        return "GLOBAL_CONFLICT"
