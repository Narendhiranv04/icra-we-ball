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
    ) -> bool:
        """Check if driver drive profile mates with fastener drive recess/head."""
        d_belief = driver_track.current_semantic_belief
        f_belief = fastener_track.current_semantic_belief

        d_cat = d_belief.get("inferred_category", "HAND_DRIVER")
        d_prof = d_belief.get("drive_profile", "PHILLIPS")
        f_cat = f_belief.get("inferred_category", "SCREW")
        f_prof = f_belief.get("fastener_profile", "PHILLIPS_RECESS")

        # Power driver and standard Phillips driver mate with Phillips screws
        if d_prof == "PHILLIPS" and f_prof == "PHILLIPS_RECESS":
            return True
        if d_prof == "SLOTTED" and f_prof == "SLOTTED_RECESS":
            return True
        if d_prof == "HEX" and f_prof == "HEX_HEAD":
            return True
        return False

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
                # 1. Profile compatibility
                prof_compat = self.check_profile_compatibility(d, f)
                if not prof_compat:
                    evaluated_tuples.append({
                        "driver": d.instance_id,
                        "fastener": f.instance_id,
                        "status": "REJECTED_PROFILE_MISMATCH",
                    })
                    continue

                for w in work_surface_candidates:
                    # 2. Relational packing
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
                        # All individual and joint constraints satisfied!
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
                                "driver_category": d.current_semantic_belief.get("inferred_category"),
                                "driver_reach_m": d.current_geometric_properties.get("usable_reach_m"),
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
            # Sort by overall confidence
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
    ) -> str:
        """Diagnose exact grounded infeasibility reason matching the 7 benchmark failure categories."""
        # 1. Driver checks
        if not driver_candidates:
            # Check if driver-like tools were observed that failed reach geometry
            driver_like = [
                trk for trk in all_objects
                if trk.current_semantic_belief.get("inferred_category") in ("HAND_DRIVER", "POWER_DRIVER")
                or "driver" in trk.current_semantic_belief.get("initial_label", "").lower()
                or "screwdriver" in trk.current_semantic_belief.get("initial_label", "").lower()
                or trk.current_semantic_belief.get("drive_profile") in ("PHILLIPS", "SLOTTED")
            ]
            if driver_like:
                # We had driver candidates, but all failed reach geometry
                return "TOOL_GEOMETRY_FAILURE"
            return "NO_VALID_DRIVER"

        # 2. Fastener checks
        if not fastener_candidates:
            return "NO_VALID_FASTENER"

        # 3. Work surface checks
        if not work_surface_candidates:
            return "NO_WORK_SURFACE"

        # 4. Container checks
        if not parts_container_candidates:
            return "NO_PARTS_CONTAINER"

        # 5. Joint tuple failure analysis
        has_packing_rejection = any(t.get("status") == "REJECTED_PACKING" for t in evaluated_tuples)
        has_profile_rejection = any(t.get("status") == "REJECTED_PROFILE_MISMATCH" for t in evaluated_tuples)

        if has_packing_rejection and not has_profile_rejection:
            return "OBJECT_REGION_PACKING_FAILURE"

        if has_profile_rejection or (has_packing_rejection and has_profile_rejection):
            return "GLOBAL_CONFLICT"

        return "GLOBAL_CONFLICT"
