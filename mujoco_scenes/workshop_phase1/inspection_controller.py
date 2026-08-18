"""Incremental inspection controller and Phase 1 orchestration for Workshop (W1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_scenes.workshop_phase1.capture import ProductionInspectionCapture
from mujoco_scenes.workshop_phase1.evidence_graph import GrowingObservedGraph
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter
from mujoco_scenes.workshop_phase1.functional_search import FunctionalSatisfactionSearch
from mujoco_scenes.workshop_phase1.geometric_grounding import GeometricGrounder
from mujoco_scenes.workshop_phase1.perception import (
    InstanceProposalBackend,
    PrivilegedOracleMaskBackend,
    RGBDConnectedComponentProposalBackend,
    YOLOWorldProposalBackend,
)
from mujoco_scenes.workshop_phase1.region_grounding import RegionGrounder
from mujoco_scenes.workshop_phase1.requirements import (
    RequirementProvider,
    StaticWorkshopRequirementProvider,
)
from mujoco_scenes.workshop_phase1.semantic_grounding import SemanticGrounder
from mujoco_scenes.workshop_phase1.serialization import write_production_json
from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker
from mujoco_scenes.workshop_phase1.types import (
    EpisodeResult,
    FunctionalRequirement,
    FunctionalWitness,
    GroundingStatus,
    InspectionDecision,
    InspectionTrace,
    MaskBackendType,
    ObservedObjectTrack,
    ObservedRegion,
)

INSPECTION_SEQUENCE = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")


class WorkshopPhase1InspectionController:
    """Orchestrates Phase 1 perception, tracking, grounding, joint search, and incremental inspection."""

    def __init__(
        self,
        mask_backend: MaskBackendType = MaskBackendType.PRODUCTION,
        requirements_provider: RequirementProvider | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.mask_backend_type = mask_backend
        self.requirements_provider = requirements_provider or StaticWorkshopRequirementProvider()
        self.output_dir = output_dir

        self.capture = ProductionInspectionCapture()
        self.tracker = PersistentInstanceTracker()
        self.graph = GrowingObservedGraph()
        self.semantic_grounder = SemanticGrounder()
        self.geometric_grounder = GeometricGrounder()
        self.region_grounder = RegionGrounder()
        self.search = FunctionalSatisfactionSearch(geometric_grounder=self.geometric_grounder)
        self.fm_adapter = FMAdapter()

    def _get_proposal_backend(self, scene: Any) -> InstanceProposalBackend:
        if self.mask_backend_type == MaskBackendType.ORACLE:
            return PrivilegedOracleMaskBackend(scene)
        elif self.mask_backend_type == MaskBackendType.CONNECTED_COMPONENT:
            return RGBDConnectedComponentProposalBackend()
        else:
            return YOLOWorldProposalBackend()

    def run_episode(self, scene: Any) -> EpisodeResult:
        """Run a complete Phase 1 grounding episode on the provided scene."""
        start_time = time.perf_counter()
        proposal_backend = self._get_proposal_backend(scene)

        # 1. Obtain broad functional requirements
        requirements = self.requirements_provider.get_requirements()
        req_map = {r.function_name: r for r in requirements}

        # 2. Register initial scene graph nodes
        for r_name in INSPECTION_SEQUENCE:
            self.graph.register_inspection_region_node(r_name, f"Storage container {r_name}")
        self.graph.register_workpiece_node("workpiece_frame_joint_0001")

        trace = InspectionTrace()
        stage_idx = 0
        witness: FunctionalWitness | None = None
        all_regions: list[ObservedRegion] = []
        grounding_results_all: list[dict[str, Any]] = []
        all_evaluated_tuples: list[dict[str, Any]] = []

        is_oracle_mode = (self.mask_backend_type == MaskBackendType.ORACLE)

        # 3. Stage 0: INITIAL observation (tabletop + candidate regions)
        init_rig = self.capture.get_stage_rig_config("INITIAL")
        init_vol_cfg = init_rig.get("inspection_volume", init_rig.get("inspection_volume_m", {}))
        init_vol_min = np.array(init_vol_cfg.get("minimum_world_m", [-1.20, -0.15, 0.60]))
        init_vol_max = np.array(init_vol_cfg.get("maximum_world_m", [1.20, 0.85, 1.50]))

        init_obs = self.capture.capture_stage(scene, "INITIAL", capture_segmentation=is_oracle_mode)
        for obs in init_obs:
            obs.detected_masks = proposal_backend.predict(obs, init_vol_min, init_vol_max)

        # Discover candidate regions from initial stage
        all_regions = self.region_grounder.discover_candidate_regions(scene, init_obs)
        self.graph.update_from_observed_regions(all_regions, stage_idx=0)

        # Track visible tabletop objects
        affected_tracks = self.tracker.update_with_stage_observations(
            stage_index=0,
            source_region_id="INITIAL_TABLETOP",
            observations=init_obs,
            stage_volume_min=init_vol_min,
            stage_volume_max=init_vol_max,
        )
        self.graph.update_from_object_tracks(affected_tracks, stage_idx=0, stage_region_id="INITIAL_TABLETOP")
        self.graph.snapshot(stage_idx=0, output_dir=self.output_dir)

        trace.steps.append(
            InspectionDecision(
                stage_index=0,
                inspection_region_id="INITIAL",
                action="INITIAL_SURVEY",
                rationale="Survey open workspace and identify staging regions and exposed objects",
            )
        )

        # Grounding & search after Stage 0
        witness, evaluated_tuples, g_res = self._ground_and_search(req_map, all_regions)
        grounding_results_all.extend(g_res)
        all_evaluated_tuples.extend(evaluated_tuples)

        if witness is not None:
            trace.early_stopped = True
        else:
            # 4. Incremental inspection over storage containers
            for reg_name in INSPECTION_SEQUENCE:
                stage_idx += 1
                trace.inspected_regions.append(reg_name)
                trace.steps.append(
                    InspectionDecision(
                        stage_index=stage_idx,
                        inspection_region_id=reg_name,
                        action="INSPECT",
                        rationale=f"Open and inspect storage container {reg_name} for missing functional items",
                    )
                )

                # Open container in scene
                scene.open_container(reg_name)

                # Capture fresh calibrated 5-view observation
                stage_rig = self.capture.get_stage_rig_config(reg_name)
                vol_cfg = stage_rig.get("inspection_volume", stage_rig.get("inspection_volume_m", {}))
                vol_min = np.array(vol_cfg.get("minimum_world_m", [-1.20, -0.30, 0.35]))
                vol_max = np.array(vol_cfg.get("maximum_world_m", [1.20, 0.85, 1.50]))

                stage_obs = self.capture.capture_stage(scene, reg_name, capture_segmentation=is_oracle_mode)
                for obs in stage_obs:
                    obs.detected_masks = proposal_backend.predict(obs, vol_min, vol_max)

                # Update persistent tracks
                stage_tracks = self.tracker.update_with_stage_observations(
                    stage_index=stage_idx,
                    source_region_id=reg_name,
                    observations=stage_obs,
                    stage_volume_min=vol_min,
                    stage_volume_max=vol_max,
                )
                self.graph.update_from_object_tracks(stage_tracks, stage_idx=stage_idx, stage_region_id=reg_name)
                self.graph.snapshot(stage_idx=stage_idx, output_dir=self.output_dir)

                # Re-ground and search
                witness, evaluated_tuples, g_res = self._ground_and_search(req_map, all_regions)
                grounding_results_all.extend(g_res)
                all_evaluated_tuples.extend(evaluated_tuples)

                if witness is not None:
                    trace.early_stopped = True
                    break

        total_time = time.perf_counter() - start_time
        trace.total_stages = stage_idx + 1

        # 5. Final diagnosis
        all_tracks = list(self.tracker.tracks.values())
        if witness is not None:
            status = "FEASIBLE"
            rejection_reason = None
        else:
            status = "INFEASIBLE"
            drivers, fasteners, surfaces, containers = self._filter_valid_candidates(req_map, all_regions)
            rejection_reason = self.search.diagnose_infeasibility(
                all_objects=all_tracks,
                all_regions=all_regions,
                driver_candidates=drivers,
                fastener_candidates=fasteners,
                work_surface_candidates=surfaces,
                parts_container_candidates=containers,
                evaluated_tuples=all_evaluated_tuples,
            )

        metrics = {
            "total_inference_time_s": round(total_time, 4),
            "discovered_object_count": len(all_tracks),
            "discovered_region_count": len(all_regions),
            "stages_executed": trace.total_stages,
            "inspected_storage_regions": trace.inspected_regions,
            "early_stopped": trace.early_stopped,
            "semantic_model_calls": self.semantic_grounder.total_semantic_calls,
            "geometric_model_calls": self.geometric_grounder.total_geometric_calls,
            "fm_requirement_calls": self.fm_adapter.metrics.requirement_calls,
            "fm_search_prior_calls": self.fm_adapter.metrics.search_prior_calls,
            "total_fm_calls": self.fm_adapter.metrics.total_calls,
        }

        result = EpisodeResult(
            status=status,
            rejection_reason=rejection_reason,
            witness=witness,
            trace=trace,
            metrics=metrics,
            diagnostics={
                "evaluated_tuple_count": len(all_evaluated_tuples),
                "evaluated_tuples": all_evaluated_tuples,
            },
        )

        # 6. Save episode artifacts
        if self.output_dir is not None:
            self._save_artifacts(
                requirements=requirements,
                trace=trace,
                tracks=all_tracks,
                regions=all_regions,
                grounding_results=grounding_results_all,
                evaluated_tuples=all_evaluated_tuples,
                witness=witness,
                rejection_reason=rejection_reason,
                result=result,
                metrics=metrics,
            )

        return result

    def _ground_and_search(
        self,
        req_map: dict[str, FunctionalRequirement],
        regions: list[ObservedRegion],
    ) -> tuple[FunctionalWitness | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """Perform semantic & geometric grounding across all current tracks/regions and search for valid witness."""
        all_tracks = list(self.tracker.tracks.values())
        drivers, fasteners, surfaces, containers = self._filter_valid_candidates(req_map, regions)

        witness, evaluated_tuples = self.search.search_witness(
            driver_candidates=drivers,
            fastener_candidates=fasteners,
            work_surface_candidates=surfaces,
            parts_container_candidates=containers,
        )

        # Record grounding entries
        grounding_entries: list[dict[str, Any]] = []
        for trk in all_tracks:
            for req in req_map.values():
                if req.entity_type.value == "OBJECT":
                    s_res = self.semantic_grounder.ground_object_for_requirement(trk, req)
                    g_res = self.geometric_grounder.ground_object_geometry(trk, req)
                    comb_status = (
                        GroundingStatus.PASS
                        if (s_res.semantic_status == GroundingStatus.PASS and g_res.geometric_status == GroundingStatus.PASS)
                        else GroundingStatus.FAIL
                    )
                    grounding_entries.append({
                        "entity_id": trk.instance_id,
                        "function": req.function_name,
                        "semantic_status": s_res.semantic_status.value,
                        "geometric_status": g_res.geometric_status.value,
                        "combined_status": comb_status.value,
                        "rejections": s_res.rejection_reasons + g_res.rejection_reasons,
                    })

        return witness, evaluated_tuples, grounding_entries

    def _filter_valid_candidates(
        self,
        req_map: dict[str, FunctionalRequirement],
        regions: list[ObservedRegion],
    ) -> tuple[list[ObservedObjectTrack], list[ObservedObjectTrack], list[ObservedRegion], list[ObservedRegion]]:
        """Filter tracks and regions that strictly PASS both semantic and geometric checks."""
        all_tracks = list(self.tracker.tracks.values())
        drivers: list[ObservedObjectTrack] = []
        fasteners: list[ObservedObjectTrack] = []
        surfaces: list[ObservedRegion] = []
        containers: list[ObservedRegion] = []

        driver_req = req_map.get("CAN_DRIVE_SCREW")
        fastener_req = req_map.get("CAN_FASTEN")
        surface_req = req_map.get("WORK_SURFACE")
        container_req = req_map.get("SMALL_PARTS_CONTAINER")

        if driver_req:
            for trk in all_tracks:
                s_res = self.semantic_grounder.ground_object_for_requirement(trk, driver_req)
                g_res = self.geometric_grounder.ground_object_geometry(trk, driver_req)
                if s_res.semantic_status == GroundingStatus.PASS and g_res.geometric_status == GroundingStatus.PASS:
                    drivers.append(trk)

        if fastener_req:
            for trk in all_tracks:
                s_res = self.semantic_grounder.ground_object_for_requirement(trk, fastener_req)
                g_res = self.geometric_grounder.ground_object_geometry(trk, fastener_req)
                if s_res.semantic_status == GroundingStatus.PASS and g_res.geometric_status == GroundingStatus.PASS:
                    fasteners.append(trk)

        if surface_req:
            for reg in regions:
                s_res = self.semantic_grounder.ground_region_for_requirement(reg, surface_req)
                g_res = self.geometric_grounder.ground_region_geometry(reg, surface_req)
                if s_res.semantic_status == GroundingStatus.PASS and g_res.geometric_status == GroundingStatus.PASS:
                    surfaces.append(reg)

        if container_req:
            for reg in regions:
                s_res = self.semantic_grounder.ground_region_for_requirement(reg, container_req)
                g_res = self.geometric_grounder.ground_region_geometry(reg, container_req)
                if s_res.semantic_status == GroundingStatus.PASS and g_res.geometric_status == GroundingStatus.PASS:
                    containers.append(reg)

        return drivers, fasteners, surfaces, containers

    def _save_artifacts(
        self,
        requirements: list[FunctionalRequirement],
        trace: InspectionTrace,
        tracks: list[ObservedObjectTrack],
        regions: list[ObservedRegion],
        grounding_results: list[dict[str, Any]],
        evaluated_tuples: list[dict[str, Any]],
        witness: FunctionalWitness | None,
        rejection_reason: str | None,
        result: EpisodeResult,
        metrics: dict[str, Any],
    ) -> None:
        """Write all episode artifacts ensuring leak-free sanitization."""
        out = self.output_dir
        if out is None:
            return
        out.mkdir(parents=True, exist_ok=True)

        write_production_json([r.to_dict() for r in requirements], out / "task_requirements.json")
        write_production_json([s.__dict__ for s in trace.steps], out / "inspection_trace.json")
        write_production_json([t.to_dict() for t in tracks], out / "tracks.json")
        write_production_json([r.to_dict() for r in regions], out / "regions.json")
        write_production_json(grounding_results, out / "grounding_results.json")
        write_production_json(evaluated_tuples, out / "functional_search.json")

        if witness is not None:
            write_production_json(witness.to_dict(), out / "witness.json")
        else:
            write_production_json({"status": "INFEASIBLE", "rejection_reason": rejection_reason}, out / "infeasibility.json")

        write_production_json(result.to_dict(), out / "episode_summary.json")
        write_production_json(metrics, out / "timings.json")
