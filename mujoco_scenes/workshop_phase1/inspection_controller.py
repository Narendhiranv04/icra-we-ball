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
from mujoco_scenes.workshop_phase1.semantic_grounding import (
    ObjectSemanticBackend,
    PrivilegedOracleSemanticBackend,
    ProductionSemanticBackend,
    SemanticGrounder,
)
from mujoco_scenes.workshop_phase1.serialization import write_production_json
from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker
from mujoco_scenes.workshop_phase1.types import (
    AblationType,
    EpisodeResult,
    FunctionalRequirement,
    FunctionalWitness,
    GroundingStatus,
    InspectionDecision,
    InspectionTrace,
    MaskBackendType,
    ObservedObjectTrack,
    ObservedRegion,
    SemanticBackendType,
)

INSPECTION_SEQUENCE = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")


class WorkshopPhase1InspectionController:
    """Orchestrates Phase 1 perception, tracking, grounding, joint search, and incremental inspection."""

    def __init__(
        self,
        mask_backend: MaskBackendType = MaskBackendType.PRODUCTION,
        semantic_backend: SemanticBackendType = SemanticBackendType.PRODUCTION,
        ablation: AblationType = AblationType.NONE,
        requirements_provider: RequirementProvider | None = None,
        requirements_source: str = "static",
        inspection_policy: str = "fixed_sequence",
        output_dir: Path | None = None,
    ) -> None:
        self.mask_backend_type = mask_backend
        self.semantic_backend_type = semantic_backend
        self.ablation = ablation
        self.requirements_source = requirements_source
        self.inspection_policy = inspection_policy
        self.requirements_provider = requirements_provider or StaticWorkshopRequirementProvider()
        self.output_dir = output_dir

        # Apply ablation overrides
        if self.ablation == AblationType.ORACLE_MASK:
            self.mask_backend_type = MaskBackendType.ORACLE
        elif self.ablation == AblationType.ORACLE_SEMANTICS:
            self.semantic_backend_type = SemanticBackendType.ORACLE

        self.capture = ProductionInspectionCapture()
        self.tracker = PersistentInstanceTracker()
        self.graph = GrowingObservedGraph()
        self.region_grounder = RegionGrounder()
        self.fm_adapter = FMAdapter()

    def _get_proposal_backend(self, scene: Any) -> InstanceProposalBackend:
        if self.mask_backend_type == MaskBackendType.ORACLE:
            return PrivilegedOracleMaskBackend(scene)
        elif self.mask_backend_type == MaskBackendType.CONNECTED_COMPONENT:
            return RGBDConnectedComponentProposalBackend()
        else:
            return YOLOWorldProposalBackend()

    def _get_semantic_grounder(self, scene: Any) -> SemanticGrounder:
        if self.semantic_backend_type == SemanticBackendType.ORACLE:
            backend: ObjectSemanticBackend = PrivilegedOracleSemanticBackend(scene)
        else:
            backend = ProductionSemanticBackend()
        return SemanticGrounder(backend=backend)

    def run_episode(self, scene: Any) -> EpisodeResult:
        """Run a complete Phase 1 grounding episode on the provided scene."""
        start_time = time.perf_counter()
        proposal_backend = self._get_proposal_backend(scene)
        semantic_grounder = self._get_semantic_grounder(scene)

        # 1. Obtain broad functional requirements
        if self.requirements_source == "fm":
            task_desc = getattr(scene, "task_instruction", "Repair frame joint assembly in workshop")
            fm_reqs = self.fm_adapter.generate_task_requirements(task_desc)
            # Parse FM output to FunctionalRequirement instances
            from mujoco_scenes.workshop_phase1.types import EntityType
            requirements: list[FunctionalRequirement] = []
            for item in fm_reqs.get("object_functions", []):
                requirements.append(
                    FunctionalRequirement(
                        requirement_id=f"req_{item['name'].lower()}",
                        entity_type=EntityType.OBJECT,
                        function_name=item["name"],
                        description=item["description"],
                        priority=item.get("rank", 1),
                    )
                )
            for item in fm_reqs.get("region_functions", []):
                requirements.append(
                    FunctionalRequirement(
                        requirement_id=f"req_{item['name'].lower()}",
                        entity_type=EntityType.FUNCTIONAL_REGION,
                        function_name=item["name"],
                        description=item["description"],
                        priority=item.get("rank", 1),
                    )
                )
        else:
            requirements = self.requirements_provider.get_requirements()

        req_map = {r.function_name: r for r in requirements}
        proposal_backend.set_requirements(requirements, getattr(scene, "task_instruction", ""))

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

        is_oracle_mask = (self.mask_backend_type == MaskBackendType.ORACLE)

        # 3. Stage 0: INITIAL observation (tabletop + candidate regions)
        init_rig = self.capture.get_stage_rig_config("INITIAL")
        init_vol_cfg = init_rig.get("inspection_volume", init_rig.get("inspection_volume_m", {}))
        init_vol_min = np.array(init_vol_cfg.get("minimum_world_m", [-1.20, -0.15, 0.60]))
        init_vol_max = np.array(init_vol_cfg.get("maximum_world_m", [1.20, 0.85, 1.50]))

        init_obs = self.capture.capture_stage(scene, "INITIAL", capture_segmentation=is_oracle_mask)
        if self.ablation == AblationType.SINGLE_VIEW and init_obs:
            init_obs = [init_obs[0]]  # Only camera 0

        for obs in init_obs:
            obs.detected_masks = proposal_backend.predict(obs, init_vol_min, init_vol_max)

        # Observe target recess geometry from RGB-D
        target_evidence = GeometricGrounder.observe_target_recess(init_obs)
        geometric_grounder = GeometricGrounder(target_evidence=target_evidence)
        search_engine = FunctionalSatisfactionSearch(geometric_grounder=geometric_grounder)

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

        trace.steps.append(
            InspectionDecision(
                stage_index=0,
                inspection_region_id="INITIAL",
                action="INITIAL_SURVEY",
                rationale="Survey open workspace and identify staging regions and exposed objects",
            )
        )

        # Grounding & search after Stage 0
        witness, evaluated_tuples, g_res = self._ground_and_search(
            req_map=req_map,
            regions=all_regions,
            semantic_grounder=semantic_grounder,
            geometric_grounder=geometric_grounder,
            search_engine=search_engine,
            stage_idx=0,
        )
        grounding_results_all.extend(g_res)
        all_evaluated_tuples.extend(evaluated_tuples)
        self.graph.snapshot(stage_idx=0, output_dir=self.output_dir)

        # Determine inspection sequence order
        if self.inspection_policy == "fm_ranked":
            descriptors = {r: f"Storage region {r}" for r in INSPECTION_SEQUENCE}
            active_sequence = self.fm_adapter.generate_inspection_priors("Find repair tools and fasteners", descriptors)
        else:
            active_sequence = list(INSPECTION_SEQUENCE)

        if witness is not None:
            trace.early_stopped = True
        else:
            # 4. Incremental inspection over storage containers
            for reg_name in active_sequence:
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

                if self.ablation == AblationType.NO_PERSISTENCE:
                    self.tracker.reset()

                # Capture fresh calibrated observation
                stage_rig = self.capture.get_stage_rig_config(reg_name)
                vol_cfg = stage_rig.get("inspection_volume", stage_rig.get("inspection_volume_m", {}))
                vol_min = np.array(vol_cfg.get("minimum_world_m", [-1.20, -0.30, 0.35]))
                vol_max = np.array(vol_cfg.get("maximum_world_m", [1.20, 0.85, 1.50]))

                stage_obs = self.capture.capture_stage(scene, reg_name, capture_segmentation=is_oracle_mask)
                if self.ablation == AblationType.SINGLE_VIEW and stage_obs:
                    stage_obs = [stage_obs[0]]

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

                # Re-ground and search
                witness, evaluated_tuples, g_res = self._ground_and_search(
                    req_map=req_map,
                    regions=all_regions,
                    semantic_grounder=semantic_grounder,
                    geometric_grounder=geometric_grounder,
                    search_engine=search_engine,
                    stage_idx=stage_idx,
                )
                grounding_results_all.extend(g_res)
                all_evaluated_tuples.extend(evaluated_tuples)
                self.graph.snapshot(stage_idx=stage_idx, output_dir=self.output_dir)

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
            drivers, fasteners, surfaces, containers = self._filter_valid_candidates(
                req_map, all_regions, semantic_grounder, geometric_grounder
            )
            rejection_reason = search_engine.diagnose_infeasibility(
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
            "semantic_model_calls": semantic_grounder.total_semantic_calls,
            "geometric_model_calls": geometric_grounder.total_geometric_calls,
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
        semantic_grounder: SemanticGrounder,
        geometric_grounder: GeometricGrounder,
        search_engine: FunctionalSatisfactionSearch,
        stage_idx: int,
    ) -> tuple[FunctionalWitness | None, list[dict[str, Any]], list[dict[str, Any]]]:
        all_tracks = list(self.tracker.tracks.values())
        drivers, fasteners, surfaces, containers = self._filter_valid_candidates(
            req_map, regions, semantic_grounder, geometric_grounder
        )

        witness, evaluated_tuples = search_engine.search_witness(
            driver_candidates=drivers,
            fastener_candidates=fasteners,
            work_surface_candidates=surfaces,
            parts_container_candidates=containers,
        )

        # Record grounding entries and graph edges
        from mujoco_scenes.workshop_phase1.types import FunctionGroundingResult
        grounding_entries: list[dict[str, Any]] = []
        typed_results: list[FunctionGroundingResult] = []

        bypass_geo = (self.ablation in (AblationType.SEMANTIC_ONLY, AblationType.NO_GEOMETRY))

        for trk in all_tracks:
            for req in req_map.values():
                if req.entity_type.value == "OBJECT":
                    s_res = semantic_grounder.ground_object_for_requirement(trk, req)
                    g_res = geometric_grounder.ground_object_geometry(trk, req)
                    g_status = GroundingStatus.PASS if bypass_geo else g_res.geometric_status

                    comb_status = (
                        GroundingStatus.PASS
                        if (s_res.semantic_status == GroundingStatus.PASS and g_status == GroundingStatus.PASS)
                        else GroundingStatus.FAIL
                    )
                    entry_res = FunctionGroundingResult(
                        entity_id=trk.instance_id,
                        requirement_id=req.requirement_id,
                        function_name=req.function_name,
                        semantic_status=s_res.semantic_status,
                        semantic_score=s_res.semantic_score,
                        semantic_evidence=s_res.semantic_evidence,
                        geometric_status=g_status,
                        geometric_score=g_res.geometric_score,
                        geometric_evidence=g_res.geometric_evidence,
                        combined_status=comb_status,
                        rejection_reasons=s_res.rejection_reasons + g_res.rejection_reasons,
                    )
                    typed_results.append(entry_res)
                    grounding_entries.append({
                        "entity_id": trk.instance_id,
                        "function": req.function_name,
                        "semantic_status": s_res.semantic_status.value,
                        "geometric_status": g_status.value,
                        "combined_status": comb_status.value,
                        "rejections": entry_res.rejection_reasons,
                    })

        self.graph.update_from_grounding_results(typed_results, stage_idx=stage_idx)

        return witness, evaluated_tuples, grounding_entries

    def _filter_valid_candidates(
        self,
        req_map: dict[str, FunctionalRequirement],
        regions: list[ObservedRegion],
        semantic_grounder: SemanticGrounder,
        geometric_grounder: GeometricGrounder,
    ) -> tuple[list[ObservedObjectTrack], list[ObservedObjectTrack], list[ObservedRegion], list[ObservedRegion]]:
        all_tracks = list(self.tracker.tracks.values())
        drivers: list[ObservedObjectTrack] = []
        fasteners: list[ObservedObjectTrack] = []
        surfaces: list[ObservedRegion] = []
        containers: list[ObservedRegion] = []

        bypass_geo = (self.ablation in (AblationType.SEMANTIC_ONLY, AblationType.NO_GEOMETRY))

        driver_req = req_map.get("CAN_DRIVE_SCREW")
        fastener_req = req_map.get("CAN_FASTEN")
        surface_req = req_map.get("WORK_SURFACE")
        container_req = req_map.get("SMALL_PARTS_CONTAINER")

        if driver_req:
            for trk in all_tracks:
                s_res = semantic_grounder.ground_object_for_requirement(trk, driver_req)
                g_res = geometric_grounder.ground_object_geometry(trk, driver_req)
                g_ok = True if bypass_geo else (g_res.geometric_status == GroundingStatus.PASS)
                if s_res.semantic_status == GroundingStatus.PASS and g_ok:
                    drivers.append(trk)

        if fastener_req:
            for trk in all_tracks:
                s_res = semantic_grounder.ground_object_for_requirement(trk, fastener_req)
                g_res = geometric_grounder.ground_object_geometry(trk, fastener_req)
                g_ok = True if bypass_geo else (g_res.geometric_status == GroundingStatus.PASS)
                if s_res.semantic_status == GroundingStatus.PASS and g_ok:
                    fasteners.append(trk)

        if surface_req:
            for reg in regions:
                s_res = semantic_grounder.ground_region_for_requirement(reg, surface_req)
                g_res = geometric_grounder.ground_region_geometry(reg, surface_req)
                g_ok = True if bypass_geo else (g_res.geometric_status == GroundingStatus.PASS)
                if s_res.semantic_status == GroundingStatus.PASS and g_ok:
                    surfaces.append(reg)

        if container_req:
            for reg in regions:
                s_res = semantic_grounder.ground_region_for_requirement(reg, container_req)
                g_res = geometric_grounder.ground_region_geometry(reg, container_req)
                g_ok = True if bypass_geo else (g_res.geometric_status == GroundingStatus.PASS)
                if s_res.semantic_status == GroundingStatus.PASS and g_ok:
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
