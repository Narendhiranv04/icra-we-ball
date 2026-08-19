"""Active inspection controller, multi-view execution loop, and ablation harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

import numpy as np

from mujoco_scenes.workshop_phase1.capture import MultiViewCameraRig
from mujoco_scenes.workshop_phase1.evidence_graph import GrowingObservedGraph
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
    ManualWorkshopFMContract,
    RequirementProvider,
)
from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker
from mujoco_scenes.workshop_phase1.semantic_grounding import (
    ObjectSemanticBackend,
    PrivilegedOracleSemanticBackend,
    ProductionSemanticBackend,
    SemanticGrounder,
)
from mujoco_scenes.workshop_phase1.types import (
    AblationType,
    EpisodeResult,
    FunctionalCandidate,
    FunctionGroundingResult,
    FunctionalRequirement,
    FunctionalWitness,
    GroundingStatus,
    InspectionDecision,
    InspectionTrace,
    MaskBackendType,
    ObservedMask,
    ObservedObjectTrack,
    ObservedRegion,
    ProposalMode,
    SemanticBackendType,
    TargetGeometryEvidence,
    ViewObservation,
    combine_status,
)


class WorkshopPhase1InspectionController:
    """Manages multi-view capture, persistent evidence accumulation, and joint functional search."""

    def __init__(
        self,
        scene: Any | None = None,
        config_path: Path | str | None = None,
        ablation: AblationType = AblationType.NONE,
        mask_backend: MaskBackendType = MaskBackendType.PRODUCTION,
        semantic_backend: SemanticBackendType = SemanticBackendType.PRODUCTION,
        proposal_mode: ProposalMode = ProposalMode.YOLO_ONLY,
        output_dir: Path | str | None = None,
    ) -> None:
        self.scene = scene
        self.config_path = config_path
        self.ablation = ablation
        self.mask_backend_type = mask_backend
        self.semantic_backend_type = semantic_backend
        self.proposal_mode = proposal_mode
        self.output_dir = Path(output_dir) if output_dir else None

        # Load YAML configurations
        self.raw_config = self._load_config(config_path)

        # 1. FM Contract
        fm_contract_path = self.raw_config.get("pipeline", {}).get("fm_contract_path")
        self.fm_contract = ManualWorkshopFMContract(
            Path(fm_contract_path) if fm_contract_path else None
        )
        self.requirements = self.fm_contract.get_requirements()
        self.prompts = self.fm_contract.get_detector_prompts()
        self.alias_to_canonical = self.fm_contract.get_alias_to_canonical_map()

        # 2. Camera Rig
        self.camera_rig = MultiViewCameraRig(scene=self.scene) if self.scene is not None else None

        # 3. Perception Backend
        self.proposal_backend = self._init_proposal_backend()
        self.proposal_backend.set_vocabulary(self.prompts, self.alias_to_canonical)

        # For oracle-mask ablation with YOLO semantics
        self._yolo_aux_backend = None
        if (self.mask_backend_type == MaskBackendType.ORACLE or self.ablation == AblationType.ORACLE_MASK) and self.semantic_backend_type != SemanticBackendType.ORACLE and self.ablation != AblationType.ORACLE_SEMANTICS:
            det_cfg = self.raw_config.get("perception", {}).get("detector", {})
            self._yolo_aux_backend = YOLOWorldProposalBackend(
                weights_path=det_cfg.get("checkpoint"),
                confidence_threshold=det_cfg.get("confidence_threshold", 0.05),
                nms_iou_threshold=det_cfg.get("nms_iou_threshold", 0.45),
                inference_size=det_cfg.get("inference_size", 640),
                device=det_cfg.get("device", "cpu"),
            )
            self._yolo_aux_backend.set_vocabulary(self.prompts, self.alias_to_canonical)

        # 4. Semantic Grounder
        self.semantic_backend = self._init_semantic_backend()
        self.semantic_grounder = SemanticGrounder(backend=self.semantic_backend)

        # 5. Persistent Tracker
        track_cfg = self.raw_config.get("tracking", {})
        self.tracker = PersistentInstanceTracker(
            cluster_distance_threshold_m=track_cfg.get("cluster_distance_threshold_m", 0.040),
            track_match_distance_threshold_m=track_cfg.get("track_match_distance_threshold_m", 0.045),
            voxel_size_m=track_cfg.get("voxel_size_m", 0.003),
            min_cluster_points=track_cfg.get("min_cluster_points", 10),
        )

        # 6. Region Grounder
        self.region_grounder = RegionGrounder()

        # 7. Geometric Grounder & Search
        geom_cfg = self.raw_config.get("grounding", {}).get("geometry", {})
        self.geometric_grounder = GeometricGrounder(
            min_driver_reach_m=geom_cfg.get("min_driver_reach_m", 0.025),
            min_fastener_length_m=geom_cfg.get("min_fastener_length_m", 0.022),
            min_surface_area_m2=geom_cfg.get("min_surface_area_m2", 0.015),
            min_container_volume_m3=geom_cfg.get("min_container_volume_m3", 0.0001),
            staging_margin_multiplier=geom_cfg.get("staging_margin_multiplier", 1.20),
        )
        self.functional_search = FunctionalSatisfactionSearch(geometric_grounder=self.geometric_grounder)

        # 8. Observed Evidence Graph
        self.graph = GrowingObservedGraph()

        # 9. Inspection Policy
        insp_cfg = self.raw_config.get("inspection", {})
        self.inspection_sequence = insp_cfg.get("sequence", ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"])
        self.early_stop_enabled = insp_cfg.get("early_stop", True)

        self.trace = InspectionTrace()
        self.target_evidence = TargetGeometryEvidence()
        self.candidate_regions: list[ObservedRegion] = []

    def _load_config(self, config_path: Path | str | None) -> dict[str, Any]:
        default_path = Path(__file__).resolve().parent.parent / "configs" / "workshop_phase1.yaml"
        target_path = Path(config_path) if config_path else default_path
        if target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def _init_proposal_backend(self) -> InstanceProposalBackend:
        if self.mask_backend_type == MaskBackendType.ORACLE or self.ablation == AblationType.ORACLE_MASK:
            return PrivilegedOracleMaskBackend(scene=self.scene)
        if self.mask_backend_type == MaskBackendType.CONNECTED_COMPONENT:
            return RGBDConnectedComponentProposalBackend()

        det_cfg = self.raw_config.get("perception", {}).get("detector", {})
        return YOLOWorldProposalBackend(
            weights_path=det_cfg.get("checkpoint"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.05),
            nms_iou_threshold=det_cfg.get("nms_iou_threshold", 0.45),
            inference_size=det_cfg.get("inference_size", 640),
            device=det_cfg.get("device", "cpu"),
            max_detections=det_cfg.get("max_detections", 100),
        )

    def _init_semantic_backend(self) -> ObjectSemanticBackend:
        if self.semantic_backend_type == SemanticBackendType.ORACLE or self.ablation == AblationType.ORACLE_SEMANTICS:
            return PrivilegedOracleSemanticBackend(scene=self.scene)
        return ProductionSemanticBackend()

    def run_episode(self, scene: Any | None = None) -> EpisodeResult:
        """Run the full Phase 1 incremental inspection loop."""
        if scene is not None:
            self.scene = scene
            self.camera_rig = MultiViewCameraRig(scene=self.scene)
            if self.mask_backend_type == MaskBackendType.ORACLE or self.ablation == AblationType.ORACLE_MASK:
                self.proposal_backend = PrivilegedOracleMaskBackend(scene=self.scene)
            if self.semantic_backend_type == SemanticBackendType.ORACLE or self.ablation == AblationType.ORACLE_SEMANTICS:
                self.semantic_backend = PrivilegedOracleSemanticBackend(scene=self.scene)
                self.semantic_grounder = SemanticGrounder(backend=self.semantic_backend)
        """Run the full Phase 1 incremental inspection loop."""
        # Register static workpiece and search regions in evidence graph
        self.graph.register_workpiece_node()
        for s_reg in self.inspection_sequence:
            self.graph.register_inspection_region_node(s_reg, f"Search container {s_reg}")

        # Stage 0: Initial workbench observation
        stage_0_obs = self._capture_and_process_stage(stage_idx=0, source_region_id="INITIAL_WORKBENCH")

        # Estimate target recess from RGB-D
        self.target_evidence = self.geometric_grounder.observe_target_recess(stage_0_obs, scene=self.scene)
        if self.target_evidence.validity == GroundingStatus.PASS:
            self.geometric_grounder.target_hole_depth_m = (
                self.target_evidence.estimated_recess_depth_m
                if self.target_evidence.estimated_recess_depth_m is not None
                else 0.030
            )
            self.geometric_grounder.target_hole_diameter_m = (
                self.target_evidence.estimated_opening_diameter_m
                if self.target_evidence.estimated_opening_diameter_m is not None
                else 0.007
            )

        # Discover candidate regions
        self.candidate_regions = self.region_grounder.discover_candidate_regions(self.scene, stage_0_obs)
        self.graph.update_from_observed_regions(self.candidate_regions, stage_idx=0)

        # Evaluate stage 0
        witness, rej_reason = self._evaluate_grounding_and_search(stage_idx=0, source_region_id="INITIAL_WORKBENCH")

        if witness is not None and self.early_stop_enabled:
            self.trace.early_stopped = True
            self.trace.steps.append(
                InspectionDecision(
                    stage_index=0,
                    inspection_region_id="INITIAL_WORKBENCH",
                    action="STOP_FEASIBLE",
                    rationale="Complete verified witness found on initial workbench observation.",
                )
            )
            return EpisodeResult(
                status="FEASIBLE",
                rejection_reason=None,
                witness=witness,
                trace=self.trace,
                metrics={"stages_executed": 1, "total_tracks": len(self.tracker.tracks)},
            )

        # Step through inspection sequence
        for s_idx, region_name in enumerate(self.inspection_sequence, start=1):
            self.trace.inspected_regions.append(region_name)

            # Open container in physics
            if hasattr(self.scene, "open_container"):
                try:
                    self.scene.open_container(region_name)
                except Exception:
                    pass

            # For NO_PERSISTENCE ablation: reset tracker and caches
            if self.ablation == AblationType.NO_PERSISTENCE:
                self.tracker.reset()
                self.semantic_grounder.reset_cache()

            stage_obs = self._capture_and_process_stage(stage_idx=s_idx, source_region_id=region_name)
            witness, rej_reason = self._evaluate_grounding_and_search(stage_idx=s_idx, source_region_id=region_name)

            if witness is not None and self.early_stop_enabled:
                self.trace.early_stopped = True
                self.trace.steps.append(
                    InspectionDecision(
                        stage_index=s_idx,
                        inspection_region_id=region_name,
                        action="STOP_FEASIBLE",
                        rationale=f"Complete verified witness found after inspecting {region_name}.",
                    )
                )
                return EpisodeResult(
                    status="FEASIBLE",
                    rejection_reason=None,
                    witness=witness,
                    trace=self.trace,
                    metrics={"stages_executed": s_idx + 1, "total_tracks": len(self.tracker.tracks)},
                )

            self.trace.steps.append(
                InspectionDecision(
                    stage_index=s_idx,
                    inspection_region_id=region_name,
                    action="INSPECT",
                    rationale=f"Witness not yet complete; continue to next region.",
                )
            )

        # Exhausted all inspection stages
        final_status = "INFEASIBLE" if rej_reason != "INSUFFICIENT_EVIDENCE" else "INSUFFICIENT_EVIDENCE"
        return EpisodeResult(
            status=final_status,
            rejection_reason=rej_reason,
            witness=None,
            trace=self.trace,
            metrics={"stages_executed": len(self.inspection_sequence) + 1, "total_tracks": len(self.tracker.tracks)},
        )

    def _capture_and_process_stage(self, stage_idx: int, source_region_id: str) -> list[ViewObservation]:
        """Capture multi-view RGB-D and predict instance masks."""
        is_oracle_mask = (self.mask_backend_type == MaskBackendType.ORACLE or self.ablation == AblationType.ORACLE_MASK)
        raw_obs = self.camera_rig.capture_stage_observations(
            stage_region=source_region_id,
            capture_segmentation=is_oracle_mask,
        )

        # SINGLE_FRONT_VIEW ablation: filter strictly to front camera
        if self.ablation == AblationType.SINGLE_VIEW or self.ablation == AblationType.SINGLE_FRONT_VIEW:
            raw_obs = [obs for obs in raw_obs if obs.camera_id == "workshop_camera_front"]
            if not raw_obs:
                raw_obs = [self.camera_rig.capture_stage_observations(stage_region=source_region_id, capture_segmentation=is_oracle_mask)[0]]

        # Determine stage inspection volume bounds
        stage_min, stage_max = self._get_stage_volume_bounds(source_region_id)

        # Run instance proposal on each view
        for obs in raw_obs:
            masks = self.proposal_backend.predict(
                observation=obs,
                stage_volume_min=stage_min,
                stage_volume_max=stage_max,
                volume_margin_m=0.08,
            )

            # If using oracle masks WITH YOLO semantics (ORACLE_MASK ablation), associate YOLO semantic detections by 2D IoU
            if (self.mask_backend_type == MaskBackendType.ORACLE or self.ablation == AblationType.ORACLE_MASK) and self.semantic_backend_type != SemanticBackendType.ORACLE and self.ablation != AblationType.ORACLE_SEMANTICS and self._yolo_aux_backend:
                yolo_masks = self._yolo_aux_backend.predict(
                    observation=obs,
                    stage_volume_min=stage_min,
                    stage_volume_max=stage_max,
                    volume_margin_m=0.08,
                )
                for om in masks:
                    ox1, oy1, ox2, oy2 = om.bounding_box_xyxy
                    om_area = max(1, (ox2 - ox1) * (oy2 - oy1))
                    best_iou = 0.0
                    best_yolo = None
                    for ym in yolo_masks:
                        yx1, yy1, yx2, yy2 = ym.bounding_box_xyxy
                        ix1, iy1 = max(ox1, yx1), max(oy1, yy1)
                        ix2, iy2 = min(ox2, yx2), min(oy2, yy2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            union = om_area + (yx2 - yx1) * (yy2 - yy1) - inter
                            iou = inter / max(1, union)
                            if iou > best_iou:
                                best_iou = iou
                                best_yolo = ym

                    if best_yolo and best_iou > 0.15:
                        om.canonical_label = best_yolo.canonical_label
                        om.raw_label = best_yolo.raw_label
                        om.confidence = best_yolo.confidence
                        om.predicted_label = best_yolo.canonical_label
                    else:
                        om.canonical_label = "unknown"
                        om.raw_label = "unknown"
                        om.confidence = 0.0
                        om.predicted_label = "unknown"

            obs.detected_masks = masks

        # Update persistent tracking
        self.tracker.update_with_stage_observations(
            stage_index=stage_idx,
            source_region_id=source_region_id,
            observations=raw_obs,
            stage_volume_min=stage_min,
            stage_volume_max=stage_max,
        )

        return raw_obs

    def _get_stage_volume_bounds(self, source_region_id: str) -> tuple[np.ndarray, np.ndarray]:
        if source_region_id == "LEFT_DRAWER":
            return np.array([-0.65, -0.20, 0.35]), np.array([-0.10, 0.45, 0.75])
        elif source_region_id == "RIGHT_DRAWER":
            return np.array([0.10, -0.20, 0.35]), np.array([0.65, 0.45, 0.75])
        elif source_region_id == "TOOL_CABINET":
            return np.array([0.10, 0.30, 0.60]), np.array([0.80, 0.95, 1.25])
        # Default workbench/scene volume
        return np.array([-1.20, -0.30, 0.35]), np.array([1.20, 1.20, 1.40])

    def _evaluate_grounding_and_search(
        self,
        stage_idx: int,
        source_region_id: str,
    ) -> tuple[FunctionalWitness | None, str | None]:
        """Perform semantic and geometric grounding pass, update graph, and search witness."""
        all_tracks = list(self.tracker.tracks.values())
        self.graph.update_from_object_tracks(all_tracks, stage_idx=stage_idx, stage_region_id=source_region_id)

        driver_req = next((r for r in self.requirements if r.function_name == "CAN_DRIVE_SCREW"), None)
        fastener_req = next((r for r in self.requirements if r.function_name == "CAN_FASTEN"), None)
        surface_req = next((r for r in self.requirements if r.function_name == "WORK_SURFACE"), None)
        container_req = next((r for r in self.requirements if r.function_name == "SMALL_PARTS_CONTAINER"), None)

        grounding_results = []
        driver_candidates = []
        fastener_candidates = []
        surface_candidates = []
        container_candidates = []

        is_semantic_only = (self.ablation == AblationType.SEMANTIC_ONLY or self.ablation == AblationType.NO_GEOMETRY)

        # Ground objects
        for trk in all_tracks:
            # 1. CAN_DRIVE_SCREW
            if driver_req:
                s_res = self.semantic_grounder.ground_object_for_requirement(trk, driver_req)
                if s_res.semantic_status == GroundingStatus.PASS:
                    trk.current_semantic_belief.update(s_res.semantic_evidence)
                g_res = self.geometric_grounder.ground_object_geometry(trk, driver_req)
                c_status = s_res.semantic_status if is_semantic_only else combine_status(s_res.semantic_status, g_res.geometric_status)
                f_res = FunctionGroundingResult(
                    entity_id=trk.instance_id,
                    requirement_id=driver_req.requirement_id,
                    function_name=driver_req.function_name,
                    semantic_status=s_res.semantic_status,
                    semantic_score=s_res.semantic_score,
                    semantic_evidence=s_res.semantic_evidence,
                    geometric_status=g_res.geometric_status,
                    geometric_score=g_res.geometric_score,
                    geometric_evidence=g_res.geometric_evidence,
                    combined_status=c_status,
                    rejection_reasons=s_res.rejection_reasons + g_res.rejection_reasons,
                )
                grounding_results.append(f_res)
                if c_status == GroundingStatus.PASS:
                    driver_candidates.append(trk)

            # 2. CAN_FASTEN
            if fastener_req:
                s_res = self.semantic_grounder.ground_object_for_requirement(trk, fastener_req)
                if s_res.semantic_status == GroundingStatus.PASS:
                    trk.current_semantic_belief.update(s_res.semantic_evidence)
                g_res = self.geometric_grounder.ground_object_geometry(trk, fastener_req)
                c_status = s_res.semantic_status if is_semantic_only else combine_status(s_res.semantic_status, g_res.geometric_status)
                f_res = FunctionGroundingResult(
                    entity_id=trk.instance_id,
                    requirement_id=fastener_req.requirement_id,
                    function_name=fastener_req.function_name,
                    semantic_status=s_res.semantic_status,
                    semantic_score=s_res.semantic_score,
                    semantic_evidence=s_res.semantic_evidence,
                    geometric_status=g_res.geometric_status,
                    geometric_score=g_res.geometric_score,
                    geometric_evidence=g_res.geometric_evidence,
                    combined_status=c_status,
                    rejection_reasons=s_res.rejection_reasons + g_res.rejection_reasons,
                )
                grounding_results.append(f_res)
                if c_status == GroundingStatus.PASS:
                    fastener_candidates.append(trk)

        # Ground regions
        for reg in self.candidate_regions:
            # 3. WORK_SURFACE
            if surface_req:
                s_res = self.semantic_grounder.ground_region_for_requirement(reg, surface_req)
                if s_res.semantic_status == GroundingStatus.PASS:
                    reg.current_semantic_belief.update(s_res.semantic_evidence)
                g_res = self.geometric_grounder.ground_region_geometry(reg, surface_req)
                c_status = s_res.semantic_status if is_semantic_only else combine_status(s_res.semantic_status, g_res.geometric_status)
                f_res = FunctionGroundingResult(
                    entity_id=reg.region_instance_id,
                    requirement_id=surface_req.requirement_id,
                    function_name=surface_req.function_name,
                    semantic_status=s_res.semantic_status,
                    semantic_score=s_res.semantic_score,
                    semantic_evidence=s_res.semantic_evidence,
                    geometric_status=g_res.geometric_status,
                    geometric_score=g_res.geometric_score,
                    geometric_evidence=g_res.geometric_evidence,
                    combined_status=c_status,
                    rejection_reasons=s_res.rejection_reasons + g_res.rejection_reasons,
                )
                grounding_results.append(f_res)
                if c_status == GroundingStatus.PASS:
                    surface_candidates.append(reg)

            # 4. SMALL_PARTS_CONTAINER
            if container_req:
                s_res = self.semantic_grounder.ground_region_for_requirement(reg, container_req)
                if s_res.semantic_status == GroundingStatus.PASS:
                    reg.current_semantic_belief.update(s_res.semantic_evidence)
                g_res = self.geometric_grounder.ground_region_geometry(reg, container_req)
                c_status = s_res.semantic_status if is_semantic_only else combine_status(s_res.semantic_status, g_res.geometric_status)
                f_res = FunctionGroundingResult(
                    entity_id=reg.region_instance_id,
                    requirement_id=container_req.requirement_id,
                    function_name=container_req.function_name,
                    semantic_status=s_res.semantic_status,
                    semantic_score=s_res.semantic_score,
                    semantic_evidence=s_res.semantic_evidence,
                    geometric_status=g_res.geometric_status,
                    geometric_score=g_res.geometric_score,
                    geometric_evidence=g_res.geometric_evidence,
                    combined_status=c_status,
                    rejection_reasons=s_res.rejection_reasons + g_res.rejection_reasons,
                )
                grounding_results.append(f_res)
                if c_status == GroundingStatus.PASS:
                    container_candidates.append(reg)

        self.graph.update_from_grounding_results(grounding_results, stage_idx=stage_idx)
        self.graph.snapshot(stage_idx=stage_idx, output_dir=self.output_dir)

        # NO_JOINT_COUPLING ablation: choose independent unary-best candidates without profile/packing check
        if self.ablation == AblationType.NO_JOINT_COUPLING:
            if driver_candidates and fastener_candidates and surface_candidates and container_candidates:
                d_best = max(driver_candidates, key=lambda x: x.overall_confidence)
                f_best = max(fastener_candidates, key=lambda x: x.overall_confidence)
                s_best = surface_candidates[0]
                c_best = container_candidates[0]
                wit = FunctionalWitness(
                    driver_id=d_best.instance_id,
                    fastener_id=f_best.instance_id,
                    work_surface_id=s_best.region_instance_id,
                    parts_container_id=c_best.region_instance_id,
                    overall_confidence=float(d_best.overall_confidence * f_best.overall_confidence),
                    verification_details={"ablation": "NO_JOINT_COUPLING", "unary_selected": True},
                )
                return wit, None

        # Full joint search
        witness, evaluated_tuples = self.functional_search.search_witness(
            driver_candidates=driver_candidates,
            fastener_candidates=fastener_candidates,
            work_surface_candidates=surface_candidates,
            parts_container_candidates=container_candidates,
        )

        if witness is not None:
            return witness, None

        rej_reason = self.functional_search.diagnose_infeasibility(
            all_objects=all_tracks,
            all_regions=self.candidate_regions,
            driver_candidates=driver_candidates,
            fastener_candidates=fastener_candidates,
            work_surface_candidates=surface_candidates,
            parts_container_candidates=container_candidates,
            evaluated_tuples=evaluated_tuples,
        )

        return None, rej_reason
