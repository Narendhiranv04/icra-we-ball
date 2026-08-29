"""Active inspection controller, multi-view execution loop, and ablation harness."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple
from PIL import Image
import yaml

import cv2
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
    FMRequirementProvider,
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
from mujoco_scenes.workshop_phase1.visual_profile import apply_workshop_visual_profile


class WorkshopPhase1InspectionController:
    """Manages multi-view capture, persistent evidence accumulation, and joint functional search."""

    def __init__(
        self,
        scene: Any | None = None,
        config_path: Path | str | None = None,
        ablation: AblationType | None = None,
        mask_backend: MaskBackendType | None = None,
        semantic_backend: SemanticBackendType | None = None,
        proposal_mode: ProposalMode = ProposalMode.YOLO_ONLY,
        output_dir: Path | str | None = None,
    ) -> None:
        self.scene = scene
        self.config_path = config_path
        self.proposal_mode = proposal_mode
        self.output_dir = Path(output_dir) if output_dir else None

        # Load YAML configurations
        self.raw_config = self._load_config(config_path)
        pipeline_defaults = self.raw_config.get("pipeline", {})
        self.ablation = ablation or AblationType(str(pipeline_defaults.get("ablation", "none")).upper())
        self.mask_backend_type = mask_backend or MaskBackendType(str(pipeline_defaults.get("mask_backend", "production")).upper())
        self.semantic_backend_type = semantic_backend or SemanticBackendType(str(pipeline_defaults.get("semantic_backend", "production")).upper())
        inspection_policy = str(pipeline_defaults.get("inspection_policy", "fixed")).lower()
        if inspection_policy != "fixed":
            raise RuntimeError(f"Inspection policy {inspection_policy!r} is not configured; use 'fixed'.")

        # 1. FM Contract
        pipeline_cfg = self.raw_config.get("pipeline", {})
        visual_profile_path = pipeline_cfg.get("visual_profile_path")
        self.visual_profile_path = Path(visual_profile_path) if visual_profile_path else None
        if self.scene is not None and self.visual_profile_path is not None:
            self.visual_profile = apply_workshop_visual_profile(
                self.scene, self.visual_profile_path)
        else:
            self.visual_profile = None
        rig_config_path = pipeline_cfg.get("inspection_rig_config_path")
        self.inspection_rig_config_path = Path(rig_config_path) if rig_config_path else None
        fm_contract_path = pipeline_cfg.get("fm_contract_path")
        requirements_source = str(pipeline_cfg.get("requirements_source", "static")).lower()
        self.requirement_provider = (
            FMRequirementProvider() if requirements_source == "fm" else
            ManualWorkshopFMContract(Path(fm_contract_path) if fm_contract_path else None)
        )
        self.fm_contract = self.requirement_provider
        self.requirements = self.requirement_provider.get_requirements()
        ranked_vocabulary = self.requirement_provider.get_ranked_detector_vocabulary()
        vocabulary_budget = int(self.raw_config.get("perception", {}).get(
            "max_detector_vocabulary_size", 32))
        if vocabulary_budget <= 0:
            raise ValueError("perception.max_detector_vocabulary_size must be positive")
        self.detector_vocabulary = ranked_vocabulary[:vocabulary_budget]
        self.prompts = [entry["detector_label"] for entry in self.detector_vocabulary]
        self.detector_label_to_canonical = self.requirement_provider.get_detector_label_to_canonical_map()
        self.alias_to_canonical = self.requirement_provider.get_alias_to_canonical_map()

        # 2. Camera Rig
        self.camera_rig = MultiViewCameraRig(
            scene=self.scene,
            width=int(pipeline_cfg.get("image_width", 1280)),
            height=int(pipeline_cfg.get("image_height", 720)),
            rig_config_path=self.inspection_rig_config_path,
        ) if self.scene is not None else None

        # 3. Perception Backend
        self.proposal_backend = self._init_proposal_backend()
        self.proposal_backend.set_vocabulary(self.prompts, self.detector_label_to_canonical)

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
                max_detections=det_cfg.get("max_detections", 100),
                min_points_per_mask=self.raw_config.get("perception", {}).get("min_points_per_mask", 8),
                supplemental_prompts=det_cfg.get("supplemental_prompts"),
                supplemental_confidence_threshold=det_cfg.get("supplemental_confidence_threshold"),
                proposal_crop_confidence_threshold=det_cfg.get("proposal_crop_confidence_threshold"),
                **self._duplicate_config(),
                **self._multi_scale_config(),
            )
            self._yolo_aux_backend.set_vocabulary(self.prompts, self.detector_label_to_canonical)

        # 4. Semantic Grounder
        self.semantic_backend = self._init_semantic_backend()
        self.semantic_grounder = SemanticGrounder(backend=self.semantic_backend)

        # 5. Persistent Tracker
        track_cfg = self.raw_config.get("tracking", {})
        fusion_cfg = self.raw_config.get("semantic_fusion", self.raw_config.get("fusion", {}))
        self.tracker = PersistentInstanceTracker(
            cluster_distance_threshold_m=track_cfg.get("cluster_distance_threshold_m", 0.040),
            track_match_distance_threshold_m=track_cfg.get("track_match_distance_threshold_m", 0.045),
            voxel_size_m=track_cfg.get("voxel_size_m", 0.003),
            min_cluster_points=track_cfg.get("min_cluster_points", 10),
            volume_margin_m=self.raw_config.get("perception", {}).get("volume_margin_m", 0.08),
            min_points_per_mask=self.raw_config.get("perception", {}).get("min_points_per_mask", 8),
            stage_object_merge_distance_threshold_m=track_cfg.get(
                "stage_object_merge_distance_threshold_m", 0.0),
            fusion_config=fusion_cfg,
        )

        # 6. Region Grounder. Region semantic categories come only from the
        # active manual future-FM contract.
        region_categories = {
            category
            for requirement in self.requirements
            if requirement.entity_type.value in {"REGION", "FUNCTIONAL_REGION"}
            for category in requirement.accepted_categories
        }
        self.region_categories = region_categories
        self.object_categories = {
            entry["canonical_label"] for entry in self.detector_vocabulary
        } - region_categories
        geometry_path = self.raw_config.get("pipeline", {}).get("geometry_config_path")
        self.region_grounder = RegionGrounder(region_categories, geometry_path)

        # 7. Category-free Geometric Grounder & Search
        self.geometric_grounder = GeometricGrounder(geometry_config_path=geometry_path)
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
        self.detection_diagnostics: list[dict[str, Any]] = []

    def _duplicate_config(self) -> dict[str, Any]:
        cfg = self.raw_config.get("perception", {}).get("duplicate_suppression", {})
        return {
            "duplicate_box_iou_threshold": cfg.get("box_iou_threshold", 0.65),
            "duplicate_mask_overlap_threshold": cfg.get("mask_overlap_threshold", 0.72),
            "duplicate_centroid_distance_m": cfg.get("centroid_distance_m", 0.018),
            "duplicate_aabb_overlap_threshold": cfg.get("aabb_overlap_threshold", 0.45),
        }

    def _multi_scale_config(self) -> dict[str, Any]:
        cfg = self.raw_config.get("perception", {}).get("multi_scale", {})
        return {
            "enable_full_frame": cfg.get("full_frame", True),
            "enable_stage_crop": cfg.get("stage_crop", True),
            "enable_stage_tiles": cfg.get("stage_tiles", False),
            "tile_overlap_fraction": cfg.get("tile_overlap_fraction", 0.15),
        }

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
            supplemental_prompts=det_cfg.get("supplemental_prompts", []),
            supplemental_confidence_threshold=det_cfg.get("supplemental_confidence_threshold"),
            min_points_per_mask=self.raw_config.get("perception", {}).get("min_points_per_mask", 8),
            **self._duplicate_config(),
            **self._multi_scale_config(),
        )

    def _init_semantic_backend(self) -> ObjectSemanticBackend:
        if self.semantic_backend_type == SemanticBackendType.ORACLE or self.ablation == AblationType.ORACLE_SEMANTICS:
            return PrivilegedOracleSemanticBackend(scene=self.scene)
        return ProductionSemanticBackend()

    @staticmethod
    def _skipped_geometry_result(entity_id: str, requirement: FunctionalRequirement) -> FunctionGroundingResult:
        return FunctionGroundingResult(
            entity_id=entity_id,
            requirement_id=requirement.requirement_id,
            function_name=requirement.function_name,
            semantic_status=GroundingStatus.UNKNOWN,
            semantic_score=0.0,
            semantic_evidence={},
            geometric_status=GroundingStatus.UNKNOWN,
            geometric_score=0.0,
            geometric_evidence={"ablation": "GEOMETRY_NOT_INVOKED"},
            combined_status=GroundingStatus.UNKNOWN,
            rejection_reasons=[],
        )

    def run_episode(self, scene: Any | None = None) -> EpisodeResult:
        """Run the full Phase 1 incremental inspection loop."""
        if scene is not None:
            self.scene = scene
            pipeline_cfg = self.raw_config.get("pipeline", {})
            if self.visual_profile_path is not None:
                self.visual_profile = apply_workshop_visual_profile(
                    self.scene, self.visual_profile_path)
            self.camera_rig = MultiViewCameraRig(
                scene=self.scene,
                width=int(pipeline_cfg.get("image_width", 1280)),
                height=int(pipeline_cfg.get("image_height", 720)),
                rig_config_path=self.inspection_rig_config_path,
            )
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
        self.target_evidence = self.geometric_grounder.observe_target_recess(
            stage_0_obs, scene=self.scene, config=self.geometric_grounder.config)
        self.geometric_grounder.target_evidence = self.target_evidence

        # Candidate regions were discovered and associated during stage capture.

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
                self.region_grounder.reset_persistent_evidence()

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
        if witness is not None:
            return EpisodeResult(
                status="FEASIBLE",
                rejection_reason=None,
                witness=witness,
                trace=self.trace,
                metrics={
                    "stages_executed": len(self.inspection_sequence) + 1,
                    "total_tracks": len(self.tracker.tracks),
                    "early_stopped": False,
                },
            )
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

        # Historical SINGLE_FRONT_VIEW is the cross-environment DETAIL ablation.
        if self.ablation == AblationType.SINGLE_VIEW or self.ablation == AblationType.SINGLE_FRONT_VIEW:
            raw_obs = [obs for obs in raw_obs if obs.camera_id == "DETAIL"]
            if not raw_obs:
                raise RuntimeError("Single-view ablation requires canonical DETAIL")

        # Reuse the one-time FM/manual contract output, but hidden-storage
        # stages search for objects only. No FM call or ranking is repeated.
        stage_prompts = (
            self.prompts
            if source_region_id in {"INITIAL", "INITIAL_WORKBENCH"}
            else [
                entry["detector_label"]
                for entry in self.detector_vocabulary
                if entry["canonical_label"] in self.object_categories
            ]
        )
        self.proposal_backend.set_vocabulary(
            stage_prompts, self.detector_label_to_canonical
        )
        if self._yolo_aux_backend is not None:
            self._yolo_aux_backend.set_vocabulary(
                stage_prompts, self.detector_label_to_canonical
            )

        # Determine stage inspection volume bounds
        stage_min, stage_max = self._get_stage_volume_bounds(source_region_id)

        # Run instance proposal on each view
        perception_cfg = self.raw_config.get("perception", {})
        volume_margin = float(perception_cfg.get("volume_margin_m", 0.08))
        for obs in raw_obs:
            masks = self.proposal_backend.predict(
                observation=obs,
                stage_volume_min=stage_min,
                stage_volume_max=stage_max,
                volume_margin_m=volume_margin,
            )
            obs.region_semantic_detections = (
                [] if is_oracle_mask else
                [mask for mask in masks if mask.canonical_label.lower() in self.region_categories]
            )

            # If using oracle masks WITH YOLO semantics (ORACLE_MASK ablation), associate YOLO semantic detections by 2D IoU
            if (self.mask_backend_type == MaskBackendType.ORACLE or self.ablation == AblationType.ORACLE_MASK) and self.semantic_backend_type != SemanticBackendType.ORACLE and self.ablation != AblationType.ORACLE_SEMANTICS and self._yolo_aux_backend:
                yolo_masks = self._yolo_aux_backend.predict(
                    observation=obs,
                    stage_volume_min=stage_min,
                    stage_volume_max=stage_max,
                    volume_margin_m=volume_margin,
                )
                # Object-specific proposal crop semantics:
                # Isolate proposal with bounding crop from RGB and run YOLO-World
                # with full vocabulary AND each supplemental prompt individually.
                active_supplementals = [
                    p for p in self._yolo_aux_backend.supplemental_prompts
                    if p in self._yolo_aux_backend._prompts
                ]
                for om_idx, om in enumerate(masks):
                    ox1, oy1, ox2, oy2 = om.bounding_box_xyxy
                    bw, bh = ox2 - ox1, oy2 - oy1
                    px = max(int(bw * 0.35), 4)
                    py = max(int(bh * 0.35), 4)
                    cx1 = max(0, ox1 - px)
                    cy1 = max(0, oy1 - py)
                    cx2 = min(obs.rgb.shape[1], ox2 + px)
                    cy2 = min(obs.rgb.shape[0], oy2 + py)
                    min_sz = 320
                    if cx2 - cx1 < min_sz:
                        ex = min_sz - (cx2 - cx1)
                        cx1 = max(0, cx1 - ex // 2)
                        cx2 = min(obs.rgb.shape[1], cx2 + ex - ex // 2)
                    if cy2 - cy1 < min_sz:
                        ey = min_sz - (cy2 - cy1)
                        cy1 = max(0, cy1 - ey // 2)
                        cy2 = min(obs.rgb.shape[0], cy2 + ey - ey // 2)
                    crop = obs.rgb[cy1:cy2, cx1:cx2].copy()
                    if crop.size > 0 and getattr(self._yolo_aux_backend, "_model", None) is not None:
                        # Run all-prompts pass + each supplemental single-prompt pass
                        for sup_prompt in [None, *active_supplementals]:
                            try:
                                self._yolo_aux_backend._activate_prompt_pass(sup_prompt)
                                pass_conf = (
                                    self._yolo_aux_backend.supplemental_confidence_threshold
                                    if sup_prompt is not None
                                    else self._yolo_aux_backend.proposal_crop_confidence_threshold
                                )
                                results = self._yolo_aux_backend._model.predict(
                                    source=Image.fromarray(crop, mode="RGB"),
                                    conf=pass_conf,
                                    iou=self._yolo_aux_backend.nms_iou_threshold,
                                    imgsz=self._yolo_aux_backend.inference_size,
                                    device=self._yolo_aux_backend.device,
                                    max_det=self._yolo_aux_backend.max_detections,
                                    verbose=False,
                                )
                                if results and results[0].boxes is not None:
                                    for box, conf, cls_id in zip(
                                        results[0].boxes.xyxy.cpu().numpy(),
                                        results[0].boxes.conf.cpu().numpy(),
                                        results[0].boxes.cls.cpu().numpy(),
                                    ):
                                        raw_label = str(results[0].names[int(cls_id)])
                                        canonical_label = self._yolo_aux_backend._alias_to_canonical.get(
                                            raw_label.lower(), raw_label.lower()
                                        )
                                        print(f"PROPOSAL CROP DETECTED: {canonical_label} conf={conf:.4f} src={sup_prompt} bbox={box}")
                                        bx1 = max(0, min(obs.rgb.shape[1] - 1, int(np.floor(box[0])) + cx1))
                                        by1 = max(0, min(obs.rgb.shape[0] - 1, int(np.floor(box[1])) + cy1))
                                        bx2 = max(0, min(obs.rgb.shape[1], int(np.ceil(box[2])) + cx1))
                                        by2 = max(0, min(obs.rgb.shape[0], int(np.ceil(box[3])) + cy1))
                                        sup_tag = "" if sup_prompt is None else f"_sup_{sup_prompt.lower().replace(' ', '_')}"
                                        yolo_masks.append(ObservedMask(
                                            detection_id=f"det_{obs.camera_id}_proposal_crop_{om_idx:03d}{sup_tag}_{cls_id}",
                                            camera_id=obs.camera_id,
                                            binary_mask=om.binary_mask,
                                            bounding_box_xyxy=(bx1, by1, bx2, by2),
                                            confidence=float(conf),
                                            canonical_label=canonical_label,
                                            raw_label=raw_label,
                                            predicted_label=canonical_label,
                                            backend_name="yolo_world",
                                            refined_mask_area=om.refined_mask_area,
                                            depth_point_count=om.depth_point_count,
                                            centroid_world_m=om.centroid_world_m,
                                            cloud_bounds_world_m=om.cloud_bounds_world_m,
                                            raw_yolo_bbox_xyxy=(bx1, by1, bx2, by2),
                                            refined_bbox_xyxy=(bx1, by1, bx2, by2),
                                            gated_points_world_m=om.gated_points_world_m,
                                            gated_pixel_indices_yx=om.gated_pixel_indices_yx,
                                            inference_source="proposal_crop",
                                            physical_support_quality=1.0,
                                        ))
                            except Exception as e:
                                print(f"PROPOSAL CROP EXCEPTION: {e}")

                # Oracle masks replace only object-instance masks. Raw YOLO
                # furniture detections remain a separate region channel.
                obs.region_semantic_detections = [
                    mask for mask in yolo_masks
                    if mask.canonical_label.lower() in self.region_categories
                ]
                object_yolo_masks = [
                    mask for mask in yolo_masks
                    if mask.canonical_label.lower() in self.object_categories
                ]
                for om in masks:
                    ox1, oy1, ox2, oy2 = om.bounding_box_xyxy
                    om_area = max(1, (ox2 - ox1) * (oy2 - oy1))
                    best_score = 0.0
                    best_yolo = None
                    for ym in object_yolo_masks:
                        yx1, yy1, yx2, yy2 = ym.bounding_box_xyxy
                        ix1, iy1 = max(ox1, yx1), max(oy1, yy1)
                        ix2, iy2 = min(ox2, yx2), min(oy2, yy2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            union = om_area + (yx2 - yx1) * (yy2 - yy1) - inter
                            iou = inter / max(1, union)
                            mask_overlap = int(np.count_nonzero(om.binary_mask & ym.binary_mask)) / max(
                                1, min(int(np.count_nonzero(om.binary_mask)), int(np.count_nonzero(ym.binary_mask))))
                            distance = float(np.linalg.norm(
                                np.asarray(om.centroid_world_m) - np.asarray(ym.centroid_world_m)))
                            proximity = max(0.0, 1.0 - distance / 0.08)
                            assoc_cfg = self.raw_config.get("perception", {}).get("association", {})
                            crop_assoc_mult = float(assoc_cfg.get("proposal_crop_association_multiplier", 20.0))
                            mult = crop_assoc_mult if ym.inference_source == "proposal_crop" else 1.0
                            ym_conf = ym.confidence
                            if om.cloud_bounds_world_m:
                                min_b = om.cloud_bounds_world_m.get("minimum_world_m")
                                max_b = om.cloud_bounds_world_m.get("maximum_world_m")
                                if min_b and max_b:
                                    prior_cfg = self.raw_config.get("semantic_physical_prior", {})
                                    if prior_cfg.get("enabled", False):
                                        max_dim = float(np.max(np.asarray(max_b, dtype=float) - np.asarray(min_b, dtype=float)))
                                        small_thresh = prior_cfg.get("small_object_max_dimension_m", 0.08)
                                        if max_dim < small_thresh:
                                            multipliers = prior_cfg.get("small_object", {})
                                        else:
                                            multipliers = prior_cfg.get("large_object", {})
                                        
                                        if ym.canonical_label in multipliers:
                                            mult *= float(multipliers[ym.canonical_label])
                            # Add confidence to break ties between multiple proposal crop detections on the same crop
                            score = mult * (0.35 * iou + 0.35 * mask_overlap + 0.30 * proximity + 0.05 * ym_conf)
                            eligible = iou >= 0.05 or mask_overlap >= 0.20 or distance <= 0.06
                            if eligible and score > best_score:
                                best_score = score
                                best_yolo = ym
                                best_yolo_conf = ym_conf

                    if best_yolo is not None:
                        om.canonical_label = best_yolo.canonical_label
                        om.raw_label = best_yolo.raw_label
                        om.confidence = best_yolo_conf
                        om.predicted_label = best_yolo.canonical_label
                        om.inference_source = best_yolo.inference_source
                        alternatives = []
                        if best_yolo.semantic_alternatives:
                            alternatives = list(best_yolo.semantic_alternatives)
                        for ym in object_yolo_masks:
                            if ym is not best_yolo:
                                yx1, yy1, yx2, yy2 = ym.bounding_box_xyxy
                                ix1, iy1 = max(ox1, yx1), max(oy1, yy1)
                                ix2, iy2 = min(ox2, yx2), min(oy2, yy2)
                                if ix2 > ix1 and iy2 > iy1:
                                    inter = (ix2 - ix1) * (iy2 - iy1)
                                    union = om_area + (yx2 - yx1) * (yy2 - yy1) - inter
                                    iou = inter / max(1, union)
                                    mask_overlap = int(np.count_nonzero(om.binary_mask & ym.binary_mask)) / max(
                                        1, min(int(np.count_nonzero(om.binary_mask)), int(np.count_nonzero(ym.binary_mask))))
                                    distance = float(np.linalg.norm(
                                        np.asarray(om.centroid_world_m) - np.asarray(ym.centroid_world_m)))
                                    if iou >= 0.05 or mask_overlap >= 0.20 or distance <= 0.06:
                                        alternatives.append({
                                            "canonical_label": ym.canonical_label,
                                            "raw_label": ym.raw_label,
                                            "confidence": ym.confidence,
                                            "inference_source": ym.inference_source,
                                        })
                        
                        # Apply size heuristic to all gathered alternatives
                        filtered_alts = []
                        for alt in alternatives:
                            alt_conf = alt["confidence"]
                            alt_label = alt["canonical_label"]
                            if om.cloud_bounds_world_m:
                                min_b = om.cloud_bounds_world_m.get("minimum_world_m")
                                max_b = om.cloud_bounds_world_m.get("maximum_world_m")
                                if min_b and max_b:
                                    prior_cfg = self.raw_config.get("semantic_physical_prior", {})
                                    if prior_cfg.get("enabled", False):
                                        max_dim = float(np.max(np.asarray(max_b, dtype=float) - np.asarray(min_b, dtype=float)))
                                        small_thresh = prior_cfg.get("small_object_max_dimension_m", 0.08)
                                        if max_dim < small_thresh:
                                            multipliers = prior_cfg.get("small_object", {})
                                        else:
                                            multipliers = prior_cfg.get("large_object", {})
                                        
                                        if alt_label in multipliers:
                                            alt_conf *= multipliers[alt_label]
                            if alt_conf > 1e-5:
                                alt["confidence"] = alt_conf
                                filtered_alts.append(alt)
                        om.semantic_alternatives = filtered_alts
                    else:
                        om.canonical_label = "unknown"
                        om.raw_label = "unknown"
                        om.confidence = 0.0
                        om.predicted_label = "unknown"

            obs.detected_masks = (
                masks if is_oracle_mask else
                [mask for mask in masks if mask.canonical_label.lower() in self.object_categories]
            )

            backend_for_diagnostics = self._yolo_aux_backend if self._yolo_aux_backend is not None else self.proposal_backend
            current_diagnostics = list(getattr(
                backend_for_diagnostics, "last_diagnostics", []))
            for record in current_diagnostics:
                self.detection_diagnostics.append({
                    **record,
                    "stage_index": stage_idx,
                    "source_region_id": source_region_id,
                })
            artifact_cfg = self.raw_config.get("artifacts", {})
            save_all_overlays = bool(artifact_cfg.get("save_all_detection_overlays", False))
            if (self.output_dir is not None and not is_oracle_mask
                    and (save_all_overlays or
                         (obs.camera_id == "DETAIL" and stage_idx <= 1))):
                self._save_detection_visual(
                    obs, masks, current_diagnostics, stage_idx, source_region_id)

        # Update persistent tracking
        self.tracker.update_with_stage_observations(
            stage_index=stage_idx,
            source_region_id=source_region_id,
            observations=raw_obs,
            stage_volume_min=stage_min,
            stage_volume_max=stage_max,
        )

        # Region instances are stable; semantic observations accumulate across
        # every available stage rather than being frozen at the initial view.
        self.candidate_regions = self.region_grounder.discover_candidate_regions(
            self.scene, raw_obs, stage_index=stage_idx)
        self.graph.update_from_observed_regions(self.candidate_regions, stage_idx=stage_idx)

        return raw_obs

    def _save_detection_visual(
        self, observation: ViewObservation, masks: list[ObservedMask],
        diagnostics: list[dict[str, Any]], stage_idx: int, source_region_id: str,
    ) -> None:
        """Persist a compact RGB-only detector sanity overlay; never used by decisions."""
        visual_dir = self.output_dir / "representative_visuals"
        visual_dir.mkdir(parents=True, exist_ok=True)
        safe_region = source_region_id.lower().replace(" ", "_")
        safe_camera = observation.camera_id.lower().replace(" ", "_")
        stem = f"stage_{stage_idx:02d}_{safe_region}_{safe_camera}"
        if bool(self.raw_config.get("artifacts", {}).get("save_raw_rgb", False)):
            raw_dir = self.output_dir / "raw_rgb"
            raw_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(raw_dir / f"{stem}.jpg"),
                cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR),
            )
        canvas = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
        status_colors = {
            "ACCEPTED": (0, 200, 0),
            "ACCEPTED_PRE_DEDUP": (0, 200, 0),
            "SUPPRESSED_DUPLICATE": (0, 0, 255),
        }
        for record in diagnostics:
            raw = record.get("raw_yolo_bbox_xyxy")
            if not raw:
                continue
            status = str(record.get("status", "PREDICTION"))
            color = status_colors.get(status, (0, 165, 255))
            cv2.rectangle(canvas, (raw[0], raw[1]), (raw[2], raw[3]), color, 1)
            label = (f"{status} {record.get('canonical_label', 'unknown')} "
                     f"{float(record.get('confidence', 0.0)):.3f}")
            cv2.putText(canvas, label, (raw[0], min(canvas.shape[0] - 3, raw[3] + 11)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        for mask in masks:
            raw = mask.raw_yolo_bbox_xyxy or mask.bounding_box_xyxy
            refined = mask.refined_bbox_xyxy or mask.bounding_box_xyxy
            cv2.rectangle(canvas, (raw[0], raw[1]), (raw[2], raw[3]), (0, 170, 255), 1)
            cv2.rectangle(canvas, (refined[0], refined[1]), (refined[2], refined[3]), (0, 255, 0), 2)
            label = f"RETAINED {mask.canonical_label} {mask.confidence:.3f} {mask.inference_source}"
            cv2.putText(canvas, label, (raw[0], max(12, raw[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(
            str(visual_dir / f"stage_{stage_idx:03d}_{safe_region}_{safe_camera}.jpg"),
            canvas,
        )

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
                g_res = (self._skipped_geometry_result(trk.instance_id, driver_req)
                         if is_semantic_only else
                         self.geometric_grounder.ground_object_geometry(trk, driver_req))
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
                g_res = (self._skipped_geometry_result(trk.instance_id, fastener_req)
                         if is_semantic_only else
                         self.geometric_grounder.ground_object_geometry(trk, fastener_req))
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
                g_res = (self._skipped_geometry_result(reg.region_instance_id, surface_req)
                         if is_semantic_only else
                         self.geometric_grounder.ground_region_geometry(reg, surface_req))
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
                g_res = (self._skipped_geometry_result(reg.region_instance_id, container_req)
                         if is_semantic_only else
                         self.geometric_grounder.ground_region_geometry(reg, container_req))
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

        # SEMANTIC_ONLY and NO_JOINT_COUPLING choose independent candidates.
        # The former candidate pools are semantic-only; the latter are
        # semantic+unary-geometry pools. Neither invokes relational search.
        if self.ablation in {AblationType.SEMANTIC_ONLY, AblationType.NO_GEOMETRY,
                             AblationType.NO_JOINT_COUPLING}:
            if driver_candidates and fastener_candidates:
                d_best = max(driver_candidates, key=lambda x: x.overall_confidence)
                f_best = max(fastener_candidates, key=lambda x: x.overall_confidence)
                wit = FunctionalWitness(
                    driver_id=d_best.instance_id,
                    fastener_id=f_best.instance_id,
                    work_surface_id="MAIN_WORKBENCH_ZONE",
                    parts_container_id=None,
                    overall_confidence=float(d_best.overall_confidence * f_best.overall_confidence),
                    verification_details={"ablation": self.ablation.value, "unary_selected": True,
                                          "geometry_used_for_decision": self.ablation == AblationType.NO_JOINT_COUPLING,
                                          "fixed_insertion_target": "MAIN_WORKBENCH_ZONE"},
                )
                return wit, None

        # Full joint search
        witness, evaluated_tuples = self.functional_search.search_witness(
            driver_candidates=driver_candidates,
            fastener_candidates=fastener_candidates,
            work_surface_candidates=surface_candidates,
            parts_container_candidates=container_candidates,
            requirements=self.requirements,
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
            grounding_results=grounding_results,
            requirements=self.requirements,
            has_unresolved_evidence=any(
                result.combined_status == GroundingStatus.UNKNOWN
                for result in grounding_results
            ),
        )

        return None, rej_reason
