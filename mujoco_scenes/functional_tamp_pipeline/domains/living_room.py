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


TASK = (
    "Prepare the living room for two people watching television: put one cup "
    "and saucer on each personal support and the remote on a shared support."
)
LOCAL_MODEL = Path(__file__).resolve().parents[3] / "semantic_model_cache/yolov8m-worldv2.pt"


def run_to_plan(
    *, variant_label: str, internal_variant: str, mode: str,
    specification: FunctionalSpecification, output_dir: Path,
) -> PipelineResult:
    phase1 = output_dir / "observed_grounding"
    rig_path = output_dir / "resolved_rig.yaml"
    name = scene_name(internal_variant)
    write_resolved_integrated_rig(name, rig_path)
    detector, semantic_config = create_region_semantic_detector(
        checkpoint=str(LOCAL_MODEL), confidence_threshold=0.03,
        vocabulary_path=specification.metadata["semantic_vocabulary_path"],
    )
    scene = L2LivingRoomRegionScene(name, robot="none")
    if specification.raw_requirements:
        task_path = output_dir / "functional_task_contract.yaml"
        task_path.write_text(
            yaml.safe_dump(specification.raw_requirements[0], sort_keys=False),
            encoding="utf-8",
        )
        task_config = str(task_path)
    else:
        task_config = str(specification.metadata.get("contract_path"))
    run = IntegratedLivingRoomRegionRun(
        phase1, scene_name=name,
        task_config=task_config,
        evaluation_config=DEFAULT_EVALUATION_CONFIG,
        rig_config=rig_path, semantic_detector=detector,
        semantic_config=semantic_config, width=1280, height=960,
    ).run(scene)
    status = run.production_result["status"]
    if status != "COMPLETE":
        return PipelineResult(
            domain="living_room", variant=variant_label, mode=mode,
            status="INFEASIBLE", failure_reason=str(
                run.production_result.get("reason", "NO_GLOBAL_REGION_ASSIGNMENT")
            ),
        )
    plan_dir = output_dir / "action_sequence"
    planning = run_living_room_symbolic_pipeline(phase1, plan_dir)
    plan_payload = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    actions = tuple({
        "action_index": index + 1,
        "operator": row["operator"],
        "arguments": list(row["arguments"].values()),
    } for index, row in enumerate(plan_payload["actions"]))
    assignment = {row["slot_id"]: row["region_id"] for row in run.production_result["assignments"]}
    return PipelineResult(
        domain="living_room", variant=variant_label, mode=mode,
        status="ACTION_SEQUENCE_READY", assignment=assignment, plan=actions,
        search_statistics=planning.get("search_statistics", {}),
    )
