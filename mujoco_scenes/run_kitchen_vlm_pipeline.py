"""Run Kitchen Qwen requirements through observed search to an action plan.

The online path receives only the goal, raw initial RGB views, observable
storage-region descriptions, and evidence revealed by opening those regions.
No oracle outcome, configured hidden-content list, GT assignment, or expected
GT action file is passed into Qwen, satisfaction, allocation, or planning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from .final_paper_variant_labels import (
    VARIANT_LABELS,
    paper_variant_label,
    resolve_variant_name,
)
from .scene_loader import KitchenScene
from .sequential_inspection import run_sequential_inspection
from .symbolic_planning import (
    SymbolicCompilationError,
    compile_plan_and_save,
)
from .kitchen_vlm_functional_graph import (
    KITCHEN_OBSERVABLE_REGIONS,
    compile_vlm_functional_graph,
)
from .workshop_phase1.fm_adapter import FMAdapter


KITCHEN_INITIAL_CAMERAS = (
    "left_shoulder_camera",
    "right_shoulder_camera",
    "overhead_camera",
    "side_camera",
    "front_camera",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(path)


def _render_initial(scene: KitchenScene, output: Path, width: int, height: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for camera in KITCHEN_INITIAL_CAMERAS:
        path = output / f"{camera}.png"
        Image.fromarray(scene.render_frame(camera=camera, width=width, height=height)).save(path)
        paths.append(path)
    return paths


def _action_mapping_trace(
    *,
    task_contract: dict[str, Any],
    witness: dict[str, Any],
    opened: list[str],
    plan_records: list[dict[str, Any]],
    plan_dir: Path,
) -> dict[str, Any]:
    assignments = {}
    assignment_path = plan_dir / "grounded_role_assignments.json"
    if assignment_path.exists():
        assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
    symbolic = task_contract.get("symbolic_task", {})
    target_requirements = symbolic.get("target_requirements", {})
    coffee_requirement = target_requirements.get("coffee", {})
    soup_requirement = target_requirements.get("soup", {})
    coffee_target_role = str(coffee_requirement.get("witness_role", ""))
    soup_target_role = str(soup_requirement.get("witness_role", ""))
    coffee_group_id = str(coffee_requirement.get("requires_operation_group", ""))
    soup_group_id = str(soup_requirement.get("requires_operation_group", ""))
    groups = task_contract.get("operation_groups", {})
    coffee_tool_role = str(groups.get(coffee_group_id, {}).get("tool_role", ""))
    soup_tool_role = str(groups.get(soup_group_id, {}).get("tool_role", ""))

    roles_by_object: dict[str, list[str]] = {}
    for object_id in assignments.get("coffee_targets", []):
        roles_by_object.setdefault(object_id, []).append(coffee_target_role)
    for object_id in assignments.get("soup_targets", []):
        roles_by_object.setdefault(object_id, []).append(soup_target_role)
    for role, object_id in assignments.get("source_roles", {}).items():
        roles_by_object.setdefault(object_id, []).append(role)
    for row in assignments.get("coffee_stirring", []):
        roles_by_object.setdefault(row["utensil_object_id"], []).append(coffee_tool_role)
    for row in assignments.get("soup_serving", []):
        roles_by_object.setdefault(row["utensil_object_id"], []).append(soup_tool_role)
    mapped_actions = []
    for record in plan_records:
        mapped_actions.append({
            **record,
            "argument_role_bindings": {
                argument: sorted(set(roles_by_object.get(argument, [])))
                for argument in record["arguments"]
                if argument in roles_by_object
            },
            "derivation": (
                "generic symbolic operator selected by deterministic search over "
                "the observed state, verified assignments, causal dependencies, "
                "and goal facts"
            ),
        })
    return {
        "schema_version": 1,
        "purpose": "EXPLICIT_FUNCTIONAL_WITNESS_TO_ACTION_SEQUENCE_MAPPING",
        "physical_execution_primitives_included": False,
        "inspection_actions": [
            {"step": index, "action": "OPEN", "region_id": region}
            for index, region in enumerate(opened, 1)
        ],
        "functional_satisfaction": {
            "status": witness.get("status"),
            "stage": witness.get("stage"),
            "selected_witness": witness.get("selected_witness"),
            "operation_assignments": witness.get("operation_assignments", []),
        },
        "grounded_role_assignments": assignments,
        "general_task_model_used": {
            "causal_dependencies": task_contract.get("symbolic_task", {}).get(
                "causal_dependencies", []
            ),
            "target_requirements": task_contract.get("symbolic_task", {}).get(
                "target_requirements", {}
            ),
            "source_roles": task_contract.get("symbolic_task", {}).get(
                "source_roles", {}
            ),
        },
        "planner": {
            "algorithm": "deterministic_astar_symbolic_state_search",
            "operator_source": "generic Kitchen symbolic action schemas",
            "gt_action_sequence_input": None,
        },
        "mapped_task_actions": mapped_actions,
    }


def run_pipeline(
    variant: str,
    *,
    output_root: Path,
    semantic_backend: str = "yolo_world",
    semantic_model: str | None = None,
    width: int = 640,
    height: int = 480,
    fm_adapter: FMAdapter | None = None,
) -> dict[str, Any]:
    internal_variant = resolve_variant_name("kitchen", variant)
    if internal_variant not in VARIANT_LABELS["kitchen"]:
        raise ValueError(f"Unknown Kitchen variant: {variant}")

    # This public naming convention selects the simulator scene without loading
    # the benchmark's intended outcomes or per-variant hidden-content manifest.
    variant_code = internal_variant.split("_", 1)[0]
    scene_name = f"S1_integrated_kitchen_object_function_feasibility_{variant_code}"
    scene = KitchenScene(scene_name, include_robot=False, robot="none")
    run_root = output_root / paper_variant_label("kitchen", internal_variant)
    image_paths = _render_initial(scene, run_root / "initial_observation", width, height)

    adapter = fm_adapter or FMAdapter()
    fm_calls_before = adapter.metrics.total_calls
    task_instruction = str(getattr(scene.config, "goal", "")) or (
        "Prepare and serve coffee and soup for two people using the available "
        "kitchenware. Stir both coffees and provide each soup bowl with a "
        "suitable utensil."
    )
    raw_graph = adapter.generate_kitchen_functional_graph(
        task_instruction,
        KITCHEN_OBSERVABLE_REGIONS,
        observation_images=image_paths,
    )
    fm_calls_for_run = adapter.metrics.total_calls - fm_calls_before
    if fm_calls_for_run != 1:
        raise RuntimeError(
            f"Kitchen VLM pipeline requires exactly one Qwen call; observed {fm_calls_for_run}"
        )
    task_contract, detector_vocabularies, transformation_trace = (
        compile_vlm_functional_graph(
            raw_graph,
            task_instruction=task_instruction,
            observable_regions=tuple(KITCHEN_OBSERVABLE_REGIONS),
        )
    )
    order = list(raw_graph["inspection_order"])
    requirements_path = run_root / "vlm_task_requirements.yaml"
    _atomic_json(
        run_root / "01_raw_vlm_functional_graph.json", raw_graph
    )
    _atomic_json(
        run_root / "02_vlm_graph_to_task_contract.json", transformation_trace
    )
    _atomic_yaml(requirements_path, task_contract)
    object_vocabulary_path = run_root / "vlm_object_detector_vocabulary.yaml"
    _atomic_yaml(object_vocabulary_path, detector_vocabularies["object"])

    def all_functional_evidence_complete(session) -> bool:
        return (session.latest_witness or {}).get("status") == "COMPLETE"

    session = run_sequential_inspection(
        scene,
        order,
        runs_root=run_root / "observed_search",
        run_id="phase1",
        width=width,
        height=height,
        task_requirements=task_contract,
        stop_on_complete=True,
        semantic_backend=semantic_backend,
        semantic_model=semantic_model,
        semantic_vocabulary_path=object_vocabulary_path,
        grounding_mode="joint",
        completion_predicate=all_functional_evidence_complete,
        record_oracle_diagnostics=False,
    )
    witness = session.latest_witness or {}
    events = [
        json.loads(line)
        for line in session.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exhausted = any(event.get("event") == "INSPECTION_ORDER_EXHAUSTED" for event in events)
    complete = witness.get("status") == "COMPLETE"
    selected_witness = witness.get("selected_witness") or {}
    source_state = {
        "complete": complete,
        "assignments": {
            source_id: list(selected_witness.get(source["witness_role"], []))
            for source_id, source in task_contract["symbolic_task"][
                "source_roles"
            ].items()
        },
    }
    opened = [
        event["region_id"] for event in events
        if event.get("event") == "REGION_OPENED"
    ]

    plan_dir = run_root / "action_sequence"
    planning_error = None
    plan_records: list[dict[str, Any]] = []
    if complete:
        try:
            compile_plan_and_save(session.run_dir, task_contract, output_dir=plan_dir)
            plan_records = json.loads((plan_dir / "generated_plan.json").read_text())
        except SymbolicCompilationError as error:
            planning_error = str(error)
            complete = False

    ordered_actions = [
        {"step": index, "phase": "INSPECTION", "action": "OPEN", "arguments": [region]}
        for index, region in enumerate(opened, 1)
    ]
    for record in plan_records:
        ordered_actions.append({
            "step": len(ordered_actions) + 1,
            "phase": "TASK",
            "action": record["action"],
            "arguments": record["arguments"],
            "rendered": record["rendered"],
        })
    _atomic_json(
        run_root / "03_grounding_to_action_mapping.json",
        _action_mapping_trace(
            task_contract=task_contract,
            witness=witness,
            opened=opened,
            plan_records=plan_records,
            plan_dir=plan_dir,
        ),
    )
    _atomic_json(run_root / "ordered_action_sequence.json", ordered_actions)
    (run_root / "ordered_action_sequence.txt").write_text(
        "\n".join(
            f"{row['step']:03d}. {row.get('rendered') or row['action'] + '(' + ', '.join(row['arguments']) + ')'}"
            for row in ordered_actions
        ) + ("\n" if ordered_actions else ""),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "environment": "kitchen",
        "variant": paper_variant_label("kitchen", internal_variant),
        "terminal_status": (
            "ACTION_SEQUENCE_READY"
            if complete
            else "PLANNING_FAILED"
            if planning_error
            else "INFEASIBLE_AFTER_SEARCH"
        ),
        "vlm_initial_satisfaction_decision": raw_graph[
            "initial_satisfaction_assessment"
        ],
        "evidence_initial_satisfaction_decision": (
            witness.get("stage") == 0 and complete
        ),
        "vlm_inspection_order": order,
        "fm_calls": fm_calls_for_run,
        "opened_regions": opened,
        "inspection_exhausted": exhausted,
        "witness_status": witness.get("status"),
        "source_evidence": source_state,
        "planning_error": planning_error,
        "action_count": len(ordered_actions),
        "boundaries": {
            "qwen_received_goal_rgb_and_observable_region_interface_only": True,
            "qwen_received_oracle_or_hidden_contents": False,
            "semantic_grounding_started": True,
            "geometry_verification_started": True,
            "allocation_started": True,
            "planning_started": complete or planning_error is not None,
            "physical_execution_started": False,
            "reviewed_task_contract_used": False,
            "role_or_property_alias_mapping_used": False,
        },
    }
    _atomic_json(run_root / "pipeline_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="K1-K12 or internal variant name")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/kitchen_vlm_pipeline"))
    parser.add_argument("--semantic-backend", default="yolo_world")
    parser.add_argument("--semantic-model")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    result = run_pipeline(
        args.variant,
        output_root=args.output_root,
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
