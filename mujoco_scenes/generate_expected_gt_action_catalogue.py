"""Generate the readable expected GT action catalogue for all paper variants."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from typing import Any

from .final_paper_variant_labels import VARIANT_LABELS, paper_variant_label
from .kitchen_ground_truth_planner import (
    generate_ground_truth_plan,
    solve_ground_truth_assignment,
)
from .kitchen_ground_truth_state import initialize_oracle_world_state
from .run_kitchen_ground_truth_execution import load_variants_config
from .run_living_room_execution import DEFAULT_PHASE2_ROOT, EXPECTED_VARIANTS
from .living_room_variants import load_living_room_variant_contract
from .scene_loader import KitchenScene
from .workshop_ground_truth_planner import (
    generate_gt_plan,
    load_variant_specs,
    solve_gt_assignment,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "EXPECTED_GT_ACTIONS"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True))


def action_arguments(action: dict[str, Any]) -> list[Any]:
    arguments = action.get("arguments")
    if isinstance(arguments, list):
        return arguments
    if isinstance(arguments, dict):
        ordered_keys = ("object", "region", "carrying", "target_pose")
        return [arguments[key] for key in ordered_keys if key in arguments]
    if (
        str(action.get("operator", "")).upper() == "MOVE"
        and isinstance(action.get("result"), dict)
        and action["result"].get("final_pose") is not None
    ):
        return [action["result"]["final_pose"]]
    return [
        action[key]
        for key in ("object_id", "region_id", "object", "region", "carrying", "target_pose")
        if key in action and action[key] is not None
    ]


def normalized_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator": str(action.get("operator", "UNKNOWN")).upper(),
        "arguments": action_arguments(action),
        **({"reason": action["reason"]} if action.get("reason") else {}),
    }


def format_actions(
    environment: str,
    paper_variant: str,
    internal_variant: str,
    intended_outcome: str,
    description: str,
    actions: list[dict[str, Any]],
) -> str:
    rows = [
        f"Environment: {environment}",
        f"Variant: {paper_variant}",
        f"Internal variant: {internal_variant}",
        f"Intended outcome: {intended_outcome}",
        f"Description: {description}",
        f"Expected GT actions: {len(actions)}",
        "",
    ]
    for index, action in enumerate(actions, 1):
        arguments = ", ".join(
            json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            for value in action_arguments(action)
        )
        reason = f"  # {action['reason']}" if action.get("reason") else ""
        rows.append(f"{index:03d}. {str(action.get('operator', 'UNKNOWN')).upper()}({arguments}){reason}")
    return "\n".join(rows)


def _kitchen_expected(variant: str) -> tuple[str, str, list[dict[str, Any]]]:
    spec = load_variants_config()["variants"][variant]
    with redirect_stdout(io.StringIO()):
        scene = KitchenScene(spec["scene_name"], include_robot=True, robot="google")
    assignment = solve_ground_truth_assignment(
        scene, variant, spec.get("intended_outcome", "FEASIBLE")
    )
    state = initialize_oracle_world_state(scene._object_instance_records)
    actions = generate_ground_truth_plan(assignment, state)
    return (
        spec.get("intended_outcome", "UNKNOWN"),
        spec.get("description", ""),
        actions,
    )


def _living_expected(variant: str) -> tuple[str, str, list[dict[str, Any]]]:
    intended = "FEASIBLE" if EXPECTED_VARIANTS[variant] == "COMPLETE" else "INFEASIBLE"
    spec = load_living_room_variant_contract()["variants"][variant]
    plan_path = DEFAULT_PHASE2_ROOT / variant / "plan.json"
    if plan_path.exists():
        actions = _read(plan_path).get("actions", [])
        description = spec.get("description", "Execute the frozen symbolic PICK/PLACE task plan.")
    else:
        compilation = _read(DEFAULT_PHASE2_ROOT / variant / "compilation_result.json")
        reason = compilation.get("reason") or compilation.get("status") or "FUNCTIONAL_WITNESS_NOT_COMPLETE"
        actions = [{
            "operator": "TERMINATE_INFEASIBLE",
            "arguments": [reason],
            "reason": "No complete functional region assignment exists.",
        }]
        description = (
            f"{spec.get('description', 'Reject after exhaustive grounding')} "
            f"Grounding termination: {reason}."
        )
    return intended, description, actions


def _workshop_expected(variant: str) -> tuple[str, str, list[dict[str, Any]]]:
    spec = load_variant_specs()[variant]
    assignment = solve_gt_assignment(variant)
    return (
        spec["intended_outcome"],
        spec.get("description", ""),
        generate_gt_plan(assignment),
    )


def build_catalogue(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    builders = {
        "kitchen": _kitchen_expected,
        "living_room": _living_expected,
        "workshop": _workshop_expected,
    }
    records: list[dict[str, Any]] = []
    for environment, variants in VARIANT_LABELS.items():
        for variant in variants:
            label = paper_variant_label(environment, variant)
            intended, description, raw_actions = builders[environment](variant)
            actions = [normalized_action(action) for action in raw_actions]
            destination = output_root / environment / label
            payload = {
                "schema_version": 1,
                "environment": environment,
                "variant": label,
                "internal_variant": variant,
                "intended_outcome": intended,
                "description": description,
                "total_actions": len(actions),
                "actions": actions,
            }
            _write_json(destination / "expected_gt_actions.json", payload)
            _write(
                destination / "expected_gt_actions.txt",
                format_actions(
                    environment, label, variant, intended, description, actions
                ),
            )
            records.append({
                "environment": environment,
                "variant": label,
                "internal_variant": variant,
                "intended_outcome": intended,
                "description": description,
                "total_actions": len(actions),
                "text": str((destination / "expected_gt_actions.txt").relative_to(output_root)),
            })
    manifest = {"schema_version": 1, "total_variants": len(records), "records": records}
    _write_json(output_root / "manifest.json", manifest)
    index_rows = [
        "# Expected GT action catalogue",
        "",
        "These are the authoritative high-level GT task actions expected before physical execution.",
        "Every final-paper run saves the physically executed sequence beside this plan and reports an exact comparison.",
        "",
        "| Variant | Environment | Outcome | What it does | Actions | File |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for record in records:
        index_rows.append(
            f"| `{record['variant']}` | {record['environment']} | {record['intended_outcome']} | "
            f"{record['description']} | {record['total_actions']} | "
            f"[{record['text']}]({record['text']}) |"
        )
    _write(output_root / "README.md", "\n".join(index_rows))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_catalogue(args.output_root)
    print(f"Wrote {manifest['total_variants']} expected GT action sequences to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
