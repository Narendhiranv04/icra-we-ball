"""Build the observed-state artifacts required by kitchen physical execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from .kitchen_execution_entities import (
    KitchenExecutionEntityResolver,
    build_phase_b_inventory,
)
from .scene_loader import KitchenScene
from .symbolic_planning import SymbolicCompilationError, compile_plan_and_save


DEFAULT_TASK = (
    Path(__file__).resolve().parent
    / "configs"
    / "s1_integrated_kitchen_object_function.yaml"
)


class KitchenExecutionBundleError(RuntimeError):
    """Raised when frozen observed IDs cannot safely enter execution."""


@dataclass(frozen=True)
class KitchenExecutionBundle:
    output_dir: Path
    scene: Any
    inventory: dict[str, Any]
    resolution: dict[str, Any]
    registry: dict[str, Any]
    witness: dict[str, Any]
    plan: list[dict[str, Any]]
    symbolic_result: dict[str, Any]


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KitchenExecutionBundleError(
            f"Execution artifact must contain a JSON object: {path}"
        )
    return payload


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_kitchen_execution_bundle(
    phase1_run_dir: str | Path,
    *,
    output_dir: str | Path,
    task_requirements: str | Path = DEFAULT_TASK,
    scene_factory: Callable[..., Any] = KitchenScene,
    resolver: KitchenExecutionEntityResolver | None = None,
    include_all_observed_objects: bool = False,
) -> KitchenExecutionBundle:
    """Compile a complete witness and resolve its IDs in a fresh robot scene.

    The symbolic planner receives only the frozen observed registry and witness.
    Backend body names enter later, inside the execution-only resolver.
    """
    phase1 = Path(phase1_run_dir).resolve()
    output = Path(output_dir).resolve()
    required = (
        phase1 / "object_registry.json",
        phase1 / "observed_graph.json",
        phase1 / "latest_witness.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise KitchenExecutionBundleError(
            "Phase-1 run is missing required artifacts: " + ", ".join(missing)
        )

    registry = _read(phase1 / "object_registry.json")
    _read(phase1 / "observed_graph.json")
    witness = _read(phase1 / "latest_witness.json")
    if str(witness.get("status", "")).upper() != "COMPLETE":
        reasons = ", ".join(map(str, witness.get("reason_codes", ())))
        detail = f" ({reasons})" if reasons else ""
        raise KitchenExecutionBundleError(
            f"Phase-1 witness is {witness.get('status')}, not COMPLETE{detail}"
        )

    output.mkdir(parents=True, exist_ok=True)
    try:
        symbolic_result = compile_plan_and_save(
            phase1,
            task_requirements,
            output_dir=output,
        )
    except SymbolicCompilationError as error:
        raise KitchenExecutionBundleError(
            f"Symbolic compilation failed: {error}"
        ) from error
    assignments = symbolic_result["compiled"]["role_assignments"]
    plan = symbolic_result["plan"]
    inventory = build_phase_b_inventory(
        registry,
        assignments,
        plan,
        include_all_observed_objects=include_all_observed_objects,
    )

    scene = scene_factory(
        inventory["scene_name"],
        include_robot=True,
        robot="google",
    )
    entity_resolver = resolver or KitchenExecutionEntityResolver()
    observed_regions = {
        str(row["source_context"]["source_container"])
        for row in inventory["objects"]
        if row["source_context"].get("source_container")
    }
    candidates = entity_resolver.candidates_from_scene(
        scene,
        observed_regions=observed_regions,
    )
    resolution = entity_resolver.resolve(inventory, candidates)
    if not resolution.get("all_resolved") or not resolution.get("one_to_one"):
        unresolved = resolution.get("unresolved_object_ids", [])
        raise KitchenExecutionBundleError(
            "Observed IDs did not resolve one-to-one in the Google-robot "
            f"execution scene: {unresolved}"
        )

    _write(output / "execution_inventory.json", inventory)
    _write(output / "execution_entity_resolution.json", resolution)
    _write(output / "object_registry.json", registry)
    _write(output / "functional_witness.json", witness)
    _write(output / "planner_output.json", plan)
    _write(
        output / "execution_bundle_manifest.json",
        {
            "schema_version": 1,
            "phase1_run_dir": str(phase1),
            "task_requirements": str(Path(task_requirements).resolve()),
            "scene_name": inventory["scene_name"],
            "robot": "google",
            "planner": "deterministic_astar_symbolic_state_search",
            "planner_received_backend_names": False,
            "execution_resolution_all_resolved": True,
            "action_count": len(plan),
            "observed_regions": sorted(observed_regions),
        },
    )
    return KitchenExecutionBundle(
        output,
        scene,
        inventory,
        resolution,
        registry,
        witness,
        plan,
        symbolic_result,
    )
