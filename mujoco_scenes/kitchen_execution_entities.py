"""Observed-evidence entity resolution at the kitchen execution boundary.

Generic object IDs and functional assignments stay frozen.  Backend body
names enter only in the output of :class:`KitchenExecutionEntityResolver`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .kitchen_execution_policy import CONTAINER_WORKSPACES, KitchenWorkspace


CONFIG_PATH = Path(__file__).parent / "configs/kitchen_execution_semantics.json"


class SourceKind(str, Enum):
    TABLE = "TABLE"
    DRAWER = "DRAWER"
    CUPBOARD = "CUPBOARD"
    BOX = "BOX"
    SUPPORT = "SUPPORT"


@dataclass(frozen=True)
class ObjectSourceContext:
    object_id: str
    source_kind: SourceKind
    source_container: str | None
    required_workspace: KitchenWorkspace
    container_must_be_open: bool
    observed_source_region: str
    observed_source_stage: int
    observed_measurement_cloud_path: str | None


@dataclass(frozen=True)
class ExecutionCandidate:
    backend_body: str
    semantic_label: str
    grasp_family: str
    source_region: str
    centroid_world_m: tuple[float, float, float]


def load_execution_semantics(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def source_context(object_id: str, record: dict[str, Any]) -> ObjectSourceContext:
    region = str(record.get("source_region") or "UNKNOWN")
    if region in {"INITIAL", "countertop", "table"}:
        kind, container, workspace = SourceKind.TABLE, None, KitchenWorkspace.HOME
    elif region in {"D1", "D2"}:
        kind, container, workspace = SourceKind.DRAWER, region, CONTAINER_WORKSPACES[region]
    elif region in {"C1", "C2"}:
        kind, container, workspace = SourceKind.CUPBOARD, region, CONTAINER_WORKSPACES[region]
    elif region == "B1":
        kind, container, workspace = SourceKind.BOX, region, CONTAINER_WORKSPACES[region]
    else:
        kind, container, workspace = SourceKind.SUPPORT, None, KitchenWorkspace.HOME
    centroid = record.get("centroid_world_m", {})
    return ObjectSourceContext(
        object_id=object_id,
        source_kind=kind,
        source_container=container,
        required_workspace=workspace,
        container_must_be_open=container is not None,
        observed_source_region=region,
        observed_source_stage=int(record.get("first_seen_stage", 0)),
        observed_measurement_cloud_path=centroid.get("measurement_cloud_path"),
    )


def validated_semantic_label(record: dict[str, Any]) -> str | None:
    validated = record.get("semantics", {}).get("validated") or {}
    label = validated.get("canonical_label")
    return str(label) if label else None


def semantic_label_with_provenance(
    object_id: str,
    record: dict[str, Any],
    roles: dict[str, set[str]],
    forced_semantics: dict[str, str],
    config: dict[str, Any],
) -> tuple[str | None, str, str | None]:
    """Resolve execution semantics without consulting a backend object name."""
    if object_id in forced_semantics:
        source_role = next(iter(sorted(roles.get(object_id, ()))), None)
        return forced_semantics[object_id], "FROZEN_SOURCE_ROLE", source_role
    observed = validated_semantic_label(record)
    if observed is not None:
        return observed, "OBSERVED_SEMANTIC_DETECTOR", None
    calibration = record.get("execution_scene_calibration") or {}
    current_observed = calibration.get("current_observed_semantic_label")
    if current_observed is not None:
        return (
            current_observed,
            "EXECUTION_CALIBRATION_OBSERVED_SEMANTIC",
            None,
        )
    fallback = config.get("functional_role_fallback_labels", {})
    for role in sorted(roles.get(object_id, ())):
        if role in fallback:
            return fallback[role], "FROZEN_FUNCTIONAL_ROLE_FALLBACK", role
    return None, "UNAVAILABLE", None


def _selected_roles(
    assignments: dict[str, Any],
    source_role_labels: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    roles: dict[str, set[str]] = {}
    forced_semantics: dict[str, str] = {}
    for object_id in assignments.get("coffee_targets", []):
        roles.setdefault(object_id, set()).add("coffee_vessel")
    for object_id in assignments.get("soup_targets", []):
        roles.setdefault(object_id, set()).add("soup_bowl")
    for row in assignments.get("coffee_stirring", []):
        object_id = row.get("tool_object_id") or row.get("source_object_id")
        if object_id is None and row.get("relation_checks"):
            object_id = row["relation_checks"][0].get("from_object")
        if object_id:
            roles.setdefault(object_id, set()).add("coffee_stirrer")
    for row in assignments.get("soup_serving", []):
        object_id = row.get("tool_object_id") or row.get("source_object_id")
        if object_id is None and row.get("relation_checks"):
            object_id = row["relation_checks"][0].get("from_object")
        if object_id:
            roles.setdefault(object_id, set()).add("soup_utensil")
    for role, object_id in assignments.get("source_roles", {}).items():
        roles.setdefault(object_id, set()).add(role)
        forced_semantics[object_id] = source_role_labels.get(role, role)
    return roles, forced_semantics


def build_phase_b_inventory(
    registry: dict[str, Any],
    assignments: dict[str, Any],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    config = load_execution_semantics()
    roles, forced_semantics = _selected_roles(
        assignments, config.get("source_role_semantic_labels", {})
    )
    usage: dict[str, list[dict[str, Any]]] = {}
    for action in plan:
        for index, argument in enumerate(action.get("arguments", [])):
            if argument in registry["objects"]:
                usage.setdefault(argument, []).append(
                    {"step": action["step"], "action": action["action"].upper(), "argument_index": index}
                )
    calibrated = {
        object_id for object_id, record in registry["objects"].items()
        if record.get("execution_scene_calibration") is not None
    }
    relevant = sorted(set(roles) | set(usage) | calibrated)
    rows = []
    for object_id in relevant:
        record = registry["objects"][object_id]
        context = source_context(object_id, record)
        semantic_label, semantic_label_source, originating_role = (
            semantic_label_with_provenance(
                object_id, record, roles, forced_semantics, config
            )
        )
        rows.append(
            {
                "generic_object_id": object_id,
                "semantic_label": semantic_label,
                "semantic_label_source": semantic_label_source,
                "originating_functional_role": originating_role,
                "selected_functions": sorted(roles.get(object_id, ())),
                "phase2_usage": usage.get(object_id, []),
                "observed_centroid_world_m": record["centroid_world_m"]["value"],
                # Stage-local measured geometry is carried across the execution
                # boundary so placement may reason about payload footprints.
                # These are observed values, never MuJoCo geom sizes.
                "observed_dimensions_m": {
                    axis: property_record.get("value")
                    for axis, property_record in record.get("dimensions_m", {}).items()
                    if property_record.get("status") in {"MEASURED", "DERIVED"}
                },
                "source_context": {
                    **asdict(context),
                    "source_kind": context.source_kind.value,
                    "required_workspace": context.required_workspace.value,
                },
                "execution_scene_calibration": deepcopy(
                    record.get("execution_scene_calibration")
                ),
                "backend_binding_present": False,
            }
        )
    return {
        "schema_version": 1,
        "scene_name": registry["scene_name"],
        "inference_boundary": "FROZEN_OBSERVED_STATE_AND_WITNESS_ONLY",
        "planner_received_backend_names": False,
        "evaluation_instance_tokens_excluded": True,
        "objects": rows,
    }


class KitchenExecutionEntityResolver:
    """Deterministic one-to-one centroid matching after semantic/source gates."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_execution_semantics()

    def classify_backend_kind(self, kind: str) -> tuple[str, str] | None:
        lowered = kind.lower()
        for rule in self.config["rules"]:
            if any(token in lowered for token in rule["contains"]):
                return rule["canonical_label"], rule["grasp_family"]
        return None

    def candidates_from_scene(
        self, scene, *, observed_regions: set[str] | None = None
    ) -> list[ExecutionCandidate]:
        mujoco.mj_forward(scene.model, scene.data)
        candidates = []
        for body, kind, container in scene._object_instance_records:
            # A closed, uninspected backend object cannot participate in the
            # resolution of visible evidence.
            if (
                container is not None
                and container not in scene.state.opened_containers
                and container not in (observed_regions or set())
            ):
                continue
            classification = self.classify_backend_kind(kind)
            if classification is None:
                continue
            body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id < 0:
                continue
            label, family = classification
            candidates.append(
                ExecutionCandidate(
                    backend_body=body,
                    semantic_label=label,
                    grasp_family=family,
                    source_region=container or "countertop",
                    centroid_world_m=tuple(float(x) for x in scene.data.xpos[body_id]),
                )
            )
        return candidates

    def resolve(
        self,
        inventory: dict[str, Any],
        candidates: list[ExecutionCandidate],
    ) -> dict[str, Any]:
        compatibility = self.config["semantic_compatibility"]
        table_limit = float(self.config["maximum_centroid_error_m"])
        storage_limit = float(
            self.config.get("storage_fixture_centroid_error_m", table_limit)
        )
        edges = []
        rejection_rows = []
        for row in inventory["objects"]:
            observed = np.asarray(row["observed_centroid_world_m"], float)
            allowed = set(compatibility.get(row["semantic_label"], [row["semantic_label"]]))
            region = row["source_context"]["observed_source_region"]
            normalized_region = "countertop" if region in {"INITIAL", "table"} else region
            # Storage evidence is captured after its articulated fixture has
            # opened, whereas execution association begins from the closed
            # fixture.  Keep semantic and source-region gates strict, but use
            # a separately bounded displacement allowance for that known
            # frame change.  Table evidence retains the original gate.
            limit = table_limit if normalized_region == "countertop" else storage_limit
            for candidate in candidates:
                semantic_ok = candidate.semantic_label in allowed
                source_ok = candidate.source_region == normalized_region
                error = float(np.linalg.norm(observed - candidate.centroid_world_m))
                if semantic_ok and source_ok and error <= limit:
                    edges.append((error, row["generic_object_id"], candidate.backend_body, candidate))
                else:
                    rejection_rows.append({
                        "generic_object_id": row["generic_object_id"],
                        "backend_body": candidate.backend_body,
                        "semantic_consistent": semantic_ok,
                        "source_context_consistent": source_ok,
                        "centroid_error_m": error,
                    })
        assigned_objects: set[str] = set()
        assigned_bodies: set[str] = set()
        accepted = []
        for error, object_id, body, candidate in sorted(edges, key=lambda x: (x[0], x[1], x[2])):
            if object_id in assigned_objects or body in assigned_bodies:
                continue
            row = next(item for item in inventory["objects"] if item["generic_object_id"] == object_id)
            assigned_objects.add(object_id)
            assigned_bodies.add(body)
            accepted.append({
                "generic_object_id": object_id,
                "semantic_label": row["semantic_label"],
                "semantic_label_source": row.get("semantic_label_source"),
                "originating_functional_role": row.get(
                    "originating_functional_role"
                ),
                "selected_functions": row["selected_functions"],
                "observed_centroid_world_m": row["observed_centroid_world_m"],
                "observed_source_context": row["source_context"],
                "physical_backend_body": body,
                "backend_initial_centroid_world_m": list(candidate.centroid_world_m),
                "centroid_error_m": error,
                "semantic_consistent": True,
                "source_context_consistent": True,
                "one_to_one": True,
                "accepted": True,
                "grasp_family": candidate.grasp_family,
                "resolution_method": "SEMANTIC_SOURCE_GATE_THEN_NEAREST_CENTROID_V1",
                "simulation_backend_only": True,
            })
        unresolved = sorted(
            row["generic_object_id"] for row in inventory["objects"]
            if row["generic_object_id"] not in assigned_objects
        )
        return {
            "schema_version": 1,
            "scene_name": inventory["scene_name"],
            "planner_received_backend_names": False,
            "resolution_method": "SEMANTIC_SOURCE_GATE_THEN_NEAREST_CENTROID_V1",
            "accepted": sorted(accepted, key=lambda row: row["generic_object_id"]),
            "unresolved_object_ids": unresolved,
            "all_resolved": not unresolved,
            "one_to_one": len(assigned_bodies) == len(accepted),
            "maximum_centroid_error_m": max((row["centroid_error_m"] for row in accepted), default=None),
            "rejected_candidate_edges": rejection_rows,
        }
