"""Canonical terminal manipulation subgoals and post-terminal goal coverage evaluation.

This module is strictly post-terminal benchmark evaluation infrastructure.
It must NEVER be accessible to baseline perception, planning, or refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..contracts import SerializableContract
from .base import (
    EvaluationContractError,
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    effect_exists,
    physical_on,
    required_sequence,
    required_string,
)

CANONICAL_SUBGOAL_COUNTS: Mapping[str, int] = {
    "kitchen": 12,
    "living_room": 5,
    "workshop": 3,
}


def canonical_subgoal_count(domain: str) -> int:
    """Return the exact canonical terminal subgoal count for a domain."""
    norm = domain.strip().lower().replace("-", "_")
    if norm not in CANONICAL_SUBGOAL_COUNTS:
        raise EvaluationContractError(f"unsupported benchmark domain: {domain!r}")
    return CANONICAL_SUBGOAL_COUNTS[norm]


@dataclass(frozen=True)
class TerminalSubgoal(SerializableContract):
    """One atomic terminal task condition required by the manipulation task.

    Represented in canonical triple format: (subject, predicate, target).
    """

    subject: str
    predicate: str
    target: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise EvaluationContractError("subgoal subject must not be empty")
        if not self.predicate.strip():
            raise EvaluationContractError("subgoal predicate must not be empty")
        if not self.target.strip():
            raise EvaluationContractError("subgoal target must not be empty")


@dataclass(frozen=True)
class SubgoalEvaluationResult(SerializableContract):
    """Result of evaluating a single terminal subgoal against physical state."""

    subgoal: TerminalSubgoal
    passed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubgoalCoverageEvaluation(SerializableContract):
    """Canonical terminal subgoal coverage evaluation for a run."""

    domain: str
    variant: str
    total_subgoals: int
    passed_subgoals: int
    coverage: float
    results: tuple[SubgoalEvaluationResult, ...]

    def __post_init__(self) -> None:
        if self.total_subgoals <= 0:
            raise EvaluationContractError("total_subgoals must be positive")
        if not (0 <= self.passed_subgoals <= self.total_subgoals):
            raise EvaluationContractError("passed_subgoals must be between 0 and total")


def canonical_terminal_subgoals(
    domain: str,
    hidden_context: HiddenBenchmarkContext,
) -> tuple[TerminalSubgoal, ...]:
    """Resolve the exact canonical GT terminal subgoals for a benchmark variant."""
    norm = domain.strip().lower().replace("-", "_")
    requirements = hidden_context.requirements

    if norm == "living_room":
        left_payloads = required_sequence(requirements, "left_payloads")
        right_payloads = required_sequence(requirements, "right_payloads")
        remote = required_string(requirements, "remote")
        left_support = required_string(requirements, "left_support")
        right_support = required_string(requirements, "right_support")
        shared_support = required_string(requirements, "shared_support")

        subgoals = []
        for p in left_payloads:
            subgoals.append(
                TerminalSubgoal(
                    subject=str(p),
                    predicate="ON",
                    target=left_support,
                    description=f"{p} on {left_support}",
                )
            )
        for p in right_payloads:
            subgoals.append(
                TerminalSubgoal(
                    subject=str(p),
                    predicate="ON",
                    target=right_support,
                    description=f"{p} on {right_support}",
                )
            )
        subgoals.append(
            TerminalSubgoal(
                subject=remote,
                predicate="ON",
                target=shared_support,
                description=f"{remote} on {shared_support}",
            )
        )
        return tuple(subgoals)

    if norm == "kitchen":
        coffee_vessels = required_sequence(requirements, "coffee_vessels")
        soup_vessels = required_sequence(requirements, "soup_vessels")
        serving_support = required_string(requirements, "serving_support")
        water_content = str(requirements.get("water_content", "water"))
        coffee_content = str(requirements.get("coffee_content", "coffee"))

        subgoals = []
        # 4 manipulation subgoals per coffee vessel = 8
        for cv in coffee_vessels:
            cv_str = str(cv)
            subgoals.append(
                TerminalSubgoal(
                    subject=cv_str,
                    predicate="ON",
                    target=serving_support,
                    description=f"{cv_str} physically served on {serving_support}",
                )
            )
            subgoals.append(
                TerminalSubgoal(
                    subject=cv_str,
                    predicate="HAS_CONTENT",
                    target=water_content,
                    description=f"{cv_str} has {water_content} delivered",
                )
            )
            subgoals.append(
                TerminalSubgoal(
                    subject=cv_str,
                    predicate="HAS_CONTENT",
                    target=coffee_content,
                    description=f"{cv_str} has {coffee_content} delivered",
                )
            )
            subgoals.append(
                TerminalSubgoal(
                    subject=cv_str,
                    predicate="STIRRED",
                    target="true",
                    description=f"{cv_str} stirred with suitable stirrer",
                )
            )

        # 2 manipulation subgoals per soup vessel = 4
        for sv in soup_vessels:
            sv_str = str(sv)
            subgoals.append(
                TerminalSubgoal(
                    subject=sv_str,
                    predicate="ON",
                    target=serving_support,
                    description=f"{sv_str} physically served on {serving_support}",
                )
            )
            subgoals.append(
                TerminalSubgoal(
                    subject=sv_str,
                    predicate="CONTAINS",
                    target="suitable_soup_utensil",
                    description=f"{sv_str} stably contains suitable soup utensil",
                )
            )
        return tuple(subgoals)

    if norm == "workshop":
        target = required_string(requirements, "target")
        workbench = required_string(requirements, "workbench")
        return (
            TerminalSubgoal(
                subject="screw",
                predicate="INSERTED_IN",
                target=target,
                description=f"compatible screw inserted into {target} with valid geometry",
            ),
            TerminalSubgoal(
                subject="joint",
                predicate="FASTENED",
                target="true",
                description="joint physically repaired / screw fully driven",
            ),
            TerminalSubgoal(
                subject="driver",
                predicate="ON",
                target=workbench,
                description=f"selected driver left safely on {workbench}",
            ),
        )

    raise EvaluationContractError(f"unsupported benchmark domain: {domain!r}")


def evaluate_terminal_subgoals(
    terminal_state: TerminalStateSnapshot,
    effect_ledger: Sequence[Mapping[str, Any] | SerializableContract],
    hidden_context: HiddenBenchmarkContext,
) -> SubgoalCoverageEvaluation:
    """Evaluate each canonical terminal subgoal against the physical snapshot."""
    if not isinstance(terminal_state, TerminalStateSnapshot):
        raise EvaluationContractError("evaluator requires a TerminalStateSnapshot")
    if not isinstance(hidden_context, HiddenBenchmarkContext):
        raise EvaluationContractError("evaluator requires a HiddenBenchmarkContext")

    domain = terminal_state.domain
    norm = domain.strip().lower().replace("-", "_")
    requirements = hidden_context.requirements
    subgoals = canonical_terminal_subgoals(domain, hidden_context)
    ledger_dicts = [
        item.to_dict() if isinstance(item, SerializableContract) else dict(item)
        for item in effect_ledger
    ]

    results: list[SubgoalEvaluationResult] = []

    if norm == "living_room":
        for sg in subgoals:
            obj_state = terminal_state.objects.get(sg.subject, {})
            passed = physical_on(obj_state, sg.target)
            results.append(
                SubgoalEvaluationResult(
                    subgoal=sg,
                    passed=passed,
                    evidence={"support": obj_state.get("support")},
                )
            )

    elif norm == "kitchen":
        water_sources = required_sequence(requirements, "water_sources")
        coffee_sources = required_sequence(requirements, "coffee_sources")
        stirrers = required_sequence(requirements, "suitable_stirrers")
        soup_utensils = required_sequence(requirements, "suitable_soup_utensils")
        contained = terminal_state.relations.get("contained_in", {})
        if not isinstance(contained, Mapping):
            contained = {}

        for sg in subgoals:
            obj_state = terminal_state.objects.get(sg.subject, {})
            passed = False
            evidence: dict[str, Any] = {}

            if sg.predicate == "ON":
                passed = physical_on(obj_state, sg.target)
                evidence["support"] = obj_state.get("support")

            elif sg.predicate == "HAS_CONTENT":
                content = sg.target
                sources = water_sources if content == "water" else coffee_sources
                passed = any(
                    effect_exists(ledger_dicts, "POUR_COMPLETED", (source, sg.subject, content))
                    for source in sources
                )
                evidence["content"] = content
                evidence["verified_pour"] = passed

            elif sg.predicate == "STIRRED":
                passed = any(
                    effect_exists(ledger_dicts, "STIR_COMPLETED", (tool, sg.subject))
                    for tool in stirrers
                )
                evidence["verified_stir"] = passed

            elif sg.predicate == "CONTAINS":
                bowl_contained = contained.get(sg.subject, ())
                matching_tools = [
                    str(t) for t in bowl_contained if str(t) in soup_utensils
                ]
                stable_tools = [
                    t for t in matching_tools
                    if terminal_state.objects.get(t, {}).get("contained_stably") is True
                ]
                passed = len(stable_tools) > 0
                evidence["matching_tools"] = matching_tools
                evidence["stable_tools"] = stable_tools

            results.append(SubgoalEvaluationResult(subgoal=sg, passed=passed, evidence=evidence))

    elif norm == "workshop":
        compatible_fasteners = set(required_sequence(requirements, "compatible_fasteners"))
        compatible_drivers = set(required_sequence(requirements, "compatible_drivers"))
        target = required_string(requirements, "target")
        workbench = required_string(requirements, "workbench")

        used_driver = terminal_state.measurements.get("used_driver")
        used_fastener = terminal_state.measurements.get("used_fastener")
        used_target = terminal_state.measurements.get("used_target")

        insertion = terminal_state.relations.get("insertion", {})
        if not isinstance(insertion, Mapping):
            insertion = {}

        min_depth = float(requirements.get("minimum_insertion_depth_m", 0.008))
        max_depth = float(requirements.get("maximum_insertion_depth_m", 0.018))
        radial_tolerance = float(requirements.get("radial_tolerance_m", 0.004))
        orientation_tolerance = float(requirements.get("orientation_tolerance_rad", 0.05))

        fastener = insertion.get("fastener")
        target_match = insertion.get("target") == target
        depth_ok = min_depth <= float(insertion.get("depth_m", float("-inf"))) <= max_depth
        radial_ok = float(insertion.get("radial_error_m", float("inf"))) <= radial_tolerance
        orient_ok = float(insertion.get("orientation_error_rad", float("inf"))) <= orientation_tolerance
        head_ok = insertion.get("head_above_tip") is True
        fastener_ok = fastener in compatible_fasteners if fastener else False

        insertion_geometry = bool(
            fastener_ok and target_match and depth_ok and radial_ok and orient_ok and head_ok
        )

        repaired = terminal_state.measurements.get("joint_repaired") is True
        certified_drive = bool(
            used_driver
            and used_fastener
            and used_target
            and effect_exists(
                ledger_dicts,
                "DRIVE_COMPLETED",
                (used_driver, used_fastener, used_target),
            )
        )

        driver_safe = bool(
            used_driver
            and physical_on(terminal_state.objects.get(str(used_driver), {}), workbench)
        )

        for sg in subgoals:
            if sg.predicate == "INSERTED_IN":
                results.append(
                    SubgoalEvaluationResult(
                        subgoal=sg,
                        passed=insertion_geometry,
                        evidence={
                            "fastener": fastener,
                            "target": insertion.get("target"),
                            "depth_ok": depth_ok,
                            "radial_ok": radial_ok,
                            "orientation_ok": orient_ok,
                            "head_above_tip": head_ok,
                        },
                    )
                )
            elif sg.predicate == "FASTENED":
                results.append(
                    SubgoalEvaluationResult(
                        subgoal=sg,
                        passed=bool(repaired and certified_drive),
                        evidence={
                            "joint_repaired": repaired,
                            "certified_drive": certified_drive,
                        },
                    )
                )
            elif sg.predicate == "ON":
                results.append(
                    SubgoalEvaluationResult(
                        subgoal=sg,
                        passed=driver_safe,
                        evidence={
                            "used_driver": used_driver,
                            "driver_safe": driver_safe,
                        },
                    )
                )

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    coverage = passed_count / total_count if total_count > 0 else 0.0

    return SubgoalCoverageEvaluation(
        domain=domain,
        variant=hidden_context.variant,
        total_subgoals=total_count,
        passed_subgoals=passed_count,
        coverage=coverage,
        results=tuple(results),
    )


def initial_snapshot_from_config(
    domain: str,
    variant: str,
    config_root: Path | None = None,
) -> TerminalStateSnapshot:
    """Reconstruct the initial benchmark scene snapshot deterministically from variant configuration.

    This is strictly benchmark evaluation infrastructure for post-hoc analysis.
    It MUST NEVER be called by baseline perception, planning, or refinement.
    """
    if config_root is None:
        config_root = Path(__file__).resolve().parents[3] / "configs"

    from mujoco_scenes.final_paper_variant_labels import resolve_variant_name

    norm = domain.strip().lower().replace("-", "_")
    internal = resolve_variant_name(norm, variant)

    if norm == "living_room":
        cfg_file = config_root / "living_room_variants.yaml"
        if not cfg_file.is_file():
            raise EvaluationContractError(f"missing living room config: {cfg_file}")
        doc = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        var_data = doc.get("variants", {}).get(internal, {})
        obj_locs = var_data.get("object_locations", {})
        regions = doc.get("regions", {})

        objects_dict: dict[str, Any] = {}
        all_objs = {
            "a2_drink_left",
            "a2_drink_right",
            "a2_snack_left",
            "a2_snack_right",
            "a2_remote_payload",
        }
        all_supports = {
            "a2_personal_left",
            "a2_personal_right",
            "a2_control_table",
            "a2_staging_surface",
        }
        for s in all_supports:
            objects_dict[s] = {"present": True}

        for obj in all_objs:
            region_name = obj_locs.get(obj)
            support_name = (
                regions.get(region_name, "a2_staging_surface")
                if region_name
                else "a2_staging_surface"
            )
            objects_dict[obj] = {
                "present": True,
                "support": support_name,
                "released": True,
                "stable": True,
                "inside_support_footprint": True,
                "support_contact": True,
                "floor_contact": False,
                "invalid_penetration": False,
            }
        return TerminalStateSnapshot(
            domain="living_room",
            objects=objects_dict,
            relations={},
            measurements={},
            predicted_infeasible=False,
        )

    if norm == "kitchen":
        cfg_file = config_root / "kitchen_feasibility_variants.yaml"
        if not cfg_file.is_file():
            raise EvaluationContractError(f"missing kitchen config: {cfg_file}")
        doc = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        var_data = doc.get("variants", {}).get(internal, {})
        ct_objs = var_data.get("countertop_objects", {})
        cont_contents = var_data.get("container_contents", {})

        objects_dict: dict[str, Any] = {"serving_area": {"present": True}}
        for spot, obj in ct_objs.items():
            objects_dict[obj] = {
                "present": True,
                "support": spot,
                "released": True,
                "stable": True,
                "inside_support_footprint": True,
                "support_contact": True,
                "floor_contact": False,
                "invalid_penetration": False,
            }
        for cont, items in cont_contents.items():
            objects_dict[cont] = {"present": True}
            for item in items:
                objects_dict[item] = {
                    "present": True,
                    "support": cont,
                    "released": True,
                    "stable": True,
                    "inside_support_footprint": True,
                    "support_contact": True,
                    "floor_contact": False,
                    "invalid_penetration": False,
                }
        return TerminalStateSnapshot(
            domain="kitchen",
            objects=objects_dict,
            relations={"contained_in": {}},
            measurements={},
            predicted_infeasible=False,
        )

    if norm == "workshop":
        cfg_file = config_root / "workshop_variants.yaml"
        if not cfg_file.is_file():
            raise EvaluationContractError(f"missing workshop config: {cfg_file}")
        doc = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        var_data = doc.get("variants", {}).get(internal, {})
        storage = var_data.get("storage_contents", {})

        objects_dict = {"MAIN_WORKBENCH_ZONE": {"present": True}}
        for reg, items in storage.items():
            objects_dict[reg] = {"present": True}
            for item in items:
                objects_dict[item] = {
                    "present": True,
                    "support": reg,
                    "released": True,
                    "stable": True,
                    "inside_support_footprint": True,
                    "support_contact": True,
                    "floor_contact": False,
                    "invalid_penetration": False,
                }
        return TerminalStateSnapshot(
            domain="workshop",
            objects=objects_dict,
            relations={"insertion": {}},
            measurements={"joint_repaired": False},
            predicted_infeasible=False,
        )

    raise EvaluationContractError(f"unsupported benchmark domain: {domain!r}")
