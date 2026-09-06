"""Object-to-problem interpretation using only independent baseline outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_json, atomic_write_text, sha256_text
from .contracts import (
    FixedSceneEvidence,
    GeneratedPDDLProblem,
    ObjectEstimate,
    ObjectEstimateStatus,
    PDDLValidationResult,
    ProblemSource,
    ViLaInObservation,
)
from .domains.registry import DomainDefinition
from .fm import FMCallRecord, FMCallType, FMRequest, RecordedFMClient
from .fact_selection import (
    FactSelectionError,
    compile_problem,
    parse_fact_selection,
)
from .pddl import validate_problem
from .prompts import (
    build_goal_reachability_prompt,
    build_goal_state_prompt,
    build_initial_consistency_prompt,
    build_initial_state_prompt,
    build_object_completeness_prompt,
    build_object_estimation_prompt,
    build_schema_regeneration_prompt,
)
from .symbolic_contract import (
    LiteralValidationError,
    SymbolicContractViolation,
    VariantActionContract,
    build_variant_action_contract,
    enumerate_grounded_facts,
    initial_state_diagnostics,
    missing_manipulable_types,
    relaxed_goal_diagnostics,
    validate_goal_fragment,
    validate_initial_fragment,
)


class InterpreterOutputError(ValueError):
    """Raised when a baseline model output cannot form a valid problem."""


@dataclass(frozen=True)
class InterpreterModels:
    object_estimator_model: str
    object_estimator_revision: str | None
    reasoning_model: str
    reasoning_model_revision: str | None


@dataclass(frozen=True)
class InterpretationResult:
    object_estimates: tuple[ObjectEstimate, ...]
    problem: GeneratedPDDLProblem
    validation: PDDLValidationResult
    calls: tuple[FMCallRecord, ...]


class ViLaInInterpreter:
    """Run object, initial-state, and goal calls and assemble one problem."""

    def __init__(
        self,
        *,
        object_client: RecordedFMClient,
        reasoning_client: RecordedFMClient,
        models: InterpreterModels,
        symbolic_contract: VariantActionContract | None = None,
        fixed_scene_evidence_provider: Any | None = None,
    ) -> None:
        self.object_client = object_client
        self.reasoning_client = reasoning_client
        self.models = models
        self.symbolic_contract = symbolic_contract
        self.fixed_scene_evidence_provider = fixed_scene_evidence_provider

    def interpret(
        self,
        *,
        task_instruction: str,
        domain: DomainDefinition,
        observations: Sequence[ViLaInObservation],
        observation_root: str | Path,
        output_root: str | Path,
    ) -> InterpretationResult:
        if not task_instruction.strip():
            raise ValueError("task_instruction must not be empty")
        destination = Path(output_root)
        symbolic_contract = self.symbolic_contract or build_variant_action_contract(
            domain, "unspecified"
        )
        if symbolic_contract.domain != domain.key:
            raise ValueError("symbolic contract domain differs from interpreter domain")
        symbolic_contract_path = atomic_write_json(
            destination / "interpreter" / "variant_action_contract.json",
            symbolic_contract.to_dict(),
        )

        fixed_scene_evidence: tuple[FixedSceneEvidence, ...] = ()
        if self.fixed_scene_evidence_provider is not None:
            fixed_scene_evidence = tuple(
                self.fixed_scene_evidence_provider(
                    observations=observations,
                    observation_root=observation_root,
                )
            )
        atomic_write_json(
            destination / "perception" / "fixed_scene_evidence.json",
            {"fixtures": [item.to_dict() for item in fixed_scene_evidence]},
        )

        object_prompt = build_object_estimation_prompt(
            task_instruction=task_instruction,
            domain=domain,
            observations=observations,
        )
        object_response, object_call = self.object_client.invoke(
            FMRequest(
                call_type=FMCallType.OBJECT_ESTIMATION,
                model=self.models.object_estimator_model,
                revision=self.models.object_estimator_revision,
                messages=object_prompt.messages(),
                image_artifacts=object_prompt.image_artifacts,
                response_format="json",
            ),
            destination / "perception" / "call",
        )
        estimates = normalize_object_estimates(
            object_response.raw_text,
            domain=domain,
            observations=observations,
            observation_root=observation_root,
        )
        estimates, excluded_fixed = _movable_estimates(estimates, symbolic_contract)
        atomic_write_json(
            destination / "perception" / "excluded_fixed_visual_entities.json",
            {
                "reason": "fixed locations use the neutral structural inventory",
                "objects": [item.to_dict() for item in excluded_fixed],
            },
        )
        calls: list[FMCallRecord] = [object_call]
        missing_types = missing_manipulable_types(
            {item.object_id: item.pddl_type for item in estimates}, symbolic_contract
        )
        completeness: dict[str, Any] = {
            "missing_capability_types_before_recheck": list(missing_types),
            "recheck_performed": False,
            "additional_object_count": 0,
        }
        if missing_types:
            recheck_prompt = build_object_completeness_prompt(
                task_instruction=task_instruction,
                domain=domain,
                observations=observations,
                existing_objects=estimates,
                missing_types=missing_types,
            )
            recheck_response, recheck_call = self.object_client.invoke(
                FMRequest(
                    call_type=FMCallType.OBJECT_ESTIMATION,
                    model=self.models.object_estimator_model,
                    revision=self.models.object_estimator_revision,
                    messages=recheck_prompt.messages(),
                    image_artifacts=recheck_prompt.image_artifacts,
                    response_format="json",
                    metadata={
                        "bounded_object_completeness_recheck": True,
                        "missing_capability_types": list(missing_types),
                    },
                ),
                destination / "perception" / "completeness_recheck_call",
            )
            additional = normalize_object_estimates(
                recheck_response.raw_text,
                domain=domain,
                observations=observations,
                observation_root=observation_root,
            )
            additional, _ = _movable_estimates(additional, symbolic_contract)
            invalid_types = sorted({item.pddl_type for item in additional}.difference(missing_types))
            if invalid_types:
                raise InterpreterOutputError(
                    "object completeness recheck returned unrequested types: "
                    + ", ".join(invalid_types)
                )
            estimates = _merge_completeness_estimates(
                estimates, additional, missing_types=missing_types
            )
            calls.append(recheck_call)
            completeness.update(
                recheck_performed=True,
                additional_object_count=len(additional),
                missing_capability_types_after_recheck=list(
                    missing_manipulable_types(
                        {item.object_id: item.pddl_type for item in estimates},
                        symbolic_contract,
                    )
                ),
            )
        atomic_write_json(
            destination / "perception" / "object_completeness.json", completeness
        )
        atomic_write_json(
            destination / "perception" / "object_estimates.json",
            {"objects": [estimate.to_dict() for estimate in estimates]},
        )

        object_inventory = dict(symbolic_contract.structural_inventory)
        for estimate in estimates:
            existing = object_inventory.get(estimate.object_id)
            if existing is not None and existing != estimate.pddl_type:
                raise InterpreterOutputError(
                    f"observed object {estimate.object_id!r} conflicts with "
                    "neutral structural inventory"
                )
            object_inventory[estimate.object_id] = estimate.pddl_type
        fact_candidates = enumerate_grounded_facts(
            object_inventory, symbolic_contract
        )
        fact_universe_path = atomic_write_json(
            destination / "interpreter" / "grounded_fact_universe.json",
            {
                "object_types": object_inventory,
                "candidate_count": len(fact_candidates),
                "facts": [fact.to_dict() for fact in fact_candidates],
            },
        )

        initial_prompt = build_initial_state_prompt(
            task_instruction=task_instruction,
            domain=domain,
            observations=observations,
            objects=estimates,
            fact_candidates=fact_candidates,
            symbolic_contract=symbolic_contract,
            fixed_scene_evidence=fixed_scene_evidence,
        )
        initial_response, initial_call = self._invoke_reasoning(
            FMCallType.INITIAL_STATE,
            initial_prompt.messages(),
            destination / "interpreter" / "initial_state_call",
            image_artifacts=initial_prompt.image_artifacts,
            metadata={
                "selection_field": "true_fact_ids",
                "fact_candidates": {
                    fact.fact_id: fact.literal for fact in fact_candidates
                },
            },
        )
        calls.append(initial_call)
        try:
            initial_facts = parse_fact_selection(
                initial_response.raw_text,
                field="true_fact_ids",
                candidates=fact_candidates,
            )
        except FactSelectionError as error:
            errors = [error.to_dict()]
            atomic_write_json(
                destination / "interpreter" / "initial_state_validation_errors.json",
                {"errors": errors},
            )
            repair = build_schema_regeneration_prompt(
                module="initial_state",
                previous_output=initial_response.raw_text,
                validation_errors=errors,
                valid_fact_ids=tuple(fact.fact_id for fact in fact_candidates),
            )
            initial_response, repair_call = self._invoke_reasoning(
                FMCallType.INITIAL_STATE,
                repair.messages(),
                destination / "interpreter" / "initial_state_regeneration_call",
                metadata={
                    "selection_field": "true_fact_ids",
                    "fact_candidates": {
                        fact.fact_id: fact.literal for fact in fact_candidates
                    },
                },
            )
            calls.append(repair_call)
            initial_call = repair_call
            try:
                initial_facts = parse_fact_selection(
                    initial_response.raw_text,
                    field="true_fact_ids",
                    candidates=fact_candidates,
                )
            except FactSelectionError as final_error:
                raise InterpreterOutputError(
                    "fact-selection regeneration remained invalid: " + str(final_error)
                ) from final_error

        initial_diagnostics = initial_state_diagnostics(initial_facts)
        if initial_diagnostics:
            pre_correction_initial = initial_facts
            atomic_write_json(
                destination / "interpreter" / "initial_state_diagnostics.json",
                {"diagnostics": list(initial_diagnostics), "correction_performed": True},
            )
            correction_prompt = build_initial_consistency_prompt(
                task_instruction=task_instruction,
                previous_fact_ids=tuple(fact.fact_id for fact in initial_facts),
                diagnostics=initial_diagnostics,
                fact_candidates=fact_candidates,
                observations=observations,
                objects=estimates,
                fixed_scene_evidence=fixed_scene_evidence,
            )
            corrected_response, corrected_call = self._invoke_reasoning(
                FMCallType.INITIAL_STATE,
                correction_prompt.messages(),
                destination / "interpreter" / "initial_state_consistency_call",
                image_artifacts=correction_prompt.image_artifacts,
                metadata={
                    "selection_field": "true_fact_ids",
                    "bounded_consistency_correction": True,
                    "fact_candidates": {
                        fact.fact_id: fact.literal for fact in fact_candidates
                    },
                    "diagnostics": list(initial_diagnostics),
                },
            )
            calls.append(corrected_call)
            try:
                corrected_initial = parse_fact_selection(
                    corrected_response.raw_text,
                    field="true_fact_ids",
                    candidates=fact_candidates,
                )
            except FactSelectionError as error:
                raise InterpreterOutputError(
                    "initial-state consistency correction was invalid: " + str(error)
                ) from error
            remaining_initial = initial_state_diagnostics(corrected_initial)
            correction_fallback = False
            if remaining_initial:
                # The bounded FM repair may make a valid selection worse. Fall
                # back to the original evidence and conservatively omit every
                # conflicting fact; never guess which mutually exclusive fact
                # was true.
                conflicting_ids = {
                    str(fact_id)
                    for diagnostic in initial_diagnostics
                    for fact_id in diagnostic.get("fact_ids", ())
                }
                corrected_initial = tuple(
                    fact
                    for fact in pre_correction_initial
                    if fact.fact_id not in conflicting_ids
                )
                remaining_initial = initial_state_diagnostics(corrected_initial)
                correction_fallback = True
            atomic_write_json(
                destination / "interpreter" / "initial_state_consistency_result.json",
                {
                    "selected_true_fact_ids": [fact.fact_id for fact in corrected_initial],
                    "remaining_diagnostics": list(remaining_initial),
                    "conservative_conflict_omission_fallback": correction_fallback,
                },
            )
            if remaining_initial:
                raise InterpreterOutputError(
                    "initial-state consistency correction remained contradictory"
                )
            initial_facts = corrected_initial
            initial_call = corrected_call
        else:
            atomic_write_json(
                destination / "interpreter" / "initial_state_diagnostics.json",
                {"diagnostics": [], "correction_performed": False},
            )

        selected_initial_path = atomic_write_json(
            destination / "interpreter" / "selected_initial_facts.json",
            {
                "true_fact_ids": [fact.fact_id for fact in initial_facts],
                "facts": [fact.to_dict() for fact in initial_facts],
            },
        )

        goal_prompt = build_goal_state_prompt(
            task_instruction=task_instruction,
            domain=domain,
            objects=estimates,
            initial_state_fragment="\n".join(fact.literal for fact in initial_facts),
            fact_candidates=fact_candidates,
            declared_object_types=object_inventory,
            symbolic_contract=symbolic_contract,
        )
        goal_response, goal_call = self._invoke_reasoning(
            FMCallType.GOAL_STATE,
            goal_prompt.messages(),
            destination / "interpreter" / "goal_state_call",
            metadata={
                "selection_field": "goal_fact_ids",
                "fact_candidates": {
                    fact.fact_id: fact.literal
                    for fact in fact_candidates
                    if fact.goal_eligible
                },
            },
        )
        calls.append(goal_call)
        goal_candidates = tuple(
            fact for fact in fact_candidates if fact.goal_eligible
        )
        try:
            goal_facts = parse_fact_selection(
                goal_response.raw_text,
                field="goal_fact_ids",
                candidates=goal_candidates,
            )
            if not goal_facts:
                raise FactSelectionError(
                    "EMPTY_GOAL_SELECTION", "goal_fact_ids",
                    "goal selection must contain at least one fact ID",
                )
        except FactSelectionError as error:
            errors = [error.to_dict()]
            atomic_write_json(
                destination / "interpreter" / "goal_state_validation_errors.json",
                {"errors": errors},
            )
            repair = build_schema_regeneration_prompt(
                module="goal_state",
                previous_output=goal_response.raw_text,
                validation_errors=errors,
                valid_fact_ids=tuple(fact.fact_id for fact in goal_candidates),
            )
            goal_response, repair_call = self._invoke_reasoning(
                FMCallType.GOAL_STATE,
                repair.messages(),
                destination / "interpreter" / "goal_state_regeneration_call",
                metadata={
                    "selection_field": "goal_fact_ids",
                    "fact_candidates": {
                        fact.fact_id: fact.literal for fact in goal_candidates
                    },
                },
            )
            calls.append(repair_call)
            goal_call = repair_call
            try:
                goal_facts = parse_fact_selection(
                    goal_response.raw_text,
                    field="goal_fact_ids",
                    candidates=goal_candidates,
                )
                if not goal_facts:
                    raise FactSelectionError(
                        "EMPTY_GOAL_SELECTION", "goal_fact_ids",
                        "goal selection must contain at least one fact ID",
                    )
            except FactSelectionError as final_error:
                raise InterpreterOutputError(
                    "fact-selection regeneration remained invalid: " + str(final_error)
                ) from final_error

        reachability = relaxed_goal_diagnostics(
            initial_facts, goal_facts, object_inventory, symbolic_contract
        )
        if reachability:
            atomic_write_json(
                destination / "interpreter" / "goal_reachability_diagnostics.json",
                {"diagnostics": list(reachability), "correction_performed": True},
            )
            correction_prompt = build_goal_reachability_prompt(
                task_instruction=task_instruction,
                previous_goal_ids=tuple(fact.fact_id for fact in goal_facts),
                diagnostics=reachability,
                fact_candidates=goal_candidates,
            )
            corrected_response, corrected_call = self._invoke_reasoning(
                FMCallType.GOAL_STATE,
                correction_prompt.messages(),
                destination / "interpreter" / "goal_reachability_correction_call",
                metadata={
                    "selection_field": "goal_fact_ids",
                    "bounded_reachability_correction": True,
                    "fact_candidates": {
                        fact.fact_id: fact.literal for fact in goal_candidates
                    },
                    "diagnostics": list(reachability),
                },
            )
            calls.append(corrected_call)
            try:
                corrected_goals = parse_fact_selection(
                    corrected_response.raw_text,
                    field="goal_fact_ids",
                    candidates=goal_candidates,
                )
                if not corrected_goals:
                    corrected_goals = goal_facts
            except FactSelectionError as error:
                raise InterpreterOutputError(
                    "goal reachability correction was invalid: " + str(error)
                ) from error
            goal_facts = corrected_goals
            remaining = relaxed_goal_diagnostics(
                initial_facts, goal_facts, object_inventory, symbolic_contract
            )
            atomic_write_json(
                destination / "interpreter" / "goal_reachability_result.json",
                {
                    "selected_goal_fact_ids": [fact.fact_id for fact in goal_facts],
                    "remaining_diagnostics": list(remaining),
                },
            )
        else:
            atomic_write_json(
                destination / "interpreter" / "goal_reachability_diagnostics.json",
                {"diagnostics": [], "correction_performed": False},
            )

        problem_text, objects_fragment, init_fragment, goal_fragment = compile_problem(
            domain=domain,
            problem_name=f"vilain-{domain.key}-attempt-00",
            object_types=object_inventory,
            initial_facts=initial_facts,
            goal_facts=goal_facts,
        )
        selected_goal_path = atomic_write_json(
            destination / "interpreter" / "selected_goal_facts.json",
            {
                "goal_fact_ids": [fact.fact_id for fact in goal_facts],
                "facts": [fact.to_dict() for fact in goal_facts],
            },
        )
        initial_fragment_path = atomic_write_text(
            destination / "interpreter" / "initial_state.pddlfrag",
            objects_fragment + "\n" + init_fragment + "\n",
        )
        goal_fragment_path = atomic_write_text(
            destination / "interpreter" / "goal_state.pddlfrag",
            goal_fragment + "\n",
        )

        problem_path = atomic_write_text(
            destination / "interpreter" / "problem_initial.pddl", problem_text
        )
        raw_fragments_path = atomic_write_json(
            destination / "interpreter" / "generation_artifacts.json",
            {
                "object_response_artifact": object_call.raw_response_artifact,
                "initial_response_artifact": initial_call.raw_response_artifact,
                "goal_response_artifact": goal_call.raw_response_artifact,
                "initial_fragment_artifact": str(initial_fragment_path),
                "goal_fragment_artifact": str(goal_fragment_path),
                "problem_artifact": str(problem_path),
                "variant_action_contract_artifact": str(symbolic_contract_path),
                "grounded_fact_universe_artifact": str(fact_universe_path),
                "selected_initial_facts_artifact": str(selected_initial_path),
                "selected_goal_facts_artifact": str(selected_goal_path),
            },
        )

        validation = validate_problem(
            problem_text,
            domain,
            expected_domain_sha256=domain.sha256,
        )
        if not validation.valid:
            raise InterpreterOutputError(
                "generated PDDL problem is invalid: " + "; ".join(validation.diagnostics)
            )
        problem = GeneratedPDDLProblem(
            attempt_index=0,
            source=ProblemSource.INITIAL,
            domain_name=domain.name,
            domain_sha256=domain.sha256,
            problem_text=problem_text,
            declared_objects=tuple(sorted(object_inventory)),
            initial_atoms=_atom_strings(init_fragment),
            goal_atoms=_atom_strings(goal_fragment),
            raw_response_artifact=str(raw_fragments_path),
            problem_sha256=sha256_text(problem_text),
        )
        return InterpretationResult(
            object_estimates=estimates,
            problem=problem,
            validation=validation,
            calls=tuple(calls),
        )

    def _invoke_reasoning(
        self,
        call_type: FMCallType,
        messages: Sequence[Mapping[str, str]],
        output_root: Path,
        image_artifacts: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Any, FMCallRecord]:
        return self.reasoning_client.invoke(
            FMRequest(
                call_type=call_type,
                model=self.models.reasoning_model,
                revision=self.models.reasoning_model_revision,
                messages=messages,
                image_artifacts=tuple(image_artifacts),
                response_format="json",
                metadata=dict(metadata or {}),
            ),
            output_root,
        )


def normalize_object_estimates(
    raw_text: str,
    *,
    domain: DomainDefinition,
    observations: Sequence[ViLaInObservation],
    observation_root: str | Path,
) -> tuple[ObjectEstimate, ...]:
    try:
        loaded = json.loads(_strip_outer_fence(raw_text, language="json"))
    except json.JSONDecodeError as error:
        truncated = _looks_truncated_json(raw_text)
        label = "truncated JSON" if truncated else "valid JSON"
        raise InterpreterOutputError(f"object response is not {label}: {error}") from error
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("objects"), list):
        raise InterpreterOutputError("object response must contain an objects array")

    frame_index = _frame_index(observations)
    stage_order = {
        observation.stage_id: index for index, observation in enumerate(observations)
    }
    normalized_rows: list[dict[str, Any]] = []
    for row in loaded["objects"]:
        if not isinstance(row, Mapping):
            raise InterpreterOutputError("every object estimate must be an object")
        # Some OpenAI-compatible vision servers name this otherwise-identical
        # visual field ``visual_label``. Normalize that wire-format alias at
        # the interpreter boundary; the baseline contract remains ``label``.
        label_row = row
        if "label" not in row and "visual_label" in row:
            label_row = {**row, "label": row["visual_label"]}
        label = _required_text(label_row, "label")
        pddl_type = _required_text(row, "pddl_type").lower()
        if pddl_type not in domain.type_hierarchy:
            raise InterpreterOutputError(f"unknown PDDL type {pddl_type!r}")
        detections = _normalize_detections(
            row.get("detections"),
            frame_index,
            stage_order,
            Path(observation_root),
        )
        sort_key = (
            _normalize_identifier(label),
            min(stage_order[item["stage_id"]] for item in detections),
            min((item["xyxy"][0] + item["xyxy"][2]) / 2 for item in detections),
            min((item["xyxy"][1] + item["xyxy"][3]) / 2 for item in detections),
        )
        normalized_rows.append(
            {
                "label": label.strip(),
                "pddl_type": pddl_type,
                "description": str(row.get("description", "")).strip(),
                "detections": detections,
                "status": _status(row.get("status", "OBSERVED")),
                "sort_key": sort_key,
            }
        )

    normalized_rows.sort(key=lambda item: item["sort_key"])
    label_counts: dict[str, int] = {}
    estimates: list[ObjectEstimate] = []
    for row in normalized_rows:
        base_id = _normalize_identifier(row["label"])
        label_counts[base_id] = label_counts.get(base_id, 0) + 1
        object_id = f"{base_id}_{label_counts[base_id]}"
        localization_detections = sorted(
            row["detections"],
            key=lambda item: (
                "overhead" not in item["camera_id"].lower(),
                -item["confidence"],
                stage_order[item["stage_id"]],
                item["camera_id"],
            ),
        )
        centroids = ()
        for detection in localization_detections:
            centroid = _detection_centroid(
                detection,
                frame_index=frame_index,
                observation_root=Path(observation_root),
            )
            if centroid is not None and all(abs(value) <= 10.0 for value in centroid):
                centroids = (centroid,)
                break
        centroid, covariance = _combine_centroids(centroids)
        estimates.append(
            ObjectEstimate(
                object_id=object_id,
                label=row["label"],
                pddl_type=row["pddl_type"],
                description=row["description"],
                detections=tuple(row["detections"]),
                estimated_centroid_m=centroid,
                centroid_covariance=covariance,
                observation_stage_ids=tuple(
                    sorted(
                        {item["stage_id"] for item in row["detections"]},
                        key=stage_order.__getitem__,
                    )
                ),
                status=row["status"],
            )
        )
    return tuple(estimates)


def _merge_completeness_estimates(
    initial: Sequence[ObjectEstimate],
    additional: Sequence[ObjectEstimate],
    *,
    missing_types: Sequence[str],
) -> tuple[ObjectEstimate, ...]:
    """Merge an absent-type recheck while preserving unique deterministic IDs."""
    result = list(initial)
    used_ids = {item.object_id for item in result}
    for item in additional:
        if item.pddl_type not in missing_types:
            raise InterpreterOutputError("completeness merge received a non-missing type")
        same_label = next(
            (
                index
                for index, prior in enumerate(result)
                if _normalize_identifier(prior.label) == _normalize_identifier(item.label)
                and prior.pddl_type not in missing_types
            ),
            None,
        )
        if same_label is not None:
            prior = result[same_label]
            result[same_label] = ObjectEstimate(
                object_id=prior.object_id,
                label=prior.label,
                pddl_type=item.pddl_type,
                description=item.description or prior.description,
                detections=item.detections,
                estimated_centroid_m=item.estimated_centroid_m,
                centroid_covariance=item.centroid_covariance,
                observation_stage_ids=item.observation_stage_ids,
                status=item.status,
            )
            continue
        object_id = item.object_id
        if object_id in used_ids:
            base, _, suffix = object_id.rpartition("_")
            index = int(suffix) if suffix.isdigit() else 1
            while f"{base}_{index}" in used_ids:
                index += 1
            object_id = f"{base}_{index}"
        result.append(
            ObjectEstimate(
                object_id=object_id,
                label=item.label,
                pddl_type=item.pddl_type,
                description=item.description,
                detections=item.detections,
                estimated_centroid_m=item.estimated_centroid_m,
                centroid_covariance=item.centroid_covariance,
                observation_stage_ids=item.observation_stage_ids,
                status=item.status,
            )
        )
        used_ids.add(object_id)
    return tuple(sorted(result, key=lambda item: item.object_id))


def _movable_estimates(
    estimates: Sequence[ObjectEstimate], contract: VariantActionContract
) -> tuple[tuple[ObjectEstimate, ...], tuple[ObjectEstimate, ...]]:
    def subtype(actual: str, expected: str) -> bool:
        current: str | None = actual
        visited: set[str] = set()
        while current is not None and current not in visited:
            if current == expected:
                return True
            visited.add(current)
            current = contract.type_hierarchy.get(current)
        return False

    movable = tuple(item for item in estimates if subtype(item.pddl_type, "movable"))
    fixed = tuple(item for item in estimates if not subtype(item.pddl_type, "movable"))
    return movable, fixed


def _normalize_detections(
    value: Any,
    frame_index: Mapping[tuple[str, str], Any],
    stage_order: Mapping[str, int],
    observation_root: Path,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise InterpreterOutputError("object detections must be a non-empty array")
    detections: list[dict[str, Any]] = []
    for detection in value:
        if not isinstance(detection, Mapping):
            raise InterpreterOutputError("each detection must be an object")
        if "stage_id" in detection:
            stage_id = _required_text(detection, "stage_id")
        elif len(stage_order) == 1:
            stage_id = next(iter(stage_order))
        else:
            raise InterpreterOutputError("stage_id must be a non-empty string")
        camera_id = _required_text(detection, "camera_id")
        if (stage_id, camera_id) not in frame_index:
            raise InterpreterOutputError(
                f"detection references unknown frame {stage_id!r}/{camera_id!r}"
            )
        frame = frame_index[(stage_id, camera_id)]
        depth = np.load(observation_root / frame.depth_path, allow_pickle=False)
        if depth.ndim != 2:
            raise InterpreterOutputError("observation depth must be a 2-D array")
        height, width = depth.shape
        normalized_box = detection.get("bbox_1000")
        box = detection.get("xyxy")
        if normalized_box is not None:
            if not isinstance(normalized_box, list) or len(normalized_box) != 4:
                raise InterpreterOutputError("detection bbox_1000 must contain four numbers")
            try:
                coordinates = tuple(float(item) for item in normalized_box)
            except (TypeError, ValueError) as error:
                raise InterpreterOutputError("bbox_1000 coordinates must be numeric") from error
            if not all(math.isfinite(item) for item in coordinates) or not (
                0.0 <= coordinates[0] < coordinates[2] <= 1000.0
                and 0.0 <= coordinates[1] < coordinates[3] <= 1000.0
            ):
                raise InterpreterOutputError(
                    "detection bbox_1000 must satisfy normalized bounds"
                )
            box = [
                coordinates[0] / 1000.0 * width,
                coordinates[1] / 1000.0 * height,
                coordinates[2] / 1000.0 * width,
                coordinates[3] / 1000.0 * height,
            ]
        if not isinstance(box, list) or len(box) != 4:
            raise InterpreterOutputError(
                "detection must contain bbox_1000 or four-number xyxy"
            )
        try:
            xyxy = tuple(float(item) for item in box)
            confidence = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError) as error:
            raise InterpreterOutputError("detection coordinates must be numeric") from error
        if not all(math.isfinite(item) for item in xyxy) or not (
            0.0 <= xyxy[0] < xyxy[2] <= width
            and 0.0 <= xyxy[1] < xyxy[3] <= height
        ):
            raise InterpreterOutputError(
                "detection xyxy must lie inside the corresponding image"
            )
        if not 0.0 <= confidence <= 1.0:
            raise InterpreterOutputError("detection confidence must be between zero and one")
        detections.append(
            {
                "stage_id": stage_id,
                "camera_id": camera_id,
                "xyxy": xyxy,
                "confidence": confidence,
            }
        )
    detections.sort(
        key=lambda item: (
            stage_order[item["stage_id"]],
            item["camera_id"],
            item["xyxy"],
        )
    )
    return tuple(detections)


def _detection_centroid(
    detection: Mapping[str, Any],
    *,
    frame_index: Mapping[tuple[str, str], Any],
    observation_root: Path,
) -> tuple[float, float, float] | None:
    frame = frame_index[(detection["stage_id"], detection["camera_id"])]
    depth = np.load(observation_root / frame.depth_path, allow_pickle=False)
    calibration = json.loads(
        (observation_root / frame.calibration_path).read_text(encoding="utf-8")
    )
    x1, y1, x2, y2 = detection["xyxy"]
    height, width = depth.shape
    left = max(0, min(width, int(math.floor(x1))))
    right = max(0, min(width, int(math.ceil(x2))))
    top = max(0, min(height, int(math.floor(y1))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    if left >= right or top >= bottom:
        return None
    crop = np.asarray(depth[top:bottom, left:right], dtype=float)
    usable = crop[np.isfinite(crop) & (crop > 0)]
    if usable.size == 0:
        return None
    z = float(np.median(usable))
    intrinsics = np.asarray(calibration["intrinsics"], dtype=float)
    extrinsics = np.asarray(calibration["extrinsics"], dtype=float)
    if intrinsics.shape != (3, 3) or extrinsics.shape != (4, 4):
        raise InterpreterOutputError("camera calibration has invalid matrix dimensions")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    if fx <= 0 or fy <= 0:
        raise InterpreterOutputError("camera focal lengths must be positive")
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    camera_point = np.array(
        [
            (u - float(intrinsics[0, 2])) * z / fx,
            (v - float(intrinsics[1, 2])) * z / fy,
            z,
            1.0,
        ]
    )
    world_point = extrinsics @ camera_point
    if not np.all(np.isfinite(world_point[:3])):
        return None
    return tuple(float(item) for item in world_point[:3])


def _combine_centroids(
    centroids: Sequence[tuple[float, float, float]],
) -> tuple[
    tuple[float, float, float] | None,
    tuple[tuple[float, float, float], ...] | None,
]:
    if not centroids:
        return None, None
    values = np.asarray(centroids, dtype=float)
    mean = tuple(float(item) for item in values.mean(axis=0))
    if len(centroids) == 1:
        return mean, None
    covariance_array = np.cov(values, rowvar=False)
    covariance = tuple(
        tuple(float(item) for item in row) for row in covariance_array
    )
    return mean, covariance


def _frame_index(
    observations: Sequence[ViLaInObservation],
) -> dict[tuple[str, str], Any]:
    if not observations:
        raise InterpreterOutputError("at least one observation is required")
    result: dict[tuple[str, str], Any] = {}
    for observation in observations:
        for frame in observation.camera_frames:
            key = (observation.stage_id, frame.camera_id)
            if key in result:
                raise InterpreterOutputError(f"duplicate observation frame {key!r}")
            result[key] = frame
    return result


def _extract_initial_fragments(raw_text: str) -> tuple[str, str]:
    forms = _top_level_forms(_strip_outer_fence(raw_text))
    if len(forms) != 2 or [_form_head(form) for form in forms] != [":objects", ":init"]:
        raise InterpreterOutputError(
            "initial-state response must contain exactly :objects then :init"
        )
    return forms[0], forms[1]


def _validate_initial_output(
    raw_text: str,
    *,
    estimates: Sequence[ObjectEstimate],
    domain: DomainDefinition,
    symbolic_contract: VariantActionContract,
) -> tuple[str, str, dict[str, str]]:
    objects_fragment, init_fragment = _extract_initial_fragments(raw_text)
    declared_object_types = _declared_object_types(objects_fragment)
    _validate_object_declarations(estimates, declared_object_types, domain)
    for object_id, object_type in declared_object_types.items():
        if object_type != "object" and object_type not in domain.type_hierarchy:
            raise SymbolicContractViolation(
                (
                    LiteralValidationError(
                        "INVALID_OBJECT_TYPE",
                        "objects",
                        object_id,
                        expected="declared domain type",
                        actual=object_type,
                        object_id=object_id,
                    ),
                )
            )
    validate_initial_fragment(
        init_fragment, declared_object_types, symbolic_contract
    )
    return objects_fragment, init_fragment, declared_object_types


def _best_effort_declared_types(raw_text: str) -> dict[str, str]:
    try:
        objects_fragment, _ = _extract_initial_fragments(raw_text)
        return _declared_object_types(objects_fragment)
    except InterpreterOutputError:
        return {}


def _structured_errors(
    error: InterpreterOutputError | SymbolicContractViolation,
    section: str,
) -> list[dict[str, Any]]:
    if isinstance(error, SymbolicContractViolation):
        return [item.to_dict() for item in error.errors]
    return [
        LiteralValidationError(
            "INVALID_MODULE_OUTPUT",
            section,
            "",
            detail=str(error),
        ).to_dict()
    ]


def _extract_goal_fragment(raw_text: str) -> str:
    forms = _top_level_forms(_strip_outer_fence(raw_text))
    if len(forms) != 1 or _form_head(forms[0]) != ":goal":
        raise InterpreterOutputError("goal response must contain exactly one :goal form")
    return forms[0]


def _top_level_forms(text: str) -> tuple[str, ...]:
    cleaned = re.sub(r";[^\n]*", "", text).strip()
    forms: list[str] = []
    depth = 0
    start: int | None = None
    outside: list[str] = []
    for index, character in enumerate(cleaned):
        if character == "(":
            if depth == 0:
                start = index
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise InterpreterOutputError("response has an unexpected closing parenthesis")
            if depth == 0 and start is not None:
                forms.append(cleaned[start:index + 1].strip())
                start = None
        elif depth == 0 and not character.isspace():
            outside.append(character)
    if depth != 0:
        raise InterpreterOutputError("response has unbalanced parentheses")
    if outside:
        raise InterpreterOutputError("response contains text outside PDDL forms")
    return tuple(forms)


def _strip_outer_fence(text: str, *, language: str | None = None) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    opening = lines[0].strip().lower()
    allowed = {"```"}
    if language:
        allowed.add(f"```{language.lower()}")
    else:
        allowed.update({"```pddl", "```lisp"})
    if opening not in allowed:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _looks_truncated_json(text: str) -> bool:
    candidate = text.strip()
    if candidate.startswith("```json"):
        candidate = "\n".join(candidate.splitlines()[1:])
    stack: list[str] = []
    quoted = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in candidate:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or stack.pop() != pairs[character]:
                return False
    return quoted or bool(stack)


def _form_head(form: str) -> str:
    match = re.match(r"\(\s*([^\s()]+)", form)
    return match.group(1).lower() if match else ""


def _assemble_problem(
    *,
    domain_name: str,
    problem_name: str,
    objects_fragment: str,
    init_fragment: str,
    goal_fragment: str,
) -> str:
    return (
        f"(define (problem {problem_name})\n"
        f"  (:domain {domain_name})\n"
        f"  {_indent_form(objects_fragment)}\n"
        f"  {_indent_form(init_fragment)}\n"
        f"  {_indent_form(goal_fragment)}\n"
        ")\n"
    )


def _indent_form(form: str) -> str:
    lines = form.strip().splitlines()
    return ("\n  ").join(line.rstrip() for line in lines)


def _declared_object_types(objects_fragment: str) -> dict[str, str]:
    inner = re.sub(r"^\s*\(\s*:objects\b|\)\s*$", "", objects_fragment, flags=re.I)
    tokens = re.findall(r"[^\s()]+", inner)
    result: dict[str, str] = {}
    pending: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == "-":
            if not pending or index + 1 >= len(tokens):
                raise InterpreterOutputError("malformed typed object declarations")
            object_type = tokens[index + 1].lower()
            for name in pending:
                if name in result:
                    raise InterpreterOutputError("object declarations contain duplicate names")
                result[name] = object_type
            pending = []
            index += 2
        else:
            pending.append(tokens[index].lower())
            index += 1
    for name in pending:
        if name in result:
            raise InterpreterOutputError("object declarations contain duplicate names")
        result[name] = "object"
    return result


def _validate_object_declarations(
    estimates: Sequence[ObjectEstimate],
    declared_types: Mapping[str, str],
    domain: DomainDefinition,
) -> None:
    estimate_types = {estimate.object_id: estimate.pddl_type for estimate in estimates}
    for object_id, estimate_type in estimate_types.items():
        declared_type = declared_types.get(object_id)
        if declared_type is None:
            raise InterpreterOutputError(
                f"generated object declarations omit observed object {object_id!r}"
            )
        if declared_type != estimate_type:
            raise InterpreterOutputError(
                f"observed object {object_id!r} changed type from "
                f"{estimate_type!r} to {declared_type!r}"
            )
    for object_id, object_type in declared_types.items():
        if _type_descends_from(object_type, "movable", domain.type_hierarchy):
            if object_id not in estimate_types:
                raise InterpreterOutputError(
                    f"generated problem declares unobserved movable object {object_id!r}"
                )


def _type_descends_from(
    actual: str,
    ancestor: str,
    hierarchy: Mapping[str, str | None],
) -> bool:
    current: str | None = actual
    visited: set[str] = set()
    while current is not None and current not in visited:
        if current == ancestor:
            return True
        visited.add(current)
        current = hierarchy.get(current)
    return False


def _atom_strings(fragment: str) -> tuple[str, ...]:
    atoms = []
    for match in re.finditer(r"\([^()]+\)", fragment):
        atom = " ".join(match.group(0).lower().split())
        if _form_head(atom) not in {":init", ":goal", "and", "or", "not"}:
            atoms.append(atom)
    return tuple(atoms)


def _normalize_identifier(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    if not normalized:
        raise InterpreterOutputError("object label cannot form a stable identifier")
    if normalized[0].isdigit():
        normalized = "object_" + normalized
    return normalized


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InterpreterOutputError(f"{key} must be a non-empty string")
    return value


def _status(value: Any) -> ObjectEstimateStatus:
    try:
        return ObjectEstimateStatus(str(value).upper())
    except ValueError as error:
        raise InterpreterOutputError(f"unknown object status {value!r}") from error
