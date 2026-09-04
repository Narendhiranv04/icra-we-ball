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
    GeneratedPDDLProblem,
    ObjectEstimate,
    ObjectEstimateStatus,
    PDDLValidationResult,
    ProblemSource,
    ViLaInObservation,
)
from .domains.registry import DomainDefinition
from .fm import FMCallRecord, FMCallType, FMRequest, RecordedFMClient
from .pddl import validate_problem
from .prompts import (
    build_goal_state_prompt,
    build_initial_state_prompt,
    build_object_estimation_prompt,
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
    ) -> None:
        self.object_client = object_client
        self.reasoning_client = reasoning_client
        self.models = models

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
        atomic_write_json(
            destination / "perception" / "object_estimates.json",
            {"objects": [estimate.to_dict() for estimate in estimates]},
        )

        initial_prompt = build_initial_state_prompt(
            task_instruction=task_instruction,
            domain=domain,
            objects=estimates,
        )
        initial_response, initial_call = self.reasoning_client.invoke(
            FMRequest(
                call_type=FMCallType.INITIAL_STATE,
                model=self.models.reasoning_model,
                revision=self.models.reasoning_model_revision,
                messages=initial_prompt.messages(),
                response_format="pddl",
            ),
            destination / "interpreter" / "initial_state_call",
        )
        objects_fragment, init_fragment = _extract_initial_fragments(
            initial_response.raw_text
        )
        declared_object_types = _declared_object_types(objects_fragment)
        _validate_object_declarations(estimates, declared_object_types, domain)
        initial_fragment_path = atomic_write_text(
            destination / "interpreter" / "initial_state.pddlfrag",
            objects_fragment + "\n" + init_fragment + "\n",
        )

        goal_prompt = build_goal_state_prompt(
            task_instruction=task_instruction,
            domain=domain,
            objects=estimates,
            initial_state_fragment=objects_fragment + "\n" + init_fragment,
        )
        goal_response, goal_call = self.reasoning_client.invoke(
            FMRequest(
                call_type=FMCallType.GOAL_STATE,
                model=self.models.reasoning_model,
                revision=self.models.reasoning_model_revision,
                messages=goal_prompt.messages(),
                response_format="pddl",
            ),
            destination / "interpreter" / "goal_state_call",
        )
        goal_fragment = _extract_goal_fragment(goal_response.raw_text)
        goal_fragment_path = atomic_write_text(
            destination / "interpreter" / "goal_state.pddlfrag",
            goal_fragment + "\n",
        )

        problem_text = _assemble_problem(
            domain_name=domain.name,
            problem_name=f"vilain-{domain.key}-attempt-00",
            objects_fragment=objects_fragment,
            init_fragment=init_fragment,
            goal_fragment=goal_fragment,
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
            declared_objects=tuple(declared_object_types),
            initial_atoms=_atom_strings(init_fragment),
            goal_atoms=_atom_strings(goal_fragment),
            raw_response_artifact=str(raw_fragments_path),
            problem_sha256=sha256_text(problem_text),
        )
        return InterpretationResult(
            object_estimates=estimates,
            problem=problem,
            validation=validation,
            calls=(object_call, initial_call, goal_call),
        )


def normalize_object_estimates(
    raw_text: str,
    *,
    domain: DomainDefinition,
    observations: Sequence[ViLaInObservation],
    observation_root: str | Path,
) -> tuple[ObjectEstimate, ...]:
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise InterpreterOutputError(f"object response is not valid JSON: {error}") from error
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
        label = _required_text(row, "label")
        pddl_type = _required_text(row, "pddl_type").lower()
        if pddl_type not in domain.type_hierarchy:
            raise InterpreterOutputError(f"unknown PDDL type {pddl_type!r}")
        detections = _normalize_detections(row.get("detections"), frame_index, stage_order)
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
        centroids = tuple(
            centroid
            for detection in row["detections"]
            if (
                centroid := _detection_centroid(
                    detection,
                    frame_index=frame_index,
                    observation_root=Path(observation_root),
                )
            )
            is not None
        )
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


def _normalize_detections(
    value: Any,
    frame_index: Mapping[tuple[str, str], Any],
    stage_order: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise InterpreterOutputError("object detections must be a non-empty array")
    detections: list[dict[str, Any]] = []
    for detection in value:
        if not isinstance(detection, Mapping):
            raise InterpreterOutputError("each detection must be an object")
        stage_id = _required_text(detection, "stage_id")
        camera_id = _required_text(detection, "camera_id")
        if (stage_id, camera_id) not in frame_index:
            raise InterpreterOutputError(
                f"detection references unknown frame {stage_id!r}/{camera_id!r}"
            )
        box = detection.get("xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise InterpreterOutputError("detection xyxy must contain four numbers")
        try:
            xyxy = tuple(float(item) for item in box)
            confidence = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError) as error:
            raise InterpreterOutputError("detection coordinates must be numeric") from error
        if not all(math.isfinite(item) for item in xyxy) or not (
            xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3]
        ):
            raise InterpreterOutputError("detection xyxy must define a finite positive box")
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
    forms = _top_level_forms(raw_text)
    if len(forms) != 2 or [_form_head(form) for form in forms] != [":objects", ":init"]:
        raise InterpreterOutputError(
            "initial-state response must contain exactly :objects then :init"
        )
    return forms[0], forms[1]


def _extract_goal_fragment(raw_text: str) -> str:
    forms = _top_level_forms(raw_text)
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
