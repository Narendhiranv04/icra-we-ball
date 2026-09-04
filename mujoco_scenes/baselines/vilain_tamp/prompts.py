"""Paper-oriented prompts for the independent ViLaIn-TAMP model calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import yaml

from .contracts import ObjectEstimate, ViLaInObservation
from .domains.registry import DomainDefinition
from .observations import prompt_observation_payload


@dataclass(frozen=True)
class PromptBundle:
    system_text: str
    user_text: str
    image_artifacts: tuple[str, ...] = ()

    def messages(self) -> tuple[Mapping[str, str], ...]:
        return (
            {"role": "system", "content": self.system_text},
            {"role": "user", "content": self.user_text},
        )


def build_object_estimation_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    observations: Sequence[ViLaInObservation],
) -> PromptBundle:
    observation_payload = prompt_observation_payload(observations)
    image_artifacts = tuple(
        frame.rgb_path
        for observation in observations
        for frame in observation.camera_frames
    )
    return PromptBundle(
        system_text=(
            "You are the object-estimation component of an independent "
            "ViLaIn-TAMP pipeline. Use only the supplied RGB images and fixed "
            "domain descriptions. Do not output an action sequence."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Fixed domain knowledge:\n{_domain_knowledge_text(domain)}\n\n"
            "Observation ordering:\n"
            f"{_json(observation_payload)}\n\n"
            "Return one JSON object with an `objects` array. Each object must "
            "contain a concise visual label, PDDL type, description, and one "
            "or more detections with camera ID, xyxy pixel box, and confidence. "
            "Use stable provisional IDs; report ambiguity instead of inventing "
            "an unseen object."
        ),
        image_artifacts=image_artifacts,
    )


def build_initial_state_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    objects: Sequence[ObjectEstimate],
) -> PromptBundle:
    return PromptBundle(
        system_text=(
            "Infer a PDDL initial state for ViLaIn-TAMP from the supplied "
            "object estimates and immutable domain."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Immutable domain PDDL:\n{domain.text}\n"
            f"Fixed domain knowledge:\n{_domain_knowledge_text(domain)}\n\n"
            f"Object estimates:\n{_json([item.to_dict() for item in objects])}\n\n"
            "Return only PDDL object declarations followed by an `:init` "
            "fragment. Use only declared domain types and predicates. Do not "
            "return JSON, action schemas, a plan, or a goal."
        ),
    )


def build_goal_state_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    objects: Sequence[ObjectEstimate],
    initial_state_fragment: str,
) -> PromptBundle:
    return PromptBundle(
        system_text=(
            "Infer the PDDL goal for ViLaIn-TAMP without changing the fixed "
            "domain or prescribing actions."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Immutable domain PDDL:\n{domain.text}\n"
            f"Fixed domain knowledge:\n{_domain_knowledge_text(domain)}\n\n"
            f"Object estimates:\n{_json([item.to_dict() for item in objects])}\n\n"
            f"Estimated initial state:\n{initial_state_fragment}\n\n"
            "Return only one non-empty PDDL `:goal` fragment using declared "
            "objects and known predicates. Do not return JSON or an action plan."
        ),
    )


def build_corrective_planning_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    object_estimates: Sequence[ObjectEstimate],
    initial_problem: str,
    current_problem: str,
    current_failure: Mapping[str, Any],
    correction_history: Sequence[Mapping[str, Any]] = (),
    prior_problem_hashes: Sequence[str] = (),
    prior_error_summaries: Sequence[str] = (),
) -> PromptBundle:
    _reject_evaluator_context(current_failure)
    for entry in correction_history:
        _reject_evaluator_context(entry)
    history = {
        "complete_correction_records": [dict(item) for item in correction_history],
        "prior_problem_hashes": list(prior_problem_hashes),
        "prior_error_summaries": list(prior_error_summaries),
    }
    return PromptBundle(
        system_text=(
            "You perform bounded Corrective Planning for ViLaIn-TAMP. The "
            "domain is immutable. Revise the problem, never the domain, and "
            "never output an action sequence."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Immutable domain PDDL (SHA-256 {domain.sha256}):\n{domain.text}\n"
            f"Fixed domain knowledge:\n{_domain_knowledge_text(domain)}\n\n"
            f"Original object estimates:\n"
            f"{_json([item.to_dict() for item in object_estimates])}\n\n"
            f"Original problem:\n{initial_problem}\n\n"
            f"Current failed problem:\n{current_problem}\n\n"
            f"Current structured failure:\n{_json(dict(current_failure))}\n\n"
            f"Correction history:\n{_json(history)}\n\n"
            "Return exactly one complete replacement PDDL problem and nothing "
            "else. Keep the domain name and domain definitions unchanged. Use "
            "only observed objects and failure-supported corrections."
        ),
    )


def _domain_knowledge_text(domain: DomainDefinition) -> str:
    loaded = yaml.safe_load(domain.knowledge_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"domain knowledge must be a mapping: {domain.knowledge_path}")
    public = {
        "descriptions": dict(loaded.get("descriptions", {})),
        "types": dict(loaded.get("types", {})),
        "predicates": dict(loaded.get("predicates", {})),
        "actions": dict(loaded.get("actions", {})),
    }
    return _json(public)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _reject_evaluator_context(value: Any) -> None:
    forbidden_key_parts = (
        "benchmark",
        "ground_truth",
        "actual_task_success",
        "expected_answer",
        "feasibility_label",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in forbidden_key_parts):
                raise ValueError("benchmark evaluator data is forbidden in CP prompts")
            _reject_evaluator_context(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_evaluator_context(item)
