"""Paper-oriented prompts for the independent ViLaIn-TAMP model calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import yaml

from .contracts import FixedSceneEvidence, ObjectEstimate, ViLaInObservation
from .domains.registry import DomainDefinition
from .observations import prompt_observation_payload
from .symbolic_contract import (
    GroundedFact,
    VariantActionContract,
    grouped_fact_payload,
    object_type_table,
)


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
            f"Legal type capability guide:\n{_type_capability_text(domain)}\n\n"
            "Observation ordering:\n"
            f"{_json(observation_payload)}\n\n"
            "Return JSON only: no Markdown fence, prose, or reasoning. Schema: "
            "{\"objects\":[{\"label\":str,\"pddl_type\":str,"
            "\"description\":str,\"detections\":[{\"stage_id\":str,"
            "\"camera_id\":str,\"bbox_1000\":[x1,y1,x2,y2],"
            "\"confidence\":number}]}]}. Use exactly one best-view detection per "
            "object. Coordinates are normalized integers satisfying "
            "0<=x1<x2<=1000 and 0<=y1<y2<=1000; do not emit pixel coordinates. "
            "Use only listed stage/camera IDs and domain PDDL types. Keep "
            "description empty unless needed to distinguish visually similar "
            "objects. Inventory every clearly visible task-relevant manipulable "
            "entity, including distinct objects with similar appearance. Assign "
            "exactly one legal PDDL type to each entity. The domain type names "
            "describe capabilities: use their fixed definitions, not an expected "
            "answer. Do not omit a visible entity merely because its exact label "
            "is uncertain; use a neutral visual label and describe the uncertainty. "
            "Do not infer hidden entities. Report ambiguity instead of inventing "
            "unseen objects. "
            "Return only movable/manipulable entity types; fixed storage, surfaces, "
            "supports, targets, seats, and the robot are supplied separately by the "
            "public structural inventory and must not be returned as visual objects."
        ),
        image_artifacts=image_artifacts,
    )


def build_object_completeness_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    observations: Sequence[ViLaInObservation],
    existing_objects: Sequence[ObjectEstimate],
    missing_types: Sequence[str],
) -> PromptBundle:
    """Request one same-image, type-capability inventory recheck."""
    base = build_object_estimation_prompt(
        task_instruction=task_instruction,
        domain=domain,
        observations=observations,
    )
    return PromptBundle(
        system_text=base.system_text,
        user_text=(
            base.user_text
            + "\n\nThis is the single bounded completeness recheck of the SAME "
            "images. The first inventory contains:\n"
            + _json(
                [
                    {"label": item.label, "pddl_type": item.pddl_type}
                    for item in existing_objects
                ]
            )
            + "\n\nNo observed entity can currently instantiate these fixed "
            "domain capability types:\n"
            + _json(list(missing_types))
            + "\nRe-check all supplied views for clearly visible entities of only "
            "those missing types. A returned entity may be a newly noticed object "
            "or a visually supported reclassification of an item from the first "
            "inventory; reuse its exact prior label in the latter case. Return only "
            "missing-type candidates, not unchanged prior objects. Do not name a "
            "required hidden object, "
            "invent an object, or infer benchmark truth. If none is clearly "
            "visible, return {\"objects\":[]}."
        ),
        image_artifacts=base.image_artifacts,
    )


def build_initial_state_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    observations: Sequence[ViLaInObservation] = (),
    objects: Sequence[ObjectEstimate],
    fact_candidates: Sequence[GroundedFact],
    symbolic_contract: VariantActionContract | None = None,
    fixed_scene_evidence: Sequence[FixedSceneEvidence] = (),
) -> PromptBundle:
    contract = symbolic_contract
    type_table = (
        object_type_table(
            {item.object_id: item.pddl_type for item in objects}, contract
        )
        if contract is not None
        else {}
    )
    contract_text = (
        "Variant action contract (capabilities, not a plan):\n"
        + _json(contract.to_dict())
        + "\n\n"
        if contract is not None
        else ""
    )
    image_artifacts = tuple(
        frame.rgb_path
        for observation in observations
        for frame in observation.camera_frames
    )
    observation_payload = (
        prompt_observation_payload(observations) if observations else {}
    )

    movable_objects_payload = [
        {
            "object_id": item.object_id,
            "label": item.label,
            "pddl_type": item.pddl_type,
            "estimated_centroid_m": (
                [round(float(x), 3) for x in item.estimated_centroid_m]
                if item.estimated_centroid_m is not None
                else None
            ),
            "detections": [
                {
                    "stage_id": str(d.get("stage_id", "")),
                    "camera_id": str(d.get("camera_id", "")),
                    "bbox_1000": [int(v) for v in d.get("xyxy", d.get("bbox_1000", []))],
                    "confidence": round(float(d.get("confidence", 1.0)), 3),
                }
                for d in item.detections
            ],
        }
        for item in objects
    ]

    fixed_evidence_payload = [
        {
            "symbolic_id": item.symbolic_id,
            "pddl_type": item.pddl_type,
            "physically_present": item.physically_present,
            "world_centroid_m": (
                [round(float(x), 3) for x in item.centroid_m]
                if item.centroid_m is not None
                else None
            ),
            "camera_detections": [
                {
                    "stage_id": str(p.get("stage_id", "")),
                    "camera_id": str(p.get("camera_id", "")),
                    "bbox_1000": [int(v) for v in p.get("bbox_1000", [])],
                }
                for p in item.camera_projections
            ],
            "description": item.description,
        }
        for item in fixed_scene_evidence
    ]

    obs_text = (
        f"Observation ordering:\n{_json(observation_payload)}\n\n"
        if observation_payload
        else ""
    )
    fixed_text = (
        f"Neutral fixed-scene support evidence:\n{_json(fixed_evidence_payload)}\n\n"
        if fixed_evidence_payload
        else ""
    )

    return PromptBundle(
        system_text=(
            "Infer a PDDL initial state for ViLaIn-TAMP from the supplied RGB scene "
            "images, movable object detections, neutral fixed-scene evidence, and immutable domain."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Fixed domain knowledge:\n{_domain_knowledge_text(domain)}\n\n"
            f"{obs_text}"
            f"Movable object estimates:\n{_json(movable_objects_payload)}\n\n"
            f"Exact object ID/type table:\n{_json(dict(type_table))}\n\n"
            f"{fixed_text}"
            f"{contract_text}"
            "Legal positive grounded facts grouped by predicate:\n"
            f"{_json(_plain_grouped_facts(fact_candidates))}\n\n"
            "Select only facts that are true NOW in the initial state from the "
            "supplied scene images, detected movable object bounding boxes/centroids, and "
            "neutral fixed-scene evidence. The task instruction states desired outcomes, "
            "not facts already achieved.\n"
            "- For at(object, location): select it only when the object's visual and spatial "
            "evidence supports that location.\n"
            "- For present(location) or accessible(location): use actual physical fixed-scene "
            "evidence. If a fixture is physically absent, do not select it.\n"
            "- A single-gripper robot cannot be handempty and holding, cannot hold "
            "multiple objects, and one object cannot be at multiple locations.\n"
            "- Closed storage is not open or accessible merely because its exterior is visible.\n"
            "- Be conservative: do not mark goal effects such as stirred, fastened, inserted, "
            "supports, inside, or contains as initially true unless directly established by the scene.\n"
            "False facts are omitted. "
            "Return compact JSON only with exactly this schema: "
            '{"true_fact_ids":["f0001"]}. Use only listed IDs. Return no raw '
            "PDDL, variables, negative literals, Markdown, explanation, plan, "
            "or chain-of-thought."
        ),
        image_artifacts=image_artifacts,
    )


def build_initial_consistency_prompt(
    *,
    task_instruction: str,
    previous_fact_ids: Sequence[str],
    diagnostics: Sequence[Mapping[str, Any]],
    fact_candidates: Sequence[GroundedFact],
    observations: Sequence[ViLaInObservation] = (),
    objects: Sequence[ObjectEstimate] = (),
    fixed_scene_evidence: Sequence[FixedSceneEvidence] = (),
) -> PromptBundle:
    image_artifacts = tuple(
        frame.rgb_path
        for observation in observations
        for frame in observation.camera_frames
    )
    movable_payload = (
        [
            {
                "object_id": item.object_id,
                "label": item.label,
                "pddl_type": item.pddl_type,
                "estimated_centroid_m": (
                    [round(float(x), 3) for x in item.estimated_centroid_m]
                    if item.estimated_centroid_m is not None
                    else None
                ),
            }
            for item in objects
        ]
        if objects
        else []
    )
    fixed_payload = (
        [
            {
                "symbolic_id": item.symbolic_id,
                "pddl_type": item.pddl_type,
                "physically_present": item.physically_present,
                "world_centroid_m": (
                    [round(float(x), 3) for x in item.centroid_m]
                    if item.centroid_m is not None
                    else None
                ),
            }
            for item in fixed_scene_evidence
        ]
        if fixed_scene_evidence
        else []
    )
    movable_text = (
        f"Movable object estimates:\n{_json(movable_payload)}\n\n"
        if movable_payload
        else ""
    )
    fixed_text = (
        f"Neutral fixed-scene support evidence:\n{_json(fixed_payload)}\n\n"
        if fixed_payload
        else ""
    )
    return PromptBundle(
        system_text=(
            "Correct one ViLaIn-TAMP initial fact selection using the supplied scene "
            "images and generic physical consistency rules. Never output a plan or PDDL."
        ),
        user_text=(
            f"Task instruction (desired outcome, not current truth):\n"
            f"{task_instruction}\n\n"
            f"{movable_text}"
            f"{fixed_text}"
            f"Previous true fact IDs:\n{_json(list(previous_fact_ids))}\n\n"
            f"Physical consistency diagnostics:\n{_json(list(diagnostics))}\n\n"
            f"Legal initial facts:\n{_json(_plain_grouped_facts(fact_candidates))}\n\n"
            "Return a conservative, physically consistent initial state. Preserve "
            "only facts supported by the scene observation or fixed physical structure. "
            "Desired task effects are not initially true merely because the task "
            "requests them. Return exactly {\"true_fact_ids\":[...]} using listed "
            "IDs only, with no PDDL, plan, explanation, or benchmark answer."
        ),
        image_artifacts=image_artifacts,
    )


def build_goal_state_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    objects: Sequence[ObjectEstimate],
    initial_state_fragment: str,
    fact_candidates: Sequence[GroundedFact],
    declared_object_types: Mapping[str, str] | None = None,
    symbolic_contract: VariantActionContract | None = None,
) -> PromptBundle:
    type_table = dict(declared_object_types or {
        item.object_id: item.pddl_type for item in objects
    })
    contract_text = (
        "Variant action contract (capabilities, not a plan):\n"
        + _json(symbolic_contract.to_dict())
        + "\n\n"
        if symbolic_contract is not None
        else ""
    )
    return PromptBundle(
        system_text=(
            "Infer the PDDL goal for ViLaIn-TAMP without changing the fixed "
            "domain or prescribing actions."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Fixed domain knowledge:\n{_domain_knowledge_text(domain)}\n\n"
            f"Object estimates:\n{_json([item.to_dict() for item in objects])}\n\n"
            f"Exact declared object ID/type table:\n{_json(type_table)}\n\n"
            f"Exact predicate signatures:\n"
            f"{_json(dict(domain.predicate_signatures))}\n\n"
            f"{contract_text}"
            f"Selected initial facts:\n{initial_state_fragment}\n\n"
            "Legal achievable grounded goal facts grouped by predicate:\n"
            f"{_json(_plain_grouped_facts(fact_candidates, goal_only=True))}\n\n"
            "Select the minimal set of facts required to make the task instruction "
            "true. Every selected fact must be directly justified by the task. Do "
            "not select extra desirable outcomes, maximize selected facts, require "
            "every vessel to contain every content, or choose a fact merely because "
            "it is legal. Return compact "
            'JSON only with exactly this schema: {"goal_fact_ids":["f0001"]}. '
            "Use only listed IDs. Return no raw PDDL, variables, Markdown, "
            "explanation, plan, or chain-of-thought."
        ),
    )


def build_goal_reachability_prompt(
    *,
    task_instruction: str,
    previous_goal_ids: Sequence[str],
    diagnostics: Sequence[Mapping[str, Any]],
    fact_candidates: Sequence[GroundedFact],
) -> PromptBundle:
    return PromptBundle(
        system_text=(
            "Correct one ViLaIn-TAMP goal fact selection using only a generic "
            "relaxed symbolic reachability diagnosis. Never output a plan or PDDL."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Previous goal fact IDs:\n{_json(list(previous_goal_ids))}\n\n"
            f"Generic reachability diagnostics:\n{_json(list(diagnostics))}\n\n"
            "Legal achievable grounded goal facts:\n"
            f"{_json(_plain_grouped_facts(fact_candidates, goal_only=True))}\n\n"
            "Select the minimal directly task-justified goal facts that are "
            "symbolically achievable with the observed typed objects. Do not weaken "
            "or replace the requested task merely to obtain a plan. Return exactly "
            '{"goal_fact_ids":[...]} using listed IDs only.'
        ),
    )


def build_schema_regeneration_prompt(
    *,
    module: str,
    previous_output: str,
    validation_errors: Sequence[Mapping[str, Any]],
    valid_fact_ids: Sequence[str],
) -> PromptBundle:
    """Request one schema-only repair without supplying semantic answers."""
    field = "true_fact_ids" if module == "initial_state" else "goal_fact_ids"
    return PromptBundle(
        system_text=(
            "Repair one ViLaIn-TAMP fact-selection wire response. Return JSON "
            "fact IDs only; do not infer or output an action sequence."
        ),
        user_text=(
            f"Module: {module}\n\n"
            f"Previous output:\n{previous_output}\n\n"
            f"Structured wire/schema errors:\n{_json(list(validation_errors))}\n\n"
            f"Valid fact IDs:\n{_json(list(valid_fact_ids))}\n\n"
            f'Return exactly {{"{field}":[...]}} with only valid listed IDs. '
            "Do not return PDDL, facts, object names, an explanation, a plan, a "
            "preferred replacement fact, or a benchmark answer."
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


def build_corrective_fact_selection_prompt(
    *,
    task_instruction: str,
    domain: DomainDefinition,
    current_problem: str,
    current_failure: Mapping[str, Any],
    fact_candidates: Sequence[GroundedFact],
) -> PromptBundle:
    _reject_evaluator_context(current_failure)
    return PromptBundle(
        system_text=(
            "Perform bounded ViLaIn-TAMP corrective planning by selecting from "
            "a fixed legal grounded-fact universe. Never write PDDL or a plan."
        ),
        user_text=(
            f"Task instruction:\n{task_instruction}\n\n"
            f"Current deterministic problem:\n{current_problem}\n\n"
            f"Current structured failure:\n{_json(dict(current_failure))}\n\n"
            "Legal initial facts:\n"
            f"{_json(_plain_grouped_facts(fact_candidates))}\n\n"
            "Legal achievable goal facts:\n"
            f"{_json(_plain_grouped_facts(fact_candidates, goal_only=True))}\n\n"
            "Return compact JSON only with exactly two fields: "
            '{"true_fact_ids":[...],"goal_fact_ids":[...]}. '
            "Select only IDs from the corresponding list. Return no PDDL, "
            "variables, action sequence, "
            "explanation, preferred object, or benchmark answer."
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


def _type_capability_text(domain: DomainDefinition) -> str:
    """Describe legal types from public PDDL capabilities, never scene truth."""
    common = {
        "movable": "a physically manipulable entity that can be picked",
        "location": "a place from which an object can be picked or placed",
        "storage": "an articulated location that can be opened",
        "surface": "a support location on which an object can be placed",
        "support": "a support location on which an object can be placed",
        "vessel": "a receiving container that can contain material or accept a utensil",
        "source": "a movable dispensing container whose contents can be poured into a vessel",
        "utensil": "a movable implement usable for stirring or serving",
        "driver": "a movable tool capable of driving a compatible fastener",
        "fastener": "a movable fastening item that can be inserted and driven",
        "target": "a fixed insertion or fastening target",
        "cup": "a movable drinking cup",
        "saucer": "a movable saucer",
        "remote": "a movable remote-control device",
        "seat": "a fixed seating entity used only for neutral support relations",
        "content": "a symbolic material kind; do not visually detect it as an object",
    }
    return _json(
        {
            type_name: common.get(
                type_name,
                f"legal PDDL type with parent {parent or 'object'}",
            )
            for type_name, parent in domain.type_hierarchy.items()
        }
    )


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _plain_grouped_facts(
    facts: Sequence[GroundedFact], *, goal_only: bool = False
) -> dict[str, list[dict[str, Any]]]:
    grouped = grouped_fact_payload(facts, goal_only=goal_only)
    return {
        predicate: [dict(item) for item in rows]
        for predicate, rows in grouped.items()
    }


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
