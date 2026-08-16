"""VLM functional decomposition with ranked candidate-type priors."""

from __future__ import annotations

import base64
import binascii
import copy
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parent
FUNCTION_CATALOG = ROOT / "functional_catalog.json"
SCENE_ALIASES = {
    "kitchen": "kitchen",
    "living_room": "living_room",
    "living-room": "living_room",
    "workshop": "workshop",
}
SCENES = tuple(sorted(set(SCENE_ALIASES.values())))
STATUSES = ("DECOMPOSED", "GOAL_UNSUPPORTED")
MAX_IMAGES = 8
MAX_REQUIREMENTS = 16
PROMPT_VERSION = 1
CAMERA_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
REQUIREMENT_ID = re.compile(r"^req_[1-9][0-9]{0,2}$")
FUNCTION_NAME = re.compile(r"^can_[a-z0-9_]{1,48}$")


SYSTEM_PROMPT = """\
You decompose a robot task goal into replaceable functional requirements.
Return only the requested JSON object.

Rules:
- Select only functions from the supplied function catalog. Function names are
  simple ability predicates.
- Create a requirement only when the goal needs a replaceable object or region
  to provide that function. Do not translate every manipulation action into a
  function.
- For each requirement, rank ten to fifteen common candidate types from most to
  least conventional for safe robot use. These are commonsense type priors,
  not detected scene instances.
- Every candidate must be a concrete, detector-searchable physical category.
  Never output an umbrella category from the supplied forbidden-generic list.
  Candidate types must be distinct.
- target_description names what the candidate acts on, contains, cleans, or
  supports. It must not redescribe the candidate itself.
- The images provide visible task context only. Never claim that a proposed
  candidate type is visible, present, available, graspable, or geometrically
  feasible.
- Do not emit simulator names, object IDs, coordinates, action sequences,
  inspection order, search results, or geometry results.
- Prefer rigid, robot-manipulable candidates. Do not rank a candidate merely
  because it could technically work when its use would be unsafe or clearly
  unconventional.
- Use depends_on only for dependencies between functional requirements, not
  for low-level action ordering.
- Treat a named serving destination as a fixed task target. Do not create a
  can_store requirement merely because the goal says to serve or place an
  object, unless the goal actually requires choosing an alternative region.
- A separate system will search observations for these types, semantically
  ground instances, and run target-specific geometric checks. Your ranking
  does not override those checks.
- Keep internal reasoning concise and leave enough output budget to emit the
  complete JSON object.
- Return GOAL_UNSUPPORTED only when the goal contains no function supported by
  the supplied catalog. Otherwise return DECOMPOSED. DECOMPOSED must contain at
  least one functional requirement and unsupported_reason must be exactly an
  empty string. GOAL_UNSUPPORTED must contain no requirements and must provide
  a non-empty unsupported_reason.
- Do not expose chain-of-thought.
"""


DECOMPOSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(STATUSES)},
        "scene": {"type": "string", "enum": list(SCENES)},
        "goal_summary": {"type": "string"},
        "functional_requirements": {
            "type": "array",
            "maxItems": MAX_REQUIREMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "function": {"type": "string"},
                    "candidate_kind": {
                        "type": "string",
                        "enum": ["object", "region"],
                    },
                    "purpose": {"type": "string"},
                    "target_description": {"type": "string"},
                    "ranked_candidate_types": {
                        "type": "array",
                        "minItems": 10,
                        "maxItems": 15,
                        "items": {"type": "string"},
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id",
                    "function",
                    "candidate_kind",
                    "purpose",
                    "target_description",
                    "ranked_candidate_types",
                    "depends_on",
                ],
                "additionalProperties": False,
            },
        },
        "unsupported_reason": {"type": "string"},
    },
    "required": [
        "status",
        "scene",
        "goal_summary",
        "functional_requirements",
        "unsupported_reason",
    ],
    "additionalProperties": False,
}


class FunctionalPlanningError(RuntimeError):
    """Base request, transport, or response error."""


class FunctionalPlanValidationError(FunctionalPlanningError):
    """The VLM returned data outside the decomposition contract."""


class CompletionTransport(Protocol):
    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """Return one decoded OpenAI-compatible completion response."""


@dataclass(frozen=True)
class PlannerConfig:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 300.0
    max_tokens: int = 8192
    enable_thinking: bool = True


def load_function_catalog(
    path: str | Path = FUNCTION_CATALOG,
) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported function catalog schema")
    minimum = document.get("min_ranked_candidates")
    maximum = document.get("max_ranked_candidates")
    functions = document.get("functions")
    if minimum != 10 or maximum != 15:
        raise ValueError("Function catalog must require 10-15 ranked candidates")
    forbidden = document.get("forbidden_generic_candidate_types")
    if (
        not isinstance(forbidden, list)
        or not forbidden
        or not all(_short_text(candidate, 40) for candidate in forbidden)
    ):
        raise ValueError("Function catalog must define forbidden generic types")
    if not isinstance(functions, dict) or not functions:
        raise ValueError("Function catalog must define functions")
    for name, specification in functions.items():
        if not isinstance(name, str) or not FUNCTION_NAME.fullmatch(name):
            raise ValueError(f"Invalid function name {name!r}")
        if not isinstance(specification, dict):
            raise ValueError(f"Invalid function specification for {name}")
        if set(specification) != {"candidate_kind", "description"}:
            raise ValueError(f"Invalid function fields for {name}")
        if specification["candidate_kind"] not in {"object", "region"}:
            raise ValueError(f"Invalid candidate kind for {name}")
        if not _short_text(specification["description"], 240):
            raise ValueError(f"Invalid description for {name}")
    return document


def normalize_scene(scene: str) -> str:
    normalized = SCENE_ALIASES.get(scene.strip().lower())
    if normalized is None:
        choices = ", ".join(SCENES)
        raise ValueError(f"Unsupported scene {scene!r}; choose {choices}")
    return normalized


def validate_request(request: Mapping[str, object]) -> dict[str, Any]:
    scene_value = request.get("scene")
    goal = request.get("goal")
    images = request.get("images")
    if not isinstance(scene_value, str):
        raise ValueError("scene must be a string")
    scene = normalize_scene(scene_value)
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
        raise ValueError("goal must contain between 1 and 4000 characters")
    if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
        raise ValueError(f"images must contain between 1 and {MAX_IMAGES} views")

    normalized_images = []
    seen_cameras: set[str] = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"images[{index}] must be an object")
        camera = image.get("camera")
        data_url = image.get("data_url")
        if not isinstance(camera, str) or not CAMERA_NAME.fullmatch(camera.strip()):
            raise ValueError(f"images[{index}].camera has an invalid name")
        camera = camera.strip()
        if camera in seen_cameras:
            raise ValueError(f"Duplicate camera name {camera!r}")
        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            raise ValueError(f"images[{index}].data_url must be an image data URL")
        if ";base64," not in data_url[:128]:
            raise ValueError(f"images[{index}].data_url must use base64 encoding")
        try:
            base64.b64decode(data_url.split(",", 1)[1], validate=True)
        except (binascii.Error, IndexError) as error:
            raise ValueError(f"images[{index}].data_url has invalid base64 data") from error
        seen_cameras.add(camera)
        normalized_images.append({"camera": camera, "data_url": data_url})
    return {"scene": scene, "goal": goal.strip(), "images": normalized_images}


def completion_payload(
    request: Mapping[str, object],
    config: PlannerConfig,
    catalog: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized = validate_request(request)
    function_catalog = catalog or load_function_catalog()
    task = {
        "scene": normalized["scene"],
        "goal": normalized["goal"],
        "prompt_version": PROMPT_VERSION,
        "camera_views": [image["camera"] for image in normalized["images"]],
        "function_catalog": function_catalog["functions"],
        "minimum_ranking_length": function_catalog["min_ranked_candidates"],
        "ranking_limit": function_catalog["max_ranked_candidates"],
        "forbidden_generic_candidate_types": function_catalog[
            "forbidden_generic_candidate_types"
        ],
    }
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": json.dumps(task, separators=(",", ":"), sort_keys=True),
        }
    ]
    for image in normalized["images"]:
        content.append({"type": "text", "text": f"Camera view: {image['camera']}"})
        content.append(
            {"type": "image_url", "image_url": {"url": image["data_url"]}}
        )
    response_schema = copy.deepcopy(DECOMPOSITION_SCHEMA)
    requirement_properties = response_schema["properties"][
        "functional_requirements"
    ]["items"]["properties"]
    requirement_properties["function"]["enum"] = sorted(
        function_catalog["functions"]
    )
    requirement_properties["id"]["pattern"] = REQUIREMENT_ID.pattern
    sampling = (
        {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
        if config.enable_thinking
        else {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
    )
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        **sampling,
        "max_tokens": config.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "functional_decomposition",
                "strict": True,
                "schema": response_schema,
            },
        },
    }


def _response_content(response: Mapping[str, object]) -> dict[str, Any]:
    try:
        choices = response["choices"]
        choice = choices[0]  # type: ignore[index]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise FunctionalPlanValidationError(
            "Completion response has no message content"
        ) from error
    if isinstance(content, dict):
        return content
    if content is None:
        finish_reason = choice.get("finish_reason", "unknown")
        reasoning = message.get("reasoning_content")
        detail = " after reasoning" if reasoning else ""
        raise FunctionalPlanValidationError(
            "Completion has no final JSON content"
            f"{detail} (finish_reason={finish_reason}); increase "
            "PLANNER_MAX_TOKENS or disable PLANNER_ENABLE_THINKING"
        )
    if not isinstance(content, str):
        raise FunctionalPlanValidationError(
            f"Completion content must be JSON text, received {type(content).__name__}"
        )
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise FunctionalPlanValidationError(
            "Completion content is not valid JSON"
        ) from error
    if not isinstance(result, dict):
        raise FunctionalPlanValidationError(
            "Functional decomposition must be a JSON object"
        )
    return result


def validate_decomposition(
    decomposition: Mapping[str, object],
    request: Mapping[str, object],
    catalog: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    normalized = validate_request(request)
    function_catalog = catalog or load_function_catalog()
    expected_fields = {
        "status",
        "scene",
        "goal_summary",
        "functional_requirements",
        "unsupported_reason",
    }
    if set(decomposition) != expected_fields:
        raise FunctionalPlanValidationError(
            "Decomposition fields do not match the response schema"
        )
    if decomposition.get("scene") != normalized["scene"]:
        raise FunctionalPlanValidationError(
            "Decomposition scene does not match request scene"
        )
    status = decomposition.get("status")
    if status not in STATUSES:
        raise FunctionalPlanValidationError("Unknown decomposition status")
    if not _short_text(decomposition.get("goal_summary"), 1000):
        raise FunctionalPlanValidationError(
            "goal_summary must be a non-empty string"
        )
    requirements = decomposition.get("functional_requirements")
    reason = decomposition.get("unsupported_reason")
    if not isinstance(requirements, list) or len(requirements) > MAX_REQUIREMENTS:
        raise FunctionalPlanValidationError(
            f"functional_requirements must contain at most {MAX_REQUIREMENTS} items"
        )
    if not isinstance(reason, str):
        raise FunctionalPlanValidationError("unsupported_reason must be a string")
    normalized_decomposition = dict(decomposition)
    if status == "DECOMPOSED":
        if not requirements:
            raise FunctionalPlanValidationError(
                "DECOMPOSED returned zero functional requirements"
            )
        if not _empty_reason(reason):
            raise FunctionalPlanValidationError(
                "DECOMPOSED returned a non-empty unsupported_reason"
            )
        normalized_decomposition["unsupported_reason"] = ""
    if status == "GOAL_UNSUPPORTED":
        if requirements:
            raise FunctionalPlanValidationError(
                "GOAL_UNSUPPORTED returned functional requirements"
            )
        if _empty_reason(reason):
            raise FunctionalPlanValidationError(
                "GOAL_UNSUPPORTED returned no explanation"
            )

    specifications = function_catalog["functions"]
    expected_requirement_fields = {
        "id",
        "function",
        "candidate_kind",
        "purpose",
        "target_description",
        "ranked_candidate_types",
        "depends_on",
    }
    by_id: dict[str, dict[str, Any]] = {}
    for requirement in requirements:
        if not isinstance(requirement, dict) or set(requirement) != expected_requirement_fields:
            raise FunctionalPlanValidationError(
                "Requirement fields do not match the response schema"
            )
        requirement_id = requirement["id"]
        function = requirement["function"]
        if not isinstance(requirement_id, str) or not REQUIREMENT_ID.fullmatch(
            requirement_id
        ):
            raise FunctionalPlanValidationError(
                f"Invalid requirement ID {requirement_id!r}"
            )
        if requirement_id in by_id:
            raise FunctionalPlanValidationError(
                f"Duplicate requirement ID {requirement_id!r}"
            )
        if not isinstance(function, str) or function not in specifications:
            raise FunctionalPlanValidationError(
                f"Unknown function {function!r}"
            )
        expected_kind = specifications[function]["candidate_kind"]
        if requirement["candidate_kind"] != expected_kind:
            raise FunctionalPlanValidationError(
                f"{function} requires candidate_kind={expected_kind}"
            )
        if not _short_text(requirement["purpose"], 500):
            raise FunctionalPlanValidationError(
                f"{requirement_id}.purpose must be a non-empty string"
            )
        if not _short_text(requirement["target_description"], 500):
            raise FunctionalPlanValidationError(
                f"{requirement_id}.target_description must be a non-empty string"
            )
        candidates = requirement["ranked_candidate_types"]
        minimum = function_catalog["min_ranked_candidates"]
        maximum = function_catalog["max_ranked_candidates"]
        if (
            not isinstance(candidates, list)
            or not minimum <= len(candidates) <= maximum
            or not all(_short_text(candidate, 80) for candidate in candidates)
        ):
            raise FunctionalPlanValidationError(
                f"{requirement_id}.ranked_candidate_types must contain "
                f"{minimum}-{maximum} names"
            )
        normalized_candidates = [candidate.strip().casefold() for candidate in candidates]
        if len(normalized_candidates) != len(set(normalized_candidates)):
            raise FunctionalPlanValidationError(
                f"{requirement_id} contains duplicate candidate types"
            )
        forbidden = {
            candidate.casefold()
            for candidate in function_catalog["forbidden_generic_candidate_types"]
        }
        generic = sorted(
            candidate
            for candidate in normalized_candidates
            if candidate in forbidden or candidate.split()[-1] in forbidden
        )
        if generic:
            raise FunctionalPlanValidationError(
                f"{requirement_id} contains generic candidate types: "
                + ", ".join(generic)
            )
        dependencies = requirement["depends_on"]
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            raise FunctionalPlanValidationError(
                f"{requirement_id}.depends_on must be an array of IDs"
            )
        if len(dependencies) != len(set(dependencies)):
            raise FunctionalPlanValidationError(
                f"{requirement_id} contains duplicate dependencies"
            )
        by_id[requirement_id] = requirement

    for requirement_id, requirement in by_id.items():
        for dependency in requirement["depends_on"]:
            if dependency == requirement_id or dependency not in by_id:
                raise FunctionalPlanValidationError(
                    f"{requirement_id} has an invalid dependency {dependency!r}"
                )
    _reject_dependency_cycles(by_id)
    return normalized_decomposition


def _short_text(value: object, maximum: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _empty_reason(value: str) -> bool:
    normalized = value.strip().casefold().rstrip(".")
    return normalized in {
        "",
        "none",
        "n/a",
        "not applicable",
        "no unsupported reason",
    }


def _reject_dependency_cycles(requirements: Mapping[str, Mapping[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(requirement_id: str) -> None:
        if requirement_id in visiting:
            raise FunctionalPlanValidationError(
                "Functional requirement dependencies contain a cycle"
            )
        if requirement_id in visited:
            return
        visiting.add(requirement_id)
        for dependency in requirements[requirement_id]["depends_on"]:
            visit(dependency)
        visiting.remove(requirement_id)
        visited.add(requirement_id)

    for requirement_id in requirements:
        visit(requirement_id)


class OpenAITransport:
    def __init__(self, config: PlannerConfig):
        self.config = config

    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise FunctionalPlanningError(
                f"Model server returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise FunctionalPlanningError(
                f"Cannot reach model server: {error.reason}"
            ) from error
        if not isinstance(result, dict):
            raise FunctionalPlanningError(
                "Model server response must be a JSON object"
            )
        return result


class FunctionalPlanner:
    def __init__(
        self,
        config: PlannerConfig,
        transport: CompletionTransport | None = None,
        catalog: Mapping[str, object] | None = None,
    ):
        self.config = config
        self.transport = transport or OpenAITransport(config)
        self.catalog = catalog or load_function_catalog()

    def decompose(self, request: Mapping[str, object]) -> dict[str, object]:
        payload = completion_payload(request, self.config, self.catalog)
        started = time.perf_counter()
        response = self.transport.complete(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        decomposition = validate_decomposition(
            _response_content(response), request, self.catalog
        )
        return {
            "decomposition": decomposition,
            "model": self.config.model,
            "latency_ms": round(latency_ms, 3),
            "prompt_version": PROMPT_VERSION,
            "function_catalog_schema_version": self.catalog["schema_version"],
            "search_started": False,
            "semantic_grounding_complete": False,
            "geometry_verified": False,
            "execution_started": False,
        }
