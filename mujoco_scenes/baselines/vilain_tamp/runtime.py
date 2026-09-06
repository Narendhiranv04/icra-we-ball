"""Production composition for planning-only ViLaIn-TAMP-Qwen runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

import numpy as np
import yaml

from .artifacts import atomic_write_json
from .attempt import TAMPAttemptRunner
from .config import BaselineConfig, Domain, ModelCondition
from .corrective_planning import CorrectivePlanningLoop
from .domains import load_domain
from .identity import BaselineIdentityResolver, EntityCandidate, fixed_entity_binding
from .interpreter import InterpreterModels, ViLaInInterpreter
from .live_fm import (
    DEFAULT_VLLM_BASE_URL,
    LiveModelClients,
    build_qwen_only_clients,
)
from .live_observations import create_live_observation_runtime
from .live_refinement import create_live_refinement_runtime
from .planner import FastDownwardPlanner, VALAdapter
from .runner import RunOptions, RunnerComponents
from .symbolic_contract import load_variant_action_contract


VAL_SOURCE_COMMIT = "3c7a1f330bdab0ba28a4762bb45c3f06c27fb6d4"
_TASK_CONFIG = {
    Domain.KITCHEN: ("configs/kitchen_feasibility_variants.yaml", "goal_instruction"),
    Domain.LIVING_ROOM: ("configs/living_room_variants.yaml", "task"),
    Domain.WORKSHOP: ("configs/workshop_variants.yaml", "canonical_task_instruction"),
}


class RuntimeCompositionError(RuntimeError):
    """Live components cannot be composed from the approved local runtime."""


def discover_served_model_id(
    base_url: str = DEFAULT_VLLM_BASE_URL, *, timeout_seconds: float = 10.0
) -> str:
    if base_url.rstrip("/") != DEFAULT_VLLM_BASE_URL:
        raise RuntimeCompositionError("model discovery is restricted to the local tunnel")
    try:
        with urlopen(f"{base_url}/models", timeout=timeout_seconds) as response:
            payload = json.load(response)
    except Exception as error:
        raise RuntimeCompositionError("Qwen tunnel required") from error
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeCompositionError("vLLM must expose exactly one served model")
    model_id = rows[0].get("id") if isinstance(rows[0], Mapping) else None
    if not isinstance(model_id, str) or not model_id.strip():
        raise RuntimeCompositionError("vLLM model listing has no served model ID")
    return model_id


class MuJoCoEntityCandidateProvider:
    """Expose neutral scene geometry for baseline-owned visual association."""

    def __init__(self, scene: Any, domain_key: str, stage_ids: Sequence[str]) -> None:
        self.scene = scene
        self.domain = load_domain(domain_key)
        self.stage_ids = tuple(stage_ids)

    def __call__(self, problem: Any, output_root: Path) -> tuple[EntityCandidate, ...]:
        del problem
        import mujoco

        mujoco.mj_forward(self.scene.model, self.scene.data)
        candidates: list[EntityCandidate] = []
        model = self.scene.model
        data = self.scene.data
        for body_id in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name or name.startswith("google:") or "camera_target" in name:
                continue
            geom_start = int(model.body_geomadr[body_id])
            geom_count = int(model.body_geomnum[body_id])
            if geom_count <= 0:
                continue
            lower = np.full(3, np.inf)
            upper = np.full(3, -np.inf)
            for geom_id in range(geom_start, geom_start + geom_count):
                rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
                center = (
                    np.asarray(data.geom_xpos[geom_id])
                    + rotation @ np.asarray(model.geom_aabb[geom_id, :3])
                )
                half = np.abs(rotation) @ np.asarray(model.geom_aabb[geom_id, 3:])
                lower = np.minimum(lower, center - half)
                upper = np.maximum(upper, center + half)
            if not np.all(np.isfinite(lower)):
                continue
            centroid = (lower + upper) / 2.0
            joint_count = int(model.body_jntnum[body_id])
            movable = any(
                int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE)
                for index in range(
                    int(model.body_jntadr[body_id]),
                    int(model.body_jntadr[body_id]) + joint_count,
                )
            )
            compatible_types = _neutral_entity_types(
                name, movable=movable, domain_key=self.domain.key
            )
            candidates.append(
                EntityCandidate(
                    entity_name=name,
                    broad_class="movable" if movable else "fixed-location",
                    compatible_pddl_types=compatible_types,
                    centroid_m=tuple(float(x) for x in centroid),
                    aabb_min_m=tuple(float(x) for x in lower),
                    aabb_max_m=tuple(float(x) for x in upper),
                    visible_stage_ids=self.stage_ids,
                    movable=movable,
                    evidence_artifacts=(),
                )
            )
        output_root.mkdir(parents=True, exist_ok=True)
        artifact = atomic_write_json(
            output_root / "scene_candidates.json",
            {"candidates": [item.to_dict() for item in candidates]},
        )
        return tuple(
            EntityCandidate(
                **{
                    **item.__dict__,
                    "evidence_artifacts": (str(artifact),),
                }
            )
            for item in candidates
        )


def _neutral_entity_types(
    entity_name: str, *, movable: bool, domain_key: str
) -> tuple[str, ...]:
    """Map descriptive simulator body names to neutral physical PDDL classes."""
    lower = entity_name.lower()
    if not movable:
        fixed = {
            "kitchen": ("location", "surface", "storage"),
            "living_room": ("location", "support", "seat"),
            "workshop": ("location", "surface", "storage", "target"),
        }
        return fixed[domain_key]
    classifiers = {
        "kitchen": (
            (("spoon", "fork", "knife", "utensil"), ("movable", "utensil")),
            (("bowl", "cup", "mug", "vessel"), ("movable", "vessel")),
            (("kettle", "jar", "bottle", "source"), ("movable", "source")),
        ),
        "living_room": (
            (("drink", "cup"), ("movable", "cup")),
            (("snack", "saucer", "plate"), ("movable", "saucer")),
            (("remote",), ("movable", "remote")),
        ),
        "workshop": (
            (("driver", "screwdriver"), ("movable", "driver")),
            (("screw", "fastener"), ("movable", "fastener")),
        ),
    }
    for tokens, types in classifiers[domain_key]:
        if any(token in lower for token in tokens):
            return types
    return ("movable",)


def build_live_components(
    config: BaselineConfig,
    options: RunOptions,
    *,
    base_url: str = DEFAULT_VLLM_BASE_URL,
    served_model_id: str | None = None,
    model_clients: LiveModelClients | None = None,
) -> RunnerComponents:
    """Compose existing live stages without performing observation or inference."""
    if config.model_condition is not ModelCondition.QWEN_ONLY:
        raise RuntimeCompositionError("live production runtime requires vilain_tamp_qwen")
    if options.execute:
        raise RuntimeCompositionError("Stage 22 production composition is planning-only")
    if options.fast_downward_path is None or options.val_path is None:
        raise RuntimeCompositionError("Fast Downward and VAL paths are required")

    observation_runtime = create_live_observation_runtime(
        domain=options.domain,
        variant=options.variant,
        observation_mode=options.observation_mode,
        output_root=options.output_directory / "observations",
        robot="google",
    )
    domain_definition = load_domain(options.domain.value)
    config_root = Path(__file__).resolve().parents[2] / "configs"
    symbolic_contract = load_variant_action_contract(
        domain_definition,
        options.variant,
        config_root=config_root,
    )
    fixed_bindings = _fixed_structural_bindings(
        options.domain, options.variant, symbolic_contract.structural_inventory,
        config_root,
    )
    actual_model_id = served_model_id or discover_served_model_id(base_url)
    if config.object_estimator_model != actual_model_id:
        raise RuntimeCompositionError(
            f"configured model {config.object_estimator_model!r} is not served "
            f"as {actual_model_id!r}"
        )
    clients = model_clients or build_qwen_only_clients(
        image_root=options.output_directory / "observations",
        served_model_id=actual_model_id,
        reference_revision=None,
        base_url=base_url,
        timeout_seconds=config.timeouts.model_seconds,
    )
    interpreter = ViLaInInterpreter(
        object_client=clients.object_client,
        reasoning_client=clients.reasoning_client,
        models=InterpreterModels(
            clients.object_estimator_model,
            clients.object_estimator_revision,
            clients.reasoning_model,
            clients.reasoning_model_revision,
        ),
        symbolic_contract=symbolic_contract,
    )
    symbolic_planner = FastDownwardPlanner(
        options.fast_downward_path,
        VALAdapter(
            options.val_path,
            expected_version=config.external_tools.val_version,
        ),
        expected_version=config.external_tools.fast_downward_version or "24.06",
        search_alias=config.search_configuration,
        timeout_seconds=config.timeouts.symbolic_seconds,
    )
    refinement = create_live_refinement_runtime(
        observation_runtime.scene,
        robot_name="google",
    )
    candidate_provider = MuJoCoEntityCandidateProvider(
        observation_runtime.scene,
        options.domain.value,
        ("000_initial",),
    )
    attempt_runner = TAMPAttemptRunner(
        domain=domain_definition,
        object_estimates=(),
        planner=symbolic_planner,
        identity_resolver=BaselineIdentityResolver(
            maximum_distance_m=0.75,
            ambiguity_margin_m=0.03,
        ),
        candidate_provider=candidate_provider,
        fixed_bindings=fixed_bindings,
        refiner=refinement.refiner,
        planning_scene_factory=refinement.planning_scene_factory,
    )
    corrective = CorrectivePlanningLoop(
        fm_client=clients.reasoning_client,
        attempt_runner=attempt_runner,
        model=clients.reasoning_model,
        model_revision=clients.reasoning_model_revision,
        max_corrections=options.cp_limit,
        symbolic_contract=symbolic_contract,
    )
    # The attempt runner needs the independently interpreted estimates, which
    # are created later by BaselineRunner. Bind them per run without exposing
    # any scene truth to the interpreter.
    corrective = _EstimateBindingCorrectivePlanning(corrective, attempt_runner)
    return RunnerComponents(
        task_instruction=_task_instruction(options.domain),
        observation=observation_runtime.protocol,
        interpreter=interpreter,
        corrective_planning=corrective,
    )


class _EstimateBindingCorrectivePlanning:
    def __init__(self, loop: CorrectivePlanningLoop, attempt: TAMPAttemptRunner) -> None:
        self.loop = loop
        self.attempt = attempt
        self.max_corrections = loop.max_corrections

    def run(self, *, object_estimates: Sequence[Any], **kwargs: Any) -> Any:
        self.attempt.object_estimates = tuple(object_estimates)
        return self.loop.run(object_estimates=object_estimates, **kwargs)


def _task_instruction(domain: Domain) -> str:
    relative_path, key = _TASK_CONFIG[domain]
    root = Path(__file__).resolve().parents[2]
    loaded = yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
    value = loaded.get(key) if isinstance(loaded, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeCompositionError(f"missing public task instruction for {domain.value}")
    return value.strip()


def _fixed_structural_bindings(
    domain: Domain,
    variant: str,
    inventory: Mapping[str, str],
    config_root: Path,
) -> dict[str, Any]:
    """Resolve public fixtures without reading objects, roles, or solutions."""
    if domain is Domain.KITCHEN:
        names = {
            key: key.upper() if key in {"d1", "d2", "c2", "b1", "c1"} else key
            for key in inventory
        }
    elif domain is Domain.LIVING_ROOM:
        loaded = yaml.safe_load(
            (config_root / "living_room_variants.yaml").read_text(encoding="utf-8")
        )
        regions = loaded.get("regions", {}) if isinstance(loaded, Mapping) else {}
        names = {
            str(region).lower(): str(entity)
            for region, entity in regions.items()
            if str(region).lower() in inventory
        }
        names["staging"] = "staging"
    else:
        del variant
        names = {
            key: (
                "MAIN_WORKBENCH_ZONE"
                if key.startswith("main_workbench_zone")
                else key.upper()
            )
            for key in inventory
        }
    return {
        symbolic_id: fixed_entity_binding(
            symbolic_id,
            names[symbolic_id],
            broad_class=object_type,
        )
        for symbolic_id, object_type in inventory.items()
        if object_type in {"location", "storage", "surface", "support", "target"}
    }
