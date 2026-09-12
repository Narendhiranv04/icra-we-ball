"""Evidence ablation over the FM-authored functional graph.

The GT ablation (`run_gt_evidence_ablation.py`, branch `baseline_execution`)
answers "which evidence channels does grounding need, given a perfect G_F?".
This answers the same question for the graph the FM actually produces, so the
two are directly comparable: same seven masks, same metric family, same
planner.

Two deliberate differences from the GT ablation, both forced by what is being
measured rather than by convenience:

1. **Masking is applied to the specification, not to the grounder.**  The GT
   ablation threaded an `evidence_components` parameter through `ground_graph`.
   Nothing here modifies the pipeline: a mask clears the corresponding evidence
   fields of G_F and the *unmodified* production pipeline runs on the result.
   That is not merely less invasive, it is the more honest experiment -- the
   system under test is the shipped one, byte for byte, and the ablation lives
   entirely in its input.  The two formulations coincide because grounding skips
   a check exactly when the field carrying it is empty (verified by
   `tests/test_fm_evidence_ablation.py::test_masking_matches_component_gating`).

2. **Repeats replace seeds.**  The GT ablation needed synthetic seeded
   tie-breaking because an oracle graph is deterministic, so an underconstrained
   mask would otherwise be scored on one arbitrary ordering.  The FM pipeline
   has real sampling variance: the same variant yields a different G_F on every
   trial.  Averaging over archived trials therefore measures the actual
   distribution of the method instead of a synthetic one.

Reads archived FM responses and makes no FM call, so a full sweep is
reproducible and costs no tokens.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)

# The seven non-empty subsets of the three evidence channels, named exactly as
# in the GT ablation so the two tables can be read side by side.
EVIDENCE_COMPONENTS = ("semantic", "unary", "binary")
COMPONENT_MASKS: dict[str, tuple[str, ...]] = {
    "semantic_only": ("semantic",),
    "unary_only": ("unary",),
    "binary_only": ("binary",),
    "no_binary": ("semantic", "unary"),
    "no_unary": ("semantic", "binary"),
    "no_semantic": ("unary", "binary"),
    "full": ("semantic", "unary", "binary"),
}


def resolve_components(condition: str) -> tuple[str, ...]:
    if condition not in COMPONENT_MASKS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {sorted(COMPONENT_MASKS)}")
    return COMPONENT_MASKS[condition]


def _mask_role(role: FunctionalRole, components: tuple[str, ...]) -> FunctionalRole:
    """Clear the evidence a disabled channel would have supplied.

    Entity kind and cardinality are *not* evidence -- they are the shape of the
    task, and removing them would ablate the question rather than the evidence.
    The GT ablation kept them for the same reason.
    """
    changes: dict[str, Any] = {}
    if "semantic" not in components:
        changes.update(semantic_categories=(), semantic_hints=())
    if "unary" not in components:
        changes.update(unary_predicates=(), numeric_constraints=())
    if not changes:
        return role
    # verification_mode is a claim about which checks apply; leaving it saying
    # SEMANTIC_AND_GEOMETRIC under a mask that removed one of them would make
    # the artifact describe a condition that was not run.
    if "semantic" not in components and "unary" not in components:
        changes["verification_mode"] = "STRUCTURAL_ONLY"
    elif "unary" not in components:
        changes["verification_mode"] = "SEMANTIC_ONLY"
    return replace(role, **changes)


def _mask_operation_group(group: OperationGroup, components: tuple[str, ...]) -> OperationGroup:
    if "binary" in components:
        return group
    # physical_preconditions overrides required_relations inside the grounder,
    # so clearing only the latter would leave binary evidence in force for every
    # group that declares preconditions -- an ablation that silently did nothing.
    return replace(group, required_relations=(), context_relations=(),
                   physical_preconditions=())


def mask_specification(specification: FunctionalRequirementGraph,
                       condition: str) -> FunctionalRequirementGraph:
    """Return a copy of G_F carrying only the enabled evidence channels."""
    components = resolve_components(condition)
    masked = replace(
        specification,
        nodes={name: _mask_role(role, components)
               for name, role in specification.nodes.items()},
        operation_groups=tuple(_mask_operation_group(g, components)
                               for g in specification.operation_groups),
        metadata={
            **dict(specification.metadata),
            "evidence_ablation": {
                "condition": condition,
                "enabled_evidence_components": list(components),
                "disabled_evidence_components": [c for c in EVIDENCE_COMPONENTS
                                                 if c not in components],
            },
        },
    )
    if "binary" not in components:
        # Explicit relations and provisional relation constraints are binary
        # geometry by definition.
        masked = replace(masked, relations=(), provisional_relation_constraints=())
    # `source` must survive: the pipeline refuses a replayed specification whose
    # source does not match the requested mode, and an ablation of the FM graph
    # is still the FM graph.
    assert masked.source == specification.source
    return masked


def evidence_channels_present(specification: FunctionalRequirementGraph) -> dict[str, bool]:
    """What evidence a graph actually carries, used to check a mask took effect."""
    return {
        "semantic": any(role.semantic_categories for role in specification.nodes.values()),
        "unary": any(role.unary_predicates or role.numeric_constraints
                     for role in specification.nodes.values()),
        "binary": bool(specification.relations) or any(
            g.required_relations or g.context_relations or g.physical_preconditions
            for g in specification.operation_groups),
    }


def specification_from_archived_response(raw_path: Path, domain: str,
                                         task: str) -> FunctionalRequirementGraph:
    """Rebuild the FM's G_F from an archived response, via the shipped provider."""
    from mujoco_scenes.functional_tamp_pipeline.spec_provider import provider_for_mode

    data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    raw_doc = (json.loads(data["content"])
               if isinstance(data.get("content"), str) else data)
    return provider_for_mode("vlm").provide(domain, task, [], raw_document=raw_doc)


def write_masked_specification(specification: FunctionalRequirementGraph,
                               condition: str, destination: Path) -> Path:
    """Serialise a masked G_F where the unmodified pipeline can replay it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(mask_specification(specification, condition).to_dict(),
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
