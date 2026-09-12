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

# The ablation is cumulative: each condition adds one evidence channel to the
# one before it, so a difference between two rows is attributable to the single
# channel that was added.  The seven-subset form answers a different question
# (which channels are individually redundant) and is not what is reported here.
EVIDENCE_COMPONENTS = ("semantic", "unary", "binary")
COMPONENT_MASKS: dict[str, tuple[str, ...]] = {
    "semantic_only": ("semantic",),
    "semantic_unary": ("semantic", "unary"),
    "full": ("semantic", "unary", "binary"),
}

# Human-readable names for reports, in the order the conditions nest.
CONDITION_LABELS = {
    "semantic_only": "semantic only",
    "semantic_unary": "semantic + unary geometric",
    "full": "semantic + unary + binary geometric",
}
CONDITION_ORDER = ("semantic_only", "semantic_unary", "full")


def resolve_components(condition: str) -> tuple[str, ...]:
    if condition not in COMPONENT_MASKS:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {sorted(COMPONENT_MASKS)}")
    return COMPONENT_MASKS[condition]


def numeric_properties_referenced(specification) -> set[str]:
    """Property names any role's numeric constraints actually consume.

    The unary mask must withhold exactly these from the observation and nothing
    else.  Grounding reads a numeric property from `unary_properties` first and
    then falls back to `geometry` and `geometry["properties"]`, so clearing only
    `unary_properties` leaves the evidence reachable and the ablation silently
    does nothing.  Clearing all of `geometry` is equally wrong: planning and
    execution read poses and extents from it, and a run would then fail for
    reasons having nothing to do with evidence.
    """
    names: set[str] = set()
    for role in specification.nodes.values():
        for constraint in role.numeric_constraints:
            names.add(constraint.property_name)
    return names


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
