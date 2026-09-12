"""Run the real VLM pipeline with evidence withheld from the observation.

This is the shadow layer for the evidence ablation.  It adds no behaviour to
the pipeline and modifies no pipeline file: contract validation, the structural
sanitizer, semantic compilation, executability analysis, grounding, planning and
independent validation all run exactly as they do in the live benchmark, on
exactly the same code.

## What is ablated, and what is not

**The functional graph G_F is never touched.**  G_F states the task -- which
roles exist, how many, which relations the task requires -- and that is not
evidence.  An earlier version of this file masked G_F, and it could not work:
the task-interface validator exists to reject a malformed contract, and a graph
with `required_relations` stripped is malformed by exactly that standard.  Every
condition that removed a channel died in the validator, identically, for a
reason having nothing to do with evidence.  The validator was right.

**The observed graph G_O is what changes.**  The question an evidence ablation
asks is "what if the system could not perceive this kind of evidence?", which is
a property of perception.  Withholding a channel from G_O is also precisely what
the GT ablation's `evidence_components` gating did inside the grounder.

## Where the mask is applied

At `ground_graph`, the moment the observation is handed to grounding:

    scene -> perception -> G_O -> [mask] -> ground_graph -> planning -> scoring

All three domains import `ground_graph` inside the function that calls it, so
rebinding the module attribute covers kitchen, living room and workshop through
one seam -- including kitchen's per-stage search loop, which grounds repeatedly
against a graph that grows as regions are opened.  Patching the per-domain
observation builders would have missed workshop entirely, which assembles its
graph incrementally inside an adapter rather than through a builder function.

## How it avoids touching the pipeline

The attribute is rebound for the duration of one call and restored afterwards.
The wiring lives here, in shadow code; nothing under `functional_tamp_pipeline/`
is modified, and nothing outside the context manager observes a patched module.
"""
from __future__ import annotations

import contextlib
from typing import Any

from mujoco_scenes.fm_evidence_ablation import (
    CONDITION_LABELS, EVIDENCE_COMPONENTS, numeric_properties_referenced,
    resolve_components,
)


def mask_observed_node(node, components, numeric_properties) -> None:
    """Withhold from one observation the channels this condition disables."""
    if "semantic" not in components:
        # The detector reports nothing: no category and no belief contract.
        node.canonical_category = None
        node.semantic_labels = {}
    if "unary" not in components:
        node.unary_properties = {}
        node.unary_predicates = {}
        # Withhold exactly the measurements a numeric constraint would read,
        # and leave the rest of geometry for planning. See
        # numeric_properties_referenced() for why both halves matter.
        for key in list(node.geometry):
            if key in numeric_properties:
                node.geometry.pop(key, None)
        properties = node.geometry.get("properties")
        if isinstance(properties, dict):
            for key in list(properties):
                if key in numeric_properties:
                    properties.pop(key, None)


def mask_observed_graph(graph, condition: str, specification=None):
    """Return G_O with the channels this condition disables withheld.

    Mutates in place and returns the graph, matching how the pipeline passes
    its observation around.
    """
    components = resolve_components(condition)
    numeric_properties = (numeric_properties_referenced(specification)
                          if specification is not None else set())
    for node in graph.nodes.values():
        mask_observed_node(node, components, numeric_properties)
    if "binary" not in components:
        # Observed relations are binary evidence by definition.
        graph.relations = {}
    return graph


def observed_evidence_present(graph) -> dict[str, bool]:
    """What evidence an observation actually carries, to check a mask took."""
    return {
        "semantic": any(n.canonical_category or n.semantic_labels
                        for n in graph.nodes.values()),
        "unary": any(n.unary_properties or n.unary_predicates
                     for n in graph.nodes.values()),
        "binary": bool(graph.relations),
    }


@contextlib.contextmanager
def evidence_masked(condition: str):
    """Run the real pipeline with `condition`'s evidence withheld from G_O.

    Yields a dict recording how many groundings were masked, so a caller can
    assert the mask was actually applied.  A shadow that silently failed to
    patch would report the unablated pipeline under every condition and look
    like a clean result.
    """
    from mujoco_scenes.functional_tamp_pipeline import grounding as grounding_module

    resolve_components(condition)  # reject an unknown condition before running
    original = grounding_module.ground_graph
    stats: dict[str, Any] = {"condition": condition, "groundings": 0,
                             "label": CONDITION_LABELS.get(condition, condition),
                             "evidence_after_mask": []}

    def patched(graph_f, graph_o, context=None, *args, **kwargs):
        mask_observed_graph(graph_o, condition, graph_f)
        stats["groundings"] += 1
        if len(stats["evidence_after_mask"]) < 5:
            stats["evidence_after_mask"].append(observed_evidence_present(graph_o))
        return original(graph_f, graph_o, context, *args, **kwargs)

    grounding_module.ground_graph = patched
    try:
        yield stats
    finally:
        grounding_module.ground_graph = original
