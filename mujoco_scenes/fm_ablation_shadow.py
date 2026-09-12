"""Run the real VLM pipeline end to end, with one evidence channel removed.

This is the shadow layer.  It adds no behaviour to the pipeline and changes no
pipeline file: every stage -- v3 contract validation, the structural sanitizer,
semantic compilation, executability analysis, grounding, planning, independent
validation -- runs exactly as it does in the live benchmark, on exactly the same
code.  The only difference is that the functional graph handed to grounding has
had one or more evidence channels cleared.

## Where the mask is applied, and why it must be there

The mask sits *inside the specification provider*, which is the last thing that
runs before the pipeline starts consuming G_F:

    archived FM response
      -> VLMSpecProvider.provide()      <- contract + sanitizer, UNMASKED
      -> mask_specification()           <- the ablation, and nothing else
      -> grounding, planning, scoring   <- the real pipeline, unchanged

Two earlier placements were wrong, and both are worth recording because they are
the obvious things to try:

1. **Masking before the provider** (feeding a masked graph through the replay
   path) fails, and should.  The task-interface validator's job is to reject a
   malformed FM contract, and an ablated graph is malformed by exactly that
   standard -- "operation group has empty required_relations".  Five of seven
   conditions died there, identically, for a reason having nothing to do with
   evidence.  It measured the validator.

2. **Masking inside grounding** (calling `ground_graph` directly, as the GT
   ablation does) skips the sanitizer and the validator entirely.  That is
   tolerable for an oracle graph, which is well-formed by construction, but for
   an FM graph those stages are a substantial part of the method under test, and
   skipping them measures something that is not the pipeline.

Masking after the contract stages and before grounding keeps both properties:
the FM's contract is judged exactly as production judges it, and the ablation
affects only the evidence available for grounding.

## How it avoids touching the pipeline

`run.provider_for_mode` is rebound for the duration of one call and restored
afterwards.  The wiring lives here, in shadow code; no file under
`functional_tamp_pipeline/` is modified, and nothing outside this context
manager observes a patched pipeline.
"""
from __future__ import annotations

import contextlib
from typing import Any

from mujoco_scenes.fm_evidence_ablation import mask_specification


class MaskedSpecProvider:
    """Wraps the real provider and ablates only what it returns."""

    def __init__(self, inner: Any, condition: str) -> None:
        self._inner = inner
        self._condition = condition
        self.masked_calls = 0

    def provide(self, domain, task_instruction, observation_images=None,
                raw_document=None, **kwargs):
        # The real provider runs first and in full: contract validation, the
        # structural sanitizer and semantic compilation all see the unmasked FM
        # output, so a contract failure stays a contract failure in every
        # condition rather than becoming an artifact of the mask.
        specification = self._inner.provide(
            domain, task_instruction, observation_images, raw_document, **kwargs)
        self.masked_calls += 1
        return mask_specification(specification, self._condition)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@contextlib.contextmanager
def evidence_masked(condition: str):
    """Run the real pipeline with `condition`'s evidence mask in force.

    Yields the wrapper so a caller can assert the mask was actually applied --
    a shadow that silently failed to patch would report the unablated pipeline
    seven times and look like a clean result.
    """
    from mujoco_scenes.functional_tamp_pipeline import run as run_module

    original = run_module.provider_for_mode
    holder: dict[str, Any] = {}

    def patched(mode: str):
        provider = original(mode)
        if mode != "vlm":
            return provider
        wrapper = MaskedSpecProvider(provider, condition)
        holder["wrapper"] = wrapper
        return wrapper

    run_module.provider_for_mode = patched
    try:
        yield holder
    finally:
        run_module.provider_for_mode = original
