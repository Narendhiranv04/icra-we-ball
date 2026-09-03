"""Infer Workshop roles/properties/candidates from goal and initial images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workshop_phase1.requirements import (
    CANONICAL_WORKSHOP_INSTRUCTION,
    FMRequirementProvider,
)


def build_result(
    provider: FMRequirementProvider,
    task_instruction: str,
    observation_images: list[Path],
    *,
    require_reviewed_contract: bool = False,
) -> dict:
    normalization_error = None
    try:
        requirements = provider.get_requirements(
            task_instruction, observation_images=observation_images
        )
    except ValueError as error:
        if provider.raw_decomposition is None or require_reviewed_contract:
            raise
        requirements = []
        normalization_error = str(error)
    ready = normalization_error is None
    return {
        "schema_version": 1,
        "scope": "VLM_REQUIREMENT_DECOMPOSITION_ONLY",
        "task_instruction": task_instruction,
        "initial_observation_images": provider.fm_adapter.last_observation_images,
        "raw_vlm_decomposition": provider.raw_decomposition,
        "normalized_contract": ({
            "functional_requirements": [
                requirement.to_dict() for requirement in requirements
            ],
            "ranked_detector_vocabulary": (
                provider.get_ranked_detector_vocabulary()
            ),
        } if ready else None),
        "ready_for_grounding": ready,
        "reviewed_ontology_audit": {
            "status": "PASS" if ready else "REVIEW_REQUIRED",
            "issues": [] if ready else [normalization_error],
            "note": (
                "The reviewed ontology was used only after the VLM response; "
                "it was not included in the model prompt."
            ),
        },
        "fm_calls": provider.fm_adapter.metrics.total_calls,
        "observation_search_started": False,
        "semantic_grounding_started": False,
        "geometry_verification_started": False,
        "planning_started": False,
        "execution_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruction", default=CANONICAL_WORKSHOP_INSTRUCTION
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/workshop_vlm_requirements.json"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="Initial-observation image; repeat for multiple camera views (max 8)",
    )
    arguments = parser.parse_args()
    result = build_result(
        FMRequirementProvider(), arguments.instruction, arguments.image
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nWrote {arguments.output}")


if __name__ == "__main__":
    main()
