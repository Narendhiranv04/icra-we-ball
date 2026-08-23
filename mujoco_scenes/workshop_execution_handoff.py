"""Stable boundary between Workshop functional grounding and execution.

The live contract consumes generic track/region IDs.  Replaying frozen simulator
artifacts uses their separately labelled post-hoc evaluation map to recover the
corresponding MuJoCo bodies; a real deployment must provide its own entity-handle
resolver.  Requirement generation is the only intentionally manual/FM-surrogate
piece upstream of this boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .workshop_ground_truth_planner import (
    WorkshopAssignment, load_variant_specs, solve_gt_assignment,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_L_OUTPUT = (
    ROOT / "outputs" / "workshop_yoloworld_l_five_view_close"
    / "final_14_bright_profile_frozen_v5"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_from_grounding(
    variant_id: str,
    episode: dict[str, Any],
    *,
    resolve_object: Callable[[str], str],
    resolve_region: Callable[[str], str],
    assignment_source: str = "PRODUCTION_FUNCTIONAL_GROUNDING",
) -> WorkshopAssignment:
    """Normalize a Phase-1 result into the execution planner's only input type."""
    status = str(episode.get("status", "UNKNOWN"))
    if status == "FEASIBLE":
        witness = episode.get("witness") or {}
        required = ("driver", "fastener")
        missing = [name for name in required if not witness.get(name)]
        if missing:
            raise ValueError(f"Feasible grounding decision is missing {missing}")
        source_ids = {name: str(witness[name]) for name in required}
        return WorkshopAssignment(
            variant_id=variant_id,
            intended_outcome="FEASIBLE",
            is_feasible=True,
            driver=resolve_object(source_ids["driver"]),
            fastener=resolve_object(source_ids["fastener"]),
            work_surface=(
                resolve_region(str(witness["work_surface"]))
                if witness.get("work_surface") else "MAIN_WORKBENCH_ZONE"
            ),
            assignment_source=assignment_source,
            source_ids=source_ids,
        )
    if status == "INFEASIBLE":
        reason = episode.get("rejection_reason")
        if not reason:
            raise ValueError("Infeasible grounding decision has no rejection reason")
        return WorkshopAssignment(
            variant_id=variant_id,
            intended_outcome="INFEASIBLE",
            is_feasible=False,
            rejection_reason=str(reason),
            assignment_source=assignment_source,
        )
    raise ValueError(f"Grounding status {status!r} is not executable")


def load_frozen_production_assignment(
    variant_id: str,
    artifact_root: Path = FROZEN_L_OUTPUT,
) -> WorkshopAssignment:
    """Replay the frozen L result with explicitly privileged simulator resolution."""
    variant_dir = artifact_root / variant_id
    episode = _read_json(variant_dir / "episode_result.json")
    mapping = _read_json(variant_dir / "privileged_eval_mapping.json")
    track_map = mapping.get("track_to_gt", {})
    region_map = mapping.get("region_to_gt", {})

    def object_resolver(track_id: str) -> str:
        if track_id not in track_map:
            raise ValueError(f"Frozen track {track_id} has no simulator evaluation mapping")
        return str(track_map[track_id])

    def region_resolver(region_id: str) -> str:
        if region_id not in region_map:
            raise ValueError(f"Frozen region {region_id} has no simulator evaluation mapping")
        return str(region_map[region_id])

    return decision_from_grounding(
        variant_id,
        episode,
        resolve_object=object_resolver,
        resolve_region=region_resolver,
        assignment_source="FROZEN_YOLOWORLD_L_PRODUCTION_POSTHOC_SIM_RESOLUTION",
    )


def validate_frozen_handoff_suite(artifact_root: Path = FROZEN_L_OUTPUT) -> dict[str, Any]:
    """Validate the redesigned decision-to-execution contract.

    The old 14-variant YOLO-World freeze belongs to the retired benchmark and
    must not be replayed against the replacement position/presence variants.
    Until new model artifacts are produced, this validates only the exact GT
    decision schema. It must not be reported as a production-grounding result.
    """
    rows = []
    for variant_id, spec in load_variant_specs().items():
        assignment = solve_gt_assignment(variant_id)
        exact = assignment.intended_outcome == spec["intended_outcome"]
        if assignment.is_feasible:
            exact = exact and assignment.driver == spec["expected_solution"]["driver"]
        else:
            exact = exact and assignment.rejection_reason == spec["rejection_reason"]
        rows.append({
            "variant_id": variant_id,
            "status": assignment.intended_outcome,
            "assignment_source": "GROUND_TRUTH_INTERFACE_CONTRACT",
            "exact_match": exact,
            "assignment": assignment.to_dict(),
        })
    return {
        "schema_version": 1,
        "interface": "WORKSHOP_FIXED_PAIR_DECISION_TO_EXECUTION_V2",
        "requirement_provider_status": (
            "REDESIGNED_GT_CONTRACT_VALID__PRODUCTION_GROUNDING_AND_LIVE_VLM_PENDING"
        ),
        "entity_resolution_note": (
            "The retired 14-variant frozen model artifacts are intentionally not reused. "
            "Live execution must provide track-to-entity resolution for the new variants."
        ),
        "total_variants": len(rows),
        "exact_matches": sum(row["exact_match"] for row in rows),
        "passed": all(row["exact_match"] for row in rows),
        "results": rows,
    }
