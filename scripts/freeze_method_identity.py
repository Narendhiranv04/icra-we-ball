#!/usr/bin/env python3
"""Emit the authoritative identity of the frozen method.

Everything that can change what the system does, in one file, so a later run can
be checked against it rather than assumed to match.  Reads only; runs nothing.

The identity is not a summary written by hand.  Each field is computed from the
artifact it describes, because a freeze whose numbers were typed in is a freeze
that can silently stop describing the tree it names.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def build_identity() -> dict:
    from mujoco_scenes.determinism import REPRODUCIBLE_CHILD_ENV
    from mujoco_scenes.evaluation_outcome import (
        COMPLETION_CLAIMED_STATUSES,
        INFEASIBILITY_CONCLUDED_STATUSES,
    )
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
        compute_v3_prompt_and_schema_hash,
    )
    from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
        get_robot_capability_registry_hash,
    )
    from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (
        get_runtime_semantic_ontology_hash,
    )
    from mujoco_scenes.final_paper_variant_labels import PREFIXES, VARIANT_LABELS

    runner = REPO / "scripts" / "run_live_repeat_experiment.py"
    spec = importlib.util.spec_from_file_location("_freeze_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grid = sorted(f"{PREFIXES[d]}{i + 1}"
                  for d, labels in VARIANT_LABELS.items()
                  for i in range(len(labels)))

    identity = {
        "git": {
            "sha": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "clean": _git("status", "--porcelain") == "",
            "committed_at": _git("log", "-1", "--format=%cI"),
        },
        "semantic_contract": {
            "prompt_and_schema_sha256": compute_v3_prompt_and_schema_hash(),
            "capability_registry_sha256": get_robot_capability_registry_hash(),
            "runtime_ontology_sha256": get_runtime_semantic_ontology_hash(),
            "schema_version": "3",
        },
        "perception": {
            "workshop_config": "mujoco_scenes/configs/workshop_phase1_yoloworld_l_five_view_close.yaml",
            "workshop_config_sha256": _sha256_file(
                REPO / "mujoco_scenes/configs/workshop_phase1_yoloworld_l_five_view_close.yaml"),
            "workshop_geometry_config_sha256": _sha256_file(
                REPO / "mujoco_scenes/configs/workshop_geometry_inference_yoloworld_l.yaml"),
            "semantic_grounding_config_sha256": _sha256_file(
                REPO / "mujoco_scenes/configs/semantic_grounding.yaml"),
            "determinism_module_sha256": _sha256_file(REPO / "mujoco_scenes/determinism.py"),
        },
        "determinism": {
            "reproducible_child_env": dict(sorted(REPRODUCIBLE_CHILD_ENV.items())),
            "strict_deterministic_algorithms": True,
            "note": "PYTHONHASHSEED and the thread counts are read at process "
                    "start, so they are passed to every spawned trial rather "
                    "than set in process.",
        },
        "sampler": dict(sorted(module.SAMPLER.items())),
        "evaluator": {
            "module_sha256": _sha256_file(REPO / "mujoco_scenes/evaluation_outcome.py"),
            "completion_claimed_statuses": sorted(COMPLETION_CLAIMED_STATUSES),
            "infeasibility_concluded_statuses": sorted(INFEASIBILITY_CONCLUDED_STATUSES),
        },
        "benchmark_grid": {
            "variants": grid,
            "variant_count": len(grid),
            "domains": {d: len(v) for d, v in sorted(VARIANT_LABELS.items())},
        },
    }
    payload = json.dumps(identity, sort_keys=True).encode()
    identity["identity_sha256"] = hashlib.sha256(payload).hexdigest()
    return identity


def main() -> int:
    out = REPO / "METHOD_FREEZE.json"
    identity = build_identity()
    out.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(json.dumps(identity, indent=2, sort_keys=True))
    print(f"\nwrote {out.relative_to(REPO)}")
    if not identity["git"]["clean"]:
        print("WARNING: working tree is not clean; this identity does not "
              "describe a committed state", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
