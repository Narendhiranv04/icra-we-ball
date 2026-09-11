#!/usr/bin/env python3
"""Regenerate method_freeze.json from the current source tree.

The manifest is a fingerprint of the frozen method: the active prompt and
schema, the runtime ontology and capability registry, the source files that
decide compilation and planning, and the inference configuration.  It has to be
rewritten whenever any of those deliberately change, and the integrity test then
holds the new state.  Offline; reads source only.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (  # noqa: E402
    RESPONSE_SCHEMA_V2, SYSTEM_PROMPT_V2, compute_v2_prompt_and_schema_hash)
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (  # noqa: E402
    get_robot_capability_registry_hash)
from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (  # noqa: E402
    get_runtime_semantic_ontology_hash)

PIPELINE = REPO / "mujoco_scenes" / "functional_tamp_pipeline"
MANIFEST = PIPELINE / "method_freeze.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["freeze_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    manifest["git_branch"] = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True).stdout.strip()
    manifest["active_prompt_sha256"] = hashlib.sha256(
        SYSTEM_PROMPT_V2.encode("utf-8")).hexdigest()
    manifest["response_schema_sha256"] = hashlib.sha256(
        json.dumps(RESPONSE_SCHEMA_V2, sort_keys=True).encode("utf-8")).hexdigest()
    manifest["combined_prompt_schema_sha256"] = compute_v2_prompt_and_schema_hash()
    manifest["runtime_semantic_ontology_sha256"] = get_runtime_semantic_ontology_hash()
    manifest["robot_capability_registry_sha256"] = get_robot_capability_registry_hash()
    manifest["predicate_registry_file_sha256"] = file_hash(PIPELINE / "predicate_registry.py")
    manifest["semantic_compiler_file_sha256"] = file_hash(PIPELINE / "semantic_compiler.py")
    manifest["planning_file_sha256"] = file_hash(PIPELINE / "planning.py")
    manifest["domains"] = {
        "kitchen_domain_sha256": file_hash(PIPELINE / "domains" / "kitchen.py"),
        "living_room_domain_sha256": file_hash(PIPELINE / "domains" / "living_room.py"),
        "workshop_domain_sha256": file_hash(PIPELINE / "domains" / "workshop.py"),
    }
    manifest["inference_config"]["max_tokens"] = 24000
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"rewrote {MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
