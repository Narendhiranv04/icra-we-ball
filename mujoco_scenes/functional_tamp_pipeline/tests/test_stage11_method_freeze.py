"""Unit tests for Stage 11 method freeze invariants and provenance integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    RESPONSE_SCHEMA_V2,
    SYSTEM_PROMPT_V2,
    compute_v2_prompt_and_schema_hash,
)
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    get_robot_capability_registry_hash,
)
from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (
    get_runtime_semantic_ontology_hash,
)


def _file_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_method_freeze_manifest_integrity():
    """Verify that method_freeze.json matches current code hashes exactly."""
    root = Path(__file__).resolve().parents[3]
    freeze_path = root / "mujoco_scenes" / "functional_tamp_pipeline" / "method_freeze.json"
    assert freeze_path.exists(), "method_freeze.json must exist"

    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))

    # Active prompt and schema hashes
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT_V2.encode("utf-8")).hexdigest()
    schema_hash = hashlib.sha256(json.dumps(RESPONSE_SCHEMA_V2, sort_keys=True).encode("utf-8")).hexdigest()
    combined_hash = compute_v2_prompt_and_schema_hash()

    assert manifest["active_prompt_sha256"] == prompt_hash
    assert manifest["response_schema_sha256"] == schema_hash
    assert manifest["combined_prompt_schema_sha256"] == combined_hash

    # Ontology and capability hashes
    assert manifest["runtime_semantic_ontology_sha256"] == get_runtime_semantic_ontology_hash()
    assert manifest["robot_capability_registry_sha256"] == get_robot_capability_registry_hash()

    # Core source file hashes
    pipeline_dir = root / "mujoco_scenes" / "functional_tamp_pipeline"
    assert manifest["predicate_registry_file_sha256"] == _file_hash(pipeline_dir / "predicate_registry.py")
    assert manifest["semantic_compiler_file_sha256"] == _file_hash(pipeline_dir / "semantic_compiler.py")
    assert manifest["planning_file_sha256"] == _file_hash(pipeline_dir / "planning.py")
    assert manifest["domains"]["kitchen_domain_sha256"] == _file_hash(pipeline_dir / "domains" / "kitchen.py")
    assert manifest["domains"]["living_room_domain_sha256"] == _file_hash(pipeline_dir / "domains" / "living_room.py")
    assert manifest["domains"]["workshop_domain_sha256"] == _file_hash(pipeline_dir / "domains" / "workshop.py")

    # Inference config invariants
    cfg = manifest["inference_config"]
    assert cfg["model"] == "qwen35-9b"
    assert cfg["enable_thinking"] is False
    assert cfg["temperature"] == 0.0
    assert cfg["max_tokens"] == 8192
    assert cfg["strict"] is True


def test_zero_variant_id_leakage_in_production_modules():
    """Verify production modules contain zero benchmark variant IDs."""
    root = Path(__file__).resolve().parents[3]
    pipeline_dir = root / "mujoco_scenes" / "functional_tamp_pipeline"

    production_files = [
        pipeline_dir / "role_semantic_ontology.py",
        pipeline_dir / "semantic_compiler.py",
        pipeline_dir / "vlm_spec_provider.py",
        pipeline_dir / "robot_capability_registry.py",
        pipeline_dir / "predicate_registry.py",
        pipeline_dir / "domains" / "kitchen.py",
        pipeline_dir / "domains" / "living_room.py",
        pipeline_dir / "domains" / "workshop.py",
    ]

    # Benchmark variant IDs
    variant_ids = [f"K{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 11)] + [f"W{i}" for i in range(1, 11)]

    for fpath in production_files:
        assert fpath.exists(), f"File {fpath} must exist"
        content = fpath.read_text(encoding="utf-8")
        for vid in variant_ids:
            # Check for variant ID as a standalone identifier / string
            assert f'"{vid}"' not in content, f"Leaked variant ID {vid} in {fpath.name}"
            assert f"'{vid}'" not in content, f"Leaked variant ID {vid} in {fpath.name}"


def test_zero_gt_provider_imports_in_online_path():
    """Verify online execution modules do not import GTSpecProvider."""
    root = Path(__file__).resolve().parents[3]
    pipeline_dir = root / "mujoco_scenes" / "functional_tamp_pipeline"

    online_files = [
        pipeline_dir / "vlm_spec_provider.py",
        pipeline_dir / "semantic_compiler.py",
        pipeline_dir / "relation_interpreter.py",
        pipeline_dir / "grounding.py",
        pipeline_dir / "planning.py",
        pipeline_dir / "executability.py",
        pipeline_dir / "search.py",
        pipeline_dir / "role_semantic_ontology.py",
        pipeline_dir / "robot_capability_registry.py",
    ]

    for fpath in online_files:
        content = fpath.read_text(encoding="utf-8")
        assert "GTSpecProvider" not in content, f"Forbidden GTSpecProvider found in online module {fpath.name}"
        assert "expected_plan" not in content, f"Forbidden expected_plan found in online module {fpath.name}"
        assert "expected_assignment" not in content, f"Forbidden expected_assignment found in online module {fpath.name}"
