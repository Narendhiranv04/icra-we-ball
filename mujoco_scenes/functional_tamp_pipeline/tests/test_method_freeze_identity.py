"""The freeze artifact must not claim something it cannot know.

A tracked file cannot contain the hash of the commit that contains it: writing
it dirties the tree, committing produces a new SHA, and the recorded one is
immediately stale.  The previous version recorded `sha` and `clean` anyway and
shipped saying `clean: false` against a SHA two commits behind the branch.

Behavioural identity lives here; repository identity is the annotated tag.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FREEZE = REPO / "METHOD_FREEZE.json"
SCRIPT = REPO / "scripts" / "freeze_method_identity.py"


def _freeze() -> dict:
    return json.loads(FREEZE.read_text())


def test_it_does_not_claim_a_commit_sha_or_cleanliness():
    data = _freeze()
    assert "git" not in data, "the old self-referential git block is back"
    flat = json.dumps(data)
    assert '"clean"' not in flat, (
        "a generated tracked artifact cannot honestly assert tree cleanliness: "
        "writing it is what makes the tree dirty")
    assert '"sha"' not in flat or "base_code_sha" in flat


def test_provenance_is_named_so_it_cannot_be_misread():
    prov = _freeze()["provenance"]
    assert "base_code_sha" in prov, "the generating commit must be named as such"
    assert prov["release_tag"] == "fm-tamp-final-experiment-v1"
    assert "cannot contain the hash of the commit that contains it" in prov["note"]


def test_the_behavioural_hash_ignores_provenance():
    """A method that behaves identically must have one identity.

    Regenerating from a different commit changes base_code_sha; it must not
    change what identifies the method.
    """
    data = _freeze()
    import hashlib
    behavioural = {k: v for k, v in data.items()
                   if k not in ("provenance", "behavioural_identity_sha256")}
    expected = hashlib.sha256(json.dumps(behavioural, sort_keys=True).encode()).hexdigest()
    assert data["behavioural_identity_sha256"] == expected


def test_every_behaviour_deciding_field_is_present():
    data = _freeze()
    for section in ("semantic_contract", "perception", "determinism", "sampler",
                    "evaluator", "benchmark_grid"):
        assert section in data, f"{section} missing from the behavioural identity"
    assert data["benchmark_grid"]["variant_count"] == 32
    assert data["determinism"]["reproducible_child_env"]["PYTHONHASHSEED"] == "0"


def test_the_artifact_matches_the_current_tree():
    """Regenerating must reproduce the committed file byte for byte."""
    before = FREEZE.read_text()
    subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, cwd=str(REPO))
    after = FREEZE.read_text()
    assert before == after, (
        "METHOD_FREEZE.json is stale; regenerate and commit it")
