"""The same contract must compile to the same thing, every time.

Two ways determinism can be lost quietly.  One is deliberate randomness used to
settle something the wording left open, which would make an ambiguity look
resolved and would not be reproducible.  The other is accidental: iterating a
set and letting the hash seed decide the order, which is stable within one
process and moves between runs, so it survives every same-process test.

Both are checked here: repeated compilation in one process, and compilation in
child processes started with different PYTHONHASHSEED values.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document,
    normalize_and_validate_v3_contract,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph


REPO = Path(__file__).resolve().parents[3]

WORKSHOP = ("Identify the compatible components required to complete the fastening at the "
            "marked workbench location, complete the fastening, and leave any reusable "
            "equipment used for the task safely on the workbench.")


def _contract():
    def role(rid, function, **kw):
        base = {"id": rid, "entity_kind": "OBJECT", "function": function, "description": "",
                "required_count": 1, "binding_policy": "REUSABLE",
                "candidate_categories": [], "required_properties": []}
        base.update(kw)
        return base

    return {
        "schema_version": 3, "status": "SUPPORTED", "task_summary": "fasten and tidy up",
        "task_contract": {
            "functional_roles": [
                role("tool", "instrument used to drive the fastener",
                     candidate_categories=["screwdriver", "wrench"]),
                role("part", "the component to be secured at the marked place",
                     binding_policy="DISTINCT", candidate_categories=["screw", "bolt"]),
                role("site", "the marked place on the bench where the fastening must occur",
                     entity_kind="REGION", binding_policy="SHARED",
                     candidate_categories=["marked spot"]),
                role("cupboard", "a closed structure that may hold the part or the tool",
                     entity_kind="REGION", binding_policy="SHARED",
                     candidate_categories=["cabinet", "drawer"]),
            ],
            "functional_relations": [
                {"id": "fit", "relation": "the part must fit the marked place",
                 "participant_roles": ["part", "site"], "required": True},
                {"id": "engage", "relation": "the tool engages the part",
                 "participant_roles": ["tool", "part"], "required": True},
            ],
            "operation_pairings": [
                {"id": "assemble", "operation": "fasten the part at the marked place",
                 "participant_roles": ["tool", "part", "site"], "operation_count": 1},
            ],
        },
        "observation_guidance": {"visible_candidates_per_role": {},
                                 "inspectable_regions": [], "inspection_order": []},
        "unsupported_reason": "",
    }


def _fingerprint(raw, domain, instruction):
    normalized, _ = normalize_and_validate_v3_contract(
        raw, domain=domain, task_instruction=instruction)
    canonical = convert_v3_to_canonical_document(
        normalized, domain=domain, task_instruction=instruction)
    graph = compile_candidate_graph(domain, instruction, canonical)
    trace = (graph.metadata or {})["canonicalization_trace"]
    return json.dumps({
        "nodes": {name: [role.entity_kind, role.count, role.binding_policy,
                         list(role.canonical_role_candidates or ())]
                  for name, role in sorted(graph.nodes.items())},
        "groups": [[g.id, g.function, g.tool_role, g.target_role, g.context_role,
                    g.required_target_count, list(g.physical_preconditions or ())]
                   for g in graph.operation_groups or ()],
        "relations": [[r.subject_role, r.predicate, r.object_role]
                      for r in graph.relations or ()],
        "executable": graph.online_executable_contract_complete,
        "unresolved_relations": [
            str(row.get("raw_phrase") or row.get("relation"))
            for row in trace.get("unresolved_required_relations", [])],
        "unresolved_operations": [
            str(row.get("operation")) for row in trace.get("unresolved_required_operations", [])],
        "context_only": [str(row.get("raw_role", {}).get("id"))
                         for row in trace.get("context_only_roles", [])],
    }, sort_keys=True, default=str)


def test_repeated_compilation_in_one_process_is_identical():
    first = _fingerprint(_contract(), "workshop", WORKSHOP)
    for _ in range(4):
        assert _fingerprint(_contract(), "workshop", WORKSHOP) == first


CHILD = """
import json, sys
sys.path.insert(0, %(repo)r)
from mujoco_scenes.functional_tamp_pipeline.tests.test_semantic_determinism import (
    _contract, _fingerprint, WORKSHOP)
print(_fingerprint(_contract(), "workshop", WORKSHOP))
"""


def test_compilation_does_not_depend_on_the_hash_seed():
    """A set iterated without sorting is stable in one process and not between."""
    script = CHILD % {"repo": str(REPO)}
    seen = set()
    for seed in ("0", "1", "7", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO))
        out = subprocess.run([sys.executable, "-c", script], env=env, cwd=str(REPO),
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"{len(seen)} distinct compilations across hash seeds"


def test_no_unseeded_randomness_reaches_the_semantic_front_half():
    """Randomness is allowed only through an explicitly seeded generator.

    The module-level functions of `random` share one process-wide generator
    that nobody in this pipeline seeds, so a call to any of them would make a
    decision that cannot be reproduced.  Constructing `random.Random(seed)` is
    fine and is what the seeded search-policy ablation does; string literals
    and CLI help mentioning the word are not calls and are not flagged.
    """
    package = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            value = node.func.value
            root = value.attr if isinstance(value, ast.Attribute) else getattr(value, "id", "")
            if root != "random":
                continue
            if node.func.attr == "Random":
                continue
            offenders.append(f"{path.name}:{node.lineno}: random.{node.func.attr}(...)")
    assert not offenders, offenders
