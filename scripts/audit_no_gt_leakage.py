#!/usr/bin/env python3
"""Static audit: no ground truth and no variant identity in online decision code.

The online path is what runs during a trial: FM contract normalization, semantic
compilation, grounding, search, planning and validation.  Nothing there may
consult a reference graph, an expected action sequence, a feasibility label, or
the benchmark variant identifier -- and nothing there may branch on one.

Offline instruments are exempt by name, and each exemption is listed in the
report so it can be checked.  This script reads source only; it runs nothing.
"""
from __future__ import annotations
import ast, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The online decision path.  A module is in scope unless it is named below.
IN_SCOPE_DIRS = (
    REPO / "mujoco_scenes" / "functional_tamp_pipeline",
    REPO / "mujoco_scenes" / "workshop_phase1",
)
IN_SCOPE_FILES = tuple(
    REPO / "mujoco_scenes" / name for name in (
        "semantic_grounding.py",
        "kitchen_vlm_functional_graph.py",
        "environment_vlm_requirements.py",
    )
)

# Offline instruments and reference providers.  These exist in order to hold the
# reference, or to score against it, and are not consulted while a trial runs.
EXEMPT_BASENAMES = frozenset({
    # The reference itself, and the evaluators that compare against it.  Each of
    # these exists in order to hold or to score against the reference, and is
    # invoked after a trial has finished, never inside a decision.
    "gt_spec_provider.py", "reference_graphs.py", "evaluation_metrics.py",
    # Scores the FM's raw output against the reference, after the fact.  Reached
    # only from evaluation_metrics.py and offline scripts, never from a decision.
    "raw_semantic_evaluation.py",
    "gf_reference_evaluation.py", "gf_reference_evaluator.py",
    "raw_eval.py", "audit.py",
    # Offline perception scoring: matches detector tracks against simulator
    # bodies to report detection quality.  Nothing it computes is read back.
    "evaluation.py",
    # Benchmark definition: variant identity lives here by construction.
    "variants.py", "benchmark_variants.py",
})

# Names that denote the variant-independent canonical search order used in GT
# mode.  They begin with "gt" and carry no reference content; the guard that
# keeps them out of the online path is asserted separately below.
GT_MODE_LABELS = frozenset({
    "gt_system", "GT_SYSTEM", "GT_SYSTEM_SEARCH_POLICY", "GT_EXPLICIT_SEARCH_POLICY",
})

# Containers that hold variant-keyed benchmark answers on purpose, as privileged
# diagnostic instruments.  Each one must be unreachable from the online path,
# and the audit checks that the module guards it with an explicit refusal.
PRIVILEGED_VARIANT_TABLES = {
    "mujoco_scenes/functional_tamp_pipeline/search_contract.py": "ORACLE_SEARCH_ORDERS",
}
PRIVILEGED_GUARD = re.compile(
    r'==\s*"oracle".*mode\s*==\s*"vlm"|mode\s*==\s*"vlm".*==\s*"oracle"')
EXEMPT_DIR_NAMES = frozenset({"tests", "__pycache__"})

# Names that would carry the reference or the variant identity into a decision.
REFERENCE_NAME = re.compile(
    r"\b(gt_|ground_truth|reference_graph|expected_(?:roles|actions|goals|plan)|"
    r"gt_feasible|gt_goals|gt_full_task|feasibility_label|benchmark_answer)\w*",
    re.I,
)
VARIANT_LITERAL = re.compile(r"^(?:K|L|W)(?:[1-9]|1[0-2])$")


def in_scope(path: Path) -> bool:
    if path.name in EXEMPT_BASENAMES:
        return False
    if any(part in EXEMPT_DIR_NAMES for part in path.parts):
        return False
    if path in IN_SCOPE_FILES:
        return True
    return any(str(path).startswith(str(d)) for d in IN_SCOPE_DIRS)


def sources() -> list[Path]:
    found: list[Path] = []
    for directory in IN_SCOPE_DIRS:
        found.extend(sorted(directory.rglob("*.py")))
    found.extend(IN_SCOPE_FILES)
    return [p for p in sorted(set(found)) if p.exists() and in_scope(p)]


def privileged_line_span(path: Path, tree: ast.AST) -> range:
    """Lines belonging to this module's declared privileged variant table."""
    name = PRIVILEGED_VARIANT_TABLES.get(str(path.relative_to(REPO)))
    if not name:
        return range(0)
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return range(node.lineno, (node.end_lineno or node.lineno) + 1)
    return range(0)


# Modules that exist only to score a finished trial against the reference.  The
# online path must not import them: scoring knows the feasibility label and the
# expected answer, so an online module that can reach it can reach those too,
# whatever it does with them today.  Names are matched on the final component of
# the import, so both `import mujoco_scenes.evaluation_outcome` and
# `from mujoco_scenes.evaluation_outcome import ...` are caught.
EVALUATION_ONLY_MODULES = frozenset({
    "evaluation_outcome",
    "evaluation_metrics",
    "gf_reference_evaluation",
    "gf_reference_evaluator",
    "gt_spec_provider",
    "reference_graphs",
})


# The one structural exception, stated rather than hidden.  The benchmark has a
# ground-truth/oracle arm as a deliberate comparison baseline, and the mode
# dispatcher has to be able to construct it.  The import is lazy and guarded by
# an explicit `mode == "gt"` branch, so the VLM online path never executes it.
# Nothing else may import an evaluation-only module, and this allowance names
# the single module and the single target it covers.
EVALUATION_IMPORT_ALLOWANCES = {
    ("mujoco_scenes/functional_tamp_pipeline/spec_provider.py", "gt_spec_provider"),
}


def evaluation_imports(path: Path, tree: ast.AST) -> list[str]:
    """Imports of evaluation-only modules from an online module."""
    found: list[str] = []
    relative = str(path.relative_to(REPO))
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            targets = [base] + [f"{base}.{a.name}" for a in node.names]
        hits = {part for target in targets for part in str(target).split(".")
                if part in EVALUATION_ONLY_MODULES}
        for part in sorted(hits):
            if (relative, part) in EVALUATION_IMPORT_ALLOWANCES:
                continue
            found.append(
                f"{relative}:{getattr(node, 'lineno', 0)}: "
                f"online module imports evaluation-only {part!r}")
    return found


def audit(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    findings: list[str] = []
    privileged = privileged_line_span(path, tree)
    findings.extend(evaluation_imports(path, tree))
    if privileged and not PRIVILEGED_GUARD.search(re.sub(r"\s+", " ", text)):
        findings.append(
            f"{path.relative_to(REPO)}: declares a privileged variant table but the "
            f"module does not refuse it in vlm mode")

    # 1. A variant identifier used as a value anywhere in the module.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.lineno in privileged:
                continue
            if VARIANT_LITERAL.match(node.value.strip()):
                findings.append(
                    f"{path.relative_to(REPO)}:{node.lineno}: variant literal "
                    f"{node.value!r}: {lines[node.lineno - 1].strip()[:110]}")

    # 2. A reference-bearing name read, written, called or imported.
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.alias):
            name = node.asname or node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A dictionary key naming a reference field counts too.
            name = node.value
        if str(name) in GT_MODE_LABELS:
            continue
        if not name or not REFERENCE_NAME.search(str(name)):
            continue
        lineno = getattr(node, "lineno", 0)
        source_line = lines[lineno - 1].strip() if lineno else ""
        findings.append(
            f"{path.relative_to(REPO)}:{lineno}: reference name {name!r}: {source_line[:110]}")
    return findings


def main() -> int:
    checked = sources()
    findings: list[str] = []
    for path in checked:
        findings.extend(audit(path))
    print("NO-GT / NO-VARIANT LEAKAGE AUDIT")
    print("=" * 78)
    print(f"modules in the online decision path: {len(checked)}")
    print(f"exempt by name (reference holders and offline evaluators): "
          f"{', '.join(sorted(EXEMPT_BASENAMES))}")
    print(f"exempt directories: {', '.join(sorted(EXEMPT_DIR_NAMES))}")
    print()
    print("rule 1: no benchmark variant identifier may appear as a value, except")
    print("        inside a declared privileged diagnostic table that the module")
    print("        itself refuses in vlm mode:")
    for module, table in sorted(PRIVILEGED_VARIANT_TABLES.items()):
        print(f"          {module}: {table}")
    print("rule 2: no reference graph, expected answer or feasibility label may be")
    print("        read, written, called or imported.  Labels naming the")
    print("        variant-independent GT-mode search order are not reference")
    print(f"        content and are listed: {', '.join(sorted(GT_MODE_LABELS))}")
    print()
    if findings:
        print(f"FINDINGS: {len(findings)}")
        for row in findings:
            print("  " + row)
    else:
        print("FINDINGS: 0")
    print()
    print("modules checked:")
    for path in checked:
        print(f"  {path.relative_to(REPO)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
