#!/usr/bin/env python3
"""Analyze a packaged 3 x 32 V3 development distribution.

Reports stage progression, per-domain and per-variant consistency, and a
frequency-ranked taxonomy of recurring raw FM semantic patterns.

Every detector here is ground-truth free.  None of them reads an expected role
list, an expected action sequence, or a feasibility label.  They inspect only
the FM contract itself, the deterministic error codes the runtime produced, and
-- for cross-variant divergence -- the fact that all variants of one domain
share a single task instruction, so the *required* task semantics ought to be
invariant across them.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

DOMAINS = {
    "kitchen": [f"K{i}" for i in range(1, 13)],
    "living_room": [f"L{i}" for i in range(1, 11)],
    "workshop": [f"W{i}" for i in range(1, 11)],
}

STAGES = ["strict_v3_valid", "task_valid", "graph_compiled",
          "grounding_reached", "complete_grounding", "astar_reached", "success"]

# ---------------------------------------------------------------- detectors
_NON_PHYSICAL = re.compile(
    r"\b(identify|determine|verify|check|assess|decide|choose|select|ensure|"
    r"confirm|inspect|evaluate|plan|consider)\b", re.I)
_CURRENT_STATE = re.compile(
    r"\b(currently|initially|at present|already|is located|is placed|is sitting|"
    r"stored on|resting on|starts? on|present location)\b", re.I)
_COMPOSITE = re.compile(r"\b(and then|then|after that|,\s*and\b|\band\b.*\band\b)", re.I)
_SENTENCE = re.compile(r"^[A-Z].*\.\s*$")


def _contract(raw):
    return (raw or {}).get("task_contract") or {}


def classify_raw(raw):
    """Return the set of generic semantic patterns present in one raw contract."""
    tags = set()
    if raw is None:
        tags.add("unparseable_response")
        return tags
    tc = _contract(raw)
    roles = tc.get("functional_roles") or []
    rels = tc.get("functional_relations") or []
    ops = tc.get("operation_pairings") or []
    guidance = (raw or {}).get("observation_guidance") or {}

    if raw.get("status") == "UNSUPPORTED":
        tags.add("declared_unsupported")
        if not roles:
            tags.add("empty_contract")
    if not roles:
        tags.add("no_roles_declared")

    declared = {r.get("id") for r in roles}

    # B. function / count / policy consistency
    for r in roles:
        cnt, pol = r.get("required_count"), r.get("binding_policy")
        if cnt and cnt > 1 and pol == "SHARED":
            tags.add("wrong_binding_policy_shared_with_multiple_instances")
        if cnt and cnt > 1 and pol == "REUSABLE":
            tags.add("reusable_with_multiple_instances")
        fn = (r.get("function") or "")
        if _CURRENT_STATE.search(fn):
            tags.add("current_state_leakage_in_role_function")
        if _NON_PHYSICAL.search(fn):
            tags.add("non_physical_role_function")
    if len(declared) != len(roles):
        tags.add("duplicate_role_id")

    # C. relation quality
    for rel in rels:
        phrase = rel.get("relation") or ""
        parts = rel.get("participant_roles") or []
        if _CURRENT_STATE.search(phrase):
            tags.add("incidental_current_state_relation")
        if _SENTENCE.match(phrase.strip()) or len(phrase.split()) > 8:
            tags.add("relation_phrased_as_sentence")
        if any(p not in declared for p in parts):
            tags.add("relation_references_undeclared_role")
        if len(set(parts)) != len(parts):
            tags.add("duplicate_relation_participant")

    # D. operation quality
    for op in ops:
        phrase = op.get("operation") or ""
        parts = op.get("participant_roles") or []
        if _NON_PHYSICAL.search(phrase):
            tags.add("non_physical_directive_operation")
        if _COMPOSITE.search(phrase):
            tags.add("composite_operation")
        if any(p not in declared for p in parts):
            tags.add("operation_references_undeclared_role")
        if len(set(parts)) != len(parts):
            tags.add("duplicate_operation_participant")
        if _SENTENCE.match(phrase.strip()) or len(phrase.split()) > 8:
            tags.add("operation_phrased_as_sentence")

    # E. observation guidance
    vis = guidance.get("visible_candidates_per_role") or {}
    if isinstance(vis, dict) and any(k not in declared for k in vis):
        tags.add("guidance_references_undeclared_role")
    regions = guidance.get("inspectable_regions") or []
    order = guidance.get("inspection_order") or []
    rids = [r.get("id") for r in regions if isinstance(r, dict)]
    if sorted(x for x in order if isinstance(x, str)) != sorted(x for x in rids if x):
        tags.add("inspection_order_mismatch")
    for reg in regions:
        if isinstance(reg, dict) and re.search(
                r"\b(contains|holding|inside is|will find|has a|stores the)\b",
                reg.get("reason", ""), re.I):
            tags.add("claims_hidden_contents")
    return tags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = json.loads((args.root / "stage_progression_all.json").read_text())
    by_key = {(r["domain"], r["variant"], r["trial"]): r for r in rows}

    report = {"total_trials": len(rows)}

    # ---- overall + per-domain stage progression
    def stage_counts(subset):
        return {s: sum(1 for r in subset if r.get(s)) for s in STAGES}

    report["overall"] = stage_counts(rows)
    report["per_domain"] = {}
    for dom in DOMAINS:
        sub = [r for r in rows if r["domain"] == dom]
        d = stage_counts(sub)
        d["trials"] = len(sub)
        d["outcome_categories"] = dict(collections.Counter(
            r.get("outcome_category") for r in sub))
        report["per_domain"][dom] = d

    report["outcome_categories"] = dict(collections.Counter(
        r.get("outcome_category") for r in rows))

    # ---- per-variant consistency (x of 3)
    per_variant = {}
    for dom, variants in DOMAINS.items():
        for v in variants:
            trials = [r for r in rows if r["domain"] == dom and r["variant"] == v]
            if not trials:
                continue
            per_variant[v] = {
                "domain": dom, "trials": len(trials),
                **{s: sum(1 for r in trials if r.get(s)) for s in STAGES},
                "outcomes": [r.get("outcome_category") for r in trials],
            }
    report["per_variant"] = per_variant

    # ---- consistency histogram: how many variants are k-of-3 at each stage
    report["consistency_histogram"] = {
        s: dict(collections.Counter(pv[s] for pv in per_variant.values()))
        for s in STAGES
    }

    # ---- deterministic error-code frequency
    codes = collections.Counter()
    for r in rows:
        for field in ("strict_v3_error", "task_error", "compile_error"):
            val = r.get(field)
            if val:
                m = re.search(r"([A-Z][A-Z_]{6,})", str(val))
                codes[m.group(1) if m else str(val).split(":")[0]] += 1
    report["error_codes"] = dict(codes.most_common())

    # ---- raw FM semantic pattern taxonomy
    tags = collections.Counter()
    tags_by_domain = collections.defaultdict(collections.Counter)
    for dom, variants in DOMAINS.items():
        for v in variants:
            for t in ("trial_01", "trial_02", "trial_03"):
                p = args.root / dom / v / t / "raw_v3.json"
                if not p.exists():
                    continue
                try:
                    raw = json.loads(p.read_text())
                except Exception:
                    raw = None
                for tag in classify_raw(raw):
                    tags[tag] += 1
                    tags_by_domain[dom][tag] += 1
    report["raw_semantic_patterns"] = dict(tags.most_common())
    report["raw_semantic_patterns_by_domain"] = {
        d: dict(c.most_common()) for d, c in tags_by_domain.items()}

    # ---- within-domain role divergence (instruction is fixed per domain)
    divergence = {}
    for dom, variants in DOMAINS.items():
        counts = []
        for v in variants:
            for t in ("trial_01", "trial_02", "trial_03"):
                p = args.root / dom / v / t / "raw_v3.json"
                if p.exists():
                    try:
                        raw = json.loads(p.read_text())
                    except Exception:
                        continue
                    counts.append(len(_contract(raw).get("functional_roles") or []))
        if counts:
            divergence[dom] = {
                "role_count_distribution": dict(sorted(collections.Counter(counts).items())),
                "min": min(counts), "max": max(counts),
                "mean": round(sum(counts) / len(counts), 2),
            }
    report["within_domain_role_count_divergence"] = divergence

    out = args.out or (args.root / "distribution_analysis.json")
    out.write_text(json.dumps(report, indent=2) + "\n")

    # ---- human-readable summary
    n = len(rows) or 1
    print(f"=== {len(rows)} trials ===")
    print("\nSTAGE PROGRESSION (overall)")
    for s in STAGES:
        c = report["overall"][s]
        print(f"  {s:22} {c:3d}/{len(rows)}  {100*c/n:5.1f}%")
    print("\nPER DOMAIN")
    for dom, d in report["per_domain"].items():
        tot = d["trials"] or 1
        print(f"  {dom:12} n={d['trials']:3d} " + " ".join(
            f"{s.split('_')[0][:6]}={d[s]:3d}" for s in STAGES))
    print("\nOUTCOME CATEGORIES")
    for k, v in sorted(report["outcome_categories"].items(), key=lambda x: -x[1]):
        print(f"  {str(k):34} {v:3d}  {100*v/n:5.1f}%")
    print("\nTOP DETERMINISTIC ERROR CODES")
    for k, v in list(report["error_codes"].items())[:15]:
        print(f"  {v:3d}  {k[:70]}")
    print("\nRAW FM SEMANTIC PATTERNS (ranked)")
    for k, v in list(report["raw_semantic_patterns"].items())[:25]:
        print(f"  {v:3d}  {k}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
