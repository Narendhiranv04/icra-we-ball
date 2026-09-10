#!/usr/bin/env python3
"""Compare compiled G_F against the reference G_F, per domain.

Reports how much of the reference structure the compiled graph recovers and how
much it adds that the reference does not have. Used offline only, for
measurement; nothing here feeds the online pipeline.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

INSTR = {
    "kitchen": "Prepare and serve one coffee and one soup for each of two people. Make each coffee using coffee and water and stir it before serving. Serve each soup bowl with its own suitable eating utensil.",
    "living_room": "Prepare the living room for two people to enjoy refreshments while watching television. Provide each person with their own refreshment setting nearby, and place the entertainment control where it is accessible to both people.",
    "workshop": "Identify the compatible components required to complete the fastening at the marked workbench location, complete the fastening, and leave any reusable equipment used for the task safely on the workbench.",
}


def reference(domain):
    from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
    g = GTSpecProvider().provide(domain, INSTR[domain])
    roles = set(g.nodes)
    rels = {(r.subject_role, r.predicate, r.object_role) for r in (g.relations or ())}
    ops = {(o.tool_role, o.target_role) for o in (g.operation_groups or ())}
    return roles, rels, ops


def compiled(domain, raw):
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
        normalize_v3_live_document, validate_v3_live_contract, convert_v3_to_canonical_document)
    from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
    n, _ = normalize_v3_live_document(raw, domain=domain, task_instruction=INSTR[domain])
    validate_v3_live_contract(n, domain=None)
    c = convert_v3_to_canonical_document(n, domain=domain, task_instruction=INSTR[domain])
    g = compile_candidate_graph(domain, INSTR[domain], c)
    roles = set(g.nodes)
    rels = {(r.subject_role, r.predicate, r.object_role) for r in (g.relations or ())}
    ops = {(o.tool_role, o.target_role) for o in (g.operation_groups or ())}
    return roles, rels, ops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    stats = collections.defaultdict(lambda: collections.Counter())
    for f in sorted(glob.glob(str(args.root / "*/*/trial_*/raw_v3.json"))):
        domain = f.split(os.sep)[-4]
        if domain not in INSTR:
            continue
        try:
            raw = json.load(open(f))
            c_roles, c_rels, c_ops = compiled(domain, raw)
        except Exception:
            stats[domain]["uncompiled"] += 1
            continue
        g_roles, g_rels, g_ops = reference(domain)
        s = stats[domain]
        s["trials"] += 1
        s["role_hit"] += len(c_roles & g_roles)
        s["role_ref"] += len(g_roles)
        s["role_extra"] += len(c_roles - g_roles)
        s["rel_hit"] += len(c_rels & g_rels)
        s["rel_ref"] += len(g_rels)
        s["rel_extra"] += len(c_rels - g_rels)
        s["op_hit"] += len(c_ops & g_ops)
        s["op_ref"] += len(g_ops)
        s["op_extra"] += len(c_ops - g_ops)

    print(f"{'domain':12} {'trials':>6} {'roles':>16} {'relations':>16} {'operations':>16}")
    tot = collections.Counter()
    for domain, s in stats.items():
        tot.update(s)
        print(f"{domain:12} {s['trials']:6d} "
              f"{s['role_hit']:5d}/{s['role_ref']:<5d}+{s['role_extra']:<4d} "
              f"{s['rel_hit']:5d}/{s['rel_ref']:<5d}+{s['rel_extra']:<4d} "
              f"{s['op_hit']:5d}/{s['op_ref']:<5d}+{s['op_extra']:<4d}")
    print(f"\n{'TOTAL':12} {tot['trials']:6d} "
          f"{tot['role_hit']:5d}/{tot['role_ref']:<5d}+{tot['role_extra']:<4d} "
          f"{tot['rel_hit']:5d}/{tot['rel_ref']:<5d}+{tot['rel_extra']:<4d} "
          f"{tot['op_hit']:5d}/{tot['op_ref']:<5d}+{tot['op_extra']:<4d}")
    print("\nread as: recovered/reference +extra")
    for k in ("role", "rel", "op"):
        r = 100 * tot[f"{k}_hit"] / max(tot[f"{k}_ref"], 1)
        print(f"  {k:4} recall {r:5.1f}%   extras {tot[f'{k}_extra']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
