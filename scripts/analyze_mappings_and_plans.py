#!/usr/bin/env python3
"""Report the role-to-object mapping and action sequence for every replayed trial.

Flags mappings and plans that are internally illogical, so that repairs target a
real inconsistency rather than being forced to make a number move.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

FEASIBLE = {
    "kitchen": {f"K{i}" for i in range(1, 7)},
    "living_room": {f"L{i}" for i in range(1, 7)},
    "workshop": {f"W{i}" for i in range(1, 9)},
}


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _as_list(assigned):
    if assigned is None:
        return []
    return list(assigned) if isinstance(assigned, (list, tuple)) else [assigned]


def audit(assignment, graph, plan, validation):
    """Internal-consistency checks on a mapping and the plan built from it."""
    problems = []
    nodes = (graph or {}).get("nodes") or {}

    # 1. One object serving two roles that each demand their own instance.
    owner = {}
    for role, assigned in (assignment or {}).items():
        for obj in _as_list(assigned):
            owner.setdefault(obj, []).append(role)
    for obj, roles in owner.items():
        if len(roles) > 1:
            shared_ok = all(
                (nodes.get(r) or {}).get("binding_policy") in {"SHARED", "REUSABLE"}
                for r in roles
            )
            if not shared_ok:
                problems.append(f"object {obj} bound to distinct roles {sorted(roles)}")

    # 2. A role given fewer instances than it requires.
    for role, node in nodes.items():
        need = node.get("minimum_count") or node.get("count") or 1
        got = len(_as_list((assignment or {}).get(role)))
        if got and got < need:
            problems.append(f"role {role} needs {need} instance(s), bound {got}")

    # 3. The plan reusing one object where the role demands distinct instances.
    used = collections.Counter()
    for a in plan or []:
        for arg in a.get("arguments", []):
            used[arg] += 1
    for role, node in nodes.items():
        if node.get("binding_policy") == "DISTINCT":
            objs = _as_list((assignment or {}).get(role))
            if len(set(objs)) > 1:
                placed = {o for o in objs if used.get(o)}
                if len(placed) == 1 and used[next(iter(placed))] > 2:
                    problems.append(
                        f"role {role} is DISTINCT but the plan reuses {next(iter(placed))}")

    # 4. Goals the plan did not reach.
    missing = (validation or {}).get("missing_goals") or []
    if missing:
        problems.append(f"unmet goals: {missing}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--trial", default="trial_01")
    ap.add_argument("--only", default=None, help="comma separated variants")
    args = ap.parse_args()

    keep = set(args.only.split(",")) if args.only else None
    rows = []
    for domain in ("kitchen", "living_room", "workshop"):
        base = args.root / args.trial / domain
        if not base.exists():
            continue
        for vdir in sorted(base.iterdir(), key=lambda p: (p.name[0], int(p.name[1:]))):
            v = vdir.name
            if keep and v not in keep:
                continue
            d = vdir / "vlm"
            result = _load(d / "result.json") or {}
            grounding = _load(d / "graph_grounding_result.json") or {}
            graph = _load(d / "functional_requirement_graph.json") or {}
            plan = result.get("candidate_plan") or []
            validation = (_load(d / "action_plan.json") or {}).get("validation")
            rows.append({
                "domain": domain, "variant": v,
                "feasible": v in FEASIBLE[domain],
                "status": result.get("status"),
                "outcome": result.get("outcome_category"),
                "assignment": grounding.get("assignment"),
                "roles": sorted((graph.get("nodes") or {})),
                "ops": [(o.get("id"), o.get("capability_id"))
                        for o in (graph.get("operation_groups") or [])],
                "plan": [f"{a['operator']}({','.join(a['arguments'])})" for a in plan],
                "problems": audit(grounding.get("assignment"), graph, plan, validation),
            })

    for r in rows:
        mark = "OK " if r["status"] == "ACTION_SEQUENCE_READY" else "   "
        print(f"{mark}{r['variant']:4} [{'feasible' if r['feasible'] else 'infeasible'}] {r['status']}")
        print(f"     roles      : {r['roles']}")
        print(f"     operations : {r['ops']}")
        print(f"     mapping    : {r['assignment']}")
        if r["plan"]:
            print(f"     plan       : {' -> '.join(r['plan'])}")
        else:
            print("     plan       : (none)")
        for p in r["problems"]:
            print(f"     ! {p}")
        print()

    print("=" * 70)
    print(f"{len(rows)} variants | success "
          f"{sum(1 for r in rows if r['status'] == 'ACTION_SEQUENCE_READY')}")
    counts = collections.Counter(p.split()[0] for r in rows for p in r["problems"])
    for k, v in counts.most_common():
        print(f"   {v:3d}  problems starting '{k}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
