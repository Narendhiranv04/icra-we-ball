#!/usr/bin/env python3
"""Compare each produced action plan against the expected GT action sequence.

Offline scoring only.  Reads the catalogue in EXPECTED_GT_ACTIONS/, which is
generated from the benchmark definition, and the action_plan.json each trial
produced.  Nothing here runs during a trial and nothing it computes is fed back
into the pipeline.

A produced plan binds observed instance ids ("object_0003") where the catalogue
names scene bodies ("workshop_medium_phillips_screw").  The comparison is
therefore structural: the operator sequence must match exactly, and the
argument *positions* must agree on every non-object argument (regions, fixed
anchors) while object arguments must be used consistently -- the same observed
instance wherever the catalogue names the same body, and different instances
wherever it names different bodies.  That is what it means for the plan to be
the GT plan, without pretending the runtime knows the body names.
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CATALOGUE = REPO / "EXPECTED_GT_ACTIONS"

# Arguments that name a place the runtime and the catalogue share a name for.
SHARED_NAME = re.compile(
    r"^(?:[A-Z][A-Z0-9_]+|workshop_frame_joint|repair_target)$")
OBSERVED_INSTANCE = re.compile(r"^object_\d+$")
# Opening a storage region to look inside it is inspection, not task progress;
# the planner records those separately and the catalogue lists them inline.
EXPLORATORY = {"OPEN"}


def load_expected(domain: str, variant: str):
    path = CATALOGUE / domain / variant / "expected_gt_actions.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    actions = data.get("actions") or data.get("expected_actions") or []
    return data, [a for a in actions if str(a.get("operator")) not in EXPLORATORY]


def load_produced(run_dir: Path):
    path = run_dir / "action_plan.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [a for a in (data.get("actions") or [])
            if str(a.get("operator")) not in EXPLORATORY]


def arguments(action):
    value = action.get("arguments")
    if isinstance(value, list):
        return [str(x) for x in value]
    return [] if value is None else [str(value)]


def compare(expected, produced):
    """Structural equality, with a consistent body -> instance correspondence."""
    if len(expected) != len(produced):
        return False, f"length {len(produced)} != expected {len(expected)}"
    body_to_instance: dict[str, str] = {}
    instance_to_body: dict[str, str] = {}
    for index, (want, got) in enumerate(zip(expected, produced), start=1):
        if str(want.get("operator")) != str(got.get("operator")):
            return False, (f"step {index}: operator {got.get('operator')} != "
                           f"{want.get('operator')}")
        want_args, got_args = arguments(want), arguments(got)
        if len(want_args) != len(got_args):
            return False, f"step {index}: arity {len(got_args)} != {len(want_args)}"
        for position, (w, g) in enumerate(zip(want_args, got_args)):
            if SHARED_NAME.match(w):
                if w != g:
                    return False, f"step {index} arg {position}: {g} != {w}"
                continue
            if not OBSERVED_INSTANCE.match(g):
                return False, f"step {index} arg {position}: {g} is not an observed instance"
            if body_to_instance.setdefault(w, g) != g:
                return False, (f"step {index} arg {position}: {w} was bound to "
                               f"{body_to_instance[w]} and is now {g}")
            if instance_to_body.setdefault(g, w) != w:
                return False, (f"step {index} arg {position}: {g} stands for both "
                               f"{instance_to_body[g]} and {w}")
    return True, "exact structural match"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(args.replay_root)
    rows = []
    for run_dir in sorted(root.glob("trial_*/*/*/vlm")):
        domain, variant, trial = run_dir.parents[1].name, run_dir.parent.name, run_dir.parents[2].name
        domain, variant = run_dir.parents[1].name, run_dir.parent.name
        trial = run_dir.parents[2].name
        loaded = load_expected(domain, variant)
        produced = load_produced(run_dir)
        if loaded is None:
            rows.append({"domain": domain, "variant": variant, "trial": trial,
                         "status": "NO_CATALOGUE_ENTRY"})
            continue
        meta, expected = loaded
        if produced is None:
            rows.append({"domain": domain, "variant": variant, "trial": trial,
                         "intended_outcome": meta.get("intended_outcome"),
                         "expected_steps": len(expected), "produced_steps": None,
                         "status": "NO_PLAN_PRODUCED", "detail": ""})
            continue
        ok, detail = compare(expected, produced)
        rows.append({"domain": domain, "variant": variant, "trial": trial,
                     "intended_outcome": meta.get("intended_outcome"),
                     "expected_steps": len(expected), "produced_steps": len(produced),
                     "status": "EXACT_GT_ACTION_SEQUENCE" if ok else "DIFFERS",
                     "detail": detail})
    rows.sort(key=lambda r: (r["domain"], r["variant"], r["trial"]))
    exact = sum(1 for r in rows if r["status"] == "EXACT_GT_ACTION_SEQUENCE")
    planned = sum(1 for r in rows if r.get("produced_steps"))
    print(f"trials with a produced plan: {planned}")
    print(f"plans matching the GT action sequence exactly: {exact}")
    print()
    for r in rows:
        if not r.get("produced_steps"):
            continue
        mark = "OK " if r["status"] == "EXACT_GT_ACTION_SEQUENCE" else "   "
        print(f"  {mark}{r['domain'][:4]:4s} {r['variant']:4s}/{r['trial'][-2:]} "
              f"{r['produced_steps']}/{r['expected_steps']} steps  {r['detail'][:90]}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps({
            "trials_with_plan": planned, "exact_gt_action_sequence": exact, "rows": rows,
        }, indent=2) + "\n")
        keys = ["domain", "variant", "trial", "intended_outcome", "expected_steps",
                "produced_steps", "status", "detail"]
        with out.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nwrote {out.with_suffix('.json')} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
