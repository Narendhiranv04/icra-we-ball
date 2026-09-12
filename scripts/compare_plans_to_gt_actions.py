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


# Each domain writes its plan under a different name and depth.
PLAN_PATHS = (
    "action_plan.json",
    "action_sequence/action_plan.json",
    "action_sequence/plan.json",
    "plan.json",
)


def load_produced(run_dir: Path):
    for relative in PLAN_PATHS:
        path = run_dir / relative
        if path.exists():
            data = json.loads(path.read_text())
            actions = data.get("actions") if isinstance(data, dict) else data
            return [a for a in (actions or [])
                    if str(a.get("operator")) not in EXPLORATORY]
    return None


def _argument(value):
    """One argument as a plain name.

    Living Room writes arguments as {"object": "object_0002"}; the other two
    write bare strings.  Both mean the same thing here.
    """
    if isinstance(value, dict):
        for key in ("object", "instance", "instance_id", "id", "name", "region", "target"):
            if key in value:
                return str(value[key])
        return json.dumps(value, sort_keys=True)
    return str(value)


# Living Room stores an action's arguments as a mapping rather than a list.
# The catalogue lists them positionally, so the mapping is read in the order the
# operators take: what is moved, then where it goes.
_ARGUMENT_ORDER = ("object", "instance", "instance_id", "fastener", "tool",
                   "region", "target", "support", "destination", "anchor")


# One destination, two names.  The serving destination is called
# `serving_area` by the ground-truth executor, the oracle world state and this
# catalogue, and `dining_table` by the compiled pipeline and by the goal
# evaluator in evaluation_metrics.py, which checks `at(cup, dining_table)`.
# Both refer to the same place; the inconsistency is between the benchmark's own
# artifacts, not something a plan can get right or wrong.  Normalising here
# keeps the comparison about actions instead of vocabulary, and changes no
# runtime behaviour.
DESTINATION_ALIASES = {"dining_table": "serving_area"}


def canonical_argument(value: str) -> str:
    return DESTINATION_ALIASES.get(str(value), str(value))


def arguments(action):
    value = action.get("arguments")
    if isinstance(value, list):
        return [canonical_argument(_argument(x)) for x in value]
    if isinstance(value, dict):
        ordered = [value[key] for key in _ARGUMENT_ORDER if key in value]
        extra = [v for k, v in sorted(value.items()) if k not in _ARGUMENT_ORDER]
        return [canonical_argument(_argument(x)) for x in ordered + extra]
    return [] if value is None else [_argument(value)]


def compare_unordered(expected, produced):
    """The same steps in any order, under a consistent body correspondence.

    Many orderings satisfy the same goal: the reference pours from each source
    in turn, and the planner may serve every utensil first.  Neither is more
    correct, so this asks whether the *set* of steps agrees, and the strict
    ordered check above is reported alongside it rather than instead of it.
    """
    if len(expected) != len(produced):
        return False, f"length {len(produced)} != expected {len(expected)}"

    # Search for a consistent correspondence with backtracking.  A greedy scan
    # is unsound here: a one-argument step like PICK matches many candidates, so
    # committing to the first one can bind a body to the wrong instance and then
    # dead-end on a later multi-argument step, reporting a difference where a
    # consistent correspondence exists.  K5 failed exactly that way -- PICK bound
    # the kettle to the first produced PICK and the POURs then had nowhere to go.
    used = [False] * len(produced)

    def assign(index, body_to_instance, instance_to_body):
        if index == len(expected):
            return True
        want = expected[index]
        want_args = arguments(want)
        for position, candidate in enumerate(produced):
            if used[position]:
                continue
            if str(candidate.get("operator")) != str(want.get("operator")):
                continue
            got_args = arguments(candidate)
            if len(want_args) != len(got_args):
                continue
            b2i, i2b = dict(body_to_instance), dict(instance_to_body)
            ok = True
            for w, g in zip(want_args, got_args):
                if not OBSERVED_INSTANCE.match(g):
                    if w != g:
                        ok = False
                        break
                    continue
                if b2i.setdefault(w, g) != g or i2b.setdefault(g, w) != w:
                    ok = False
                    break
            if not ok:
                continue
            used[position] = True
            if assign(index + 1, b2i, i2b):
                return True
            used[position] = False
        return False

    if assign(0, {}, {}):
        return True, "same steps, different order"
    unmatched = expected[sum(used)] if sum(used) < len(expected) else expected[0]
    return False, (f"no consistent correspondence; first unmatched "
                   f"{unmatched.get('operator')}{arguments(unmatched)}")


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
            # What kind of argument this is, is decided by what the produced
            # plan put there.  An observed instance id stands for a scene body
            # and has to correspond consistently; anything else is a name the
            # runtime and the catalogue share -- a region, a support, a fixed
            # anchor -- and has to be identical.
            if not OBSERVED_INSTANCE.match(g):
                if w != g:
                    return False, f"step {index} arg {position}: {g} != {w}"
                continue
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
        unordered_ok, unordered_detail = (True, detail) if ok else compare_unordered(
            expected, produced)
        rows.append({"domain": domain, "variant": variant, "trial": trial,
                     "intended_outcome": meta.get("intended_outcome"),
                     "expected_steps": len(expected), "produced_steps": len(produced),
                     "status": ("EXACT_GT_ACTION_SEQUENCE" if ok
                                else "SAME_STEPS_DIFFERENT_ORDER" if unordered_ok
                                else "DIFFERS"),
                     "detail": detail if ok else (
                         unordered_detail if unordered_ok else detail)})
    rows.sort(key=lambda r: (r["domain"], r["variant"], r["trial"]))
    # Scored over the trials the pipeline reported complete.  A partial plan on a
    # trial that failed was never meant to be the reference sequence, and
    # counting it as a mismatch would say nothing.
    successes = set()
    replay_rows = root / "BASELINE_REPLAY.json"
    if replay_rows.exists():
        successes = {
            (r["domain"], r["variant"], r["trial"])
            for r in json.loads(replay_rows.read_text()) if r.get("success")
        }
    for r in rows:
        r["pipeline_success"] = (r["domain"], r["variant"], r["trial"]) in successes
    scored = [r for r in rows if r.get("produced_steps") and r["pipeline_success"]]
    exact = [r for r in scored if r["status"] == "EXACT_GT_ACTION_SEQUENCE"]
    same = [r for r in scored if r["status"] == "SAME_STEPS_DIFFERENT_ORDER"]
    print(f"trials with a produced plan: {sum(1 for r in rows if r.get('produced_steps'))}")
    print(f"of which the pipeline reported complete: {len(scored)}")
    print(f"  exact GT action sequence:        {len(exact)}")
    print(f"  same steps, different order:     {len(same)}")
    print(f"  differs:                         {len(scored) - len(exact) - len(same)}")
    print()
    for r in scored:
        mark = {"EXACT_GT_ACTION_SEQUENCE": "EXACT", "SAME_STEPS_DIFFERENT_ORDER": "ORDER"}.get(
            r["status"], "     ")
        print(f"  {mark:5s} {r['domain'][:4]:4s} {r['variant']:4s}/{r['trial'][-2:]} "
              f"{r['produced_steps']}/{r['expected_steps']} steps  {r['detail'][:80]}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps({
            "trials_with_plan": sum(1 for r in rows if r.get("produced_steps")),
            "scored_successes": len(scored),
            "exact_gt_action_sequence": len(exact),
            "same_steps_different_order": len(same),
            "rows": rows,
        }, indent=2) + "\n")
        keys = ["domain", "variant", "trial", "intended_outcome", "pipeline_success",
                "expected_steps", "produced_steps", "status", "detail"]
        with out.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nwrote {out.with_suffix('.json')} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
