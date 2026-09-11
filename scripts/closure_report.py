#!/usr/bin/env python3
"""Compare two frozen-replay artifacts and print the closure report tables.

Offline only.  The ground-truth columns already present in each row are used to
report whether a completion was confirmed; nothing here influences the pipeline.
"""
from __future__ import annotations
import argparse, collections, json, re, statistics
from pathlib import Path

DOMAINS = ("kitchen", "living_room", "workshop")
STAGES = (
    ("wire_valid", "strict_v3_valid"),
    ("task_valid", "task_valid"),
    ("canonical_graph_emitted", "canonical_graph_emitted"),
    ("executable_contract_complete", "executable_contract_complete"),
    ("grounding_reached", "grounding_reached"),
    ("complete_grounding", "complete_grounding"),
    ("astar_invoked", "astar_reached"),
    ("success", "success"),
)
OUTCOMES = ("TASK_SPECIFICATION_FAILURE", "GRAPH_COMPILATION_FAILURE",
            "OBJECT_DISCOVERY_FAILURE", "FUNCTIONAL_ASSIGNMENT_FAILURE",
            "PLANNING_FAILURE", "SUCCESS", "None")


def emitted(row):
    if "canonical_graph_emitted" in row:
        return bool(row["canonical_graph_emitted"])
    return bool(row.get("graph_compiled"))


def block(rows):
    out = {"trials": len(rows)}
    for name, key in STAGES:
        if name == "canonical_graph_emitted":
            out[name] = sum(1 for r in rows if emitted(r))
        else:
            out[name] = sum(1 for r in rows if r.get(key))
    out["gt_goals"] = sum(r.get("gt_goals_satisfied", 0) or 0 for r in rows)
    out["feasible_fully_done"] = sum(
        1 for r in rows if r.get("feasible") and r.get("gt_full_task_satisfied"))
    # Two very different things used to be added together here.  A completion on
    # a variant the benchmark declares infeasible is unsound: the pipeline
    # claimed to finish a task that cannot be finished, and one of those is a
    # defect.  A completion on a *feasible* variant whose achieved goals differ
    # from the reference set may be a defect or may be a difference of
    # interpretation between the reference and the compiled goal, and reading it
    # as a false completion made the soundness figure unusable -- it moved
    # whenever the reference did.  They are counted apart.
    out["online_false_completions"] = sum(
        1 for r in rows if r.get("success") and not r.get("feasible"))
    out["gt_goal_mismatch_completions"] = sum(
        1 for r in rows
        if r.get("success") and r.get("feasible") and not r.get("gt_full_task_satisfied"))
    out["outcome_correct"] = sum(1 for r in rows if r.get("outcome_correct"))
    out["harness_failures"] = sum(
        1 for r in rows if r.get("harness_failure") or r.get("pipeline_status") == "PIPELINE_EXCEPTION")
    out["fm_calls"] = sum(r.get("semantic_vlm_requests", 0) or 0 for r in rows)
    out["max_astar"] = max([r.get("astar_invocations", 0) or 0 for r in rows] or [0])
    return out


CAUSES = (
    ("no expressed physical operation survived",
     r"none survived compilation"),
    ("required operation has no runtime capability",
     r"NO_RUNTIME_CAPABILITY_MATCHES_THIS_OPERATION_PHRASE|NO_CAPABILITY_SIGNATURE_ACCEPTS"),
    ("operation participant has no runtime role",
     r"SLOT_PARTICIPANT_HAS_NO_RUNTIME_ROLE"),
    ("expressed operation disabled during compilation",
     r"Operations disabled during compilation|UNSUPPORTED_OPERATOR|UNINSTANTIABLE"),
    ("required relation has no canonical reading",
     r"UNINTERPRETABLE_REQUIRED_RELATION|NO_LEGAL_ORIENTATION_FOR_PARTICIPANTS|Uninterpretable required relations"),
    ("role could not be typed",
     r"Unresolved roles|AMBIGUOUS_ROLE_MAPPING|UNRESOLVED_SEMANTIC"),
    ("sanitizer dropped a declared reference",
     r"Sanitizer reported semantic incompleteness"),
    ("no functional role compiled", r"No valid functional roles compiled"),
)


def classify(row):
    text = " ".join(str(x) for x in (
        (row.get("executable_contract_missing_reasons") or [])
        + (row.get("contract_missing_reasons") or [])))
    found = [name for name, pattern in CAUSES if re.search(pattern, text, re.I)]
    return found or ["unclassified"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    b = {(r["variant"], r["trial"]): r for r in before}
    a = {(r["variant"], r["trial"]): r for r in after}

    keys = ["trials"] + [n for n, _ in STAGES] + [
        "gt_goals", "feasible_fully_done", "online_false_completions",
        "gt_goal_mismatch_completions", "outcome_correct",
        "harness_failures", "fm_calls", "max_astar"]
    print("=" * 78)
    print("STAGES AND QUALITY, before -> after")
    print("=" * 78)
    width = max(len(k) for k in keys) + 2
    for scope, rows_b, rows_a in (
        ["TOTAL", before, after],
        *[[d, [r for r in before if r["domain"] == d], [r for r in after if r["domain"] == d]]
          for d in DOMAINS],
    ):
        bb, aa = block(rows_b), block(rows_a)
        print(f"\n-- {scope}")
        for k in keys:
            arrow = "" if bb[k] == aa[k] else ("   +" if aa[k] > bb[k] else "   -")
            print(f"   {k:<{width}} {bb[k]:4d} -> {aa[k]:4d}{arrow}")

    print("\n" + "=" * 78)
    print("OUTCOME CATEGORIES, before -> after")
    print("=" * 78)
    cb = collections.Counter(str(r.get("outcome_category")) for r in before)
    ca = collections.Counter(str(r.get("outcome_category")) for r in after)
    print(f"   {'category':<34}{'before':>8}{'after':>8}   per-domain after")
    for name in OUTCOMES:
        if not cb.get(name) and not ca.get(name):
            continue
        per = {d: sum(1 for r in after if r["domain"] == d and str(r.get("outcome_category")) == name)
               for d in DOMAINS}
        print(f"   {name:<34}{cb.get(name,0):>8}{ca.get(name,0):>8}   "
              f"k={per['kitchen']} l={per['living_room']} w={per['workshop']}")

    print("\n" + "=" * 78)
    print("GRAPH COMPILATION FAILURES BY ROOT CAUSE (after)")
    print("=" * 78)
    tally = collections.Counter()
    per_domain = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    gcf = [r for r in after if r.get("outcome_category") == "GRAPH_COMPILATION_FAILURE"]
    for r in gcf:
        for cause in classify(r):
            tally[cause] += 1
            per_domain[cause][r["domain"]] += 1
            if len(examples[cause]) < 4:
                examples[cause].append(f"{r['variant']}/{r['trial'][-2:]}")
    print(f"   total graph compilation failures: {len(gcf)}")
    print(f"   (a trial can carry more than one cause)")
    for cause, count in tally.most_common():
        d = per_domain[cause]
        print(f"   {count:3d}  {cause:<48} k={d['kitchen']} l={d['living_room']} w={d['workshop']}"
              f"   e.g. {', '.join(examples[cause])}")

    print("\n" + "=" * 78)
    print("COMPLETIONS THAT SHOULD NOT HAVE HAPPENED")
    print("=" * 78)
    print("   ONLINE_FALSE_COMPLETION -- success on a variant the benchmark calls")
    print("   infeasible.  This is the soundness figure and it must be zero.")
    for label, rows in (("before", before), ("after", after)):
        inf = [r for r in rows if r.get("success") and not r.get("feasible")]
        print(f"      {label}: {len(inf)}  " + (", ".join(
            f"{r['variant']}/{r['trial'][-2:]}" for r in inf) or "none"))
    print("   GT_GOAL_MISMATCH_COMPLETION -- success on a feasible variant whose")
    print("   achieved goals differ from the reference set.  Reported separately")
    print("   because it can be a difference of interpretation rather than a defect.")
    for label, rows in (("before", before), ("after", after)):
        fc = [r for r in rows
              if r.get("success") and r.get("feasible") and not r.get("gt_full_task_satisfied")]
        print(f"      {label}: {len(fc)}  " + (", ".join(
            f"{r['variant']}/{r['trial'][-2:]} ({r.get('gt_goals_satisfied')}/{r.get('gt_goals_total')})"
            for r in fc) or "none"))

    print("\n" + "=" * 78)
    print("RUNTIME (seconds per trial)")
    print("=" * 78)
    for label, rows in (("before", before), ("after", after)):
        times = sorted(r.get("runtime_sec", 0) or 0 for r in rows)
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
        print(f"   {label}: median {statistics.median(times):7.1f}   p95 {p95:7.1f}   "
              f"max {max(times):7.1f}   over 120s: {sum(1 for t in times if t > 120)}")
    print("   slowest after:")
    for r in sorted(after, key=lambda x: -(x.get("runtime_sec", 0) or 0))[:6]:
        pb = b.get((r["variant"], r["trial"]), {}).get("runtime_sec", 0)
        print(f"      {r['variant']:4s}/{r['trial'][-2:]}  {pb:8.1f} -> {r.get('runtime_sec', 0):8.1f}")

    print("\n" + "=" * 78)
    print("PER VARIANT (three trials each)")
    print("=" * 78)
    print(f"   {'var':4s} {'wire':>5}{'task':>5}{'graph':>6}{'exec':>5}{'grnd':>5}{'cgnd':>5}"
          f"{'plan':>5}{'succ':>5}{'gtok':>5} {'gt goals':>10}  dominant remaining cause")
    for domain in DOMAINS:
        variants = sorted({r["variant"] for r in after if r["domain"] == domain},
                          key=lambda v: int(v[1:]))
        for var in variants:
            rows = [r for r in after if r["variant"] == var]
            prev = [b[(var, r["trial"])] for r in rows if (var, r["trial"]) in b]
            blk = block(rows)
            gtb = sum(r.get("gt_goals_satisfied", 0) or 0 for r in prev)
            causes = collections.Counter()
            for r in rows:
                if r.get("outcome_category") in (None, "SUCCESS"):
                    continue
                if r.get("outcome_category") == "GRAPH_COMPILATION_FAILURE":
                    for c in classify(r):
                        causes[c] += 1
                else:
                    causes[str(r.get("outcome_category"))] += 1
            top = causes.most_common(1)[0][0] if causes else "-"
            print(f"   {var:4s} {blk['wire_valid']:>5}{blk['task_valid']:>5}"
                  f"{blk['canonical_graph_emitted']:>6}{blk['executable_contract_complete']:>5}"
                  f"{blk['grounding_reached']:>5}{blk['complete_grounding']:>5}"
                  f"{blk['astar_invoked']:>5}{blk['success']:>5}{blk['outcome_correct']:>5}"
                  f" {gtb:4d} ->{blk['gt_goals']:3d}  {top}")

    print("\n" + "=" * 78)
    print("TRIALS THAT WENT BACKWARDS")
    print("=" * 78)
    any_regression = False
    for key, row in sorted(a.items()):
        old = b.get(key)
        if not old:
            continue
        worse = []
        if old.get("success") and not row.get("success"):
            worse.append("lost reported completion")
        if (old.get("gt_goals_satisfied") or 0) > (row.get("gt_goals_satisfied") or 0):
            worse.append(f"gt {old['gt_goals_satisfied']} -> {row['gt_goals_satisfied']}")
        if old.get("complete_grounding") and not row.get("complete_grounding"):
            worse.append("lost complete grounding")
        if emitted(old) and not emitted(row):
            worse.append("lost canonical graph")
        if old.get("executable_contract_complete") and not row.get("executable_contract_complete"):
            worse.append("lost executable contract")
        if worse:
            any_regression = True
            print(f"   {key[0]:4s}/{key[1][-2:]}  {', '.join(worse)}"
                  f"   [{old.get('outcome_category')} -> {row.get('outcome_category')}]")
    if not any_regression:
        print("   none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
