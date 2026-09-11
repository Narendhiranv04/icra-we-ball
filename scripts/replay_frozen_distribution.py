#!/usr/bin/env python3
"""Replay a whole frozen V3 distribution in parallel with zero FM calls.

Every trial's archived raw semantic response is fed back through the current
deterministic pipeline, so any movement in the stage table is attributable to
the compiler/grounder rather than to model sampling.  Writes
BASELINE_REPLAY.{json,csv}, a summary JSON, and a compact markdown report.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures as cf, csv, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DOMAINS = {"kitchen": [f"K{i}" for i in range(1, 13)],
           "living_room": [f"L{i}" for i in range(1, 11)],
           "workshop": [f"W{i}" for i in range(1, 11)]}
STAGES = ["strict_v3_valid", "task_valid", "canonical_graph_emitted",
          "executable_contract_complete", "grounding_reached",
          "complete_grounding", "astar_reached", "success", "outcome_correct"]


def jobs(root: Path, out: Path):
    for t in (1, 2, 3):
        trial = f"trial_{t:02d}"
        for dom, variants in DOMAINS.items():
            for var in variants:
                raw = root / trial / dom / var / "vlm" / "fm_diagnostics" / "fm_call_001.json"
                frozen = root / dom / var / trial / "raw_v3.json"
                if not raw.exists() and not frozen.exists():
                    continue
                yield dom, var, trial, (raw if raw.exists() else frozen), out / trial, \
                    out / "rows" / f"{dom}__{var}__{trial}.json"


def run(job, timeout):
    dom, var, trial, raw, outroot, rowout = job
    if rowout.exists():
        return dom, var, trial, "cached"
    # PYTHONHASHSEED is read at interpreter start, so it has to be in the child's
    # environment; no in-process call can set it afterwards.  Without it Python
    # randomises string hashing per process, set iteration order changes, and a
    # choice among equally ranked candidates goes a different way.  Two replays
    # of the same archived responses disagreed on kitchen/K7/trial_01 for exactly
    # this reason: seed 1 leaves coffee_container unseated with an 8-action plan,
    # seeds 2 and 3 leave coffee_source unseated with a 20-action plan.
    env = dict(os.environ, PYTHONPATH=str(REPO), TAMP_FM_SCHEMA_VERSION="3",
               TAMP_REPO=str(REPO), MUJOCO_GL="egl", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONHASHSEED="0")
    cmd = [sys.executable, str(HERE / "replay_frozen_trial.py"), "--domain", dom, "--variant", var,
           "--trial", trial, "--raw", str(raw), "--output-root", str(outroot),
           "--row-out", str(rowout)]
    log = outroot / "logs" / f"{dom}__{var}__{trial}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w") as stream:
            p = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=stream,
                               stderr=subprocess.STDOUT, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    if not rowout.exists():
        rowout.parent.mkdir(parents=True, exist_ok=True)
        rowout.write_text(json.dumps({
            "domain": dom, "variant": var, "trial": trial, "harness_failure": str(rc),
            "strict_v3_valid": False, "task_valid": False, "graph_compiled": False,
            "grounding_reached": False, "complete_grounding": False, "astar_reached": False,
            "success": False, "outcome_correct": False,
            "outcome_category": None, "pipeline_status": "HARNESS_FAILURE",
        }, indent=2) + "\n")
    return dom, var, trial, rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--label", default="")
    ap.add_argument("--domains", default="", help="comma-separated subset, e.g. living_room")
    ap.add_argument("--variants", default="", help="comma-separated subset, e.g. L1,L4,L5")
    ap.add_argument("--trials", default="", help="comma-separated subset, e.g. trial_01")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    todo = list(jobs(args.root, args.out))
    if args.domains:
        keep = {d.strip() for d in args.domains.split(",") if d.strip()}
        todo = [j for j in todo if j[0] in keep]
    if args.variants:
        keep = {v.strip() for v in args.variants.split(",") if v.strip()}
        todo = [j for j in todo if j[1] in keep]
    if args.trials:
        keep = {t.strip() for t in args.trials.split(",") if t.strip()}
        todo = [j for j in todo if j[2] in keep]
    print(f"{len(todo)} frozen trials to replay -> {args.out}", flush=True)
    t0 = time.perf_counter()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for dom, var, trial, rc in pool.map(lambda j: run(j, args.timeout), todo):
            done += 1
            print(f"[{done}/{len(todo)}] {dom}/{var}/{trial} rc={rc} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted((args.out / "rows").glob("*.json"))]
    (args.out / "BASELINE_REPLAY.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen and not isinstance(r[k], (dict, list)):
                seen.add(k); keys.append(k)
    with (args.out / "BASELINE_REPLAY.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    summarize(rows, args.out, args.label)
    return 0


def summarize(rows, out: Path, label: str):
    def counts(sub):
        return {s: sum(1 for r in sub if r.get(s)) for s in STAGES}
    report = {"label": label, "total_trials": len(rows), "overall": counts(rows),
              "outcome_categories": dict(collections.Counter(r.get("outcome_category") for r in rows)),
              "per_domain": {}, "per_variant": {}}
    for dom, variants in DOMAINS.items():
        sub = [r for r in rows if r["domain"] == dom]
        d = counts(sub); d["trials"] = len(sub)
        d["outcome_categories"] = dict(collections.Counter(r.get("outcome_category") for r in sub))
        report["per_domain"][dom] = d
        for var in variants:
            tr = [r for r in sub if r["variant"] == var]
            if not tr:
                continue
            report["per_variant"][var] = {"domain": dom, "trials": len(tr), **counts(tr),
                "outcomes": [r.get("outcome_category") for r in tr],
                "stops": sorted({str(r.get("failure_reason") or "")[:90] for r in tr})}
    # Soundness, kept apart from agreement with the reference goal set: a
    # completion on a variant the benchmark calls infeasible is a defect, while
    # a completion whose achieved goals merely differ from the reference may be
    # a difference of interpretation.  Adding them together made the soundness
    # figure move whenever the reference did.
    report["online_false_completions"] = [
        f"{r['domain']}/{r['variant']}/{r['trial']}" for r in rows
        if r.get("success") and not r.get("feasible")]
    report["gt_goal_mismatch_completions"] = [
        f"{r['domain']}/{r['variant']}/{r['trial']}" for r in rows
        if r.get("success") and r.get("feasible") and not r.get("gt_full_task_satisfied")]
    report["stop_reasons"] = dict(collections.Counter(
        str(r.get("failure_reason") or "")[:120] for r in rows).most_common())
    report["compile_missing_reason_heads"] = dict(collections.Counter(
        str((r.get("contract_missing_reasons") or [""])[0])[:110] for r in rows
        if r.get("contract_missing_reasons")).most_common())
    report["disabled_operation_codes"] = dict(collections.Counter(
        c for r in rows for c in (r.get("disabled_operation_codes") or [])).most_common())
    (out / "BASELINE_REPLAY_SUMMARY.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    n = len(rows) or 1
    lines = [f"# Frozen zero-call replay {label}", "",
             f"trials: {len(rows)}", "", "| stage | count | pct |", "| --- | ---: | ---: |"]
    for s in STAGES:
        lines.append(f"| {s} | {report['overall'][s]} | {100*report['overall'][s]/n:.1f}% |")
    lines += ["", "| domain | n | " + " | ".join(STAGES) + " |",
              "| --- | ---: |" + " ---: |" * len(STAGES)]
    for dom, d in report["per_domain"].items():
        lines.append(f"| {dom} | {d['trials']} | " + " | ".join(str(d[s]) for s in STAGES) + " |")
    lines += ["", "## outcome categories"]
    for k, v in sorted(report["outcome_categories"].items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines += ["", "## per domain outcome categories"]
    for dom, d in report["per_domain"].items():
        lines.append(f"- **{dom}**: " + ", ".join(
            f"{k}={v}" for k, v in sorted(d["outcome_categories"].items(), key=lambda x: -x[1])))
    lines += ["", "## top stop reasons"]
    for k, v in list(report["stop_reasons"].items())[:25]:
        lines.append(f"- {v:3d}  {k}")
    lines += ["", "## disabled operation codes"]
    for k, v in report["disabled_operation_codes"].items():
        lines.append(f"- {v:3d}  {k}")
    (out / "BASELINE_REPLAY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    raise SystemExit(main())
