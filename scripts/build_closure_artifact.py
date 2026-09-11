#!/usr/bin/env python3
"""Assemble the closure artifact from a frozen replay, offline and reproducibly.

Reads a before and an after frozen-replay row set, the archived FM call records,
and the current source tree, and writes the named closure files.  It makes no FM
call, runs no simulation, and never lets a reference value reach a decision --
the ground-truth columns already present in each row are reported as evidence
about the run, which is what an evaluation is for.
"""
from __future__ import annotations
import argparse, ast, collections, csv, json, os, re, statistics, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

DOMAINS = ("kitchen", "living_room", "workshop")
STAGES = (
    ("strict_v3_valid", "wire contract valid"),
    ("task_valid", "task contract valid"),
    ("canonical_graph_emitted", "canonical graph emitted"),
    ("executable_contract_complete", "executable contract complete"),
    ("grounding_reached", "grounding reached"),
    ("complete_grounding", "complete grounding"),
    ("astar_reached", "A* invoked"),
    ("success", "success"),
    ("outcome_correct", "outcome correct"),
)


def counts(rows):
    out = {"trials": len(rows)}
    for key, _ in STAGES:
        out[key] = sum(1 for r in rows if r.get(key))
    out["online_false_completion"] = sum(
        1 for r in rows if r.get("success") and not r.get("feasible"))
    out["gt_goal_mismatch_completion"] = sum(
        1 for r in rows if r.get("success") and r.get("feasible")
        and not r.get("gt_full_task_satisfied"))
    out["harness_failures"] = sum(
        1 for r in rows if r.get("harness_failure")
        or r.get("pipeline_status") == "PIPELINE_EXCEPTION")
    out["fm_semantic_requests"] = sum(r.get("semantic_vlm_requests", 0) or 0 for r in rows)
    out["max_astar_invocations"] = max(
        [r.get("astar_invocations", 0) or 0 for r in rows] or [0])
    return out


CAUSES = (
    ("required operation has no runtime capability",
     r"NO_RUNTIME_CAPABILITY_MATCHES_THIS_OPERATION_PHRASE|NO_CAPABILITY_SIGNATURE_ACCEPTS"),
    ("operation participant has no runtime role", r"SLOT_PARTICIPANT_HAS_NO_RUNTIME_ROLE"),
    ("no expressed physical operation survived compilation", r"none survived compilation"),
    ("expressed operation disabled during compilation",
     r"Operations disabled during compilation|UNSUPPORTED_OPERATOR|UNINSTANTIABLE"),
    ("required relation has no canonical reading",
     r"UNINTERPRETABLE_REQUIRED_RELATION|NO_LEGAL_ORIENTATION_FOR_PARTICIPANTS|"
     r"Uninterpretable required relations"),
    ("role could not be typed", r"Unresolved roles|AMBIGUOUS_ROLE_MAPPING|UNRESOLVED_SEMANTIC"),
    ("sanitizer dropped a declared reference", r"Sanitizer reported semantic incompleteness"),
    ("no functional role compiled", r"No valid functional roles compiled"),
    ("relation participants collapse to one", r"TOO_FEW_DISTINCT_PARTICIPANTS"),
)


def root_causes(row):
    text = " ".join(str(x) for x in (
        (row.get("executable_contract_missing_reasons") or [])
        + (row.get("contract_missing_reasons") or [])))
    found = [name for name, pattern in CAUSES if re.search(pattern, text, re.I)]
    return found or ["unclassified"]


def fm_call_audit(archive: Path):
    records, lengths, reasons = [], [], collections.Counter()
    for path in sorted(archive.glob("*/*/trial_*/fm_call_record.json")):
        data = json.loads(path.read_text())
        usage = data.get("usage") or {}
        reasons[str(data.get("finish_reason"))] += 1
        tokens = usage.get("completion_tokens") or 0
        lengths.append(tokens)
        records.append({
            "trial": str(path.parent.relative_to(archive)),
            "model": data.get("model"),
            "finish_reason": data.get("finish_reason"),
            "completion_tokens": tokens,
            "json_parse_success": bool(data.get("json_parse_success")),
            "parse_error": data.get("parse_error"),
        })
    finished = sorted(r["completion_tokens"] for r in records if r["finish_reason"] == "stop")
    return {
        "calls": len(records),
        "models": sorted({str(r["model"]) for r in records}),
        "finish_reasons": dict(reasons),
        "truncated_trials": [r["trial"] for r in records if r["finish_reason"] == "length"],
        "completion_tokens": {
            "median_of_finished": statistics.median(finished) if finished else None,
            "max_of_finished": max(finished) if finished else None,
            "over_8192_of_finished": sum(1 for t in finished if t > 8192),
            "finished": len(finished),
        },
        "records": records,
    }


def one_semantic_call_per_trial(rows):
    offenders = [
        f"{r['domain']}/{r['variant']}/{r['trial']}={r.get('semantic_vlm_requests')}"
        for r in rows if (r.get("semantic_vlm_requests") or 0) > 1
    ]
    return offenders


def one_astar_per_trial(rows):
    return [
        f"{r['domain']}/{r['variant']}/{r['trial']}={r.get('astar_invocations')}"
        for r in rows if (r.get("astar_invocations") or 0) > 1
    ]


def write_rows(path_stem: Path, rows):
    path_stem.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n")
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen and not isinstance(row[key], (dict, list)):
                seen.add(key)
                keys.append(key)
    with path_stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def run(command, **kwargs):
    return subprocess.run(command, cwd=str(REPO), capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(REPO)), **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze-sha", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    archive = Path(args.archive).resolve()

    write_rows(out / "FROZEN_REPLAY_BEFORE", before)
    write_rows(out / "FROZEN_REPLAY_AFTER", after)

    summary = {
        "TOTAL": {"before": counts(before), "after": counts(after)},
        **{d.upper(): {"before": counts([r for r in before if r["domain"] == d]),
                       "after": counts([r for r in after if r["domain"] == d])}
           for d in DOMAINS},
    }
    (out / "STAGE_TABLE.json").write_text(json.dumps(summary, indent=2) + "\n")

    causes = collections.Counter()
    per_trial = {}
    for row in after:
        if row.get("success"):
            continue
        found = root_causes(row)
        per_trial[f"{row['domain']}/{row['variant']}/{row['trial']}"] = {
            "outcome_category": row.get("outcome_category"),
            "feasible": row.get("feasible"),
            "outcome_correct": row.get("outcome_correct"),
            "causes": found,
            "missing_reasons": [
                str(x)[:400] for x in (row.get("executable_contract_missing_reasons") or [])][:6],
        }
        for name in found:
            causes[name] += 1
    (out / "FAILURE_ROOT_CAUSES.json").write_text(json.dumps({
        "tally": dict(causes.most_common()),
        "per_trial": per_trial,
    }, indent=2) + "\n")

    audit = fm_call_audit(archive)
    (out / "MODEL_CONFIG.json").write_text(json.dumps({
        "archive": str(archive.relative_to(REPO)) if archive.is_relative_to(REPO) else str(archive),
        "fm_call_audit": {k: v for k, v in audit.items() if k != "records"},
        "per_call": audit["records"],
        "frozen_generation_budget_default": 24000,
        "decoding": {"temperature": 0.0, "top_p": 1.0, "enable_thinking": False},
        "semantic_requests_per_trial_rule": 1,
    }, indent=2) + "\n")

    (out / "SEMANTIC_ACCOUNTING_AUDIT.json").write_text(json.dumps({
        "one_semantic_request_per_trial": {
            "violations": one_semantic_call_per_trial(after),
            "total_requests_in_replay": counts(after)["fm_semantic_requests"],
            "note": "A frozen replay makes no FM call; the archived run made one per trial.",
        },
        "at_most_one_astar_per_trial": {"violations": one_astar_per_trial(after)},
        "online_false_completion": {
            "count": counts(after)["online_false_completion"],
            "trials": [f"{r['domain']}/{r['variant']}/{r['trial']}"
                       for r in after if r.get("success") and not r.get("feasible")],
        },
        "gt_goal_mismatch_completion": {
            "count": counts(after)["gt_goal_mismatch_completion"],
            "trials": [f"{r['domain']}/{r['variant']}/{r['trial']}" for r in after
                       if r.get("success") and r.get("feasible")
                       and not r.get("gt_full_task_satisfied")],
        },
        "harness_failures": {"count": counts(after)["harness_failures"]},
    }, indent=2) + "\n")

    leakage = run([sys.executable, str(REPO / "scripts" / "audit_no_gt_leakage.py")])
    (out / "NO_GT_LEAKAGE_AUDIT.txt").write_text(
        leakage.stdout + ("\nSTDERR\n" + leakage.stderr if leakage.stderr else ""))

    repro = run([sys.executable, "-m", "pytest",
                 "mujoco_scenes/functional_tamp_pipeline/tests/test_semantic_determinism.py",
                 "-q", "-p", "no:randomly"])
    front_hashes = []
    for seed in ("0", "1", "7", "12345"):
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "hash_front_half.py"),
             "--frozen-root", str(archive)],
            cwd=str(REPO), capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONHASHSEED=seed))
        front_hashes.append((seed, proc.stdout.strip() or proc.stderr.strip()[-200:]))
    (out / "REPRODUCIBILITY_AUDIT.txt").write_text(
        "REPRODUCIBILITY AUDIT\n" + "=" * 78 + "\n\n"
        "1. Semantic front half, compiled from every archived contract, hashed once\n"
        "   per PYTHONHASHSEED.  Identical hashes mean no set-iteration order or\n"
        "   other process-local nondeterminism reaches the compiled graph.\n\n"
        + "".join(f"   PYTHONHASHSEED={seed}: {value}\n" for seed, value in front_hashes)
        + "\n2. Determinism test suite\n\n" + repro.stdout[-3000:] + "\n")

    manifest = {
        "freeze_commit": args.freeze_sha or run(
            ["git", "rev-parse", "HEAD"]).stdout.strip(),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip(),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_archive": str(archive.relative_to(REPO)) if archive.is_relative_to(REPO) else str(archive),
        "before_rows": str(Path(args.before)),
        "after_rows": str(Path(args.after)),
        "git_status_not_clean": bool(run(["git", "status", "--porcelain",
                                          "mujoco_scenes", "scripts"]).stdout.strip()),
        "python": sys.version.split()[0],
    }
    (out / "FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    for path in sorted(out.iterdir()):
        print("  ", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
