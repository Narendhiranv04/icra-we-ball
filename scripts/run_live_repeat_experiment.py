#!/usr/bin/env python3
"""Run the live experiment as N independent repetitions of the whole matrix.

Each repetition is one full pass over all 32 variants, and every trial inside it
makes exactly one semantic FM call.  Repetitions are *independent samples of the
same system*, not attempts: nothing here looks at how a repetition scored, and
nothing selects among them.  Aggregation is a separate step over whatever the
runs produced.

Resumable, and deliberately unable to lose or overwrite a completed trial.  A
repetition is skipped only when its own manifest says it finished, and each
repetition writes to its own directory, so a rerun cannot touch an archived raw
response.  A repetition that failed stays on disk, is reported, and is not
silently redone -- rerunning it is an explicit choice, because a failure that
quietly disappears and is retried until it passes is cherry-picking.

Starts, stops and configures nothing about vLLM.  Makes no semantic request of
its own: preflight only asks the endpoint which models it serves.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _require_model(base_url: str, required: str) -> dict:
    """The endpoint must serve exactly the model the experiment names.

    There used to be a fallback to whatever the endpoint happened to serve
    first.  That silently benchmarks a different model than the one the run
    claims, which is the one bookkeeping error no amount of later analysis can
    detect or undo.  Absent means abort.

    Returns the endpoint's own record for that model, so the run freezes the
    served revision and not only the name a caller typed.
    """
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=15) as response:
        rows = json.load(response)["data"]
    served = {str(row.get("id")): row for row in rows}
    if not served:
        raise SystemExit("Preflight blocked: endpoint exposes no models")
    if required not in served:
        raise SystemExit(
            f"Preflight blocked: required model {required!r} is not served. "
            f"The endpoint offers {sorted(served)}. Refusing to benchmark a different model.")
    return served[required]


def _model_revision(record: dict) -> str:
    """A stable fingerprint of the served model as the endpoint describes it.

    Two endpoints can serve the same model *name* over different weights or
    configurations.  The name alone therefore does not identify what was
    benchmarked; this hashes the endpoint's own record of it so a later run
    against different weights is refused rather than pooled.
    """
    import hashlib
    stable = {key: record[key] for key in sorted(record)
              if key not in {"created"} and isinstance(record[key], (str, int, float, bool))}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


# Everything that changes what the model returns, set explicitly rather than
# inherited.  A shell variable must not be able to alter an experiment quietly,
# so each is passed and each is recorded, and the two come from one dict.
#
# The values are not chosen here: they are the sampler of the archived 3x32 V3
# distribution (benchmark_reports/v3_qwen_distribution_3x32_20260910T053937,
# model_config), which is the configuration every offline measurement of this
# pipeline was made against.  Running live under a different sampler would make
# the live numbers unattributable -- a drop could be the pipeline or could be
# the sampler, with no way to tell them apart.  In particular thinking is on and
# the temperature is 0.6: the repetitions exist to measure that variance, and a
# greedy sampler would make ten repetitions ten copies of one draw.
# test_the_live_sampler_is_the_one_the_offline_numbers_were_measured_on pins
# these against the archived manifest.
FROZEN_DISTRIBUTION = (
    "benchmark_reports/v3_qwen_distribution_3x32_20260910T053937/collection_manifest.json")
SAMPLER = {
    # 28000, not the archive's 24000.  The hard limit is 28247: the model's
    # context is 32768 and the worst observed prompt is 4521 tokens.
    #
    # This does NOT fix truncation.  All five unparseable archived calls spent
    # their entire 24000 budget on reasoning and emitted no content, and
    # re-issuing those same five inputs recovers 5/5 at 28000 *and* 5/5 at
    # 24000, consuming at most 14699 tokens -- so the ceiling was never the
    # binding constraint and truncation is a stochastic runaway that would hit
    # any ceiling.  The increase is justified only because the observed
    # convergent tail (14699) already exceeds the archive's maximum (12222), so
    # a draw that would finish has room to; it buys headroom, not a lower
    # failure rate.
    "TAMP_FM_MAX_TOKENS": "28000",
    "TAMP_FM_SCHEMA_VERSION": "3",
    "TAMP_FM_TEMPERATURE": "0.6",
    "TAMP_FM_TOP_P": "0.95",
    "TAMP_FM_TOP_K": "20",
    "TAMP_FM_PRESENCE_PENALTY": "1.0",
    "TAMP_FM_REPETITION_PENALTY": "1.0",
    "TAMP_FM_ENABLE_THINKING": "true",
    "TAMP_FM_VIEWS": "3",
}


def _frozen_identity(*, base_url: str, variant_set: str, run_type: str) -> dict[str, str]:
    """Everything that has to be identical across every repetition.

    Whatever is not recorded here cannot be checked later, so this carries the
    whole decision surface of the experiment: the code, the semantic contract
    (prompt, schema, capabilities, ontology), the endpoint, every sampler value
    that reaches the model, which variants the matrix covers, and what kind of
    run this is.  A second run that differs in any of it is a different
    experiment and is refused rather than pooled.
    """
    sys.path.insert(0, str(REPO))
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (  # noqa: E402
        SYSTEM_PROMPT_V3, compute_v3_prompt_and_schema_hash,
    )
    from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (  # noqa: E402
        get_robot_capability_registry_hash,
    )
    from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (  # noqa: E402
        get_runtime_semantic_ontology_hash,
    )
    try:
        prompt_hash = compute_v3_prompt_and_schema_hash()
    except Exception:
        import hashlib
        prompt_hash = hashlib.sha256(SYSTEM_PROMPT_V3.encode()).hexdigest()
    identity = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "prompt_and_schema_hash": prompt_hash,
        "capability_registry_hash": get_robot_capability_registry_hash(),
        "runtime_ontology_hash": get_runtime_semantic_ontology_hash(),
        "base_url": base_url,
        "variant_set": variant_set,
        "run_type": run_type,
    }
    # The sampler is frozen from the single dict that is also passed to the
    # evaluator, so what is recorded cannot drift from what was sent.
    identity.update({key.lower(): value for key, value in sorted(SAMPLER.items())})
    from mujoco_scenes.determinism import REPRODUCIBLE_CHILD_ENV
    identity.update({k.lower(): v for k, v in sorted(REPRODUCIBLE_CHILD_ENV.items())})
    return identity


def _complete(repeat_dir: Path) -> bool:
    """Whether this repetition finished and said so itself."""
    manifest = repeat_dir / "repeat_manifest.json"
    if not manifest.is_file():
        return False
    try:
        return bool(json.loads(manifest.read_text()).get("finished"))
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path,
                        default=Path("benchmark_reports/live_repeat_experiment"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True,
                        help="exact model id the endpoint must serve; no fallback")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--variants", default=None,
                        help="comma-separated subset; omit for all 32")
    parser.add_argument("--run-type", required=True, choices=("smoke", "full"),
                        help="what this run is for. Recorded and frozen, so a smoke "
                             "run's trials can never be pooled into the reported "
                             "experiment by sharing an output root.")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="redo repetitions that did not finish. Off by default: a "
                             "failure that is quietly retried until it passes is "
                             "cherry-picking, so asking for it has to be deliberate.")
    args = parser.parse_args()
    os.chdir(REPO)

    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("Preflight blocked: working tree is not clean")
    try:
        record = _require_model(args.base_url, args.model)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Preflight blocked: vLLM endpoint unavailable: {exc}")

    model = record["id"]
    variant_set = ",".join(sorted(v.strip() for v in args.variants.split(","))) \
        if args.variants else "ALL"
    identity = _frozen_identity(base_url=args.base_url, variant_set=variant_set,
                                run_type=args.run_type)
    identity["model"] = model
    identity["model_revision"] = _model_revision(record)
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_root / "frozen_identity.json"
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text())
        drifted = {k: (previous.get(k), identity[k])
                   for k in identity if previous.get(k) != identity[k]}
        if drifted:
            raise SystemExit(
                "Preflight blocked: this run would not be comparable with the "
                f"repetitions already in {args.output_root}: {drifted}")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")

    # The repeat count is the run's stopping rule, not part of what a single
    # repetition means, so it is recorded as the history of what was asked for
    # rather than drift-checked -- but it is recorded, because an experiment
    # extended after seeing its numbers is a different claim from one planned.
    history_path = args.output_root / "repeat_count_history.json"
    history = []
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []
    history.append({"requested_repeats": args.repeats,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "git_sha": identity["git_sha"]})
    history_path.write_text(json.dumps(history, indent=2) + "\n")

    print(f"model {model}  revision {identity['model_revision'][:12]}  "
          f"git {identity['git_sha'][:12]}  "
          f"prompt {identity['prompt_and_schema_hash'][:12]}  "
          f"run_type {args.run_type}  variants {variant_set}  repeats {args.repeats}",
          flush=True)

    skipped: list[int] = []
    failed: list[int] = []
    ran: list[int] = []
    for index in range(1, args.repeats + 1):
        repeat_dir = args.output_root / f"repeat_{index:02d}"
        if _complete(repeat_dir):
            skipped.append(index)
            print(f"[repeat {index:02d}] already finished; left untouched", flush=True)
            continue
        attempts = sorted(p for p in repeat_dir.glob("attempt_*") if p.is_dir())
        if attempts and not args.rerun_failed:
            failed.append(index)
            print(f"[repeat {index:02d}] {len(attempts)} unfinished attempt(s); preserved. "
                  f"Pass --rerun-failed to add another deliberately.", flush=True)
            continue
        # Each attempt gets its own directory and nothing ever writes into one
        # that exists.  A retry that reused the directory would both collide
        # with the evaluator's refusal to overwrite a live run and destroy the
        # earlier attempt's raw responses, which are evidence about the model
        # whatever the attempt's outcome was.
        attempt_dir = repeat_dir / f"attempt_{len(attempts) + 1:02d}"
        if attempt_dir.exists():
            raise SystemExit(f"Preflight blocked: {attempt_dir} already exists")
        attempt_dir.mkdir(parents=True)
        # PYTHONHASHSEED is read at interpreter start, so it must be in the
        # child's environment.  Unset, Python randomises string hashing per
        # process and a choice among equally ranked grounding candidates goes a
        # different way between runs, which would put uncontrolled variance into
        # an experiment whose whole purpose is to measure FM variance.
        from mujoco_scenes.determinism import REPRODUCIBLE_CHILD_ENV
        env = dict(os.environ, TAMP_FM_BASE_URL=args.base_url, TAMP_FM_MODEL=model,
                   PYTHONPATH=".", **REPRODUCIBLE_CHILD_ENV, **SAMPLER)
        command = [sys.executable, "scripts/evaluate_vlm_functional_tamp.py",
                   "--mode", "vlm", "--spec-source", "live",
                   "--output-root", str(attempt_dir)]
        if args.variants:
            command += ["--variants", args.variants]
        started = time.time()
        code = subprocess.run(command, env=env).returncode
        elapsed = round(time.time() - started, 1)
        after = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        invariants_path = attempt_dir / "invariants.json"
        invariants = {}
        if invariants_path.is_file():
            try:
                invariants = json.loads(invariants_path.read_text())
            except Exception:
                invariants = {"errors": ["invariants.json unreadable"]}
        errors = list(invariants.get("errors", []))
        if code:
            errors.append(f"evaluator exit code {code}")
        if after != identity["git_sha"] or dirty:
            errors.append("code or working tree changed during the repetition")
        (repeat_dir / "repeat_manifest.json").write_text(json.dumps({
            "repeat_index": index, "finished": not errors, "errors": errors,
            "authoritative_attempt": attempt_dir.name, "attempts": len(attempts) + 1,
            "seconds": elapsed, **identity,
        }, indent=2) + "\n")
        (ran if not errors else failed).append(index)
        print(f"[repeat {index:02d}] {'ok' if not errors else 'FAILED ' + str(errors)} "
              f"in {elapsed}s", flush=True)

    print(f"\nran {ran}\nskipped (already finished) {skipped}\nfailed/preserved {failed}")
    print("\nAggregate with:  python scripts/aggregate_live_repeats.py --root "
          f"{args.output_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
