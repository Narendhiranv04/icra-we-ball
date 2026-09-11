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


def _require_model(base_url: str, required: str) -> str:
    """The endpoint must serve exactly the model the experiment names.

    There used to be a fallback to whatever the endpoint happened to serve
    first.  That silently benchmarks a different model than the one the run
    claims, which is the one bookkeeping error no amount of later analysis can
    detect or undo.  Absent means abort.
    """
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=15) as response:
        ids = [row["id"] for row in json.load(response)["data"]]
    if not ids:
        raise SystemExit("Preflight blocked: endpoint exposes no models")
    if required not in ids:
        raise SystemExit(
            f"Preflight blocked: required model {required!r} is not served. "
            f"The endpoint offers {sorted(ids)}. Refusing to benchmark a different model.")
    return required


# Everything that changes what the model returns, set explicitly rather than
# inherited.  A shell variable must not be able to alter an experiment quietly,
# so each is passed and each is recorded, and the two come from one dict.
SAMPLER = {
    "TAMP_FM_MAX_TOKENS": "24000",
    "TAMP_FM_SCHEMA_VERSION": "3",
    "TAMP_FM_TEMPERATURE": "0.0",
    "TAMP_FM_TOP_P": "1.0",
    "TAMP_FM_TOP_K": "-1",
    "TAMP_FM_PRESENCE_PENALTY": "0.0",
    "TAMP_FM_REPETITION_PENALTY": "1.0",
    "TAMP_FM_ENABLE_THINKING": "false",
    "TAMP_FM_VIEWS": "3",
}


def _frozen_identity() -> dict[str, str]:
    """Everything that has to be identical across every repetition."""
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
    return {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "prompt_and_schema_hash": prompt_hash,
        "capability_registry_hash": get_robot_capability_registry_hash(),
        "runtime_ontology_hash": get_runtime_semantic_ontology_hash(),
        "schema_version": "3",
        "max_tokens": "24000",
        "temperature": "0.0",
        "enable_thinking": "false",
    }


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
    parser.add_argument("--rerun-failed", action="store_true",
                        help="redo repetitions that did not finish. Off by default: a "
                             "failure that is quietly retried until it passes is "
                             "cherry-picking, so asking for it has to be deliberate.")
    args = parser.parse_args()
    os.chdir(REPO)

    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("Preflight blocked: working tree is not clean")
    try:
        model = _require_model(args.base_url, args.model)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Preflight blocked: vLLM endpoint unavailable: {exc}")

    identity = _frozen_identity()
    identity["model"] = model
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

    print(f"model {model}  git {identity['git_sha'][:12]}  "
          f"prompt {identity['prompt_and_schema_hash'][:12]}  repeats {args.repeats}",
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
        env = dict(os.environ, TAMP_FM_BASE_URL=args.base_url, TAMP_FM_MODEL=model,
                   PYTHONPATH=".", **SAMPLER)
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
