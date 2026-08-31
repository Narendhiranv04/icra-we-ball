#!/usr/bin/env python3
"""Pass 3.6B.2: Full 32-Variant VLM Phase-3 Evaluation Orchestrator (Hardened Provenance & Accounting)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

from mujoco_scenes.final_paper_variant_labels import resolve_variant_name
from mujoco_scenes.functional_tamp_pipeline.audit import (
    audit_prompt_leakage,
    compute_provenance_fingerprint,
    get_git_info,
)
from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import TASK as KITCHEN_TASK
from mujoco_scenes.functional_tamp_pipeline.domains.living_room import TASK as LIVING_ROOM_TASK
from mujoco_scenes.workshop_phase1.requirements import (
    CANONICAL_WORKSHOP_INSTRUCTION,
)
from mujoco_scenes.workshop_phase1.fm_adapter import SYSTEM_PROMPT
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import (
    evaluate_gf_against_reference,
)
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph
from mujoco_scenes.functional_tamp_pipeline.run import run_pipeline
from scripts.validate_phase3_vlm_replay import validate_vlm_replay


ROOT = Path(__file__).resolve().parent.parent
DOMAINS = {
    "kitchen": [f"K{i}" for i in range(1, 13)],
    "living_room": [f"L{i}" for i in range(1, 11)],
    "workshop": [f"W{i}" for i in range(1, 11)],
}

TASK_INSTRUCTIONS = {
    "kitchen": KITCHEN_TASK,
    "living_room": LIVING_ROOM_TASK,
    "workshop": CANONICAL_WORKSHOP_INSTRUCTION,
}

EXPECTED_STATUSES = {
    "kitchen": {f"K{i}": "ACTION_SEQUENCE_READY" if i <= 6 else "INFEASIBLE" for i in range(1, 13)},
    "living_room": {f"L{i}": "ACTION_SEQUENCE_READY" if i <= 6 else "INFEASIBLE" for i in range(1, 11)},
    "workshop": {f"W{i}": "ACTION_SEQUENCE_READY" if i <= 8 else "INFEASIBLE" for i in range(1, 11)},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def clean_case_directory(case_dir: Path) -> None:
    """Ensure target case directory is completely fresh with no leftover artifacts from previous runs."""
    if case_dir.exists():
        for item in case_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def ensure_tunnel_alive() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(f"{os.environ.get('TAMP_FM_BASE_URL', 'http://127.0.0.1:18000/v1')}/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return True
    except Exception:
        pass
    # Re-establish tunnel
    for host in ("10.10.16.68", "gvlab2.iiit.ac.in"):
        subprocess.run([
            "ssh", "-i", "/home/naren/keyfile",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=10",
            "-N", "-f", "-L", "18000:127.0.0.1:8000",
            f"long-horizon@{host}"
        ], check=False)
        time.sleep(1.0)
        try:
            import urllib.request
            req = urllib.request.Request(f"{os.environ.get('TAMP_FM_BASE_URL', 'http://127.0.0.1:18000/v1')}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return True
        except Exception:
            continue
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full 32-variant VLM evaluation matrix.")
    parser.add_argument("bundle_dir", nargs="?", default=None, help="Target output bundle directory")
    parser.add_argument("--bundle-dir", dest="bundle_dir_opt", default=None, help="Target output bundle directory")
    parser.add_argument("--no-resume", "--fresh", dest="no_resume", action="store_true", default=False,
                        help="Refuse stale case reuse and execute all cases fresh")
    parser.add_argument("--domain", dest="filter_domain", default=None, help="Filter to specific domain")
    parser.add_argument("--variant", dest="filter_variant", default=None, help="Filter to specific variant (e.g. K1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ts = datetime.now(timezone.utc)
    target_env = os.environ.get("TARGET_BUNDLE_DIR")
    target_path = args.bundle_dir_opt or args.bundle_dir or target_env
    no_resume = bool(args.no_resume or os.environ.get("TAMP_NO_RESUME", "").lower() in {"1", "true", "yes"})

    # Capture initial git provenance before writing benchmark directories
    git_info = get_git_info(ROOT)
    git_commit = git_info["git_commit"]
    git_dirty = git_info["git_dirty"]
    git_dirty_hash = git_info["git_dirty_source_hash"]
    is_clean_source_tree = git_info["is_clean_source_tree"]

    if target_path:
        bundle_dir = Path(target_path)
        ts_str = bundle_dir.name.replace("phase36b2_full_vlm_matrix_", "")
    else:
        ts_str = start_ts.strftime("%Y%m%d_%H%M%S")
        bundle_dir = ROOT / "runs" / "phase3_benchmarks" / f"phase36b2_full_vlm_matrix_{ts_str}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== PASS 3.6B.2 FULL 32-VARIANT VLM MATRIX EVALUATION ===")
    print(f"Target directory: {bundle_dir}")
    print(f"Resume allowed: {not no_resume}")
    print(f"Source clean: {is_clean_source_tree} (dirty: {git_dirty}, dirty_hash: {str(git_dirty_hash)[:8]})")
    print(f"Started at UTC: {start_ts.isoformat()}\n")

    # Set up environment variables
    os.environ["TAMP_FM_BASE_URL"] = os.environ.get("TAMP_FM_BASE_URL", "http://127.0.0.1:18000/v1")
    os.environ["TAMP_FM_MODEL"] = os.environ.get("TAMP_FM_MODEL", "qwen35-9b")
    os.environ["TAMP_FM_API_KEY"] = os.environ.get("TAMP_FM_API_KEY", "dummy")
    os.environ["MUJOCO_MENAGERIE_PATH"] = os.environ.get("MUJOCO_MENAGERIE_PATH", "/home/naren/third_party/mujoco_menagerie")

    # 1. Environment Provenance
    env_dir = bundle_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    (env_dir / "run_id.txt").write_text(f"phase36b2_full_vlm_matrix_{ts_str}\n")
    (env_dir / "model_endpoint.txt").write_text(f"{os.environ.get('TAMP_FM_BASE_URL')}\n")
    (env_dir / "python_version.txt").write_text(f"{sys.version}\n")
    (env_dir / "evaluation_protocol.txt").write_text(
        "PASS 3.6B.2: Full 32-variant VLM evaluation on Qwen 3.5 9B with lossless executable IR, "
        "provider replay (deterministic downstream solver check), random order search replays (seeds 0-9 for >=2 candidate regions), "
        "and offline GT reference evaluation.\n"
    )

    (env_dir / "git_state.txt").write_text(
        f"commit: {git_commit}\n"
        f"dirty: {git_dirty}\n"
        f"dirty_source_hash: {git_dirty_hash}\n"
        f"is_clean_source_tree: {is_clean_source_tree}\n"
    )

    # Test curl endpoint
    try:
        import urllib.request
        req = urllib.request.Request(f"{os.environ.get('TAMP_FM_BASE_URL')}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models_data = json.loads(resp.read().decode("utf-8"))
        write_json(env_dir / "local_model_connectivity.json", {"status": "SUCCESS", "models": models_data})
        write_json(env_dir / "remote_model_connectivity.json", {"status": "SUCCESS", "models": models_data})
        (env_dir / "tunnel_state.txt").write_text("CONNECTED\n")
    except Exception as e:
        write_json(env_dir / "local_model_connectivity.json", {"status": "FAILED", "error": str(e)})
        write_json(env_dir / "remote_model_connectivity.json", {"status": "FAILED", "error": str(e)})
        (env_dir / "tunnel_state.txt").write_text(f"FAILED: {e}\n")

    # 2. Case Registry
    case_registry = {}
    for domain, variants in DOMAINS.items():
        if args.filter_domain and domain != args.filter_domain:
            continue
        for v in variants:
            if args.filter_variant and v != args.filter_variant:
                continue
            case_id = f"{domain}_{v}"
            internal = resolve_variant_name(domain, v)
            case_registry[case_id] = {
                "case_id": case_id,
                "domain": domain,
                "paper_label": v,
                "internal_variant": internal,
                "exact_task_instruction": TASK_INSTRUCTIONS[domain],
                "dir_name": case_id,
            }
    write_json(bundle_dir / "case_registry.json", case_registry)

    # 3. Matrix Execution
    all_trials_records = []
    case_summaries = {}
    gt_spec_provider = GTSpecProvider()

    newly_executed_count = 0
    reused_count = 0
    newly_executed_runtime_seconds = 0.0
    newly_executed_trials_count = 0
    reused_trials_count = 0

    for case_idx, (case_id, cinfo) in enumerate(case_registry.items(), start=1):
        domain = cinfo["domain"]
        variant = cinfo["paper_label"]
        internal_variant = cinfo["internal_variant"]
        task_instruction = cinfo["exact_task_instruction"]
        expected_status = EXPECTED_STATUSES[domain][variant]

        current_fingerprint = compute_provenance_fingerprint(
            domain=domain,
            variant=variant,
            model=os.environ.get("TAMP_FM_MODEL", "qwen35-9b"),
            base_url=os.environ.get("TAMP_FM_BASE_URL", "http://127.0.0.1:18000/v1"),
            task_instruction=task_instruction,
            search_order="auto",
            repo_root=ROOT,
        )

        print(f"[{case_idx}/{len(case_registry)}] Evaluating {case_id} ({internal_variant})...", flush=True)
        ensure_tunnel_alive()
        case_dir = bundle_dir / case_id

        val_dir = case_dir / "validation"
        inputs_dir = case_dir / "inputs"
        live_dir = case_dir / "live"
        provider_replay_dir = case_dir / "provider_replay"
        random_replays_dir = case_dir / "random_replays"
        ref_dir = case_dir / "reference"

        # Check if case is already completed under the identical configuration fingerprint
        can_resume = (
            not no_resume
            and (case_dir / "case_manifest.json").is_file()
            and (val_dir / "case_replay_validation.json").is_file()
        )
        if can_resume:
            case_manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
            saved_fp = case_manifest.get("provenance_fingerprint", {})
            saved_fp_sha = saved_fp.get("fingerprint_sha256")
            curr_fp_sha = current_fingerprint.get("fingerprint_sha256")

            if saved_fp_sha and saved_fp_sha == curr_fp_sha:
                print(f"  -> Case {case_id} matches configuration fingerprint ({curr_fp_sha[:8]}), reusing existing artifacts...", flush=True)
                reused_count += 1
                case_summaries[case_id] = case_manifest
                ref_eval_dict = {}
                if (ref_dir / "gf_reference_evaluation.json").is_file():
                    ref_eval_dict = json.loads((ref_dir / "gf_reference_evaluation.json").read_text(encoding="utf-8"))

                # Live manifest and result
                live_manifest = {}
                live_result_data = {}
                live_m_vlm = live_dir / "runtime_output" / domain / variant / "vlm" / "run_manifest.json"
                live_r_vlm = live_dir / "runtime_output" / domain / variant / "vlm" / "result.json"
                if live_m_vlm.is_file():
                    live_manifest = json.loads(live_m_vlm.read_text(encoding="utf-8"))
                if live_r_vlm.is_file():
                    live_result_data = json.loads(live_r_vlm.read_text(encoding="utf-8"))

                inspected_live = live_result_data.get("inspected_regions", [])
                plan_live = live_result_data.get("plan", [])

                all_trials_records.append({
                    "case_id": case_id,
                    "domain": domain,
                    "variant": variant,
                    "internal_variant": internal_variant,
                    "trial_type": "live",
                    "search_order": "auto",
                    "seed": "",
                    "expected_status": expected_status,
                    "actual_status": case_manifest.get("live_terminal_status", ""),
                    "status_match": case_manifest.get("live_status_matches_expected", False),
                    "inspected_count": len(inspected_live),
                    "inspected_regions": ";".join(inspected_live),
                    "plan_length": len(plan_live) if plan_live else 0,
                    "runtime_sec": round(case_manifest.get("runtime_seconds", 0.0), 4),
                    "ref_complete": ref_eval_dict.get("reference_complete", False),
                    "exact_structural_match": ref_eval_dict.get("exact_structural_match", False),
                    "role_precision": ref_eval_dict.get("role_identity_precision", 0.0),
                    "role_recall": ref_eval_dict.get("role_identity_recall", 0.0),
                    "relation_precision": ref_eval_dict.get("relation_precision", 0.0),
                    "relation_recall": ref_eval_dict.get("relation_recall", 0.0),
                    "spec_sha256": live_manifest.get("specification_sha256", ""),
                })
                reused_trials_count += 1

                # Provider replay
                prov_m_vlm = provider_replay_dir / "runtime_output" / domain / variant / "vlm" / "run_manifest.json"
                prov_r_vlm = provider_replay_dir / "runtime_output" / domain / variant / "vlm" / "result.json"
                if prov_m_vlm.is_file():
                    prov_manifest = json.loads(prov_m_vlm.read_text(encoding="utf-8"))
                    prov_result_data = json.loads(prov_r_vlm.read_text(encoding="utf-8")) if prov_r_vlm.is_file() else {}
                    inspected_prov = prov_result_data.get("inspected_regions", [])
                    plan_prov = prov_result_data.get("plan", [])
                    all_trials_records.append({
                        "case_id": case_id,
                        "domain": domain,
                        "variant": variant,
                        "internal_variant": internal_variant,
                        "trial_type": "provider_replay",
                        "search_order": "provider",
                        "seed": "",
                        "expected_status": expected_status,
                        "actual_status": case_manifest.get("provider_replay_status", ""),
                        "status_match": (case_manifest.get("provider_replay_status") == expected_status),
                        "inspected_count": len(inspected_prov),
                        "inspected_regions": ";".join(inspected_prov),
                        "plan_length": len(plan_prov) if plan_prov else 0,
                        "runtime_sec": round(prov_manifest.get("pipeline_runtime_seconds", 0.0), 4),
                        "ref_complete": ref_eval_dict.get("reference_complete", False),
                        "exact_structural_match": ref_eval_dict.get("exact_structural_match", False),
                        "role_precision": ref_eval_dict.get("role_identity_precision", 0.0),
                        "role_recall": ref_eval_dict.get("role_identity_recall", 0.0),
                        "relation_precision": ref_eval_dict.get("relation_precision", 0.0),
                        "relation_recall": ref_eval_dict.get("relation_recall", 0.0),
                        "spec_sha256": prov_manifest.get("specification_sha256", ""),
                    })
                    reused_trials_count += 1

                # Random replays
                for seed_p in sorted(random_replays_dir.glob("seed_*")):
                    seed_num = int(seed_p.name.replace("seed_", ""))
                    seed_m_path = seed_p / "runtime_output" / domain / variant / "vlm" / "run_manifest.json"
                    seed_r_path = seed_p / "runtime_output" / domain / variant / "vlm" / "result.json"
                    if seed_m_path.is_file():
                        sm = json.loads(seed_m_path.read_text(encoding="utf-8"))
                        sr = json.loads(seed_r_path.read_text(encoding="utf-8")) if seed_r_path.is_file() else {}
                        inspected_rand = sr.get("inspected_regions", [])
                        all_trials_records.append({
                            "case_id": case_id,
                            "domain": domain,
                            "variant": variant,
                            "internal_variant": internal_variant,
                            "trial_type": "random_replay",
                            "search_order": "random",
                            "seed": str(seed_num),
                            "expected_status": expected_status,
                            "actual_status": sm.get("terminal_status", ""),
                            "status_match": (sm.get("terminal_status") == expected_status),
                            "inspected_count": len(inspected_rand),
                            "inspected_regions": ";".join(inspected_rand),
                            "plan_length": 0,
                            "runtime_sec": round(sm.get("pipeline_runtime_seconds", 0.0), 4),
                            "ref_complete": ref_eval_dict.get("reference_complete", False),
                            "exact_structural_match": ref_eval_dict.get("exact_structural_match", False),
                            "role_precision": ref_eval_dict.get("role_identity_precision", 0.0),
                            "role_recall": ref_eval_dict.get("role_identity_recall", 0.0),
                            "relation_precision": ref_eval_dict.get("relation_precision", 0.0),
                            "relation_recall": ref_eval_dict.get("relation_recall", 0.0),
                            "spec_sha256": sm.get("specification_sha256", ""),
                        })
                        reused_trials_count += 1
                continue
            else:
                print(f"  -> Case {case_id} fingerprint mismatch (saved: {str(saved_fp_sha)[:8]}, current: {curr_fp_sha[:8]}); executing fresh...", flush=True)

        # Ensure target case directory is clean with no stale artifacts
        clean_case_directory(case_dir)
        for d in (inputs_dir, live_dir, provider_replay_dir, random_replays_dir, ref_dir, val_dir):
            d.mkdir(parents=True, exist_ok=True)

        # --- A. LIVE VLM RUN ---
        newly_executed_count += 1
        live_out_root = live_dir / "runtime_output"
        fm_diag_dir = live_dir / "fm_diagnostics"
        fm_diag_dir.mkdir(parents=True, exist_ok=True)

        # Ensure raw VLM responses are retained in fm_diagnostics
        os.environ["TAMP_FM_DIAGNOSTICS_DIR"] = str(fm_diag_dir)

        live_start = time.perf_counter()

        live_result = run_pipeline(
            domain=domain,
            variant=variant,
            mode="vlm",
            dry_run=True,
            output_root=live_out_root,
        )
        live_elapsed = time.perf_counter() - live_start
        newly_executed_runtime_seconds += live_elapsed

        (live_dir / "exit_code.txt").write_text("0\n")
        (live_dir / "command.txt").write_text(
            f"run_pipeline(domain={domain!r}, variant={variant!r}, mode=vlm, dry_run=True)\n"
        )

        live_run_dir = live_out_root / domain / variant / "vlm"
        live_manifest_path = live_run_dir / "run_manifest.json"
        live_manifest = json.loads(live_manifest_path.read_text(encoding="utf-8")) if live_manifest_path.exists() else {}

        # Copy any additional FM diagnostic files created in live_run_dir
        diag_files = list(live_run_dir.glob("**/fm_call_*.json")) or list(live_run_dir.glob("fm_call_*.json"))
        for df in diag_files:
            if df.parent != fm_diag_dir:
                shutil.copy(df, fm_diag_dir / df.name)

        all_fm_diag = sorted(fm_diag_dir.glob("fm_call_*.json"))
        raw_response_available = len(all_fm_diag) > 0
        if not raw_response_available:
            write_json(fm_diag_dir / "raw_diagnostics_unavailable.json", {
                "raw_response_available": False,
                "reason": "No raw FM diagnostic captured by transport or provider for this run",
                "case_id": case_id,
            })

        # Copy captured inputs to inputs_dir
        captured_inputs = list(live_run_dir.glob("vlm_inputs/*.png")) or list(live_run_dir.glob("*.png"))
        for img_p in captured_inputs:
            shutil.copy(img_p, inputs_dir / img_p.name)

        # Deterministic Prompt Leakage Audit strictly over outgoing request payloads
        if all_fm_diag:
            diag_payloads = []
            for df in all_fm_diag:
                try:
                    diag_payloads.append(json.loads(df.read_text(encoding="utf-8")))
                except Exception:
                    pass
            prompt_leak_report = audit_prompt_leakage(diag_payloads, domain=domain)
            prompt_leak_report["live_request_diagnostics_inspected"] = True
        else:
            prompt_leak_report = audit_prompt_leakage(
                {"system_prompt": SYSTEM_PROMPT, "task_instruction": task_instruction},
                domain=domain,
            )
            prompt_leak_report["live_request_diagnostics_inspected"] = False

        prompt_leakage_audit = {
            "case_id": case_id,
            "domain": domain,
            "variant": variant,
            **prompt_leak_report,
        }
        write_json(live_dir / "prompt_leakage_audit.json", prompt_leakage_audit)

        # Load live G_F
        live_spec_file = live_run_dir / "functional_specification.json"
        live_gf = FunctionalRequirementGraph.from_dict(json.loads(live_spec_file.read_text())) if live_spec_file.exists() else None

        candidate_regions = tuple(live_gf.candidate_regions) if live_gf else ()
        region_ranking = tuple(live_gf.region_ranking) if live_gf else ()

        # --- B. PROVIDER REPLAY (Deterministic Downstream Solver Check) ---
        prov_replay_out_root = provider_replay_dir / "runtime_output"
        prov_replay_result = None
        if live_spec_file.exists():
            prov_replay_result = run_pipeline(
                domain=domain,
                variant=variant,
                mode="vlm",
                specification_json=live_spec_file,
                search_order="auto",
                dry_run=True,
                output_root=prov_replay_out_root,
            )
            (provider_replay_dir / "command.txt").write_text(
                f"run_pipeline(domain={domain!r}, variant={variant!r}, mode=vlm, specification_json={str(live_spec_file)!r}, search_order=auto, dry_run=True)\n"
            )

        prov_replay_run_dir = prov_replay_out_root / domain / variant / "vlm"
        prov_replay_manifest_path = prov_replay_run_dir / "run_manifest.json"
        prov_replay_manifest = json.loads(prov_replay_manifest_path.read_text(encoding="utf-8")) if prov_replay_manifest_path.exists() else {}

        # --- C. RANDOM REPLAYS (Seeds 0..9) ---
        random_replay_results = {}
        should_run_random = (domain in {"kitchen", "workshop"} and len(candidate_regions) >= 2 and live_spec_file.exists())

        if should_run_random:
            for seed in range(10):
                seed_str = f"seed_{seed:02d}"
                seed_dir = random_replays_dir / seed_str
                seed_dir.mkdir(parents=True, exist_ok=True)
                seed_out_root = seed_dir / "runtime_output"

                r_res = run_pipeline(
                    domain=domain,
                    variant=variant,
                    mode="vlm",
                    specification_json=live_spec_file,
                    search_order="random",
                    search_seed=seed,
                    dry_run=True,
                    output_root=seed_out_root,
                )
                seed_run_dir = seed_out_root / domain / variant / "vlm"
                seed_manifest_p = seed_run_dir / "run_manifest.json"
                seed_manifest = json.loads(seed_manifest_p.read_text(encoding="utf-8")) if seed_manifest_p.exists() else {}

                random_replay_results[seed] = {
                    "seed": seed,
                    "status": r_res.status,
                    "run_dir": str(seed_run_dir),
                    "inspected_regions": list(r_res.inspected_regions),
                    "manifest": seed_manifest,
                }

        # --- D. REFERENCE G_F & OFFLINE EVALUATION ---
        ref_gf = gt_spec_provider.provide(domain, task_instruction, [])
        write_json(ref_dir / "reference_gf.json", ref_gf.to_dict())
        write_json(ref_dir / "reference_generation_manifest.json", {
            "domain": domain,
            "variant": variant,
            "task_instruction": task_instruction,
            "source": "GTSpecProvider",
            "sha256": sha256_file(ref_dir / "reference_gf.json"),
        })

        if live_gf:
            ref_eval = evaluate_gf_against_reference(live_gf, reference_graph=ref_gf)
            ref_eval_dict = {
                "domain": ref_eval.domain,
                "exact_structural_match": ref_eval.exact_structural_match,
                "reference_complete": ref_eval.reference_complete,
                "role_identity_precision": ref_eval.role_identity_precision,
                "role_identity_recall": ref_eval.role_identity_recall,
                "role_exact_precision": ref_eval.role_exact_precision,
                "role_exact_recall": ref_eval.role_exact_recall,
                "relation_precision": ref_eval.relation_precision,
                "relation_recall": ref_eval.relation_recall,
                "operation_group_identity_precision": ref_eval.operation_group_identity_precision,
                "operation_group_identity_recall": ref_eval.operation_group_identity_recall,
                "operation_group_exact_precision": ref_eval.operation_group_exact_precision,
                "operation_group_exact_recall": ref_eval.operation_group_exact_recall,
                "missing_roles": list(ref_eval.missing_roles),
                "extra_roles": list(ref_eval.extra_roles),
                "matched_roles": list(ref_eval.matched_roles),
                "missing_relations": [list(r) for r in ref_eval.missing_relations],
                "extra_relations": [list(r) for r in ref_eval.extra_relations],
                "missing_operation_groups": list(ref_eval.missing_operation_groups),
                "extra_operation_groups": list(ref_eval.extra_operation_groups),
                "matched_operation_groups": [list(p) for p in ref_eval.matched_operation_groups],
                "role_attribute_mismatches": ref_eval.role_attribute_mismatches,
                "operation_group_mismatches": ref_eval.operation_group_mismatches,
            }
        else:
            ref_eval_dict = {"exact_structural_match": False, "reference_complete": False}
        write_json(ref_dir / "gf_reference_evaluation.json", ref_eval_dict)

        # --- E. CASE REPLAY VALIDATION (Strong Deterministic Verification) ---
        replay_val_report = {"case_id": case_id, "provider_replay_valid": False, "random_replays_valid": True, "checks": []}
        if live_run_dir.exists() and prov_replay_run_dir.exists():
            val_ok, val_details = validate_vlm_replay(live_run_dir, prov_replay_run_dir, expect_replay_search="provider")
            replay_val_report["provider_replay_valid"] = val_ok
            replay_val_report["provider_replay_details"] = val_details

        for seed, s_info in random_replay_results.items():
            s_run_dir = Path(s_info["run_dir"])
            val_ok, val_details = validate_vlm_replay(live_run_dir, s_run_dir, expect_replay_search="random", expect_seed=seed)
            if not val_ok:
                replay_val_report["random_replays_valid"] = False
            replay_val_report[f"seed_{seed:02d}_valid"] = val_ok

        write_json(val_dir / "case_replay_validation.json", replay_val_report)

        # --- F. CASE MANIFEST ---
        case_manifest = {
            "case_id": case_id,
            "domain": domain,
            "variant": variant,
            "internal_variant": internal_variant,
            "provenance_fingerprint": current_fingerprint,
            "expected_terminal_status": expected_status,
            "live_terminal_status": live_result.status,
            "live_status_matches_expected": (live_result.status == expected_status),
            "provider_replay_status": prov_replay_result.status if prov_replay_result else None,
            "candidate_regions": list(candidate_regions),
            "region_ranking": list(region_ranking),
            "reference_complete": ref_eval_dict.get("reference_complete", False),
            "exact_structural_match": ref_eval_dict.get("exact_structural_match", False),
            "role_precision": ref_eval_dict.get("role_identity_precision", 0.0),
            "role_recall": ref_eval_dict.get("role_identity_recall", 0.0),
            "relation_precision": ref_eval_dict.get("relation_precision", 0.0),
            "relation_recall": ref_eval_dict.get("relation_recall", 0.0),
            "provider_replay_valid": replay_val_report.get("provider_replay_valid", False),
            "random_replays_valid": replay_val_report.get("random_replays_valid", True),
            "random_replays_count": len(random_replay_results),
            "runtime_seconds": round(live_elapsed, 4),
        }
        write_json(case_dir / "case_manifest.json", case_manifest)
        case_summaries[case_id] = case_manifest

        # Record CSV row for live run
        all_trials_records.append({
            "case_id": case_id,
            "domain": domain,
            "variant": variant,
            "internal_variant": internal_variant,
            "trial_type": "live",
            "search_order": "auto",
            "seed": "",
            "expected_status": expected_status,
            "actual_status": live_result.status,
            "status_match": (live_result.status == expected_status),
            "inspected_count": len(live_result.inspected_regions),
            "inspected_regions": ";".join(live_result.inspected_regions),
            "plan_length": len(live_result.plan) if live_result.plan else 0,
            "runtime_sec": round(live_elapsed, 4),
            "ref_complete": ref_eval_dict.get("reference_complete", False),
            "exact_structural_match": ref_eval_dict.get("exact_structural_match", False),
            "role_precision": ref_eval_dict.get("role_identity_precision", 0.0),
            "role_recall": ref_eval_dict.get("role_identity_recall", 0.0),
            "relation_precision": ref_eval_dict.get("relation_precision", 0.0),
            "relation_recall": ref_eval_dict.get("relation_recall", 0.0),
            "spec_sha256": live_manifest.get("specification_sha256", ""),
        })
        newly_executed_trials_count += 1

        # Record CSV row for provider replay
        if prov_replay_result:
            all_trials_records.append({
                "case_id": case_id,
                "domain": domain,
                "variant": variant,
                "internal_variant": internal_variant,
                "trial_type": "provider_replay",
                "search_order": "provider",
                "seed": "",
                "expected_status": expected_status,
                "actual_status": prov_replay_result.status,
                "status_match": (prov_replay_result.status == expected_status),
                "inspected_count": len(prov_replay_result.inspected_regions),
                "inspected_regions": ";".join(prov_replay_result.inspected_regions),
                "plan_length": len(prov_replay_result.plan) if prov_replay_result.plan else 0,
                "runtime_sec": round(prov_replay_manifest.get("pipeline_runtime_seconds", 0.0), 4),
                "ref_complete": ref_eval_dict.get("reference_complete", False),
                "exact_structural_match": ref_eval_dict.get("exact_structural_match", False),
                "role_precision": ref_eval_dict.get("role_identity_precision", 0.0),
                "role_recall": ref_eval_dict.get("role_identity_recall", 0.0),
                "relation_precision": ref_eval_dict.get("relation_precision", 0.0),
                "relation_recall": ref_eval_dict.get("relation_recall", 0.0),
                "spec_sha256": prov_replay_manifest.get("specification_sha256", ""),
            })
            newly_executed_trials_count += 1

        # Record CSV rows for random replays
        for seed, s_info in random_replay_results.items():
            all_trials_records.append({
                "case_id": case_id,
                "domain": domain,
                "variant": variant,
                "internal_variant": internal_variant,
                "trial_type": "random_replay",
                "search_order": "random",
                "seed": str(seed),
                "expected_status": expected_status,
                "actual_status": s_info["status"],
                "status_match": (s_info["status"] == expected_status),
                "inspected_count": len(s_info["inspected_regions"]),
                "inspected_regions": ";".join(s_info["inspected_regions"]),
                "plan_length": 0,
                "runtime_sec": round(s_info["manifest"].get("pipeline_runtime_seconds", 0.0), 4),
                "ref_complete": ref_eval_dict.get("reference_complete", False),
                "exact_structural_match": ref_eval_dict.get("exact_structural_match", False),
                "role_precision": ref_eval_dict.get("role_identity_precision", 0.0),
                "role_recall": ref_eval_dict.get("role_identity_recall", 0.0),
                "relation_precision": ref_eval_dict.get("relation_precision", 0.0),
                "relation_recall": ref_eval_dict.get("relation_recall", 0.0),
                "spec_sha256": s_info["manifest"].get("specification_sha256", ""),
            })
            newly_executed_trials_count += 1

        rc = ref_eval_dict.get("reference_complete")
        rp = ref_eval_dict.get("role_identity_precision")
        rr = ref_eval_dict.get("role_identity_recall")
        print(f"  -> Live status: {live_result.status} (expected: {expected_status}) | Ref complete: {rc} | Role F1: {rp}/{rr}\n", flush=True)

    # 4. Top-level Summaries
    total_time = (datetime.now(timezone.utc) - start_ts).total_seconds()
    summed_case_time = sum(c.get("runtime_seconds", 0.0) for c in case_summaries.values())

    # Write results.csv
    csv_fields = [
        "case_id", "domain", "variant", "internal_variant", "trial_type", "search_order", "seed",
        "expected_status", "actual_status", "status_match", "inspected_count", "inspected_regions",
        "plan_length", "runtime_sec", "ref_complete", "exact_structural_match",
        "role_precision", "role_recall", "relation_precision", "relation_recall", "spec_sha256"
    ]
    with open(bundle_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_trials_records)

    # Aggregate statistics
    live_records = [r for r in all_trials_records if r["trial_type"] == "live"]
    status_counts = {}
    for r in live_records:
        status_counts[r["actual_status"]] = status_counts.get(r["actual_status"], 0) + 1

    expected_matches = sum(1 for r in live_records if r["status_match"])
    ref_completes = sum(1 for r in live_records if r["ref_complete"])
    structural_matches = sum(1 for r in live_records if r["exact_structural_match"])

    mean_role_p = sum(r["role_precision"] for r in live_records) / len(live_records) if live_records else 0.0
    mean_role_r = sum(r["role_recall"] for r in live_records) / len(live_records) if live_records else 0.0
    mean_rel_p = sum(r["relation_precision"] for r in live_records) / len(live_records) if live_records else 0.0
    mean_rel_r = sum(r["relation_recall"] for r in live_records) / len(live_records) if live_records else 0.0

    replay_records = [r for r in all_trials_records if r["trial_type"] == "provider_replay"]
    provider_replay_cases_count = sum(1 for c in case_summaries.values() if c.get("provider_replay_status") is not None)
    provider_validation_passes = sum(1 for c in case_summaries.values() if c.get("provider_replay_valid") is True)
    provider_status_matches = sum(
        1 for r in replay_records
        if r["actual_status"] == case_summaries[r["case_id"]]["live_terminal_status"]
    )

    summary = {
        "benchmark": "PASS_3_6B_2_VLM_FULL_EVALUATION",
        "benchmark_provenance_version": "phase3_6b2_hardened_v2",
        "model": os.environ.get("TAMP_FM_MODEL", "qwen35-9b"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_dirty_source_hash": git_dirty_hash,
        "is_clean_source_tree": is_clean_source_tree,
        "started_at_utc": start_ts.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation_wall_time_seconds": round(total_time, 2),
        "summed_live_case_runtime_seconds": round(summed_case_time, 2),
        "summed_newly_executed_runtime_seconds": round(newly_executed_runtime_seconds, 2),
        "total_cases": len(case_registry),
        "number_of_cases_executed_this_invocation": newly_executed_count,
        "number_of_cases_reused": reused_count,
        "total_trial_records": len(all_trials_records),
        "newly_executed_trial_count": newly_executed_trials_count,
        "reused_trial_count": reused_trials_count,
        "total_trials_executed": len(all_trials_records),  # Backward compatibility alias
        "total_runtime_seconds": round(total_time, 2),     # Backward compatibility alias
        "live_status_distribution": status_counts,
        "live_matches_expected_terminal_status_count": expected_matches,
        "live_matches_expected_terminal_status_rate": round(expected_matches / len(live_records), 4) if live_records else 0.0,
        "reference_evaluator": {
            "reference_complete_count": ref_completes,
            "reference_complete_rate": round(ref_completes / len(live_records), 4) if live_records else 0.0,
            "exact_structural_match_count": structural_matches,
            "exact_structural_match_rate": round(structural_matches / len(live_records), 4) if live_records else 0.0,
            "mean_role_identity_precision": round(mean_role_p, 4),
            "mean_role_identity_recall": round(mean_role_r, 4),
            "mean_relation_precision": round(mean_rel_p, 4),
            "mean_relation_recall": round(mean_rel_r, 4),
        },
        "provider_replay_evaluation": {
            "description": "Deterministic Downstream Solver Replay of Saved Functional Specification (via validate_vlm_replay)",
            "provider_replay_cases_total": provider_replay_cases_count,
            "provider_replay_validation_pass_count": provider_validation_passes,
            "provider_replay_validation_pass_rate": round(provider_validation_passes / provider_replay_cases_count, 4) if provider_replay_cases_count else 1.0,
            "provider_replay_terminal_status_match_count": provider_status_matches,
            "provider_replay_terminal_status_match_rate": round(provider_status_matches / len(replay_records), 4) if replay_records else 1.0,
        },
        "replay_determinism": {
            "description": "Provider Replay (Deterministic Downstream Replay of Saved Functional Specification; does not measure stochastic VLM reproducibility)",
            "provider_replay_trials": provider_replay_cases_count,
            "provider_replay_exact_match_count": provider_validation_passes,
            "provider_replay_exact_match_rate": round(provider_validation_passes / provider_replay_cases_count, 4) if provider_replay_cases_count else 1.0,
        },
        "cases": case_summaries,
    }
    write_json(bundle_dir / "summary.json", summary)

    # Write README.md
    model_name = os.environ.get('TAMP_FM_MODEL', 'qwen35-9b')
    val_pass_pct = f"{provider_validation_passes / provider_replay_cases_count * 100:.1f}%" if provider_replay_cases_count else "N/A"
    status_match_pct = f"{provider_status_matches / len(replay_records) * 100:.1f}%" if replay_records else "N/A"
    readme_content = f"""# Pass 3.6B.2: Full 32-Variant Live VLM Evaluation Report

- **Model**: `{model_name}`
- **Git Commit**: `{git_commit}` (dirty: `{git_dirty}`, is_clean: `{is_clean_source_tree}`)
- **Total Cases**: {len(case_registry)} (Kitchen K1-K12, Living Room L1-L10, Workshop W1-W10)
- **Newly Executed Cases**: {newly_executed_count}
- **Reused Cases (Matching Fingerprint)**: {reused_count}
- **Invocation Wall Time**: {total_time:.2f} seconds
- **Summed Live Case Runtime**: {summed_case_time:.2f} seconds
- **Summed Newly Executed Runtime**: {newly_executed_runtime_seconds:.2f} seconds
- **Total Trial Records**: {len(all_trials_records)} (newly executed: {newly_executed_trials_count}, reused: {reused_trials_count})

## Summary Metrics
- **Live Terminal Status Match with Reference Expected**: {expected_matches}/{len(live_records)} ({expected_matches/len(live_records)*100:.1f}%)
- **Reference Complete Rate**: {ref_completes}/{len(live_records)} ({ref_completes/len(live_records)*100:.1f}%)
- **Exact Structural Match Rate**: {structural_matches}/{len(live_records)} ({structural_matches/len(live_records)*100:.1f}%)
- **Mean Role Precision / Recall**: {mean_role_p:.3f} / {mean_role_r:.3f}
- **Mean Relation Precision / Recall**: {mean_rel_p:.3f} / {mean_rel_r:.3f}
- **Provider Replay Deterministic Validation Pass Rate**: {provider_validation_passes}/{provider_replay_cases_count} ({val_pass_pct})
- **Provider Replay Terminal Status Agreement**: {provider_status_matches}/{len(replay_records)} ({status_match_pct})
"""
    (bundle_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # Compute SHA256SUMS.txt for all files
    print("Generating SHA256SUMS.txt...", flush=True)
    all_files = sorted([p for p in bundle_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"])
    lines = []
    for fp in all_files:
        rel = fp.relative_to(bundle_dir)
        lines.append(f"{sha256_file(fp)}  {rel}")
    (bundle_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Artifacts preserved at: {bundle_dir}")
    print(f"Total cases: {len(case_registry)} (executed: {newly_executed_count}, reused: {reused_count}) | Total trial records: {len(all_trials_records)}")
    print(f"Live terminal status distribution: {status_counts}")
    print(f"Reference complete: {ref_completes}/{len(live_records)} | Mean role recall: {mean_role_r:.3f}\n")


if __name__ == "__main__":
    main()
