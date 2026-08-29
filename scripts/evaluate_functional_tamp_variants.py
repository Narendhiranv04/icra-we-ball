#!/usr/bin/env python3
import subprocess
import json
import os
import sys
import time

DOMAINS = {
    "kitchen": [f"K{i}" for i in range(1, 13)],
    "living_room": [f"L{i}" for i in range(1, 11)],
    "workshop": [f"W{i}" for i in range(1, 11)],
}

EXPECTED = {
    "kitchen": {v: "ACTION_SEQUENCE_READY" if int(v[1:]) <= 6 else "INFEASIBLE" for v in DOMAINS["kitchen"]},
    "living_room": {v: "ACTION_SEQUENCE_READY" if int(v[1:]) <= 6 else "INFEASIBLE" for v in DOMAINS["living_room"]},
    "workshop": {v: "ACTION_SEQUENCE_READY" if int(v[1:]) <= 8 else "INFEASIBLE" for v in DOMAINS["workshop"]},
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default=f"/tmp/pass2b2_full_sweep_{int(time.time())}")
    args = parser.parse_args()
    
    if os.path.exists(args.output_root):
        print(f"Error: Output root {args.output_root} already exists. Please delete it or use a fresh path.")
        sys.exit(1)
        
    os.makedirs(args.output_root, exist_ok=True)
    
    results = []
    attempted_count = 0
    completed_count = 0
    matching_count = 0

    print(f"{'Domain':<12} {'Variant':<8} {'Expected':<22} {'Actual':<22} {'Match':<6} {'Inspections':<12} {'PlanLen':<8} {'Grounding':<10} {'Runtime':<8} {'FailureReason'}")
    print("-" * 130)

    for domain, variants in DOMAINS.items():
        for variant in variants:
            attempted_count += 1
            expected = EXPECTED[domain][variant]
            
            cmd = [
                sys.executable,
                "-m", "mujoco_scenes.functional_tamp_pipeline.run",
                "--domain", domain,
                "--variant", variant,
                "--mode", "gt",
                "--dry-run",
                "--output-root", args.output_root
            ]
            
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."})
            runtime = time.time() - t0
            
            res_file = os.path.join(args.output_root, domain, variant, "gt", "result.json")
            actual = "ERROR_NO_RESULT"
            failure_reason = None
            inspections = 0
            plan_len = 0
            is_completed = False
            
            if os.path.exists(res_file):
                try:
                    with open(res_file, "r") as f:
                        data = json.load(f)
                        actual = data.get("status", "UNKNOWN")
                        failure_reason = data.get("failure_reason")
                        inspections = len(data.get("inspected_regions", []))
                        plan_len = len(data.get("plan", []))
                        is_completed = True
                except Exception as e:
                    actual = "ERROR_INVALID_RESULT"
                    failure_reason = str(e)
            elif proc.returncode != 0:
                actual = "CRASH"
                failure_reason = f"return_code={proc.returncode}"
            
            if is_completed:
                completed_count += 1
            match = "YES" if actual == expected else "NO"
            if match == "YES":
                matching_count += 1
                
            # Try to get grounding status if satisfaction.json or graph_grounding_result.json exists
            sat_file = os.path.join(args.output_root, domain, variant, "gt", "satisfaction.json")
            if os.path.exists(sat_file):
                with open(sat_file, "r") as f:
                    sat_data = json.load(f)
                    grounding = sat_data.get("status", "UNKNOWN")
            else:
                ggr_file = os.path.join(args.output_root, domain, variant, "gt", "graph_grounding_result.json")
            results.append({
                "domain": domain, "variant": variant, "expected_status": expected, "actual_status": actual,
                "plan_length": plan_len, "combined_high_level_count": inspections + plan_len,
                "grounding_complete": grounding == "COMPLETE",
                "grounded_assignment_audit": len(audit_data.get("violations", [])) == 0 if audit_data else (actual == "ACTION_SEQUENCE_READY"),
                "plan_replay_valid": plan_replay_valid,
            })


    with open(os.path.join(args.output_root, "summary.csv"), "w") as f:
        f.write("Domain,Variant,Expected,Actual,Match,Inspections,PlanLen,CombinedCount,Grounding,PlanValid,AccessValid,Runtime,FailureReason\n")
        for r in results:
            f.write(f"{r['domain']},{r['variant']},{r['expected_status']},{r['actual_status']},{r['match']},{r['inspected_regions']},{r['plan_length']},{r['combined_high_level_count']},{r['grounding_status']},{r['plan_replay_valid']},{r['accessibility_valid']},{r['runtime_sec']:.2f},{r['failure_reason']}\n")
    print("-" * 130)
    print(f"Attempted: {attempted_count} / 32")
    print(f"Match: {matching_count} / {attempted_count}")

    main()
