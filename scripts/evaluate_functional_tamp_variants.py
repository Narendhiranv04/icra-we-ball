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
    "kitchen": {v: "ACTION_SEQUENCE_READY" if v in ["K1", "K3", "K5"] else ("INCOMPLETE" if v in ["K2", "K4", "K6"] else "INFEASIBLE") for v in DOMAINS["kitchen"]},
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
            
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    data = json.load(f)
                    actual = data.get("status", "UNKNOWN")
                    failure_reason = data.get("failure_reason")
                    inspections = len(data.get("inspected_regions", []))
                    plan_len = len(data.get("plan", []))
            elif proc.returncode != 0:
                actual = "CRASH"
                failure_reason = f"return_code={proc.returncode}"
            
            completed_count += 1
            match = "YES" if actual == expected else "NO"
            if match == "YES":
                matching_count += 1
                
            # Try to get grounding status if satisfaction.json exists
            grounding = "UNKNOWN"
            sat_file = os.path.join(args.output_root, domain, variant, "gt", "satisfaction.json")
            if os.path.exists(sat_file):
                with open(sat_file, "r") as f:
                    sat_data = json.load(f)
                    grounding = sat_data.get("status", "UNKNOWN")
                    
            print(f"{domain:<12} {variant:<8} {expected:<22} {actual:<22} {match:<6} {inspections:<12} {plan_len:<8} {grounding:<10} {runtime:<8.2f} {str(failure_reason)}")
            
            results.append({
                "domain": domain, "variant": variant, "expected_status": expected, "actual_status": actual,
                "match": match, "return_code": proc.returncode, "inspected_regions": inspections,
                "plan_length": plan_len, "combined_high_level_count": inspections + plan_len,
                "runtime_sec": runtime, "failure_reason": failure_reason, "result_json_path": res_file,
                "grounding_status": grounding
            })

    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    with open(os.path.join(args.output_root, "summary.csv"), "w") as f:
        f.write("Domain,Variant,Expected,Actual,Match,Inspections,PlanLen,Grounding,Runtime,FailureReason\n")
        for r in results:
            f.write(f"{r['domain']},{r['variant']},{r['expected_status']},{r['actual_status']},{r['match']},{r['inspected_regions']},{r['plan_length']},{r['grounding_status']},{r['runtime_sec']:.2f},{r['failure_reason']}\n")

    print("-" * 130)
    print(f"Attempted: {attempted_count} / 32")
    print(f"Completed: {completed_count} / 32")
    print(f"Match: {matching_count} / {attempted_count}")

if __name__ == "__main__":
    main()
