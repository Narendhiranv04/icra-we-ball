#!/usr/bin/env python3
import subprocess
import json
import os
import sys

def main():
    variants = [f"W{i}" for i in range(1, 11)]
    output_dir = "/tmp/pass2b1_evaluation"
    
    results = {}
    
    for v in variants:
        print(f"Evaluating {v}...")
        cmd = [
            "/home/naren/miniconda3/bin/python",
            "-m", "mujoco_scenes.functional_tamp_pipeline.run",
            "--domain", "workshop",
            "--variant", v,
            "--mode", "gt",
            "--dry-run",
            "--output-root", output_dir
        ]
        
        proc = subprocess.run(cmd, capture_output=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."})
        
        res_file = os.path.join(output_dir, "workshop", v, "gt", "satisfaction.json")
        status = "ERROR"
        if os.path.exists(res_file):
            with open(res_file, "r") as f:
                data = json.load(f)
                if data.get("status") == "COMPLETE":
                    status = "ACTION_SEQUENCE_READY"
                else:
                    status = data.get("status", "UNKNOWN")
        else:
            if "ACTION_SEQUENCE_READY" in proc.stdout.decode("utf-8") or "ACTION_SEQUENCE_READY" in proc.stderr.decode("utf-8"):
                 status = "ACTION_SEQUENCE_READY"
            elif "INFEASIBLE" in proc.stdout.decode("utf-8") or "INFEASIBLE" in proc.stderr.decode("utf-8"):
                 status = "INFEASIBLE"
            else:
                 print("STDOUT:", proc.stdout.decode("utf-8")[-500:])
                 print("STDERR:", proc.stderr.decode("utf-8")[-500:])
                 
        results[v] = status
        print(f"{v}: {status}")
        
    print("\n--- FINAL REPORT ---")
    for v, s in results.items():
        print(f"{v}: {s}")

if __name__ == "__main__":
    main()
