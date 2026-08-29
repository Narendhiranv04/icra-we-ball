#!/usr/bin/env python3
import subprocess
import os
import sys

VARIANTS = ["W1", "W2", "W5", "W7", "W8", "W9", "W10"]
EXPECTED = {
    "W1": "ACTION_SEQUENCE_READY",
    "W2": "ACTION_SEQUENCE_READY",
    "W5": "ACTION_SEQUENCE_READY",
    "W7": "ACTION_SEQUENCE_READY",
    "W8": "ACTION_SEQUENCE_READY",
    "W9": "INFEASIBLE",
    "W10": "INFEASIBLE"
}

def main():
    results = {v: {"pass": 0, "fail": 0} for v in VARIANTS}
    
    for i in range(1, 4):
        output_root = f"/tmp/pass2b2_repeat_{i}"
        os.makedirs(output_root, exist_ok=True)
        
        for v in VARIANTS:
            cmd = [
                sys.executable,
                "-m", "mujoco_scenes.functional_tamp_pipeline.run",
                "--domain", "workshop",
                "--variant", v,
                "--mode", "gt",
                "--dry-run",
                "--output-root", output_root
            ]
            proc = subprocess.run(cmd, capture_output=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "."})
            
            import json
            res_file = os.path.join(output_root, "workshop", v, "gt", "result.json")
            actual = "CRASH"
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    actual = json.load(f).get("status", "UNKNOWN")
            
            if actual == EXPECTED[v]:
                results[v]["pass"] += 1
            else:
                results[v]["fail"] += 1
                
    print("Robustness Results:")
    for v in VARIANTS:
        print(f"{v} {results[v]['pass']}/3 (Expected: {EXPECTED[v]})")

if __name__ == "__main__":
    main()
